"""replicated source env 的机器人/对象导入与 fixed-base 修正。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from linkerbot_sim.assets.robot_import import import_robot_asset
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    RobotSceneInstanceConfig,
    resolve_controller_profile,
)
from linkerbot_sim.assets.root_pose import (
    RootPoseConfig,
    apply_root_pose,
    apply_root_pose_transform,
    mjcf_fixed_root_joint_paths_without_body0,
)
from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    apply_solver_iteration_overrides,
    merge_solver_configs,
)
from linkerbot_sim.assets.usd_overrides import (
    apply_robot_gravity_policy,
    apply_robot_usd_overrides,
)
from linkerbot_sim.assets.instance_paths import validate_disjoint_instance_prim_paths
from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.controllers.projection import robot_usd_override_configs
from linkerbot_sim.configuration.objects import (
    DynamicChainObjectProfileConfig,
    RigidObjectProfileConfig,
)
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.isaac.physics.newton.render import prepare_newton_render_subtree
from linkerbot_sim.objects.runtime import (
    RuntimeObjectConfig,
    RuntimeObjectHandle,
    add_runtime_objects,
)
from linkerbot_sim.robots.tcp_binding import (
    resolve_physical_tcp_binding,
    unique_rigid_body_path,
)

from .layout import env_local_prim_path
from .types import SourceReplicatedRobot


def define_source_environment(
    stage: object,
    env_root: str,
    *,
    prepare_newton_render_topology: bool = False,
) -> None:
    """创建 source env 根 Xform 与父 Scope。"""

    from pxr import Sdf, UsdGeom

    parent = Sdf.Path(env_root).GetParentPath()
    if str(parent) != "/" and not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Scope.Define(stage, parent)
    if stage.GetPrimAtPath(env_root).IsValid():
        raise RuntimeError(f"source environment root already exists: {env_root}")
    UsdGeom.Xform.Define(stage, env_root)
    if prepare_newton_render_topology:
        # Kaleidoscope source env 是 Newton render prototype；在任何 child/reference
        # 暴露前发布最终 root topology，clone 与 manager 后续都只更新 matrix value。
        apply_root_pose_transform(
            stage,
            env_root,
            RootPoseConfig(),
            prepare_newton_render_topology=True,
        )


def source_object_configs(
    scene_settings: object,
    *,
    env_root: str,
) -> tuple[RuntimeObjectConfig, ...]:
    """把严格 scene object 实例转成 env-local runtime asset 配置。"""

    result: list[RuntimeObjectConfig] = []
    for instance in tuple(getattr(scene_settings, "objects")):
        profile_name = str(instance.object_profile)
        profile = getattr(instance, "resolved_profile", None)
        if profile is None:
            raise RuntimeError(
                f"scene object {instance.name!r} has no resolved object profile"
            )
        if not isinstance(
            profile, (RigidObjectProfileConfig, DynamicChainObjectProfileConfig)
        ):
            raise TypeError(
                f"scene object {instance.name!r} has an invalid resolved object profile"
            )
        result.append(
            RuntimeObjectConfig(
                name=str(instance.name),
                root_pose=_root_pose(instance.root_pose),
                object_profile=profile_name,
                profile=profile,
                prim_path=env_local_prim_path(env_root, str(instance.prim_path)),
                runtime_handle=None,
            )
        )
    return tuple(result)


def validate_single_dynamic_rigid_object(
    configs: Sequence[RuntimeObjectConfig],
    *,
    expected_name: str,
) -> None:
    """验证状态 port 可完整覆盖场景中所有动态对象。

    Kaleidoscope 当前只有一组 ``object.*`` state 字段，因此允许任意数量静态 rigid，
    但必须恰好有一个非静态 rigid，且其名称与 task 声明一致。dynamic chain 含多个刚体，
    在逐对象/逐 body 状态 schema 出现前必须 fail closed。
    """

    dynamic_names: list[str] = []
    for config in configs:
        if config.kind == "dynamic_chain":
            raise ValueError(
                "single-dynamic-object state contract does not support dynamic_chain"
            )
        if config.kind != "rigid":
            raise ValueError(f"unsupported scene object kind: {config.kind!r}")
        if not isinstance(config.profile, RigidObjectProfileConfig):
            raise ValueError(
                "single-dynamic-object state contract requires rigid profiles"
            )
        if not config.profile.physics.static:
            dynamic_names.append(config.name)
    if len(dynamic_names) != 1:
        raise ValueError(
            "single-dynamic-object state contract requires exactly one non-static "
            f"rigid object, found {dynamic_names}"
        )
    if dynamic_names[0] != expected_name:
        raise ValueError(
            "task dynamic object does not match the unique non-static rigid object: "
            f"expected {dynamic_names[0]!r}, got {expected_name!r}"
        )


def import_source_objects(
    stage: object,
    *,
    configs: Sequence[RuntimeObjectConfig],
    physics_backend: str,
    prepare_newton_render_topology: bool,
) -> tuple[RuntimeObjectHandle, ...]:
    """导入 source env 对象；clone 之前只存在这一份 USD 拓扑。"""

    return add_runtime_objects(
        stage,
        tuple(configs),
        physics_backend=physics_backend,
        prepare_newton_render_topology=prepare_newton_render_topology,
    )


def import_source_robots(
    stage: object,
    *,
    scene_settings: object,
    env_root: str,
    controller_bundle: str,
    controller_bundles: Mapping[str, ControllerProfiles],
    solver_type: str | None,
    physics_backend: str,
    prepare_newton_render_topology: bool,
    object_configs: Sequence[RuntimeObjectConfig],
) -> tuple[SourceReplicatedRobot, ...]:
    """按明确物理后端导入全部机器人，并重绑 MJCF world joint。

    Isaac MJCF importer 的 fixed-base joint 默认 ``body0`` 为空，代表固定到绝对 world。
    这种 joint 即使位于 clone 子树内也不会随 env root 平移，导致所有机器人 reset 到同一
    世界位置。这里创建一个无碰撞、kinematic 的 env anchor，并把 ``body0`` 指向它；
    source 与每个 clone 因而保持完全同构，能够继续使用 ``replicate_physics=true``。
    """

    if prepare_newton_render_topology and physics_backend != "newton":
        raise RuntimeError(
            "Newton render topology intent requires physics_backend='newton'"
        )
    robots = tuple(getattr(scene_settings, "robots"))
    planned_robot_paths = {
        str(instance.label): env_local_prim_path(
            env_root, f"/World/Robots/{instance.label}"
        )
        for instance in robots
    }
    validate_disjoint_instance_prim_paths(
        robot_paths=planned_robot_paths,
        object_paths={item.name: item.prim_path for item in object_configs},
    )
    result: list[SourceReplicatedRobot] = []
    fixed_joint_paths: list[str] = []
    for robot_id, instance in enumerate(robots):
        label = str(instance.label)
        profile_name = str(instance.robot_profile)
        profile = getattr(instance, "resolved_profile", None)
        if not isinstance(profile, RobotProfileSettings):
            raise TypeError(
                f"scene robot {label!r} has no valid resolved robot profile"
            )
        scene_instance = RobotSceneInstanceConfig(
            robot_profile=profile_name,
            root_pose=_root_pose(instance.root_pose),
            robot_id=robot_id,
            label=label,
            controller_profile=getattr(instance, "controller_profile", None),
        )
        execution = RobotExecutionConfig.from_profile(
            profile,
            scene_instance=scene_instance,
        )
        controller_name = resolve_controller_profile(
            scene_instance,
            execution.robot,
            controller_bundle,
        )
        try:
            controllers = controller_bundles[controller_name]
        except KeyError as exc:
            raise RuntimeError(
                f"controller bundle {controller_name!r} is outside the resolved "
                "configuration graph"
            ) from exc
        if not isinstance(controllers, ControllerProfiles):
            raise TypeError(
                f"resolved controller bundle {controller_name!r} has invalid type"
            )
        execution = replace(
            execution,
            robot=replace(
                execution.robot,
                name=f"kaleidoscope_{label}",
                prim_path=env_local_prim_path(env_root, execution.robot.prim_path),
            ),
        )
        articulation_path, asset_path, imported_root_path = import_robot_asset(
            execution.robot,
            physics_backend=physics_backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
            root_pose=execution.root_pose,
        )
        apply_root_pose(
            stage,
            imported_root_path,
            execution.root_pose,
            prepare_newton_render_topology=prepare_newton_render_topology,
        )
        if prepare_newton_render_topology:
            # source env 只 author 一次；先冻结 prototype body，后续 internal-reference
            # render clone 才能继承稳定的 op order。
            prepare_newton_render_subtree(
                stage=stage,
                subtree_root=imported_root_path,
            )
        overrides = robot_usd_override_configs(controllers)
        if execution.robot.contact_material is not None:
            overrides = execution.robot.contact_material.apply_to_configs(overrides)
        if physics_backend == "physx":
            overrides = execution.robot.physx.overrides.apply_to_configs(overrides)
        apply_robot_usd_overrides(
            imported_root_path,
            overrides,
            driven_joint_names=tuple(execution.controlled_joints),
            mjcf_path=(asset_path if execution.robot.asset_type == "mjcf" else None),
            mimic_path=(
                asset_path if execution.robot.asset_type in {"mjcf", "urdf"} else None
            ),
            component_mapping=execution.robot.component_mapping,
            native_mimic=execution.robot.asset_type in {"mjcf", "urdf"},
            physics_backend=physics_backend,
        )
        # SolverIterationConfig 只对应 PhysX schema。Newton 的 solver、迭代数和
        # contact pipeline 由 IsaacNewtonCudaSpec 在 manager finalize 前一次性冻结。
        solver = None
        if physics_backend == "physx":
            if solver_type is None:
                raise ValueError("PhysX replicated scene requires solver_type")
            solver = merge_solver_configs(
                SolverIterationConfig(solver_type=str(solver_type)),
                execution.robot.physx.solver_iterations,
            )
        elif physics_backend != "newton":
            raise ValueError("physics_backend must be physx or newton")
        if solver is not None:
            apply_solver_iteration_overrides(
                stage,
                articulation_path,
                solver,
                component_mapping=execution.robot.component_mapping,
                physics_backend=physics_backend,
            )
        apply_robot_gravity_policy(
            imported_root_path,
            execution.robot.gravity_policy,
            component_mapping=execution.robot.component_mapping,
            physics_backend=physics_backend,
        )
        tcp = _physical_tcp_binding(
            stage=stage,
            imported_root_path=imported_root_path,
            profile=profile,
        )
        result.append(
            SourceReplicatedRobot(
                robot_id=robot_id,
                label=label,
                profile_name=profile_name,
                profile=profile,
                controller_bundle_name=controller_name,
                controller_profiles=controllers,
                execution=execution,
                asset_path=asset_path,
                asset_type=execution.robot.asset_type,
                articulation_path=str(articulation_path),
                imported_root_path=str(imported_root_path),
                controlled_joints=tuple(execution.controlled_joints),
                tcp_frame_name=tcp[0],
                tcp_parent_frame_name=tcp[1],
                tcp_parent_body_path=tcp[2],
                tcp_offset_xyz=tcp[3],
                tcp_offset_rpy=tcp[4],
            )
        )
        if execution.robot.asset_type == "mjcf":
            fixed_joint_paths.extend(
                mjcf_fixed_root_joint_paths_without_body0(stage, imported_root_path)
            )
    if fixed_joint_paths:
        anchor_path = _bind_fixed_joints_to_environment_anchor(
            stage,
            env_root=env_root,
            joint_paths=tuple(fixed_joint_paths),
        )
        if prepare_newton_render_topology:
            prepare_newton_render_subtree(
                stage=stage,
                subtree_root=anchor_path,
            )
    return tuple(result)


def _bind_fixed_joints_to_environment_anchor(
    stage: object,
    *,
    env_root: str,
    joint_paths: tuple[str, ...],
) -> str:
    """让 imported MJCF fixed joints 随 replicated env root 移动。"""

    from pxr import Sdf, UsdGeom, UsdPhysics

    anchor_path = f"{env_root}/__fixed_world_anchor"
    anchor = UsdGeom.Xform.Define(stage, anchor_path).GetPrim()
    rigid = UsdPhysics.RigidBodyAPI.Apply(anchor)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    target = Sdf.Path(anchor_path)
    for joint_path in joint_paths:
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid() or prim.GetTypeName() != "PhysicsFixedJoint":
            raise RuntimeError(f"invalid MJCF fixed-base joint: {joint_path}")
        relationship = prim.GetRelationship("physics:body0")
        if relationship.GetTargets():
            raise RuntimeError(
                f"refusing to replace a non-world fixed joint body0: {joint_path}"
            )
        relationship.SetTargets([target])
    return anchor_path


def _physical_tcp_binding(
    *,
    stage: object,
    imported_root_path: str,
    profile: RobotProfileSettings,
) -> tuple[
    str,
    str,
    str,
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """把已校验 robot profile 的 TCP 投影为物理 rigid parent 与固定偏移。"""

    return resolve_physical_tcp_binding(
        stage=stage,
        imported_root_path=imported_root_path,
        profile=profile,
    ).as_legacy_tuple()


def _unique_rigid_body_path(stage: object, *, root_path: str, body_name: str) -> str:
    """Compatibility wrapper for callers of the former private resolver."""

    return unique_rigid_body_path(
        stage,
        root_path=root_path,
        body_name=body_name,
    )


def _root_pose(value: object) -> RootPoseConfig:
    return RootPoseConfig(
        xyz=tuple(float(item) for item in getattr(value, "xyz")),
        rpy=tuple(float(item) for item in getattr(value, "rpy")),
    )


__all__ = [
    "define_source_environment",
    "import_source_objects",
    "import_source_robots",
    "source_object_configs",
    "validate_single_dynamic_rigid_object",
]
