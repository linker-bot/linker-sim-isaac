"""tiled Isaac scene 的 robot 导入、身份绑定与 clone path 辅助函数。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from linkerbot_sim.assets.robot_import import import_robot_asset
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    RobotSceneInstanceConfig,
    robot_instances_from_env_config,
    resolve_controller_profile,
)
from linkerbot_sim.assets.root_pose import apply_root_pose
from linkerbot_sim.assets.solver_overrides import (
    apply_solver_iteration_overrides,
    merge_solver_configs,
    scene_solver_settings,
)
from linkerbot_sim.assets.usd_overrides import (
    apply_robot_gravity_policy,
    apply_robot_usd_overrides,
)
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.controllers.config import ControllerProfiles, physx_override_configs
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.scene.paths import (
    env_local_suffix,
    make_env_local_prim_path,
    prim_paths_from_suffix,
)
from linkerbot_sim.tiled.scene.types import ImportedTiledRobot, TiledRobotInstance
from linkerbot_sim.tiled.scene.utils import _print_status
from linkerbot_sim.robots.capabilities import (
    PlanningBindingConfig,
    robot_kind_from_profile,
)


def tiled_robot_instances_from_env_config(
    env_config: Mapping[str, object],
) -> tuple[TiledRobotInstance, ...]:
    """从公共 canonical loader 提取 tiled builder 的机器人实例。

    Stable labels are the canonical tiled dictionary keys; session IDs stay as
    explicit data and never derive from a side/name alias.
    """

    instances = robot_instances_from_env_config(env_config)
    return tuple(
        TiledRobotInstance(
            robot_id=instance.robot_id,
            label=instance.label,
            profile_name=instance.robot_profile,
            scene_instance=instance,
        )
        for instance in instances
    )


def env_local_robot_execution(
    *,
    robot_profile_config: Mapping[str, object],
    scene_instance: RobotSceneInstanceConfig,
    env_root: str,
    robot_name: str,
) -> RobotExecutionConfig:
    """把一个 robot profile 改写为 env-local ``RobotExecutionConfig``。

    canonical list schema 先应用实例 prim path，tiled builder 再统一改写到
    ``/World/envs/env_0/...``，避免不同 env 的路径冲突。
    """

    execution = RobotExecutionConfig.from_mapping(
        robot_profile_config,
        scene_instance=scene_instance,
    )
    robot = replace(
        execution.robot,
        name=robot_name,
        prim_path=make_env_local_prim_path(env_root, execution.robot.prim_path),
    )
    return replace(execution, robot=robot)


def _import_env_zero_robots(
    *,
    stage: object,
    env_config: Mapping[str, object],
    tiled_config: TiledEnvConfig,
    controller_bundle: str,
    controller_bundle_loader: Callable[[str], ControllerProfiles],
    env_roots: Sequence[str],
    status_prefix: str | None,
    robot_instances: Sequence[TiledRobotInstance] | None = None,
) -> dict[str, ImportedTiledRobot]:
    """导入 env_0 下的所有机器人，并记录 clone 后每个 env 的路径。"""

    env_zero = env_roots[0]
    result: dict[str, ImportedTiledRobot] = {}
    controller_cache: dict[str, ControllerProfiles] = {}
    instances = (
        tiled_robot_instances_from_env_config(env_config)
        if robot_instances is None
        else tuple(robot_instances)
    )
    for instance in instances:
        robot_profile = load_profile_yaml("robot", instance.profile_name)
        kind = robot_kind_from_profile(robot_profile)
        binding = PlanningBindingConfig.from_profile(robot_profile, kind=kind)
        execution = env_local_robot_execution(
            robot_profile_config=robot_profile,
            scene_instance=instance.scene_instance,
            env_root=env_zero,
            robot_name=f"tiled_{instance.label}",
        )
        controller_profile = resolve_controller_profile(
            instance.scene_instance,
            execution.robot,
            controller_bundle,
        )
        if controller_profile not in controller_cache:
            controller_cache[controller_profile] = controller_bundle_loader(
                controller_profile
            )
        imported = _import_robot_to_env_zero(
            stage=stage,
            env_config=env_config,
            robot_execution=execution,
            controller_profiles=controller_cache[controller_profile],
        )
        articulation_suffix = env_local_suffix(env_zero, imported["articulation_path"])
        imported_root_suffix = env_local_suffix(
            env_zero, imported["imported_root_path"]
        )
        tiled_robot = ImportedTiledRobot(
            name=instance.label,
            profile_name=instance.profile_name,
            controller_profile=controller_profile,
            execution=execution,
            articulation_root_suffix=articulation_suffix,
            imported_root_suffix=imported_root_suffix,
            articulation_paths=prim_paths_from_suffix(env_roots, articulation_suffix),
            imported_root_paths=prim_paths_from_suffix(env_roots, imported_root_suffix),
            asset_path=imported["asset_path"],
            asset_type=execution.robot.asset_type,
            controlled_joints=tuple(execution.controlled_joints),
            gravity_policy=imported["gravity_policy"],
            gravity_counts=imported["gravity_counts"],
            solver_counts=imported["solver_counts"],
            robot_id=instance.robot_id,
            label=instance.label,
            kind=kind.value,
            supports_planning=bool(kind.has_arm and binding.enabled),
        )
        result[instance.label] = tiled_robot
        _print_status(
            status_prefix,
            "ROBOT "
            f"label={instance.label} profile={instance.profile_name} "
            f"controller_profile={controller_profile} "
            f"articulation_suffix={articulation_suffix} "
            f"asset={tiled_robot.asset_path} gravity={tiled_robot.gravity_counts} "
            f"solver={tiled_robot.solver_counts}",
        )
    return result


def _import_robot_to_env_zero(
    *,
    stage: object,
    env_config: Mapping[str, object],
    robot_execution: RobotExecutionConfig,
    controller_profiles: ControllerProfiles,
) -> dict[str, object]:
    """执行单个机器人 reset 前的 USD/PhysX 导入和覆盖。"""

    articulation_path, asset_path, imported_root_path = import_robot_asset(
        robot_execution.robot
    )
    apply_root_pose(stage, imported_root_path, robot_execution.root_pose)
    physx_configs = robot_execution.robot.physx_overrides.apply_to_configs(
        physx_override_configs(controller_profiles)
    )
    apply_robot_usd_overrides(
        imported_root_path,
        physx_configs,
        driven_joint_names=tuple(robot_execution.controlled_joints),
        mjcf_path=asset_path if robot_execution.robot.asset_type == "mjcf" else None,
        mimic_path=(
            asset_path if robot_execution.robot.asset_type in {"mjcf", "urdf"} else None
        ),
        component_mapping=robot_execution.robot.component_mapping,
        native_mimic=robot_execution.robot.asset_type == "urdf",
    )
    solver_config = merge_solver_configs(
        scene_solver_settings(env_config),
        robot_execution.robot.solver_iterations,
    )
    solver_counts = (
        apply_solver_iteration_overrides(
            stage,
            articulation_path,
            solver_config,
            component_mapping=robot_execution.robot.component_mapping,
        )
        if solver_config is not None
        else {"configured": 0}
    )
    gravity_policy = robot_execution.robot.gravity_policy
    gravity_counts = apply_robot_gravity_policy(
        imported_root_path,
        gravity_policy,
        component_mapping=robot_execution.robot.component_mapping,
    )
    return {
        "articulation_path": articulation_path,
        "imported_root_path": imported_root_path,
        "asset_path": asset_path,
        "gravity_policy": gravity_policy,
        "gravity_counts": gravity_counts,
        "solver_counts": solver_counts,
    }
