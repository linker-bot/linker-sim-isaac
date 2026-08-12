"""单份 USD prototype 到 Newton 多 world 场景的装配器。

本模块只拥有资产拓扑、world 布局和 raw Newton view 绑定；它不知道强化学习任务、动作、
奖励或 Gymnasium。USD stage 只物化 source env，其他环境由 NewtonRuntime 在同一 CUDA
model 中复制为彼此隔离的 worlds。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np

from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.isaac.physics.newton.views import (
    NewtonArticulationView,
    NewtonRigidBodyView,
)

from .assets import (
    define_source_environment,
    import_source_objects,
    import_source_robots,
    source_object_configs,
    validate_single_dynamic_rigid_object,
)
from .layout import (
    environment_origins,
    environment_root_paths,
    paths_from_suffix,
    relative_prim_suffix,
)
from .types import ImportedReplicatedRobot, ReplicatedNewtonScene
from .views import finalize_replicated_robot_views


@dataclass(frozen=True, slots=True)
class _NewtonManagerRobotTopology:
    """manager finalize 阶段需要的最小 replicated robot 描述。

    source robot 只含 env_0 的导入路径；NewtonRuntime 的 equality 审计却必须按
    world 精确解析每个 replica 的 joint label。这个冷路径值对象在 model finalize 前补齐
    全部 root paths，同时不伪造一个尚未创建的 articulation view。
    """

    asset_path: object
    imported_root_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NewtonWorldPlan:
    """由 Newton builder 派生的 multi-world 执行计划。"""

    env_root_paths: tuple[str, ...]
    env_origins: np.ndarray

    @classmethod
    def from_environment_settings(
        cls,
        settings: object,
        *,
        num_envs: int,
    ) -> "NewtonWorldPlan":
        roots = environment_root_paths(settings, num_envs=num_envs)
        # Newton 用 world id 隔离接触；所有 world 共址既避免无意义的大坐标，也明确表明
        # 零间距是 multi-world 后端合同，不是用户可选择的复制策略 selector。
        origins = environment_origins(
            settings,
            num_envs=num_envs,
            spacing_m=0.0,
        )
        return cls(env_root_paths=roots, env_origins=origins)

    @property
    def world_count(self) -> int:
        return len(self.env_root_paths)


def build_replicated_newton_scene(
    *,
    stage: object,
    runtime: object,
    scene_settings: object,
    environment_settings: object,
    num_envs: int,
    dynamic_object_name: str,
    controller_bundle: str,
    controller_bundles: Mapping[str, ControllerProfiles],
    prepare_newton_render_topology: bool,
) -> ReplicatedNewtonScene:
    """导入一次 source env，并由唯一 NewtonRuntime finalize 多 world model。"""

    if getattr(runtime, "kind", None) != "newton_cuda":
        raise TypeError("Newton replicated scene requires newton_cuda runtime")
    initialize = getattr(runtime, "initialize_worlds", None)
    if not callable(initialize):
        raise TypeError("newton_cuda runtime does not expose initialize_worlds")

    plan = NewtonWorldPlan.from_environment_settings(
        environment_settings,
        num_envs=num_envs,
    )
    roots = plan.env_root_paths
    origins = plan.env_origins
    source_root = roots[0]
    object_configs = source_object_configs(scene_settings, env_root=source_root)
    validate_single_dynamic_rigid_object(
        object_configs,
        expected_name=dynamic_object_name,
    )
    # 在 stage 与 manager model 都未发生变更时拒绝不完整的对象状态闭包。
    define_source_environment(
        stage,
        source_root,
        prepare_newton_render_topology=prepare_newton_render_topology,
    )
    object_handles = import_source_objects(
        stage,
        configs=object_configs,
        physics_backend="newton",
        prepare_newton_render_topology=prepare_newton_render_topology,
    )
    source_robots = import_source_robots(
        stage,
        scene_settings=scene_settings,
        env_root=source_root,
        controller_bundle=controller_bundle,
        controller_bundles=controller_bundles,
        solver_type=None,
        physics_backend="newton",
        prepare_newton_render_topology=prepare_newton_render_topology,
        object_configs=object_configs,
    )

    # 先从唯一 USD prototype 派生全部 Newton world path。manager 在 finalize 后立即审计
    # native equality，因此此时就必须拿到 N 条 root path；不能把只有 env_0 的
    # SourceReplicatedRobot 传进去，再等 view 构造阶段才扩展。
    robot_layouts: list[
        tuple[object, tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    ] = []
    manager_robots: dict[str, _NewtonManagerRobotTopology] = {}
    for source in source_robots:
        articulation_paths = paths_from_suffix(
            roots,
            relative_prim_suffix(source_root, source.articulation_path),
        )
        imported_root_paths = paths_from_suffix(
            roots,
            relative_prim_suffix(source_root, source.imported_root_path),
        )
        tcp_body_paths = paths_from_suffix(
            roots,
            relative_prim_suffix(source_root, source.tcp_parent_body_path),
        )
        robot_layouts.append(
            (source, articulation_paths, imported_root_paths, tcp_body_paths)
        )
        manager_robots[source.label] = _NewtonManagerRobotTopology(
            asset_path=source.asset_path,
            imported_root_paths=imported_root_paths,
        )

    # NewtonRuntime 必须在所有 prototype 资产就位后一次性 finalize。提前创建 view
    # 会把它绑定到不完整或即将失效的 model/state/control buffer。
    initialize(
        env_root_paths=roots,
        env_origins=origins,
        robots=manager_robots,
        object_handles=object_handles,
    )
    if int(getattr(runtime, "world_count", -1)) != plan.world_count:
        raise RuntimeError(
            "newton_cuda finalized an unexpected world count: "
            "actual="
            f"{getattr(runtime, 'world_count', None)!r}, expected={plan.world_count}"
        )

    world_indices = tuple(range(num_envs))
    robots: list[ImportedReplicatedRobot] = []
    for (
        source,
        articulation_paths,
        imported_root_paths,
        tcp_body_paths,
    ) in robot_layouts:
        articulation_view = NewtonArticulationView(
            runtime,
            paths=articulation_paths,
            world_indices=world_indices,
            name=f"kaleidoscope_{source.label}_newton_articulation",
        )
        robots.append(
            ImportedReplicatedRobot(
                robot_id=source.robot_id,
                label=source.label,
                profile_name=source.profile_name,
                profile=source.profile,
                controller_bundle_name=source.controller_bundle_name,
                controller_profiles=source.controller_profiles,
                execution=source.execution,
                asset_path=source.asset_path,
                asset_type=source.asset_type,
                articulation_paths=articulation_paths,
                imported_root_paths=imported_root_paths,
                controlled_joints=source.controlled_joints,
                tcp_frame_name=source.tcp_frame_name,
                tcp_parent_frame_name=source.tcp_parent_frame_name,
                tcp_body_paths=tcp_body_paths,
                tcp_offset_xyz=source.tcp_offset_xyz,
                tcp_offset_rpy=source.tcp_offset_rpy,
                articulation_view=articulation_view,
            )
        )

    object_paths = {
        config.name: paths_from_suffix(
            roots,
            relative_prim_suffix(source_root, config.prim_path),
        )
        for config in object_configs
    }
    scene = ReplicatedNewtonScene(
        env_root_paths=roots,
        env_origins=np.ascontiguousarray(origins, dtype=np.float32),
        robots=tuple(robots),
        object_handles=object_handles,
        object_prim_paths=object_paths,
    )
    finalized = finalize_replicated_robot_views(scene)
    assert isinstance(finalized, ReplicatedNewtonScene)
    return finalized


def create_newton_tcp_rigid_views(
    scene: ReplicatedNewtonScene,
    *,
    runtime: object,
) -> dict[str, NewtonRigidBodyView]:
    """为每个机器人创建覆盖全部 worlds 的 physical TCP parent view。"""

    worlds = tuple(range(scene.num_envs))
    return {
        robot.label: NewtonRigidBodyView(
            runtime,
            paths=robot.tcp_body_paths,
            world_indices=worlds,
            name=f"kaleidoscope_{robot.label}_newton_tcp",
        )
        for robot in scene.robots
    }


def create_newton_dynamic_object_rigid_view(
    scene: ReplicatedNewtonScene,
    *,
    runtime: object,
    object_name: str,
) -> NewtonRigidBodyView:
    """为任务唯一动态 FREE rigid body 创建覆盖全部 worlds 的 exact-path view。"""

    handles = tuple(
        handle for handle in scene.object_handles if handle.name == object_name
    )
    if len(handles) != 1:
        raise RuntimeError(
            f"dynamic object {object_name!r} must match exactly one imported object"
        )
    handle = handles[0]
    if handle.kind != "rigid" or bool(getattr(handle.model, "static", False)):
        raise RuntimeError(
            f"Kaleidoscope dynamic object {object_name!r} must be a non-static rigid body"
        )
    try:
        paths = tuple(scene.object_prim_paths[object_name])
    except KeyError as exc:
        raise RuntimeError(
            f"dynamic object {object_name!r} has no replicated prim paths"
        ) from exc
    return NewtonRigidBodyView(
        runtime,
        paths=paths,
        world_indices=tuple(range(scene.num_envs)),
        name=f"kaleidoscope_object_{_identifier(object_name)}_newton",
    )


def newton_command_joint_limits(
    robot: ImportedReplicatedRobot,
    *,
    runtime: object,
    device: object,
) -> tuple[object, object]:
    """从 Newton model 的 owner arrays 取得首个 world 的 command-space 上下限。

    该函数只在启动冷路径调用；Warp array 通过 ``wp.to_torch`` 零拷贝映射到同一 CUDA
    device，再由 command index 选择所需列，不允许先转 NumPy。
    """

    import torch
    import warp as wp

    view = robot.articulation_view
    command_indices = robot.command_joint_indices
    if command_indices is None:
        raise RuntimeError("Newton robot command binding has not been finalized")
    qd_rows = np.asarray(view.binding.qd_indices, dtype=np.int64)
    if qd_rows.ndim != 2 or qd_rows.shape[0] != robot.articulation_view.count:
        raise RuntimeError(
            "Newton articulation returned invalid generalized DOF mapping"
        )
    first_world_global = torch.as_tensor(
        qd_rows[0],
        device=device,
        dtype=torch.int64,
    )
    selected = first_world_global.index_select(
        0,
        torch.as_tensor(command_indices, device=device, dtype=torch.int64),
    )
    model = getattr(runtime, "model", None)
    if model is None:
        raise RuntimeError("newton_cuda runtime has no finalized model")
    owner_stream = view.owner_stream
    external = torch.cuda.ExternalStream(
        int(owner_stream.cuda_stream),
        device=torch.device(device),
    )
    caller = torch.cuda.current_stream(torch.device(device))
    external.wait_stream(caller)
    with torch.cuda.stream(external):
        lower = wp.to_torch(getattr(model, "joint_limit_lower"))
        upper = wp.to_torch(getattr(model, "joint_limit_upper"))
        if lower.device != torch.device(device) or upper.device != torch.device(device):
            raise RuntimeError(
                "Newton joint-limit buffers are not on the configured device"
            )
        selected_lower = lower.index_select(0, selected).to(dtype=torch.float32)
        selected_upper = upper.index_select(0, selected).to(dtype=torch.float32)
    caller.wait_stream(external)
    return selected_lower, selected_upper


def _identifier(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in value)
    return result or "object"


__all__ = [
    "NewtonWorldPlan",
    "build_replicated_newton_scene",
    "create_newton_dynamic_object_rigid_view",
    "create_newton_tcp_rigid_views",
    "newton_command_joint_limits",
]
