"""replicated PhysX articulation/rigid raw view 的冷路径装配。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from linkerbot_sim.isaac.physics.core_api import create_rigid_prim_core_view
from linkerbot_sim.robots.joint_groups import resolve_joint_indices
from linkerbot_sim.robots.mimic.runtime import resolve_mimic_follower_controls

from .types import ReplicatedNewtonScene, ReplicatedPhysxScene


def finalize_replicated_robot_views(
    scene: ReplicatedPhysxScene | ReplicatedNewtonScene,
) -> ReplicatedPhysxScene | ReplicatedNewtonScene:
    """冻结 articulation command columns，并绑定唯一 target writers。

    原生 URDF/MJCF mimic follower 由 PhysX 约束执行，不能再作为独立 action 列；否则
    上层 target 与约束会同时驱动同一自由度。这里沿用资产声明计算 follower 集合，最终
    command order 始终是 raw articulation ``dof_names`` 的稳定子序列。
    """

    robots = []
    for robot in scene.robots:
        names = tuple(str(name) for name in robot.articulation_view.dof_names)
        requested = resolve_joint_indices(names, list(robot.controlled_joints))
        mimic_path = robot.asset_path if robot.asset_type in {"mjcf", "urdf"} else None
        followers = {
            item.dependent_joint
            for item in resolve_mimic_follower_controls(names, mimic_path)
        }
        indices = np.asarray(
            [int(index) for index in requested if names[int(index)] not in followers],
            dtype=np.int64,
        )
        if indices.size == 0:
            raise RuntimeError(
                f"robot {robot.label!r} has no active command joints after mimic filtering"
            )
        command_names = tuple(names[int(index)] for index in indices)
        bind_controllable = getattr(
            robot.articulation_view,
            "bind_controllable_dofs",
            None,
        )
        if callable(bind_controllable):
            # Newton native equality follower 只有 solver 一个执行者。把过滤后的 command
            # names 显式绑定到 raw view，后续任何 follower target 写入都会 fail closed。
            bind_controllable(command_names)
        prepare = getattr(robot.articulation_view, "prepare_dof_selection", None)
        if callable(prepare):
            # selector 与 Warp staging 在启动冷路径预分配，训练 step 不得首次分配或把
            # CUDA selector 搬回 host。
            prepare(dof_indices=tuple(int(index) for index in indices))
        robots.append(
            robot.with_command_binding(
                names=command_names,
                indices=indices,
            )
        )
    return replace(scene, robots=tuple(robots))


def create_tcp_rigid_views(scene: ReplicatedPhysxScene) -> dict[str, object]:
    """为每个 robot 的 physical TCP parent link 创建 GPU raw rigid view。"""

    result: dict[str, object] = {}
    for robot in scene.robots:
        result[robot.label] = create_rigid_prim_core_view(
            paths=robot.tcp_body_paths,
            name=f"kaleidoscope_{robot.label}_tcp",
            physics_backend="physx",
        )
    return result


def create_dynamic_object_rigid_view(
    scene: ReplicatedPhysxScene,
    *,
    object_name: str,
) -> object:
    """创建 task 唯一动态刚体对象的 GPU raw view，并拒绝静态/链式对象。"""

    handles = [handle for handle in scene.object_handles if handle.name == object_name]
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
    return create_rigid_prim_core_view(
        paths=paths,
        name=f"kaleidoscope_object_{_identifier(object_name)}",
        physics_backend="physx",
    )


def command_joint_limits(robot: object) -> object:
    """读取 PhysX 冷态 DOF limits，并返回 command columns 的 host 索引。

    Isaac Sim 6 可能在 CUDA articulation view 上仍返回 CPU limits；调用方只允许在场景
    装配期把这份结构元数据复制一次到 GPU，不能把该行为复用到逐步状态读取。
    """

    indices = getattr(robot, "command_joint_indices", None)
    if indices is None:
        raise RuntimeError("replicated robot views have not been finalized")
    limits = robot.articulation_view.get_dof_limits()
    if limits is None:
        raise RuntimeError(f"robot {robot.label!r} did not expose PhysX DOF limits")
    return limits, np.ascontiguousarray(indices, dtype=np.int64)


def _identifier(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in value)
    return result or "object"


__all__ = [
    "command_joint_limits",
    "create_dynamic_object_rigid_view",
    "create_tcp_rigid_views",
    "finalize_replicated_robot_views",
]
