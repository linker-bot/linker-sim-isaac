"""canonical ``SingleSceneRuntime`` 的快照读取、事务恢复与目标描述。

SingleSceneRuntime 只有一个 scene，因此对象位姿按 ``scene-local`` 保存；机器人通过稳定 label
匹配，不能依赖会话内 robot ID。恢复前会一次性完成兼容性检查并采集所有回滚值，首次
PhysX 写入后发生的异常则交给补偿事务处理。
"""

from __future__ import annotations

from collections.abc import Mapping

from linkerbot_sim.snapshots.compatibility import (
    SnapshotTargetDescriptor,
    require_snapshot_compatibility,
)
from linkerbot_sim.snapshots.runtime_objects import (
    _imported_asset_fingerprint,
    _reset_execution_observers,
    _restore_robot_snapshot_to_execution,
    _restore_runtime_objects,
    _robot_snapshot_from_execution,
    _robot_target_from_execution,
    _runtime_object_snapshots,
    _runtime_object_targets,
)
from linkerbot_sim.snapshots.schema import (
    SimulationSnapshot,
    SnapshotMetadata,
    SnapshotRestoreResult,
)
from linkerbot_sim.snapshots.transactions import (
    mutation_transaction,
    require_runtime_mutable,
)


def get_single_scene_snapshot(runtime: object) -> SimulationSnapshot:
    """按稳定 label 读取 N-robot ``SingleSceneRuntime`` 的完整逻辑状态。

    返回值包含 command joint 的位置、速度及控制 target，也包含 runtime object 的根/子
    刚体局部位姿。这里读取的是恢复所需状态，而不是可直接写回 Isaac 的底层句柄。
    """

    robots = {}
    for robot_id, robot_runtime in runtime.robots_by_id.items():
        robots[robot_runtime.label] = _robot_snapshot_from_execution(
            label=robot_runtime.label,
            robot_id=robot_id,
            execution=robot_runtime.execution,
            robot_profile=robot_runtime.profile_name,
            asset_fingerprint=_imported_asset_fingerprint(robot_runtime.imported),
        )
    return SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="single_scene",
            coordinate_frame="scene-local",
            info={
                "robot_labels": [robot.label for robot in robots.values()],
                "config_fingerprint": getattr(runtime, "config_fingerprint", None),
            },
        ),
        robots=robots,
        objects=_runtime_object_snapshots(
            stage=getattr(runtime.session, "stage", None),
            handles=getattr(runtime, "object_handles", ()),
            state_views=getattr(runtime, "object_state_views", {}),
        ),
    )


def set_single_scene_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot | Mapping[str, object],
    *,
    label_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """按稳定 label 事务式恢复快照；不完整回滚会永久 fail-stop。

    ``label_map`` 显式描述 source label 到 scene label 的映射；``strict`` 决定缺少关节或
    对象时是拒绝整次操作还是返回 partial 结果。无论哪种模式，兼容性检查与回滚快照采集
    都在首次写入前完成，避免已知错误造成半恢复。
    """

    require_runtime_mutable(runtime, operation="set_single_scene_snapshot")
    parsed = _snapshot_from_input(snapshot)
    target = single_scene_target_descriptor(runtime)
    compatibility = require_snapshot_compatibility(
        parsed,
        target,
        label_map=label_map,
        strict=strict,
    )
    # 原始快照与控制器 effort cache 必须在首个 articulation/PhysX setter 前全部捕获。
    # 否则后续机器人的“旧值”可能已经混入本次恢复写入，失去事务基准。
    original = get_single_scene_snapshot(runtime)
    original_compatibility = require_snapshot_compatibility(
        original,
        target,
        strict=True,
    )
    missing_object_state = set(compatibility.object_mappings).difference(
        original_compatibility.object_mappings
    )
    if missing_object_state:
        raise RuntimeError(
            "cannot capture rollback state for scene objects: "
            f"{sorted(missing_object_state)}"
        )
    effort_caches = {
        label: _controller_effort_cache(runtime.robot_by_label(label).execution)
        for label in compatibility.robot_mappings
    }

    restored: list[str] = []
    restored_objects: tuple[str, ...] = ()
    with mutation_transaction(
        runtime, operation="set_single_scene_snapshot"
    ) as transaction:
        for target_label, mapping in compatibility.robot_mappings.items():
            robot = runtime.robot_by_label(target_label)
            original_mapping = original_compatibility.robot_mappings[target_label]
            transaction.add_rollback(
                f"robot {target_label}",
                lambda robot=robot, original_robot=original.robots[original_mapping.source_label], original_mapping=original_mapping, effort_cache=effort_caches[target_label]: (
                    _restore_scene_robot(
                        robot.execution,
                        original_robot,
                        mapping=original_mapping,
                        effort_cache=effort_cache,
                    )
                ),
            )
            _restore_robot_snapshot_to_execution(
                robot.execution,
                parsed.robots[mapping.source_label],
                mapping=mapping,
            )
            restored.append(target_label)
        if compatibility.object_mappings:
            transaction.add_rollback(
                "scene objects",
                lambda: _restore_scene_objects(
                    runtime,
                    original,
                    compatibility=original_compatibility,
                    expected=tuple(compatibility.object_mappings),
                ),
            )
            restored_objects = _restore_scene_objects(
                runtime,
                parsed,
                compatibility=compatibility,
                expected=tuple(compatibility.object_mappings),
            )
        # observer/cache reset 与碰撞缓存失效无法仅靠快照重建，因此先标记不可逆。
        transaction.mark_irreversible("execution observer cache reset")
        for robot in runtime.robots_by_id.values():
            _reset_execution_observers(robot.execution)
        collision_registry = getattr(runtime, "collision_registry", None)
        mark_dirty = getattr(collision_registry, "mark_dirty", None)
        if callable(mark_dirty):
            transaction.mark_irreversible("collision registry invalidation")
            mark_dirty()
    return SnapshotRestoreResult(
        accepted=True,
        robots=tuple(restored),
        objects=restored_objects,
        partial=compatibility.partial,
    )


def single_scene_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """按稳定 label 描述 scene 快照恢复目标，不暴露 session robot ID。

    descriptor 只携带兼容性判断所需的 profile、asset fingerprint、关节名和对象 body 名；
    它不包含可变物理状态，可在真正恢复前安全构建。
    """

    robots = {
        robot.label: _robot_target_from_execution(
            label=robot.label,
            execution=robot.execution,
            robot_profile=robot.profile_name,
            asset_fingerprint=_imported_asset_fingerprint(robot.imported),
        )
        for robot in runtime.robots_by_id.values()
    }
    return SnapshotTargetDescriptor(
        runtime_kind="single_scene",
        robots=robots,
        objects=_runtime_object_targets(getattr(runtime, "object_handles", ())),
    )


def _snapshot_from_input(
    snapshot: SimulationSnapshot | Mapping[str, object],
) -> SimulationSnapshot:
    """接受已解析快照或 canonical JSON mapping，并统一返回 schema 对象。"""

    if isinstance(snapshot, SimulationSnapshot):
        return snapshot
    if isinstance(snapshot, Mapping):
        return SimulationSnapshot.from_mapping(snapshot)
    raise ValueError("snapshot must be a SimulationSnapshot or JSON object")


def _controller_effort_cache(execution: object) -> object:
    """复制可选 effort 缓存，使回滚同时恢复控制器的后续行为。

    只恢复 articulation 状态仍可能让下一控制步使用新缓存，因此缓存属于事务状态。测试
    adapter 可能没有该属性，此时用 sentinel 区分“属性缺失”和“属性值为 None”。
    """

    controller = execution.joint_controller
    if not hasattr(controller, "last_commanded_efforts"):
        return _missing
    value = getattr(controller, "last_commanded_efforts")
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value


def _restore_scene_robot(
    execution: object,
    source_robot: object,
    *,
    mapping: object,
    effort_cache: object,
) -> None:
    """把单机器人物理状态与控制器缓存作为一个补偿动作恢复。"""

    _restore_robot_snapshot_to_execution(execution, source_robot, mapping=mapping)
    if effort_cache is not _missing:
        value = effort_cache
        copy = getattr(value, "copy", None)
        setattr(
            execution.joint_controller,
            "last_commanded_efforts",
            copy() if callable(copy) else value,
        )


def _restore_scene_objects(
    runtime: object,
    snapshot: SimulationSnapshot,
    *,
    compatibility: object,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    """恢复全部请求对象，并拒绝底层 adapter 静默少写。

    即使 compatibility 层已经确认对象存在，底层 view 仍可能没有实际写入；这里比较
    ``expected`` 与返回值，把这种情况提升为事务异常并触发回滚。
    """

    restored = _restore_runtime_objects(
        runtime,
        snapshot,
        compatibility=compatibility,
    )
    missing = set(expected).difference(restored)
    if missing:
        raise RuntimeError(f"scene object restore did not write: {sorted(missing)}")
    return restored


_missing = object()


__all__ = [
    "get_single_scene_snapshot",
    "single_scene_target_descriptor",
    "set_single_scene_snapshot",
]
