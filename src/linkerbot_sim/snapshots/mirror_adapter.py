"""canonical Mirror 场景资源的 CPU/NumPy 快照读取与事务恢复。

MirrorSceneResources 只有一个 scene，因此对象位姿按 ``scene-local`` 保存；机器人通过稳定 label
匹配，不能依赖会话内 robot ID。恢复前会一次性完成兼容性检查并采集所有回滚值，首次
PhysX 写入后发生的异常则交给补偿事务处理。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

from linkerbot_sim.snapshots.compatibility import (
    SnapshotTargetDescriptor,
    require_snapshot_compatibility,
)
from linkerbot_sim.snapshots.runtime_objects import (
    COMMAND_TARGET_MODES_INFO_KEY,
    _command_joint_names,
    _command_target_modes,
    _imported_asset_fingerprint,
    _preflight_runtime_object_restore,
    _reset_execution_observers,
    _restore_robot_snapshot_to_execution,
    _restore_runtime_objects,
    _robot_snapshot_from_execution,
    _robot_target_from_execution,
    _runtime_object_snapshots,
    _runtime_object_targets,
    _snapshot_command_target_modes,
)
from linkerbot_sim.snapshots.schema import (
    SceneSnapshot,
    SnapshotMetadata,
    SnapshotRestoreResult,
)
from linkerbot_sim.snapshots.transactions import (
    mutation_transaction,
    require_runtime_mutable,
)
from linkerbot_sim.controllers.control_mode import (
    require_control_mode,
    require_expected_generation,
)


NEWTON_SOLVER_STATE_INFO_KEY = "linkerbot.snapshot.newton_solver_integration_state"
CONTROL_MODE_INFO_KEY = "linkerbot.snapshot.control_mode"
CONTROLLER_PROFILE_FINGERPRINTS_INFO_KEY = (
    "linkerbot.snapshot.controller_profile_fingerprints"
)


@dataclass(frozen=True)
class _NewtonSolverRestorePlan:
    """首个物理写入前冻结的 Newton persistent-state 恢复动作。"""

    payload: Mapping[str, object] | None
    validate: Callable[[Mapping[str, object]], None]
    restore: Callable[[Mapping[str, object]], None]
    reset_to_baseline: Callable[[], None]

    def apply(self) -> None:
        """Newton-origin 恢复 payload；跨引擎恢复则使用 committed baseline。"""

        if self.payload is None:
            self.reset_to_baseline()
            return
        self.restore(self.payload)


def get_mirror_snapshot(runtime: object) -> SceneSnapshot:
    """按稳定 label 读取 N-robot ``MirrorSceneResources`` 的完整逻辑状态。

    返回值包含 command joint 的位置、速度及控制 target，也包含 runtime object 的根/子
    刚体局部位姿。这里读取的是恢复所需状态，而不是可直接写回 Isaac 的底层句柄。
    """

    robots = {}
    command_target_modes = {}
    for robot_id, robot_runtime in runtime.robots_by_id.items():
        robots[robot_runtime.label] = _robot_snapshot_from_execution(
            label=robot_runtime.label,
            robot_id=robot_id,
            execution=robot_runtime.execution,
            robot_profile=robot_runtime.profile_name,
            asset_fingerprint=_imported_asset_fingerprint(robot_runtime.imported),
        )
        controller = robot_runtime.execution.joint_controller
        names = robots[robot_runtime.label].command_joint_names
        modes = _command_target_modes(controller, command_count=len(names))
        command_target_modes[robot_runtime.label] = dict(zip(names, modes, strict=True))
    metadata_info: dict[str, object] = {
        "robot_labels": [robot.label for robot in robots.values()],
        "config_fingerprint": getattr(runtime, "config_fingerprint", None),
        COMMAND_TARGET_MODES_INFO_KEY: command_target_modes,
    }
    control_state = _runtime_control_mode_state(runtime)
    if control_state is not None:
        metadata_info[CONTROL_MODE_INFO_KEY] = control_state
    controller_fingerprints = {
        robot.label: fingerprint
        for robot in runtime.robots_by_id.values()
        if (
            fingerprint := getattr(
                robot,
                "controller_profile_fingerprint",
                None,
            )
        )
        is not None
    }
    if controller_fingerprints:
        metadata_info[CONTROLLER_PROFILE_FINGERPRINTS_INFO_KEY] = (
            controller_fingerprints
        )
    solver_state = _capture_newton_solver_state(runtime)
    if solver_state is not None:
        metadata_info[NEWTON_SOLVER_STATE_INFO_KEY] = solver_state
    return SceneSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="mirror",
            coordinate_frame="scene-local",
            info=metadata_info,
        ),
        robots=robots,
        objects=_runtime_object_snapshots(
            stage=getattr(runtime.session, "stage", None),
            handles=getattr(runtime, "object_handles", ()),
            state_views=getattr(runtime, "object_state_views", {}),
        ),
    )


def set_mirror_snapshot(
    runtime: object,
    snapshot: SceneSnapshot | Mapping[str, object],
    *,
    label_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """按稳定 label 事务式恢复快照；不完整回滚会永久 fail-stop。

    ``label_map`` 显式描述 source label 到 scene label 的映射；``strict`` 决定缺少关节或
    对象时是拒绝整次操作还是返回 partial 结果。无论哪种模式，兼容性检查与回滚快照采集
    都在首次写入前完成，避免已知错误造成半恢复。
    """

    require_runtime_mutable(runtime, operation="set_mirror_snapshot")
    parsed = _snapshot_from_input(snapshot)
    _validate_snapshot_control_mode(runtime, parsed)
    target = mirror_target_descriptor(runtime)
    compatibility = require_snapshot_compatibility(
        parsed,
        target,
        label_map=label_map,
        strict=strict,
    )
    _validate_controller_profile_fingerprints(
        runtime,
        parsed,
        compatibility=compatibility,
    )
    restore_modes = _validated_restore_command_modes(
        runtime,
        parsed,
        compatibility=compatibility,
    )
    _preflight_runtime_object_restore(
        runtime,
        parsed,
        compatibility=compatibility,
    )
    solver_restore = _preflight_newton_solver_state(runtime, parsed)
    # 原始快照与控制器 cache 必须在首个 articulation/physics setter 前全部捕获。
    # 否则后续机器人的“旧值”可能已经混入本次恢复写入，失去事务基准。
    original = get_mirror_snapshot(runtime)
    original_compatibility = require_snapshot_compatibility(
        original,
        target,
        strict=True,
    )
    _preflight_runtime_object_restore(
        runtime,
        original,
        compatibility=original_compatibility,
    )
    missing_object_state = set(compatibility.object_mappings).difference(
        original_compatibility.object_mappings
    )
    if missing_object_state:
        raise RuntimeError(
            "cannot capture rollback state for scene objects: "
            f"{sorted(missing_object_state)}"
        )
    controller_caches = {
        label: _controller_cache(runtime.robot_by_label(label).execution)
        for label in compatibility.robot_mappings
    }
    original_solver_state: Mapping[str, object] | None = None
    if solver_restore is not None:
        captured = original.metadata.info.get(NEWTON_SOLVER_STATE_INFO_KEY)
        if not isinstance(captured, Mapping):
            raise RuntimeError(
                "cannot capture rollback state for Newton solver integration"
            )
        # rollback payload 同样必须在首个 articulation setter 前通过目标 runtime 校验；
        # 否则后续失败时才发现补偿动作不可执行，会把可预知错误升级为 fail-stop。
        original_solver_state = deepcopy(dict(captured))
        solver_restore.validate(original_solver_state)

    restored: list[str] = []
    restored_objects: tuple[str, ...] = ()
    with mutation_transaction(runtime, operation="set_mirror_snapshot") as transaction:
        for target_label, mapping in compatibility.robot_mappings.items():
            robot = runtime.robot_by_label(target_label)
            original_mapping = original_compatibility.robot_mappings[target_label]
            transaction.add_rollback(
                f"robot {target_label}",
                lambda robot=robot, original_robot=original.robots[original_mapping.source_label], original_mapping=original_mapping, command_modes=restore_modes[target_label], controller_cache=controller_caches[target_label]: (
                    _restore_scene_robot(
                        robot.execution,
                        original_robot,
                        mapping=original_mapping,
                        command_modes=command_modes,
                        controller_cache=controller_cache,
                    )
                ),
            )
            _restore_robot_snapshot_to_execution(
                robot.execution,
                parsed.robots[mapping.source_label],
                mapping=mapping,
                command_modes=restore_modes[target_label],
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
        if solver_restore is not None:
            assert original_solver_state is not None
            transaction.add_rollback(
                "Newton solver integration state",
                lambda: solver_restore.restore(original_solver_state),
            )
            solver_restore.apply()
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


def _capture_newton_solver_state(runtime: object) -> dict[str, object] | None:
    """在 Mirror 显式快照边界捕获 Newton-only persistent solver state。"""

    session = getattr(runtime, "session", None)
    physics = getattr(session, "physics_runtime", None)
    capture = getattr(physics, "capture_solver_integration_state_host", None)
    if not callable(capture):
        return None
    payload = capture()
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "Newton capture_solver_integration_state_host must return a mapping"
        )
    return deepcopy(dict(payload))


def _preflight_newton_solver_state(
    runtime: object,
    snapshot: SceneSnapshot,
) -> _NewtonSolverRestorePlan | None:
    """冻结 Newton 冷恢复动作；PhysX-origin 快照显式选择 baseline reset。"""

    session = getattr(runtime, "session", None)
    physics = getattr(session, "physics_runtime", None)
    capture = getattr(physics, "capture_solver_integration_state_host", None)
    validate = getattr(physics, "validate_solver_integration_state_host", None)
    restore = getattr(physics, "set_solver_integration_state_host", None)
    reset = getattr(physics, "reset_solver_integration_state_host", None)
    methods = {
        "capture_solver_integration_state_host": capture,
        "validate_solver_integration_state_host": validate,
        "set_solver_integration_state_host": restore,
        "reset_solver_integration_state_host": reset,
    }
    if not any(callable(method) for method in methods.values()):
        return None
    missing = sorted(name for name, method in methods.items() if not callable(method))
    if missing:
        raise RuntimeError(
            "Newton runtime must expose the complete Mirror solver integration API; "
            f"missing={missing}"
        )
    assert callable(validate) and callable(restore) and callable(reset)
    payload = snapshot.metadata.info.get(NEWTON_SOLVER_STATE_INFO_KEY)
    if payload is None:
        # PhysX 没有 Newton persistent state。保留跨引擎恢复，但必须清除目标 Newton
        # runtime 的旧时间线，不能让恢复结果依赖调用前已经运行了多少步。
        return _NewtonSolverRestorePlan(
            payload=None,
            validate=validate,
            restore=restore,
            reset_to_baseline=reset,
        )
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"metadata.info[{NEWTON_SOLVER_STATE_INFO_KEY!r}] must be an object"
        )
    validate(payload)
    return _NewtonSolverRestorePlan(
        payload=deepcopy(dict(payload)),
        validate=validate,
        restore=restore,
        reset_to_baseline=reset,
    )


def mirror_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
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
        runtime_kind="mirror",
        robots=robots,
        objects=_runtime_object_targets(getattr(runtime, "object_handles", ())),
    )


def _snapshot_from_input(
    snapshot: SceneSnapshot | Mapping[str, object],
) -> SceneSnapshot:
    """接受已解析快照或 canonical JSON mapping，并统一返回 schema 对象。"""

    if isinstance(snapshot, SceneSnapshot):
        return snapshot
    if isinstance(snapshot, Mapping):
        return SceneSnapshot.from_mapping(snapshot)
    raise ValueError("snapshot must be a SceneSnapshot or JSON object")


def _runtime_control_mode_state(runtime: object) -> dict[str, object] | None:
    """Read the product-owned global mode without touching an engine handle."""

    provider = getattr(runtime, "control_mode_state_provider", None)
    if not callable(provider):
        return None
    state = provider()
    if isinstance(state, Mapping):
        active_mode = state.get("active_mode")
        generation = state.get("generation")
    else:
        active_mode = getattr(state, "active_mode", None)
        generation = getattr(state, "generation", None)
    return {
        "active_mode": require_control_mode(
            active_mode,
            label="runtime active_mode",
        ),
        "generation": require_expected_generation(generation),
    }


def _validate_snapshot_control_mode(
    runtime: object,
    snapshot: SceneSnapshot,
) -> None:
    """Reject cross-mode restore before any compatibility or engine mutation."""

    runtime_state = _runtime_control_mode_state(runtime)
    if runtime_state is None:
        return
    raw = snapshot.metadata.info.get(CONTROL_MODE_INFO_KEY)
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError(f"metadata.info[{CONTROL_MODE_INFO_KEY!r}] must be an object")
    unknown = sorted(set(raw) - {"active_mode", "generation"})
    missing = sorted({"active_mode", "generation"} - set(raw))
    if missing or unknown:
        raise ValueError(
            f"snapshot control mode fields mismatch: missing={missing}, "
            f"unknown={unknown}"
        )
    source_mode = require_control_mode(
        raw["active_mode"],
        label="snapshot active_mode",
    )
    require_expected_generation(raw["generation"])
    if source_mode != runtime_state["active_mode"]:
        raise ValueError(
            "snapshot control mode mismatch: "
            f"snapshot={source_mode!r}, runtime={runtime_state['active_mode']!r}"
        )


def _validate_controller_profile_fingerprints(
    runtime: object,
    snapshot: SceneSnapshot,
    *,
    compatibility: object,
) -> None:
    """Validate resolved controller bundle contents across label mappings."""

    raw = snapshot.metadata.info.get(CONTROLLER_PROFILE_FINGERPRINTS_INFO_KEY)
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError(
            "snapshot controller profile fingerprints metadata must be an object"
        )
    for target_label, mapping in compatibility.robot_mappings.items():
        source_fingerprint = raw.get(mapping.source_label)
        if not isinstance(source_fingerprint, str) or not source_fingerprint:
            raise ValueError(
                "snapshot controller profile fingerprint is missing for "
                f"robot {mapping.source_label!r}"
            )
        target_fingerprint = getattr(
            runtime.robot_by_label(target_label),
            "controller_profile_fingerprint",
            None,
        )
        if not isinstance(target_fingerprint, str) or not target_fingerprint:
            raise ValueError(
                "target controller profile fingerprint is unavailable for "
                f"robot {target_label!r}"
            )
        if source_fingerprint != target_fingerprint:
            raise ValueError(
                "snapshot controller profile fingerprint mismatch: "
                f"source={mapping.source_label!r}, target={target_label!r}"
            )


@dataclass(frozen=True)
class _ControllerCache:
    """首次写入前复制的 controller 事务状态。"""

    efforts: object
    control_targets: object


def _controller_cache(execution: object) -> _ControllerCache:
    """复制 effort 与 ControlTargets 缓存，使回滚恢复后续控制行为。

    只恢复 articulation 状态仍可能让下一控制步使用新缓存，因此缓存属于事务状态。测试
    adapter 可能没有该属性，此时用 sentinel 区分“属性缺失”和“属性值为 None”。
    """

    controller = execution.joint_controller
    effort_cache = _missing
    if hasattr(controller, "last_commanded_efforts"):
        value = getattr(controller, "last_commanded_efforts")
        copy = getattr(value, "copy", None)
        effort_cache = copy() if callable(copy) else value
    target_cache = _missing
    snapshot_targets = getattr(controller, "snapshot_control_targets_cache", None)
    if callable(snapshot_targets):
        if not callable(getattr(controller, "restore_control_targets_cache", None)):
            raise RuntimeError(
                "controller target cache snapshot requires a matching restore method"
            )
        target_cache = snapshot_targets()
    return _ControllerCache(efforts=effort_cache, control_targets=target_cache)


def _restore_scene_robot(
    execution: object,
    source_robot: object,
    *,
    mapping: object,
    command_modes: tuple[str, ...],
    controller_cache: _ControllerCache,
) -> None:
    """把单机器人物理状态与控制器缓存作为一个补偿动作恢复。"""

    _restore_robot_snapshot_to_execution(
        execution,
        source_robot,
        mapping=mapping,
        command_modes=command_modes,
    )
    if controller_cache.efforts is not _missing:
        value = controller_cache.efforts
        copy = getattr(value, "copy", None)
        setattr(
            execution.joint_controller,
            "last_commanded_efforts",
            copy() if callable(copy) else value,
        )
    if controller_cache.control_targets is not _missing:
        execution.joint_controller.restore_control_targets_cache(
            controller_cache.control_targets
        )


def _validated_restore_command_modes(
    runtime: object,
    snapshot: SceneSnapshot,
    *,
    compatibility: object,
) -> dict[str, tuple[str, ...]]:
    """在首次 physics 写入前校验快照与目标 controller 的逻辑 target 模式。"""

    result: dict[str, tuple[str, ...]] = {}
    for target_label, mapping in compatibility.robot_mappings.items():
        execution = runtime.robot_by_label(target_label).execution
        controller = execution.joint_controller
        target_names = _command_joint_names(execution.articulation, controller)
        target_modes = _command_target_modes(
            controller,
            command_count=len(target_names),
        )
        apply_targets = getattr(controller, "apply_targets", None)
        if callable(apply_targets) and not callable(
            getattr(execution, "articulation_action_type", None)
        ):
            raise RuntimeError(
                f"robot {target_label!r} mode-aware restore requires "
                "articulation_action_type"
            )
        if any(mode != "position" for mode in target_modes) and not callable(
            apply_targets
        ):
            raise RuntimeError(
                f"robot {target_label!r} controller does not support non-position "
                "target restore"
            )

        source_label = mapping.source_label
        source_modes = _snapshot_command_target_modes(
            snapshot,
            source_label=source_label,
        )
        if source_modes is None:
            if any(mode != "position" for mode in target_modes):
                raise ValueError(
                    f"snapshot is missing {COMMAND_TARGET_MODES_INFO_KEY!r}; "
                    f"robot {target_label!r} is not all-position control"
                )
            result[target_label] = target_modes
            continue

        source_robot = snapshot.robots[source_label]
        source_names = source_robot.command_joint_names
        if target_names and not source_names:
            raise ValueError(
                f"snapshot command target modes for {source_label!r} do not describe "
                "any command joints"
            )
        command_mapping = mapping.command_joints
        if command_mapping is not None:
            for source_index, target_index in zip(
                command_mapping.source_indices,
                command_mapping.target_indices,
                strict=True,
            ):
                source_name = source_names[int(source_index)]
                target_name = target_names[int(target_index)]
                source_mode = source_modes[source_name]
                target_mode = target_modes[int(target_index)]
                if source_mode != target_mode:
                    raise ValueError(
                        "snapshot command target mode mismatch: "
                        f"source={source_label!r}.{source_name}({source_mode}), "
                        f"target={target_label!r}.{target_name}({target_mode})"
                    )
        result[target_label] = target_modes
    return result


def _restore_scene_objects(
    runtime: object,
    snapshot: SceneSnapshot,
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
    "CONTROL_MODE_INFO_KEY",
    "CONTROLLER_PROFILE_FINGERPRINTS_INFO_KEY",
    "NEWTON_SOLVER_STATE_INFO_KEY",
    "get_mirror_snapshot",
    "mirror_target_descriptor",
    "set_mirror_snapshot",
]
