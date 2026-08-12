"""snapshot adapter 共享的机器人、对象 descriptor、指纹与状态恢复 helper。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path

import numpy as np

from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.objects.runtime import runtime_object_prim_path
from linkerbot_sim.objects.state_views import SceneObjectStateView
from linkerbot_sim.snapshots.compatibility import (
    ObjectTargetDescriptor,
    RobotTargetDescriptor,
)
from linkerbot_sim.snapshots.schema import (
    ObjectSnapshot,
    RobotSnapshot,
    SceneSnapshot,
)
from linkerbot_sim.isaac.scene.pose import (
    apply_prim_local_pose_and_zero_velocity,
    read_prim_world_pose,
)
from linkerbot_sim.utils.tensors import tensor_like_to_numpy

COMMAND_TARGET_MODES_INFO_KEY = "linkerbot.snapshot.command_target_modes"
_COMMAND_TARGET_MODES = frozenset({"position", "velocity", "effort"})


def _robot_snapshot_from_execution(
    *,
    label: str,
    robot_id: int,
    execution: object,
    robot_profile: str | None,
    asset_fingerprint: str | None,
) -> RobotSnapshot:
    """从一个 RobotRuntime execution 读取 command-joint 快照。"""

    articulation = execution.articulation
    controller = execution.joint_controller
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    joint_names = _command_joint_names(articulation, controller)
    # articulation API 返回全 DOF；snapshot 只截取 controller 管理的 command joints，
    # 保持与 replicated-environment adapter 的坐标语义一致。
    positions = tensor_like_to_numpy(
        articulation.get_joint_positions(), dtype=float
    ).reshape(-1)
    velocities = tensor_like_to_numpy(
        articulation.get_joint_velocities(), dtype=float
    ).reshape(-1)
    command_modes = _command_target_modes(
        controller, command_count=command_indices.size
    )
    command_targets = _command_targets_from_execution(
        articulation,
        controller,
        command_indices=command_indices,
        command_modes=command_modes,
        positions=positions,
    )
    return RobotSnapshot(
        label=label,
        robot_id=robot_id,
        robot_profile=robot_profile,
        asset_fingerprint=asset_fingerprint,
        joint_names=joint_names,
        joint_positions=positions[command_indices],
        joint_velocities=velocities[command_indices],
        command_joint_names=joint_names,
        command_targets=command_targets,
    )


def _robot_target_from_execution(
    *,
    label: str,
    execution: object,
    robot_profile: str | None,
    asset_fingerprint: str | None,
) -> RobotTargetDescriptor:
    """从 execution 对象提取目标机器人关节名字和资产指纹。"""

    joint_names = _command_joint_names(
        execution.articulation,
        execution.joint_controller,
    )
    return RobotTargetDescriptor(
        label=label,
        robot_profile=robot_profile,
        asset_fingerprint=asset_fingerprint,
        joint_names=joint_names,
        command_joint_names=joint_names,
    )


def _restore_robot_snapshot_to_execution(
    execution: object,
    source_robot: RobotSnapshot,
    *,
    mapping: object,
    command_modes: tuple[str, ...] | None = None,
) -> None:
    """把一个 ``RobotSnapshot`` 写回 RobotRuntime execution 的 articulation。"""

    articulation = execution.articulation
    controller = execution.joint_controller
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    q = tensor_like_to_numpy(articulation.get_joint_positions(), dtype=float).reshape(
        -1
    )
    dq = tensor_like_to_numpy(articulation.get_joint_velocities(), dtype=float).reshape(
        -1
    )
    # mapping.joints.target_indices 是 command-joint 空间索引；先转换成 articulation DOF
    # index，再写入全量 q/dq，避免覆盖非 command DOF。
    target_command_indices = command_indices[mapping.joints.target_indices]
    q[target_command_indices] = source_robot.joint_positions[
        mapping.joints.source_indices
    ]
    dq[target_command_indices] = source_robot.joint_velocities[
        mapping.joints.source_indices
    ]
    articulation.set_joint_positions(q)
    articulation.set_joint_velocities(dq)
    resolved_modes = (
        _command_target_modes(controller, command_count=command_indices.size)
        if command_modes is None
        else _validated_command_modes(command_modes, command_indices.size)
    )
    command_mapping = mapping.command_joints
    if source_robot.command_targets is not None and command_mapping is not None:
        target_slots = np.asarray(command_mapping.target_indices, dtype=int)
        target_values = source_robot.command_targets[command_mapping.source_indices]
    else:
        target_slots = np.asarray(mapping.joints.target_indices, dtype=int)
        target_values = None

    apply_targets = getattr(controller, "apply_targets", None)
    action_type = getattr(execution, "articulation_action_type", None)
    if callable(apply_targets):
        if not callable(action_type):
            raise RuntimeError(
                "mode-aware command target restore requires articulation_action_type"
            )
        cached = _controller_control_targets_cache(controller, expected_size=q.size)
        if cached is None:
            target_positions = _articulation_position_targets(
                articulation,
                fallback=q,
            )
            target_velocities = _articulation_velocity_targets(
                articulation,
                fallback=dq,
            )
            target_efforts = _articulation_applied_efforts(
                articulation,
                fallback=np.zeros(q.size, dtype=float),
            )
        else:
            target_positions = cached.positions.copy()
            target_velocities = cached.velocities.copy()
            target_efforts = cached.efforts.copy()
        for offset, target_slot in enumerate(target_slots):
            dof_index = int(command_indices[int(target_slot)])
            mode = resolved_modes[int(target_slot)]
            value = (
                float(target_values[offset])
                if target_values is not None
                else _hold_target_for_mode(
                    mode,
                    dof_index=dof_index,
                    positions=q,
                    velocities=dq,
                )
            )
            if mode == "position":
                target_positions[dof_index] = value
            elif mode == "velocity":
                target_velocities[dof_index] = value
            else:
                target_efforts[dof_index] = value
        targets = ControlTargets(target_positions, target_velocities, target_efforts)
        rebuild_targets = getattr(controller, "targets_from_full_state", None)
        if callable(rebuild_targets):
            # JointController 需要根据恢复后的 master 实际状态刷新 mimic follower 目标。
            targets = rebuild_targets(
                targets.positions,
                targets.velocities,
                targets.efforts,
            )
        apply_targets(action_type, targets)
        return

    if any(mode != "position" for mode in resolved_modes):
        raise RuntimeError(
            "legacy articulation target restore only supports position control"
        )
    target_command_indices = command_indices[target_slots]
    if target_values is None:
        target_values = q[target_command_indices]
    _set_articulation_position_targets(
        execution,
        values=target_values,
        joint_indices=target_command_indices,
    )
    if hasattr(controller, "last_commanded_efforts"):
        controller.last_commanded_efforts = np.full(q.size, np.nan, dtype=float)


def _command_target_modes(
    controller: object,
    *,
    command_count: int,
) -> tuple[str, ...]:
    """读取 controller 的逻辑 target 模式；旧 controller 视为全 position。"""

    values = getattr(controller, "command_target_modes", None)
    if callable(values):
        values = values()
    if values is None:
        return ("position",) * int(command_count)
    return _validated_command_modes(values, command_count)


def _validated_command_modes(values: object, command_count: int) -> tuple[str, ...]:
    """校验 command-space target 模式枚举和长度。"""

    try:
        modes = tuple(str(mode) for mode in values)
    except TypeError as exc:
        raise RuntimeError("controller command_target_modes must be iterable") from exc
    if len(modes) != int(command_count):
        raise RuntimeError(
            "controller command_target_modes length must match command_indices: "
            f"modes={len(modes)}, commands={int(command_count)}"
        )
    invalid = [mode for mode in modes if mode not in _COMMAND_TARGET_MODES]
    if invalid:
        raise RuntimeError(f"unsupported controller command target modes: {invalid}")
    return modes


def _snapshot_command_target_modes(
    snapshot: SceneSnapshot,
    *,
    source_label: str,
) -> dict[str, str] | None:
    """解析一个 source robot 的 namespaced mode metadata；旧快照返回 ``None``。"""

    if COMMAND_TARGET_MODES_INFO_KEY not in snapshot.metadata.info:
        return None
    raw_modes = snapshot.metadata.info[COMMAND_TARGET_MODES_INFO_KEY]
    if not isinstance(raw_modes, Mapping):
        raise ValueError(
            f"metadata.info[{COMMAND_TARGET_MODES_INFO_KEY!r}] must be an object"
        )
    source_entry = raw_modes.get(source_label)
    if not isinstance(source_entry, Mapping):
        raise ValueError(
            f"snapshot command target modes are missing robot {source_label!r}"
        )
    source_names = snapshot.robots[source_label].command_joint_names
    actual_names = tuple(source_entry.keys())
    if any(not isinstance(name, str) for name in actual_names):
        raise ValueError(
            f"snapshot command target modes for {source_label!r} require string "
            "joint names"
        )
    missing = [name for name in source_names if name not in source_entry]
    extra = [name for name in actual_names if name not in source_names]
    if missing or extra:
        raise ValueError(
            f"snapshot command target modes for {source_label!r} do not match "
            f"command joints: missing={missing}, extra={extra}"
        )
    result: dict[str, str] = {}
    for name in source_names:
        mode = source_entry[name]
        if not isinstance(mode, str) or mode not in _COMMAND_TARGET_MODES:
            raise ValueError(
                f"snapshot command target mode for {source_label!r}.{name} "
                f"is invalid: {mode!r}"
            )
        result[name] = mode
    return result


def _command_targets_from_execution(
    articulation: object,
    controller: object,
    *,
    command_indices: np.ndarray,
    command_modes: tuple[str, ...],
    positions: np.ndarray,
) -> np.ndarray:
    """按每个 command joint 的 active mode 捕获逻辑 target 标量。"""

    cached = _controller_control_targets_cache(
        controller,
        expected_size=positions.size,
    )
    position_targets: np.ndarray | None = None
    velocity_targets: np.ndarray | None = None
    effort_targets: np.ndarray | None = None
    result = np.empty(command_indices.size, dtype=float)
    for slot, (dof_index, mode) in enumerate(
        zip(command_indices, command_modes, strict=True)
    ):
        index = int(dof_index)
        if cached is not None:
            values = {
                "position": cached.positions,
                "velocity": cached.velocities,
                "effort": cached.efforts,
            }[mode]
            result[slot] = values[index]
            continue
        if mode == "position":
            if position_targets is None:
                position_targets = _articulation_position_targets(
                    articulation,
                    fallback=positions,
                )
            result[slot] = position_targets[index]
        elif mode == "velocity":
            if velocity_targets is None:
                velocity_targets = _articulation_velocity_targets(
                    articulation,
                    expected_size=positions.size,
                )
            result[slot] = velocity_targets[index]
        else:
            if effort_targets is None:
                effort_targets = _controller_or_articulation_effort_targets(
                    articulation,
                    controller,
                    expected_size=positions.size,
                    required_indices=command_indices[
                        np.asarray(
                            [item == "effort" for item in command_modes], dtype=bool
                        )
                    ],
                )
            result[slot] = effort_targets[index]
    if not np.all(np.isfinite(result)):
        raise RuntimeError("captured command targets must contain finite values")
    return result


def _controller_control_targets_cache(
    controller: object,
    *,
    expected_size: int,
) -> ControlTargets | None:
    """读取并重建 controller 缓存，确保返回值不共享可变数组。"""

    snapshot = getattr(controller, "snapshot_control_targets_cache", None)
    values = (
        snapshot()
        if callable(snapshot)
        else getattr(controller, "last_control_targets", None)
    )
    if values is None:
        return None
    try:
        copied = ControlTargets(values.positions, values.velocities, values.efforts)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "controller returned an invalid ControlTargets cache"
        ) from exc
    if copied.positions.size != int(expected_size):
        raise RuntimeError(
            "controller ControlTargets cache length must match articulation DOFs: "
            f"targets={copied.positions.size}, dofs={int(expected_size)}"
        )
    return copied


def _controller_or_articulation_effort_targets(
    articulation: object,
    controller: object,
    *,
    expected_size: int,
    required_indices: np.ndarray,
) -> np.ndarray:
    """优先读取 controller 已下发 effort，缺失项回退 articulation applied effort。"""

    values = np.full(int(expected_size), np.nan, dtype=float)
    cached = getattr(controller, "last_commanded_efforts", None)
    if cached is not None:
        candidate = np.asarray(cached, dtype=float).reshape(-1)
        if candidate.size != int(expected_size):
            raise RuntimeError(
                "controller last_commanded_efforts length must match articulation DOFs"
            )
        finite = np.isfinite(candidate)
        values[finite] = candidate[finite]
    missing = required_indices[~np.isfinite(values[required_indices])]
    if missing.size:
        applied = _articulation_applied_efforts(
            articulation,
            fallback=None,
            expected_size=expected_size,
        )
        if applied is not None:
            values[missing] = applied[missing]
    if not np.all(np.isfinite(values[required_indices])):
        raise RuntimeError(
            "cannot capture effort command targets: controller cache and articulation "
            "applied efforts are unavailable"
        )
    return values


def _hold_target_for_mode(
    mode: str,
    *,
    dof_index: int,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> float:
    """为没有 command_targets 的快照生成确定的无旧缓存恢复目标。"""

    if mode == "position":
        return float(positions[dof_index])
    if mode == "velocity":
        return float(velocities[dof_index])
    return 0.0


def _articulation_position_targets(
    articulation: object,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    """读取完整 position drive target，旧 Core 无目标时退化为当前位置 hold。"""

    getter = getattr(articulation, "get_joint_position_targets", None)
    if callable(getter):
        values = getter()
    else:
        get_applied_action = getattr(articulation, "get_applied_action", None)
        action = get_applied_action() if callable(get_applied_action) else None
        values = None if action is None else getattr(action, "joint_positions", None)
    if values is None:
        return np.asarray(fallback, dtype=float).reshape(-1).copy()
    targets = tensor_like_to_numpy(values, dtype=float).reshape(-1)
    if targets.shape != fallback.shape or not np.all(np.isfinite(targets)):
        raise RuntimeError(
            "articulation position targets must be finite and match joint positions: "
            f"targets={targets.shape}, positions={fallback.shape}"
        )
    return targets.copy()


def _articulation_velocity_targets(
    articulation: object,
    *,
    fallback: np.ndarray | None = None,
    expected_size: int | None = None,
) -> np.ndarray:
    """读取完整 velocity target；仅恢复基准允许显式 fallback。"""

    getter = getattr(articulation, "get_joint_velocity_targets", None)
    if callable(getter):
        values = getter()
    else:
        get_applied_action = getattr(articulation, "get_applied_action", None)
        action = get_applied_action() if callable(get_applied_action) else None
        values = None if action is None else getattr(action, "joint_velocities", None)
    if values is None:
        if fallback is None:
            raise RuntimeError("articulation does not expose velocity targets")
        return np.asarray(fallback, dtype=float).reshape(-1).copy()
    targets = tensor_like_to_numpy(values, dtype=float).reshape(-1)
    required_size = (
        np.asarray(fallback).reshape(-1).size if fallback is not None else expected_size
    )
    if required_size is not None and targets.size != int(required_size):
        raise RuntimeError(
            "articulation velocity targets do not match joint state shape"
        )
    if not np.all(np.isfinite(targets)):
        raise RuntimeError("articulation velocity targets must contain finite values")
    return targets.copy()


def _articulation_applied_efforts(
    articulation: object,
    *,
    fallback: np.ndarray | None,
    expected_size: int | None = None,
) -> np.ndarray | None:
    """读取完整 applied effort；不可用时按调用方要求返回 fallback 或 ``None``。"""

    values = None
    getter = getattr(articulation, "get_applied_joint_efforts", None)
    if callable(getter):
        try:
            values = getter()
        except (AttributeError, RuntimeError, TypeError):
            values = None
    if values is None:
        get_applied_action = getattr(articulation, "get_applied_action", None)
        action = get_applied_action() if callable(get_applied_action) else None
        values = None if action is None else getattr(action, "joint_efforts", None)
    if values is None:
        return (
            None
            if fallback is None
            else np.asarray(fallback, dtype=float).reshape(-1).copy()
        )
    efforts = tensor_like_to_numpy(values, dtype=float).reshape(-1)
    required_size = (
        np.asarray(fallback).reshape(-1).size if fallback is not None else expected_size
    )
    if required_size is not None and efforts.size != int(required_size):
        raise RuntimeError(
            "articulation applied efforts do not match joint state shape"
        )
    if not np.all(np.isfinite(efforts)):
        if fallback is None:
            return None
        return np.asarray(fallback, dtype=float).reshape(-1).copy()
    return efforts.copy()


def _set_articulation_position_targets(
    execution: object,
    *,
    values: object,
    joint_indices: object,
) -> None:
    """同步快照的 position drive target，兼容 Experimental 与 legacy Core。"""

    articulation = execution.articulation
    setter = getattr(articulation, "set_joint_position_targets", None)
    if callable(setter):
        setter(values, joint_indices=joint_indices)
        return
    apply_action = getattr(articulation, "apply_action", None)
    action_type = getattr(execution, "articulation_action_type", None)
    if callable(apply_action) and callable(action_type):
        apply_action(
            action_type(
                joint_positions=np.asarray(values, dtype=float),
                joint_indices=np.asarray(joint_indices, dtype=int),
            )
        )
        return
    raise RuntimeError("articulation does not expose a position drive target writer")


def _command_joint_names(articulation: object, controller: object) -> tuple[str, ...]:
    """按 controller 配置返回 command joint 名字。"""

    names = getattr(controller, "command_joint_names", None)
    if names is not None:
        return tuple(str(name) for name in names)
    dof_names = tuple(str(name) for name in getattr(articulation, "dof_names", ()))
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    return tuple(dof_names[int(index)] for index in command_indices)


def _runtime_object_snapshots(
    *,
    stage: object | None,
    handles: Sequence[object],
    state_views: Mapping[str, SceneObjectStateView] | None = None,
) -> dict[str, ObjectSnapshot]:
    """读取 MirrorSceneResources object handles 对应的 scene-local 对象快照。

    Newton dynamic-chain 可同时保存精确 owner q/qd envelope 与 maximal body state：前者用于
    topology/ABI 相同的无损恢复，后者是跨 backend、部分映射和旧 target 的可移植退路。
    """

    if stage is None:
        return {}
    views = {} if state_views is None else state_views
    result: dict[str, ObjectSnapshot] = {}
    for handle in handles:
        name = _runtime_object_name(handle)
        prim_path = runtime_object_prim_path(handle)
        if not name or prim_path is None:
            continue
        body_names, body_paths = _runtime_object_body_paths(handle)
        state_view = views.get(name)
        if state_view is not None:
            state_view.require_velocity_support(object_name=name)
        pose = (
            state_view.root_world_pose()
            if state_view is not None and state_view.has_live_root
            else read_prim_world_pose(stage, prim_path)
        )
        if pose is None:
            continue
        position, orientation = pose
        kwargs: dict[str, object] = {}
        velocities = (
            state_view.root_velocities()
            if state_view is not None and state_view.has_live_root
            else _read_prim_rigid_body_velocities(stage, prim_path)
        )
        if velocities is not None:
            kwargs["linear_velocities"], kwargs["angular_velocities"] = velocities
        if state_view is not None:
            generalized = state_view.generalized_state()
            if generalized is not None:
                # generalized_state() 已保证 signature/names/q/qd 五字段同时生成；不要把
                # 它们拆成可选字段，否则序列化后将失去检测 q/qd 错列所需的 ABI 身份。
                kwargs.update(generalized)
        if body_names:
            # MirrorSceneResources 没有 env origin，root/child body 都按 scene-local pose 保存。
            body_positions = []
            body_orientations = []
            body_linear_velocities = []
            body_angular_velocities = []
            body_velocities_complete = True
            live_body_poses = None
            live_body_velocities = None
            if state_view is not None and state_view.body_view is not None:
                if state_view.body_names != body_names:
                    raise RuntimeError(
                        f"Scene object {name!r} body view names do not match snapshot paths"
                    )
                live_body_poses = state_view.body_world_poses()
                live_body_velocities = state_view.body_velocities()
            for body_index, body_path in enumerate(body_paths):
                body_pose = (
                    None
                    if live_body_poses is None
                    else (
                        live_body_poses[0][body_index],
                        live_body_poses[1][body_index],
                    )
                )
                if live_body_poses is None:
                    body_pose = read_prim_world_pose(stage, body_path)
                if body_pose is None:
                    break
                body_position, body_orientation = body_pose
                body_positions.append(body_position)
                body_orientations.append(body_orientation)
                body_velocities = (
                    None
                    if live_body_velocities is None
                    else (
                        live_body_velocities[0][body_index],
                        live_body_velocities[1][body_index],
                    )
                )
                if state_view is None or state_view.body_view is None:
                    body_velocities = _read_prim_rigid_body_velocities(stage, body_path)
                if body_velocities is None:
                    body_velocities_complete = False
                else:
                    body_linear, body_angular = body_velocities
                    body_linear_velocities.append(body_linear)
                    body_angular_velocities.append(body_angular)
            if len(body_positions) == len(body_names):
                kwargs.update(
                    {
                        "body_names": body_names,
                        "body_positions_local": np.vstack(body_positions),
                        "body_orientations_wxyz": np.vstack(body_orientations),
                    }
                )
                if body_velocities_complete:
                    kwargs.update(
                        {
                            "body_linear_velocities": np.vstack(body_linear_velocities),
                            "body_angular_velocities": np.vstack(
                                body_angular_velocities
                            ),
                        }
                    )
        result[name] = ObjectSnapshot(
            name=name,
            object_profile=_runtime_object_profile(handle),
            positions_local=position,
            orientations_wxyz=orientation,
            **kwargs,
        )
    return result


def _runtime_object_targets(
    handles: Sequence[object],
) -> dict[str, ObjectTargetDescriptor]:
    """根据 MirrorSceneResources object handles 构建 object target descriptors。"""

    result = {}
    for handle in handles:
        name = _runtime_object_name(handle)
        if not name:
            continue
        body_names, _ = _runtime_object_body_paths(handle)
        result[name] = ObjectTargetDescriptor(
            name=name,
            object_profile=_runtime_object_profile(handle),
            body_names=body_names,
        )
    return result


def _restore_runtime_objects(
    runtime: object,
    snapshot: SceneSnapshot,
    *,
    compatibility: object,
) -> tuple[str, ...]:
    """把 snapshot objects 优先写回 live physics view，静态对象使用 USD。

    对 Newton dynamic-chain，只有完整 body mapping、相同 generalized ABI 且无需 replicated-origin
    换算时才写 owner q/qd；其它情况使用 maximal body state。选择 fallback 是坐标/兼容性
    决策，不代表 body pose 比 solver owner state 更权威。
    """

    stage = getattr(getattr(runtime, "session", None), "stage", None)
    if stage is None or not snapshot.objects:
        return ()
    handles_by_name = {
        _runtime_object_name(handle): handle
        for handle in getattr(runtime, "object_handles", ())
        if _runtime_object_name(handle)
    }
    restored: list[str] = []
    for target_name, mapping in compatibility.object_mappings.items():
        obj = snapshot.objects[mapping.source_name]
        handle = handles_by_name.get(target_name)
        if handle is None:
            continue
        prim_path = runtime_object_prim_path(handle)
        state_view = getattr(runtime, "object_state_views", {}).get(target_name)
        if state_view is not None:
            state_view.require_velocity_support(object_name=target_name)
        dynamic_chain_live = bool(
            obj.body_names
            and state_view is not None
            and state_view.root_view is None
            and state_view.body_view is not None
        )
        generalized_owner_restore = bool(
            dynamic_chain_live
            and obj.generalized_q is not None
            # replicated-env 快照的 FREE-root q 使用 world frame，并携带 source env origin；
            # Mirror 没有对应的 target env origin 可计算 delta，因此跨 runtime 时改走
            # scene-local maximal-body fallback，绝不能原样写入 source world 的 q。
            and obj.generalized_world_origin is None
            and state_view is not None
            and state_view.has_generalized_state
            and _object_mapping_covers_complete_body_state(
                obj,
                state_view=state_view,
                mapping=mapping,
            )
        )
        if (
            not dynamic_chain_live
            and prim_path is not None
            and _apply_prim_local_pose_and_velocity(
                stage,
                prim_path,
                obj.positions_local,
                obj.orientations_wxyz,
                obj.linear_velocities,
                obj.angular_velocities,
                state_view=state_view,
            )
        ):
            restored.append(target_name)
        if obj.body_names and mapping.bodies is not None:
            body_names, body_paths = _runtime_object_body_paths(handle)
            body_path_by_name = dict(zip(body_names, body_paths, strict=True))
            assert obj.body_positions_local is not None
            assert obj.body_orientations_wxyz is not None
            if generalized_owner_restore:
                assert state_view is not None
                assert obj.generalized_q is not None
                assert obj.generalized_qd is not None
                state_view.set_generalized_state(
                    signature=obj.generalized_signature,
                    q_names=obj.generalized_q_names,
                    qd_names=obj.generalized_qd_names,
                    q=obj.generalized_q,
                    qd=obj.generalized_qd,
                )
                # q/qd 是唯一 owner 写入；body transforms 由 Newton view 内部 FK 派生，
                # 此后不能再执行下面的逐 body setter，否则会形成两套互相冲突的权威状态。
                restored.append(target_name)
                continue
            if dynamic_chain_live:
                if (
                    obj.body_linear_velocities is None
                    or obj.body_angular_velocities is None
                ):
                    raise ValueError(
                        "dynamic-chain object snapshot is missing required body velocities"
                    )
                assert state_view is not None
                source_indices = np.asarray(
                    mapping.bodies.source_indices, dtype=int
                ).reshape(-1)
                target_indices = np.asarray(
                    [
                        state_view.body_names.index(name)
                        for name in mapping.bodies.names
                    ],
                    dtype=int,
                )
                state_view.set_body_states(
                    body_indices=target_indices,
                    positions=obj.body_positions_local[source_indices],
                    orientations_wxyz=obj.body_orientations_wxyz[source_indices],
                    linear_velocities=obj.body_linear_velocities[source_indices],
                    angular_velocities=obj.body_angular_velocities[source_indices],
                )
                restored.append(target_name)
                continue
            for source_index, body_name in zip(
                mapping.bodies.source_indices,
                mapping.bodies.names,
                strict=True,
            ):
                body_path = body_path_by_name.get(body_name)
                if body_path is None:
                    continue
                _apply_prim_local_pose_and_velocity(
                    stage,
                    body_path,
                    obj.body_positions_local[int(source_index)],
                    obj.body_orientations_wxyz[int(source_index)],
                    (
                        None
                        if obj.body_linear_velocities is None
                        else obj.body_linear_velocities[int(source_index)]
                    ),
                    (
                        None
                        if obj.body_angular_velocities is None
                        else obj.body_angular_velocities[int(source_index)]
                    ),
                    state_view=state_view,
                    body_index=(
                        None
                        if state_view is None or state_view.body_view is None
                        else state_view.body_names.index(body_name)
                    ),
                )
    return tuple(restored)


def _object_mapping_covers_complete_body_state(
    obj: ObjectSnapshot,
    *,
    state_view: SceneObjectStateView,
    mapping: object,
) -> bool:
    """仅在 body 集合完全一致时允许 generalized 原子路径。

    strict=False 的名称交集适合 maximal body copy，却无法表达“只写半条 articulation”的
    q/qd；这种映射必须回退，不能用 generalized fast path 顺带覆盖未请求的 body。
    """

    bodies = getattr(mapping, "bodies", None)
    if bodies is None:
        return False
    names = tuple(str(name) for name in getattr(bodies, "names", ()))
    return len(names) == len(obj.body_names) == len(state_view.body_names) and set(
        names
    ) == set(obj.body_names) == set(state_view.body_names)


def _preflight_runtime_object_restore(
    runtime: object,
    snapshot: SceneSnapshot,
    *,
    compatibility: object,
) -> None:
    """在任何 scene setter 运行前预检 Newton object owner state。

    事务层会分别预检待写快照和 rollback 快照；只有二者都能按目标 topology/ABI 恢复，才
    允许第一次 mutation。其它后端、replicated-origin 跨 runtime 以及部分 body mapping 明确
    走 maximal-body 路径，不在这里伪装成 generalized 验证成功。
    """

    views = getattr(runtime, "object_state_views", {})
    for target_name, mapping in compatibility.object_mappings.items():
        obj = snapshot.objects[mapping.source_name]
        if obj.generalized_q is None:
            continue
        if obj.generalized_world_origin is not None:
            continue
        state_view = views.get(target_name) if isinstance(views, Mapping) else None
        if (
            state_view is None
            or not state_view.has_generalized_state
            or not _object_mapping_covers_complete_body_state(
                obj,
                state_view=state_view,
                mapping=mapping,
            )
        ):
            # 其它后端 target 与部分 body 映射继续使用可移植 maximal-body restore。
            continue
        assert obj.generalized_qd is not None
        state_view.preflight_generalized_state(
            signature=obj.generalized_signature,
            q_names=obj.generalized_q_names,
            qd_names=obj.generalized_qd_names,
            q=obj.generalized_q,
            qd=obj.generalized_qd,
        )


def _read_prim_rigid_body_velocities(
    stage: object,
    prim_path: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取 prim 自身的 USD 刚体速度；非刚体 root 返回 ``None``。"""

    from pxr import Sdf, UsdPhysics

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return None
    api = UsdPhysics.RigidBodyAPI(prim)
    values = []
    for getter_name in ("GetVelocityAttr", "GetAngularVelocityAttr"):
        attr = getattr(api, getter_name)()
        value = attr.Get() if attr is not None and attr.IsValid() else None
        values.append(
            np.zeros(3, dtype=float)
            if value is None
            else np.asarray(value, dtype=float).reshape(3)
        )
    # USD Physics authoring attr 使用 deg/s；canonical snapshot 和 live PhysX view 使用 rad/s。
    return values[0], np.deg2rad(values[1])


def _apply_prim_local_pose_and_velocity(
    stage: object,
    prim_path: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    linear_velocity: np.ndarray | None,
    angular_velocity: np.ndarray | None,
    *,
    state_view: SceneObjectStateView | None = None,
    body_index: int | None = None,
) -> bool:
    """恢复 pose，并为 live rigid view 写回完整的速度对。"""

    if (linear_velocity is None) != (angular_velocity is None):
        raise ValueError("object velocity requires both linear and angular components")
    live_state_target = state_view is not None and (
        (body_index is None and state_view.has_live_root)
        or (body_index is not None and state_view.body_view is not None)
    )
    if live_state_target and linear_velocity is None:
        raise ValueError("object snapshot is missing required velocity state")

    if live_state_target:
        assert linear_velocity is not None
        assert angular_velocity is not None
        if body_index is None:
            state_view.set_root_state(
                position=position,
                orientation_wxyz=orientation_wxyz,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
        else:
            state_view.set_body_world_pose(
                body_index=body_index,
                position=position,
                orientation_wxyz=orientation_wxyz,
            )
            state_view.set_body_velocities(
                body_index=body_index,
                linear=linear_velocity,
                angular=angular_velocity,
            )
        return True
    applied = apply_prim_local_pose_and_zero_velocity(
        stage,
        prim_path,
        position,
        orientation_wxyz,
    )
    if not applied:
        return applied
    if linear_velocity is None and angular_velocity is None:
        return True
    from pxr import Gf, Sdf, UsdPhysics

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"object prim {prim_path!r} does not have RigidBodyAPI")
    api = UsdPhysics.RigidBodyAPI(prim)
    for getter_name, velocity, angular in (
        ("GetVelocityAttr", linear_velocity, False),
        ("GetAngularVelocityAttr", angular_velocity, True),
    ):
        if velocity is None:
            continue
        xyz = np.asarray(velocity, dtype=float).reshape(3)
        if angular:
            xyz = np.rad2deg(xyz)
        getattr(api, getter_name)().Set(
            Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        )
    return True


def _reset_execution_observers(execution: object | None) -> None:
    """恢复状态后重置 execution 上可能缓存旧采样的 observer。"""

    if execution is None:
        return
    for name in ("state_observer", "camera_observer"):
        observer = getattr(execution, name, None)
        reset = getattr(observer, "reset", None)
        if callable(reset):
            reset()


def _runtime_object_name(handle: object) -> str:
    """从 runtime object handle 中提取对外使用的稳定对象名。"""

    runtime_handle = getattr(handle, "runtime_handle", None)
    if runtime_handle is not None:
        return str(runtime_handle)
    name = getattr(handle, "name", None)
    return "" if name is None else str(name)


def _runtime_object_profile(handle: object) -> str | None:
    """读取 object profile 名称；缺失时返回 ``None``。"""

    config = getattr(handle, "config", None)
    if hasattr(config, "object_profile"):
        return str(getattr(config, "object_profile"))
    return None


def _runtime_object_body_paths(
    handle: object,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """读取 dynamic/multi-body object 的 child body 名字和 prim paths。"""

    model = getattr(handle, "model", None)
    if isinstance(model, Mapping):
        bodies = list(model.get("bodies", ()) or ())
    else:
        bodies = list(getattr(model, "bodies", ()) or ())
    names = []
    paths = []
    for body in bodies:
        path_getter = getattr(body, "GetPath", None)
        body_path = str(path_getter() if callable(path_getter) else body)
        name_getter = getattr(body, "GetName", None)
        body_name = str(
            name_getter() if callable(name_getter) else body_path.rsplit("/", 1)[-1]
        )
        names.append(body_name)
        paths.append(body_path)
    return tuple(names), tuple(paths)


def _imported_asset_fingerprint(imported: object | None) -> str | None:
    """计算已导入仿真资产的稳定指纹。"""

    if imported is None:
        return None
    asset_path = getattr(imported, "asset_path", None)
    if asset_path is None:
        return None
    return _asset_fingerprint_from_path(asset_path)


def _asset_fingerprint_from_path(asset_path: str | Path) -> str:
    """基于规范路径和文件内容返回跨模式共用的 SHA-256 指纹。"""

    path = Path(asset_path)
    digest = hashlib.sha256(str(path.resolve()).encode())
    if path.is_file():
        digest.update(path.read_bytes())
    return digest.hexdigest()
