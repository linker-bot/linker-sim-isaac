"""双臂 motion 的 planner 调用、轨迹执行和 hand/raw 辅助。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from linkerbot_sim.app.motion.dual_arm_cspace import (
    dual_cspace_goal_to_command,
    dual_cspace_trajectory_from_motion_result,
    dual_cspace_vector_from_side_commands,
)
from linkerbot_sim.app.motion.runtime import (
    cumotion_boundary,
    MotionPlanningFailed,
    phase_for_move,
    trajectory_sample_times,
)
from linkerbot_sim.app.motion.specs import (
    CommandOverlaySpec,
    DualHandMoveSpec,
    HandMoveSpec,
    RawJointSequenceMoveSpec,
    RawJointSequenceSideSpec,
)
from linkerbot_sim.execution.dual_runtime import DualRobotRuntime, RobotSideRuntime
from linkerbot_sim.execution.dual_steps import (
    DualCommandExecutionInterrupted,
    DualCommandPositionTrajectoryStep,
    DualRawCommandTargetSequenceStep,
)
from linkerbot_sim.planning.dual_arm_cspace_partition import (
    DualArmJointPartitions,
    split_dual_arm_trajectory_to_commands,
)
from linkerbot_sim.planning.requests import MotionRequest, SpecifiedPathRequest
from linkerbot_sim.robots.classification import component_for_name
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class DualMoveExecutionResult:
    """单个双臂 move 执行后的滚动状态，用于串联后续 move。"""

    step: int
    cspace_q: np.ndarray
    left_command: np.ndarray
    right_command: np.ndarray


def plan_dual_motion_trajectory(
    *,
    context,
    request: MotionRequest | SpecifiedPathRequest,
    tcp_frame_name: str,
    config,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
    move_index: int,
    side: str | None,
    execution_mode: str,
) -> JointTrajectory:
    """调用双臂 cuMotion planner，并把结果转成融合 C-space 轨迹。"""

    planner = context.make_motion_planner(
        tcp_frame_name=tcp_frame_name,
        config=config,
    )
    print(
        "DUAL_ARM_CUMOTION_PLAN_START "
        f"move={move_index} side={side} tcp={tcp_frame_name} phase={phase} "
        f"type={type(request).__name__} execution={execution_mode} "
        f"pipeline={config.planning_pipeline}",
        flush=True,
    )
    result = cumotion_boundary("motion planner", planner.plan, request)
    if not result.success:
        message = (
            "cuMotion dual-arm path planning failed: "
            f"phase={phase} status={result.status} "
            f"message={result.diagnostics.message}"
        )
        raise MotionPlanningFailed(
            message,
            phase=phase,
            status=str(result.status),
            solver_message=str(result.diagnostics.message),
            move_index=move_index,
            side=side,
            tcp_frame_name=tcp_frame_name,
            component="motion planner",
        )
    trajectory = dual_cspace_trajectory_from_motion_result(
        result,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )
    metrics = result.diagnostics.metrics
    print(
        "DUAL_ARM_CUMOTION_PLAN_OK "
        f"move={move_index} side={side} tcp={tcp_frame_name} phase={phase} "
        f"pipeline={config.planning_pipeline} status={result.status} "
        f"samples={len(trajectory)} "
        f"path_waypoints={int(metrics.get('num_waypoints', 0.0))} "
        f"path_length={metrics.get('path_length', 0.0):.6g}",
        flush=True,
    )
    return trajectory


def execute_dual_cspace_trajectory(
    *,
    runtime: DualRobotRuntime,
    partitions: DualArmJointPartitions,
    trajectory: JointTrajectory,
    joint_names: Sequence[str],
    left_start_command: np.ndarray,
    right_start_command: np.ndarray,
    step: int,
    phase: str,
    overlays: Sequence[CommandOverlaySpec] = (),
    sample_dt: float = 1.0 / 60.0,
    default_duration_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> DualMoveExecutionResult:
    """把融合 C-space 轨迹拆成左右 command-space 轨迹并执行。"""

    if len(trajectory) == 0:
        raise ValueError(f"trajectory for {phase!r} cannot be empty")
    goal_q = np.asarray(trajectory.positions[-1], dtype=float).reshape(-1)
    left_target = dual_cspace_goal_to_command(
        side_runtime=runtime.left,
        base_command=left_start_command,
        joint_names=joint_names,
        goal_q=goal_q,
    )
    right_target = dual_cspace_goal_to_command(
        side_runtime=runtime.right,
        base_command=right_start_command,
        joint_names=joint_names,
        goal_q=goal_q,
    )
    left_trajectory, right_trajectory = split_dual_arm_trajectory_to_commands(
        dual_arm_trajectory=trajectory,
        partitions=partitions,
        left_command_joint_names=runtime.left.joint_controller.command_joint_names,
        right_command_joint_names=runtime.right.joint_controller.command_joint_names,
        left_start_command=left_start_command,
        right_start_command=right_start_command,
        left_target_command=left_target,
        right_target_command=right_target,
        phase=phase,
    )
    left_trajectory, right_trajectory = _apply_sync_overlays_to_trajectories(
        tuple(overlays),
        left_trajectory=left_trajectory,
        right_trajectory=right_trajectory,
        runtime=runtime,
        left_start_command=left_start_command,
        right_start_command=right_start_command,
        sample_dt=sample_dt,
        default_duration_s=default_duration_s,
        phase=phase,
    )
    step = DualCommandPositionTrajectoryStep(
        left_trajectory=left_trajectory,
        right_trajectory=right_trajectory,
        phase=phase,
        should_stop=should_stop,
    ).run(runtime, step)
    return DualMoveExecutionResult(
        step=step,
        cspace_q=goal_q,
        left_command=np.asarray(left_trajectory.positions[-1], dtype=float),
        right_command=np.asarray(right_trajectory.positions[-1], dtype=float),
    )


def execute_hand_move(
    move: HandMoveSpec,
    *,
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    should_stop: Callable[[], bool] | None = None,
) -> DualMoveExecutionResult:
    """把单手 move 包装成 DualHandMoveSpec，复用双手执行路径。"""

    dual = DualHandMoveSpec(
        left=move if normalize_side(move.side) == "left" else None,
        right=move if normalize_side(move.side) == "right" else None,
        duration_s=move.duration_s,
        phase=move.phase,
    )
    return execute_dual_hand_move(
        dual,
        runtime=runtime,
        current_q=current_q,
        left_command=left_command,
        right_command=right_command,
        step=step,
        sample_dt=sample_dt,
        should_stop=should_stop,
    )


def execute_dual_hand_move(
    move: DualHandMoveSpec,
    *,
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    should_stop: Callable[[], bool] | None = None,
) -> DualMoveExecutionResult:
    """执行左右手 command-space 线性插值动作，不改变机械臂 C-space。"""

    duration_s = _dual_hand_duration(move)
    phase = phase_for_move(move)
    left_target = (
        _hand_target_command(
            runtime.left,
            start_command=left_command,
            hand=move.left,
        )
        if move.left is not None
        else left_command
    )
    right_target = (
        _hand_target_command(
            runtime.right,
            start_command=right_command,
            hand=move.right,
        )
        if move.right is not None
        else right_command
    )
    left_trajectory = (
        _command_linear_trajectory(
            command_joint_names=runtime.left.joint_controller.command_joint_names,
            start_command=left_command,
            target_command=left_target,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
        )
        if move.left is not None
        else None
    )
    right_trajectory = (
        _command_linear_trajectory(
            command_joint_names=runtime.right.joint_controller.command_joint_names,
            start_command=right_command,
            target_command=right_target,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
        )
        if move.right is not None
        else None
    )
    step = DualCommandPositionTrajectoryStep(
        left_trajectory=left_trajectory,
        right_trajectory=right_trajectory,
        phase=phase,
        should_stop=should_stop,
    ).run(runtime, step)
    return DualMoveExecutionResult(
        step=step,
        cspace_q=np.asarray(current_q, dtype=float).reshape(-1),
        left_command=np.asarray(left_target, dtype=float).reshape(-1),
        right_command=np.asarray(right_target, dtype=float).reshape(-1),
    )


def execute_raw_joint_sequence_move(
    move: RawJointSequenceMoveSpec,
    *,
    runtime: DualRobotRuntime,
    joint_names: Sequence[str],
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    should_stop: Callable[[], bool] | None = None,
) -> DualMoveExecutionResult:
    """按 physics step 执行调用方提供的原始左右 command target 序列。"""

    phase = move.phase or "raw_joint_sequence"
    left_positions = _raw_sequence_side_matrix(
        move.left,
        side_runtime=runtime.left,
        current_command=left_command,
        label="left",
    )
    right_positions = _raw_sequence_side_matrix(
        move.right,
        side_runtime=runtime.right,
        current_command=right_command,
        label="right",
    )
    _validate_raw_side_sample_counts(left_positions, right_positions)
    step = DualRawCommandTargetSequenceStep(
        left_positions=left_positions,
        right_positions=right_positions,
        step_interval=int(move.step_interval),
        phase=phase,
        should_stop=should_stop,
    ).run(runtime, step)
    next_left = (
        np.asarray(left_positions[-1], dtype=float).reshape(-1)
        if left_positions is not None
        else np.asarray(left_command, dtype=float).reshape(-1)
    )
    next_right = (
        np.asarray(right_positions[-1], dtype=float).reshape(-1)
        if right_positions is not None
        else np.asarray(right_command, dtype=float).reshape(-1)
    )
    next_q = dual_cspace_vector_from_side_commands(
        joint_names=joint_names,
        left_command_joint_names=runtime.left.joint_controller.command_joint_names,
        right_command_joint_names=runtime.right.joint_controller.command_joint_names,
        left_command=next_left,
        right_command=next_right,
    )
    return DualMoveExecutionResult(
        step=step,
        cspace_q=next_q,
        left_command=next_left,
        right_command=next_right,
    )


def execute_overlays(
    overlays: Sequence[CommandOverlaySpec],
    *,
    timing: str,
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    default_duration_s: float | None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """执行指定 timing 的手部 overlays，并返回更新后的 step 与左右 command。"""

    current_left = np.asarray(left_command, dtype=float).reshape(-1)
    current_right = np.asarray(right_command, dtype=float).reshape(-1)
    for overlay in overlays:
        if overlay.timing != timing:
            continue
        move = _dual_hand_move_from_overlay(
            overlay,
            duration_s=default_duration_s,
            phase=f"overlay_{timing}",
        )
        result = execute_dual_hand_move(
            move,
            runtime=runtime,
            current_q=current_q,
            left_command=current_left,
            right_command=current_right,
            step=step,
            sample_dt=sample_dt,
            should_stop=should_stop,
        )
        step = result.step
        current_left = result.left_command
        current_right = result.right_command
    return step, current_left, current_right


def finish_arm_move_with_after_overlays(
    result: DualMoveExecutionResult,
    *,
    overlays: Sequence[CommandOverlaySpec],
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    sample_dt: float,
    default_duration_s: float | None,
    should_stop: Callable[[], bool] | None = None,
) -> DualMoveExecutionResult:
    """主臂动作完成后执行 after overlays，并把结果合并为新的滚动状态。"""

    step, left_command, right_command = execute_overlays(
        overlays,
        timing="after",
        runtime=runtime,
        current_q=current_q,
        left_command=result.left_command,
        right_command=result.right_command,
        step=result.step,
        sample_dt=sample_dt,
        default_duration_s=default_duration_s,
        should_stop=should_stop,
    )
    return DualMoveExecutionResult(
        step=step,
        cspace_q=np.asarray(current_q, dtype=float).reshape(-1),
        left_command=left_command,
        right_command=right_command,
    )


def raise_if_requested_stop(should_stop: Callable[[], bool] | None) -> None:
    """在可中断边界检查外部停止请求，并抛出统一中断异常。"""

    if should_stop is not None and should_stop():
        raise DualCommandExecutionInterrupted("dual command execution interrupted")


def normalize_side(side: str) -> str:
    """把 side 规范化为 left/right。"""

    normalized = str(side).lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return normalized


def _apply_sync_overlays_to_trajectories(
    overlays: Sequence[CommandOverlaySpec],
    *,
    left_trajectory: JointTrajectory,
    right_trajectory: JointTrajectory,
    runtime: DualRobotRuntime,
    left_start_command: np.ndarray,
    right_start_command: np.ndarray,
    sample_dt: float,
    default_duration_s: float | None,
    phase: str,
) -> tuple[JointTrajectory, JointTrajectory]:
    """把 sync overlays 的手部轨迹合并进主臂左右 command 轨迹。"""

    left = left_trajectory
    right = right_trajectory
    left_target = np.asarray(left.positions[-1], dtype=float).reshape(-1)
    right_target = np.asarray(right.positions[-1], dtype=float).reshape(-1)
    for overlay in overlays:
        if overlay.timing != "sync":
            continue
        if overlay.left_hand is not None:
            left_target = _hand_target_command(
                runtime.left,
                start_command=left_target,
                hand=_hand_with_default_duration(
                    overlay.left_hand,
                    default_duration_s,
                ),
            )
            left = _overlay_hand_trajectory(
                base=left,
                side_runtime=runtime.left,
                start_command=left_start_command,
                target_command=left_target,
                hand=overlay.left_hand,
                sample_dt=sample_dt,
                default_duration_s=default_duration_s,
                phase=phase,
            )
        if overlay.right_hand is not None:
            right_target = _hand_target_command(
                runtime.right,
                start_command=right_target,
                hand=_hand_with_default_duration(
                    overlay.right_hand,
                    default_duration_s,
                ),
            )
            right = _overlay_hand_trajectory(
                base=right,
                side_runtime=runtime.right,
                start_command=right_start_command,
                target_command=right_target,
                hand=overlay.right_hand,
                sample_dt=sample_dt,
                default_duration_s=default_duration_s,
                phase=phase,
            )
    return left, right


def _overlay_hand_trajectory(
    *,
    base: JointTrajectory,
    side_runtime: RobotSideRuntime,
    start_command: np.ndarray,
    target_command: np.ndarray,
    hand: HandMoveSpec,
    sample_dt: float,
    default_duration_s: float | None,
    phase: str,
) -> JointTrajectory:
    """为一侧手部 overlay 生成 command 轨迹，并覆盖 base 中的手部列。"""

    duration_s = (
        float(hand.duration_s)
        if hand.duration_s is not None
        else float(default_duration_s or base.times[-1])
    )
    hand_trajectory = _command_linear_trajectory(
        command_joint_names=side_runtime.joint_controller.command_joint_names,
        start_command=start_command,
        target_command=target_command,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )
    return _merge_hand_columns(
        base=base,
        overlay=hand_trajectory,
        command_joint_names=side_runtime.joint_controller.command_joint_names,
        hand_joint_names=_hand_command_joint_names(
            side_runtime.joint_controller.command_joint_names
        ),
    )


def _merge_hand_columns(
    *,
    base: JointTrajectory,
    overlay: JointTrajectory,
    command_joint_names: Sequence[str],
    hand_joint_names: Sequence[str],
) -> JointTrajectory:
    """按 base 时间轴重采样 overlay，并只替换手部 command 列。"""

    names = tuple(str(name) for name in command_joint_names)
    hand_indices = [
        index for index, name in enumerate(names) if name in set(hand_joint_names)
    ]
    if not hand_indices:
        return base
    positions = base.positions.copy()
    velocities = base.velocities.copy()
    accelerations = base.accelerations.copy()
    jerks = base.jerks.copy()
    efforts = base.efforts.copy()
    for index in hand_indices:
        for matrix, overlay_matrix in (
            (positions, overlay.positions),
            (velocities, overlay.velocities),
            (accelerations, overlay.accelerations),
            (jerks, overlay.jerks),
            (efforts, overlay.efforts),
        ):
            matrix[:, index] = np.asarray(
                [
                    np.interp(t, overlay.times, overlay_matrix[:, index])
                    for t in base.times
                ],
                dtype=float,
            )
    return JointTrajectory.from_samples(
        times=base.times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        efforts=efforts,
        phases=base.phases,
        joint_names=base.joint_names,
    )


def _hand_target_command(
    side_runtime: RobotSideRuntime,
    *,
    start_command: np.ndarray,
    hand: HandMoveSpec,
) -> np.ndarray:
    """把手部目标写入一侧 command 向量，支持按关节名或按手部顺序输入。"""

    if normalize_side(hand.side) != side_runtime.side:
        raise ValueError(
            f"hand target side {hand.side!r} does not match runtime side "
            f"{side_runtime.side!r}"
        )
    target = np.asarray(start_command, dtype=float).reshape(-1).copy()
    command_names = tuple(
        str(name) for name in side_runtime.joint_controller.command_joint_names
    )
    if target.size != len(command_names):
        raise ValueError(
            f"{side_runtime.side} command shape mismatch: "
            f"{len(command_names)} names, {target.size} values"
        )
    if isinstance(hand.joint_positions, Mapping):
        index_by_name = {name: index for index, name in enumerate(command_names)}
        for name, value in hand.joint_positions.items():
            key = str(name)
            if key not in index_by_name:
                raise ValueError(
                    f"{side_runtime.side} hand joint {key!r} is not in command-space"
                )
            target[index_by_name[key]] = float(value)
        return target
    hand_names = _hand_command_joint_names(command_names)
    values = np.asarray(hand.joint_positions, dtype=float).reshape(-1)
    if values.size > len(hand_names):
        raise ValueError(
            f"{side_runtime.side} hand target has {values.size} values, "
            f"but only {len(hand_names)} hand command joints exist"
        )
    index_by_name = {name: index for index, name in enumerate(command_names)}
    for name, value in zip(hand_names, values):
        target[index_by_name[name]] = float(value)
    return target


def _raw_sequence_side_matrix(
    side: RawJointSequenceSideSpec | None,
    *,
    side_runtime: RobotSideRuntime,
    current_command: np.ndarray,
    label: str,
) -> np.ndarray | None:
    """把单侧 raw sequence 规范化为 shape=(samples, command_dof) 的矩阵。"""

    if side is None:
        return None
    command_names = tuple(
        str(name) for name in side_runtime.joint_controller.command_joint_names
    )
    base = np.asarray(current_command, dtype=float).reshape(-1)
    if base.size != len(command_names):
        raise ValueError(
            f"{label} command shape mismatch: {len(command_names)} names, {base.size} values"
        )
    positions = side.joint_positions
    if isinstance(positions, Mapping):
        matrix = _raw_mapping_sequence_matrix(
            positions,
            command_joint_names=command_names,
            base_command=base,
            label=label,
        )
    else:
        matrix = np.asarray(positions, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"{label} raw joint sequence must have shape (N, dof)")
        if matrix.shape[1] != len(command_names):
            raise ValueError(
                f"{label} raw joint sequence expected {len(command_names)} columns, "
                f"got {matrix.shape[1]}"
            )
    if matrix.shape[0] == 0:
        raise ValueError(f"{label} raw joint sequence cannot be empty")
    return matrix


def _raw_mapping_sequence_matrix(
    positions: Mapping[str, Sequence[float]],
    *,
    command_joint_names: Sequence[str],
    base_command: np.ndarray,
    label: str,
) -> np.ndarray:
    """把按关节名给出的 raw sequence 展开成完整 command-space 矩阵。"""

    if not positions:
        raise ValueError(f"{label} raw joint sequence mapping cannot be empty")
    index_by_name = {
        str(name): index for index, name in enumerate(command_joint_names)
    }
    sample_count: int | None = None
    columns: dict[int, np.ndarray] = {}
    for name, values in positions.items():
        key = str(name)
        if key not in index_by_name:
            raise ValueError(f"{label} raw joint {key!r} is not in command-space")
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{label} raw joint {key!r} samples cannot be empty")
        if sample_count is None:
            sample_count = int(array.size)
        elif array.size != sample_count:
            raise ValueError(f"{label} raw joint sequence sample counts must match")
        columns[index_by_name[key]] = array
    assert sample_count is not None
    matrix = np.tile(
        np.asarray(base_command, dtype=float).reshape(1, -1),
        (sample_count, 1),
    )
    for index, values in columns.items():
        matrix[:, index] = values
    return matrix


def _validate_raw_side_sample_counts(
    left_positions: np.ndarray | None,
    right_positions: np.ndarray | None,
) -> None:
    """校验 raw sequence 左右两侧样本数一致，避免一侧提前结束。"""

    counts = [
        int(matrix.shape[0])
        for matrix in (left_positions, right_positions)
        if matrix is not None
    ]
    if not counts:
        raise ValueError("RawJointSequenceMoveSpec requires at least one side")
    if len(set(counts)) != 1:
        raise ValueError("left/right raw joint sequence sample counts must match")


def _hand_command_joint_names(command_joint_names: Sequence[str]) -> tuple[str, ...]:
    """从一侧 command-space 名称中过滤出 hand 组件关节。"""

    return tuple(
        str(name)
        for name in command_joint_names
        if component_for_name(str(name)) == "hand"
    )


def _command_linear_trajectory(
    *,
    command_joint_names: Sequence[str],
    start_command: np.ndarray,
    target_command: np.ndarray,
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """在 command-space 中生成一条线性插值轨迹。"""

    start = np.asarray(start_command, dtype=float).reshape(-1)
    target = np.asarray(target_command, dtype=float).reshape(-1)
    if start.size != target.size:
        raise ValueError(
            f"command trajectory shape mismatch: {start.size} vs {target.size}"
        )
    times = trajectory_sample_times(duration_s=duration_s, sample_dt=sample_dt)
    duration = max(float(duration_s), float(sample_dt))
    alpha = (times / duration).reshape(-1, 1)
    positions = start.reshape(1, -1) + alpha * (target - start).reshape(1, -1)
    positions[-1] = target
    return joint_trajectory_from_positions(
        times=times,
        positions=positions,
        joint_names=tuple(str(name) for name in command_joint_names),
        phase=phase,
    )


def _dual_hand_duration(move: DualHandMoveSpec) -> float:
    """解析双手动作时长；父级缺省时取左右子手动作的最大时长。"""

    if move.duration_s is not None:
        return float(move.duration_s)
    durations = [
        float(hand.duration_s)
        for hand in (move.left, move.right)
        if hand is not None and hand.duration_s is not None
    ]
    if durations:
        return max(durations)
    raise ValueError("DualHandMoveSpec requires duration_s")


def _dual_hand_move_from_overlay(
    overlay: CommandOverlaySpec,
    *,
    duration_s: float | None,
    phase: str,
) -> DualHandMoveSpec:
    """把 overlay 转成可直接执行的 DualHandMoveSpec。"""

    return DualHandMoveSpec(
        left=_hand_with_default_duration(overlay.left_hand, duration_s),
        right=_hand_with_default_duration(overlay.right_hand, duration_s),
        duration_s=duration_s,
        phase=phase,
    )


def _hand_with_default_duration(
    hand: HandMoveSpec | None,
    duration_s: float | None,
) -> HandMoveSpec | None:
    """给 overlay 手部动作补默认时长；显式时长优先。"""

    if hand is None or hand.duration_s is not None:
        return hand
    if duration_s is None:
        raise ValueError("hand overlay duration_s is required")
    return replace(hand, duration_s=float(duration_s))
