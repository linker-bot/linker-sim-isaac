"""单臂 cuMotion 通用动作执行封装。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np

from linkerbot_sim.app.motion.single_arm_cspace import (
    cspace_goal_to_command,
    cspace_linear_trajectory,
    current_command,
    current_cspace_command,
)
from linkerbot_sim.app.motion.single_arm_execution import (
    execute_cspace_trajectory,
    plan_cspace_trajectory,
    SingleMoveExecutionResult,
)
from linkerbot_sim.app.motion.specs import (
    CSpaceDeltaPlanMoveSpec,
    CSpaceGoalPlanMoveSpec,
    CumotionMoveSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    MoveSpec,
    RawJointSequenceMoveSpec,
    RawJointSequenceSideSpec,
    SpecifiedPathMoveSpec,
    normalize_move_sequence,
    specified_path_planner_config,
)
from linkerbot_sim.app.motion.runtime import (
    cspace_trajectory_from_motion_result,
    duration_for_move,
    explicit_tcp_frame_name,
    phase_for_move,
    solve_ik_request,
)
from linkerbot_sim.app.runtime.single_robot import SingleRobotRuntime
from linkerbot_sim.backends.cumotion.context import (
    CuMotionContext,
    resolve_tcp_frame_name,
)
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.steps import (
    CommandExecutionInterrupted,
    CommandPositionTrajectoryStep,
    HoldCommandPositionTargetStep,
    SmoothCommandPositionTargetStep,
)
from linkerbot_sim.planning.requests import (
    IKRequest,
    MotionRequest,
    SpecifiedPathRequest,
)
from linkerbot_sim.robots.classification import component_for_name
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)


DEFAULT_HOLD_REFRESH_DURATION_S = 0.25

__all__ = [
    "DEFAULT_HOLD_REFRESH_DURATION_S",
    "SingleArmCuMotionExecutionSession",
    "cspace_goal_to_command",
    "cspace_linear_trajectory",
    "cspace_trajectory_from_motion_result",
    "current_command",
    "current_cspace_command",
    "hold_single_current_pose",
    "run_single_arm_cumotion_motion",
]


def run_single_arm_cumotion_motion(
    runtime: SingleRobotRuntime,
    *,
    moves: Sequence[MoveSpec],
    start_step: int = 0,
) -> int:
    """执行客户传入的单臂 cuMotion move 序列，并返回累计 physics step。"""

    with SingleArmCuMotionExecutionSession(runtime) as session:
        return session.execute_moves(moves, start_step=start_step)


class SingleArmCuMotionExecutionSession:
    """Long-lived single-arm cuMotion execution context."""

    def __init__(
        self,
        runtime: SingleRobotRuntime,
    ) -> None:
        """加载单臂 cuMotion context，并缓存采样周期。"""

        self.runtime = runtime
        self.execution = runtime.execution
        physics_dt = float(self.execution.simulation_world.get_physics_dt())
        self.sample_dt = max(physics_dt, 1.0e-6)
        self.context = CuMotionContext(runtime.robot_cumotion)
        self._closed = False
        try:
            self.context.clear_collision_world()
            self.joint_names = tuple(self.context.joint_names())
            self.step = 0
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """释放长期持有的 cuMotion context manager。"""

        if self._closed:
            return
        self._closed = True

    def __enter__(self) -> "SingleArmCuMotionExecutionSession":
        """让 session 可用于 with 语句。"""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """退出 with 语句时关闭 cuMotion context。"""

        self.close()

    def execute_moves(
        self,
        moves: Sequence[MoveSpec],
        *,
        start_step: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        """Execute a move sequence using the already loaded cuMotion context."""

        move_specs = normalize_move_sequence(moves)
        step = self.step if start_step is None else int(start_step)
        cspace_q, command = self.refresh_current_state()
        for move_index, move in enumerate(move_specs, start=1):
            _raise_if_requested_stop(should_stop, step=step)
            result = _run_single_move(
                move,
                move_index=move_index,
                context=self.context,
                current_q=cspace_q,
                joint_names=self.joint_names,
                runtime=self.execution,
                command=command,
                step=step,
                sample_dt=self.sample_dt,
                motion_planner_config=self.runtime.motion_planner_config,
                should_stop=should_stop,
            )
            step = result.step
            cspace_q = result.cspace_q
            command = result.command
        self.step = step
        return step

    def refresh_current_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Read current command-space and C-space vectors from Isaac."""

        command = current_command(self.execution)
        cspace_q = current_cspace_command(
            self.execution,
            joint_names=self.joint_names,
            current_command_values=command,
        )
        return cspace_q, command


def hold_single_current_pose(
    runtime: ExecutionRuntime,
    *,
    step: int,
    simulation_app,
    refresh_duration_s: float = DEFAULT_HOLD_REFRESH_DURATION_S,
) -> int:
    """GUI 调试时在完整动作结束后保持当前 command target。"""

    if simulation_app is None:
        current = current_command(runtime)
        return HoldCommandPositionTargetStep(
            target_command=current,
            duration=refresh_duration_s,
            phase="hold",
        ).run(runtime, step)
    while simulation_app.is_running():
        current = current_command(runtime)
        step = HoldCommandPositionTargetStep(
            target_command=current,
            duration=refresh_duration_s,
            phase="hold",
        ).run(runtime, step)
    return step


def _run_single_move(
    move: MoveSpec,
    *,
    move_index: int,
    context,
    current_q: np.ndarray,
    joint_names: Sequence[str],
    runtime: ExecutionRuntime,
    command: np.ndarray,
    step: int,
    sample_dt: float,
    motion_planner_config,
    should_stop: Callable[[], bool] | None = None,
) -> SingleMoveExecutionResult:
    """执行一个单臂 move，并返回更新后的 step、C-space 和 command-space 状态。"""

    _raise_if_requested_stop(should_stop, step=step)

    if isinstance(move, HandMoveSpec):
        return _execute_single_hand_move(
            move,
            runtime=runtime,
            current_q=current_q,
            command=command,
            step=step,
            should_stop=should_stop,
        )

    if isinstance(move, RawJointSequenceMoveSpec):
        return _execute_single_raw_joint_sequence_move(
            move,
            runtime=runtime,
            joint_names=joint_names,
            command=command,
            step=step,
            sample_dt=sample_dt,
            should_stop=should_stop,
        )

    tcp_frame_name = _move_tcp_frame_name(move, context)
    _reject_single_overlays(move)
    phase = phase_for_move(move)
    duration_s = duration_for_move(move)

    if isinstance(move, IkOffsetMoveSpec):
        move.validate(require_side=False)
        fk = context.make_forward_kinematics()
        current_pose = fk.compute_pose(current_q, tcp_frame_name)
        ik_defaults = context.config.kinematics.ik
        request = IKRequest(
            target_position=(
                np.asarray(current_pose.position, dtype=float).reshape(3)
                + np.asarray(move.tcp_offset, dtype=float).reshape(3)
            ),
            target_orientation=current_pose.orientation,
            tcp_frame_name=tcp_frame_name,
            warm_start_ik_cspace_seed=current_q,
            position_tolerance=ik_defaults.position_tolerance,
            orientation_tolerance=ik_defaults.orientation_tolerance,
            avoid_collisions=False,
        )
        ik_result = solve_ik_request(
            context,
            request,
            tcp_frame_name=tcp_frame_name,
            label="single-arm",
        )
        trajectory = cspace_linear_trajectory(
            start_q=current_q,
            goal_q=ik_result.joint_positions,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
        )
        result = execute_cspace_trajectory(
            runtime=runtime,
            trajectory=trajectory,
            joint_names=joint_names,
            command_start=command,
            step=step,
            phase=phase,
            should_stop=should_stop,
        )
        print(
            "CUMOTION_IK_OK "
            f"move={move_index} tcp={tcp_frame_name} phase={phase} "
            f"position_error={float(ik_result.position_error):.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return result

    if isinstance(move, CSpaceGoalPlanMoveSpec):
        move.validate(require_side=False)
        goal_q = _single_joint_absolute_goal(
            base_q=current_q,
            joint_positions=move.joint_positions,
        )
        request = MotionRequest(
            current_q=current_q,
            goal_q=goal_q,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        trajectory = plan_cspace_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=motion_planner_config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
        )
        return execute_cspace_trajectory(
            runtime=runtime,
            trajectory=trajectory,
            joint_names=joint_names,
            command_start=command,
            step=step,
            phase=phase,
            should_stop=should_stop,
        )

    if isinstance(move, CSpaceDeltaPlanMoveSpec):
        move.validate(require_side=False)
        goal_q = np.asarray(current_q, dtype=float).reshape(-1).copy()
        for index, delta in enumerate(move.joint_deltas):
            if index >= goal_q.size:
                break
            goal_q[index] += float(delta)
        request = MotionRequest(
            current_q=current_q,
            goal_q=goal_q,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        trajectory = plan_cspace_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=motion_planner_config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
        )
        return execute_cspace_trajectory(
            runtime=runtime,
            trajectory=trajectory,
            joint_names=joint_names,
            command_start=command,
            step=step,
            phase=phase,
            should_stop=should_stop,
        )

    if isinstance(move, SpecifiedPathMoveSpec):
        move.validate(require_side=False)
        request = SpecifiedPathRequest(
            current_q=current_q,
            path=move.path,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        config = specified_path_planner_config(
            motion_planner_config,
            path=move.path,
        )
        trajectory = plan_cspace_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
        )
        return execute_cspace_trajectory(
            runtime=runtime,
            trajectory=trajectory,
            joint_names=joint_names,
            command_start=command,
            step=step,
            phase=phase,
            should_stop=should_stop,
        )

    if isinstance(move, CumotionMoveSpec):
        move.validate(require_side=False)
        if move.execution != "single":
            raise ValueError("single-arm runtime only accepts execution='single'")
        if isinstance(move.request, IKRequest):
            request = replace(
                move.request,
                tcp_frame_name=move.request.tcp_frame_name or tcp_frame_name,
                warm_start_ik_cspace_seed=(
                    move.request.warm_start_ik_cspace_seed
                    if move.request.warm_start_ik_cspace_seed is not None
                    else current_q
                ),
            )
            ik_result = solve_ik_request(
                context,
                request,
                tcp_frame_name=tcp_frame_name,
                label="single-arm",
            )
            trajectory = cspace_linear_trajectory(
                start_q=current_q,
                goal_q=ik_result.joint_positions,
                joint_names=joint_names,
                duration_s=duration_s,
                sample_dt=sample_dt,
                phase=phase,
            )
            return execute_cspace_trajectory(
                runtime=runtime,
                trajectory=trajectory,
                joint_names=joint_names,
                command_start=command,
                step=step,
                phase=phase,
                should_stop=should_stop,
            )
        request = replace(
            move.request,
            tcp_frame_name=move.request.tcp_frame_name or tcp_frame_name,
            duration_s=(
                move.request.duration_s
                if move.request.duration_s is not None
                else duration_s
            ),
        )
        config = (
            specified_path_planner_config(motion_planner_config, path=request.path)
            if isinstance(request, SpecifiedPathRequest)
            else motion_planner_config
        )
        trajectory = plan_cspace_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
        )
        return execute_cspace_trajectory(
            runtime=runtime,
            trajectory=trajectory,
            joint_names=joint_names,
            command_start=command,
            step=step,
            phase=phase,
            should_stop=should_stop,
        )

    raise TypeError(f"unsupported move spec type: {type(move).__name__}")


def _move_tcp_frame_name(move: MoveSpec, context) -> str:
    """返回 move 指定的 frame；未指定时使用 robot YAML 默认 frame。"""

    return resolve_tcp_frame_name(
        context,
        tcp_frame_name=explicit_tcp_frame_name(move),
        label="tcp_frame_name",
    )


def _execute_single_hand_move(
    move: HandMoveSpec,
    *,
    runtime: ExecutionRuntime,
    current_q: np.ndarray,
    command: np.ndarray,
    step: int,
    should_stop: Callable[[], bool] | None,
) -> SingleMoveExecutionResult:
    """执行单手 command-space 目标，不改变机械臂 C-space。"""

    move.validate(require_side=False)
    phase = phase_for_move(move)
    target = _hand_target_command(
        runtime,
        start_command=command,
        hand=move,
    )
    step = SmoothCommandPositionTargetStep(
        start_command=command,
        target_command=target,
        duration=duration_for_move(move),
        phase=phase,
        should_stop=should_stop,
    ).run(runtime, step)
    return SingleMoveExecutionResult(
        step=step,
        cspace_q=np.asarray(current_q, dtype=float).reshape(-1),
        command=np.asarray(target, dtype=float).reshape(-1),
    )


def _execute_single_raw_joint_sequence_move(
    move: RawJointSequenceMoveSpec,
    *,
    runtime: ExecutionRuntime,
    joint_names: Sequence[str],
    command: np.ndarray,
    step: int,
    sample_dt: float,
    should_stop: Callable[[], bool] | None,
) -> SingleMoveExecutionResult:
    """按 physics step 执行单臂 raw command target 序列。"""

    phase = move.phase or "raw_joint_sequence"
    positions = _single_raw_sequence_matrix(
        move,
        runtime=runtime,
        current_command=command,
    )
    interval = int(move.step_interval)
    if interval <= 0:
        raise ValueError("step_interval must be a positive integer")
    if interval > 1:
        positions = np.repeat(positions, interval, axis=0)
    times = np.asarray(
        [(index + 1) * float(sample_dt) for index in range(positions.shape[0])],
        dtype=float,
    )
    trajectory = joint_trajectory_from_positions(
        times=times,
        positions=positions,
        joint_names=runtime.joint_controller.command_joint_names,
        phase=phase,
    )
    step = CommandPositionTrajectoryStep(
        trajectory,
        should_stop=should_stop,
    ).run(runtime, step)
    next_command = np.asarray(positions[-1], dtype=float).reshape(-1)
    return SingleMoveExecutionResult(
        step=step,
        cspace_q=current_cspace_command(
            runtime,
            joint_names=joint_names,
            current_command_values=next_command,
        ),
        command=next_command,
    )


def _hand_target_command(
    runtime: ExecutionRuntime,
    *,
    start_command: np.ndarray,
    hand: HandMoveSpec,
) -> np.ndarray:
    """把手部目标写入单臂 command 向量，支持按关节名或按手部顺序输入。"""

    target = np.asarray(start_command, dtype=float).reshape(-1).copy()
    command_names = tuple(
        str(name) for name in runtime.joint_controller.command_joint_names
    )
    if target.size != len(command_names):
        raise ValueError(
            f"single command shape mismatch: {len(command_names)} names, "
            f"{target.size} values"
        )
    if isinstance(hand.joint_positions, Mapping):
        index_by_name = {name: index for index, name in enumerate(command_names)}
        for name, value in hand.joint_positions.items():
            key = str(name)
            if key not in index_by_name:
                raise ValueError(f"single hand joint {key!r} is not in command-space")
            target[index_by_name[key]] = float(value)
        return target
    hand_names = _hand_command_joint_names(command_names)
    values = np.asarray(hand.joint_positions, dtype=float).reshape(-1)
    if values.size > len(hand_names):
        raise ValueError(
            f"single hand target has {values.size} values, "
            f"but only {len(hand_names)} hand command joints exist"
        )
    index_by_name = {name: index for index, name in enumerate(command_names)}
    for name, value in zip(hand_names, values):
        target[index_by_name[name]] = float(value)
    return target


def _single_raw_sequence_matrix(
    move: RawJointSequenceMoveSpec,
    *,
    runtime: ExecutionRuntime,
    current_command: np.ndarray,
) -> np.ndarray:
    """把 raw sequence 规范化为 shape=(samples, command_dof) 的矩阵。"""

    sides = [side for side in (move.left, move.right) if side is not None]
    if len(sides) != 1:
        raise ValueError("single-arm raw_joint_sequence requires exactly one side")
    return _raw_sequence_side_matrix(
        sides[0],
        runtime=runtime,
        current_command=current_command,
    )


def _raw_sequence_side_matrix(
    side: RawJointSequenceSideSpec,
    *,
    runtime: ExecutionRuntime,
    current_command: np.ndarray,
) -> np.ndarray:
    """把单侧 raw sequence 展开成完整 command-space 矩阵。"""

    command_names = tuple(
        str(name) for name in runtime.joint_controller.command_joint_names
    )
    base = np.asarray(current_command, dtype=float).reshape(-1)
    if base.size != len(command_names):
        raise ValueError(
            f"single command shape mismatch: {len(command_names)} names, "
            f"{base.size} values"
        )
    positions = side.joint_positions
    if isinstance(positions, Mapping):
        matrix = _raw_mapping_sequence_matrix(
            positions,
            command_joint_names=command_names,
            base_command=base,
        )
    else:
        matrix = np.asarray(positions, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("single raw joint sequence must have shape (N, dof)")
        if matrix.shape[1] != len(command_names):
            raise ValueError(
                f"single raw joint sequence expected {len(command_names)} columns, "
                f"got {matrix.shape[1]}"
            )
    if matrix.shape[0] == 0:
        raise ValueError("single raw joint sequence cannot be empty")
    return matrix


def _raw_mapping_sequence_matrix(
    positions: Mapping[str, Sequence[float]],
    *,
    command_joint_names: Sequence[str],
    base_command: np.ndarray,
) -> np.ndarray:
    """把按关节名给出的 raw sequence 展开成完整 command-space 矩阵。"""

    if not positions:
        raise ValueError("single raw joint sequence mapping cannot be empty")
    index_by_name = {str(name): index for index, name in enumerate(command_joint_names)}
    sample_count: int | None = None
    columns: dict[int, np.ndarray] = {}
    for name, values in positions.items():
        key = str(name)
        if key not in index_by_name:
            raise ValueError(f"single raw joint {key!r} is not in command-space")
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"single raw joint {key!r} samples cannot be empty")
        if sample_count is None:
            sample_count = int(array.size)
        elif array.size != sample_count:
            raise ValueError("single raw joint sequence sample counts must match")
        columns[index_by_name[key]] = array
    assert sample_count is not None
    matrix = np.tile(
        np.asarray(base_command, dtype=float).reshape(1, -1),
        (sample_count, 1),
    )
    for index, values in columns.items():
        matrix[:, index] = values
    return matrix


def _single_joint_absolute_goal(
    *,
    base_q: np.ndarray,
    joint_positions: Sequence[float],
) -> np.ndarray:
    """在单臂 C-space 中写入绝对关节角目标。"""

    goal = np.asarray(base_q, dtype=float).reshape(-1).copy()
    for index, position in enumerate(joint_positions):
        if index >= goal.size:
            break
        goal[index] = float(position)
    return goal


def _hand_command_joint_names(command_joint_names: Sequence[str]) -> tuple[str, ...]:
    """从 command-space 名称中过滤出 hand 组件关节。"""

    return tuple(
        str(name)
        for name in command_joint_names
        if component_for_name(str(name)) == "hand"
    )


def _reject_single_overlays(move: MoveSpec) -> None:
    """单臂 runtime 暂不合并 overlay；调用方可把 hand 作为独立 move 发送。"""

    overlays = getattr(move, "overlays", ())
    if overlays:
        raise ValueError(
            "single-arm runtime does not support overlays; "
            "send hand commands as separate moves"
        )


def _raise_if_requested_stop(
    should_stop: Callable[[], bool] | None,
    *,
    step: int,
) -> None:
    """把外部 cancel/estop/quit 请求转成单臂中断异常。"""

    if should_stop is not None and should_stop():
        raise CommandExecutionInterrupted(
            "single command execution interrupted",
            step=step,
        )
