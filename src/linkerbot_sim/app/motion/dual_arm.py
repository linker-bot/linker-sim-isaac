"""双臂 cuMotion 通用动作执行封装。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from linkerbot_sim.app.motion.dual_arm_cspace import (
    current_command,
    current_dual_cspace_command,
    dual_cspace_goal_to_command,
    dual_cspace_linear_trajectory,
    dual_cspace_vector_from_side_commands,
    dual_cspace_trajectory_from_motion_result,
    IkMotionGoal,
    selected_side_dual_trajectory,
    side_joint_absolute_goal,
    side_joint_delta_goal,
    solve_side_ik_goal,
)
from linkerbot_sim.app.motion.dual_arm_execution import (
    DualMoveExecutionResult,
    execute_dual_cspace_trajectory,
    execute_dual_hand_move,
    execute_hand_move,
    execute_overlays,
    execute_raw_joint_sequence_move,
    finish_arm_move_with_after_overlays,
    normalize_side,
    plan_dual_motion_trajectory,
    raise_if_requested_stop,
)
from linkerbot_sim.app.motion.dual_arm_semantics import (
    dual_arm_semantics_from_robot_configs,
)
from linkerbot_sim.app.motion.specs import (
    CSpaceDeltaPlanMoveSpec,
    CSpaceGoalPlanMoveSpec,
    CumotionMoveSpec,
    DualHandMoveSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    MoveSpec,
    RawJointSequenceMoveSpec,
    SpecifiedPathMoveSpec,
    normalize_move_sequence,
    specified_path_planner_config,
)
from linkerbot_sim.app.motion.runtime import (
    duration_for_move,
    explicit_tcp_frame_name,
    MotionPlanningFailed,
    phase_for_move,
    solve_ik_request,
)
from linkerbot_sim.app.runtime.dual_robot import (
    DualRobotAppRuntime,
    load_dual_robot_runtime_config,
)
from linkerbot_sim.assets.robot_loader import dual_robot_root_poses_from_env_config
from linkerbot_sim.backends.cumotion.profile_config import (
    merged_robot_config_with_cumotion_profile,
    motion_planner_config_from_profile,
    robot_cumotion_config,
)
from linkerbot_sim.backends.cumotion.context import (
    CuMotionContext,
    resolve_tcp_frame_name,
)
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.execution.dual_runtime import DualRobotRuntime
from linkerbot_sim.execution.dual_steps import (
    DualCommandPositionTargetStep,
)
from linkerbot_sim.planning.dual_arm_cspace_partition import (
    DualArmJointPartitions,
    selected_side_goal,
)
from linkerbot_sim.planning.requests import IKRequest, MotionRequest, SpecifiedPathRequest


DEFAULT_HOLD_REFRESH_DURATION_S = 0.25

__all__ = [
    "DEFAULT_HOLD_REFRESH_DURATION_S",
    "DualArmCuMotionExecutionSession",
    "DualArmCuMotionRunResult",
    "DualArmCuMotionSummary",
    "IkMotionGoal",
    "current_command",
    "current_dual_cspace_command",
    "dual_arm_cumotion_summary",
    "dual_cspace_goal_to_command",
    "dual_cspace_linear_trajectory",
    "dual_cspace_trajectory_from_motion_result",
    "dual_cspace_vector_from_side_commands",
    "hold_dual_current_pose",
    "run_dual_arm_cumotion_motion",
    "run_dual_arm_cumotion_motion_result",
    "selected_side_dual_trajectory",
    "side_joint_absolute_goal",
    "side_joint_delta_goal",
    "solve_side_ik_goal",
]


@dataclass(frozen=True)
class DualArmCuMotionSummary:
    """双臂 cuMotion 配置摘要，供脚本打印和 dry-run 检查。"""

    env: str
    cumotion_profile: str
    planning_pipeline: str
    left_default_tcp_frame: str
    right_default_tcp_frame: str
    left_flange_frame: str
    right_flange_frame: str


@dataclass(frozen=True)
class DualArmCuMotionRunResult:
    """双臂 move 序列执行结果。

    ``success=False`` 表示 cuMotion 正常返回了求解失败；这时 Isaac runtime 仍可继续保持或接收
    后续命令。配置错误和运行时异常不会被包装成这个结果。
    """

    success: bool
    step: int
    failed_move_index: int | None = None
    phase: str | None = None
    status: str | None = None
    message: str = ""
    side: str | None = None
    tcp_frame_name: str | None = None
    component: str | None = None


class DualArmCuMotionExecutionSession:
    """Long-lived dual-arm cuMotion execution context."""

    def __init__(
        self,
        runtime: DualRobotAppRuntime,
        *,
        cumotion_profile: str = "default",
    ) -> None:
        """加载双臂 cuMotion context、规划配置和左右 C-space 分区。"""

        self.runtime = runtime
        self.execution = runtime.execution
        self.cumotion_profile = str(cumotion_profile)
        self.dual_arm_semantics = dual_arm_semantics_from_robot_configs(
            runtime.side_robot_configs
        )
        self.default_tcp_by_side = {
            "left": self.dual_arm_semantics.left_default_tcp_frame,
            "right": self.dual_arm_semantics.right_default_tcp_frame,
        }

        physics_dt = float(self.execution.simulation_world.get_physics_dt())
        self.sample_dt = max(physics_dt, 1.0e-6)

        cumotion_profile_data = load_profile_yaml("cumotion", self.cumotion_profile)
        robot_config = merged_robot_config_with_cumotion_profile(
            runtime.robot_config,
            cumotion_profile_data,
        )
        cumotion_config = robot_cumotion_config(
            robot_config,
            dual_root_poses=dual_robot_root_poses_from_env_config(
                runtime.env_config
            ),
        )
        self.motion_planner_config = motion_planner_config_from_profile(
            cumotion_profile_data
        )
        self.context = CuMotionContext(cumotion_config)
        self._closed = False
        try:
            self.context.clear_collision_world()
            self.joint_names = tuple(self.context.joint_names())
            self.partitions = DualArmJointPartitions.from_joint_names(
                self.joint_names,
                left_joint_names=self.dual_arm_semantics.left_arm_joints,
                right_joint_names=self.dual_arm_semantics.right_arm_joints,
            )
            self.step = 0
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """释放长期持有的 cuMotion context manager。"""

        if self._closed:
            return
        self._closed = True

    def __enter__(self) -> "DualArmCuMotionExecutionSession":
        """让 session 可用于 with 语句，复用已加载的 cuMotion context。"""

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

        result = self.execute_moves_result(
            moves,
            start_step=start_step,
            should_stop=should_stop,
            raise_on_failure=True,
        )
        return result.step

    def execute_moves_result(
        self,
        moves: Sequence[MoveSpec],
        *,
        start_step: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        raise_on_failure: bool = False,
    ) -> DualArmCuMotionRunResult:
        """Execute moves and return a structured result for recoverable planner failures."""

        move_specs = normalize_move_sequence(moves)
        step = self.step if start_step is None else int(start_step)
        current_q, left_command, right_command = self.refresh_current_state()

        for move_index, move in enumerate(move_specs, start=1):
            raise_if_requested_stop(should_stop)
            try:
                result = _run_dual_move(
                    move,
                    move_index=move_index,
                    context=self.context,
                    partitions=self.partitions,
                    current_q=current_q,
                    joint_names=self.joint_names,
                    runtime=self.execution,
                    left_command=left_command,
                    right_command=right_command,
                    step=step,
                    sample_dt=self.sample_dt,
                    motion_planner_config=self.motion_planner_config,
                    default_tcp_by_side=self.default_tcp_by_side,
                    should_stop=should_stop,
                )
            except MotionPlanningFailed as exc:
                if raise_on_failure:
                    raise
                failure = _dual_run_failure_result(
                    exc,
                    move=move,
                    move_index=move_index,
                    step=step,
                )
                self.step = step
                return failure
            step = result.step
            current_q = result.cspace_q
            left_command = result.left_command
            right_command = result.right_command

        self.step = step
        return DualArmCuMotionRunResult(success=True, step=step)

    def refresh_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read current command-space and merged C-space vectors from Isaac."""

        left_command = current_command(self.execution.left)
        right_command = current_command(self.execution.right)
        current_q = dual_cspace_vector_from_side_commands(
            joint_names=self.joint_names,
            left_command_joint_names=(
                self.execution.left.joint_controller.command_joint_names
            ),
            right_command_joint_names=(
                self.execution.right.joint_controller.command_joint_names
            ),
            left_command=left_command,
            right_command=right_command,
        )
        return current_q, left_command, right_command


def dual_arm_cumotion_summary(
    *,
    env: str,
    cumotion_profile: str,
) -> DualArmCuMotionSummary:
    """加载 profile 并返回不创建后端 context 的 cuMotion 摘要。"""

    runtime_config = load_dual_robot_runtime_config(env=env)
    semantics = dual_arm_semantics_from_robot_configs(runtime_config.side_robot_configs)
    cumotion_config = load_profile_yaml("cumotion", cumotion_profile)
    motion_config = motion_planner_config_from_profile(cumotion_config)
    return DualArmCuMotionSummary(
        env=env,
        cumotion_profile=cumotion_profile,
        planning_pipeline=motion_config.planning_pipeline,
        left_default_tcp_frame=semantics.left_default_tcp_frame,
        right_default_tcp_frame=semantics.right_default_tcp_frame,
        left_flange_frame=semantics.left_flange_frame,
        right_flange_frame=semantics.right_flange_frame,
    )


def run_dual_arm_cumotion_motion(
    runtime: DualRobotAppRuntime,
    *,
    moves: Sequence[MoveSpec],
    start_step: int = 0,
    cumotion_profile: str = "default",
) -> int:
    """执行客户传入的双臂 cuMotion move 序列，并返回累计 physics step。"""

    with DualArmCuMotionExecutionSession(
        runtime,
        cumotion_profile=cumotion_profile,
    ) as session:
        return session.execute_moves(moves, start_step=start_step)


def run_dual_arm_cumotion_motion_result(
    runtime: DualRobotAppRuntime,
    *,
    moves: Sequence[MoveSpec],
    start_step: int = 0,
    cumotion_profile: str = "default",
) -> DualArmCuMotionRunResult:
    """执行双臂 cuMotion move 序列，并把可恢复求解失败作为结果返回。"""

    with DualArmCuMotionExecutionSession(
        runtime,
        cumotion_profile=cumotion_profile,
    ) as session:
        return session.execute_moves_result(
            moves,
            start_step=start_step,
            raise_on_failure=False,
        )


def hold_dual_current_pose(
    runtime: DualRobotRuntime,
    *,
    step: int,
    simulation_app,
    refresh_duration_s: float = DEFAULT_HOLD_REFRESH_DURATION_S,
) -> int:
    """GUI 调试时在完整动作结束后保持当前左右 command target。"""

    while simulation_app.is_running():
        current_left = current_command(runtime.left)
        current_right = current_command(runtime.right)
        step = DualCommandPositionTargetStep(
            left_start_command=current_left,
            right_start_command=current_right,
            left_target_command=current_left,
            right_target_command=current_right,
            duration=refresh_duration_s,
            phase="dual_hold",
        ).run(runtime, step)
    return step


def _run_dual_move(
    move: MoveSpec,
    *,
    move_index: int,
    context,
    partitions: DualArmJointPartitions,
    current_q: np.ndarray,
    joint_names: Sequence[str],
    runtime: DualRobotRuntime,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    motion_planner_config,
    default_tcp_by_side: dict[str, str],
    should_stop: Callable[[], bool] | None = None,
) -> DualMoveExecutionResult:
    """分派并执行一个双臂 move，统一维护 C-space 和左右 command-space 状态。"""

    if isinstance(move, HandMoveSpec):
        move.validate()
        return execute_hand_move(
            move,
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            should_stop=should_stop,
        )

    if isinstance(move, DualHandMoveSpec):
        move.validate()
        return execute_dual_hand_move(
            move,
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            should_stop=should_stop,
        )

    if isinstance(move, RawJointSequenceMoveSpec):
        move.validate()
        return execute_raw_joint_sequence_move(
            move,
            runtime=runtime,
            joint_names=joint_names,
            left_command=left_command,
            right_command=right_command,
            step=step,
            should_stop=should_stop,
        )

    execution_mode = _dual_execution_mode(move)
    side = None if execution_mode == "dual_cspace" else _move_side_required(move)
    tcp_frame_name = _move_tcp_frame_name(
        move,
        context=context,
        default_tcp_by_side=default_tcp_by_side,
        side=side,
    )
    duration_s = duration_for_move(move)
    phase = phase_for_move(
        move,
        side=side,
        dual_cspace=execution_mode == "dual_cspace",
    )

    if isinstance(move, IkOffsetMoveSpec):
        move.validate(require_side=True)
        step, left_command, right_command = execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        ik_goal = solve_side_ik_goal(
            context=context,
            partitions=partitions,
            current_q=current_q,
            side=side,
            tcp_frame_name=tcp_frame_name,
            offset=np.asarray(move.tcp_offset, dtype=float),
            orientation_mode=move.orientation_mode,
            target_orientation=(
                None
                if move.target_orientation is None
                else np.asarray(move.target_orientation, dtype=float)
            ),
        )
        trajectory = dual_cspace_linear_trajectory(
            start_q=current_q,
            goal_q=ik_goal.goal_q,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
        )
        result = execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        print(
            "DUAL_ARM_CUMOTION_IK_OK "
            f"move={move_index} side={ik_goal.side} tcp={ik_goal.tcp_frame_name} "
            f"phase={phase} position_error={ik_goal.position_error:.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=ik_goal.goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CumotionMoveSpec) and isinstance(move.request, IKRequest):
        move.validate(require_side=execution_mode != "dual_cspace")
        step, left_command, right_command = execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        request = _ik_request_with_runtime_defaults(
            move.request,
            current_q=current_q,
            tcp_frame_name=tcp_frame_name,
        )
        ik_result = solve_ik_request(
            context,
            request,
            tcp_frame_name=tcp_frame_name,
            label="dual-arm",
        )
        goal_q = (
            np.asarray(ik_result.joint_positions, dtype=float).reshape(-1)
            if execution_mode == "dual_cspace"
            else selected_side_goal(
                base_q=current_q,
                solved_q=ik_result.joint_positions,
                partitions=partitions,
                active_side=side,
            )
        )
        trajectory = dual_cspace_linear_trajectory(
            start_q=current_q,
            goal_q=goal_q,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
        )
        result = execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        print(
            "DUAL_ARM_CUMOTION_IK_OK "
            f"move={move_index} side={side} tcp={tcp_frame_name} phase={phase} "
            f"position_error={float(ik_result.position_error):.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CSpaceGoalPlanMoveSpec):
        move.validate(require_side=True)
        step, left_command, right_command = execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        goal_q = side_joint_absolute_goal(
            base_q=current_q,
            partitions=partitions,
            side=side,
            joint_positions=move.joint_positions,
        )
        request = MotionRequest(
            current_q=current_q,
            goal_q=goal_q,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        trajectory = plan_dual_motion_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=motion_planner_config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
            side=side,
            execution_mode=execution_mode,
        )
        trajectory = selected_side_dual_trajectory(
            trajectory=trajectory,
            base_q=current_q,
            partitions=partitions,
            active_side=side,
        )
        result = execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CSpaceDeltaPlanMoveSpec):
        move.validate(require_side=True)
        step, left_command, right_command = execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        goal_q = side_joint_delta_goal(
            base_q=current_q,
            partitions=partitions,
            side=side,
            deltas=move.joint_deltas,
        )
        request = MotionRequest(
            current_q=current_q,
            goal_q=goal_q,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        trajectory = plan_dual_motion_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=motion_planner_config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
            side=side,
            execution_mode=execution_mode,
        )
        trajectory = selected_side_dual_trajectory(
            trajectory=trajectory,
            base_q=current_q,
            partitions=partitions,
            active_side=side,
        )
        result = execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, SpecifiedPathMoveSpec):
        move.validate(require_side=True)
        step, left_command, right_command = execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        request = SpecifiedPathRequest(
            current_q=current_q,
            path=move.path,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        specified_config = specified_path_planner_config(
            motion_planner_config,
            path=move.path,
        )
        trajectory = plan_dual_motion_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=specified_config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
            side=side,
            execution_mode=execution_mode,
        )
        trajectory = selected_side_dual_trajectory(
            trajectory=trajectory,
            base_q=current_q,
            partitions=partitions,
            active_side=side,
        )
        result = execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=np.asarray(trajectory.positions[-1], dtype=float),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CumotionMoveSpec):
        move.validate(require_side=execution_mode != "dual_cspace")
        step, left_command, right_command = execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        request = _planning_request_with_runtime_defaults(
            move.request,
            duration_s=duration_s,
            tcp_frame_name=tcp_frame_name,
        )
        config = (
            specified_path_planner_config(motion_planner_config, path=request.path)
            if isinstance(request, SpecifiedPathRequest)
            else motion_planner_config
        )
        trajectory = plan_dual_motion_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
            side=side,
            execution_mode=execution_mode,
        )
        if execution_mode != "dual_cspace":
            trajectory = selected_side_dual_trajectory(
                trajectory=trajectory,
                base_q=current_q,
                partitions=partitions,
                active_side=side,
            )
        result = execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=np.asarray(trajectory.positions[-1], dtype=float),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    raise TypeError(f"unsupported move spec type: {type(move).__name__}")


def _dual_run_failure_result(
    exc: MotionPlanningFailed,
    *,
    move: MoveSpec,
    move_index: int,
    step: int,
) -> DualArmCuMotionRunResult:
    """把可恢复规划失败转成脚本/交互层可以直接报告的结果。"""

    return DualArmCuMotionRunResult(
        success=False,
        step=step,
        failed_move_index=exc.move_index or move_index,
        phase=exc.phase or _failure_phase_for_move(move),
        status=exc.status,
        message=exc.solver_message or str(exc),
        side=exc.side or _failure_side_for_move(move),
        tcp_frame_name=exc.tcp_frame_name or explicit_tcp_frame_name(move),
        component=exc.component,
    )


def _failure_phase_for_move(move: MoveSpec) -> str:
    """Best-effort phase lookup used only for recoverable failure reporting."""

    try:
        execution_mode = getattr(move, "execution", "single")
        side = None if execution_mode == "dual_cspace" else getattr(move, "side", None)
        return phase_for_move(
            move,
            side=None if side is None else str(side),
            dual_cspace=execution_mode == "dual_cspace",
        )
    except Exception:
        phase = getattr(move, "phase", None)
        return str(phase) if phase else type(move).__name__


def _failure_side_for_move(move: MoveSpec) -> str | None:
    """Best-effort side lookup used only for recoverable failure reporting."""

    execution_mode = getattr(move, "execution", "single")
    if execution_mode == "dual_cspace":
        return None
    side = getattr(move, "side", None)
    return None if side is None else str(side)


def _ik_request_with_runtime_defaults(
    request: IKRequest,
    *,
    current_q: np.ndarray,
    tcp_frame_name: str,
) -> IKRequest:
    """给 IK request 补运行期默认 TCP 和 warm-start C-space seed。"""

    return replace(
        request,
        tcp_frame_name=request.tcp_frame_name or tcp_frame_name,
        warm_start_ik_cspace_seed=(
            request.warm_start_ik_cspace_seed
            if request.warm_start_ik_cspace_seed is not None
            else np.asarray(current_q, dtype=float).reshape(-1)
        ),
    )


def _planning_request_with_runtime_defaults(
    request: MotionRequest | SpecifiedPathRequest,
    *,
    duration_s: float,
    tcp_frame_name: str,
) -> MotionRequest | SpecifiedPathRequest:
    """给规划 request 补运行期默认 TCP 和 duration。"""

    return replace(
        request,
        tcp_frame_name=request.tcp_frame_name or tcp_frame_name,
        duration_s=request.duration_s if request.duration_s is not None else duration_s,
    )


def _dual_execution_mode(move: MoveSpec) -> str:
    """把 move 映射为双臂执行模式：selected_side 或 dual_cspace。"""

    if isinstance(move, CumotionMoveSpec) and move.execution == "dual_cspace":
        return "dual_cspace"
    return "selected_side"


def _move_side_required(move: MoveSpec) -> str:
    """读取双臂 selected-side move 的 side 字段。"""

    side = getattr(move, "side", None)
    if side is None:
        raise ValueError(f"{type(move).__name__} requires side in dual-arm runtime")
    return normalize_side(side)


def _move_tcp_frame_name(
    move: MoveSpec,
    *,
    context,
    default_tcp_by_side: Mapping[str, str],
    side: str | None,
) -> str:
    """解析 move 使用的 TCP frame；selected-side 模式可从左右默认 TCP 推导。"""

    value = explicit_tcp_frame_name(move)
    default = None
    if side is not None:
        default = default_tcp_by_side.get(side)
    return resolve_tcp_frame_name(
        context,
        tcp_frame_name=value,
        default_tcp_frame_name=default,
        label=f"{type(move).__name__}.tcp_frame_name",
    )
