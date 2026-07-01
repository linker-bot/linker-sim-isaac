"""单臂 cuMotion 通用动作执行封装。"""

from __future__ import annotations

from collections.abc import Sequence
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
    CartesianTcpFrameSpec,
    CumotionMoveSpec,
    IkOffsetMoveSpec,
    MoveSpec,
    SpecifiedPathMoveSpec,
    normalize_move_sequence,
    specified_path_planner_config,
    tcp_transform_from_spec,
)
from linkerbot_sim.app.motion.runtime import (
    cspace_trajectory_from_motion_result,
    duration_for_move,
    explicit_tcp_frame_name,
    phase_for_move,
    solve_ik_request,
)
from linkerbot_sim.app.runtime.single_robot import SingleRobotRuntime
from linkerbot_sim.backends.cumotion.tcp_context import make_cumotion_context
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.steps import (
    HoldCommandPositionTargetStep,
)
from linkerbot_sim.planning.requests import IKRequest, MotionRequest, SpecifiedPathRequest


DEFAULT_HOLD_REFRESH_DURATION_S = 0.25

__all__ = [
    "DEFAULT_HOLD_REFRESH_DURATION_S",
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
    tcp: CartesianTcpFrameSpec,
    moves: Sequence[MoveSpec],
    start_step: int = 0,
) -> int:
    """执行客户传入的单臂 cuMotion move 序列，并返回累计 physics step。"""

    tcp_transform = tcp_transform_from_spec(tcp)
    move_specs = normalize_move_sequence(moves)
    execution = runtime.execution
    physics_dt = float(execution.simulation_world.get_physics_dt())
    sample_dt = max(physics_dt, 1.0e-6)
    command = current_command(execution)

    step = int(start_step)
    with make_cumotion_context(runtime.robot_cumotion, tcp=tcp_transform) as context:
        context.clear_collision_world()
        joint_names = tuple(context.joint_names())
        cspace_q = current_cspace_command(
            execution,
            joint_names=joint_names,
            current_command_values=command,
        )
        for move_index, move in enumerate(move_specs, start=1):
            result = _run_single_move(
                move,
                move_index=move_index,
                context=context,
                current_q=cspace_q,
                joint_names=joint_names,
                runtime=execution,
                command=command,
                step=step,
                sample_dt=sample_dt,
                motion_planner_config=runtime.motion_planner_config,
                default_tcp_frame_name=tcp.frame_name,
            )
            step = result.step
            cspace_q = result.cspace_q
            command = result.command
    return step


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
    default_tcp_frame_name: str,
) -> SingleMoveExecutionResult:
    """执行一个单臂 move，并返回更新后的 step、C-space 和 command-space 状态。"""

    tcp_frame_name = _move_tcp_frame_name(move, default_tcp_frame_name)
    duration_s = duration_for_move(move)
    phase = phase_for_move(move)

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
        )
        print(
            "CUMOTION_IK_OK "
            f"move={move_index} tcp={tcp_frame_name} phase={phase} "
            f"position_error={float(ik_result.position_error):.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return result

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
        )

    raise TypeError(f"unsupported move spec type: {type(move).__name__}")


def _move_tcp_frame_name(move: MoveSpec, default_tcp_frame_name: str) -> str:
    """返回 move 指定的 TCP；未指定时使用运行入口传入的默认 TCP。"""

    return explicit_tcp_frame_name(move) or str(default_tcp_frame_name)
