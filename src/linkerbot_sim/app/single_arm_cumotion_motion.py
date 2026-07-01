"""单臂 cuMotion 通用动作执行封装。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from linkerbot_sim.app.cumotion_motion_specs import (
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
from linkerbot_sim.app.cumotion_motion_runtime import (
    command_indices_for_cspace_joints,
    cspace_goal_to_command_vector,
    cspace_linear_trajectory as _shared_cspace_linear_trajectory,
    cspace_trajectory_from_motion_result,
    cspace_vector_from_command,
    current_command_from_runtime,
    cumotion_boundary,
    duration_for_move,
    explicit_tcp_frame_name,
    phase_for_move,
    solve_ik_request,
)
from linkerbot_sim.app.single_robot_runtime import SingleRobotRuntime
from linkerbot_sim.backends.cumotion.tcp_context import make_cumotion_context
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.steps import (
    CommandPositionTrajectoryStep,
    HoldCommandPositionTargetStep,
)
from linkerbot_sim.planning.requests import IKRequest, MotionRequest, SpecifiedPathRequest
from linkerbot_sim.trajectories.command_trajectory import (
    command_trajectory_from_arm_trajectory,
)
from linkerbot_sim.trajectories.types import JointTrajectory


DEFAULT_HOLD_REFRESH_DURATION_S = 0.25


@dataclass(frozen=True)
class _SingleMoveExecutionResult:
    step: int
    cspace_q: np.ndarray
    command: np.ndarray


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


def current_command(runtime: ExecutionRuntime) -> np.ndarray:
    """读取 articulation 当前关节位置，并投影到 controller command-space。"""

    return current_command_from_runtime(runtime)


def current_cspace_command(
    runtime: ExecutionRuntime,
    *,
    joint_names: Sequence[str],
    current_command_values: np.ndarray | None = None,
) -> np.ndarray:
    """按 cuMotion C-space 关节名拼出当前单臂关节向量。"""

    command = (
        current_command(runtime)
        if current_command_values is None
        else np.asarray(current_command_values, dtype=float).reshape(-1)
    )
    return cspace_vector_from_command(
        joint_names=joint_names,
        command_joint_names=runtime.joint_controller.command_joint_names,
        command=command,
        label="command",
    )


def cspace_goal_to_command(
    *,
    runtime: ExecutionRuntime,
    base_command: np.ndarray,
    joint_names: Sequence[str],
    goal_q: np.ndarray,
) -> np.ndarray:
    """把 C-space goal 中的机械臂关节写回单臂 command-space。"""

    return cspace_goal_to_command_vector(
        command_joint_names=runtime.joint_controller.command_joint_names,
        base_command=base_command,
        joint_names=joint_names,
        goal_q=goal_q,
    )


def cspace_linear_trajectory(
    *,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """为单臂 IK 动作构造一条简单 C-space 插值轨迹。"""

    return _shared_cspace_linear_trajectory(
        start_q=start_q,
        goal_q=goal_q,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )


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
) -> _SingleMoveExecutionResult:
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
        result = _execute_cspace_trajectory(
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
        trajectory = _plan_cspace_trajectory(
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
        return _execute_cspace_trajectory(
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
        trajectory = _plan_cspace_trajectory(
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
        return _execute_cspace_trajectory(
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
            return _execute_cspace_trajectory(
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
        trajectory = _plan_cspace_trajectory(
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
        return _execute_cspace_trajectory(
            runtime=runtime,
            trajectory=trajectory,
            joint_names=joint_names,
            command_start=command,
            step=step,
            phase=phase,
        )

    raise TypeError(f"unsupported move spec type: {type(move).__name__}")


def _plan_cspace_trajectory(
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
) -> JointTrajectory:
    planner = context.make_motion_planner(
        tcp_frame_name=tcp_frame_name,
        config=config,
    )
    print(
        "CUMOTION_PLAN_START "
        f"move={move_index} tcp={tcp_frame_name} phase={phase} "
        f"type={type(request).__name__} pipeline={config.planning_pipeline}",
        flush=True,
    )
    result = cumotion_boundary("motion planner", planner.plan, request)
    if not result.success:
        raise RuntimeError(
            "cuMotion path planning failed: "
            f"phase={phase} status={result.status} "
            f"message={result.diagnostics.message}"
        )
    trajectory = cspace_trajectory_from_motion_result(
        result,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )
    metrics = result.diagnostics.metrics
    print(
        "CUMOTION_PLAN_OK "
        f"move={move_index} tcp={tcp_frame_name} phase={phase} "
        f"pipeline={config.planning_pipeline} status={result.status} "
        f"samples={len(trajectory)} "
        f"path_waypoints={int(metrics.get('num_waypoints', 0.0))} "
        f"path_length={metrics.get('path_length', 0.0):.6g}",
        flush=True,
    )
    return trajectory


def _execute_cspace_trajectory(
    *,
    runtime: ExecutionRuntime,
    trajectory: JointTrajectory,
    joint_names: Sequence[str],
    command_start: np.ndarray,
    step: int,
    phase: str,
) -> _SingleMoveExecutionResult:
    if len(trajectory) == 0:
        raise ValueError(f"trajectory for {phase!r} cannot be empty")
    goal_q = np.asarray(trajectory.positions[-1], dtype=float).reshape(-1)
    target_command = cspace_goal_to_command(
        runtime=runtime,
        base_command=command_start,
        joint_names=joint_names,
        goal_q=goal_q,
    )
    arm_command_indices = command_indices_for_cspace_joints(
        command_joint_names=runtime.joint_controller.command_joint_names,
        cspace_joint_names=joint_names,
    )
    command_trajectory = command_trajectory_from_arm_trajectory(
        arm_trajectory=trajectory,
        command_joint_names=runtime.joint_controller.command_joint_names,
        arm_command_indices=arm_command_indices,
        start_command=command_start,
        target_command=target_command,
        phase=phase,
    )
    step = CommandPositionTrajectoryStep(command_trajectory).run(runtime, step)
    return _SingleMoveExecutionResult(
        step=step,
        cspace_q=goal_q,
        command=np.asarray(command_trajectory.positions[-1], dtype=float),
    )


def _move_tcp_frame_name(move: MoveSpec, default_tcp_frame_name: str) -> str:
    return explicit_tcp_frame_name(move) or str(default_tcp_frame_name)
