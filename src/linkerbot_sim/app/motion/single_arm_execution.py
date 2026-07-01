"""单臂 motion 的 planner 调用和 C-space 轨迹执行辅助。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.app.motion.runtime import (
    command_indices_for_cspace_joints,
    cspace_trajectory_from_motion_result,
    cumotion_boundary,
)
from linkerbot_sim.app.motion.single_arm_cspace import cspace_goal_to_command
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.steps import CommandPositionTrajectoryStep
from linkerbot_sim.planning.requests import MotionRequest, SpecifiedPathRequest
from linkerbot_sim.trajectories.command_trajectory import (
    command_trajectory_from_arm_trajectory,
)
from linkerbot_sim.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class SingleMoveExecutionResult:
    """单个 move 执行后的滚动状态，用于串联后续 move。"""

    step: int
    cspace_q: np.ndarray
    command: np.ndarray


def plan_cspace_trajectory(
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
    """调用 cuMotion planner，并把 MotionResult 归一化为项目 JointTrajectory。"""

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


def execute_cspace_trajectory(
    *,
    runtime: ExecutionRuntime,
    trajectory: JointTrajectory,
    joint_names: Sequence[str],
    command_start: np.ndarray,
    step: int,
    phase: str,
) -> SingleMoveExecutionResult:
    """把单臂 C-space 轨迹投影到 controller command-space 并执行。"""

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
    return SingleMoveExecutionResult(
        step=step,
        cspace_q=goal_q,
        command=np.asarray(command_trajectory.positions[-1], dtype=float),
    )
