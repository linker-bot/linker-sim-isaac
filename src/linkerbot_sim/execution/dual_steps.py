"""双机器人同步执行步骤。

本模块用于左右两个 Isaac articulation 同时存在的场景。每个 physics step 内先分别向左右
controller 下发目标，再统一推进一次 ``world.step()``，避免左右机器人在仿真时间上串行执行。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.execution.dual_runtime import DualRobotRuntime, RobotSideRuntime
from linkerbot_sim.trajectories.types import JointTrajectory


class DualCommandExecutionInterrupted(RuntimeError):
    """Raised when a dual command step is stopped by an external request."""

    def __init__(self, message: str, *, step: int | None = None) -> None:
        super().__init__(message)
        self.step = step


@dataclass(frozen=True)
class DualCommandPositionTargetStep:
    """同步平滑移动左右 command-space 目标。"""

    left_target_command: np.ndarray | None
    right_target_command: np.ndarray | None
    duration: float
    phase: str
    left_start_command: np.ndarray | None = None
    right_start_command: np.ndarray | None = None
    left_base_positions: np.ndarray | None = None
    right_base_positions: np.ndarray | None = None
    should_stop: Callable[[], bool] | None = None

    def run(self, runtime: DualRobotRuntime, step: int) -> int:
        """执行同步 smoothstep 并返回累计 step。"""

        return execute_dual_smooth_command_position_target(
            runtime=runtime,
            left_target_command=self.left_target_command,
            right_target_command=self.right_target_command,
            duration=self.duration,
            phase=self.phase,
            step=step,
            left_start_command=self.left_start_command,
            right_start_command=self.right_start_command,
            left_base_positions=self.left_base_positions,
            right_base_positions=self.right_base_positions,
            should_stop=self.should_stop,
        )


@dataclass(frozen=True)
class DualCommandPositionTrajectoryStep:
    """同步播放左右 command-space 位置轨迹。"""

    left_trajectory: JointTrajectory | None
    right_trajectory: JointTrajectory | None
    phase: str | None = None
    left_base_positions: np.ndarray | None = None
    right_base_positions: np.ndarray | None = None
    should_stop: Callable[[], bool] | None = None

    def run(self, runtime: DualRobotRuntime, step: int) -> int:
        """逐样本同步播放左右轨迹并返回累计 step。"""

        return execute_dual_command_position_trajectory(
            runtime=runtime,
            left_trajectory=self.left_trajectory,
            right_trajectory=self.right_trajectory,
            step=step,
            phase=self.phase,
            left_base_positions=self.left_base_positions,
            right_base_positions=self.right_base_positions,
            should_stop=self.should_stop,
        )


@dataclass
class _SmoothSidePlan:
    controller: object
    start: np.ndarray
    target: np.ndarray
    delta: np.ndarray
    base: np.ndarray


def execute_dual_smooth_command_position_target(
    *,
    runtime: DualRobotRuntime,
    left_target_command: np.ndarray | None,
    right_target_command: np.ndarray | None,
    duration: float,
    phase: str,
    step: int,
    left_start_command: np.ndarray | None = None,
    right_start_command: np.ndarray | None = None,
    left_base_positions: np.ndarray | None = None,
    right_base_positions: np.ndarray | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """用同一个时间网格同步移动左右 command-space target。"""

    physics_dt = float(runtime.simulation_world.get_physics_dt())
    steps = max(1, int(round(float(duration) / physics_dt)))
    left_plan = _smooth_side_plan(
        runtime.left,
        target_command=left_target_command,
        start_command=left_start_command,
        base_positions=left_base_positions,
    )
    right_plan = _smooth_side_plan(
        runtime.right,
        target_command=right_target_command,
        start_command=right_start_command,
        base_positions=right_base_positions,
    )

    try:
        for local_step in range(steps):
            if (
                runtime.simulation_app is not None
                and not runtime.simulation_app.is_running()
            ):
                break
            _raise_if_stopped(should_stop, step=step)
            alpha = (local_step + 1) / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            smooth_rate = (
                (6.0 * alpha * (1.0 - alpha) / float(duration))
                if duration > 0
                else 0.0
            )
            left_targets = _targets_from_smooth_plan(left_plan, smooth, smooth_rate)
            right_targets = _targets_from_smooth_plan(right_plan, smooth, smooth_rate)
            step = _apply_dual_targets_once(
                runtime=runtime,
                left_targets=left_targets,
                right_targets=right_targets,
                phase=phase,
                step=step,
            )
            _update_plan_base(left_plan, left_targets)
            _update_plan_base(right_plan, right_targets)
    finally:
        _zero_side_velocities(runtime.left)
        _zero_side_velocities(runtime.right)
    return step


def execute_dual_command_position_trajectory(
    *,
    runtime: DualRobotRuntime,
    left_trajectory: JointTrajectory | None,
    right_trajectory: JointTrajectory | None,
    step: int,
    phase: str | None = None,
    left_base_positions: np.ndarray | None = None,
    right_base_positions: np.ndarray | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """同步播放左右 command-space 轨迹。"""

    sample_count = _dual_sample_count(left_trajectory, right_trajectory)
    left_base = _side_base_positions(runtime.left, left_base_positions)
    right_base = _side_base_positions(runtime.right, right_base_positions)
    try:
        for sample_index in range(sample_count):
            if (
                runtime.simulation_app is not None
                and not runtime.simulation_app.is_running()
            ):
                break
            _raise_if_stopped(should_stop, step=step)
            left_targets = _trajectory_sample_targets(
                runtime.left, left_trajectory, sample_index, left_base
            )
            right_targets = _trajectory_sample_targets(
                runtime.right, right_trajectory, sample_index, right_base
            )
            sample_phase = phase or _sample_phase(
                left_trajectory, right_trajectory, sample_index
            )
            step = _apply_dual_targets_once(
                runtime=runtime,
                left_targets=left_targets,
                right_targets=right_targets,
                phase=sample_phase,
                step=step,
            )
            if left_targets is not None:
                left_base = left_targets.positions
            if right_targets is not None:
                right_base = right_targets.positions
    finally:
        _zero_side_velocities(runtime.left)
        _zero_side_velocities(runtime.right)
    return step


def _apply_dual_targets_once(
    *,
    runtime: DualRobotRuntime,
    left_targets: ControlTargets | None,
    right_targets: ControlTargets | None,
    phase: str,
    step: int,
) -> int:
    if left_targets is not None:
        runtime.left.joint_controller.apply_targets(
            runtime.articulation_action_type, left_targets
        )
    if right_targets is not None:
        runtime.right.joint_controller.apply_targets(
            runtime.articulation_action_type, right_targets
        )
    runtime.simulation_world.step(render=runtime.render_enabled)
    _write_side_log(runtime.left, left_targets, step, phase, runtime)
    _write_side_log(runtime.right, right_targets, step, phase, runtime)
    return step + 1


def _write_side_log(
    side_runtime: RobotSideRuntime,
    targets: ControlTargets | None,
    step: int,
    phase: str,
    runtime: DualRobotRuntime,
) -> None:
    logger = side_runtime.drive_logger
    if logger is None or targets is None or not logger.should_write(step):
        return
    controller = side_runtime.joint_controller
    indices = np.asarray(controller.driven_indices, dtype=int)
    log_values = logger.collect_step_values(
        side_runtime.articulation, controller, targets, indices
    )
    logger.write(
        step=step,
        time_s=(step + 1) * float(runtime.simulation_world.get_physics_dt()),
        phase=f"{side_runtime.side}_{phase}",
        drive_update=True,
        **log_values,
    )


def _smooth_side_plan(
    side_runtime: RobotSideRuntime,
    *,
    target_command: np.ndarray | None,
    start_command: np.ndarray | None,
    base_positions: np.ndarray | None,
) -> _SmoothSidePlan | None:
    if start_command is None:
        start = _current_command_positions(side_runtime)
    else:
        start = np.asarray(start_command, dtype=float).reshape(-1)
    target = (
        start.copy()
        if target_command is None
        else np.asarray(target_command, dtype=float).reshape(-1)
    )
    if start.size != target.size:
        raise ValueError(
            f"{side_runtime.side} command target shape mismatch: "
            f"start {start.size}, target {target.size}"
        )
    return _SmoothSidePlan(
        controller=side_runtime.joint_controller,
        start=start,
        target=target,
        delta=target - start,
        base=_side_base_positions(side_runtime, base_positions),
    )


def _targets_from_smooth_plan(
    plan: _SmoothSidePlan | None, smooth: float, smooth_rate: float
) -> ControlTargets | None:
    if plan is None:
        return None
    command = plan.start + smooth * plan.delta
    velocity = smooth_rate * plan.delta
    return plan.controller.build_control_targets(
        command_positions=command,
        command_velocities=velocity,
        command_efforts=np.zeros_like(command),
        base_positions=plan.base,
    )


def _update_plan_base(
    plan: _SmoothSidePlan | None, targets: ControlTargets | None
) -> None:
    if plan is not None and targets is not None:
        plan.base = targets.positions


def _current_command_positions(side_runtime: RobotSideRuntime) -> np.ndarray:
    positions = np.asarray(
        side_runtime.articulation.get_joint_positions(), dtype=float
    ).reshape(-1)
    return positions[np.asarray(side_runtime.joint_controller.command_indices, dtype=int)]


def _side_base_positions(
    side_runtime: RobotSideRuntime, base_positions: np.ndarray | None
) -> np.ndarray:
    if base_positions is None:
        return np.asarray(
            side_runtime.articulation.get_joint_positions(), dtype=float
        ).reshape(-1)
    return np.asarray(base_positions, dtype=float).reshape(-1)


def _trajectory_sample_targets(
    side_runtime: RobotSideRuntime,
    trajectory: JointTrajectory | None,
    sample_index: int,
    base_positions: np.ndarray,
) -> ControlTargets | None:
    if trajectory is None or sample_index >= len(trajectory):
        command_positions = base_positions[
            np.asarray(side_runtime.joint_controller.command_indices, dtype=int)
        ]
        return side_runtime.joint_controller.build_control_targets(
            command_positions=command_positions,
            command_velocities=np.zeros_like(command_positions),
            command_efforts=np.zeros_like(command_positions),
            base_positions=base_positions,
        )
    return side_runtime.joint_controller.build_control_targets(
        command_positions=trajectory.positions[sample_index],
        command_velocities=trajectory.velocities[sample_index],
        command_efforts=trajectory.efforts[sample_index],
        base_positions=base_positions,
    )


def _dual_sample_count(
    left_trajectory: JointTrajectory | None, right_trajectory: JointTrajectory | None
) -> int:
    if left_trajectory is None and right_trajectory is None:
        raise ValueError("At least one side trajectory is required")
    return max(
        0 if left_trajectory is None else len(left_trajectory),
        0 if right_trajectory is None else len(right_trajectory),
    )


def _sample_phase(
    left_trajectory: JointTrajectory | None,
    right_trajectory: JointTrajectory | None,
    sample_index: int,
) -> str:
    for trajectory in (left_trajectory, right_trajectory):
        if trajectory is not None and sample_index < len(trajectory):
            return str(trajectory.phases[sample_index])
    return "dual_trajectory"


def _zero_side_velocities(side_runtime: RobotSideRuntime) -> None:
    side_runtime.articulation.set_joint_velocities(
        np.zeros(side_runtime.articulation.num_dof, dtype=float)
    )


def _raise_if_stopped(should_stop: Callable[[], bool] | None, *, step: int) -> None:
    if should_stop is not None and should_stop():
        raise DualCommandExecutionInterrupted(
            "dual command execution interrupted",
            step=step,
        )
