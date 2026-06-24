"""可复用的仿真执行任务原语。

这些任务只负责把已经规划好的完整 DOF 目标或轨迹下发到 Isaac world。它们不做
IK、不读取配置，也不生成新的目标；上层任务可以把它们串起来形成更复杂的流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from manipulation_project.robots.mimic import MimicFollowerTargetMapper
from manipulation_project.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class TaskRuntime:
    """执行任务原语所需的 Isaac runtime 对象。"""

    robot: object
    world: object
    articulation_action_type: object
    driven_indices: np.ndarray
    simulation_app: object | None
    render: bool
    drive_logger: object | None = None
    follower_mapper: MimicFollowerTargetMapper | None = None


class ExecutableTask(Protocol):
    """可顺序执行的任务原语协议。"""

    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        """执行任务并返回新的全局 step。"""


@dataclass(frozen=True)
class MoveJointTargetTask:
    """用 smoothstep 在两个完整 DOF 目标之间移动。"""

    start_all: np.ndarray
    target_all: np.ndarray
    duration: float
    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        return move_joint_target(
            robot=runtime.robot,
            world=runtime.world,
            articulation_action_type=runtime.articulation_action_type,
            driven_indices=runtime.driven_indices,
            start_all=self.start_all,
            target_all=self.target_all,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render=runtime.render,
            step=step,
            drive_logger=runtime.drive_logger,
            follower_mapper=runtime.follower_mapper,
        )


@dataclass(frozen=True)
class MoveFullJointTrajectoryTask:
    """按仿真时钟播放完整 DOF 关节轨迹。"""

    trajectory: JointTrajectory
    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        return move_full_joint_trajectory(
            robot=runtime.robot,
            world=runtime.world,
            articulation_action_type=runtime.articulation_action_type,
            driven_indices=runtime.driven_indices,
            trajectory=self.trajectory,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render=runtime.render,
            step=step,
            drive_logger=runtime.drive_logger,
            follower_mapper=runtime.follower_mapper,
        )


@dataclass(frozen=True)
class HoldTask:
    """持续下发同一个完整 DOF 目标，维持姿态。"""

    target_all: np.ndarray
    duration: float
    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        return hold_joint_target(
            robot=runtime.robot,
            world=runtime.world,
            articulation_action_type=runtime.articulation_action_type,
            driven_indices=runtime.driven_indices,
            target_all=self.target_all,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render=runtime.render,
            step=step,
            drive_logger=runtime.drive_logger,
            follower_mapper=runtime.follower_mapper,
        )


def step_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    driven_indices: np.ndarray,
    target_all: np.ndarray,
    render: bool,
    step: int,
    phase: str,
    target_velocity_all: np.ndarray | None = None,
    drive_logger=None,
    follower_mapper: MimicFollowerTargetMapper | None = None,
) -> int:
    """下发一帧完整 DOF 目标，并推进一个 physics step。"""

    command_target_all = np.asarray(target_all, dtype=float).copy()
    if target_velocity_all is None:
        command_velocity_all = np.zeros(robot.num_dof, dtype=float)
    else:
        command_velocity_all = np.asarray(target_velocity_all, dtype=float).copy()
    if follower_mapper is not None:
        follower_mapper.apply_from_actual(
            command_target_all,
            command_velocity_all,
            np.asarray(robot.get_joint_positions(), dtype=float),
            np.asarray(robot.get_joint_velocities(), dtype=float),
        )
    driven_position = command_target_all[driven_indices]
    driven_velocity = command_velocity_all[driven_indices]
    robot.apply_action(
        articulation_action_type(
            joint_positions=driven_position,
            joint_velocities=driven_velocity,
            joint_indices=driven_indices,
        )
    )
    world.step(render=render)
    if drive_logger is not None:
        actual_position = np.asarray(robot.get_joint_positions(), dtype=float)[driven_indices]
        actual_velocity = np.asarray(robot.get_joint_velocities(), dtype=float)[driven_indices]
        drive_logger.write(
            step=step,
            time_s=(step + 1) * float(world.get_physics_dt()),
            phase=phase,
            drive_update=True,
            desired_position=driven_position,
            actual_position=actual_position,
            desired_velocity=driven_velocity,
            actual_velocity=actual_velocity,
        )
    return step + 1


def move_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    driven_indices: np.ndarray,
    start_all: np.ndarray,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
    follower_mapper: MimicFollowerTargetMapper | None = None,
) -> int:
    """用 smoothstep 在两个完整 DOF 目标之间平滑移动。"""

    physics_dt = float(world.get_physics_dt())
    steps = max(1, int(round(duration / physics_dt)))
    delta = target_all - start_all
    for local_step in range(steps):
        if simulation_app is not None and not simulation_app.is_running():
            break
        alpha = (local_step + 1) / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        smooth_rate = (6.0 * alpha * (1.0 - alpha) / duration) if duration > 0 else 0.0
        command = start_all + smooth * delta
        velocity = smooth_rate * delta
        step = step_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            target_all=command,
            target_velocity_all=velocity,
            render=render,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))
    return step


def move_full_joint_trajectory(
    *,
    robot,
    world,
    articulation_action_type,
    driven_indices: np.ndarray,
    trajectory: JointTrajectory,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
    follower_mapper: MimicFollowerTargetMapper | None = None,
) -> int:
    """按 physics dt 播放完整 DOF 关节轨迹。"""

    if trajectory.positions.shape[1] != robot.num_dof:
        raise ValueError(f"Full DOF trajectory expected {robot.num_dof} joints, got {trajectory.positions.shape[1]}")
    physics_dt = float(world.get_physics_dt())
    if physics_dt <= 0:
        raise ValueError("world physics dt must be positive")
    start_time, end_time = trajectory.domain()
    duration = max(0.0, float(end_time - start_time))
    steps = max(1, int(round(duration / physics_dt))) if duration > 0.0 else 1
    for local_step in range(steps):
        if simulation_app is not None and not simulation_app.is_running():
            break
        alpha = (local_step + 1) / steps
        time_s = start_time + alpha * duration
        sample = trajectory.eval_all(time_s)
        step = step_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            target_all=sample.position,
            target_velocity_all=sample.velocity,
            render=render,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))
    return step


def hold_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    driven_indices: np.ndarray,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
    follower_mapper: MimicFollowerTargetMapper | None = None,
) -> int:
    """保持一个完整 DOF 目标一段时间。"""

    physics_dt = float(world.get_physics_dt())
    total_steps = max(1, int(round(duration / physics_dt))) if duration > 0 else None
    local_step = 0
    while total_steps is None or local_step < total_steps:
        if simulation_app is not None and not simulation_app.is_running():
            break
        step = step_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            target_all=target_all,
            render=render,
            step=step,
            phase=phase,
            target_velocity_all=np.zeros(robot.num_dof, dtype=float),
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        local_step += 1
    return step
