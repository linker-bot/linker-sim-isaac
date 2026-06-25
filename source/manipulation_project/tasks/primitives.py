"""可复用的仿真执行任务原语。

这些任务只负责把已经规划好的完整 DOF 目标或轨迹下发到 Isaac world。它们不做
IK、不读取配置，也不生成新的目标；上层任务可以把它们串起来形成更复杂的流程。

原语约定 ``target_all``/``trajectory.positions`` 已经是完整 articulation DOF 顺序，单位为
rad。每次下发后都会推进一个 physics step，可选 logger 记录 driven_indices 对应的目标与
实际状态，便于后续分析控制误差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from manipulation_project.robots.mimic import MimicFollowerTargetMapper
from manipulation_project.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class TaskRuntime:
    """执行任务原语所需的 Isaac runtime 对象。

    该容器把 world、robot、action 类型和可选 logger/follower mapper 聚合在一起，避免每个
    任务原语函数都携带一长串重复参数。它只保存引用，不负责资源生命周期。
    """

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
        """执行 smoothstep 关节目标移动并返回累计 step。"""

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
        """按轨迹采样时间播放完整 DOF 目标并返回累计 step。"""

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
        """保持固定目标指定时长并返回累计 step。"""

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

    # 每帧都复制目标数组，避免 follower mapper 在原地修正从动关节目标时污染调用方
    # 持有的关键帧、轨迹采样或任务配置缓存。
    command_target_all = np.asarray(target_all, dtype=float).copy()
    if target_velocity_all is None:
        command_velocity_all = np.zeros(robot.num_dof, dtype=float)
    else:
        command_velocity_all = np.asarray(target_velocity_all, dtype=float).copy()
    if follower_mapper is not None:
        # 上层传入的完整目标通常只可靠描述主动关节；mimic follower 需要用当前实际状态
        # 和 MJCF 多项式重新计算，才能避免从动关节在每帧执行时落后于主关节。
        follower_mapper.apply_from_actual(
            command_target_all,
            command_velocity_all,
            np.asarray(robot.get_joint_positions(), dtype=float),
            np.asarray(robot.get_joint_velocities(), dtype=float),
        )
    driven_position = command_target_all[driven_indices]
    driven_velocity = command_velocity_all[driven_indices]
    # 这里只下发 driven_indices 切片，而不是完整 DOF 数组。这样未参与任务的 DOF 不会被
    # action 重置；同时 logger 也按同一索引记录期望/实际值，便于误差列一一对应。
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
    # 用 round(duration / dt) 贴近用户指定时长，并至少执行一帧，保证 0 或很短 duration
    # 仍会把目标下发给控制器。
    steps = max(1, int(round(duration / physics_dt)))
    delta = target_all - start_all
    for local_step in range(steps):
        if simulation_app is not None and not simulation_app.is_running():
            break
        alpha = (local_step + 1) / steps
        # cubic smoothstep 在首尾速度为 0，适合作为简单关节过渡；同时显式计算导数，
        # 让 drive velocity target 与位置曲线一致，减少停止瞬间的抖动。
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
    # 轨迹段结束后清零 articulation 速度，避免下一段任务继承上一段残余速度造成过冲。
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

    # 此原语约定输入是完整 DOF 轨迹；如果传入命令子空间轨迹，应先经控制器扩展，
    # 否则 joint_indices 切片会和数组列数不匹配。
    if trajectory.positions.shape[1] != robot.num_dof:
        raise ValueError(f"Full DOF trajectory expected {robot.num_dof} joints, got {trajectory.positions.shape[1]}")
    physics_dt = float(world.get_physics_dt())
    if physics_dt <= 0:
        raise ValueError("world physics dt must be positive")
    start_time, end_time = trajectory.domain()
    duration = max(0.0, float(end_time - start_time))
    # 按 physics dt 重采样播放，而不是直接遍历原始样本点，保证日志步号、world.step
    # 和控制频率保持一致；原始轨迹内部再通过 ``eval_all`` 做线性插值。
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
    # duration<=0 表示无限保持，常用于 GUI 调试；有 simulation_app 时窗口关闭会退出，
    # 无 app 的测试场景应避免传入非正 duration 以免形成无限循环。
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
