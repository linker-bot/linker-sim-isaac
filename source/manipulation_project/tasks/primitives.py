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

from manipulation_project.controllers.types import JointControlSettings
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
    controller: object
    simulation_app: object | None
    render: bool
    drive_logger: object | None = None


class ExecutableTask(Protocol):
    """可顺序执行的任务原语协议。

    所有实现都接收同一个 ``TaskRuntime`` 和全局 step，并返回更新后的 step。协议不要求
    实现一定推进 world；例如控制模式切换任务只改变 runtime 配置并原样返回 step。
    """

    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        """执行任务并返回新的全局 step。

        具体实现可以推进仿真、写日志或只更新 runtime 配置；返回值始终是下一段任务继续
        计数的起点。
        """


@dataclass(frozen=True)
class MoveJointTargetTask:
    """用 smoothstep 在两个完整 DOF 目标之间移动。

    ``start_all`` 和 ``target_all`` 必须是 Isaac articulation 完整 DOF 顺序；controller 会在
    下发时裁剪主动命令关节并补齐 mimic follower。
    """

    start_all: np.ndarray
    target_all: np.ndarray
    duration: float
    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        """执行 smoothstep 关节目标移动并返回累计 step。

        每个 physics step 都会生成完整 DOF 位置/速度目标，并通过 controller 处理主动关节和
        mimic follower 的下发。
        """

        return move_joint_target(
            robot=runtime.robot,
            world=runtime.world,
            articulation_action_type=runtime.articulation_action_type,
            controller=runtime.controller,
            start_all=self.start_all,
            target_all=self.target_all,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render=runtime.render,
            step=step,
            drive_logger=runtime.drive_logger,
        )


@dataclass(frozen=True)
class MoveFullJointTrajectoryTask:
    """按仿真时钟播放完整 DOF 关节轨迹。

    轨迹列必须覆盖完整 articulation DOF，而不是 controller command space 子集。播放时会按
    ``world.get_physics_dt()`` 重采样，原始采样点不必与物理步长完全一致。
    """

    trajectory: JointTrajectory
    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        """按轨迹采样时间播放完整 DOF 目标并返回累计 step。

        轨迹会按 ``world.get_physics_dt()`` 重采样播放；返回值包含实际推进的仿真步数。
        """

        return move_full_joint_trajectory(
            robot=runtime.robot,
            world=runtime.world,
            articulation_action_type=runtime.articulation_action_type,
            controller=runtime.controller,
            trajectory=self.trajectory,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render=runtime.render,
            step=step,
            drive_logger=runtime.drive_logger,
        )


@dataclass(frozen=True)
class HoldTask:
    """持续下发同一个完整 DOF 目标，维持姿态。

    ``duration <= 0`` 表示无限保持，通常只适合带 GUI/app 生命周期的调试运行。
    """

    target_all: np.ndarray
    duration: float
    phase: str

    def run(self, runtime: TaskRuntime, step: int) -> int:
        """保持固定目标指定时长并返回累计 step。

        目标速度会置零并持续下发；当 ``duration <= 0`` 时依赖 ``simulation_app`` 生命周期退出。
        """

        return hold_joint_target(
            robot=runtime.robot,
            world=runtime.world,
            articulation_action_type=runtime.articulation_action_type,
            controller=runtime.controller,
            target_all=self.target_all,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render=runtime.render,
            step=step,
            drive_logger=runtime.drive_logger,
        )


@dataclass(frozen=True)
class SwitchControlModeTask:
    """在任务序列中切换 runtime 关节控制配置。

    该任务只更新控制器的 ``settings`` 并重新写入 articulation runtime 参数，不推进
    physics step。这样它可以作为阶段边界事件插在两个运动任务之间，下一帧目标会直接
    使用新的 position/velocity/effort 控制方式下发。
    """

    settings: JointControlSettings
    phase: str = "switch_control_mode"

    def run(self, runtime: TaskRuntime, step: int) -> int:
        """切换控制器模式并返回未改变的累计 step。

        该任务不推进 world，只把新的 ``JointControlSettings`` 写入 controller 并重新配置
        articulation runtime。
        """

        runtime.controller.settings = self.settings
        runtime.controller.configure_runtime()
        return step


def step_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    controller,
    target_all: np.ndarray,
    render: bool,
    step: int,
    phase: str,
    target_velocity_all: np.ndarray | None = None,
    target_effort_all: np.ndarray | None = None,
    drive_logger=None,
) -> int:
    """下发一帧完整 DOF 目标，并推进一个 physics step。

    ``target_all``、可选速度和 effort 都必须是完整 DOF 数组。controller 会负责把完整目标
    转换成当前控制模式需要的 Isaac action，并在日志打开时只记录 driven joints。
    """

    # 完整目标进入 controller 后会被分解成主动关节 action 和 follower position drive action。
    # 这样 position/velocity/effort 三种主动控制都能复用同一套 mimic follower 规则。
    targets = controller.targets_from_full_state(
        target_all,
        joint_velocities=target_velocity_all,
        joint_efforts=target_effort_all,
    )
    controller.apply_targets(articulation_action_type, targets)
    world.step(render=render)
    if drive_logger is not None:
        driven_indices = controller.driven_indices
        if drive_logger.should_write(step):
            log_values = drive_logger.collect_step_values(
                robot, controller, targets, driven_indices
            )
            drive_logger.write(
                step=step,
                time_s=(step + 1) * float(world.get_physics_dt()),
                phase=phase,
                drive_update=True,
                **log_values,
            )
    return step + 1


def move_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    controller,
    start_all: np.ndarray,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
) -> int:
    """用 smoothstep 在两个完整 DOF 目标之间平滑移动。

    这是任务层的简单插值 fallback，适合手部开合或不需要后端规划的完整 DOF 过渡。机械臂
    避障或限速轨迹应优先由规划后端生成 ``JointTrajectory`` 后交给
    ``move_full_joint_trajectory`` 播放。
    """

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
            controller=controller,
            target_all=command,
            target_velocity_all=velocity,
            render=render,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
        )
    # 轨迹段结束后清零 articulation 速度，避免下一段任务继承上一段残余速度造成过冲。
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))
    return step


def move_full_joint_trajectory(
    *,
    robot,
    world,
    articulation_action_type,
    controller,
    trajectory: JointTrajectory,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
) -> int:
    """按 physics dt 播放完整 DOF 关节轨迹。

    输入轨迹必须已经完成后端关节顺序到完整 DOF 顺序的映射。函数会在每个 physics step
    调用 ``trajectory.eval_all``，同时下发位置和速度目标。
    """

    # 此原语约定输入是完整 DOF 轨迹；如果传入命令子空间轨迹，应先经控制器扩展，
    # 否则 joint_indices 切片会和数组列数不匹配。
    if trajectory.positions.shape[1] != robot.num_dof:
        raise ValueError(
            f"Full DOF trajectory expected {robot.num_dof} joints, got {trajectory.positions.shape[1]}"
        )
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
            controller=controller,
            target_all=sample.position,
            target_velocity_all=sample.velocity,
            render=render,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
        )
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))
    return step


def hold_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    controller,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
) -> int:
    """保持一个完整 DOF 目标一段时间。

    该函数每步重新下发相同位置并把目标速度置零。它不检查目标是否已经到达，只按时长或
    app 生命周期决定何时退出。
    """

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
            controller=controller,
            target_all=target_all,
            render=render,
            step=step,
            phase=phase,
            target_velocity_all=np.zeros(robot.num_dof, dtype=float),
            drive_logger=drive_logger,
        )
        local_step += 1
    return step
