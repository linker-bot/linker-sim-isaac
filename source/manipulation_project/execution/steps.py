"""可复用的仿真执行步骤。

这些步骤只负责把已经规划好的目标或轨迹下发到 Isaac world。它们不做 IK、不读取配置，
也不生成新的任务目标；上层任务可以把它们串起来形成更复杂的流程。

本模块同时支持两种轨迹语义：

* 完整 DOF 轨迹：列数等于 articulation ``num_dof``，直接走 ``targets_from_full_state``。
* 命令子空间轨迹：列数等于 controller command space，走 ``build_control_targets``。

两者共享同一套私有单帧下发逻辑，避免执行层出现两套几乎相同但行为略有差异的循环。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipulation_project.controllers.types import ControlTargets, JointControlSettings
from manipulation_project.execution.runtime import ExecutionRuntime
from manipulation_project.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class SmoothJointTargetStep:
    """用 smoothstep 在两个完整 DOF 目标之间移动。"""

    start_all: np.ndarray
    target_all: np.ndarray
    duration: float
    phase: str

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """执行 smoothstep 关节目标移动并返回累计 step。"""

        return execute_smooth_joint_target(
            articulation=runtime.articulation,
            simulation_world=runtime.simulation_world,
            articulation_action_type=runtime.articulation_action_type,
            joint_controller=runtime.joint_controller,
            start_all=self.start_all,
            target_all=self.target_all,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render_enabled=runtime.render_enabled,
            step=step,
            drive_logger=runtime.drive_logger,
        )


@dataclass(frozen=True)
class FullJointTrajectoryStep:
    """按仿真时钟播放完整 DOF 关节轨迹。"""

    trajectory: JointTrajectory
    phase: str

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """按轨迹采样时间播放完整 DOF 目标并返回累计 step。"""

        return execute_full_joint_trajectory(
            articulation=runtime.articulation,
            simulation_world=runtime.simulation_world,
            articulation_action_type=runtime.articulation_action_type,
            joint_controller=runtime.joint_controller,
            trajectory=self.trajectory,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render_enabled=runtime.render_enabled,
            step=step,
            drive_logger=runtime.drive_logger,
        )


@dataclass(frozen=True)
class HoldJointTargetStep:
    """持续下发同一个完整 DOF 目标，维持姿态。"""

    target_all: np.ndarray
    duration: float
    phase: str

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """保持固定目标指定时长并返回累计 step。"""

        return execute_joint_hold(
            articulation=runtime.articulation,
            simulation_world=runtime.simulation_world,
            articulation_action_type=runtime.articulation_action_type,
            joint_controller=runtime.joint_controller,
            target_all=self.target_all,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render_enabled=runtime.render_enabled,
            step=step,
            drive_logger=runtime.drive_logger,
        )


@dataclass(frozen=True)
class SwitchControlModeStep:
    """在任务序列中切换 runtime 关节控制配置。"""

    settings: JointControlSettings
    phase: str = "switch_control_mode"

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """切换控制器模式并返回未改变的累计 step。"""

        runtime.joint_controller.settings = self.settings
        runtime.joint_controller.configure_runtime()
        return step


def _apply_control_targets_once(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    targets: ControlTargets,
    render_enabled: bool,
    step: int,
    phase: str,
    drive_logger=None,
    log_indices: np.ndarray | None = None,
) -> int:
    """下发一帧已经构造好的控制目标，并推进一个 physics step。

    ``targets`` 已经是 controller 认可的完整控制目标，内部包含主动关节、mimic follower
    以及 position/velocity/effort 三类 action 字段。本函数不再关心这些目标来自完整 DOF
    轨迹还是 command-space 轨迹，只负责“apply action -> world.step -> 可选日志”这一帧
    物理推进流程。
    """

    joint_controller.apply_targets(articulation_action_type, targets)
    simulation_world.step(render=render_enabled)
    if drive_logger is not None and drive_logger.should_write(step):
        indices = (
            np.asarray(joint_controller.driven_indices, dtype=int)
            if log_indices is None
            else np.asarray(log_indices, dtype=int).reshape(-1)
        )
        log_values = drive_logger.collect_step_values(
            articulation, joint_controller, targets, indices
        )
        drive_logger.write(
            step=step,
            time_s=(step + 1) * float(simulation_world.get_physics_dt()),
            phase=phase,
            drive_update=True,
            **log_values,
        )
    return step + 1


def _apply_full_joint_target_once(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    target_all: np.ndarray,
    render_enabled: bool,
    step: int,
    phase: str,
    target_velocity_all: np.ndarray | None = None,
    target_effort_all: np.ndarray | None = None,
    drive_logger=None,
) -> int:
    """从完整 articulation DOF 目标构造控制目标，并下发一帧。

    ``target_all`` 的长度必须等于 Isaac articulation ``num_dof``，列顺序就是
    ``articulation.dof_names``。controller 会从完整目标中抽取主动关节命令，并根据 master
    的实际状态补出 mimic follower 目标。
    """

    targets = joint_controller.targets_from_full_state(
        target_all,
        joint_velocities=target_velocity_all,
        joint_efforts=target_effort_all,
    )
    return _apply_control_targets_once(
        articulation=articulation,
        simulation_world=simulation_world,
        articulation_action_type=articulation_action_type,
        joint_controller=joint_controller,
        targets=targets,
        render_enabled=render_enabled,
        step=step,
        phase=phase,
        drive_logger=drive_logger,
    )


def _apply_command_joint_target_once(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    command_positions: np.ndarray | None,
    command_velocities: np.ndarray | None,
    command_efforts: np.ndarray | None,
    base_positions: np.ndarray | None,
    render_enabled: bool,
    step: int,
    phase: str,
    drive_logger=None,
) -> tuple[int, ControlTargets]:
    """从 controller command-space 目标构造控制目标，并下发一帧。

    command-space 只包含主动命令关节，不包含 mimic follower，也不一定覆盖完整 articulation。
    因此需要传入 ``base_positions`` 作为其它 DOF 的参考姿态，再由 controller 统一展开成
    完整 action。
    """

    targets = joint_controller.build_control_targets(
        command_positions=command_positions,
        command_velocities=command_velocities,
        command_efforts=command_efforts,
        base_positions=base_positions,
    )
    next_step = _apply_control_targets_once(
        articulation=articulation,
        simulation_world=simulation_world,
        articulation_action_type=articulation_action_type,
        joint_controller=joint_controller,
        targets=targets,
        render_enabled=render_enabled,
        step=step,
        phase=phase,
        drive_logger=drive_logger,
    )
    return next_step, targets


def execute_smooth_joint_target(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    start_all: np.ndarray,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render_enabled: bool,
    step: int,
    drive_logger=None,
) -> int:
    """用 smoothstep 在两个完整 DOF 目标之间平滑移动。"""

    physics_dt = float(simulation_world.get_physics_dt())
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
        step = _apply_full_joint_target_once(
            articulation=articulation,
            simulation_world=simulation_world,
            articulation_action_type=articulation_action_type,
            joint_controller=joint_controller,
            target_all=command,
            target_velocity_all=velocity,
            render_enabled=render_enabled,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
        )
    articulation.set_joint_velocities(np.zeros(articulation.num_dof, dtype=float))
    return step


def execute_full_joint_trajectory(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    trajectory: JointTrajectory,
    phase: str,
    simulation_app,
    render_enabled: bool,
    step: int,
    drive_logger=None,
) -> int:
    """按 physics dt 播放完整 DOF 关节轨迹。

    轨迹采样频率可以和物理步频不同：执行层以 physics dt 为时钟，在每个物理步调用
    ``trajectory.eval_all(time_s)`` 插值得到当前位置/速度/effort。这样 cuMotion 可以按自身
    时间域输出轨迹，Isaac 仍按固定物理步稳定推进。
    """

    if trajectory.positions.shape[1] != articulation.num_dof:
        raise ValueError(
            "Full DOF trajectory expected "
            f"{articulation.num_dof} joints, got {trajectory.positions.shape[1]}"
        )
    physics_dt = float(simulation_world.get_physics_dt())
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
        step = _apply_full_joint_target_once(
            articulation=articulation,
            simulation_world=simulation_world,
            articulation_action_type=articulation_action_type,
            joint_controller=joint_controller,
            target_all=sample.position,
            target_velocity_all=sample.velocity,
            target_effort_all=sample.effort,
            render_enabled=render_enabled,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
        )
    articulation.set_joint_velocities(np.zeros(articulation.num_dof, dtype=float))
    return step


def execute_command_joint_trajectory(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    trajectory: JointTrajectory,
    simulation_app,
    render_enabled: bool,
    step: int = 0,
    drive_logger=None,
    hold: bool = False,
) -> int:
    """按采样点播放命令子空间关节轨迹。

    轨迹列按 controller command space 排列，而不是完整 articulation DOF。每一帧都会通过
    ``build_control_targets`` 扩展成完整 DOF 控制目标，再复用私有单帧下发逻辑
    下发和记录日志。
    """

    full_position = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
    targets: ControlTargets | None = None
    for sample_index in range(len(trajectory)):
        if simulation_app is not None and not simulation_app.is_running():
            break
        step, targets = _apply_command_joint_target_once(
            articulation=articulation,
            simulation_world=simulation_world,
            articulation_action_type=articulation_action_type,
            joint_controller=joint_controller,
            command_positions=trajectory.positions[sample_index],
            command_velocities=trajectory.velocities[sample_index],
            command_efforts=trajectory.efforts[sample_index],
            base_positions=full_position,
            render_enabled=render_enabled,
            step=step,
            phase=trajectory.phases[sample_index],
            drive_logger=drive_logger,
        )
        full_position = targets.positions
    if hold and targets is not None:
        while simulation_app is None or simulation_app.is_running():
            joint_controller.apply_targets(articulation_action_type, targets)
            simulation_world.step(render=render_enabled)
            if simulation_app is None:
                break
    return step


def execute_joint_hold(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render_enabled: bool,
    step: int,
    drive_logger=None,
) -> int:
    """保持一个完整 DOF 目标一段时间。"""

    physics_dt = float(simulation_world.get_physics_dt())
    total_steps = max(1, int(round(duration / physics_dt))) if duration > 0 else None
    local_step = 0
    while total_steps is None or local_step < total_steps:
        if simulation_app is not None and not simulation_app.is_running():
            break
        step = _apply_full_joint_target_once(
            articulation=articulation,
            simulation_world=simulation_world,
            articulation_action_type=articulation_action_type,
            joint_controller=joint_controller,
            target_all=target_all,
            render_enabled=render_enabled,
            step=step,
            phase=phase,
            target_velocity_all=np.zeros(articulation.num_dof, dtype=float),
            drive_logger=drive_logger,
        )
        local_step += 1
    return step
