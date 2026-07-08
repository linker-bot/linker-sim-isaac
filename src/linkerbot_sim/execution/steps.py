"""可复用的仿真执行步骤。

这些步骤只负责把已经规划好的目标或轨迹下发到 Isaac world。它们不做 IK、不读取配置，
也不生成新的动作目标；上层动作脚本可以把它们串起来形成更复杂的流程。

本模块同时支持两种轨迹语义：

* 命令子空间位置轨迹：列数等于 controller command space，走 ``build_control_targets``。

传入本模块的 ``JointTrajectory`` 必须已经是“一行对应一个 physics step”的离散执行矩阵，
并且不包含首样本。cuMotion 连续轨迹函数到离散矩阵的采样、去首样本和对齐工作都应在
后端 sampler 完成；execution 只逐行播放，不再调用 ``trajectory.eval_all(...)`` 二次插值。

本模块当前只暴露主动关节位置目标执行步。力控/扭矩控制以后应新增显式命名的 step，
不要复用这些 position step。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.controllers.types import ControlTargets, JointControlSettings
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.trajectories.types import JointTrajectory


class CommandExecutionInterrupted(RuntimeError):
    """Raised when a single-robot command step is stopped by an external request."""

    def __init__(self, message: str, *, step: int | None = None) -> None:
        """保存中断发生时的累计 physics step，便于交互循环续接。"""

        super().__init__(message)
        self.step = step


@dataclass(frozen=True)
class SmoothCommandPositionTargetStep:
    """用 smoothstep 在两个 command-space 位置目标之间移动。

    command-space 只包含主动关节，不包含 mimic follower。每一帧都会交给
    ``JointController.build_control_targets``，由 controller 根据实际 master 状态补出
    follower 目标。
    """

    start_command: np.ndarray
    target_command: np.ndarray
    duration: float
    phase: str
    base_positions: np.ndarray | None = None
    should_stop: Callable[[], bool] | None = None

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """执行 command-space position smoothstep 并返回累计 step。"""

        return execute_smooth_command_position_target(
            articulation=runtime.articulation,
            simulation_world=runtime.simulation_world,
            articulation_action_type=runtime.articulation_action_type,
            joint_controller=runtime.joint_controller,
            start_command=self.start_command,
            target_command=self.target_command,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render_enabled=runtime.render_enabled,
            step=step,
            base_positions=self.base_positions,
            should_stop=self.should_stop,
            drive_logger=runtime.drive_logger,
            state_observer=runtime.state_observer,
            camera_observer=runtime.camera_observer,
        )


@dataclass(frozen=True)
class CommandPositionTrajectoryStep:
    """按采样行播放 command-space 位置轨迹。"""

    trajectory: JointTrajectory
    should_stop: Callable[[], bool] | None = None

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """逐样本播放 command-space position 轨迹并返回累计 step。"""

        return execute_command_position_trajectory(
            articulation=runtime.articulation,
            simulation_world=runtime.simulation_world,
            articulation_action_type=runtime.articulation_action_type,
            joint_controller=runtime.joint_controller,
            trajectory=self.trajectory,
            simulation_app=runtime.simulation_app,
            render_enabled=runtime.render_enabled,
            step=step,
            should_stop=self.should_stop,
            drive_logger=runtime.drive_logger,
            state_observer=runtime.state_observer,
            camera_observer=runtime.camera_observer,
        )


@dataclass(frozen=True)
class HoldCommandPositionTargetStep:
    """持续下发同一个 command-space 位置目标，维持主动关节姿态。"""

    target_command: np.ndarray
    duration: float
    phase: str
    base_positions: np.ndarray | None = None
    should_stop: Callable[[], bool] | None = None

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """保持 command-space position 目标指定时长并返回累计 step。"""

        return execute_command_position_hold(
            articulation=runtime.articulation,
            simulation_world=runtime.simulation_world,
            articulation_action_type=runtime.articulation_action_type,
            joint_controller=runtime.joint_controller,
            target_command=self.target_command,
            duration=self.duration,
            phase=self.phase,
            simulation_app=runtime.simulation_app,
            render_enabled=runtime.render_enabled,
            step=step,
            base_positions=self.base_positions,
            should_stop=self.should_stop,
            drive_logger=runtime.drive_logger,
            state_observer=runtime.state_observer,
            camera_observer=runtime.camera_observer,
        )


@dataclass(frozen=True)
class SwitchControlModeStep:
    """在动作序列中切换 runtime 关节控制配置。"""

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
    state_observer=None,
    camera_observer=None,
) -> int:
    """下发一帧已经构造好的控制目标，并推进一个 physics step。

    ``targets`` 已经是 controller 认可的完整控制目标，内部包含主动关节、mimic follower
    以及 position/velocity/effort 三类 action 字段。本函数不再关心这些目标来自完整 DOF
    轨迹还是 command-space 轨迹，只负责“apply action -> world.step -> 可选日志”这一帧
    物理推进流程。
    """

    joint_controller.apply_targets(articulation_action_type, targets)
    simulation_world.step(render=render_enabled)
    _observe_state(state_observer, simulation_world, step=step, phase=phase)
    _observe_cameras(camera_observer, simulation_world, step=step, phase=phase)
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
    state_observer=None,
    camera_observer=None,
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
        state_observer=state_observer,
        camera_observer=camera_observer,
    )
    return next_step, targets


def execute_smooth_command_position_target(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    start_command: np.ndarray,
    target_command: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render_enabled: bool,
    step: int,
    base_positions: np.ndarray | None = None,
    should_stop: Callable[[], bool] | None = None,
    drive_logger=None,
    state_observer=None,
    camera_observer=None,
) -> int:
    """用 smoothstep 在两个 command-space 位置目标之间平滑移动。

    本函数只处理主动命令关节。mimic follower 不作为入参暴露，而是在每一帧进入
    ``JointController.build_control_targets`` 时根据实际 master 状态重算。
    """

    physics_dt = float(simulation_world.get_physics_dt())
    steps = max(1, int(round(duration / physics_dt)))
    start = np.asarray(start_command, dtype=float).reshape(-1)
    target = np.asarray(target_command, dtype=float).reshape(-1)
    delta = target - start
    full_position = (
        np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
        if base_positions is None
        else np.asarray(base_positions, dtype=float).reshape(-1)
    )
    for local_step in range(steps):
        if simulation_app is not None and not simulation_app.is_running():
            break
        _raise_if_stopped(should_stop, step=step)
        alpha = (local_step + 1) / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        smooth_rate = (6.0 * alpha * (1.0 - alpha) / duration) if duration > 0 else 0.0
        command = start + smooth * delta
        velocity = smooth_rate * delta
        step, targets = _apply_command_joint_target_once(
            articulation=articulation,
            simulation_world=simulation_world,
            articulation_action_type=articulation_action_type,
            joint_controller=joint_controller,
            command_positions=command,
            command_velocities=velocity,
            command_efforts=np.zeros_like(command),
            base_positions=full_position,
            render_enabled=render_enabled,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
            state_observer=state_observer,
            camera_observer=camera_observer,
        )
        full_position = targets.positions
    articulation.set_joint_velocities(np.zeros(articulation.num_dof, dtype=float))
    return step


def execute_command_position_trajectory(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    trajectory: JointTrajectory,
    simulation_app,
    render_enabled: bool,
    step: int = 0,
    should_stop: Callable[[], bool] | None = None,
    drive_logger=None,
    state_observer=None,
    camera_observer=None,
    hold: bool = False,
) -> int:
    """按采样点播放命令子空间位置轨迹。

    轨迹列按 controller command space 排列，而不是完整 articulation DOF。每一帧都会通过
    ``build_control_targets`` 扩展成完整 DOF 控制目标，再复用私有单帧下发逻辑
    下发和记录日志。
    """

    full_position = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
    targets: ControlTargets | None = None
    for sample_index in range(len(trajectory)):
        if simulation_app is not None and not simulation_app.is_running():
            break
        _raise_if_stopped(should_stop, step=step)
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
            state_observer=state_observer,
            camera_observer=camera_observer,
        )
        full_position = targets.positions
    if hold and targets is not None:
        while simulation_app is None or simulation_app.is_running():
            _raise_if_stopped(should_stop, step=step)
            joint_controller.apply_targets(articulation_action_type, targets)
            simulation_world.step(render=render_enabled)
            _observe_state(
                state_observer,
                simulation_world,
                step=step,
                phase=trajectory.phases[-1],
            )
            _observe_cameras(
                camera_observer,
                simulation_world,
                step=step,
                phase=trajectory.phases[-1],
            )
            step += 1
            if simulation_app is None:
                break
    return step


def execute_command_position_hold(
    *,
    articulation,
    simulation_world,
    articulation_action_type,
    joint_controller,
    target_command: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render_enabled: bool,
    step: int,
    base_positions: np.ndarray | None = None,
    should_stop: Callable[[], bool] | None = None,
    drive_logger=None,
    state_observer=None,
    camera_observer=None,
) -> int:
    """保持一个 command-space 位置目标一段时间。

    只下发主动关节目标；follower 每帧由 controller 根据实际 master 状态覆盖。
    """

    physics_dt = float(simulation_world.get_physics_dt())
    total_steps = max(1, int(round(duration / physics_dt))) if duration > 0 else None
    full_position = (
        np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
        if base_positions is None
        else np.asarray(base_positions, dtype=float).reshape(-1)
    )
    local_step = 0
    target = np.asarray(target_command, dtype=float).reshape(-1)
    zero_velocity = np.zeros_like(target)
    zero_effort = np.zeros_like(target)
    while total_steps is None or local_step < total_steps:
        if simulation_app is not None and not simulation_app.is_running():
            break
        _raise_if_stopped(should_stop, step=step)
        step, targets = _apply_command_joint_target_once(
            articulation=articulation,
            simulation_world=simulation_world,
            articulation_action_type=articulation_action_type,
            joint_controller=joint_controller,
            command_positions=target,
            command_velocities=zero_velocity,
            command_efforts=zero_effort,
            base_positions=full_position,
            render_enabled=render_enabled,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
            state_observer=state_observer,
            camera_observer=camera_observer,
        )
        full_position = targets.positions
        local_step += 1
    return step


def _observe_state(state_observer, simulation_world, *, step: int, phase: str) -> None:
    """执行一帧后的可选状态采样；observer 自身负责频率和输出策略。"""

    if state_observer is None:
        return
    observe = getattr(state_observer, "observe", None)
    if observe is None:
        return
    observe(simulation_world, step=step, phase=phase)


def _observe_cameras(camera_observer, simulation_world, *, step: int, phase: str) -> None:
    """执行一帧后的可选 camera 采样；observer 自身负责频率和输出策略。"""

    if camera_observer is None:
        return
    observe = getattr(camera_observer, "observe", None)
    if observe is None:
        return
    observe(simulation_world, step=step, phase=phase)


def _raise_if_stopped(
    should_stop: Callable[[], bool] | None,
    *,
    step: int,
) -> None:
    """把外部 cancel/estop/quit 请求转成带 step 的中断异常。"""

    if should_stop is not None and should_stop():
        raise CommandExecutionInterrupted(
            "single command execution interrupted",
            step=step,
        )
