"""关节目标任务辅助函数。

该模块负责把 YAML 中的稀疏关节目标转换成控制器可执行的采样轨迹，并在 Isaac
仿真循环中逐点下发，同时记录目标/实际关节跟踪日志。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manipulation_project.controllers.implicit_drive_controller import ImplicitDriveController
from manipulation_project.logging.joint_logger import JointTrackingLogger
from manipulation_project.robots.joint_groups import target_vector_from_mapping
from manipulation_project.trajectories.joint_trajectory import build_joint_target_trajectory


@dataclass(frozen=True)
class MoveJointTargetsConfig:
    """固定时长关节目标移动的配置。

    输入字段:
        targets: 稀疏的 ``关节名 -> 目标位置(rad)`` 映射。
        duration_s: 从当前姿态移动到目标姿态的持续时间，单位秒。
        interpolation: 插值函数名称，例如 ``smoothstep``。
        sample_hz: 采样频率，单位 Hz，用于生成离散轨迹点。
        phase: 写入日志的阶段名。
    输出:
        dataclass 本身不执行计算，作为构建轨迹时的输入参数集合。
    """

    targets: dict[str, float]
    duration_s: float
    interpolation: str = "smoothstep"
    sample_hz: float = 200.0
    phase: str = "joint_target"


def build_command_trajectory_from_sparse_targets(
    *,
    dof_names: list[str],
    command_indices: np.ndarray,
    current_positions: np.ndarray,
    config: MoveJointTargetsConfig,
):
    """从稀疏关节目标构建命令空间轨迹。

    参数:
        dof_names: articulation 的完整 DOF 名称列表，顺序必须与 Isaac 数组一致。
        command_indices: 控制器实际接收命令的 DOF 索引。
        current_positions: 当前完整 DOF 位置数组，单位 rad。
        config: 关节目标任务配置。
    返回:
        ``JointTrajectory``，其 ``joint_names`` 和 ``joint_positions`` 只包含
        ``command_indices`` 对应的命令关节。
    """

    full_target = target_vector_from_mapping(dof_names, config.targets, base=current_positions)
    start = np.asarray(current_positions, dtype=float).reshape(-1)[command_indices]
    target = full_target[command_indices]
    command_names = [dof_names[int(index)] for index in command_indices]
    return build_joint_target_trajectory(
        start,
        target,
        joint_names=command_names,
        duration_s=config.duration_s,
        sample_dt=1.0 / config.sample_hz,
        interpolation=config.interpolation,
        phase=config.phase,
    )


def execute_joint_target_trajectory(
    *,
    robot,
    world,
    articulation_action_type,
    controller: ImplicitDriveController,
    trajectory,
    log_path: str | Path | None,
    render: bool,
    simulation_app=None,
    hold: bool = False,
) -> None:
    """在 Isaac 中执行采样后的关节命令轨迹。

    参数:
        robot: Isaac articulation 对象，需提供关节读写和 ``num_dof``。
        world: Isaac world，用于获取 physics dt 并推进仿真。
        articulation_action_type: Isaac 的 action 类型构造器。
        controller: ``ImplicitDriveController``，负责把命令子空间扩展成完整 DOF 目标。
        trajectory: ``JointTrajectory`` 或可迭代轨迹点序列。
        log_path: CSV 日志路径；为 ``None`` 时禁用写文件。
        render: 是否在 ``world.step`` 时渲染。
        simulation_app: 可选 Isaac app，用于 hold 阶段检测窗口是否仍在运行。
        hold: 为真时轨迹结束后持续保持最后一个目标。
    返回:
        无返回值；副作用是下发 action、推进仿真并写日志。
    """

    flush_interval_steps = max(1, int(round(0.05 / float(world.get_physics_dt()))))
    driven_joint_names = [controller.dof_names[int(index)] for index in controller.driven_indices]
    logger = JointTrackingLogger(log_path, driven_joint_names, flush_interval_steps=flush_interval_steps)
    try:
        full_target = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
        full_velocity = np.zeros(robot.num_dof, dtype=float)
        for step, point in enumerate(trajectory):
            full_target, full_velocity = controller.build_full_targets(
                point.joint_positions,
                point.joint_velocities,
                base_positions=full_target,
            )
            controller.apply(articulation_action_type, full_target, full_velocity)
            world.step(render=render)

            actual_position = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
            actual_velocity = np.asarray(robot.get_joint_velocities(), dtype=float).reshape(-1)
            logger.write(
                step=step,
                time_s=float(step) * float(world.get_physics_dt()),
                phase=point.phase,
                drive_update=True,
                desired_position=full_target[controller.driven_indices],
                actual_position=actual_position[controller.driven_indices],
                desired_velocity=full_velocity[controller.driven_indices],
                actual_velocity=actual_velocity[controller.driven_indices],
            )

        if hold:
            while simulation_app is None or simulation_app.is_running():
                controller.apply(articulation_action_type, full_target, full_velocity)
                world.step(render=render)
                if simulation_app is None:
                    break
    finally:
        logger.close()
