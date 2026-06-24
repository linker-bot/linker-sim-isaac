"""关节轨迹仿真执行器。

execution 层专门负责把已经生成好的 ``JointTrajectory`` 下发到 Isaac world。
它不负责生成轨迹，也不负责 FK/IK；这些分别属于 trajectories/tasks/backends。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipulation_project.controllers.implicit_drive_controller import ImplicitDriveController
from manipulation_project.logging.joint_logger import JointTrackingLogger


def execute_joint_trajectory(
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
        trajectory: ``JointTrajectory``，以矩阵形式保存命令关节轨迹。
        log_path: CSV 日志路径；为 ``None`` 时禁用写文件。
        render: 是否在 ``world.step`` 时渲染。
        simulation_app: 可选 Isaac app，用于 hold 阶段检测窗口是否仍在运行。
        hold: 为真时轨迹结束后持续保持最后一个目标。
    """

    flush_interval_steps = max(1, int(round(0.05 / float(world.get_physics_dt()))))
    driven_joint_names = [controller.dof_names[int(index)] for index in controller.driven_indices]
    logger = JointTrackingLogger(log_path, driven_joint_names, flush_interval_steps=flush_interval_steps)
    try:
        full_target = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
        full_velocity = np.zeros(robot.num_dof, dtype=float)
        for step in range(len(trajectory)):
            full_target, full_velocity = controller.build_full_targets(
                trajectory.positions[step],
                trajectory.velocities[step],
                base_positions=full_target,
            )
            controller.apply(articulation_action_type, full_target, full_velocity)
            world.step(render=render)

            actual_position = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
            actual_velocity = np.asarray(robot.get_joint_velocities(), dtype=float).reshape(-1)
            logger.write(
                step=step,
                time_s=float(step) * float(world.get_physics_dt()),
                phase=trajectory.phases[step],
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
