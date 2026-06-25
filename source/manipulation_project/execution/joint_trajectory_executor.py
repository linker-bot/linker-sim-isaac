"""关节轨迹仿真执行器。

本模块位于 execution 层，只负责一件事：把上游已经采样好的
``JointTrajectory`` 转换成 Isaac articulation action，并按 physics step
逐帧下发到 Isaac world。

职责边界：
    * 不生成轨迹：插值、速度规划等逻辑属于 ``trajectories``。
    * 不求解任务：抓取、直线 TCP 运动等任务逻辑属于 ``tasks``。
    * 不做 FK/IK：正逆运动学和 cuMotion 适配属于 ``backends``/``tcp``。
    * 不直接决定控制子空间：关节名到 DOF index 的映射由 controller 负责。

这里的执行器只关心“每个仿真步应该给 articulation 下发什么完整 DOF 目标”，
并在下发后记录期望/实际关节状态，方便离线检查 tracking 误差。

单位约定沿用控制器和轨迹层：关节位置为 rad，速度为 rad/s，时间为 s。执行器不会改变
DOF 顺序，也不会重新解释 trajectory 的关节名，只通过 controller 提供的索引映射下发目标。
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

    # 日志刷盘不需要每步都做，否则长轨迹会产生明显 I/O 开销。这里按约 50 ms
    # 的仿真时间间隔 flush 一次；若 physics dt 大于 50 ms，则至少每步 flush。
    flush_interval_steps = max(1, int(round(0.05 / float(world.get_physics_dt()))))

    # logger 只记录 controller 实际驱动的 DOF。未驱动 DOF 虽然也存在于完整
    # articulation target 中，但通常不是 tracking 关注对象，写入 CSV 会增加噪声。
    driven_joint_names = [controller.dof_names[int(index)] for index in controller.driven_indices]
    logger = JointTrackingLogger(log_path, driven_joint_names, flush_interval_steps=flush_interval_steps)
    try:
        # 使用当前仿真状态作为 base_positions，可以避免未控制 DOF 在第一步被意外
        # 置零；后续每一步则用上一帧 full_target 作为基准，保持命令连续。
        full_target = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
        full_velocity = np.zeros(robot.num_dof, dtype=float)
        for step in range(len(trajectory)):
            # 将命令关节子空间扩展为 Isaac 需要的完整 DOF 数组。controller 内部会
            # 根据 driven_indices 写入命令目标，并保留/推导其它 DOF 的目标值。
            full_target, full_velocity = controller.build_full_targets(
                trajectory.positions[step],
                trajectory.velocities[step],
                base_positions=full_target,
            )

            # 下发目标后立即推进一个 physics step。Isaac 的 drive controller 会在
            # 本 step 内尝试跟踪 position/velocity target。
            controller.apply(articulation_action_type, full_target, full_velocity)
            world.step(render=render)

            # 读取 step 后的实际状态，用于和刚下发的 desired target 对比。
            # 注意：这里记录的是驱动关节切片，而不是完整 DOF 数组。
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
            # GUI 调试时，轨迹结束后继续保持最后一帧目标，用户可以观察稳定后的姿态。
            # 如果没有传入 simulation_app，则只执行一次，避免在测试或脚本中卡住。
            while simulation_app is None or simulation_app.is_running():
                controller.apply(articulation_action_type, full_target, full_velocity)
                world.step(render=render)
                if simulation_app is None:
                    break
    finally:
        # 无论中途是否异常，都关闭 logger，确保 CSV 缓冲区被写出。
        logger.close()
