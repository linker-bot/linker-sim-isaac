"""关节空间轨迹生成。

输入是一组起始/目标关节位置，输出是按固定采样间隔离散化的 ``JointTrajectory``。
速度通过相邻采样点差分估计，适合给位置驱动或日志使用。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manipulation_project.trajectories.base import JointTrajectory, TrajectoryPoint
from manipulation_project.trajectories.interpolation import interpolation_fn


def build_joint_target_trajectory(
    start_positions: Sequence[float],
    target_positions: Sequence[float],
    *,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    interpolation: str = "smoothstep",
    phase: str = "joint_target",
) -> JointTrajectory:
    """采样固定时长的关节目标轨迹。

    参数:
        start_positions: 起始关节位置序列，单位 rad。
        target_positions: 目标关节位置序列，单位 rad。
        joint_names: 与位置数组一一对应的关节名。
        duration_s: 轨迹持续时间，单位 s；为 0 时只返回目标点。
        sample_dt: 采样时间间隔，单位 s。
        interpolation: 插值函数名称。
        phase: 写入每个 ``TrajectoryPoint`` 的阶段名。
    返回:
        ``JointTrajectory``，每个点包含位置、差分速度和时间戳。
    """

    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")
    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive")
    start = np.asarray(start_positions, dtype=float).reshape(-1)
    target = np.asarray(target_positions, dtype=float).reshape(-1)
    if start.shape != target.shape:
        raise ValueError(f"start/target shape mismatch: {start.shape} vs {target.shape}")
    if len(joint_names) != start.size:
        raise ValueError(f"joint_names expected {start.size} names, got {len(joint_names)}")

    fn = interpolation_fn(interpolation)
    if duration_s == 0:
        return JointTrajectory(
            [TrajectoryPoint(time_s=0.0, joint_positions=target.copy(), joint_velocities=np.zeros_like(target), phase=phase)],
            tuple(joint_names),
        )

    num_steps = max(1, int(np.ceil(duration_s / sample_dt)))
    points: list[TrajectoryPoint] = []
    previous_position = start
    previous_time = 0.0
    for step in range(num_steps + 1):
        time_s = min(duration_s, step * sample_dt)
        alpha = time_s / duration_s
        scale = fn(alpha)
        position = start + scale * (target - start)
        if step == 0:
            velocity = np.zeros_like(position)
        else:
            dt = max(1.0e-12, time_s - previous_time)
            velocity = (position - previous_position) / dt
        points.append(TrajectoryPoint(time_s=time_s, joint_positions=position, joint_velocities=velocity, phase=phase))
        previous_position = position
        previous_time = time_s
    return JointTrajectory(points, tuple(joint_names))
