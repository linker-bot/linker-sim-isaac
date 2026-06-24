"""关节空间轨迹生成。

输入是一组起始/目标关节位置，输出是按固定采样间隔离散化的 ``JointTrajectory``。
轨迹内部按 cuMotion 风格保存时间、位置、速度、加速度和 jerk 矩阵。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manipulation_project.trajectories.base import JointTrajectory
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
        ``JointTrajectory``，包含时间、位置、速度、加速度和 jerk 采样矩阵。
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
        return JointTrajectory.from_samples(
            times=np.asarray([0.0], dtype=float),
            positions=target.reshape(1, -1),
            velocities=np.zeros((1, target.size), dtype=float),
            accelerations=np.zeros((1, target.size), dtype=float),
            jerks=np.zeros((1, target.size), dtype=float),
            phases=(phase,),
            joint_names=tuple(joint_names),
        )

    num_steps = max(1, int(np.ceil(duration_s / sample_dt)))
    times = np.asarray([min(duration_s, step * sample_dt) for step in range(num_steps + 1)], dtype=float)
    positions = np.zeros((times.size, start.size), dtype=float)
    for step in range(num_steps + 1):
        time_s = times[step]
        alpha = time_s / duration_s
        scale = fn(alpha)
        positions[step] = start + scale * (target - start)

    velocities = _differentiate(positions, times)
    accelerations = _differentiate(velocities, times)
    jerks = _differentiate(accelerations, times)
    return JointTrajectory.from_samples(
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        phases=tuple(phase for _ in range(times.size)),
        joint_names=tuple(joint_names),
    )


def _differentiate(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if values.shape[0] == 1:
        return np.zeros_like(values)
    result = np.zeros_like(values)
    for index in range(1, values.shape[0]):
        dt = max(1.0e-12, float(times[index] - times[index - 1]))
        result[index] = (values[index] - values[index - 1]) / dt
    return result
