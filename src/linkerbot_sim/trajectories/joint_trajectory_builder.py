"""关节空间轨迹构造。

输入是一组已经采样好的关节位置和时间，输出是项目统一的 ``JointTrajectory``。轨迹内部
按 cuRobo 风格保存时间、位置、速度、加速度和 jerk 矩阵。

职责边界:
    * 只负责关节空间数值采样，不进行速度/加速度约束优化。
    * 不解析机器人模型、不检查关节限位；调用方必须保证输入数组已经按期望关节名顺序排列。
    * 有限差分生成的速度/加速度只用于控制目标和日志诊断，不代表严格动力学最优曲线。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.trajectories.types import JointTrajectory
from linkerbot_sim.utils.timing import differentiate_samples


def joint_trajectory_from_positions(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    joint_names: Sequence[str],
    phase: str = "trajectory",
) -> JointTrajectory:
    """从位置采样矩阵构造完整 ``JointTrajectory``。

    这是项目里“positions + times -> position/velocity/acceleration/jerk 轨迹”的
    通用入口。调用方只需要准备好采样时间、位置矩阵和关节名；速度、加速度和 jerk
    默认由有限差分自动生成。
    """

    times_array = np.asarray(times, dtype=float).reshape(-1)
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.ndim != 2:
        raise ValueError("positions must have shape (N, dof)")
    if positions_array.shape[0] != times_array.size:
        raise ValueError("times length must match positions rows")

    # 速度、加速度、jerk 通过同一时间网格上的有限差分得到。对于手写关键帧轨迹，
    # 这比全部填零更有利于 drive velocity target 和误差分析。
    velocities = differentiate_samples(positions_array, times_array)
    accelerations = differentiate_samples(velocities, times_array)
    jerks = differentiate_samples(accelerations, times_array)
    # phase 用于日志和可视化标记，不参与轨迹数学。需要逐点 phase 或 effort 曲线时，
    # 调用方应直接使用 ``JointTrajectory.from_samples(...)``，保持这个 builder 的职责简单。
    phases_tuple = tuple(str(phase) for _ in range(times_array.size))

    return JointTrajectory.from_samples(
        times=times_array,
        positions=positions_array,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        phases=phases_tuple,
        joint_names=tuple(joint_names),
    )
