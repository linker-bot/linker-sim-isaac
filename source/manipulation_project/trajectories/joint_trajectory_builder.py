"""关节空间轨迹构造。

输入是一组起始/目标关节位置，输出是按固定采样间隔离散化的 ``JointTrajectory``。轨迹内部
按 cuMotion 风格保存时间、位置、速度、加速度和 jerk 矩阵。

职责边界:
    * 只负责关节空间数值采样，不进行速度/加速度约束优化。
    * 不解析机器人模型、不检查关节限位；调用方必须保证输入数组已经按期望关节名顺序排列。
    * 有限差分生成的速度/加速度只用于控制目标和日志诊断，不代表严格动力学最优曲线。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manipulation_project.trajectories.types import JointTrajectory
from manipulation_project.trajectories.interpolation import interpolation_fn
from manipulation_project.utils.timing import differentiate_samples


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
        phase: 写入每个采样行的阶段名。
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

    # 插值函数只定义 0..1 的无量纲进度，具体持续时间由 duration_s 决定；这样同一种
    # smoothstep 可用于不同任务阶段。
    fn = interpolation_fn(interpolation)
    if duration_s == 0:
        return joint_trajectory_from_positions(
            times=np.asarray([0.0], dtype=float),
            positions=target.reshape(1, -1),
            phases=(phase,),
            joint_names=tuple(joint_names),
        )

    # 使用 ceil 保证最后一个采样覆盖 duration_s；每个时间点再用 min 钳制到终点，避免
    # duration 不是 sample_dt 整数倍时越界。
    num_steps = max(1, int(np.ceil(duration_s / sample_dt)))
    times = np.asarray([min(duration_s, step * sample_dt) for step in range(num_steps + 1)], dtype=float)
    positions = np.zeros((times.size, start.size), dtype=float)
    for step in range(num_steps + 1):
        time_s = times[step]
        alpha = time_s / duration_s
        scale = fn(alpha)
        positions[step] = start + scale * (target - start)

    return joint_trajectory_from_positions(
        times=times,
        positions=positions,
        phases=tuple(phase for _ in range(times.size)),
        joint_names=tuple(joint_names),
    )


def joint_trajectory_from_positions(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    joint_names: Sequence[str],
    phases: Sequence[str] | None = None,
    phase: str = "trajectory",
    differentiate: bool = True,
    efforts: np.ndarray | None = None,
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

    # phase 用于日志和可视化标记，不参与轨迹数学；仍要求与采样点数量一致，避免 CSV 行
    # 无法对应任务阶段。
    if phases is None:
        phases_tuple = tuple(phase for _ in range(times_array.size))
    else:
        phases_tuple = tuple(str(value) for value in phases)
        if len(phases_tuple) != times_array.size:
            raise ValueError("phases length must match trajectory samples")

    if differentiate:
        # 速度、加速度、jerk 通过同一时间网格上的有限差分得到。对于手写关键帧轨迹，
        # 这比全部填零更有利于 drive velocity target 和误差分析。
        velocities = differentiate_samples(positions_array, times_array)
        accelerations = differentiate_samples(velocities, times_array)
        jerks = differentiate_samples(accelerations, times_array)
    else:
        velocities = None
        accelerations = None
        jerks = None

    return JointTrajectory.from_samples(
        times=times_array,
        positions=positions_array,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        efforts=efforts,
        phases=phases_tuple,
        joint_names=tuple(joint_names),
    )
