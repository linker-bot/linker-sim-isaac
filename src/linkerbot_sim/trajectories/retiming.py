"""关节轨迹的执行时间网格与路径进度重采样。"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from linkerbot_sim.trajectories.types import JointTrajectory
from linkerbot_sim.utils.timing import differentiate_samples


def trajectory_sample_times(
    *,
    duration_s: float,
    sample_dt_s: float,
    include_start: bool = False,
) -> np.ndarray:
    """生成固定步长执行使用的 canonical 时间网格。

    除最后一个不足完整 tick 的端点外，样本都落在 ``sample_dt_s`` 的整数倍上。默认返回每个
    physics tick 后应执行的时间点；需要描述完整轨迹域的 batch/tiled 调用方可通过
    ``include_start=True`` 在首行加入 ``t=0``。
    """

    duration = float(duration_s)
    sample_dt = float(sample_dt_s)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be finite and non-negative")
    if not np.isfinite(sample_dt) or sample_dt <= 0.0:
        raise ValueError("sample_dt_s must be finite and positive")
    target_duration = max(duration, sample_dt)
    steps = max(1, int(np.ceil(target_duration / sample_dt)))
    tick_times = np.minimum(
        np.arange(1, steps + 1, dtype=float) * sample_dt,
        target_duration,
    )
    if include_start:
        return np.concatenate((np.asarray([0.0], dtype=float), tick_times))
    return tick_times


def retime_joint_trajectory(
    trajectory: JointTrajectory,
    *,
    duration_s: float | None,
    sample_dt_s: float | None,
    start_position: np.ndarray | None = None,
    phase: str | None = None,
    include_start: bool = False,
) -> JointTrajectory:
    """按关节路径累计进度把轨迹重采样到 canonical 执行网格。

    cuRobo single 与 batch planner 可能返回不同 dt、样本数和重复 waypoint。重定时只保留路径
    几何，并在目标时间网格上重新计算速度、加速度与 jerk，避免把源时间域的导数直接贴到新
    时间戳。``start_position`` 可显式补上求解前的当前关节位置；batch/tiled 结果使用
    ``include_start=True``，单条执行轨迹保持默认的“首个 physics tick 后下发第一行”语义。
    """

    if duration_s is None or sample_dt_s is None:
        if include_start:
            raise ValueError("include_start requires both duration_s and sample_dt_s")
        return trajectory
    target_times = trajectory_sample_times(
        duration_s=float(duration_s),
        sample_dt_s=float(sample_dt_s),
        include_start=include_start,
    )
    target_duration = float(target_times[-1])
    source_positions = _source_positions_with_start_anchor(
        trajectory,
        start_position=start_position,
    )
    progress = target_times / target_duration
    positions = _resample_positions_by_path_progress(source_positions, progress)

    if include_start:
        derivative_times = target_times
        derivative_positions = positions
        derivative_offset = 0
    else:
        derivative_times = np.concatenate(([0.0], target_times))
        derivative_positions = np.vstack([source_positions[0], positions])
        derivative_offset = 1
    velocities = differentiate_samples(derivative_positions, derivative_times)
    accelerations = differentiate_samples(velocities, derivative_times)
    jerks = differentiate_samples(accelerations, derivative_times)
    phases = tuple(
        _phase_at_progress(trajectory, progress=value, override=phase)
        for value in progress
    )
    return JointTrajectory.from_samples(
        times=target_times,
        positions=positions,
        velocities=velocities[derivative_offset:],
        accelerations=accelerations[derivative_offset:],
        jerks=jerks[derivative_offset:],
        efforts=np.zeros_like(positions),
        phases=phases,
        joint_names=trajectory.joint_names,
    )


def _source_positions_with_start_anchor(
    trajectory: JointTrajectory,
    *,
    start_position: np.ndarray | None,
) -> np.ndarray:
    """返回源路径位置，并在需要时补上显式起点。"""

    positions = np.asarray(trajectory.positions, dtype=float)
    if start_position is None:
        return positions
    start = np.asarray(start_position, dtype=float).reshape(-1)
    if start.size != positions.shape[1]:
        raise ValueError(
            f"start_position expected {positions.shape[1]} values, got {start.size}"
        )
    if not np.all(np.isfinite(start)):
        raise ValueError("start_position must contain finite values")
    if np.allclose(positions[0], start):
        return positions
    return np.vstack([start, positions])


def _resample_positions_by_path_progress(
    positions: np.ndarray,
    progress_values: np.ndarray,
) -> np.ndarray:
    """按累计关节空间距离对路径逐列插值。"""

    path = np.asarray(positions, dtype=float)
    progress = np.asarray(progress_values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(path)):
        raise ValueError("positions must contain finite values")
    if not np.all(np.isfinite(progress)):
        raise ValueError("progress values must be finite")
    if path.shape[0] == 1:
        return np.repeat(path, progress.size, axis=0)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1.0e-12:
        return np.repeat(path[-1:], progress.size, axis=0)
    normalized = cumulative / total
    keep = np.concatenate(([True], np.diff(normalized) > 1.0e-12))
    if normalized[-1] > normalized[np.flatnonzero(keep)[-1]]:
        keep[-1] = True
    sample_progress = normalized[keep]
    sample_positions = path[keep]
    queries = np.clip(progress, 0.0, 1.0)
    interpolator = make_interp_spline(
        sample_progress,
        sample_positions,
        k=1,
        axis=0,
    )
    return np.asarray(interpolator(queries), dtype=float)


def _phase_at_progress(
    trajectory: JointTrajectory,
    *,
    progress: float,
    override: str | None,
) -> str:
    """返回目标进度对应的 phase，显式 override 优先。"""

    if override is not None:
        return str(override)
    index = min(
        len(trajectory.phases) - 1,
        max(0, int(round(float(progress) * (len(trajectory.phases) - 1)))),
    )
    return trajectory.phases[index]


__all__ = ["retime_joint_trajectory", "trajectory_sample_times"]
