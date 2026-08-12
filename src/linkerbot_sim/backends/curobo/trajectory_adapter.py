"""cuRobo 轨迹结果到项目 ``JointTrajectory`` 的适配。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.types import JointTrajectory
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


def joint_trajectory_from_curobo(
    result_or_trajectory,
    *,
    joint_names: Sequence[str],
    sample_dt: float | None = None,
    phase: str = "trajectory",
) -> JointTrajectory:
    """把 cuRobo result / JointState trajectory 转成项目 ``JointTrajectory``。

    cuRobo high-level planner 常返回 ``TrajOptSolverResult``，其中更适合执行的是
    ``interpolated_trajectory``；若没有该字段，则回退到 ``trajectory`` 或对象本身。
    """

    trajectory = _extract_curobo_trajectory(result_or_trajectory)
    positions = _single_trajectory_matrix(
        _required_attr(trajectory, "position"),
        name="position",
    )
    times = _trajectory_times(
        result_or_trajectory, trajectory, positions.shape[0], sample_dt
    )
    velocities = _optional_matrix(trajectory, "velocity", positions.shape)
    accelerations = _optional_matrix(trajectory, "acceleration", positions.shape)
    if velocities is None and accelerations is None:
        return joint_trajectory_from_positions(
            times=times,
            positions=positions,
            joint_names=tuple(joint_names),
            phase=phase,
        )
    phases = tuple(str(phase) for _ in range(times.size))
    return JointTrajectory.from_samples(
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        phases=phases,
        joint_names=tuple(joint_names),
    )


def joint_trajectory_from_motion_result(
    result,
    *,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """把 cuRobo 规划结果转换为可执行的关节轨迹。

    正常 cuRobo 结果携带 ``trajectory``；图搜索等路径也可能只返回离散 ``path``。
    后一种情况在这里生成确定的时间轴，避免上层 timeline 了解 cuRobo 的多种返回形态。
    """

    if result.trajectory is not None:
        if isinstance(result.trajectory, JointTrajectory):
            return result.trajectory
        return joint_trajectory_from_curobo(
            result.trajectory,
            joint_names=tuple(joint_names),
            sample_dt=sample_dt,
            phase=phase,
        )
    if result.path is None:
        raise RuntimeError(
            f"cuRobo planner returned no executable trajectory: status={result.status}"
        )
    path = np.asarray(result.path, dtype=float)
    if path.ndim != 2 or path.shape[0] == 0:
        raise RuntimeError(
            f"cuRobo planner returned an empty path: status={result.status}"
        )
    if path.shape[0] == 1:
        times = np.asarray([max(float(duration_s), float(sample_dt))], dtype=float)
    else:
        times = np.linspace(0.0, float(duration_s), path.shape[0], dtype=float)[1:]
        path = path[1:]
        if path.shape[0] == 0:  # pragma: no cover - 前面的 shape 分支已覆盖
            path = np.asarray(result.path, dtype=float)[-1:].copy()
            times = np.asarray([float(duration_s)], dtype=float)
    return joint_trajectory_from_positions(
        times=times,
        positions=path,
        joint_names=tuple(joint_names),
        phase=phase,
    )


def _extract_curobo_trajectory(value):
    """从 result-like 对象中取出 cuRobo JointState trajectory。"""

    for name in ("interpolated_trajectory", "trajectory"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return value


def _trajectory_times(
    result, trajectory, count: int, sample_dt: float | None
) -> np.ndarray:
    """根据 cuRobo dt 字段或调用方 sample_dt 生成时间轴。"""

    for name in ("interpolated_trajectory_dt", "trajectory_dt", "dt"):
        value = getattr(result, name, None)
        if value is not None:
            dt = float(tensor_like_to_numpy(value, dtype=float).reshape(-1)[0])
            return _times_from_dt(count, dt, source=name)
    value = getattr(trajectory, "dt", None)
    if value is not None:
        dt = float(tensor_like_to_numpy(value, dtype=float).reshape(-1)[0])
        return _times_from_dt(count, dt, source="trajectory.dt")
    if sample_dt is None:
        raise ValueError(
            "cuRobo trajectory is missing a dt source; pass sample_dt explicitly"
        )
    return _times_from_dt(count, float(sample_dt), source="sample_dt")


def _times_from_dt(count: int, dt: float, *, source: str) -> np.ndarray:
    """校验采样周期并生成从零开始的时间轴。"""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"cuRobo trajectory {source} must be finite and positive")
    return np.arange(count, dtype=float) * dt


def _optional_matrix(
    trajectory, name: str, shape: tuple[int, int]
) -> np.ndarray | None:
    """读取可选 velocity/acceleration 矩阵。"""

    value = getattr(trajectory, name, None)
    if value is None:
        return None
    matrix = _single_trajectory_matrix(value, name=name)
    if matrix.shape != shape:
        raise ValueError(f"cuRobo trajectory {name} shape mismatch")
    return matrix


def _single_trajectory_matrix(value, *, name: str) -> np.ndarray:
    """把 cuRobo 单请求轨迹矩阵规范化为 ``(T, D)``。

    单问题 ``MotionPlanner`` 结果可能保留 batch/seed 等前置单例维度，例如
    ``(1, T, D)`` 或 ``(1, 1, T, D)``。项目侧单条 ``JointTrajectory`` 只消费二维矩阵；
    如果前置维度不是单例，说明调用方拿到了真正 batched 结果，应在 batch adapter 中逐行抽取。
    """

    matrix = tensor_like_to_numpy(value, dtype=float)
    if matrix.ndim < 2:
        raise ValueError(f"cuRobo trajectory {name} must have shape (T,D)")
    if matrix.ndim == 2:
        return matrix
    leading_shape = matrix.shape[: matrix.ndim - 2]
    if any(size != 1 for size in leading_shape):
        raise ValueError(
            f"batched cuRobo trajectory {name} requires per-row conversion"
        )
    return matrix.reshape(matrix.shape[-2], matrix.shape[-1])


def _required_attr(value, name: str):
    """读取必填属性。"""

    attr = getattr(value, name, None)
    if attr is None:
        raise ValueError(f"cuRobo trajectory missing {name!r}")
    return attr
