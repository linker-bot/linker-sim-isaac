"""cuRobo BatchMotionPlanner result 的 seed/trajectory 解码与重采样。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.backends.curobo.tensor_adapter import tensor_like_to_numpy
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.retiming import retime_joint_trajectory


def batch_success_matrix(result: object, *, rows: int) -> np.ndarray:
    """把任意 cuRobo success shape 规范为 ``(B, S)`` bool。"""

    value = getattr(result, "success", None)
    if value is None:
        return np.zeros((rows, 1), dtype=bool)
    success = np.asarray(tensor_like_to_numpy(value), dtype=bool)
    if success.ndim == 0:
        success = np.full((rows, 1), bool(success), dtype=bool)
    elif success.ndim == 1:
        if success.size != rows:
            raise ValueError("cuRobo batch success has wrong batch dimension")
        success = success.reshape(rows, 1)
    elif success.ndim == 2:
        if success.shape[0] != rows:
            raise ValueError("cuRobo batch success has wrong batch dimension")
    else:
        success = success.reshape(success.shape[0], -1)
        if success.shape[0] != rows:
            raise ValueError("cuRobo batch success has wrong batch dimension")
    return success


def batch_result_row_positions(
    result: object,
    *,
    row: int,
    seed_index: int,
    joint_names: tuple[str, ...],
    sample_dt_s: float,
    query_times: np.ndarray,
    duration_s: float,
    start_position: np.ndarray,
) -> np.ndarray:
    """抽取一个 batch row/seed，并按共享路径进度规则重采样。"""

    trajectory = _extract_curobo_trajectory(result)
    positions = tensor_like_to_numpy(_required_attr(trajectory, "position"))
    if positions.ndim == 4:
        row_positions = positions[int(row), int(seed_index), :, :]
    elif positions.ndim == 3:
        row_positions = positions[int(row), :, :]
    elif positions.ndim == 2:
        row_positions = positions
    else:
        raise ValueError("cuRobo batch trajectory position must be 2D, 3D or 4D")
    times = _batch_trajectory_times(
        result, trajectory, row_positions.shape[0], sample_dt_s
    )
    source_trajectory = joint_trajectory_from_positions(
        times=times,
        positions=row_positions,
        joint_names=joint_names,
        phase="curobo_batch_path",
    )
    retimed = retime_joint_trajectory(
        source_trajectory,
        duration_s=duration_s,
        sample_dt_s=sample_dt_s,
        start_position=start_position,
        phase="curobo_batch_path",
        include_start=True,
    )
    expected_times = np.asarray(query_times, dtype=float).reshape(-1)
    if retimed.times.shape != expected_times.shape or not np.allclose(
        retimed.times, expected_times
    ):
        raise ValueError("cuRobo batch target times do not match canonical grid")
    return retimed.positions


def _extract_curobo_trajectory(value: object) -> object:
    """按 cuRobo 版本优先级解析 interpolated/raw joint trajectory 容器。"""

    for name in ("interpolated_trajectory", "trajectory", "js_solution"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return value


def _batch_trajectory_times(
    result: object,
    trajectory: object,
    count: int,
    sample_dt_s: float,
) -> np.ndarray:
    """从 result/trajectory 的 dt 字段恢复时间轴，缺失时使用 request sample period。"""

    for owner, names in (
        (result, ("interpolated_trajectory_dt", "trajectory_dt", "dt")),
        (trajectory, ("dt",)),
    ):
        for name in names:
            value = getattr(owner, name, None)
            if value is None:
                continue
            dt = float(np.asarray(tensor_like_to_numpy(value)).reshape(-1)[0])
            return np.arange(count, dtype=float) * dt
    return np.arange(count, dtype=float) * float(sample_dt_s)


def _required_attr(value: object, name: str) -> object:
    """读取 cuRobo result 必需属性，并把版本不匹配转换为明确 ValueError。"""

    attr = getattr(value, name, None)
    if attr is None:
        raise ValueError(f"cuRobo batch trajectory missing {name!r}")
    return attr


__all__ = ["batch_result_row_positions", "batch_success_matrix"]
