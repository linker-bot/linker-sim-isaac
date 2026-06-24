"""笛卡尔 TCP 位姿轨迹采样。

本模块只做纯数学采样：给定起点/终点 TCP 位姿和时间序列，返回任务空间
waypoint。它不读取配置、不调用 FK/IK，也不依赖 Isaac。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipulation_project.utils.rotations import slerp_quat_wxyz
from manipulation_project.utils.timing import sample_linear_positions


@dataclass(frozen=True)
class CartesianPoseWaypoint:
    """单个 TCP 位姿 waypoint。"""

    time_s: float
    position: np.ndarray
    orientation: np.ndarray | None = None


def sample_cartesian_pose_line(
    *,
    times: np.ndarray,
    start_position,
    target_position,
    start_orientation=None,
    target_orientation=None,
) -> tuple[CartesianPoseWaypoint, ...]:
    """采样 TCP 位姿直线路径。

    位置在两点间线性插值；姿态在 wxyz 四元数之间 slerp。若起点或终点姿态为
    ``None``，所有 waypoint 的姿态都为 ``None``，表示只约束位置。
    """

    sample_times = np.asarray(times, dtype=float).reshape(-1)
    positions = sample_linear_positions(start_position, target_position, sample_times)
    orientations = slerp_quat_wxyz(start_orientation, target_orientation, sample_times)
    return tuple(
        CartesianPoseWaypoint(
            time_s=float(time_s),
            position=np.asarray(position, dtype=float).reshape(3),
            orientation=None if orientation is None else np.asarray(orientation, dtype=float).reshape(4),
        )
        for time_s, position, orientation in zip(sample_times, positions, orientations, strict=True)
    )
