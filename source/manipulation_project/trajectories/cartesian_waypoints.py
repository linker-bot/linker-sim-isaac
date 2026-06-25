"""笛卡尔 TCP 位姿 waypoint 采样。

本模块只做纯数学采样：给定起点/终点 TCP 位姿和时间序列，返回任务空间 waypoint。它不读取
配置、不调用 FK/IK，也不依赖 Isaac。

职责边界:
    * 位置在线段上做线性插值，姿态在 wxyz 四元数之间做 slerp。
    * 不检查 waypoint 是否可达，也不做碰撞或速度约束规划。
    * 不改变坐标系；起点和终点必须已经在同一 base/world 约定坐标系下。

位置单位为 m，姿态使用项目统一的 wxyz 四元数。返回 waypoint 的时间只用于后续 IK 和关节
轨迹构造，不保证满足速度或加速度约束。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipulation_project.utils.rotations import slerp_quat_wxyz
from manipulation_project.utils.timing import sample_linear_positions


@dataclass(frozen=True)
class CartesianPoseWaypoint:
    """单个 TCP 位姿 waypoint。

    ``orientation`` 为 ``None`` 表示下游 IK 只需要约束位置；否则为 wxyz 四元数。
    """

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

    # 时间序列由调用方决定，这里只按最后一个时间作为总时长进行归一化插值。
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
