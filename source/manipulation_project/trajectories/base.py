"""轨迹数据类型。

这里定义项目内部统一使用的采样点容器。轨迹点可以同时携带关节空间和 TCP 空间信息，
但具体任务通常只填其中一部分。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    """一个离散轨迹采样点。

    输入字段:
        time_s: 采样点相对轨迹起点的时间，单位 s。
        joint_positions: 关节位置数组，单位 rad，顺序由所属轨迹的 ``joint_names`` 定义。
        joint_velocities: 可选关节速度数组，单位 rad/s，顺序同 ``joint_positions``。
        tcp_position: 可选 TCP 世界或任务坐标位置，shape ``(3,)``，单位 m。
        tcp_orientation: 可选 TCP 姿态，通常为 wxyz 四元数。
        phase: 任务阶段名，用于日志标记。
    输出:
        dataclass 实例作为轨迹中的一个不可变点。
    """

    time_s: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray | None = None
    tcp_position: np.ndarray | None = None
    tcp_orientation: np.ndarray | None = None
    phase: str = "trajectory"


class JointTrajectory:
    """基于列表的简单关节轨迹容器。

    输入:
        points: 至少包含一个 ``TrajectoryPoint`` 的列表。
        joint_names: 关节名元组，定义每个点中关节数组的顺序。
    输出:
        可迭代对象，迭代时逐个返回 ``TrajectoryPoint``。
    """

    def __init__(self, points: list[TrajectoryPoint], joint_names: tuple[str, ...]):
        """创建关节轨迹并校验非空。

        参数:
            points: 轨迹点列表，必须至少包含一个点。
            joint_names: 关节名元组，定义每个点中关节数组顺序。
        返回:
            无返回值；非法空轨迹会抛出 ``ValueError``。
        """

        if not points:
            raise ValueError("Trajectory must contain at least one point")
        self.points = points
        self.joint_names = joint_names

    def __iter__(self):
        """返回轨迹点迭代器。

        参数:
            无。
        返回:
            ``self.points`` 的迭代器。
        """

        return iter(self.points)

    def __len__(self) -> int:
        """返回轨迹点数量。

        参数:
            无。
        返回:
            轨迹点数量。
        """

        return len(self.points)
