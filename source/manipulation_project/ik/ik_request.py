"""IK 请求数据类型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IKRequest:
    """单个 TCP IK 目标。

    输入字段:
        target_position: TCP 目标位置，shape ``(3,)``，单位 m。
        target_orientation: 可选 TCP 目标姿态，wxyz 四元数；为 ``None`` 时只约束位置。
        tcp_type: TCP 类型标签，当前主要用于保留扩展信息。
        warm_start: 可选关节初值，单位 rad；用于连续目标减少跳解。
        position_tolerance: 位置容差，单位 m。
        orientation_tolerance: 姿态容差，单位由 IK 后端定义，通常近似 rad。
    输出:
        传给实现 ``IKSolver.solve`` 的后端，返回 ``IKResult``。
    """

    target_position: np.ndarray
    target_orientation: np.ndarray | None = None
    tcp_type: str = "flange"
    warm_start: np.ndarray | None = None
    position_tolerance: float = 1.0e-3
    orientation_tolerance: float = 1.0e-2
