"""后端无关的碰撞对象描述。

这些数据结构只表达几何和位姿，不依赖 cuRobo 或 Isaac。后端适配层可以把同一对象
转换成 cuRobo obstacle、Isaac debug marker 或其它规划库的碰撞体。尺寸单位为米，
姿态用 4x4 齐次矩阵生成。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CollisionObject:
    """规划层使用的简化障碍物。

    ``pose`` 是 4x4 齐次矩阵，或可被转换为 4x4 的数组。``size`` 的含义由
    ``shape`` 决定：cuboid 为 ``(x, y, z)``，sphere 为 ``(radius,)``，
    capsule 为 ``(radius, length)``。``padding`` 是规划阶段的安全膨胀量，不改写原始
    ``size``，由后端适配时按几何类型转换。
    """

    name: str
    shape: str
    pose: np.ndarray
    size: tuple[float, ...]
    enabled: bool = True
    padding: float = 0.0

    def pose_matrix(self) -> np.ndarray:
        """返回 shape ``(4, 4)`` 的齐次位姿矩阵。

        返回值是新构造的 numpy view/array；调用方可以把它传给后端转换器，但不应假设它与
        原始 ``pose`` 对象共享可写语义。
        """

        return np.asarray(self.pose, dtype=float).reshape(4, 4)

    def padded_size(self) -> tuple[float, ...]:
        """按 ``padding`` 返回膨胀后的尺寸。

        cuboid 在每个维度两侧各膨胀一次，因此尺寸增加 ``2*padding``；sphere/capsule
        只增加半径，长度保持不变。未知形状保持原尺寸，交给具体后端决定是否支持。
        """

        padding = float(self.padding)
        if padding == 0.0:
            return tuple(float(value) for value in self.size)
        shape = self.shape.lower()
        if shape == "cuboid":
            return tuple(float(value) + 2.0 * padding for value in self.size)
        if shape in {"sphere", "capsule"}:
            values = [float(value) for value in self.size]
            values[0] += padding
            return tuple(values)
        return tuple(float(value) for value in self.size)
