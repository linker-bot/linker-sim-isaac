"""轻量机器人状态容器。

这些 dataclass 不直接绑定 Isaac 对象，只保存采样下来的数组和名字顺序，
适合作为控制器、任务和日志之间传递状态的中间格式。
数组单位沿用采样来源：旋转关节通常为 rad/rad/s，平移关节则为 m/m/s；该容器不做单位转换。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointState:
    """一次关节状态采样。

    输入字段:
        names: 关节名顺序。
        positions: 关节位置数组，单位通常为 rad。
        velocities: 可选关节速度数组，单位通常为 rad/s。
        efforts: 可选关节力/力矩数组，单位由仿真或驱动层定义。
    输出:
        不可变 dataclass，作为状态快照在模块之间传递。
    """

    names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray | None = None
    efforts: np.ndarray | None = None

    def index(self, name: str) -> int:
        """返回某个关节名在数组中的下标。

        参数:
            name: 需要查找的关节名。
        返回:
            ``name`` 在 ``self.names`` 中的整数下标。
        """

        return self.names.index(name)
