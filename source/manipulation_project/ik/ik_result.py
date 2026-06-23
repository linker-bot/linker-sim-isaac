"""IK 结果数据类型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IKResult:
    """IK 后端返回的统一结果。

    输入字段:
        joint_positions: 求解得到的关节位置数组，单位 rad，顺序由后端 ``joint_names`` 定义。
        success: 后端是否认为求解成功。
        position_error: TCP 位置误差，单位 m。
        orientation_error: 可选姿态误差，单位/定义取决于后端。
        message: 可选诊断信息。
    输出:
        被任务层用于写回机械臂关节目标和记录 IK 诊断。
    """

    joint_positions: np.ndarray
    success: bool
    position_error: float
    orientation_error: float | None = None
    message: str = ""
