"""控制器基础数据类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerConfig:
    """常见控制器参数。

    输入字段:
        stiffness: 位置 drive 刚度。
        damping: 位置 drive 阻尼。
        max_force: drive 最大力/力矩。
        joint_friction: 运行时关节摩擦默认值。
    输出:
        作为更具体控制器配置的共享基础结构。
    """

    stiffness: float = 1000.0
    damping: float = 50.0
    max_force: float = 100.0
    joint_friction: float = 0.5
