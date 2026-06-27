"""控制器共享数据类型。

这些类型描述项目层面的控制意图，而不是 Isaac runtime 对象本身。配置解析、执行器和控制器
都可以导入它们，不需要依赖具体 controller 实现文件。主动关节支持 position、velocity 和
effort 三类模式；mimic follower 始终使用独立的 position drive 配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from linkerbot_sim.robots.classification import component_for_name


ControlMode = Literal["position", "velocity", "effort"]
ControlMethod = Literal["implicit", "explicit", "direct"]


@dataclass(frozen=True)
class ComponentControlSettings:
    """单个部件的主动控制和 follower position drive 配置。

    输入字段:
        mode: 主动关节控制模式，支持 ``position``、``velocity`` 或 ``effort``。
        method: 主动关节控制方法。position/velocity 支持 ``implicit`` 和 ``explicit``；
            effort 使用 ``direct``。
        stiffness/damping: 主动关节显式 PD 或 implicit drive 增益。velocity 控制只使用
            ``damping``；effort 直通模式不使用这两个字段。
        max_force: 主动关节 implicit drive 最大力/力矩，也是显式 PD effort 的限幅默认值。
        effort_limit: effort 直通模式和显式 effort 的限幅；为 ``None`` 时沿用 ``max_force``。
        joint_friction: 主动关节默认摩擦；MJCF frictionloss 会覆盖同名关节。
        follower_stiffness/follower_damping: mimic follower 的 Isaac position drive 增益。
        follower_max_force: follower drive 最大力/力矩；为 ``None`` 时沿用 ``max_force``。
        follower_joint_friction: follower 默认摩擦；为 ``None`` 时沿用 ``joint_friction``。
    输出:
        作为 ``JointControlSettings`` 的 arm/hand/default 子配置。
    """

    mode: ControlMode = "position"
    method: ControlMethod = "implicit"
    stiffness: tuple[float, ...] = (1000.0,)
    damping: tuple[float, ...] = (50.0,)
    max_force: float = 100.0
    effort_limit: float | None = None
    joint_friction: float = 0.5
    follower_stiffness: tuple[float, ...] = (50000.0,)
    follower_damping: tuple[float, ...] = (50.0,)
    follower_max_force: float | None = None
    follower_joint_friction: float | None = None

    def active_effort_limit(self) -> float:
        """返回主动关节 effort 限幅。

        返回:
            effort 限幅；``effort_limit`` 缺省时使用 ``max_force``。
        """

        return float(self.max_force if self.effort_limit is None else self.effort_limit)


@dataclass(frozen=True)
class JointControlSettings:
    """按部件分组的 runtime 关节控制配置。

    输入字段:
        default: 未识别部件时使用的回退参数。
        arm: 机械臂主动/从动关节参数。
        hand: 灵巧手主动/从动关节参数。
    输出:
        传给 ``JointController`` 后用于写入 articulation runtime 参数并计算 action。
    """

    default: ComponentControlSettings = field(default_factory=ComponentControlSettings)
    arm: ComponentControlSettings | None = None
    hand: ComponentControlSettings | None = None

    def component(self, name: str) -> ComponentControlSettings:
        """根据关节名选择部件配置。

        参数:
            name: articulation DOF 名称。
        返回:
            ``ComponentControlSettings``；未知名称使用 ``default``。
        """

        group = component_for_name(name)
        if group == "arm" and self.arm is not None:
            return self.arm
        if group == "hand" and self.hand is not None:
            return self.hand
        return self.default


@dataclass(frozen=True)
class ControlTargets:
    """完整 DOF 控制目标。

    输入字段:
        positions: 完整 DOF 位置目标，单位 rad。
        velocities: 完整 DOF 速度目标，单位 rad/s。
        efforts: 完整 DOF effort 目标，量纲由 PhysX 关节类型决定。
    输出:
        ``JointController.apply_targets`` 可直接消费该对象。
    """

    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray
