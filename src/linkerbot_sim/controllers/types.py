"""控制器共享数据类型。

这些类型描述项目层面的控制意图，而不是 Isaac runtime 对象本身。配置解析、执行器和控制器
都可以导入它们，不需要依赖具体 controller 实现文件。主动关节支持 position、velocity 和
effort 三类模式；mimic follower 始终使用独立的 position drive 配置。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from linkerbot_sim.robots.classification import component_for_name


ControlMode = Literal["position", "velocity", "effort"]
ControlMethod = Literal["implicit", "explicit", "direct"]
JointParameter = float | tuple[float, ...] | Mapping[str, float]


def resolve_joint_parameter(
    value: JointParameter,
    joint_names: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    """按最终 articulation 关节顺序展开一个关节参数。

    标量广播为 ``(joint_count,)``；序列必须恰好提供相同数量的值；name-map 必须与目标
    名称集合完全一致，并按 ``joint_names`` 顺序重排。返回数组只包含有限 float，调用方
    因而可以直接逐列应用到 articulation，不会发生配置值与 DOF 顺序错位。
    """

    names = tuple(str(name) for name in joint_names)
    # mapping 的插入顺序不具有控制含义，输出顺序只能由 runtime 最终 joint_names 决定。
    if isinstance(value, Mapping):
        configured = set(value)
        expected = set(names)
        unknown = sorted(configured - expected)
        missing = sorted(expected - configured)
        if unknown or missing:
            details: list[str] = []
            if unknown:
                details.append(f"unknown={unknown}")
            if missing:
                details.append(f"missing={missing}")
            raise ValueError(
                f"{label} name-map does not match selected joints: {', '.join(details)}"
            )
        values = np.asarray([float(value[name]) for name in names], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must contain finite values")
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = np.asarray(tuple(float(item) for item in value), dtype=float).reshape(
            -1
        )
    else:
        values = np.asarray((float(value),), dtype=float)
    # 长度为 1 的序列与标量使用同一广播语义；其余序列禁止隐式截断或循环填充。
    if values.size == 1:
        values = np.full(len(names), float(values[0]), dtype=float)
    if values.size != len(names):
        raise ValueError(
            f"{label} expected a scalar or {len(names)} values, got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain finite values")
    return values


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
        follower_stiffness/follower_damping: mimic follower 的 Isaac position drive 增益。
        follower_max_force: follower drive 最大力/力矩；为 ``None`` 时沿用 ``max_force``。
    输出:
        作为 ``JointControlSettings`` 的 arm/hand/default 子配置。
    """

    mode: ControlMode = "position"
    method: ControlMethod = "implicit"
    stiffness: JointParameter = (1000.0,)
    damping: JointParameter = (50.0,)
    max_force: JointParameter = 100.0
    effort_limit: JointParameter | None = None
    follower_stiffness: JointParameter = (50000.0,)
    follower_damping: JointParameter = (50.0,)
    follower_max_force: JointParameter | None = None

    def active_effort_limits(self, joint_names: Sequence[str]) -> np.ndarray:
        """按 selected joint 顺序返回每个主动关节的 effort 限幅向量。"""

        value = self.max_force if self.effort_limit is None else self.effort_limit
        return resolve_joint_parameter(value, joint_names, label="active effort_limit")


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

    def component(
        self, name: str, *, component: str | None = None
    ) -> ComponentControlSettings:
        """根据关节名选择部件配置。

        参数:
            name: articulation DOF 名称。
        返回:
            ``ComponentControlSettings``；未知名称使用 ``default``。
        """

        group = component_for_name(name) if component is None else component
        if group == "arm" and self.arm is not None:
            return self.arm
        if group == "hand" and self.hand is not None:
            return self.hand
        return self.default


@dataclass(frozen=True)
class ControlTargets:
    """完整 DOF 控制目标。

    三个字段必须是一维、shape 完全相同的有限 float 向量；元素顺序均对应同一份完整 DOF
    列表。positions 单位为 rad，velocities 单位为 rad/s，efforts 量纲由 PhysX 关节类型
    决定。

    输入字段:
        positions: 完整 DOF 位置目标。
        velocities: 完整 DOF 速度目标。
        efforts: 完整 DOF effort 目标。
    输出:
        ``JointController.apply_targets`` 可直接消费该对象。
    """

    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray

    def __post_init__(self) -> None:
        """建立控制目标不变量，并复制数组以实现逻辑冻结。

        dataclass 的 ``frozen=True`` 只能禁止字段重新赋值，不能阻止外部持有的 numpy 输入
        原地修改；因此这里统一 reshape、校验相同 shape/有限值，再保存独立副本。成功构造
        后 controller 可把三个数组视为同一 DOF 域上的不可变目标。
        """

        arrays = {
            label: np.asarray(value, dtype=float).reshape(-1)
            for label, value in (
                ("positions", self.positions),
                ("velocities", self.velocities),
                ("efforts", self.efforts),
            )
        }
        expected_shape = arrays["positions"].shape
        for label, array in arrays.items():
            if array.shape != expected_shape:
                raise ValueError(
                    f"ControlTargets.{label} must have shape {expected_shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"ControlTargets.{label} must contain finite values")
            object.__setattr__(self, label, array.copy())
