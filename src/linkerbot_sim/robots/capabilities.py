"""机器人组件类型与 cuRobo planning 能力的纯配置诊断。

simulation asset 与 planning model 虽然声明在同一个 robot profile 中，但关节集合、frame 和
碰撞模型并不天然等价。本模块只解析显式 ``robot.kind``/``curobo`` binding，并保存完成
articulation joint mapping 后的能力结果；它不导入 Isaac 或 cuRobo，也不把“可控制关节”误判
为“可规划”。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class RobotKind(str, Enum):
    """一个 articulation 实际包含的可动组件类型。"""

    ARM = "arm"
    HAND = "hand"
    ARM_HAND = "arm_hand"

    @classmethod
    def parse(cls, value: object) -> "RobotKind":
        """解析 profile 值，并为非法枚举生成稳定配置错误。"""

        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"robot.kind must be one of: {supported}; got {value!r}"
            ) from exc

    @property
    def has_arm(self) -> bool:
        """返回该 kind 是否要求非空 arm joint group。"""

        return self in {RobotKind.ARM, RobotKind.ARM_HAND}

    @property
    def has_hand(self) -> bool:
        """返回该 kind 是否要求非空 hand joint group。"""

        return self in {RobotKind.HAND, RobotKind.ARM_HAND}


@dataclass(frozen=True)
class PlanningBindingConfig:
    """从 robot profile 读取的静态 cuRobo binding 开关和模型存在性。"""

    enabled: bool
    planning_joint_group: str | None
    has_robot_model: bool

    @classmethod
    def from_profile(
        cls,
        profile: Mapping[str, object],
        *,
        kind: RobotKind,
    ) -> "PlanningBindingConfig":
        """解析 ``curobo.enabled`` 与 model 存在性，不构造 backend config。"""

        raw = profile.get("curobo")
        if not isinstance(raw, Mapping):
            raise ValueError("robot profile requires a curobo mapping")

        robot_model = raw.get("robot")
        has_model = isinstance(robot_model, Mapping) and bool(robot_model)
        if "enabled" not in raw:
            raise ValueError("curobo.enabled is required")
        enabled_value = raw["enabled"]
        if not isinstance(enabled_value, bool):
            raise ValueError("curobo.enabled must be a boolean")
        enabled = enabled_value

        if enabled and "planning_joint_group" not in raw:
            raise ValueError(
                "curobo.planning_joint_group is required when planning is enabled"
            )
        group_value = raw.get("planning_joint_group")
        group = None if group_value is None else str(group_value).strip().lower()
        if enabled and not group:
            raise ValueError(
                "curobo.planning_joint_group is required when planning is enabled"
            )
        if group not in {None, "arm"}:
            raise ValueError("curobo.planning_joint_group must be 'arm'")
        if kind is RobotKind.HAND and enabled:
            raise ValueError("robot.kind 'hand' cannot enable cuRobo planning")
        if enabled and not has_model:
            raise ValueError("curobo.enabled requires a non-empty curobo.robot model")
        return cls(enabled, group, has_model)


@dataclass(frozen=True)
class PlanningCapability:
    """articulation finalize、joint group 校验完成后的 planning 能力。

    ``supports_planning`` 只表示普通 IK/path planning 可进入后端。场景 checker、collision
    sphere、cache 容量和 scene version 属于请求时能力，由 context 的 collision capability
    单独判断。
    """

    kind: RobotKind
    backend_enabled: bool
    planning_joint_group: str | None
    kinematics_binding_valid: bool
    arm_joint_mapping_valid: bool
    reasons: tuple[str, ...] = ()

    @property
    def supports_planning(self) -> bool:
        """返回 arm、backend、kinematics 与 joint mapping 是否全部满足。"""

        return bool(
            self.kind.has_arm
            and self.backend_enabled
            and self.kinematics_binding_valid
            and self.arm_joint_mapping_valid
        )

    def require(self, operation: str = "planning") -> None:
        """在分配 backend/GPU 资源前检查能力，并抛出可诊断错误。"""

        if self.supports_planning:
            return
        details = ", ".join(self.reasons or self._derived_reasons())
        raise RuntimeError(f"{operation} is not supported: {details}")

    def _derived_reasons(self) -> tuple[str, ...]:
        """从 capability flags 推导稳定、可序列化的失败原因。"""

        reasons: list[str] = []
        if not self.kind.has_arm:
            reasons.append(f"robot kind {self.kind.value!r} has no arm")
        if not self.backend_enabled:
            reasons.append("curobo.enabled is false")
        if not self.kinematics_binding_valid:
            reasons.append("planning model binding is invalid")
        if not self.arm_joint_mapping_valid:
            reasons.append("planning joints do not match the arm joint group")
        return tuple(reasons)


def robot_kind_from_profile(
    profile: Mapping[str, object],
) -> RobotKind:
    """读取必需的 canonical ``robot.kind`` 值。"""

    robot = profile.get("robot")
    if not isinstance(robot, Mapping):
        raise ValueError("Robot config must contain top-level robot section")
    if "kind" not in robot:
        raise ValueError("robot.kind is required")
    return RobotKind.parse(robot["kind"])


__all__ = [
    "PlanningBindingConfig",
    "PlanningCapability",
    "RobotKind",
    "robot_kind_from_profile",
]
