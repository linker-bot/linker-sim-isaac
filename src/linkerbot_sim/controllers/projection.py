"""控制器 profile 到执行设置和后端通用 USD drive seed 的纯投影。"""

from __future__ import annotations

from linkerbot_sim.assets.usd_overrides import RobotUsdOverrideConfig
from linkerbot_sim.configuration.controllers import (
    ControllerProfile,
    ControllerProfiles,
)
from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlMode,
    JointControlSettings,
)


def _component_settings(
    profile: ControllerProfile, mode: ControlMode
) -> ComponentControlSettings:
    """返回 profile 中已经严格解析的指定控制模式。"""

    if mode == "position":
        return profile.position_control
    if mode == "velocity":
        return profile.velocity_control
    if mode == "effort":
        return profile.effort_control
    raise ValueError(f"Unsupported control mode: {mode!r}")


def joint_control_settings(
    profiles: ControllerProfiles, *, mode: ControlMode = "position"
) -> JointControlSettings:
    """投影 arm/hand profile，并让缺省组显式继承 arm。"""

    arm = _component_settings(profiles.arm, mode)
    default = (
        arm if profiles.default is None else _component_settings(profiles.default, mode)
    )
    return JointControlSettings(
        default=default,
        arm=arm,
        hand=_component_settings(profiles.hand, mode),
    )


def hybrid_force_position_settings(
    profiles: ControllerProfiles,
) -> JointControlSettings:
    """Project direct arm effort while preserving implicit hand position drives."""

    arm = _component_settings(profiles.arm, "effort")
    hand = _component_settings(profiles.hand, "position")
    default_profile = profiles.arm if profiles.default is None else profiles.default
    default = _component_settings(default_profile, "position")
    if (arm.mode, arm.method) != ("effort", "direct"):
        raise ValueError("hybrid arm controller must use effort + direct")
    if (hand.mode, hand.method) != ("position", "implicit"):
        raise ValueError("hybrid hand controller must use position + implicit")
    if (default.mode, default.method) != ("position", "implicit"):
        raise ValueError("hybrid default controller must use position + implicit")
    return JointControlSettings(default=default, arm=arm, hand=hand)


def _usd_joint_parameter(value: object) -> object:
    """将 parser 的单元素广播 tuple 还原为 USD drive 标量。"""

    if isinstance(value, tuple) and len(value) == 1:
        return float(value[0])
    return value


def _robot_usd_override_config(profile: ControllerProfile) -> RobotUsdOverrideConfig:
    # Importer 写入的是 position drive 初值，与运行时选择的主动控制模式无关。
    drive = profile.position_control
    return RobotUsdOverrideConfig(
        drive_stiffness_seed=_usd_joint_parameter(drive.stiffness),
        drive_damping_seed=_usd_joint_parameter(drive.damping),
        follower_drive_stiffness_seed=_usd_joint_parameter(drive.follower_stiffness),
        follower_drive_damping_seed=_usd_joint_parameter(drive.follower_damping),
        max_force=_usd_joint_parameter(drive.max_force),
        follower_max_force=_usd_joint_parameter(drive.follower_max_force),
    )


def robot_usd_override_configs(
    profiles: ControllerProfiles,
) -> dict[str, RobotUsdOverrideConfig]:
    """生成导入后通用 USD drive 使用的 default/arm/hand seed。"""

    arm = _robot_usd_override_config(profiles.arm)
    default = (
        arm
        if profiles.default is None
        else _robot_usd_override_config(profiles.default)
    )
    return {
        "default": default,
        "arm": arm,
        "hand": _robot_usd_override_config(profiles.hand),
    }


__all__ = [
    "hybrid_force_position_settings",
    "joint_control_settings",
    "robot_usd_override_configs",
]
