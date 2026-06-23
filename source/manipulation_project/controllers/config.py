"""控制器 YAML 配置解析。

项目允许把机械臂和灵巧手的控制参数拆成独立文件，再通过一个聚合配置引用。
本模块把这些 YAML 统一解析成 runtime controller 和 USD/PhysX 覆盖所需的数据。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manipulation_project.assets.usd_overrides import PhysxOverrideConfig
from manipulation_project.controllers.implicit_drive_controller import ComponentDriveSettings, ImplicitDriveSettings
from manipulation_project.utils.config import deep_merge, load_yaml
from manipulation_project.utils.paths import repo_path


@dataclass(frozen=True)
class ControllerProfile:
    """单个部件的控制和物理参数。

    输入字段:
        name: ``arm`` 或 ``hand`` 等部件名。
        implicit_position_drive: position drive 参数。
        velocity_control: 速度控制预留参数。
        effort_control: effort 控制预留参数。
        physx: 材料、刚体和导入后 drive 初值覆盖参数。
    输出:
        可转换为 ``ComponentDriveSettings`` 和 ``PhysxOverrideConfig``。
    """

    name: str
    implicit_position_drive: dict[str, Any]
    velocity_control: dict[str, Any]
    effort_control: dict[str, Any]
    physx: dict[str, Any]


@dataclass(frozen=True)
class ControllerProfiles:
    """机械臂和灵巧手控制配置集合。"""

    arm: ControllerProfile
    hand: ControllerProfile


def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Controller section {key!r} must be a mapping")
    return dict(value)


def _path_from_reference(reference: Any) -> Path:
    if not isinstance(reference, (str, Path)):
        raise ValueError(f"Controller config reference must be a path, got {type(reference).__name__}")
    return repo_path(reference)


def load_controller_profiles(path_or_config: str | Path | Mapping[str, Any]) -> ControllerProfiles:
    """读取并展开 arm/hand 控制器配置。

    参数:
        path_or_config: YAML 路径或已经读取出的 mapping。
    返回:
        ``ControllerProfiles``，包含 arm 和 hand 两个 profile。
    """

    config = load_yaml(path_or_config) if isinstance(path_or_config, (str, Path)) else dict(path_or_config)

    profiles = config.get("profiles")
    if isinstance(profiles, Mapping):
        return ControllerProfiles(
            arm=_load_profile_from_reference("arm", profiles.get("arm")),
            hand=_load_profile_from_reference("hand", profiles.get("hand")),
        )

    # 兼容旧版单文件配置：同一套参数同时用于 arm 和 hand。
    legacy = _legacy_profile_mapping(config)
    return ControllerProfiles(
        arm=_profile_from_mapping("arm", legacy),
        hand=_profile_from_mapping("hand", legacy),
    )


def _load_profile_from_reference(name: str, reference: Any) -> ControllerProfile:
    if reference is None:
        raise ValueError(f"Controller profiles must define {name!r}")
    data = load_yaml(_path_from_reference(reference))
    return _profile_from_mapping(name, data)


def _legacy_profile_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    controller = _section(config, "controller") if "controller" in config else dict(config)
    physx = _section(config, "physx")
    active = {
        "stiffness": controller.get("stiffness", 1000.0),
        "damping": controller.get("damping", 50.0),
        "max_force": controller.get("max_force", 100.0),
        "joint_friction": controller.get("joint_friction", 0.5),
    }
    follower = {
        "stiffness": controller.get("follower_stiffness", controller.get("stiffness", 1000.0)),
        "damping": controller.get("follower_damping", controller.get("damping", 50.0)),
        "max_force": controller.get("follower_max_force", controller.get("max_force", 100.0)),
        "joint_friction": controller.get("joint_friction", 0.5),
    }
    return {
        "implicit_position_drive": {
            "active_joints": active,
            "follower_joints": follower,
        },
        "velocity_control": {},
        "effort_control": {},
        "physx": {
            "material": {
                "contact_static_friction": physx.get("contact_static_friction", 0.8),
                "contact_dynamic_friction": physx.get("contact_dynamic_friction", 0.6),
                "contact_restitution": physx.get("contact_restitution", 0.0),
            },
            "rigid_body": {
                "linear_damping": physx.get("rigid_body_linear_damping", 0.0),
                "angular_damping": physx.get("rigid_body_angular_damping", 0.1),
            },
        },
    }


def _profile_from_mapping(name: str, data: Mapping[str, Any]) -> ControllerProfile:
    target = str(data.get("target", name))
    if target != name:
        raise ValueError(f"Controller profile {name!r} has mismatched target {target!r}")
    return ControllerProfile(
        name=name,
        implicit_position_drive=_section(data, "implicit_position_drive"),
        velocity_control=_section(data, "velocity_control"),
        effort_control=_section(data, "effort_control"),
        physx=_section(data, "physx"),
    )


def _joint_section(config: Mapping[str, Any], key: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    return deep_merge(defaults, _section(config, key))


def _component_drive_settings(profile: ControllerProfile) -> ComponentDriveSettings:
    defaults = {
        "stiffness": 1000.0,
        "damping": 50.0,
        "max_force": 100.0,
        "joint_friction": 0.5,
    }
    follower_defaults = {
        "stiffness": 50000.0,
        "damping": 50.0,
        "max_force": 100.0,
        "joint_friction": defaults["joint_friction"],
    }
    active = _joint_section(profile.implicit_position_drive, "active_joints", defaults)
    follower = _joint_section(profile.implicit_position_drive, "follower_joints", follower_defaults)
    return ComponentDriveSettings(
        stiffness=(float(active["stiffness"]),),
        damping=(float(active["damping"]),),
        max_force=float(active["max_force"]),
        joint_friction=float(active["joint_friction"]),
        follower_stiffness=(float(follower["stiffness"]),),
        follower_damping=(float(follower["damping"]),),
        follower_max_force=float(follower["max_force"]),
        follower_joint_friction=float(follower["joint_friction"]),
    )


def implicit_drive_settings(profiles: ControllerProfiles) -> ImplicitDriveSettings:
    """把 arm/hand profile 转成 runtime implicit drive 设置。"""

    return ImplicitDriveSettings(
        default=_component_drive_settings(profiles.arm),
        arm=_component_drive_settings(profiles.arm),
        hand=_component_drive_settings(profiles.hand),
    )


def _physx_override_config(profile: ControllerProfile) -> PhysxOverrideConfig:
    material = _section(profile.physx, "material")
    rigid_body = _section(profile.physx, "rigid_body")
    drive = _component_drive_settings(profile)
    return PhysxOverrideConfig(
        contact_static_friction=float(material.get("contact_static_friction", 0.8)),
        contact_dynamic_friction=float(material.get("contact_dynamic_friction", 0.6)),
        contact_restitution=float(material.get("contact_restitution", 0.0)),
        joint_friction=float(drive.joint_friction),
        follower_joint_friction=float(drive.follower_joint_friction),
        rigid_body_linear_damping=float(rigid_body.get("linear_damping", 0.0)),
        rigid_body_angular_damping=float(rigid_body.get("angular_damping", 0.1)),
        drive_stiffness_seed=float(drive.stiffness[0]),
        drive_damping_seed=float(drive.damping[0]),
        follower_drive_stiffness_seed=float(drive.follower_stiffness[0]),
        follower_drive_damping_seed=float(drive.follower_damping[0]),
        max_force=float(drive.max_force),
        follower_max_force=float(drive.follower_max_force),
    )


def physx_override_configs(profiles: ControllerProfiles) -> dict[str, PhysxOverrideConfig]:
    """把 arm/hand profile 转成 USD/PhysX 覆盖配置。"""

    arm = _physx_override_config(profiles.arm)
    hand = _physx_override_config(profiles.hand)
    return {"default": arm, "arm": arm, "hand": hand}
