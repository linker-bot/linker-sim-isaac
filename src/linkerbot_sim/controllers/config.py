"""控制器 YAML 配置解析。

项目将机械臂和灵巧手的控制、材料、刚体参数拆成独立 YAML 文件。本模块从 controllers
目录或显式 profile 映射读取这些文件，并转换成 runtime controller 和 USD/PhysX 覆盖所需
的数据。

职责边界:
        * 把 YAML 中的 arm/hand profile 解析成纯 Python dataclass。
        * 生成 ``JointControlSettings``（运行时 articulation controller 用）和
            ``PhysxOverrideConfig``（导入后 USD/PhysX 覆盖用）。
        * 不读取机器人 ``dof_names``，不检查每个关节名是否存在；这一步必须等资产导入后完成。

配置约定：arm/hand profile 分别描述一个部件，``position_control``/``velocity_control``/
``effort_control`` 表示不同主动控制模式的配置。每个控制模式内的 ``active_joints`` 面向
上层命令空间，字段含义随主动模式变化；``follower_joints`` 始终面向 mimic 从动关节，
字段含义固定为 Isaac position drive 的 stiffness/damping/max_force/friction。解析阶段
只做类型、缺失字段和 target 名称校验，具体 DOF 长度会在控制器拿到 Isaac articulation 后
再校验。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from linkerbot_sim.assets.usd_overrides import PhysxOverrideConfig
from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlMethod,
    ControlMode,
    JointControlSettings,
)
from linkerbot_sim.utils.config import deep_merge, load_yaml
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class ControllerProfile:
    """单个部件的控制和物理参数。

    输入字段:
        name: ``arm`` 或 ``hand`` 等部件名。
        position_control: 位置控制参数，支持 ``method: implicit`` 或 ``method: explicit``。
        velocity_control: 速度控制参数，支持 ``method: implicit`` 或 ``method: explicit``。
        effort_control: effort 控制参数，当前支持 ``method: direct``。
        physx: 材料、刚体和导入后 drive 初值覆盖参数。
    输出:
        可转换为 ``ComponentControlSettings`` 和 ``PhysxOverrideConfig``。
    """

    name: str
    position_control: dict[str, Any]
    velocity_control: dict[str, Any]
    effort_control: dict[str, Any]
    physx: dict[str, Any]


@dataclass(frozen=True)
class ControllerProfiles:
    """机械臂和灵巧手控制配置集合。

    当前项目显式区分 arm 和 hand 两类 profile。若某个机器人没有灵巧手，调用方仍可提供
    hand profile 作为默认占位；真正的关节存在性由导入后的 articulation/controller 校验。
    """

    arm: ControllerProfile
    hand: ControllerProfile


def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    """读取可选 YAML section，并确保其为 mapping。"""

    # 缺失 section 视为空 mapping，后续与默认值合并；但如果用户显式写成列表/标量，
    # 通常表示 YAML 结构错误，应立即报告。
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Controller section {key!r} must be a mapping")
    return dict(value)


def load_controller_profiles(config_dir: str | Path) -> ControllerProfiles:
    """读取并展开 arm/hand 控制器配置。

    参数:
        config_dir: controllers 目录路径，目录内必须包含 ``arm_controller.yaml`` 和
            ``hand_controller.yaml``。
    返回:
        ``ControllerProfiles``，包含 arm 和 hand 两个 profile。
    """

    if not isinstance(config_dir, (str, Path)):
        raise TypeError(
            f"Controller config must be a directory path, got {type(config_dir).__name__}"
        )
    # controllers 配置以目录为单位加载，保证 arm/hand profile 来自同一套实验参数。
    path = repo_path(config_dir)
    if not path.is_dir():
        raise ValueError(
            f"Controller config must be a directory containing arm_controller.yaml and hand_controller.yaml: {path}"
        )
    return ControllerProfiles(
        arm=_load_profile_from_path("arm", path / "arm_controller.yaml"),
        hand=_load_profile_from_path("hand", path / "hand_controller.yaml"),
    )


def _load_profile_from_path(name: str, path: Path) -> ControllerProfile:
    """从单个 YAML 文件读取并校验指定部件 profile。"""

    if not path.is_file():
        raise FileNotFoundError(f"Controller profile {name!r} was not found: {path}")
    return _profile_from_mapping(name, load_yaml(path))


def _profile_from_mapping(name: str, data: Mapping[str, Any]) -> ControllerProfile:
    """把 YAML mapping 转成 ``ControllerProfile``。"""

    # profile 内的 target 是防呆字段：例如手部配置误命名为 arm_controller.yaml 时能尽早发现。
    target = str(data.get("target", name))
    if target != name:
        raise ValueError(
            f"Controller profile {name!r} has mismatched target {target!r}"
        )
    if "implicit_position_drive" in data:
        raise ValueError(
            f"Controller profile {name!r} uses removed section 'implicit_position_drive'; "
            "use 'position_control' with method: implicit"
        )
    return ControllerProfile(
        name=name,
        position_control=_section(data, "position_control"),
        velocity_control=_section(data, "velocity_control"),
        effort_control=_section(data, "effort_control"),
        physx=_section(data, "physx"),
    )


def _joint_section(
    config: Mapping[str, Any], key: str, defaults: Mapping[str, Any]
) -> dict[str, Any]:
    """读取 active/follower 子配置并与默认值深合并。"""

    return deep_merge(defaults, _section(config, key))


def _control_section(profile: ControllerProfile, mode: ControlMode) -> dict[str, Any]:
    """按控制模式选择 profile section。"""

    if mode == "position":
        return profile.position_control
    if mode == "velocity":
        return profile.velocity_control
    if mode == "effort":
        return profile.effort_control
    raise ValueError(f"Unsupported control mode: {mode!r}")


def _normalize_method(config: Mapping[str, Any], mode: ControlMode) -> ControlMethod:
    """读取并规范化控制方法名称。"""

    raw_method = config.get("method", config.get("type"))
    if raw_method is None and mode == "position":
        raw_method = "implicit"
    if raw_method is None and mode == "velocity":
        raw_method = "implicit"
    if raw_method is None and mode == "effort":
        raw_method = "direct"
    method = str(raw_method)
    if method == "implicit_drive":
        method = "implicit"
    allowed = {
        "position": {"implicit", "explicit"},
        "velocity": {"implicit", "explicit"},
        "effort": {"direct"},
    }[mode]
    if method not in allowed:
        raise ValueError(
            f"{mode}_control method must be one of {sorted(allowed)}, got {method!r}"
        )
    return cast(ControlMethod, method)


def _component_control_settings(
    profile: ControllerProfile, mode: ControlMode
) -> ComponentControlSettings:
    """把单个部件 profile 转成指定模式的 runtime 控制参数。"""

    # active_joints 描述当前 --control-mode 下主动关节的控制参数；当 mode=effort 时，
    # stiffness/damping 只作为默认字段保留，真实输出由 command effort 和 effort_limit 决定。
    control = _control_section(profile, mode)
    method = _normalize_method(control, mode)
    defaults = {
        "stiffness": 1000.0,
        "damping": 50.0,
        "max_force": 100.0,
        "effort_limit": None,
        "joint_friction": 0.5,
    }
    # follower_joints 不随主动模式切换控制语义。无论 active_joints 使用位置、速度还是
    # effort，mimic follower 都使用 Isaac position drive 跟随 master 实际角度，因此这里
    # 总是读取 stiffness/damping/max_force/joint_friction。
    follower_defaults = {
        "stiffness": 50000.0,
        "damping": 50.0,
        "max_force": 100.0,
        "joint_friction": defaults["joint_friction"],
    }
    active = _joint_section(control, "active_joints", defaults)
    follower = _joint_section(control, "follower_joints", follower_defaults)
    effort_limit = active.get("effort_limit")
    return ComponentControlSettings(
        mode=mode,
        method=method,
        stiffness=(float(active["stiffness"]),),
        damping=(float(active["damping"]),),
        max_force=float(active["max_force"]),
        effort_limit=None if effort_limit is None else float(effort_limit),
        joint_friction=float(active["joint_friction"]),
        follower_stiffness=(float(follower["stiffness"]),),
        follower_damping=(float(follower["damping"]),),
        follower_max_force=float(follower["max_force"]),
        follower_joint_friction=float(follower["joint_friction"]),
    )


def joint_control_settings(
    profiles: ControllerProfiles, *, mode: ControlMode = "position"
) -> JointControlSettings:
    """把 arm/hand profile 转成指定模式的 runtime 关节控制设置。

    返回值包含 ``default``、``arm`` 和 ``hand`` 三组参数。``default`` 当前沿用 arm profile，
    用于未知命名关节的保守回退。
    """

    return JointControlSettings(
        default=_component_control_settings(profiles.arm, mode),
        arm=_component_control_settings(profiles.arm, mode),
        hand=_component_control_settings(profiles.hand, mode),
    )


def _physx_override_config(profile: ControllerProfile) -> PhysxOverrideConfig:
    """把单个部件 profile 转成导入后 USD/PhysX 覆盖参数。"""

    # USD/PhysX 覆盖只需要导入后的默认 drive seed。这里固定从 position_control 读取，
    # 因为 Isaac importer 写入的是 joint drive 初值，而不是运行期的速度/effort action。
    material = _section(profile.physx, "material")
    rigid_body = _section(profile.physx, "rigid_body")
    drive = _component_control_settings(profile, "position")
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


def physx_override_configs(
    profiles: ControllerProfiles,
) -> dict[str, PhysxOverrideConfig]:
    """把 arm/hand profile 转成 USD/PhysX 覆盖配置。

    覆盖配置用于资产导入后的 USD/PhysX schema，不等同于运行时 controller action。
    ``default`` 同样沿用 arm 配置，供未知或未分类刚体使用。
    """

    arm = _physx_override_config(profiles.arm)
    hand = _physx_override_config(profiles.hand)
    return {"default": arm, "arm": arm, "hand": hand}
