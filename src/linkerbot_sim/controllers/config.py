"""控制器 YAML 配置解析。

项目将机械臂和灵巧手的控制参数拆成独立 YAML 文件。本模块从 controllers 目录读取这些
文件，并转换成 runtime controller 和 USD/PhysX drive seed 所需的数据。机器人接触材质和
刚体阻尼属于资产物理属性，由 robot YAML 的 ``robot.physics.physx`` 描述。

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Literal, cast, overload

from linkerbot_sim.assets.usd_overrides import PhysxOverrideConfig
from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlMethod,
    ControlMode,
    JointParameter,
    JointControlSettings,
)
from linkerbot_sim.utils.config import deep_merge, load_yaml
from linkerbot_sim.utils.paths import CONFIGS_ROOT, repo_path


CONTROLLER_BUNDLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ControllerProfile:
    """单个部件的控制参数。

    输入字段:
        name: ``arm`` 或 ``hand`` 等部件名。
        position_control: 位置控制参数，支持 ``method: implicit`` 或 ``method: explicit``。
        velocity_control: 速度控制参数，支持 ``method: implicit`` 或 ``method: explicit``。
        effort_control: effort 控制参数，当前支持 ``method: direct``。
    输出:
        可转换为 ``ComponentControlSettings`` 和 ``PhysxOverrideConfig``。
    """

    name: str
    position_control: dict[str, Any]
    velocity_control: dict[str, Any]
    effort_control: dict[str, Any]


@dataclass(frozen=True)
class ControllerProfiles:
    """机械臂和灵巧手控制配置集合。

    当前项目显式区分 arm 和 hand 两类 profile。若某个机器人没有灵巧手，调用方仍可提供
    hand profile 作为默认占位；真正的关节存在性由导入后的 articulation/controller 校验。
    """

    arm: ControllerProfile
    hand: ControllerProfile
    default: ControllerProfile | None = None


def _section(
    data: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    """读取可选 YAML section，并确保其为 mapping。"""

    # 缺失 section 视为空 mapping，后续与默认值合并；但如果用户显式写成列表/标量，
    # 通常表示 YAML 结构错误，应立即报告。
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
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
        default=(
            _load_profile_from_path("default", path / "default_controller.yaml")
            if (path / "default_controller.yaml").is_file()
            else None
        ),
    )


def load_controller_bundle(
    name: str,
    *,
    controllers_root: str | Path = CONFIGS_ROOT / "controllers",
) -> ControllerProfiles:
    """按安全的 bundle 名读取 ``configs/controllers/<name>``。"""

    bundle_name = normalize_controller_bundle_name(name, label="controller bundle name")
    root = repo_path(controllers_root)
    bundle_dir = root / bundle_name
    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"Controller bundle {bundle_name!r} was not found: {bundle_dir}"
        )
    if not bundle_dir.is_dir():
        raise ValueError(f"Controller bundle must be a directory: {bundle_dir}")
    return load_controller_profiles(bundle_dir)


def normalize_controller_bundle_name(value: object, *, label: str) -> str:
    """校验 controller bundle 为不能逃逸配置根目录的简单名称。"""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    name = value.strip()
    if CONTROLLER_BUNDLE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"{label} must match [A-Za-z0-9][A-Za-z0-9_-]*, got {value!r}")
    return name


def _load_profile_from_path(name: str, path: Path) -> ControllerProfile:
    """从单个 YAML 文件读取并校验指定部件 profile。"""

    if not path.is_file():
        raise FileNotFoundError(f"Controller profile {name!r} was not found: {path}")
    return _profile_from_mapping(name, load_yaml(path), source_path=path)


def _profile_from_mapping(
    name: str,
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ControllerProfile:
    """把 YAML mapping 转成 ``ControllerProfile``。"""

    source = str(source_path) if source_path is not None else "<mapping>"
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: controller profile must be a mapping")
    canonical = dict(data)
    profile_label = f"{source}: controller[{name}]"
    _reject_unknown_keys(
        canonical,
        {
            "target",
            "position_control",
            "velocity_control",
            "effort_control",
        },
        label=profile_label,
    )
    # profile 内的 target 是防呆字段：例如手部配置误命名为 arm_controller.yaml 时能尽早发现。
    target = canonical.get("target", name)
    if not isinstance(target, str):
        raise ValueError(f"{profile_label}.target must be a string")
    if target != name:
        raise ValueError(
            f"Controller profile {name!r} has mismatched target {target!r}"
        )
    profile = ControllerProfile(
        name=name,
        position_control=_section(
            canonical,
            "position_control",
            label=profile_label,
        ),
        velocity_control=_section(
            canonical,
            "velocity_control",
            label=profile_label,
        ),
        effort_control=_section(
            canonical,
            "effort_control",
            label=profile_label,
        ),
    )
    # Profile loading is the schema boundary. Validate every mode eagerly so a typo or
    # invalid gain cannot remain latent until a different runtime control mode is selected.
    for mode in ("position", "velocity", "effort"):
        _component_control_settings(profile, cast(ControlMode, mode))
    return profile


def _reject_unknown_keys(
    data: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    """拒绝映射中的未知键，并在错误中给出其完整点分路径。"""

    unsupported = sorted(str(key) for key in data if key not in allowed)
    if unsupported:
        keys = ", ".join(unsupported)
        paths = ", ".join(f"{label}.{key}" for key in unsupported)
        raise ValueError(
            f"{label} contains unsupported keys: {keys} (full paths: {paths})"
        )


def _joint_section(
    config: Mapping[str, Any],
    key: str,
    defaults: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """读取 active/follower 子配置并与默认值深合并。"""

    section = _section(config, key, label=label)
    _reject_unknown_keys(section, set(defaults), label=f"{label}.{key}")
    return deep_merge(defaults, section)


def _control_section(profile: ControllerProfile, mode: ControlMode) -> dict[str, Any]:
    """按控制模式选择 profile section。"""

    if mode == "position":
        result = profile.position_control
    elif mode == "velocity":
        result = profile.velocity_control
    elif mode == "effort":
        result = profile.effort_control
    else:
        raise ValueError(f"Unsupported control mode: {mode!r}")
    _reject_unknown_keys(
        result,
        {"method", "active_joints", "follower_joints"},
        label=f"controller[{profile.name}].{mode}_control",
    )
    return result


def _normalize_method(config: Mapping[str, Any], mode: ControlMode) -> ControlMethod:
    """读取并规范化控制方法名称。"""

    raw_method = config.get("method")
    if raw_method is None and mode == "position":
        raw_method = "implicit"
    if raw_method is None and mode == "velocity":
        raw_method = "implicit"
    if raw_method is None and mode == "effort":
        raw_method = "direct"
    if not isinstance(raw_method, str):
        raise ValueError(f"{mode}_control.method must be a string")
    method = raw_method
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
    section_label = f"controller[{profile.name}].{mode}_control"
    active = _joint_section(control, "active_joints", defaults, label=section_label)
    follower = _joint_section(
        control, "follower_joints", follower_defaults, label=section_label
    )
    effort_limit = _joint_parameter(
        active.get("effort_limit"),
        label=f"{section_label}.active_joints.effort_limit",
        nullable=True,
    )
    return ComponentControlSettings(
        mode=mode,
        method=method,
        stiffness=_joint_parameter(
            active["stiffness"],
            label=f"{section_label}.active_joints.stiffness",
            scalar_as_tuple=True,
        ),
        damping=_joint_parameter(
            active["damping"],
            label=f"{section_label}.active_joints.damping",
            scalar_as_tuple=True,
        ),
        max_force=_joint_parameter(
            active["max_force"], label=f"{section_label}.active_joints.max_force"
        ),
        effort_limit=effort_limit,
        joint_friction=_joint_parameter(
            active["joint_friction"],
            label=f"{section_label}.active_joints.joint_friction",
        ),
        follower_stiffness=_joint_parameter(
            follower["stiffness"],
            label=f"{section_label}.follower_joints.stiffness",
            scalar_as_tuple=True,
        ),
        follower_damping=_joint_parameter(
            follower["damping"],
            label=f"{section_label}.follower_joints.damping",
            scalar_as_tuple=True,
        ),
        follower_max_force=_joint_parameter(
            follower["max_force"],
            label=f"{section_label}.follower_joints.max_force",
        ),
        follower_joint_friction=_joint_parameter(
            follower["joint_friction"],
            label=f"{section_label}.follower_joints.joint_friction",
        ),
    )


def joint_control_settings(
    profiles: ControllerProfiles, *, mode: ControlMode = "position"
) -> JointControlSettings:
    """把 arm/hand profile 转成指定模式的 runtime 关节控制设置。

    返回值包含 ``default``、``arm`` 和 ``hand`` 三组参数。``default`` 当前沿用 arm profile，
    用于未知命名关节的保守回退。
    """

    default_profile = profiles.default or profiles.arm
    return JointControlSettings(
        default=_component_control_settings(default_profile, mode),
        arm=_component_control_settings(profiles.arm, mode),
        hand=_component_control_settings(profiles.hand, mode),
    )


def _physx_override_config(profile: ControllerProfile) -> PhysxOverrideConfig:
    """把单个部件 profile 转成导入后 USD/PhysX 覆盖参数。"""

    # USD/PhysX 覆盖只需要导入后的默认 drive seed。这里固定从 position_control 读取，
    # 因为 Isaac importer 写入的是 joint drive 初值，而不是运行期的速度/effort action。
    drive = _component_control_settings(profile, "position")
    return PhysxOverrideConfig(
        joint_friction=_physx_joint_parameter(drive.joint_friction),
        follower_joint_friction=_physx_joint_parameter(drive.follower_joint_friction),
        drive_stiffness_seed=_physx_joint_parameter(drive.stiffness),
        drive_damping_seed=_physx_joint_parameter(drive.damping),
        follower_drive_stiffness_seed=_physx_joint_parameter(drive.follower_stiffness),
        follower_drive_damping_seed=_physx_joint_parameter(drive.follower_damping),
        max_force=_physx_joint_parameter(drive.max_force),
        follower_max_force=_physx_joint_parameter(drive.follower_max_force),
    )


def physx_override_configs(
    profiles: ControllerProfiles,
) -> dict[str, PhysxOverrideConfig]:
    """把 arm/hand profile 转成 USD/PhysX 覆盖配置。

    覆盖配置用于资产导入后的 USD/PhysX schema，不等同于运行时 controller action。
    ``default`` 同样沿用 arm 配置，供未知或未分类刚体使用。
    """

    default = _physx_override_config(profiles.default or profiles.arm)
    arm = _physx_override_config(profiles.arm)
    hand = _physx_override_config(profiles.hand)
    return {"default": default, "arm": arm, "hand": hand}


@overload
def _joint_parameter(
    value: object,
    *,
    label: str,
    nullable: Literal[False] = False,
    scalar_as_tuple: bool = False,
) -> JointParameter: ...


@overload
def _joint_parameter(
    value: object,
    *,
    label: str,
    nullable: Literal[True],
    scalar_as_tuple: bool = False,
) -> JointParameter | None: ...


def _joint_parameter(
    value: object,
    *,
    label: str,
    nullable: bool = False,
    scalar_as_tuple: bool = False,
) -> JointParameter | None:
    """把标量、列表或关节名映射解析为不可变的数值参数。

    所有值必须非负且有限；映射键会去除首尾空白，空映射、空序列和空关节名均被拒绝。
    """

    if value is None:
        if nullable:
            return None
        raise ValueError(f"{label} cannot be null")
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{label} name-map cannot be empty")
        result: dict[str, float] = {}
        for name, item in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{label} contains an invalid joint name")
            result[name.strip()] = _non_negative_finite_number(item, f"{label}.{name}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{label} sequence cannot be empty")
        return tuple(
            _non_negative_finite_number(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    parsed = _non_negative_finite_number(value, label)
    return (parsed,) if scalar_as_tuple else parsed


def _physx_joint_parameter(value: object) -> object:
    """将单元素元组还原为 PhysX 标量，同时保留向量和关节名映射。"""

    if isinstance(value, tuple) and len(value) == 1:
        return float(value[0])
    return value


def _non_negative_finite_number(value: object, label: str) -> float:
    """解析非负有限控制器数值，拒绝布尔值、非数值、负数和无穷值。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed
