"""``configs/controllers`` 的唯一 strict mapping schema。

项目将机械臂和灵巧手的控制参数拆成独立 YAML 文件。这些文件由 catalog 从
controllers 目录读取；本模块只把已读取 mapping 转换成后端无关的冻结 profile。
机器人接触材质和刚体阻尼属于资产物理属性。显式选择的 Newton 柔顺驱动 bundle
额外拥有虚拟电机、armature、被动阻尼及主从约束参数，不修改接触配置。

职责边界:
        * 把 YAML 中的 arm/hand profile 解析成纯 Python dataclass。
        * 不生成运行时 controller 或 USD/PhysX 对象；投影由 ``controllers.projection`` 完成。
        * 不读取机器人 ``dof_names``，不检查每个关节名是否存在；这一步必须等资产导入后完成。

配置约定：arm/hand profile 分别描述一个部件，``position_control``/``velocity_control``/
``effort_control`` 表示不同主动控制模式的配置。每个控制模式内的 ``active_joints`` 面向
上层命令空间，字段含义随主动模式变化；``follower_joints`` 始终面向 mimic 从动关节，
字段含义固定为 Isaac position drive 的 stiffness/damping/max_force。关节摩擦属于
``robot.physics.physx.joint``，不由控制器 profile 重复拥有。解析阶段
只做类型、缺失字段和 target 名称校验，具体 DOF 长度会在控制器拿到 Isaac articulation 后
再校验。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from numbers import Real
import re
from types import MappingProxyType
from typing import Any, cast

from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlMethod,
    ControlMode,
    JointParameter,
    resolve_joint_parameter,
)

CONTROLLER_BUNDLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# 主动关节字段按控制语义判别。未被某种控制律消费的字段不进入该 section，避免配置中出现
# 看似可调、实际无效的参数；被消费的字段则全部必填，parser 不提供任何数值回退。
_ACTIVE_FIELDS_BY_MODE: dict[ControlMode, frozenset[str]] = {
    "position": frozenset({"stiffness", "damping", "max_force"}),
    "velocity": frozenset({"damping", "max_force"}),
    "effort": frozenset({"effort_limit"}),
}
_FOLLOWER_FIELDS = frozenset({"stiffness", "damping", "max_force"})


@dataclass(frozen=True)
class ControllerProfile:
    """单个部件的控制参数。

    输入字段:
        name: ``arm`` 或 ``hand`` 等部件名。
        position_control: 已严格解析的位置控制参数。
        velocity_control: 已严格解析的速度控制参数。
        effort_control: 已严格解析的 effort 控制参数。
    输出:
        由运行时 projection 转换为 controller 与 PhysX 所需设置。
    """

    name: str
    position_control: ComponentControlSettings
    velocity_control: ComponentControlSettings
    effort_control: ComponentControlSettings

    def __post_init__(self) -> None:
        for name in ("position_control", "velocity_control", "effort_control"):
            if not isinstance(getattr(self, name), ComponentControlSettings):
                raise TypeError(f"ControllerProfile.{name} must be parsed settings")


@dataclass(frozen=True)
class ControllerProfiles:
    """机械臂和灵巧手控制配置集合。

    当前项目显式区分 arm 和 hand 两类 profile。若某个机器人没有灵巧手，调用方仍可提供
    hand profile 作为默认占位；真正的关节存在性由导入后的 articulation/controller 校验。
    """

    arm: ControllerProfile
    hand: ControllerProfile
    default: ControllerProfile | None = None
    compliant_drive: CompliantDriveSettings | None = None


def validate_compliant_drive_selection(
    profiles: ControllerProfiles,
    *,
    engine: str,
    asset_path: str,
    mode: str = "position",
) -> None:
    """Reject unsupported use before starting Kit or importing a physical asset."""
    if profiles.compliant_drive is None:
        return
    from pathlib import Path

    unit = Path(asset_path).parent.name
    if engine != "newton" or mode != "position":
        raise ValueError("Compliant drive requires Newton position control")
    if not unit.startswith(("ar5_08_o6_", "ar5_o6_")):
        raise ValueError(f"Compliant drive supports only AR5/O6 assemblies, got {unit}")
    if any(
        (p.position_control.mode, p.position_control.method) != ("position", "implicit")
        for p in (profiles.arm, profiles.hand)
    ):
        raise ValueError("Compliant drive requires implicit native position outputs")


def _required_keys(
    data: Mapping[str, Any],
    required: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    """报告 strict mapping 中缺失的必填字段。"""

    missing = sorted(required - set(data))
    if not missing:
        return
    if len(missing) == 1:
        raise ValueError(f"{label}.{missing[0]} is required")
    fields = ", ".join(missing)
    paths = ", ".join(f"{label}.{field}" for field in missing)
    raise ValueError(
        f"{label} is missing required fields: {fields} (full paths: {paths})"
    )


def _required_section(
    data: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    """读取必填 YAML section，并确保其为 mapping。"""

    if key not in data:
        raise ValueError(f"{label}.{key} is required")
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return dict(value)


def normalize_controller_bundle_name(value: object, *, label: str) -> str:
    """校验 controller bundle 为不能逃逸配置根目录的简单名称。"""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    name = value.strip()
    if CONTROLLER_BUNDLE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"{label} must match [A-Za-z0-9][A-Za-z0-9_-]*, got {value!r}")
    return name


def controller_profile_from_mapping(
    name: str,
    data: Mapping[str, Any],
    *,
    source: str = "<mapping>",
) -> ControllerProfile:
    """把 YAML mapping 转成 ``ControllerProfile``。"""

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
    _required_keys(
        canonical,
        {"target", "position_control", "velocity_control", "effort_control"},
        label=profile_label,
    )
    # profile 内的 target 是防呆字段：例如手部配置误命名为 arm_controller.yaml 时能尽早发现。
    target = canonical["target"]
    if not isinstance(target, str):
        raise ValueError(f"{profile_label}.target must be a string")
    if target != name:
        raise ValueError(f"{profile_label}.target must equal {name!r}, got {target!r}")
    position = _required_section(
        canonical,
        "position_control",
        label=profile_label,
    )
    velocity = _required_section(
        canonical,
        "velocity_control",
        label=profile_label,
    )
    effort = _required_section(
        canonical,
        "effort_control",
        label=profile_label,
    )
    return ControllerProfile(
        name=name,
        position_control=_parse_component_control_settings(
            profile_label,
            position,
            "position",
        ),
        velocity_control=_parse_component_control_settings(
            profile_label,
            velocity,
            "velocity",
        ),
        effort_control=_parse_component_control_settings(
            profile_label,
            effort,
            "effort",
        ),
    )


def controller_profiles_from_mappings(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    source: str = "<controller bundle>",
) -> ControllerProfiles:
    """组合 catalog 已读取的 arm/hand/default profile 文档。"""

    unknown = sorted(set(profiles) - {"arm", "hand", "default", "newton_drive"})
    missing = sorted({"arm", "hand"} - set(profiles))
    if missing:
        raise ValueError(f"{source}: missing controller profiles: {missing}")
    if unknown:
        raise ValueError(f"{source}: unsupported controller profiles: {unknown}")
    return ControllerProfiles(
        compliant_drive=(
            compliant_drive_from_mapping(
                profiles["newton_drive"], label=f"{source}/newton_drive.yaml"
            )
            if "newton_drive" in profiles
            else None
        ),
        arm=controller_profile_from_mapping(
            "arm", profiles["arm"], source=f"{source}/arm_controller.yaml"
        ),
        hand=controller_profile_from_mapping(
            "hand", profiles["hand"], source=f"{source}/hand_controller.yaml"
        ),
        default=(
            controller_profile_from_mapping(
                "default",
                profiles["default"],
                source=f"{source}/default_controller.yaml",
            )
            if "default" in profiles
            else None
        ),
    )


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
    required_fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    """读取 active/follower strict mapping，不补齐任何控制参数。"""

    section = _required_section(config, key, label=label)
    section_label = f"{label}.{key}"
    _reject_unknown_keys(section, required_fields, label=section_label)
    _required_keys(section, required_fields, label=section_label)
    return section


def _normalize_method(
    config: Mapping[str, Any], mode: ControlMode, *, label: str
) -> ControlMethod:
    """读取并规范化控制方法名称。"""

    if "method" not in config:
        raise ValueError(f"{label}.method is required")
    raw_method = config["method"]
    if not isinstance(raw_method, str):
        raise ValueError(f"{label}.method must be a string")
    method = raw_method
    allowed = {
        "position": {"implicit", "explicit"},
        "velocity": {"implicit", "explicit"},
        "effort": {"direct"},
    }[mode]
    if method not in allowed:
        raise ValueError(
            f"{label}.method must be one of {sorted(allowed)}, got {method!r}"
        )
    return cast(ControlMethod, method)


def _parse_component_control_settings(
    profile_label: str,
    control: Mapping[str, Any],
    mode: ControlMode,
) -> ComponentControlSettings:
    """把一个 YAML 控制段解析为不可变 runtime 设置，并保留来源路径。"""

    # active_joints 只声明当前 mode 实际消费的字段；内部数据类型需要的非适用 gain 使用 0，
    # 这是控制律的确定性投影，不是可覆盖的配置默认值。
    section_label = f"{profile_label}.{mode}_control"
    _reject_unknown_keys(
        control,
        {"method", "active_joints", "follower_joints"},
        label=section_label,
    )
    _required_keys(
        control,
        {"method", "active_joints", "follower_joints"},
        label=section_label,
    )
    method = _normalize_method(control, mode, label=section_label)
    # follower_joints 不随主动模式切换控制语义。无论 active_joints 使用位置、速度还是
    # effort，mimic follower 都使用 Isaac position drive 跟随 master 实际角度，因此这里
    # 总是读取 stiffness/damping/max_force。关节摩擦是 PhysX 资产属性，唯一配置入口为
    # ``robot.physics.physx.joint``，不能由控制模式 profile 重复拥有。
    active = _joint_section(
        control,
        "active_joints",
        _ACTIVE_FIELDS_BY_MODE[mode],
        label=section_label,
    )
    follower = _joint_section(
        control, "follower_joints", _FOLLOWER_FIELDS, label=section_label
    )
    if mode == "position":
        stiffness = _joint_parameter(
            active["stiffness"],
            label=f"{section_label}.active_joints.stiffness",
            scalar_as_tuple=True,
        )
        damping = _joint_parameter(
            active["damping"],
            label=f"{section_label}.active_joints.damping",
            scalar_as_tuple=True,
        )
        max_force = _joint_parameter(
            active["max_force"],
            label=f"{section_label}.active_joints.max_force",
        )
        effort_limit = None
    elif mode == "velocity":
        stiffness = (0.0,)
        damping = _joint_parameter(
            active["damping"],
            label=f"{section_label}.active_joints.damping",
            scalar_as_tuple=True,
        )
        max_force = _joint_parameter(
            active["max_force"],
            label=f"{section_label}.active_joints.max_force",
        )
        effort_limit = None
    else:
        stiffness = (0.0,)
        damping = (0.0,)
        effort_limit = _joint_parameter(
            active["effort_limit"],
            label=f"{section_label}.active_joints.effort_limit",
        )
        # ``max_force`` 在 direct mode 中不参与控制；保持与唯一显式限幅相同，避免
        # ComponentControlSettings 内出现第二份、来源不明的数值。
        max_force = effort_limit
    return ComponentControlSettings(
        mode=mode,
        method=method,
        stiffness=stiffness,
        damping=damping,
        max_force=max_force,
        effort_limit=effort_limit,
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
    )


def _joint_parameter(
    value: object,
    *,
    label: str,
    scalar_as_tuple: bool = False,
) -> JointParameter:
    """把标量、列表或关节名映射解析为不可变的数值参数。

    所有值必须非负且有限；映射键会去除首尾空白，空映射、空序列和空关节名均被拒绝。
    """

    if value is None:
        raise ValueError(f"{label} cannot be null")
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{label} name-map cannot be empty")
        result: dict[str, float] = {}
        for name, item in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{label} contains an invalid joint name")
            result[name.strip()] = _non_negative_finite_number(item, f"{label}.{name}")
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{label} sequence cannot be empty")
        return tuple(
            _non_negative_finite_number(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    parsed = _non_negative_finite_number(value, label)
    return (parsed,) if scalar_as_tuple else parsed


def _non_negative_finite_number(value: object, label: str) -> float:
    """解析非负有限控制器数值，拒绝布尔值、非数值、负数和无穷值。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


__all__ = [
    "ControllerProfile",
    "ControllerProfiles",
    "controller_profile_from_mapping",
    "controller_profiles_from_mappings",
    "normalize_controller_bundle_name",
]


@dataclass(frozen=True)
class VirtualMotorSettings:
    advancing_torque: float
    viscosity: float
    output_damping: float
    passive_damping: float
    reflected_passive_damping: float
    lower: float
    upper: float


@dataclass(frozen=True)
class CompliantDriveSettings:
    arm_armature: float
    arm_passive_damping: float
    arm_force_guard: float
    hand_armature: float
    follower_damping: float
    motor_stiffness: float
    motor_speed_limit: float
    thumb_yaw: VirtualMotorSettings
    thumb_flexion: VirtualMotorSettings
    finger_flexion: VirtualMotorSettings
    equality_solref: tuple[float, float]
    equality_solimp: tuple[float, float, float, float, float]


def _mapping(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} requires exactly {sorted(keys)}")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(
            f"{label} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return number


def _motor(value: object, label: str) -> VirtualMotorSettings:
    fields = set(VirtualMotorSettings.__dataclass_fields__)
    mapping = _mapping(value, fields, label)
    values = {
        key: _number(
            item,
            f"{label}.{key}",
            positive=key in {"advancing_torque", "viscosity", "upper"},
        )
        for key, item in mapping.items()
    }
    if values["lower"] >= values["upper"]:
        raise ValueError(f"{label}.lower must be below upper")
    return VirtualMotorSettings(**values)


def compliant_drive_from_mapping(
    value: object, *, label: str
) -> CompliantDriveSettings:
    fields = set(CompliantDriveSettings.__dataclass_fields__)
    mapping = _mapping(value, fields, label)
    motors = {
        name: _motor(mapping[name], f"{label}.{name}")
        for name in ("thumb_yaw", "thumb_flexion", "finger_flexion")
    }
    scalars = {
        key: _number(
            mapping[key],
            f"{label}.{key}",
            positive=key
            in {
                "arm_armature",
                "arm_force_guard",
                "hand_armature",
                "motor_stiffness",
                "motor_speed_limit",
            },
        )
        for key in fields - motors.keys() - {"equality_solref", "equality_solimp"}
    }
    solref = mapping["equality_solref"]
    solimp = mapping["equality_solimp"]
    if not isinstance(solref, (list, tuple)) or len(solref) != 2:
        raise ValueError(f"{label}.equality_solref requires two positive numbers")
    ref = tuple(_number(x, f"{label}.equality_solref", positive=True) for x in solref)
    if not isinstance(solimp, (list, tuple)) or len(solimp) != 5:
        raise ValueError(f"{label}.equality_solimp requires five numbers")
    imp = tuple(_number(x, f"{label}.equality_solimp") for x in solimp)
    if not (0 < imp[0] <= imp[1] < 1 and imp[2] > 0 and 0 < imp[3] < 1 and imp[4] >= 1):
        raise ValueError(f"{label}.equality_solimp has invalid impedance parameters")
    return CompliantDriveSettings(
        **scalars,
        thumb_yaw=motors["thumb_yaw"],
        thumb_flexion=motors["thumb_flexion"],
        finger_flexion=motors["finger_flexion"],
        equality_solref=(ref[0], ref[1]),
        equality_solimp=(imp[0], imp[1], imp[2], imp[3], imp[4]),
    )


def bind_compliant_drive_joints(
    profiles: ControllerProfiles, joint_groups
) -> ControllerProfiles:
    """Bind the opt-in profile's vectors to declared arm/hand names before import.

    O6 USD traversal and articulation orders differ. Declared group order is the
    new profile's explicit configuration order; exact name maps survive both.
    Ordinary bundles retain their existing runtime-order semantics.
    """
    if profiles.compliant_drive is None:
        return profiles
    components = {}
    for group in ("arm", "hand"):
        names = tuple(getattr(joint_groups, group))
        component = getattr(profiles, group)
        modes = {}
        for mode in ("position_control", "velocity_control", "effort_control"):
            settings = getattr(component, mode)
            values = {}
            for field in ("stiffness", "damping", "max_force", "effort_limit"):
                value = getattr(settings, field)
                if value is not None:
                    resolved = resolve_joint_parameter(
                        value, names, label=f"{group}.{mode}.{field}"
                    )
                    values[field] = MappingProxyType(
                        dict(zip(names, (float(x) for x in resolved), strict=True))
                    )
            modes[mode] = replace(settings, **values)
        components[group] = replace(component, **modes)
    return replace(profiles, arm=components["arm"], hand=components["hand"])
