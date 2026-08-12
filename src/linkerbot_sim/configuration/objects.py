"""``configs/objects`` 的唯一 typed schema 与 mapping 解析边界。

资产来源、导入参数、物理属性和规划碰撞属性只允许出现在 ``configs/objects/*.yaml``。
YAML I/O 与路径解析由 configuration catalog 负责，本模块只校验 catalog 提供的 mapping。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Literal, TypeAlias, cast

from linkerbot_sim.configuration.robots import AssetImportConfig


_OBJECT_PROFILE_ROOT_KEYS = frozenset({"object"})


@dataclass(frozen=True)
class ObjectStateSummaryConfig:
    """dynamic-chain 顶层 object state 的确定性摘要规则。

    动态链包含多个刚体，必须显式指定 ``reference_body``，才能稳定定义对外对象位姿；该
    字段是 body 名而不是完整 prim path。
    """

    reference_body: str | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
    ) -> "ObjectStateSummaryConfig":
        """解析 ``object.state_summary``，当前只支持命名参考刚体。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unsupported = set(data) - {"reference_body"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        value = data.get("reference_body")
        if value is None:
            return cls()
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}.reference_body must be a non-empty string")
        reference_body = value.strip()
        if "/" in reference_body or "\\" in reference_body:
            raise ValueError(
                f"{label}.reference_body must be a body name, not a prim path"
            )
        return cls(reference_body=reference_body)


@dataclass(frozen=True)
class ObjectMaterialConfig:
    """所有物理后端共用的 contact material；None 表示保留资产原值。"""

    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "ObjectMaterialConfig | None":
        """解析可选对象接触材质；没有任何覆盖时返回 None。"""

        if data is None:
            return None
        allowed = {
            "static_friction",
            "dynamic_friction",
            "restitution",
        }
        unsupported = set(data) - allowed
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        config = cls(
            static_friction=_optional_non_negative_float(
                data, "static_friction", label
            ),
            dynamic_friction=_optional_non_negative_float(
                data, "dynamic_friction", label
            ),
            restitution=_optional_non_negative_float(data, "restitution", label),
        )
        if config.restitution is not None and config.restitution > 1.0:
            raise ValueError(f"{label}.restitution must be between 0 and 1")
        return config if config.has_overrides() else None

    def has_overrides(self) -> bool:
        """返回是否至少设置了一个材质字段。"""

        return any(
            value is not None
            for value in (
                self.static_friction,
                self.dynamic_friction,
                self.restitution,
            )
        )


@dataclass(frozen=True)
class ObjectPhysxMaterialConfig:
    """仅由 PhysX material schema 消费的对象材质字段。"""

    friction_combine_mode: str | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "ObjectPhysxMaterialConfig | None":
        """严格解析 ``object.physics.physx.material``。"""

        if data is None:
            return None
        unsupported = set(data) - {"friction_combine_mode"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        config = cls(friction_combine_mode=_optional_friction_combine_mode(data, label))
        return config if config.friction_combine_mode is not None else None


@dataclass(frozen=True)
class RigidObjectPlanningCollisionConfig:
    """规划后端使用的简化碰撞几何，不改变仿真碰撞体。"""

    shape: str
    size: tuple[float, ...]
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enabled: bool = True
    padding: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
    ) -> "RigidObjectPlanningCollisionConfig | None":
        """解析 backend-neutral planning shape；它不会修改仿真 collider。"""

        if data is None:
            return None
        unsupported = set(data) - {
            "shape",
            "size",
            "xyz",
            "rpy",
            "enabled",
            "padding",
        }
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        shape_value = data.get("shape")
        if not isinstance(shape_value, str):
            raise ValueError(f"{label}.shape must be a string")
        shape = shape_value.lower()
        if shape not in {"cuboid", "sphere", "capsule"}:
            raise ValueError(f"{label}.shape must be one of cuboid, sphere, capsule")
        if "size" not in data:
            raise ValueError(f"{label}.size is required")
        size = _numeric_sequence(data["size"], label=f"{label}.size")
        expected = {"cuboid": 3, "sphere": 1, "capsule": 2}[shape]
        if len(size) != expected:
            raise ValueError(f"{label}.size for {shape} must contain {expected} values")
        if any(value <= 0.0 for value in size):
            raise ValueError(f"{label}.size values must be positive")
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{label}.enabled must be a boolean")
        padding = _finite_float(data.get("padding", 0.0), f"{label}.padding")
        if padding < 0.0:
            raise ValueError(f"{label}.padding cannot be negative")
        return cls(
            shape=shape,
            size=size,
            xyz=_vec3_tuple(data.get("xyz"), label=f"{label}.xyz"),
            rpy=_vec3_tuple(data.get("rpy"), label=f"{label}.rpy"),
            enabled=enabled,
            padding=padding,
        )


@dataclass(frozen=True)
class RigidObjectPhysxConfig:
    """Rigid object 的 PhysX-only 运行时覆盖。"""

    material: ObjectPhysxMaterialConfig | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
    ) -> "RigidObjectPhysxConfig | None":
        """严格解析 ``object.physics.physx``，空 leaf 规范化为 None。"""

        if data is None:
            return None
        unsupported = set(data) - {"material"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        material = ObjectPhysxMaterialConfig.from_mapping(
            _optional_mapping(data, "material", label),
            label=f"{label}.material",
        )
        return cls(material=material) if material is not None else None


@dataclass(frozen=True)
class RigidObjectPhysicsConfig:
    """Rigid object 的通用物理属性与显式 PhysX leaf。"""

    static: bool = False
    material: ObjectMaterialConfig | None = None
    physx: RigidObjectPhysxConfig | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
    ) -> "RigidObjectPhysicsConfig":
        """解析 static、通用 material 与可选 PhysX-only override。"""

        if data is None:
            return cls()
        unsupported = set(data) - {"static", "material", "physx"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        static = data.get("static", False)
        if not isinstance(static, bool):
            raise ValueError(f"{label}.static must be a boolean")
        return cls(
            static=static,
            material=ObjectMaterialConfig.from_mapping(
                _optional_mapping(data, "material", label),
                label=f"{label}.material",
            ),
            physx=RigidObjectPhysxConfig.from_mapping(
                _optional_mapping(data, "physx", label),
                label=f"{label}.physx",
            ),
        )


@dataclass(frozen=True)
class CapsuleRopePhysxSolverConfig:
    """绳体刚体的 PhysX solver iteration 覆盖。"""

    position_iterations: int | None = None
    velocity_iterations: int | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "CapsuleRopePhysxSolverConfig | None":
        """严格解析 ``object.physics.physx.solver``。"""

        if data is None:
            return None
        unsupported = set(data) - {"position_iterations", "velocity_iterations"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        config = cls(
            position_iterations=_optional_positive_int(
                data, "position_iterations", label
            ),
            velocity_iterations=_optional_non_negative_int(
                data, "velocity_iterations", label
            ),
        )
        if config.position_iterations is None and config.velocity_iterations is None:
            return None
        return config


@dataclass(frozen=True)
class CapsuleRopePhysxConfig:
    """只允许 PhysX session 投影的绳体运行时配置。"""

    material: ObjectPhysxMaterialConfig | None = None
    solver: CapsuleRopePhysxSolverConfig | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "CapsuleRopePhysxConfig | None":
        """严格解析 ``object.physics.physx``，空 leaf 规范化为 None。"""

        if data is None:
            return None
        unsupported = set(data) - {"material", "solver"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        config = cls(
            material=ObjectPhysxMaterialConfig.from_mapping(
                _optional_mapping(data, "material", label),
                label=f"{label}.material",
            ),
            solver=CapsuleRopePhysxSolverConfig.from_mapping(
                _optional_mapping(data, "solver", label),
                label=f"{label}.solver",
            ),
        )
        return (
            config if config.material is not None or config.solver is not None else None
        )


@dataclass(frozen=True)
class CapsuleRopePhysicsConfig:
    """绳体的后端通用材质与显式 PhysX leaf。"""

    material: ObjectMaterialConfig | None = None
    physx: CapsuleRopePhysxConfig | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "CapsuleRopePhysicsConfig":
        """解析 capsule rope 运行时物理覆盖。"""

        if data is None:
            return cls()
        unsupported = set(data) - {"material", "physx"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        return cls(
            material=ObjectMaterialConfig.from_mapping(
                _optional_mapping(data, "material", label),
                label=f"{label}.material",
            ),
            physx=CapsuleRopePhysxConfig.from_mapping(
                _optional_mapping(data, "physx", label),
                label=f"{label}.physx",
            ),
        )

    def has_overrides(self) -> bool:
        """返回是否有任何需要写入 stage 的运行时物理覆盖。"""

        return self.material is not None or self.physx is not None


@dataclass(frozen=True)
class RigidObjectProfileConfig:
    """完整、不可变的 rigid object profile，不包含场景实例位姿。"""

    profile_name: str
    name: str
    kind: Literal["rigid"]
    source: Literal["usd", "urdf"]
    asset_path: str
    urdf_drive_type: Literal["none", "position"]
    import_config: "AssetImportConfig"
    physics: "RigidObjectPhysicsConfig"
    planning_collision: "RigidObjectPlanningCollisionConfig | None"

    def __post_init__(self) -> None:
        if self.kind != "rigid":
            raise ValueError("RigidObjectProfileConfig.kind must be 'rigid'")
        if self.source not in {"usd", "urdf"}:
            raise ValueError("RigidObjectProfileConfig.source must be 'usd' or 'urdf'")
        if self.source == "usd" and self.urdf_drive_type != "none":
            raise ValueError("USD rigid objects must use urdf_drive_type='none'")
        if self.import_config.fix_base is True and not self.physics.static:
            raise ValueError(
                "object.import.fix_base=true conflicts with object.physics.static=false"
            )


@dataclass(frozen=True)
class DynamicChainObjectProfileConfig:
    """完整、不可变的 dynamic-chain profile，不包含场景实例路径和位姿。"""

    profile_name: str
    name: str
    kind: Literal["dynamic_chain"]
    source: Literal["usd"]
    asset_path: str
    root_path: str
    physics: "CapsuleRopePhysicsConfig"
    state_summary: ObjectStateSummaryConfig

    def __post_init__(self) -> None:
        if self.kind != "dynamic_chain":
            raise ValueError(
                "DynamicChainObjectProfileConfig.kind must be 'dynamic_chain'"
            )
        if self.source != "usd":
            raise ValueError("DynamicChainObjectProfileConfig.source must be 'usd'")
        if not self.root_path.startswith("/"):
            raise ValueError("dynamic_chain object.root_path must be absolute")
        if self.state_summary.reference_body is None:
            raise ValueError(
                "dynamic_chain object.state_summary.reference_body is required"
            )


ObjectProfileConfig: TypeAlias = (
    RigidObjectProfileConfig | DynamicChainObjectProfileConfig
)


def object_profile_from_mapping(
    data: Mapping[str, Any],
    *,
    profile_name: str,
    source: str | None = None,
) -> ObjectProfileConfig:
    """按 ``object.kind`` 一次性解析完整 object profile 判别联合。"""

    source_label = source or f"Object profile {profile_name!r}"
    try:
        return _object_profile_from_mapping(data, profile_name=profile_name)
    except ValueError as exc:
        if str(exc).startswith(f"{source_label}:"):
            raise
        raise ValueError(f"{source_label}: {exc}") from exc


def _object_profile_from_mapping(
    data: Mapping[str, Any], *, profile_name: str
) -> ObjectProfileConfig:
    """执行不附加文件来源的 schema 校验和 typed projection。"""

    if not isinstance(data, Mapping):
        raise ValueError("object profile must be a mapping")
    _reject_unknown_paths(data, _OBJECT_PROFILE_ROOT_KEYS, "profile")
    object_cfg = _required_mapping(data, "object", "profile")
    allowed = {
        "name",
        "kind",
        "source",
        "asset_path",
        "root_path",
        "urdf_drive_type",
        "import",
        "physics",
        "planning_collision",
        "state_summary",
    }
    _reject_unknown_paths(object_cfg, allowed, "object")
    for key in ("kind", "source", "asset_path"):
        if key not in object_cfg:
            raise ValueError(f"Object profile {profile_name!r} missing object.{key}")
    for section in ("import", "physics", "planning_collision", "state_summary"):
        if section in object_cfg and not isinstance(object_cfg[section], Mapping):
            raise ValueError(f"object.{section} must be a mapping")

    kind = _required_non_empty_str(
        object_cfg["kind"], f"Object profile {profile_name!r} object.kind"
    ).lower()
    if kind not in {"rigid", "dynamic_chain"}:
        raise ValueError(
            f"Object profile {profile_name!r} object.kind must be one of "
            f"['dynamic_chain', 'rigid'], got {kind!r}"
        )
    source = _required_non_empty_str(
        object_cfg["source"], f"Object profile {profile_name!r} object.source"
    ).lower()
    if source not in {"usd", "urdf"}:
        raise ValueError(
            f"Object profile {profile_name!r} object.source must be one of "
            f"['urdf', 'usd'], got {source!r}"
        )
    name = _required_non_empty_str(
        object_cfg.get("name", profile_name),
        f"Object profile {profile_name!r} object.name",
    )
    asset_path = _required_non_empty_str(
        object_cfg["asset_path"],
        f"Object profile {profile_name!r} object.asset_path",
    )
    if kind == "rigid":
        return _rigid_profile_from_mapping(
            object_cfg,
            profile_name=profile_name,
            name=name,
            source=source,
            asset_path=asset_path,
        )
    return _dynamic_chain_profile_from_mapping(
        object_cfg,
        profile_name=profile_name,
        name=name,
        source=source,
        asset_path=asset_path,
    )


def _rigid_profile_from_mapping(
    object_cfg: Mapping[str, object],
    *,
    profile_name: str,
    name: str,
    source: str,
    asset_path: str,
) -> RigidObjectProfileConfig:
    """解析 rigid-only leaf，并保留解析后的 typed settings。"""

    if "root_path" in object_cfg:
        raise ValueError("object.root_path is only supported for dynamic_chain")
    if "state_summary" in object_cfg:
        raise ValueError(
            "object.state_summary is only supported for dynamic_chain objects"
        )
    if source == "usd" and "urdf_drive_type" in object_cfg:
        raise ValueError(
            "object.urdf_drive_type is only supported for rigid URDF objects"
        )
    import_settings = _optional_mapping(object_cfg, "import", "object")
    physics = RigidObjectPhysicsConfig.from_mapping(
        _optional_mapping(object_cfg, "physics", "object"),
        label="object.physics",
    )
    return RigidObjectProfileConfig(
        profile_name=profile_name,
        name=name,
        kind="rigid",
        source=cast(Literal["usd", "urdf"], source),
        asset_path=asset_path,
        urdf_drive_type=_object_urdf_drive_type(
            object_cfg.get("urdf_drive_type", "none"),
            label=f"Object profile {profile_name!r} object.urdf_drive_type",
        ),
        import_config=AssetImportConfig.from_mapping(
            import_settings,
            label="object.import",
            asset_type=source,
        ),
        physics=physics,
        planning_collision=RigidObjectPlanningCollisionConfig.from_mapping(
            _optional_mapping(object_cfg, "planning_collision", "object"),
            label="object.planning_collision",
        ),
    )


def _dynamic_chain_profile_from_mapping(
    object_cfg: Mapping[str, object],
    *,
    profile_name: str,
    name: str,
    source: str,
    asset_path: str,
) -> DynamicChainObjectProfileConfig:
    """解析 dynamic-chain-only leaf，并保留解析后的 typed settings。"""

    if source != "usd":
        raise ValueError("dynamic_chain object.source must be 'usd'")
    if "root_path" not in object_cfg:
        raise ValueError("dynamic_chain object.root_path is required")
    if "import" in object_cfg:
        raise ValueError("object.import is only supported for rigid objects")
    if "planning_collision" in object_cfg:
        raise ValueError(
            "object.planning_collision is only supported for rigid objects"
        )
    if "urdf_drive_type" in object_cfg:
        raise ValueError(
            "object.urdf_drive_type is only supported for rigid URDF objects"
        )
    state_summary = ObjectStateSummaryConfig.from_mapping(
        _optional_mapping(object_cfg, "state_summary", "object"),
        label="object.state_summary",
    )
    root_path = _required_non_empty_str(
        object_cfg["root_path"],
        f"Object profile {profile_name!r} object.root_path",
    )
    return DynamicChainObjectProfileConfig(
        profile_name=profile_name,
        name=name,
        kind="dynamic_chain",
        source="usd",
        asset_path=asset_path,
        root_path=root_path,
        physics=CapsuleRopePhysicsConfig.from_mapping(
            _optional_mapping(object_cfg, "physics", "object"),
            label="object.physics",
        ),
        state_summary=state_summary,
    )


def _required_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object]:
    """读取必填 mapping 字段，并生成带父路径的错误信息。"""

    value = data.get(key)
    if value is None:
        raise ValueError(f"{parent_label}.{key} is required")
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    """读取可选 mapping 字段；缺省时返回 None。"""

    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _required_non_empty_str(value: object, label: str) -> str:
    """读取并去除必填字符串两端空白，空值按完整字段路径报错。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_non_negative_float(
    data: Mapping[str, object], key: str, parent_label: str
) -> float | None:
    """读取可选的有限非负浮点字段。"""

    if key not in data:
        return None
    raw = data[key]
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise ValueError(f"{parent_label}.{key} must be a number")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{parent_label}.{key} must be finite and non-negative")
    return value


def _optional_positive_int(
    data: Mapping[str, object], key: str, parent_label: str
) -> int | None:
    """读取可选正整数字段。"""

    if key not in data:
        return None
    raw = data[key]
    if isinstance(raw, bool) or not isinstance(raw, Integral):
        raise ValueError(f"{parent_label}.{key} must be an integer")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{parent_label}.{key} must be positive")
    return value


def _optional_non_negative_int(
    data: Mapping[str, object], key: str, parent_label: str
) -> int | None:
    """读取可选非负整数字段。"""

    if key not in data:
        return None
    raw = data[key]
    if isinstance(raw, bool) or not isinstance(raw, Integral):
        raise ValueError(f"{parent_label}.{key} must be an integer")
    value = int(raw)
    if value < 0:
        raise ValueError(f"{parent_label}.{key} cannot be negative")
    return value


def _optional_friction_combine_mode(
    data: Mapping[str, object], parent_label: str
) -> str | None:
    """读取 PhysX friction combine mode，并限制在支持的枚举内。"""

    if "friction_combine_mode" not in data:
        return None
    raw = data["friction_combine_mode"]
    if not isinstance(raw, str):
        raise ValueError(f"{parent_label}.friction_combine_mode must be a string")
    value = raw.lower()
    allowed = {"average", "min", "multiply", "max"}
    if value not in allowed:
        raise ValueError(
            f"{parent_label}.friction_combine_mode must be one of "
            f"{sorted(allowed)}, got {value!r}"
        )
    return value


def _vec3_tuple(
    value: object | None,
    *,
    label: str,
) -> tuple[float, float, float]:
    """把可选 xyz/rpy 输入规范为三元 float tuple，缺省返回零向量。"""

    if value is None:
        return (0.0, 0.0, 0.0)
    values = _numeric_sequence(value, label=label)
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    return values[0], values[1], values[2]


def _numeric_sequence(value: object, *, label: str) -> tuple[float, ...]:
    """严格解析有限数值序列。"""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(
        _finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _object_urdf_drive_type(
    value: object, *, label: str
) -> Literal["none", "position"]:
    """规范化对象 URDF drive 类型并限制为当前支持的枚举。"""

    normalized = _required_non_empty_str(value, label).lower()
    if normalized not in {"none", "position"}:
        raise ValueError(f"{label} must be 'none' or 'position'")
    return cast(Literal["none", "position"], normalized)


def _reject_unknown_paths(
    data: Mapping[Any, Any], allowed: set[str] | frozenset[str], label: str
) -> None:
    """拒绝 mapping 中未由当前配置所有者声明的字段。"""

    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        paths = ", ".join(f"{label}.{key}" for key in unknown)
        raise ValueError(f"unsupported configuration field(s): {paths}")


__all__ = [
    "CapsuleRopePhysicsConfig",
    "CapsuleRopePhysxConfig",
    "CapsuleRopePhysxSolverConfig",
    "DynamicChainObjectProfileConfig",
    "ObjectMaterialConfig",
    "ObjectPhysxMaterialConfig",
    "ObjectProfileConfig",
    "ObjectStateSummaryConfig",
    "RigidObjectPhysicsConfig",
    "RigidObjectPhysxConfig",
    "RigidObjectPlanningCollisionConfig",
    "RigidObjectProfileConfig",
    "object_profile_from_mapping",
]
