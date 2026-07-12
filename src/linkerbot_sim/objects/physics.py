"""object runtime 共享的 PhysX material、root pose 与严格配置辅助函数。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real

from linkerbot_sim.assets.root_pose import RootPoseConfig


@dataclass(frozen=True)
class ObjectMaterialConfig:
    """object 的可选 contact material overrides；None 表示保留资产原值。"""

    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None
    friction_combine_mode: str | None = None

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
            "friction_combine_mode",
        }
        unsupported = set(data) - allowed
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        config = cls(
            static_friction=optional_non_negative_float(data, "static_friction", label),
            dynamic_friction=optional_non_negative_float(
                data, "dynamic_friction", label
            ),
            restitution=optional_non_negative_float(data, "restitution", label),
            friction_combine_mode=optional_friction_combine_mode(data, label),
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
                self.friction_combine_mode,
            )
        )


def apply_root_pose_to_prim(stage, prim_path: str, pose: RootPoseConfig) -> None:
    """清除现有 xform op order，并把 scene root pose 写入指定 USD prim。"""

    from pxr import Gf, Sdf, UsdGeom
    import numpy as np

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid():
        raise RuntimeError(
            f"Cannot apply root_pose; object prim not found: {prim_path}"
        )
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose.xyz))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*tuple(np.degrees(pose.rpy))))


def optional_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    """读取可选 mapping 字段；缺省时返回 None。"""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def optional_non_negative_float(
    data: Mapping[str, object], key: str, parent_label: str
) -> float | None:
    """读取可选非负浮点字段。"""

    if key not in data:
        return None
    raw = data[key]
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise ValueError(f"{parent_label}.{key} must be a number")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{parent_label}.{key} must be finite and non-negative")
    return value


def optional_positive_int(
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


def optional_non_negative_int(
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


def optional_friction_combine_mode(
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
