"""从 env profile 解析 Scene visual settings。

本模块只解析纯 Python 配置，不导入 Isaac/Omni。实际创建灯光和设置 viewport
由 ``scene_builder`` 完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite


Vec3 = tuple[float, float, float]


# 默认值归属各 settings dataclass；解析 partial mapping 时从默认实例读取，
# 避免直接构造和配置解析维护两套 fallback。
@dataclass(frozen=True)
class ViewportViewSettings:
    """GUI viewport 的 camera eye、target 与 prim path 设置。"""

    enabled: bool = True
    eye: Vec3 = (1.35, -1.65, 1.05)
    target: Vec3 = (0.0, -0.1, 0.42)
    prim_path: str = "/OmniverseKit_Persp"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ViewportViewSettings":
        """解析 visuals.viewport；缺省时使用项目默认视角。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("visuals.viewport must be a mapping")
        _reject_keys(
            data,
            {"enabled", "eye", "target", "prim_path"},
            "visuals.viewport",
        )
        defaults = cls()
        return cls(
            enabled=_optional_bool(
                data,
                "enabled",
                default=defaults.enabled,
                label="visuals.viewport",
            ),
            eye=_optional_vec3(data, "eye", defaults.eye, label="visuals.viewport"),
            target=_optional_vec3(
                data, "target", defaults.target, label="visuals.viewport"
            ),
            prim_path=_optional_path(
                data,
                "prim_path",
                defaults.prim_path,
                label="visuals.viewport",
            ),
        )


@dataclass(frozen=True)
class DistantLightSettings:
    """Scene key light 使用的 DistantLight 设置。"""

    enabled: bool = True
    path: str = "/World/KeyLight"
    intensity: float = 1200.0
    angle: float = 0.5
    color: Vec3 | None = None
    rotation_rpy: Vec3 | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "DistantLightSettings":
        """解析 visuals.lights.key；缺省时使用项目默认主光。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("visuals.lights.key must be a mapping")
        _reject_keys(
            data,
            {"enabled", "path", "intensity", "angle", "color", "rotation_rpy"},
            "visuals.lights.key",
        )
        defaults = cls()
        return cls(
            enabled=_optional_bool(
                data,
                "enabled",
                default=defaults.enabled,
                label="visuals.lights.key",
            ),
            path=_optional_path(
                data, "path", defaults.path, label="visuals.lights.key"
            ),
            intensity=_non_negative_float(
                data.get("intensity", defaults.intensity),
                label="visuals.lights.key.intensity",
            ),
            angle=_non_negative_float(
                data.get("angle", defaults.angle),
                label="visuals.lights.key.angle",
            ),
            color=_optional_vec3_or_none(
                data,
                "color",
                default=defaults.color,
                label="visuals.lights.key",
            ),
            rotation_rpy=_optional_vec3_or_none(
                data,
                "rotation_rpy",
                default=defaults.rotation_rpy,
                label="visuals.lights.key",
            ),
        )


@dataclass(frozen=True)
class DomeLightSettings:
    """ambient/fill light 使用的 DomeLight 设置。"""

    enabled: bool = True
    path: str = "/World/FillLight"
    intensity: float = 250.0
    color: Vec3 | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "DomeLightSettings":
        """解析 visuals.lights.fill；缺省时使用项目默认补光。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("visuals.lights.fill must be a mapping")
        _reject_keys(
            data,
            {"enabled", "path", "intensity", "color"},
            "visuals.lights.fill",
        )
        defaults = cls()
        return cls(
            enabled=_optional_bool(
                data,
                "enabled",
                default=defaults.enabled,
                label="visuals.lights.fill",
            ),
            path=_optional_path(
                data, "path", defaults.path, label="visuals.lights.fill"
            ),
            intensity=_non_negative_float(
                data.get("intensity", defaults.intensity),
                label="visuals.lights.fill.intensity",
            ),
            color=_optional_vec3_or_none(
                data,
                "color",
                default=defaults.color,
                label="visuals.lights.fill",
            ),
        )


@dataclass(frozen=True)
class SceneVisualSettings:
    """灯光和 GUI 视角设置。

    ``visuals`` 是 env profile 的可选顶层分组；缺省时使用 typed 默认值。
    """

    viewport: ViewportViewSettings = field(default_factory=ViewportViewSettings)
    key_light: DistantLightSettings = field(default_factory=DistantLightSettings)
    fill_light: DomeLightSettings = field(default_factory=DomeLightSettings)

    @classmethod
    def from_env_config(cls, config: Mapping[str, object]) -> "SceneVisualSettings":
        """从完整 env profile 解析 visuals 顶层分组。"""

        if "visuals" not in config:
            return cls()
        visuals = config["visuals"]
        if visuals is None:
            raise ValueError("visuals must be a mapping")
        if not isinstance(visuals, Mapping):
            raise ValueError("visuals must be a mapping")
        _reject_keys(visuals, {"viewport", "lights"}, "visuals")
        lights = _optional_mapping(
            visuals,
            "lights",
            label="visuals",
        )
        if lights is None:
            lights = {}
        _reject_keys(lights, {"key", "fill"}, "visuals.lights")
        return cls(
            viewport=ViewportViewSettings.from_mapping(
                _optional_mapping(
                    visuals,
                    "viewport",
                    label="visuals",
                )
            ),
            key_light=DistantLightSettings.from_mapping(
                _optional_mapping(
                    lights,
                    "key",
                    label="visuals.lights",
                )
            ),
            fill_light=DomeLightSettings.from_mapping(
                _optional_mapping(
                    lights,
                    "fill",
                    label="visuals.lights",
                )
            ),
        )


def _optional_mapping(
    data: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> Mapping[str, object] | None:
    """读取可选映射；字段缺失时返回 ``None``，显式 ``null`` 则按类型错误拒绝。"""

    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return value


def _optional_vec3(
    data: Mapping[str, object],
    key: str,
    default: Vec3,
    *,
    label: str,
) -> Vec3:
    """读取可选三维向量；字段缺失时返回调用方默认值。"""

    return _vec3(data.get(key, default), label=f"{label}.{key}")


def _optional_vec3_or_none(
    data: Mapping[str, object],
    key: str,
    *,
    default: Vec3 | None,
    label: str,
) -> Vec3 | None:
    """读取可选三维向量；字段缺失时使用默认值，null 时返回 None。"""

    value = data.get(key, default)
    if value is None:
        return None
    return _vec3(value, label=f"{label}.{key}")


def _vec3(value: object, *, label: str) -> Vec3:
    """把配置值解析为长度为 3 的 float tuple。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a length-3 sequence")
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    values = tuple(
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return values[0], values[1], values[2]


def _optional_path(
    data: Mapping[str, object],
    key: str,
    default: str,
    *,
    label: str,
) -> str:
    """读取 USD prim path 字符串，并要求使用绝对路径。"""

    value = data.get(key, default)
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label}.{key} must be an absolute USD prim path string")
    return value


def _optional_bool(
    data: Mapping[str, object],
    key: str,
    *,
    default: bool,
    label: str,
) -> bool:
    """读取布尔字段，拒绝字符串形式的 true/false。"""

    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean")
    return value


def _non_negative_float(value: object, *, label: str) -> float:
    """解析非负浮点值，用于灯光强度和角度等字段。"""

    number = _finite_number(value, label=label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _reject_keys(data: Mapping[str, object], allowed: set[str], label: str) -> None:
    """拒绝未知 visual 键，并在错误中保留其完整 YAML 路径。"""

    unsupported = sorted(str(key) for key in data if key not in allowed)
    if unsupported:
        raise ValueError(f"{label}.{unsupported[0]} is not supported")
