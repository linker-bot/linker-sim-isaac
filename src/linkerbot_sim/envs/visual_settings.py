"""Scene visual settings parsed from env profiles.

本模块只解析纯 Python 配置，不导入 Isaac/Omni。实际创建灯光和设置 viewport
由 ``scene_builder`` 完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


Vec3 = tuple[float, float, float]

DEFAULT_VIEWPORT_EYE: Vec3 = (1.35, -1.65, 1.05)
DEFAULT_VIEWPORT_TARGET: Vec3 = (0.0, -0.1, 0.42)
DEFAULT_VIEWPORT_PRIM_PATH = "/OmniverseKit_Persp"
DEFAULT_KEY_LIGHT_PATH = "/World/KeyLight"
DEFAULT_FILL_LIGHT_PATH = "/World/FillLight"


@dataclass(frozen=True)
class ViewportViewSettings:
    """GUI viewport view settings."""

    enabled: bool = True
    eye: Vec3 = DEFAULT_VIEWPORT_EYE
    target: Vec3 = DEFAULT_VIEWPORT_TARGET
    prim_path: str = DEFAULT_VIEWPORT_PRIM_PATH

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ViewportViewSettings":
        """解析 visuals.viewport；缺省时使用项目默认视角。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("visuals.viewport must be a mapping")
        return cls(
            enabled=_optional_bool(
                data, "enabled", default=True, label="visuals.viewport"
            ),
            eye=_optional_vec3(
                data, "eye", DEFAULT_VIEWPORT_EYE, label="visuals.viewport"
            ),
            target=_optional_vec3(
                data, "target", DEFAULT_VIEWPORT_TARGET, label="visuals.viewport"
            ),
            prim_path=_optional_path(
                data,
                "prim_path",
                DEFAULT_VIEWPORT_PRIM_PATH,
                label="visuals.viewport",
            ),
        )


@dataclass(frozen=True)
class DistantLightSettings:
    """DistantLight settings for the scene key light."""

    enabled: bool = True
    path: str = DEFAULT_KEY_LIGHT_PATH
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
        return cls(
            enabled=_optional_bool(
                data, "enabled", default=True, label="visuals.lights.key"
            ),
            path=_optional_path(
                data, "path", DEFAULT_KEY_LIGHT_PATH, label="visuals.lights.key"
            ),
            intensity=_non_negative_float(
                data.get("intensity", 1200.0), label="visuals.lights.key.intensity"
            ),
            angle=_non_negative_float(
                data.get("angle", 0.5), label="visuals.lights.key.angle"
            ),
            color=_optional_vec3_or_none(data, "color", label="visuals.lights.key"),
            rotation_rpy=_optional_vec3_or_none(
                data, "rotation_rpy", label="visuals.lights.key"
            ),
        )


@dataclass(frozen=True)
class DomeLightSettings:
    """DomeLight settings for ambient/fill light."""

    enabled: bool = True
    path: str = DEFAULT_FILL_LIGHT_PATH
    intensity: float = 250.0
    color: Vec3 | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "DomeLightSettings":
        """解析 visuals.lights.fill；缺省时使用项目默认补光。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("visuals.lights.fill must be a mapping")
        return cls(
            enabled=_optional_bool(
                data, "enabled", default=True, label="visuals.lights.fill"
            ),
            path=_optional_path(
                data, "path", DEFAULT_FILL_LIGHT_PATH, label="visuals.lights.fill"
            ),
            intensity=_non_negative_float(
                data.get("intensity", 250.0), label="visuals.lights.fill.intensity"
            ),
            color=_optional_vec3_or_none(data, "color", label="visuals.lights.fill"),
        )


@dataclass(frozen=True)
class SceneVisualSettings:
    """灯光和 GUI 视角设置。

    ``visuals`` 是 env profile 的可选顶层分组；缺省时使用旧版硬编码默认值。
    """

    viewport: ViewportViewSettings = field(default_factory=ViewportViewSettings)
    key_light: DistantLightSettings = field(default_factory=DistantLightSettings)
    fill_light: DomeLightSettings = field(default_factory=DomeLightSettings)

    @classmethod
    def from_env_config(cls, config: Mapping[str, object]) -> "SceneVisualSettings":
        """从完整 env profile 解析 visuals 顶层分组。"""

        visuals = config.get("visuals")
        if visuals is None:
            return cls()
        if not isinstance(visuals, Mapping):
            raise ValueError("visuals must be a mapping")
        if "camera" in visuals:
            raise ValueError("visuals.camera was renamed to visuals.viewport")
        lights = visuals.get("lights", {})
        if lights is None:
            lights = {}
        if not isinstance(lights, Mapping):
            raise ValueError("visuals.lights must be a mapping")
        return cls(
            viewport=ViewportViewSettings.from_mapping(visuals.get("viewport")),
            key_light=DistantLightSettings.from_mapping(lights.get("key")),
            fill_light=DomeLightSettings.from_mapping(lights.get("fill")),
        )


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
    label: str,
) -> Vec3 | None:
    """读取可选三维向量；字段缺失或 null 时返回 None。"""

    value = data.get(key)
    if value is None:
        return None
    return _vec3(value, label=f"{label}.{key}")


def _vec3(value: object, *, label: str) -> Vec3:
    """把配置值解析为长度为 3 的 float tuple。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a length-3 sequence")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    return values


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

    number = float(value)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number
