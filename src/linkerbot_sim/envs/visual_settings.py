"""Scene visual settings parsed from env profiles.

本模块只解析纯 Python 配置，不导入 Isaac/Omni。实际创建灯光和设置 viewport
由 ``scene_builder`` 完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


Vec3 = tuple[float, float, float]

DEFAULT_CAMERA_EYE: Vec3 = (1.35, -1.65, 1.05)
DEFAULT_CAMERA_TARGET: Vec3 = (0.0, -0.1, 0.42)
DEFAULT_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"
DEFAULT_KEY_LIGHT_PATH = "/World/KeyLight"
DEFAULT_FILL_LIGHT_PATH = "/World/FillLight"


@dataclass(frozen=True)
class CameraViewSettings:
    """GUI viewport camera view settings."""

    enabled: bool = True
    eye: Vec3 = DEFAULT_CAMERA_EYE
    target: Vec3 = DEFAULT_CAMERA_TARGET
    prim_path: str = DEFAULT_CAMERA_PRIM_PATH

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "CameraViewSettings":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("visuals.camera must be a mapping")
        return cls(
            enabled=_optional_bool(data, "enabled", default=True, label="visuals.camera"),
            eye=_optional_vec3(data, "eye", DEFAULT_CAMERA_EYE, label="visuals.camera"),
            target=_optional_vec3(
                data, "target", DEFAULT_CAMERA_TARGET, label="visuals.camera"
            ),
            prim_path=_optional_path(
                data,
                "prim_path",
                DEFAULT_CAMERA_PRIM_PATH,
                label="visuals.camera",
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

    camera: CameraViewSettings = field(default_factory=CameraViewSettings)
    key_light: DistantLightSettings = field(default_factory=DistantLightSettings)
    fill_light: DomeLightSettings = field(default_factory=DomeLightSettings)

    @classmethod
    def from_env_config(cls, config: Mapping[str, object]) -> "SceneVisualSettings":
        visuals = config.get("visuals")
        if visuals is None:
            return cls()
        if not isinstance(visuals, Mapping):
            raise ValueError("visuals must be a mapping")
        lights = visuals.get("lights", {})
        if lights is None:
            lights = {}
        if not isinstance(lights, Mapping):
            raise ValueError("visuals.lights must be a mapping")
        return cls(
            camera=CameraViewSettings.from_mapping(visuals.get("camera")),
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
    return _vec3(data.get(key, default), label=f"{label}.{key}")


def _optional_vec3_or_none(
    data: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> Vec3 | None:
    value = data.get(key)
    if value is None:
        return None
    return _vec3(value, label=f"{label}.{key}")


def _vec3(value: object, *, label: str) -> Vec3:
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
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean")
    return value


def _non_negative_float(value: object, *, label: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number
