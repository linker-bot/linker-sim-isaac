"""Sensor camera settings parsed from env profiles.

本模块只解析纯 Python 配置，不导入 Isaac/Omni。实际创建 camera prim、render product 和
采样数据由 sensor runtime 完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite


Vec2i = tuple[int, int]
Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

DEFAULT_CAMERA_RESOLUTION: Vec2i = (640, 480)
DEFAULT_CAMERA_FREQUENCY = 30.0
DEFAULT_CAMERA_MODALITIES = ("rgb",)
DEFAULT_CAMERA_CLIPPING_RANGE: Vec2 = (0.01, 5.0)
SUPPORTED_CAMERA_MODALITIES = frozenset(
    {"rgb", "depth", "semantic_segmentation", "instance_segmentation"}
)


@dataclass(frozen=True)
class SensorCameraIntrinsicsSettings:
    """Pinhole camera intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "SensorCameraIntrinsicsSettings | None":
        """Parse optional camera intrinsics."""

        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        return cls(
            fx=_required_positive_finite_float(data, "fx", label=label),
            fy=_required_positive_finite_float(data, "fy", label=label),
            cx=_required_finite_float(data, "cx", label=label),
            cy=_required_finite_float(data, "cy", label=label),
        )

    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        """Return the 3x3 camera matrix."""

        return (
            (self.fx, 0.0, self.cx),
            (0.0, self.fy, self.cy),
            (0.0, 0.0, 1.0),
        )


@dataclass(frozen=True)
class SensorCameraOutputSettings:
    """传感器摄像机输出设置。"""

    save_dir: str | None = None
    foxglove_topic_prefix: str | None = None
    foxglove_live_host: str = "127.0.0.1"
    foxglove_live_port: int | None = None
    foxglove_mcap_path: str | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "SensorCameraOutputSettings":
        """解析单个 camera 的 output 分组。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        return cls(
            save_dir=_optional_non_empty_str(data, "save_dir", label=label),
            foxglove_topic_prefix=_optional_topic_prefix(
                data, "foxglove_topic_prefix", label=label
            ),
            foxglove_live_host=_optional_non_empty_str(
                data, "foxglove_live_host", label=label, default="127.0.0.1"
            ),
            foxglove_live_port=_optional_positive_int_or_none(
                data, "foxglove_live_port", label=label
            ),
            foxglove_mcap_path=_optional_non_empty_str(
                data, "foxglove_mcap_path", label=label
            ),
        )


@dataclass(frozen=True)
class SensorCameraSettings:
    """单个仿真传感器摄像机配置。"""

    name: str
    prim_path: str
    enabled: bool = True
    parent_prim_path: str | None = None
    pose_xyz: Vec3 = (0.0, 0.0, 0.0)
    pose_rpy: Vec3 = (0.0, 0.0, 0.0)
    resolution: Vec2i = DEFAULT_CAMERA_RESOLUTION
    frequency: float = DEFAULT_CAMERA_FREQUENCY
    modalities: tuple[str, ...] = DEFAULT_CAMERA_MODALITIES
    clipping_range: Vec2 = DEFAULT_CAMERA_CLIPPING_RANGE
    intrinsics: SensorCameraIntrinsicsSettings | None = None
    output: SensorCameraOutputSettings = field(
        default_factory=SensorCameraOutputSettings
    )

    def __post_init__(self) -> None:
        """校验跨字段约束。"""

        if self.parent_prim_path is None:
            return
        parent_prefix = self.parent_prim_path.rstrip("/") + "/"
        if not self.prim_path.startswith(parent_prefix):
            raise ValueError(
                f"sensors.cameras.{self.name}.prim_path must be under "
                "parent_prim_path when parent_prim_path is set"
            )

    @classmethod
    def from_mapping(
        cls, name: object, data: object, *, label: str
    ) -> "SensorCameraSettings":
        """解析 sensors.cameras.<name>。"""

        camera_name = _camera_name(name, label=label)
        camera_label = f"{label}.{camera_name}"
        if not isinstance(data, Mapping):
            raise ValueError(f"{camera_label} must be a mapping")
        pose = data.get("pose", {})
        if pose is None:
            pose = {}
        if not isinstance(pose, Mapping):
            raise ValueError(f"{camera_label}.pose must be a mapping")
        return cls(
            name=camera_name,
            enabled=_optional_bool(data, "enabled", default=True, label=camera_label),
            prim_path=_required_path(data, "prim_path", label=camera_label),
            parent_prim_path=_optional_path_or_none(
                data, "parent_prim_path", label=camera_label
            ),
            pose_xyz=_optional_vec3(
                pose, "xyz", (0.0, 0.0, 0.0), label=f"{camera_label}.pose"
            ),
            pose_rpy=_optional_vec3(
                pose, "rpy", (0.0, 0.0, 0.0), label=f"{camera_label}.pose"
            ),
            resolution=_optional_resolution(
                data, "resolution", DEFAULT_CAMERA_RESOLUTION, label=camera_label
            ),
            frequency=_positive_float(
                data.get("frequency", DEFAULT_CAMERA_FREQUENCY),
                label=f"{camera_label}.frequency",
            ),
            modalities=_optional_modalities(data, "modalities", label=camera_label),
            clipping_range=_optional_clipping_range(
                data,
                "clipping_range",
                DEFAULT_CAMERA_CLIPPING_RANGE,
                label=camera_label,
            ),
            intrinsics=SensorCameraIntrinsicsSettings.from_mapping(
                data.get("intrinsics"), label=f"{camera_label}.intrinsics"
            ),
            output=SensorCameraOutputSettings.from_mapping(
                data.get("output"), label=f"{camera_label}.output"
            ),
        )


@dataclass(frozen=True)
class SceneSensorSettings:
    """env profile 中的 sensor 配置集合。"""

    cameras: tuple[SensorCameraSettings, ...] = ()

    @classmethod
    def from_env_config(cls, config: Mapping[str, object]) -> "SceneSensorSettings":
        """从完整 env profile 解析 sensors 顶层分组。"""

        sensors = config.get("sensors")
        if sensors is None:
            return cls()
        if not isinstance(sensors, Mapping):
            raise ValueError("sensors must be a mapping")
        cameras = sensors.get("cameras", {})
        if cameras is None:
            cameras = {}
        if not isinstance(cameras, Mapping):
            raise ValueError("sensors.cameras must be a mapping")
        return cls(
            cameras=tuple(
                SensorCameraSettings.from_mapping(
                    name, camera_data, label="sensors.cameras"
                )
                for name, camera_data in cameras.items()
            )
        )

    @property
    def enabled_cameras(self) -> tuple[SensorCameraSettings, ...]:
        """返回 enabled=true 的摄像机配置。"""

        return tuple(camera for camera in self.cameras if camera.enabled)


def _camera_name(value: object, *, label: str) -> str:
    """校验 camera mapping key。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} keys must be non-empty strings")
    if "/" in value or "\\" in value:
        raise ValueError(f"{label}.{value} name must not contain path separators")
    return value


def _required_path(data: Mapping[str, object], key: str, *, label: str) -> str:
    """读取必填 USD prim path。"""

    if key not in data:
        raise ValueError(f"{label}.{key} is required")
    return _path(data[key], label=f"{label}.{key}")


def _optional_path_or_none(
    data: Mapping[str, object], key: str, *, label: str
) -> str | None:
    """读取可选 USD prim path。"""

    value = data.get(key)
    if value is None:
        return None
    return _path(value, label=f"{label}.{key}")


def _path(value: object, *, label: str) -> str:
    """校验绝对 USD prim path。"""

    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute USD prim path string")
    return value


def _optional_vec3(
    data: Mapping[str, object],
    key: str,
    default: Vec3,
    *,
    label: str,
) -> Vec3:
    """读取可选三维向量。"""

    return _vec3(data.get(key, default), label=f"{label}.{key}")


def _vec3(value: object, *, label: str) -> Vec3:
    """把配置值解析为长度为 3 的 float tuple。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a length-3 sequence")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    return values


def _optional_resolution(
    data: Mapping[str, object],
    key: str,
    default: Vec2i,
    *,
    label: str,
) -> Vec2i:
    """读取图像分辨率。"""

    value = data.get(key, default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}.{key} must be a length-2 integer sequence")
    values = tuple(_positive_int(item, label=f"{label}.{key}") for item in value)
    if len(values) != 2:
        raise ValueError(f"{label}.{key} must contain exactly 2 values")
    return values


def _positive_int(value: object, *, label: str) -> int:
    """解析正整数，拒绝 bool。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} values must be positive integers")
    if value <= 0:
        raise ValueError(f"{label} values must be positive integers")
    return value


def _positive_float(value: object, *, label: str) -> float:
    """解析正浮点数。"""

    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _required_finite_float(
    data: Mapping[str, object], key: str, *, label: str
) -> float:
    """读取必填有限浮点数。"""

    if key not in data:
        raise ValueError(f"{label}.{key} is required")
    parsed = float(data[key])
    if not isfinite(parsed):
        raise ValueError(f"{label}.{key} must be finite")
    return parsed


def _required_positive_finite_float(
    data: Mapping[str, object], key: str, *, label: str
) -> float:
    """读取必填正有限浮点数。"""

    parsed = _required_finite_float(data, key, label=label)
    if parsed <= 0.0:
        raise ValueError(f"{label}.{key} must be positive")
    return parsed


def _optional_clipping_range(
    data: Mapping[str, object],
    key: str,
    default: Vec2,
    *,
    label: str,
) -> Vec2:
    """读取 near/far clipping range。"""

    value = data.get(key, default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}.{key} must be a length-2 sequence")
    values = tuple(float(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"{label}.{key} must contain exactly 2 values")
    near, far = values
    if near <= 0.0 or far <= near:
        raise ValueError(f"{label}.{key} must satisfy 0 < near < far")
    return near, far


def _optional_modalities(
    data: Mapping[str, object], key: str, *, label: str
) -> tuple[str, ...]:
    """读取并校验 camera output modalities。"""

    value = data.get(key, DEFAULT_CAMERA_MODALITIES)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}.{key} must be a non-empty sequence")
    modalities = tuple(str(item) for item in value)
    if not modalities:
        raise ValueError(f"{label}.{key} must be non-empty")
    seen: set[str] = set()
    for modality in modalities:
        if modality not in SUPPORTED_CAMERA_MODALITIES:
            raise ValueError(
                f"{label}.{key} contains unsupported modality {modality!r}"
            )
        if modality in seen:
            raise ValueError(f"{label}.{key} contains duplicate modality {modality!r}")
        seen.add(modality)
    return modalities


def _optional_non_empty_str(
    data: Mapping[str, object],
    key: str,
    *,
    label: str,
    default: str | None = None,
) -> str | None:
    """读取可选非空字符串。"""

    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _optional_topic_prefix(
    data: Mapping[str, object], key: str, *, label: str
) -> str | None:
    """读取可选 topic prefix。"""

    value = _optional_non_empty_str(data, key, label=label)
    if value is not None and not value.startswith("/"):
        raise ValueError(f"{label}.{key} must start with /")
    return value


def _optional_positive_int_or_none(
    data: Mapping[str, object], key: str, *, label: str
) -> int | None:
    """读取可选正整数。"""

    value = data.get(key)
    if value is None:
        return None
    return _positive_int(value, label=f"{label}.{key}")


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
