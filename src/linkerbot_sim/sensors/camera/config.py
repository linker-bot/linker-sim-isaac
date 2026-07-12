"""从 env profile 解析 sensor camera settings。

本模块只解析纯 Python 配置，不导入 Isaac/Omni。实际创建 camera prim、render product 和
采样数据由 sensor runtime 完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

from linkerbot_sim.utils.config import require_loopback_host


Vec2i = tuple[int, int]
Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

# 支持集合是所有相机共享的校验约束，不属于某个实例的默认配置。
SUPPORTED_CAMERA_MODALITIES = frozenset(
    {"rgb", "depth", "semantic_segmentation", "instance_segmentation"}
)


# 实例默认值归属各 settings dataclass；解析器通过类字段复用同一来源。
@dataclass(frozen=True)
class SensorCameraIntrinsicsSettings:
    """以 pixel 为单位的 pinhole camera intrinsics。"""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "SensorCameraIntrinsicsSettings | None":
        """解析可选 camera intrinsics；省略时保留 Isaac 默认参数。"""

        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_keys(data, {"fx", "fy", "cx", "cy"}, label=label)
        return cls(
            fx=_required_positive_finite_float(data, "fx", label=label),
            fy=_required_positive_finite_float(data, "fy", label=label),
            cx=_required_finite_float(data, "cx", label=label),
            cy=_required_finite_float(data, "cy", label=label),
        )

    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        """返回 row-major 3x3 camera intrinsic matrix。"""

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

    @property
    def has_consumer(self) -> bool:
        """返回是否配置了会实际消费 camera frame 的输出端。"""

        return any(
            value is not None
            for value in (
                self.save_dir,
                self.foxglove_live_port,
                self.foxglove_mcap_path,
            )
        )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "SensorCameraOutputSettings":
        """解析单个 camera 的 output 分组。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_keys(
            data,
            {
                "save_dir",
                "foxglove_topic_prefix",
                "foxglove_live_host",
                "foxglove_live_port",
                "foxglove_mcap_path",
            },
            label=label,
        )
        return cls(
            save_dir=_optional_non_empty_str(data, "save_dir", label=label),
            foxglove_topic_prefix=_optional_topic_prefix(
                data, "foxglove_topic_prefix", label=label
            ),
            foxglove_live_host=require_loopback_host(
                data.get("foxglove_live_host", cls.foxglove_live_host),
                label=f"{label}.foxglove_live_host",
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
    resolution: Vec2i = (640, 480)
    frequency: float = 30.0
    # None means no tiled selector was declared; tiled validation rejects it.
    env_ids: tuple[int, ...] | None = None
    modalities: tuple[str, ...] = ("rgb",)
    clipping_range: Vec2 = (0.01, 5.0)
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
        _reject_keys(
            data,
            {
                "enabled",
                "prim_path",
                "parent_prim_path",
                "pose",
                "resolution",
                "frequency",
                "env_ids",
                "modalities",
                "clipping_range",
                "intrinsics",
                "output",
            },
            label=camera_label,
        )
        pose = data.get("pose", {})
        if pose is None:
            pose = {}
        if not isinstance(pose, Mapping):
            raise ValueError(f"{camera_label}.pose must be a mapping")
        _reject_keys(pose, {"xyz", "rpy"}, label=f"{camera_label}.pose")
        return cls(
            name=camera_name,
            enabled=_optional_bool(
                data, "enabled", default=cls.enabled, label=camera_label
            ),
            prim_path=_required_path(data, "prim_path", label=camera_label),
            parent_prim_path=_optional_path_or_none(
                data, "parent_prim_path", label=camera_label
            ),
            pose_xyz=_optional_vec3(
                pose, "xyz", cls.pose_xyz, label=f"{camera_label}.pose"
            ),
            pose_rpy=_optional_vec3(
                pose, "rpy", cls.pose_rpy, label=f"{camera_label}.pose"
            ),
            resolution=_optional_resolution(
                data, "resolution", cls.resolution, label=camera_label
            ),
            frequency=_positive_float(
                data.get("frequency", cls.frequency),
                label=f"{camera_label}.frequency",
            ),
            env_ids=_optional_env_ids(data, label=camera_label),
            modalities=_optional_modalities(
                data, "modalities", cls.modalities, label=camera_label
            ),
            clipping_range=_optional_clipping_range(
                data,
                "clipping_range",
                cls.clipping_range,
                label=camera_label,
            ),
            intrinsics=SensorCameraIntrinsicsSettings.from_mapping(
                data.get("intrinsics"), label=f"{camera_label}.intrinsics"
            ),
            output=SensorCameraOutputSettings.from_mapping(
                data.get("output"), label=f"{camera_label}.output"
            ),
        )


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
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    values = tuple(
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return values[0], values[1], values[2]


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
    if len(value) != 2:
        raise ValueError(f"{label}.{key} must contain exactly 2 values")
    return values[0], values[1]


def _positive_int(value: object, *, label: str) -> int:
    """解析正整数，拒绝 bool。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} values must be positive integers")
    if value <= 0:
        raise ValueError(f"{label} values must be positive integers")
    return value


def _positive_float(value: object, *, label: str) -> float:
    """解析正浮点数。"""

    parsed = _finite_number(value, label=label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return parsed


def _required_finite_float(
    data: Mapping[str, object], key: str, *, label: str
) -> float:
    """读取必填有限浮点数。"""

    if key not in data:
        raise ValueError(f"{label}.{key} is required")
    return _finite_number(data[key], label=f"{label}.{key}")


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
    if len(value) != 2:
        raise ValueError(f"{label}.{key} must contain exactly 2 values")
    values = tuple(
        _finite_number(item, label=f"{label}.{key}[{index}]")
        for index, item in enumerate(value)
    )
    near, far = values
    if near <= 0.0 or far <= near:
        raise ValueError(f"{label}.{key} must satisfy 0 < near < far")
    return near, far


def _optional_modalities(
    data: Mapping[str, object],
    key: str,
    default: tuple[str, ...],
    *,
    label: str,
) -> tuple[str, ...]:
    """读取并校验 camera output modalities。"""

    value = data.get(key, default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}.{key} must be a non-empty sequence")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label}.{key} must contain strings")
    modalities = tuple(value)
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


def _optional_env_ids(
    data: Mapping[str, object], *, label: str
) -> tuple[int, ...] | None:
    """解析可选的 tiled camera 环境范围，要求序列非空、元素为整数且不重复。"""

    if "env_ids" not in data:
        return None
    value = data["env_ids"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}.env_ids must be a non-empty sequence of integers")
    env_ids: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label}.env_ids[{index}] must be an integer")
        if item < 0:
            raise ValueError(f"{label}.env_ids[{index}] must be nonnegative")
        env_ids.append(item)
    if not env_ids:
        raise ValueError(f"{label}.env_ids must be non-empty")
    if len(set(env_ids)) != len(env_ids):
        raise ValueError(f"{label}.env_ids cannot contain duplicates")
    return tuple(env_ids)


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


def _non_empty_str_with_default(
    data: Mapping[str, object],
    key: str,
    *,
    label: str,
    default: str,
) -> str:
    value = _optional_non_empty_str(data, key, label=label, default=default)
    assert value is not None
    return value


def _finite_number(value: object, *, label: str) -> float:
    """解析有限 YAML 数值，不接受布尔值或数值字符串的隐式转换。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


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


def _reject_keys(data: Mapping[str, object], allowed: set[str], *, label: str) -> None:
    """拒绝未知键，防止 camera scope 拼写错误意外退化为选择全部环境。"""

    unsupported = set(data) - allowed
    if unsupported:
        names = sorted(str(key) for key in unsupported)
        keys = ", ".join(names)
        paths = ", ".join(f"{label}.{key}" for key in names)
        raise ValueError(
            f"{label} contains unsupported keys: {keys} (full paths: {paths})"
        )
