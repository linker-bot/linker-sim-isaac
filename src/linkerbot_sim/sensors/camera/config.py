"""Mirror camera runtime 使用的 typed settings。

YAML 只由 ``configuration.scenes`` 与 ``configuration.outputs`` 解析；本模块不做第二次
mapping 解析，也不导入 Isaac/Omni。实际创建 camera prim、render product 和采样数据由
sensor runtime 完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from linkerbot_sim.utils.config import require_loopback_host


Vec2i = tuple[int, int]
Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

SUPPORTED_CAMERA_MODALITIES = frozenset(
    {"rgb", "depth", "semantic_segmentation", "instance_segmentation"}
)


@dataclass(frozen=True)
class SensorCameraIntrinsicsSettings:
    """以 pixel 为单位的 pinhole camera intrinsics。"""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        _require_positive_finite(self.fx, label="camera intrinsics fx")
        _require_positive_finite(self.fy, label="camera intrinsics fy")
        _require_finite(self.cx, label="camera intrinsics cx")
        _require_finite(self.cy, label="camera intrinsics cy")

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

    def __post_init__(self) -> None:
        _require_optional_non_empty_string(
            self.save_dir,
            label="camera output save_dir",
        )
        _require_optional_non_empty_string(
            self.foxglove_topic_prefix,
            label="camera output foxglove_topic_prefix",
        )
        if (
            self.foxglove_topic_prefix is not None
            and not self.foxglove_topic_prefix.startswith("/")
        ):
            raise ValueError("camera output foxglove_topic_prefix must start with /")
        require_loopback_host(
            self.foxglove_live_host,
            label="camera output foxglove_live_host",
        )
        if self.foxglove_live_port is not None:
            _require_positive_int(
                self.foxglove_live_port,
                label="camera output foxglove_live_port",
            )
        _require_optional_non_empty_string(
            self.foxglove_mcap_path,
            label="camera output foxglove_mcap_path",
        )

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
    # None 表示没有声明复制环境 selector；Mirror 配置会拒绝非空 selector。
    env_ids: tuple[int, ...] | None = None
    modalities: tuple[str, ...] = ("rgb",)
    clipping_range: Vec2 = (0.01, 5.0)
    intrinsics: SensorCameraIntrinsicsSettings | None = None
    output: SensorCameraOutputSettings = field(
        default_factory=SensorCameraOutputSettings
    )

    def __post_init__(self) -> None:
        """校验 runtime typed camera 的字段与跨字段约束。"""

        _require_camera_name(self.name)
        _require_path(self.prim_path, label=f"camera {self.name} prim_path")
        if not isinstance(self.enabled, bool):
            raise TypeError(f"camera {self.name} enabled must be a boolean")
        if self.parent_prim_path is not None:
            _require_path(
                self.parent_prim_path,
                label=f"camera {self.name} parent_prim_path",
            )
        _require_vec3(self.pose_xyz, label=f"camera {self.name} pose_xyz")
        _require_vec3(self.pose_rpy, label=f"camera {self.name} pose_rpy")
        _require_resolution(self.resolution, label=f"camera {self.name} resolution")
        _require_positive_finite(self.frequency, label=f"camera {self.name} frequency")
        _require_env_ids(self.env_ids, label=f"camera {self.name} env_ids")
        _require_modalities(self.modalities, label=f"camera {self.name} modalities")
        _require_clipping_range(
            self.clipping_range,
            label=f"camera {self.name} clipping_range",
        )
        if self.intrinsics is not None and not isinstance(
            self.intrinsics, SensorCameraIntrinsicsSettings
        ):
            raise TypeError(f"camera {self.name} intrinsics has invalid type")
        if not isinstance(self.output, SensorCameraOutputSettings):
            raise TypeError(f"camera {self.name} output has invalid type")
        if self.parent_prim_path is not None and not self.prim_path.startswith(
            self.parent_prim_path.rstrip("/") + "/"
        ):
            raise ValueError(
                f"sensors.cameras.{self.name}.prim_path must be under "
                "parent_prim_path when parent_prim_path is set"
            )


def _require_camera_name(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("camera name must be a non-empty string")
    if "/" in value or "\\" in value:
        raise ValueError("camera name must not contain path separators")


def _require_path(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute USD prim path string")


def _require_vec3(value: object, *, label: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{label} must be a length-3 tuple")
    for index, item in enumerate(value):
        _require_finite(item, label=f"{label}[{index}]")


def _require_resolution(value: object, *, label: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a length-2 tuple")
    for item in value:
        _require_positive_int(item, label=label)


def _require_env_ids(value: object, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} must be a non-empty tuple of integers")
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label}[{index}] must be an integer")
        if item < 0:
            raise ValueError(f"{label}[{index}] must be nonnegative")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} cannot contain duplicates")


def _require_modalities(value: object, *, label: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} must be a non-empty tuple")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} cannot contain duplicates")
    for modality in value:
        if not isinstance(modality, str):
            raise ValueError(f"{label} must contain strings")
        if modality not in SUPPORTED_CAMERA_MODALITIES:
            raise ValueError(f"{label} contains unsupported modality {modality!r}")


def _require_clipping_range(value: object, *, label: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a length-2 tuple")
    near = _require_finite(value[0], label=f"{label}[0]")
    far = _require_finite(value[1], label=f"{label}[1]")
    if near <= 0.0 or far <= near:
        raise ValueError(f"{label} must satisfy 0 < near < far")


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_positive_finite(value: object, *, label: str) -> float:
    number = _require_finite(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _require_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _require_optional_non_empty_string(value: object, *, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{label} must be a non-empty string")


__all__ = [
    "SUPPORTED_CAMERA_MODALITIES",
    "SensorCameraIntrinsicsSettings",
    "SensorCameraOutputSettings",
    "SensorCameraSettings",
]
