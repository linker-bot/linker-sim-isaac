"""Mirror 与 Kaleidoscope 分离的单环境场景模板配置。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .objects import ObjectProfileConfig
    from .robots import RobotProfileSettings

from .common import (
    ConfigurationError,
    as_bool,
    as_float,
    as_float_tuple,
    as_int,
    as_string,
    as_string_tuple,
    require_keys,
    strict_mapping,
)


@dataclass(frozen=True)
class PoseSettings:
    """以米和 XYZ 弧度表示的静态根位姿。"""

    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "PoseSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(mapping, required={"xyz", "rpy"}, label=label)
        return cls(
            xyz=as_float_tuple(mapping["xyz"], label=f"{label}.xyz", length=3),  # type: ignore[arg-type]
            rpy=as_float_tuple(mapping["rpy"], label=f"{label}.rpy", length=3),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class RobotInstanceSettings:
    """场景内机器人身份、profile 引用与 catalog 绑定结果。"""

    label: str
    robot_profile: str
    root_pose: PoseSettings
    controller_profile: str | None = None
    resolved_profile: "RobotProfileSettings | None" = None

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "RobotInstanceSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping,
            required={"label", "robot_profile", "root_pose"},
            optional={"controller_profile"},
            label=label,
        )
        controller_profile = mapping.get("controller_profile")
        return cls(
            label=as_string(mapping["label"], label=f"{label}.label"),
            robot_profile=as_string(
                mapping["robot_profile"], label=f"{label}.robot_profile"
            ),
            root_pose=PoseSettings.from_mapping(
                mapping["root_pose"], label=f"{label}.root_pose"
            ),
            controller_profile=(
                None
                if controller_profile is None
                else as_string(
                    controller_profile,
                    label=f"{label}.controller_profile",
                )
            ),
        )


@dataclass(frozen=True)
class ObjectInstanceSettings:
    """场景内对象身份、profile 引用与 catalog 绑定结果。"""

    name: str
    object_profile: str
    prim_path: str
    root_pose: PoseSettings
    resolved_profile: "ObjectProfileConfig | None" = None

    def __post_init__(self) -> None:
        if not self.prim_path.startswith("/"):
            raise ConfigurationError(
                "scene.objects[].prim_path must be an absolute USD path"
            )

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "ObjectInstanceSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping,
            required={"name", "object_profile", "prim_path", "root_pose"},
            label=label,
        )
        return cls(
            name=as_string(mapping["name"], label=f"{label}.name"),
            object_profile=as_string(
                mapping["object_profile"], label=f"{label}.object_profile"
            ),
            prim_path=as_string(mapping["prim_path"], label=f"{label}.prim_path"),
            root_pose=PoseSettings.from_mapping(
                mapping["root_pose"], label=f"{label}.root_pose"
            ),
        )


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{label} must be a sequence")
    return value


def _instances(
    mapping: dict[str, object], *, label: str
) -> tuple[tuple[RobotInstanceSettings, ...], tuple[ObjectInstanceSettings, ...]]:
    robot_values = _sequence(mapping["robots"], label=f"{label}.robots")
    object_values = _sequence(mapping["objects"], label=f"{label}.objects")
    robots = tuple(
        RobotInstanceSettings.from_mapping(item, label=f"{label}.robots[{index}]")
        for index, item in enumerate(robot_values)
    )
    objects = tuple(
        ObjectInstanceSettings.from_mapping(item, label=f"{label}.objects[{index}]")
        for index, item in enumerate(object_values)
    )
    if not robots:
        raise ConfigurationError(f"{label}.robots requires at least one robot")
    if len({item.label for item in robots}) != len(robots):
        raise ConfigurationError(f"{label}.robots label must be unique")
    if len({item.name for item in objects}) != len(objects):
        raise ConfigurationError(f"{label}.objects name must be unique")
    prim_paths = [item.prim_path for item in objects]
    if len(set(prim_paths)) != len(prim_paths):
        raise ConfigurationError(f"{label}.objects prim_path must be unique")
    return robots, objects


@dataclass(frozen=True)
class CameraIntrinsicsSettings:
    """以 pixel 为单位的 OpenCV pinhole 相机内参。"""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "CameraIntrinsicsSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(mapping, required={"fx", "fy", "cx", "cy"}, label=label)
        return cls(
            fx=as_float(mapping["fx"], label=f"{label}.fx", strictly_positive=True),
            fy=as_float(mapping["fy"], label=f"{label}.fy", strictly_positive=True),
            cx=as_float(mapping["cx"], label=f"{label}.cx"),
            cy=as_float(mapping["cy"], label=f"{label}.cy"),
        )


@dataclass(frozen=True)
class CameraSettings:
    """Mirror 场景中的物理相机；输出 sink 单独属于 outputs profile。"""

    camera_id: str
    parent_prim_path: str
    prim_path: str
    pose: PoseSettings
    resolution: tuple[int, int]
    frequency_hz: float
    modalities: tuple[str, ...]
    clipping_range_m: tuple[float, float]
    intrinsics: CameraIntrinsicsSettings | None = None

    def __post_init__(self) -> None:
        if not self.parent_prim_path.startswith("/") or not self.prim_path.startswith(
            "/"
        ):
            raise ConfigurationError(
                "scene.cameras prim path must be an absolute USD path"
            )
        if not self.prim_path.startswith(self.parent_prim_path.rstrip("/") + "/"):
            raise ConfigurationError(
                "camera prim_path must be under the parent_prim_path namespace"
            )
        if (
            self.clipping_range_m[0] <= 0
            or self.clipping_range_m[1] <= self.clipping_range_m[0]
        ):
            raise ConfigurationError(
                "camera clipping_range_m must satisfy 0 < near < far"
            )

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "CameraSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "id",
            "parent_prim_path",
            "prim_path",
            "pose",
            "resolution",
            "frequency_hz",
            "modalities",
            "clipping_range_m",
        }
        require_keys(mapping, required=required, optional={"intrinsics"}, label=label)
        resolution_raw = _sequence(mapping["resolution"], label=f"{label}.resolution")
        if len(resolution_raw) != 2:
            raise ConfigurationError(f"{label}.resolution must be [width, height]")
        return cls(
            camera_id=as_string(mapping["id"], label=f"{label}.id"),
            parent_prim_path=as_string(
                mapping["parent_prim_path"], label=f"{label}.parent_prim_path"
            ),
            prim_path=as_string(mapping["prim_path"], label=f"{label}.prim_path"),
            pose=PoseSettings.from_mapping(mapping["pose"], label=f"{label}.pose"),
            resolution=(
                as_int(resolution_raw[0], label=f"{label}.resolution[0]", minimum=1),
                as_int(resolution_raw[1], label=f"{label}.resolution[1]", minimum=1),
            ),
            frequency_hz=as_float(
                mapping["frequency_hz"],
                label=f"{label}.frequency_hz",
                strictly_positive=True,
            ),
            modalities=as_string_tuple(
                mapping["modalities"], label=f"{label}.modalities"
            ),
            clipping_range_m=as_float_tuple(
                mapping["clipping_range_m"], label=f"{label}.clipping_range_m", length=2
            ),  # type: ignore[arg-type]
            intrinsics=(
                CameraIntrinsicsSettings.from_mapping(
                    mapping["intrinsics"], label=f"{label}.intrinsics"
                )
                if "intrinsics" in mapping
                else None
            ),
        )


@dataclass(frozen=True)
class ViewportSettings:
    """场景 GUI viewport 的相机视角。"""

    enabled: bool = True
    eye: tuple[float, float, float] = (1.35, -1.65, 1.05)
    target: tuple[float, float, float] = (0.0, -0.1, 0.42)
    prim_path: str = "/OmniverseKit_Persp"

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "ViewportSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping, required={"enabled", "eye", "target", "prim_path"}, label=label
        )
        prim_path = as_string(mapping["prim_path"], label=f"{label}.prim_path")
        if not prim_path.startswith("/"):
            raise ConfigurationError(f"{label}.prim_path must be an absolute USD path")
        return cls(
            enabled=as_bool(mapping["enabled"], label=f"{label}.enabled"),
            eye=as_float_tuple(mapping["eye"], label=f"{label}.eye", length=3),  # type: ignore[arg-type]
            target=as_float_tuple(mapping["target"], label=f"{label}.target", length=3),  # type: ignore[arg-type]
            prim_path=prim_path,
        )

    @classmethod
    def from_visual_mapping(
        cls,
        value: Mapping[str, object] | None,
        *,
        label: str = "visuals.viewport",
    ) -> "ViewportSettings":
        """解析允许省略字段的启动视觉配置。"""

        if value is None:
            return cls()
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping,
            required=set(),
            optional={"enabled", "eye", "target", "prim_path"},
            label=label,
        )
        defaults = cls()
        prim_path = as_string(
            mapping.get("prim_path", defaults.prim_path),
            label=f"{label}.prim_path",
        )
        if not prim_path.startswith("/"):
            raise ConfigurationError(f"{label}.prim_path must be an absolute USD path")
        return cls(
            enabled=as_bool(
                mapping.get("enabled", defaults.enabled),
                label=f"{label}.enabled",
            ),
            eye=as_float_tuple(
                mapping.get("eye", defaults.eye),
                label=f"{label}.eye",
                length=3,
            ),  # type: ignore[arg-type]
            target=as_float_tuple(
                mapping.get("target", defaults.target),
                label=f"{label}.target",
                length=3,
            ),  # type: ignore[arg-type]
            prim_path=prim_path,
        )


@dataclass(frozen=True)
class DistantLightSettings:
    """用于场景主光的 DistantLight 设置。"""

    enabled: bool = True
    path: str = "/World/KeyLight"
    intensity: float = 1200.0
    angle: float = 0.5
    color: tuple[float, float, float] | None = None
    rotation_rpy: tuple[float, float, float] | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
        *,
        label: str = "visuals.lights.key",
    ) -> "DistantLightSettings":
        mapping = {} if value is None else strict_mapping(value, label=label)
        require_keys(
            mapping,
            required=set(),
            optional={
                "enabled",
                "path",
                "intensity",
                "angle",
                "color",
                "rotation_rpy",
            },
            label=label,
        )
        defaults = cls()
        return cls(
            enabled=as_bool(
                mapping.get("enabled", defaults.enabled), label=f"{label}.enabled"
            ),
            path=_absolute_prim_path(
                mapping.get("path", defaults.path), label=f"{label}.path"
            ),
            intensity=as_float(
                mapping.get("intensity", defaults.intensity),
                label=f"{label}.intensity",
                minimum=0.0,
            ),
            angle=as_float(
                mapping.get("angle", defaults.angle),
                label=f"{label}.angle",
                minimum=0.0,
            ),
            color=_optional_vec3(
                mapping.get("color", defaults.color), label=f"{label}.color"
            ),
            rotation_rpy=_optional_vec3(
                mapping.get("rotation_rpy", defaults.rotation_rpy),
                label=f"{label}.rotation_rpy",
            ),
        )


@dataclass(frozen=True)
class DomeLightSettings:
    """用于场景环境补光的 DomeLight 设置。"""

    enabled: bool = True
    path: str = "/World/FillLight"
    intensity: float = 250.0
    color: tuple[float, float, float] | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
        *,
        label: str = "visuals.lights.fill",
    ) -> "DomeLightSettings":
        mapping = {} if value is None else strict_mapping(value, label=label)
        require_keys(
            mapping,
            required=set(),
            optional={"enabled", "path", "intensity", "color"},
            label=label,
        )
        defaults = cls()
        return cls(
            enabled=as_bool(
                mapping.get("enabled", defaults.enabled), label=f"{label}.enabled"
            ),
            path=_absolute_prim_path(
                mapping.get("path", defaults.path), label=f"{label}.path"
            ),
            intensity=as_float(
                mapping.get("intensity", defaults.intensity),
                label=f"{label}.intensity",
                minimum=0.0,
            ),
            color=_optional_vec3(
                mapping.get("color", defaults.color), label=f"{label}.color"
            ),
        )


@dataclass(frozen=True)
class SceneVisualSettings:
    """场景灯光和 GUI 视角；不包含 Kit/RTX 运行时资源。"""

    viewport: ViewportSettings = field(default_factory=ViewportSettings)
    key_light: DistantLightSettings = field(default_factory=DistantLightSettings)
    fill_light: DomeLightSettings = field(default_factory=DomeLightSettings)

    @classmethod
    def from_scene_mapping(cls, config: Mapping[str, object]) -> "SceneVisualSettings":
        if "visuals" not in config:
            return cls()
        visuals = strict_mapping(config["visuals"], label="visuals")
        require_keys(
            visuals,
            required=set(),
            optional={"viewport", "lights"},
            label="visuals",
        )
        lights = strict_mapping(visuals.get("lights", {}), label="visuals.lights")
        require_keys(
            lights,
            required=set(),
            optional={"key", "fill"},
            label="visuals.lights",
        )
        return cls(
            viewport=ViewportSettings.from_visual_mapping(
                _optional_mapping(visuals, "viewport", label="visuals"),
            ),
            key_light=DistantLightSettings.from_mapping(
                _optional_mapping(lights, "key", label="visuals.lights"),
            ),
            fill_light=DomeLightSettings.from_mapping(
                _optional_mapping(lights, "fill", label="visuals.lights"),
            ),
        )


def _optional_mapping(
    mapping: Mapping[str, object], key: str, *, label: str
) -> Mapping[str, object] | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label}.{key} must be a mapping")
    return value


def _absolute_prim_path(value: object, *, label: str) -> str:
    path = as_string(value, label=label)
    if not path.startswith("/"):
        raise ConfigurationError(f"{label} must be an absolute USD path")
    return path


def _optional_vec3(value: object, *, label: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return as_float_tuple(value, label=label, length=3)  # type: ignore[return-value]


@dataclass(frozen=True)
class LightSettings:
    light_id: str
    path: str
    intensity: float
    angle: float | None

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "LightSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping,
            required={"id", "path", "intensity"},
            optional={"angle"},
            label=label,
        )
        path = as_string(mapping["path"], label=f"{label}.path")
        if not path.startswith("/"):
            raise ConfigurationError(f"{label}.path must be an absolute USD path")
        return cls(
            light_id=as_string(mapping["id"], label=f"{label}.id"),
            path=path,
            intensity=as_float(
                mapping["intensity"], label=f"{label}.intensity", strictly_positive=True
            ),
            angle=(
                as_float(
                    mapping["angle"], label=f"{label}.angle", strictly_positive=True
                )
                if "angle" in mapping
                else None
            ),
        )


@dataclass(frozen=True)
class MirrorSceneSettings:
    """现实映像场景：允许相机与视觉事实，但不包含物理后端选择。"""

    scene_id: str
    description: str
    gravity_z: float
    add_ground: bool
    ground_height: float
    physics_frequency_hz: float
    render_frequency_hz: float
    robots: tuple[RobotInstanceSettings, ...]
    objects: tuple[ObjectInstanceSettings, ...]
    cameras: tuple[CameraSettings, ...]
    viewport: ViewportSettings
    lights: tuple[LightSettings, ...]
    planning_startup: Literal["lazy", "prewarm"] = "lazy"

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "scene"
    ) -> "MirrorSceneSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "id",
            "description",
            "gravity_z",
            "add_ground",
            "ground_height",
            "physics_frequency_hz",
            "render_frequency_hz",
            "robots",
            "objects",
            "cameras",
            "viewport",
            "lights",
            "planning_startup",
        }
        require_keys(mapping, required=required, label=label)
        robots, objects = _instances(mapping, label=label)
        camera_values = _sequence(mapping["cameras"], label=f"{label}.cameras")
        cameras = tuple(
            CameraSettings.from_mapping(item, label=f"{label}.cameras[{index}]")
            for index, item in enumerate(camera_values)
        )
        light_values = _sequence(mapping["lights"], label=f"{label}.lights")
        lights = tuple(
            LightSettings.from_mapping(item, label=f"{label}.lights[{index}]")
            for index, item in enumerate(light_values)
        )
        if len({item.camera_id for item in cameras}) != len(cameras):
            raise ConfigurationError(f"{label}.cameras id must be unique")
        if len({item.light_id for item in lights}) != len(lights):
            raise ConfigurationError(f"{label}.lights id must be unique")
        return cls(
            scene_id=as_string(mapping["id"], label=f"{label}.id"),
            description=as_string(mapping["description"], label=f"{label}.description"),
            gravity_z=as_float(mapping["gravity_z"], label=f"{label}.gravity_z"),
            add_ground=as_bool(mapping["add_ground"], label=f"{label}.add_ground"),
            ground_height=as_float(
                mapping["ground_height"], label=f"{label}.ground_height"
            ),
            physics_frequency_hz=as_float(
                mapping["physics_frequency_hz"],
                label=f"{label}.physics_frequency_hz",
                strictly_positive=True,
            ),
            render_frequency_hz=as_float(
                mapping["render_frequency_hz"],
                label=f"{label}.render_frequency_hz",
                strictly_positive=True,
            ),
            robots=robots,
            objects=objects,
            cameras=cameras,
            viewport=ViewportSettings.from_mapping(
                mapping["viewport"], label=f"{label}.viewport"
            ),
            lights=lights,
            planning_startup=as_string(
                mapping["planning_startup"],
                label=f"{label}.planning_startup",
                choices={"lazy", "prewarm"},
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class KaleidoscopeSceneSettings:
    """并行 RL 的无渲染单环境模板。

    类型本身没有 cameras、viewport 或 render_frequency 字段，因而这些概念不可能被
    “解析但忽略”。环境数量与命名由 mode root 持有，复制和隔离机制则由物理后端派生。
    """

    scene_id: str
    description: str
    gravity_z: float
    add_ground: bool
    ground_height: float
    physics_frequency_hz: float
    robots: tuple[RobotInstanceSettings, ...]
    objects: tuple[ObjectInstanceSettings, ...]

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "scene"
    ) -> "KaleidoscopeSceneSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "id",
            "description",
            "gravity_z",
            "add_ground",
            "ground_height",
            "physics_frequency_hz",
            "robots",
            "objects",
        }
        require_keys(mapping, required=required, label=label)
        robots, objects = _instances(mapping, label=label)
        return cls(
            scene_id=as_string(mapping["id"], label=f"{label}.id"),
            description=as_string(mapping["description"], label=f"{label}.description"),
            gravity_z=as_float(mapping["gravity_z"], label=f"{label}.gravity_z"),
            add_ground=as_bool(mapping["add_ground"], label=f"{label}.add_ground"),
            ground_height=as_float(
                mapping["ground_height"], label=f"{label}.ground_height"
            ),
            physics_frequency_hz=as_float(
                mapping["physics_frequency_hz"],
                label=f"{label}.physics_frequency_hz",
                strictly_positive=True,
            ),
            robots=robots,
            objects=objects,
        )


__all__ = [
    "CameraIntrinsicsSettings",
    "CameraSettings",
    "DistantLightSettings",
    "DomeLightSettings",
    "KaleidoscopeSceneSettings",
    "LightSettings",
    "MirrorSceneSettings",
    "ObjectInstanceSettings",
    "PoseSettings",
    "RobotInstanceSettings",
    "SceneVisualSettings",
    "ViewportSettings",
]
