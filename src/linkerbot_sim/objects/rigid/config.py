"""Rigid object 的纯配置模型与 env profile 展开。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path

from linkerbot_sim.assets.robot_config import AssetImportConfig
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.objects.config import (
    ObjectProfileConfig,
    expanded_object_mapping,
    object_scene_instances_from_env_config,
)
from linkerbot_sim.objects.physics import ObjectMaterialConfig, optional_mapping
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class RigidObjectPlanningCollisionConfig:
    """规划后端使用的简化碰撞几何，不改变 PhysX 碰撞体。"""

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
        """解析 backend-neutral planning shape；它不会修改 PhysX collider。"""

        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
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
        result = cls(
            shape=shape,
            size=size,
            xyz=_vec3_tuple(data.get("xyz"), label=f"{label}.xyz"),
            rpy=_vec3_tuple(data.get("rpy"), label=f"{label}.rpy"),
            enabled=enabled,
            padding=padding,
        )
        return result


@dataclass(frozen=True)
class RigidObjectPhysicsConfig:
    """Rigid object 的 static 与物理材质覆盖。"""

    static: bool = False
    material: ObjectMaterialConfig | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
    ) -> "RigidObjectPhysicsConfig":
        """解析 static 开关与可选 PhysX material override。"""

        if data is None:
            return cls()
        unsupported_keys = set(data) - {"static", "material"}
        if unsupported_keys:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported_keys))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        static = data.get("static", False)
        if not isinstance(static, bool):
            raise ValueError(f"{label}.static must be a boolean")
        material_data = optional_mapping(data, "material", label)
        return cls(
            static=static,
            material=ObjectMaterialConfig.from_mapping(
                material_data,
                label=f"{label}.material",
            ),
        )


@dataclass(frozen=True)
class RigidObjectConfig:
    """单个 rigid object 的资产来源、stage 路径与运行时覆盖。"""

    name: str
    asset_type: str
    asset_path: Path
    prim_path: str
    root_pose: RootPoseConfig = RootPoseConfig()
    physics: RigidObjectPhysicsConfig = RigidObjectPhysicsConfig()
    planning_collision: RigidObjectPlanningCollisionConfig | None = None
    urdf_drive_type: str = "none"
    import_config: AssetImportConfig = field(default_factory=AssetImportConfig)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        index: int,
    ) -> "RigidObjectConfig":
        """解析 profile 展开后的低层配置。"""

        if not isinstance(data, Mapping):
            raise ValueError(f"objects[{index}] must be a mapping")
        allowed = {
            "name",
            "source",
            "asset_path",
            "prim_path",
            "root_pose",
            "physics",
            "planning_collision",
            "urdf_drive_type",
            "import",
        }
        unsupported = sorted(str(key) for key in data if key not in allowed)
        if unsupported:
            paths = ", ".join(f"objects[{index}].{key}" for key in unsupported)
            raise ValueError(f"unsupported configuration field(s): {paths}")

        name = str(data.get("name", f"object_{index}"))
        if not name:
            raise ValueError(f"objects[{index}].name cannot be empty")
        asset_type = str(data.get("source", "usd")).lower()
        if asset_type not in {"usd", "urdf"}:
            raise ValueError(
                f"objects[{index}].source must be 'usd' or 'urdf', got {asset_type!r}"
            )
        if asset_type == "usd" and "import" in data:
            raise ValueError(f"objects[{index}].import is not supported for USD assets")
        if "asset_path" not in data:
            raise ValueError(f"objects[{index}].asset_path is required")
        if "prim_path" not in data:
            raise ValueError(f"objects[{index}].prim_path is required")
        prim_path = str(data["prim_path"])
        if not prim_path.startswith("/"):
            raise ValueError(f"objects[{index}].prim_path must be an absolute USD path")
        result = cls(
            name=name,
            asset_type=asset_type,
            asset_path=repo_path(str(data["asset_path"])),
            prim_path=prim_path,
            root_pose=RootPoseConfig.from_mapping(
                optional_mapping(data, "root_pose", f"objects[{index}]")
            ),
            physics=RigidObjectPhysicsConfig.from_mapping(
                optional_mapping(data, "physics", f"objects[{index}]"),
                label=f"objects[{index}].physics",
            ),
            planning_collision=RigidObjectPlanningCollisionConfig.from_mapping(
                optional_mapping(
                    data,
                    "planning_collision",
                    f"objects[{index}]",
                ),
                label=f"objects[{index}].planning_collision",
            ),
            urdf_drive_type=str(data.get("urdf_drive_type", "none")),
            import_config=AssetImportConfig.from_mapping(
                optional_mapping(data, "import", f"objects[{index}]"),
                label=f"objects[{index}].import",
                asset_type=asset_type,
            ),
        )
        if (
            result.asset_type == "urdf"
            and result.import_config.fix_base is True
            and not result.physics.static
        ):
            raise ValueError(
                f"objects[{index}].import.fix_base=true conflicts with "
                "physics.static=false"
            )
        return result


def rigid_objects_from_env_config(
    config: Mapping[str, object],
) -> tuple[RigidObjectConfig, ...]:
    """展开 env YAML 中所有 kind=rigid 的 object profile。"""

    instances = object_scene_instances_from_env_config(config)
    profiles = {
        item.object_profile: ObjectProfileConfig.from_profile(item.object_profile)
        for item in instances
    }
    rigid_objects: list[RigidObjectConfig] = []
    for index, item in enumerate(instances):
        profile = profiles[item.object_profile]
        if profile.kind != "rigid":
            continue
        rigid_objects.append(
            RigidObjectConfig.from_mapping(
                expanded_object_mapping(item, profile),
                index=index,
            )
        )
    return tuple(rigid_objects)


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


__all__ = [
    "RigidObjectConfig",
    "RigidObjectPhysicsConfig",
    "RigidObjectPlanningCollisionConfig",
    "rigid_objects_from_env_config",
]
