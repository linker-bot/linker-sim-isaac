"""机器人 profile 到单场景资产执行配置的投影。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from linkerbot_sim.configuration.robots import (
    AssetImportConfig,
    RobotContactMaterialSettings,
    RobotGravityPolicy,
    RobotProfileSettings,
    RobotPhysxSettings,
)
from linkerbot_sim.robots.classification import RobotComponentMapping
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class RobotAssetConfig:
    """解析后的机器人 profile 与一个场景实例路径的执行组合。"""

    asset_type: str
    asset_path: Path
    prim_path: str
    name: str = "robot"
    controller_profile: str | None = None
    urdf_drive_type: str = "position"
    import_config: AssetImportConfig = field(default_factory=AssetImportConfig)
    gravity_policy: RobotGravityPolicy = field(default_factory=RobotGravityPolicy)
    contact_material: RobotContactMaterialSettings | None = None
    physx: RobotPhysxSettings = field(default_factory=RobotPhysxSettings)
    component_mapping: RobotComponentMapping = field(
        default_factory=RobotComponentMapping
    )

    @classmethod
    def from_profile(
        cls,
        profile: RobotProfileSettings,
        *,
        prim_path: str,
        name: str | None = None,
    ) -> "RobotAssetConfig":
        """把 catalog 已解析的 typed profile 投影为 importer 配置。"""

        if not isinstance(profile, RobotProfileSettings):
            raise TypeError("profile must be RobotProfileSettings")
        return cls(
            asset_type=profile.asset_type,
            asset_path=repo_path(profile.asset_path),
            prim_path=_absolute_prim_path(prim_path, "robot instance prim_path"),
            name=profile.name if name is None else name,
            controller_profile=profile.controller_profile,
            urdf_drive_type=profile.urdf_drive_type,
            import_config=profile.import_config,
            gravity_policy=profile.gravity_policy,
            contact_material=profile.contact_material,
            physx=profile.physx,
            component_mapping=profile.component_mapping,
        )


def _absolute_prim_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be a canonical absolute USD path")
    if value == "/" or value.endswith("/") or "//" in value:
        raise ValueError(f"{label} must be a canonical absolute USD path")
    return value


__all__ = ["RobotAssetConfig"]
