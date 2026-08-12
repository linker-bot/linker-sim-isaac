"""已解析 rigid object profile 的 stage/runtime 投影。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.objects import (
    RigidObjectPhysicsConfig,
    RigidObjectPlanningCollisionConfig,
)
from linkerbot_sim.configuration.robots import AssetImportConfig


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


__all__ = [
    "RigidObjectConfig",
]
