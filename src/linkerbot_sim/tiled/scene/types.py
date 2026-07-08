"""Shared data structures for tiled Isaac scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linkerbot_sim.app.runtime.objects import RuntimeObjectHandle
from linkerbot_sim.assets.robot_loader import RobotExecutionConfig, RobotGravityPolicy
from linkerbot_sim.tiled.config import TiledEnvConfig


@dataclass(frozen=True)
class TiledRobotInstance:
    """env profile 中一个可被 tiled builder 导入的机器人实例。"""

    name: str
    profile_name: str
    scene_instance: object


@dataclass(frozen=True)
class ImportedTiledRobot:
    """reset 前已经导入 env_0 并记录 clone 后路径的机器人摘要。"""

    name: str
    profile_name: str
    execution: RobotExecutionConfig
    articulation_root_suffix: str
    imported_root_suffix: str
    articulation_paths: tuple[str, ...]
    imported_root_paths: tuple[str, ...]
    asset_path: Path
    asset_type: str
    controlled_joints: tuple[str, ...]
    gravity_policy: RobotGravityPolicy
    gravity_counts: dict[str, int]
    solver_counts: dict[str, int]

    @property
    def mjcf_path(self) -> Path | None:
        """MJCF 资产路径；URDF 机器人没有 mimic/equality 文件需要解析。"""

        return self.asset_path if self.asset_type == "mjcf" else None


@dataclass(frozen=True)
class TiledArticulationView:
    """一个机器人实例对应的 batched ``Articulation`` view。"""

    name: str
    view: object
    articulation_paths: tuple[str, ...]
    command_joint_names: tuple[str, ...]
    command_joint_indices: np.ndarray


@dataclass(frozen=True)
class IsaacTiledScene:
    """构建完成但尚未 reset/finalize 的 tiled scene 描述。"""

    config: TiledEnvConfig
    env_root_paths: tuple[str, ...]
    env_origins: np.ndarray
    clone_positions: np.ndarray
    robots: dict[str, ImportedTiledRobot]
    articulation_views: dict[str, TiledArticulationView]
    object_handles: tuple[RuntimeObjectHandle, ...]
    object_prim_paths: dict[str, tuple[str, ...]]
    robot_root_pose_overrides_applied: int
    object_pose_overrides_applied: int
    collision_filtering_applied: bool
