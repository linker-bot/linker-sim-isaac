"""tiled Isaac scene 构建阶段共享的 immutable data structures。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linkerbot_sim.objects.runtime import RuntimeObjectHandle
from linkerbot_sim.assets.robot_config import RobotGravityPolicy
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    RobotSceneInstanceConfig,
)
from linkerbot_sim.robots.mimic.runtime import MimicFollowerControl
from linkerbot_sim.tiled.config import TiledEnvConfig


@dataclass(frozen=True)
class TiledRobotInstance:
    """env profile 中一个可被 tiled builder 导入的机器人实例。"""

    robot_id: int
    label: str
    profile_name: str
    scene_instance: RobotSceneInstanceConfig


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
    controller_profile: str = "default"
    robot_id: int = 0
    label: str = ""
    kind: str = "arm"
    supports_planning: bool = False

    @property
    def mimic_path(self) -> Path | None:
        """返回原生格式能够声明 follower 关系的资产路径。"""

        return self.asset_path if self.asset_type in {"mjcf", "urdf"} else None


@dataclass(frozen=True)
class TiledArticulationView:
    """一个机器人实例对应的 batched ``Articulation`` view。"""

    name: str
    view: object
    articulation_paths: tuple[str, ...]
    command_joint_names: tuple[str, ...]
    command_joint_indices: np.ndarray
    runtime_mimic_controls: tuple[MimicFollowerControl, ...] = ()
    robot_id: int = 0
    label: str = ""
    controller_profile: str = "default"


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

    @property
    def robots_by_id(self) -> dict[int, ImportedTiledRobot]:
        """按 session robot ID 构造 imported robot 主索引。"""

        return {robot.robot_id: robot for robot in self.robots.values()}

    @property
    def robot_id_by_label(self) -> dict[str, int]:
        """构造稳定 label 到 session ID 的反向索引。"""

        return {
            (robot.label or name): robot.robot_id for name, robot in self.robots.items()
        }

    def robot_label(self, robot_id: int) -> str:
        """按 ID 返回稳定 label，并在错误中列出当前可用 IDs。"""

        try:
            robot = self.robots_by_id[int(robot_id)]
        except KeyError as exc:
            raise KeyError(
                f"unknown robot_id {robot_id!r}; available={sorted(self.robots_by_id)}"
            ) from exc
        return robot.label
