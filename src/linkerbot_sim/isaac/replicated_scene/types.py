"""replicated PhysX scene 构造阶段的不可变资源描述。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

from linkerbot_sim.assets.robot_instances import RobotExecutionConfig
from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.objects.runtime import RuntimeObjectHandle


@dataclass(frozen=True, slots=True)
class SourceReplicatedRobot:
    """只存在于 source env 导入阶段的机器人拓扑。"""

    robot_id: int
    label: str
    profile_name: str
    profile: RobotProfileSettings
    controller_bundle_name: str
    controller_profiles: ControllerProfiles
    execution: RobotExecutionConfig
    asset_path: Path
    asset_type: str
    articulation_path: str
    imported_root_path: str
    controlled_joints: tuple[str, ...]
    tcp_frame_name: str
    tcp_parent_frame_name: str
    tcp_parent_body_path: str
    tcp_offset_xyz: tuple[float, float, float]
    tcp_offset_rpy: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ImportedReplicatedRobot:
    """导入 source env 后可复制的机器人拓扑与 raw view 绑定信息。"""

    robot_id: int
    label: str
    profile_name: str
    profile: RobotProfileSettings
    controller_bundle_name: str
    controller_profiles: ControllerProfiles
    execution: RobotExecutionConfig
    asset_path: Path
    asset_type: str
    articulation_paths: tuple[str, ...]
    imported_root_paths: tuple[str, ...]
    controlled_joints: tuple[str, ...]
    tcp_frame_name: str
    tcp_parent_frame_name: str
    tcp_body_paths: tuple[str, ...]
    tcp_offset_xyz: tuple[float, float, float]
    tcp_offset_rpy: tuple[float, float, float]
    articulation_view: object
    command_joint_names: tuple[str, ...] = ()
    command_joint_indices: np.ndarray | None = None

    def with_command_binding(
        self,
        *,
        names: tuple[str, ...],
        indices: np.ndarray,
    ) -> "ImportedReplicatedRobot":
        """在 ``World.reset`` 后冻结 command-space 顺序。"""

        return replace(
            self,
            command_joint_names=names,
            command_joint_indices=np.ascontiguousarray(indices, dtype=np.int64),
        )


@dataclass(frozen=True, slots=True)
class ReplicatedPhysxScene:
    """已构建的同构 PhysX stage；不拥有 World 或 SimulationApp。"""

    env_root_paths: tuple[str, ...]
    env_origins: np.ndarray
    robots: tuple[ImportedReplicatedRobot, ...]
    object_handles: tuple[RuntimeObjectHandle, ...]
    object_prim_paths: Mapping[str, tuple[str, ...]]
    collision_isolation_strategy: Literal["env_ids"] = field(
        init=False,
        default="env_ids",
    )

    @property
    def num_envs(self) -> int:
        return len(self.env_root_paths)


@dataclass(frozen=True, slots=True)
class ReplicatedNewtonScene:
    """单份 USD prototype 已复制进独立 Newton worlds 的场景描述。

    destination env 不要求在 USD stage 中物化；路径仍按 world 顺序保存，供 Newton
    topology label、articulation view 和 rigid-body view 做精确绑定。
    """

    env_root_paths: tuple[str, ...]
    env_origins: np.ndarray
    robots: tuple[ImportedReplicatedRobot, ...]
    object_handles: tuple[RuntimeObjectHandle, ...]
    object_prim_paths: Mapping[str, tuple[str, ...]]
    collision_isolation_strategy: Literal["separate_worlds"] = field(
        init=False,
        default="separate_worlds",
    )

    @property
    def num_envs(self) -> int:
        return len(self.env_root_paths)


__all__ = [
    "ImportedReplicatedRobot",
    "ReplicatedNewtonScene",
    "ReplicatedPhysxScene",
    "SourceReplicatedRobot",
]
