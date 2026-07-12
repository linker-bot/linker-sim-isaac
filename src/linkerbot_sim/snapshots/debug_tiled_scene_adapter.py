"""无 Isaac 的 debug Tiled Scene runtime snapshot adapter。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.snapshots.compatibility import (
    RobotTargetDescriptor,
    SnapshotTargetDescriptor,
)
from linkerbot_sim.snapshots.schema import (
    RobotSnapshot,
    SimulationSnapshot,
    SnapshotMetadata,
)


def get_debug_tiled_scene_snapshot(
    runtime: object,
    *,
    env_id: int,
) -> SimulationSnapshot:
    """从 debug Tiled Scene runtime 读取一个已经校验过的 env。"""

    selected = int(env_id)
    joint_names = tuple(
        f"joint_{index}" for index in range(runtime.adapter.command_dim)
    )
    return SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="tiled_scene_debug",
            source_env_id=selected,
            step=int(runtime.step),
            time_s=float(runtime.time_s),
            coordinate_frame="env-local",
            info={"per_env": runtime.config.metadata_for_env(selected)},
        ),
        robots={
            "debug": RobotSnapshot(
                label="debug",
                robot_id=0,
                joint_names=joint_names,
                joint_positions=np.asarray(
                    runtime.current_positions[selected],
                    dtype=float,
                ),
                joint_velocities=np.zeros(runtime.adapter.command_dim, dtype=float),
                command_joint_names=joint_names,
                command_targets=np.asarray(
                    runtime.current_positions[selected],
                    dtype=float,
                ),
            )
        },
        objects={},
    )


def debug_tiled_scene_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 debug Tiled Scene runtime 构建最小 target descriptor。"""

    joint_names = tuple(
        f"joint_{index}" for index in range(runtime.adapter.command_dim)
    )
    return SnapshotTargetDescriptor(
        runtime_kind="tiled_scene_debug",
        robots={
            "debug": RobotTargetDescriptor(
                label="debug",
                joint_names=joint_names,
                command_joint_names=joint_names,
            )
        },
        objects={},
    )


__all__ = [
    "debug_tiled_scene_target_descriptor",
    "get_debug_tiled_scene_snapshot",
]
