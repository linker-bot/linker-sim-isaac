"""运行时无关的仿真快照公共入口。

snapshot 层只描述机器人/对象状态和兼容性检查，不直接依赖 Isaac 或 cuRobo。canonical
SingleSceneRuntime 与 TiledSceneRuntime 通过 adapters 共享同一套 canonical schema。
"""

from linkerbot_sim.snapshots.compatibility import (
    JointMapping,
    ObjectCompatibilityMapping,
    ObjectTargetDescriptor,
    RobotCompatibilityMapping,
    RobotTargetDescriptor,
    SnapshotCompatibilityError,
    SnapshotCompatibilityResult,
    SnapshotTargetDescriptor,
    check_snapshot_compatibility,
    require_snapshot_compatibility,
)
from linkerbot_sim.snapshots.dispatch import (
    get_snapshot,
    set_snapshot,
)
from linkerbot_sim.snapshots.single_scene_adapter import (
    get_single_scene_snapshot,
    set_single_scene_snapshot,
    single_scene_target_descriptor,
)
from linkerbot_sim.snapshots.tiled_scene_adapter import (
    clone_tiled_env_state,
    get_tiled_scene_snapshot,
    set_tiled_scene_snapshot,
    tiled_scene_target_descriptor,
)
from linkerbot_sim.snapshots.schema import (
    SNAPSHOT_SCHEMA,
    ObjectSnapshot,
    RobotSnapshot,
    SimulationSnapshot,
    SnapshotMetadata,
    SnapshotRestoreResult,
)

# __all__ 同时暴露 schema、compatibility 和 adapter API；调用方无需知道内部文件拆分。
__all__ = [
    "SNAPSHOT_SCHEMA",
    "JointMapping",
    "ObjectCompatibilityMapping",
    "ObjectSnapshot",
    "ObjectTargetDescriptor",
    "RobotCompatibilityMapping",
    "RobotSnapshot",
    "RobotTargetDescriptor",
    "SimulationSnapshot",
    "SnapshotCompatibilityError",
    "SnapshotCompatibilityResult",
    "SnapshotMetadata",
    "SnapshotRestoreResult",
    "SnapshotTargetDescriptor",
    "check_snapshot_compatibility",
    "clone_tiled_env_state",
    "get_single_scene_snapshot",
    "get_snapshot",
    "get_tiled_scene_snapshot",
    "require_snapshot_compatibility",
    "set_single_scene_snapshot",
    "set_snapshot",
    "set_tiled_scene_snapshot",
    "single_scene_target_descriptor",
    "tiled_scene_target_descriptor",
]
