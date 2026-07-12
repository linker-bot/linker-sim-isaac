"""snapshot 公共 get/set API 的 canonical runtime 形状分派。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from linkerbot_sim.snapshots.single_scene_adapter import (
    get_single_scene_snapshot,
    set_single_scene_snapshot,
)
from linkerbot_sim.snapshots.schema import (
    SimulationSnapshot,
    SnapshotRestoreResult,
)
from linkerbot_sim.snapshots.tiled_scene_adapter import (
    get_tiled_scene_snapshot,
    set_tiled_scene_snapshot,
)


def get_snapshot(
    runtime: object,
    *,
    env_id: int | None = None,
) -> SimulationSnapshot:
    """根据 canonical runtime 形状分派到 Single Scene 或 Tiled Scene reader。"""

    if _looks_like_single_scene_runtime(runtime):
        return get_single_scene_snapshot(runtime)
    if _looks_like_tiled_scene_runtime(runtime):
        if env_id is None:
            raise ValueError("env_id is required for tiled_scene snapshot reads")
        return get_tiled_scene_snapshot(runtime, env_id=int(env_id))
    raise ValueError("unsupported runtime type for get_snapshot")


def set_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot | Mapping[str, object],
    *,
    env_ids: Sequence[int] | np.ndarray | None = None,
    label_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """根据 canonical runtime 形状分派到 Single Scene 或 Tiled Scene writer。"""

    if _looks_like_single_scene_runtime(runtime):
        return set_single_scene_snapshot(
            runtime,
            snapshot,
            label_map=label_map,
            strict=strict,
        )
    if _looks_like_tiled_scene_runtime(runtime):
        if env_ids is None:
            raise ValueError("env_ids is required for tiled_scene snapshot restores")
        return set_tiled_scene_snapshot(
            runtime,
            snapshot,
            env_ids=env_ids,
            label_map=label_map,
            strict=strict,
        )
    raise ValueError("unsupported runtime type for set_snapshot")


def _looks_like_tiled_scene_runtime(runtime: object) -> bool:
    """识别 debug/Isaac Tiled Scene runtime 的最小公开状态形状。"""

    is_debug = hasattr(runtime, "current_positions") and hasattr(runtime, "adapter")
    return is_debug or hasattr(runtime, "scene")


def _looks_like_single_scene_runtime(runtime: object) -> bool:
    """识别 canonical ``SingleSceneRuntime`` 的注册表形状。"""

    return (
        hasattr(runtime, "robots_by_id")
        and hasattr(runtime, "robot_id_by_label")
        and hasattr(runtime, "robot_registry")
    )


__all__ = ["get_snapshot", "set_snapshot"]
