"""selected tiled env 的 object state 读取、初始快照与恢复编排。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from linkerbot_sim.tiled.state.object_views import (
    _read_object_view_state,
    _restore_object_view_state,
)
from linkerbot_sim.tiled.state.usd_pose import (
    apply_prim_local_pose_and_zero_velocity,
    read_prim_world_pose,
)


def read_tiled_object_states(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    env_origins: np.ndarray,
    env_ids: np.ndarray,
    object_pose_views: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """读取 selected env 的 object world/local pose，返回 JSON-compatible dict。

    返回值同时保留 world/local：world pose 便于诊断和同场景恢复，local position 用于跨 env
    clone 与 runtime-neutral snapshot。可用 rigid view 时优先读取 PhysX 状态，否则在主线程从
    USD prim 读取。
    """

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
    objects: dict[str, object] = {}
    for object_name, prim_paths in object_prim_paths.items():
        view_state = _read_object_view_state(
            object_pose_views.get(str(object_name))
            if object_pose_views is not None
            else None,
            object_name=str(object_name),
            env_ids=selected,
            env_origins=origins,
        )
        if view_state is not None:
            objects[str(object_name)] = view_state
            continue
        positions: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        valid_env_ids: list[int] = []
        for env_id in selected:
            if int(env_id) >= len(prim_paths):
                continue
            pose = read_prim_world_pose(stage, str(prim_paths[int(env_id)]))
            if pose is None:
                continue
            position, orientation = pose
            positions.append(position)
            orientations.append(orientation)
            valid_env_ids.append(int(env_id))
        if not positions:
            continue
        position_array = np.vstack(positions)
        orientation_array = np.vstack(orientations)
        local_positions = position_array - origins[np.asarray(valid_env_ids, dtype=int)]
        objects[str(object_name)] = {
            "env_ids": valid_env_ids,
            "positions_world": position_array.tolist(),
            "positions_local": local_positions.tolist(),
            "orientations_wxyz": orientation_array.tolist(),
        }
    return objects


def capture_tiled_object_pose_snapshot(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    env_origins: np.ndarray,
    env_ids: np.ndarray,
    object_pose_views: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """缓存 tiled objects 的初始 env-local pose，供 reset 走统一恢复路径。"""

    state = read_tiled_object_states(
        stage=stage,
        object_prim_paths=object_prim_paths,
        env_origins=env_origins,
        env_ids=env_ids,
        object_pose_views=object_pose_views,
    )
    snapshot: dict[str, dict[str, object]] = {}
    for object_name, object_state in state.items():
        if not isinstance(object_state, Mapping):
            continue
        entry: dict[str, object] = {
            "env_ids": np.asarray(object_state.get("env_ids", ()), dtype=int).reshape(
                -1
            ),
            "positions_world": np.asarray(
                object_state.get("positions_world", ()), dtype=float
            ).reshape(-1, 3),
            "positions_local": np.asarray(
                object_state.get("positions_local", ()), dtype=float
            ).reshape(-1, 3),
            "orientations_wxyz": np.asarray(
                object_state.get("orientations_wxyz", ()), dtype=float
            ).reshape(-1, 4),
        }
        if "body_names" in object_state:
            entry["body_names"] = tuple(
                str(name) for name in object_state["body_names"]
            )
        if "body_positions_world" in object_state:
            entry["body_positions_world"] = np.asarray(
                object_state.get("body_positions_world", ()), dtype=float
            )
        if "body_positions_local" in object_state:
            entry["body_positions_local"] = np.asarray(
                object_state.get("body_positions_local", ()), dtype=float
            )
        if "body_orientations_wxyz" in object_state:
            entry["body_orientations_wxyz"] = np.asarray(
                object_state.get("body_orientations_wxyz", ()), dtype=float
            )
        for field in (
            "linear_velocities",
            "angular_velocities",
            "body_linear_velocities",
            "body_angular_velocities",
        ):
            if field in object_state:
                entry[field] = np.asarray(object_state.get(field, ()), dtype=float)
        snapshot[str(object_name)] = entry
    return snapshot


def restore_tiled_object_pose_snapshot(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    snapshot: Mapping[str, Mapping[str, object]],
    env_ids: np.ndarray,
    env_origins: np.ndarray | None = None,
    object_pose_views: Mapping[str, object] | None = None,
) -> int:
    """把 object pose 写回 selected env，并在 rigid view/prim 路径清零刚体速度。"""

    selected = {int(env_id) for env_id in np.asarray(env_ids, dtype=int).reshape(-1)}
    restored = 0
    for object_name, object_state in snapshot.items():
        paths = object_prim_paths.get(str(object_name))
        if paths is None:
            continue
        snapshot_env_ids = np.asarray(
            object_state.get("env_ids", ()), dtype=int
        ).reshape(-1)
        world_positions = np.asarray(
            object_state.get("positions_world", ()), dtype=float
        ).reshape(-1, 3)
        positions = np.asarray(
            object_state.get("positions_local", ()), dtype=float
        ).reshape(-1, 3)
        orientations = np.asarray(
            object_state.get("orientations_wxyz", ()), dtype=float
        ).reshape(-1, 4)
        linear_velocities = (
            np.asarray(object_state.get("linear_velocities", ()), dtype=float).reshape(
                -1, 3
            )
            if "linear_velocities" in object_state
            else None
        )
        angular_velocities = (
            np.asarray(object_state.get("angular_velocities", ()), dtype=float).reshape(
                -1, 3
            )
            if "angular_velocities" in object_state
            else None
        )
        row_count = min(
            snapshot_env_ids.size,
            positions.shape[0],
            orientations.shape[0],
        )
        object_view = (
            object_pose_views.get(str(object_name))
            if object_pose_views is not None
            else None
        )
        view_restore = getattr(object_view, "restore_object_state", None)
        if callable(view_restore):
            restored += int(
                view_restore(
                    object_name=str(object_name),
                    object_state=object_state,
                    selected_env_ids=selected,
                    env_origins=env_origins,
                )
            )
            continue
        view_restored = _restore_object_view_state(
            object_view,
            object_name=str(object_name),
            snapshot_env_ids=snapshot_env_ids[:row_count],
            positions_world=world_positions[:row_count],
            positions_local=positions[:row_count],
            orientations_wxyz=orientations[:row_count],
            linear_velocities=linear_velocities,
            angular_velocities=angular_velocities,
            env_origins=env_origins,
            selected_env_ids=selected,
        )
        if view_restored is not None:
            restored += view_restored
            continue
        for row in range(row_count):
            env_id = int(snapshot_env_ids[row])
            if env_id not in selected or env_id >= len(paths):
                continue
            if apply_prim_local_pose_and_zero_velocity(
                stage,
                str(paths[env_id]),
                positions[row],
                orientations[row],
            ):
                restored += 1
    return restored


__all__ = [
    "capture_tiled_object_pose_snapshot",
    "read_tiled_object_states",
    "restore_tiled_object_pose_snapshot",
]
