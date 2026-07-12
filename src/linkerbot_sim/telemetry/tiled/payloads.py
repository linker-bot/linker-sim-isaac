"""把 tiled state response 转换为 JSON、JointStates 与 scene marker 纯数据。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


SceneMarkers = dict[str, list[tuple[str, np.ndarray]]]


def build_json_payload(
    state_response: Mapping[str, object],
    *,
    event: str,
    trigger_response: Mapping[str, object] | None,
) -> dict[str, object]:
    """构造写入 ``/tiled/state`` 的 JSON payload。"""

    payload = {
        "event": str(event),
        "step": int(state_response.get("step", 0)),
        "time_s": float(state_response.get("time_s", 0.0)),
        "env_ids": list(state_response.get("env_ids", [])),
        "state": state_response.get("state", {}),
    }
    if trigger_response is not None:
        payload["trigger"] = {
            key: value
            for key, value in trigger_response.items()
            if key not in {"state", "joint_positions"}
        }
    return payload


def selected_joint_state_arrays(
    state_response: Mapping[str, object],
    *,
    selected_env_id: int,
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """提取一个 env 的扁平标准 JointStates 数组。"""

    state = state_response.get("state")
    if not isinstance(state, Mapping):
        return None
    row_index = _selected_row_index(
        state_response,
        selected_env_id=selected_env_id,
    )
    if row_index is None:
        return None
    robots = state.get("robots")
    if isinstance(robots, Mapping):
        return _robot_joint_state_arrays(robots, row_index=row_index)
    if "joint_positions" in state:
        positions = np.asarray(state["joint_positions"], dtype=float)
        if positions.ndim != 2 or row_index >= positions.shape[0]:
            return None
        names = [f"command_{index}" for index in range(positions.shape[1])]
        velocities = np.zeros(positions.shape[1], dtype=float)
        return names, positions[row_index], velocities
    return None


def selected_joint_efforts(
    state_response: Mapping[str, object],
    *,
    selected_env_id: int,
) -> np.ndarray | None:
    """按 robot 顺序拼接 primary env 的 measured effort；缺失时返回 None。"""

    state = state_response.get("state")
    if not isinstance(state, Mapping):
        return None
    row_index = _selected_row_index(
        state_response,
        selected_env_id=selected_env_id,
    )
    robots = state.get("robots")
    if row_index is None or not isinstance(robots, Mapping):
        return None
    efforts: list[np.ndarray] = []
    for robot_state in robots.values():
        if not isinstance(robot_state, Mapping):
            continue
        values = robot_state.get("measured_efforts")
        if values is None:
            return None
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or row_index >= array.shape[0]:
            return None
        efforts.append(array[row_index])
    return np.concatenate(efforts) if efforts else None


def selected_scene_markers(
    state_response: Mapping[str, object],
    *,
    selected_env_id: int,
) -> SceneMarkers:
    """提取 selected env 的 object/TCP marker 点。"""

    state = state_response.get("state")
    if not isinstance(state, Mapping):
        return {"objects": [], "tcps": []}
    row_index = _selected_row_index(
        state_response,
        selected_env_id=selected_env_id,
    )
    markers: SceneMarkers = {"objects": [], "tcps": []}
    objects = state.get("objects")
    if isinstance(objects, Mapping):
        for object_name, object_state in objects.items():
            if not isinstance(object_state, Mapping):
                continue
            point = _marker_row_for_env(
                object_state.get("positions_world"),
                object_state=object_state,
                fallback_row_index=row_index,
                selected_env_id=selected_env_id,
            )
            if point is not None:
                markers["objects"].append((str(object_name), point))
    robots = state.get("robots")
    if isinstance(robots, Mapping):
        for robot_name, robot_state in robots.items():
            if not isinstance(robot_state, Mapping):
                continue
            point = _marker_row_for_env(
                robot_state.get("tcp_positions_world"),
                object_state=robot_state,
                fallback_row_index=row_index,
                selected_env_id=selected_env_id,
            )
            if point is not None:
                markers["tcps"].append((str(robot_name), point))
    point = _marker_row_for_env(
        state.get("tcp_positions_world"),
        object_state=state,
        fallback_row_index=row_index,
        selected_env_id=selected_env_id,
    )
    if point is not None:
        markers["tcps"].append(("debug", point))
    return markers


def _selected_row_index(
    state_response: Mapping[str, object],
    *,
    selected_env_id: int,
) -> int | None:
    """把 public env ID 转成当前 response ``env_ids`` 中的局部 row index。"""

    env_ids = tuple(int(item) for item in state_response.get("env_ids", ()))
    if not env_ids:
        return None
    try:
        return env_ids.index(int(selected_env_id))
    except ValueError:
        return None


def _robot_joint_state_arrays(
    robots: Mapping[str, object],
    *,
    row_index: int,
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """按 robot label 拼接一个 env row 的 names、positions 与 velocities。"""

    names: list[str] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for robot_name, robot_state in robots.items():
        if not isinstance(robot_state, Mapping):
            continue
        joint_positions = robot_state.get("joint_positions")
        joint_names = tuple(str(name) for name in robot_state.get("joint_names", ()))
        if joint_positions is None:
            continue
        values = np.asarray(joint_positions, dtype=float)
        if values.ndim != 2 or row_index >= values.shape[0]:
            continue
        if not joint_names:
            joint_names = tuple(f"joint_{index}" for index in range(values.shape[1]))
        velocity = _robot_velocity_row(
            robot_state,
            row_index=row_index,
            width=values.shape[1],
        )
        names.extend(f"{robot_name}/{joint_name}" for joint_name in joint_names)
        positions.append(values[row_index])
        velocities.append(velocity)
    if not names:
        return None
    return names, np.concatenate(positions), np.concatenate(velocities)


def _robot_velocity_row(
    robot_state: Mapping[str, object],
    *,
    row_index: int,
    width: int,
) -> np.ndarray:
    """读取 shape 匹配的 velocity row；缺失或不兼容时返回零向量。"""

    values = robot_state.get("joint_velocities")
    if values is None:
        return np.zeros(int(width), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or row_index >= array.shape[0] or array.shape[1] != int(width):
        return np.zeros(int(width), dtype=float)
    return array[row_index]


def _marker_row_for_env(
    values: object,
    *,
    object_state: Mapping[str, object],
    fallback_row_index: int | None,
    selected_env_id: int,
) -> np.ndarray | None:
    """按 object 自带 env IDs 或 response fallback row 选择 marker position。"""

    if "env_ids" not in object_state:
        return _marker_row(values, row_index=fallback_row_index)
    row_index = _row_index_from_env_ids(
        object_state.get("env_ids"),
        selected_env_id=selected_env_id,
    )
    return _marker_row(values, row_index=row_index)


def _marker_row(values: object, *, row_index: int | None) -> np.ndarray | None:
    """安全读取 `(E,3)` marker matrix 中的一行并返回副本。"""

    if values is None or row_index is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or row_index >= array.shape[0]:
        return None
    return array[row_index].copy()


def _row_index_from_env_ids(
    env_ids: object,
    *,
    selected_env_id: int,
) -> int | None:
    """在任意 env ID sequence 中查找 selected env 的局部 row index。"""

    if env_ids is None:
        return None
    try:
        values = tuple(int(item) for item in env_ids)
    except TypeError:
        return None
    try:
        return values.index(int(selected_env_id))
    except ValueError:
        return None


__all__ = [
    "SceneMarkers",
    "build_json_payload",
    "selected_joint_efforts",
    "selected_joint_state_arrays",
    "selected_scene_markers",
]
