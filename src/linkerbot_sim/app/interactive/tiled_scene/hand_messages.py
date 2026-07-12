"""canonical hand motion payload 的同步子轨构造与 trajectory buffer 入队。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.message_utils import (
    json_number,
    json_numeric_array,
    optional_json_string,
    reject_unknown_fields,
    selected_variable_width_rows,
    strict_optional_bool,
)
from linkerbot_sim.tiled.playback.buffer import TiledTrajectoryBuffer
from linkerbot_sim.tiled.playback.models import PlaybackJointTrack


def load_interactive_hand_motion(
    buffer: TiledTrajectoryBuffer,
    hand_payload: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    robot_name: str,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
) -> dict[str, object]:
    """把独立 hand motion 解析成同步子轨并载入 trajectory buffer。"""

    reject_unknown_fields(
        hand_payload,
        {
            "type",
            "env_ids",
            "robot_id",
            "duration_s",
            "joint_positions",
            "request_id",
            "source",
            "replace",
            "queue",
        },
        label="hand",
    )
    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    duration_s = _required_hand_duration_s(hand_payload)
    current = np.asarray(current_positions, dtype=float)
    if current.ndim != 2 or current.shape != (
        selected.size,
        len(command_joint_names),
    ):
        raise ValueError(
            "current_positions must match selected envs and command joints"
        )
    joint_track = _hand_joint_track_from_payload(
        hand_payload,
        current_positions=current,
        command_joint_names=command_joint_names,
        duration_s=duration_s,
        timing="sync",
        label="hand",
    )
    times = (
        np.asarray([0.0], dtype=float)
        if duration_s <= 0.0
        else np.asarray([0.0, duration_s], dtype=float)
    )
    positions = np.repeat(current[:, None, :], int(times.size), axis=1)
    replace = strict_optional_bool(
        hand_payload,
        "replace",
        default=False,
        label="hand",
    )
    append = (
        strict_optional_bool(
            hand_payload,
            "queue",
            default=True,
            label="hand",
        )
        and not replace
    )
    loaded = buffer.load(
        robot_name=robot_name,
        env_ids=selected,
        times=times,
        positions=positions,
        joint_names=command_joint_names,
        request_id=optional_json_string(
            hand_payload,
            "request_id",
            label="hand.request_id",
        ),
        source=(
            optional_json_string(
                hand_payload,
                "source",
                label="hand.source",
            )
            or "interactive_hand"
        ),
        replace=replace,
        joint_tracks=(joint_track,),
        append=append,
        dynamic_base=append,
    )
    return {
        "robot": str(robot_name),
        "env_ids": list(loaded),
        "duration_s": float(duration_s),
        "joint_names": list(command_joint_names),
        "joint_track_count": 1,
        "queued": bool(append),
    }


def _hand_joint_track_from_payload(
    payload: Mapping[str, object],
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    duration_s: float | None,
    timing: str,
    label: str,
) -> PlaybackJointTrack:
    """把 joint-name target mapping 转成内部部分关节播放轨。"""

    if "joint_positions" not in payload:
        raise ValueError(f"{label}.joint_positions is required")
    positions = payload["joint_positions"]
    if not isinstance(positions, Mapping):
        raise ValueError(
            f"{label}.joint_positions must be a mapping from joint name to target"
        )
    current = np.asarray(current_positions, dtype=float)
    command_names = tuple(str(name) for name in command_joint_names)
    if current.ndim != 2 or current.shape[1] != len(command_names):
        raise ValueError("current_positions must match command_joint_names")
    index_by_name = {name: index for index, name in enumerate(command_names)}
    joint_indices: list[int] = []
    target_columns: list[np.ndarray] = []
    for name, value in positions.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.joint_positions keys must be non-empty strings")
        joint_name = name
        if joint_name not in index_by_name:
            raise ValueError(f"unknown tiled hand playback joint name: {joint_name}")
        joint_indices.append(index_by_name[joint_name])
        target_columns.append(
            _joint_track_target_column(
                value,
                selected_count=current.shape[0],
                label=f"{label}.joint_positions[{joint_name!r}]",
            )
        )
    if not joint_indices:
        raise ValueError(f"{label}.joint_positions cannot be empty")
    if len(set(joint_indices)) != len(joint_indices):
        raise ValueError(f"{label}.joint_positions contains duplicate joints")
    indices = np.asarray(joint_indices, dtype=int)
    targets = np.stack(target_columns, axis=1)
    starts = current[:, indices]
    return PlaybackJointTrack(
        joint_indices=tuple(int(index) for index in indices),
        start_positions=starts,
        target_positions=targets,
        duration_s=duration_s,
        timing=timing,
    )


def _joint_track_target_column(
    value: object,
    *,
    selected_count: int,
    label: str,
) -> np.ndarray:
    """解析部分关节轨目标，支持标量或 selected-env 向量。"""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        target = json_number(value, label=label)
        return np.full(int(selected_count), target, dtype=float)
    array = json_numeric_array(value, label=label)
    if array.ndim == 1:
        if array.size != int(selected_count):
            raise ValueError(f"{label} must be scalar or have len(env_ids) values")
        return array.astype(float, copy=True)
    column = selected_variable_width_rows(
        array,
        selected_count=selected_count,
        label=label,
    )
    if column.shape[1] != 1:
        raise ValueError(f"{label} must be a scalar or one value per selected env")
    return column[:, 0].copy()


def _required_hand_duration_s(
    payload: Mapping[str, object],
) -> float:
    """解析独立 hand motion 时长。"""

    if "duration_s" not in payload:
        raise ValueError("hand duration_s is required")
    duration = json_number(payload["duration_s"], label="hand.duration_s")
    if duration < 0.0:
        raise ValueError("hand duration_s cannot be negative")
    return duration


__all__ = ["load_interactive_hand_motion"]
