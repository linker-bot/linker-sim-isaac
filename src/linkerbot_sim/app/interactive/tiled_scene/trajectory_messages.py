"""canonical trajectory JSON 的纯解析与 ``TiledTrajectoryBuffer`` 写入。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.message_utils import (
    are_generated_command_names,
    json_numeric_array,
    optional_json_string,
    optional_str_tuple,
    reject_unknown_fields,
    strict_optional_bool,
)
from linkerbot_sim.tiled.playback.buffer import TiledTrajectoryBuffer


def load_interactive_trajectory(
    buffer: TiledTrajectoryBuffer,
    trajectory: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    robot_name: str,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
) -> dict[str, object]:
    """解析顶层 trajectory payload，并在全部校验完成后写入 buffer。"""

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    payload = trajectory
    reject_unknown_fields(
        payload,
        {
            "type",
            "env_ids",
            "robot_id",
            "times",
            "positions",
            "joint_names",
            "request_id",
            "source",
            "replace",
            "queue",
        },
        label="load_trajectory",
    )
    times = _trajectory_times(payload)
    full_positions, joint_names = _trajectory_positions_for_command_space(
        payload,
        times=times,
        selected_env_ids=selected,
        current_positions=current_positions,
        command_joint_names=command_joint_names,
    )
    loaded = buffer.load(
        robot_name=robot_name,
        env_ids=selected,
        times=times,
        positions=full_positions,
        joint_names=joint_names,
        request_id=optional_json_string(
            payload,
            "request_id",
            label="load_trajectory.request_id",
        ),
        source=(
            optional_json_string(
                payload,
                "source",
                label="load_trajectory.source",
            )
            or "interactive"
        ),
        replace=strict_optional_bool(
            payload,
            "replace",
            default=True,
            label="trajectory",
        ),
        append=strict_optional_bool(
            payload,
            "queue",
            default=False,
            label="trajectory",
        ),
    )
    return {
        "robot": str(robot_name),
        "env_ids": list(loaded),
        "samples": int(times.size),
        "joint_names": list(joint_names),
    }


def single_trajectory_robot_name(
    robot_names: tuple[str, ...] | None,
    *,
    default: str,
) -> str:
    """解析 trajectory_step 这类单机器人语义消息的 robot 名。"""

    if robot_names is None:
        return str(default)
    if len(robot_names) != 1:
        raise ValueError("exactly one robot is required")
    return str(robot_names[0])


def _trajectory_times(payload: Mapping[str, object]) -> np.ndarray:
    """解析严格递增且非空的轨迹采样时间。"""

    if "times" not in payload:
        raise ValueError("trajectory.times is required")
    times = json_numeric_array(
        payload["times"],
        label="trajectory.times",
    ).reshape(-1)
    if times.size == 0:
        raise ValueError("trajectory.times cannot be empty")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory.times must be strictly increasing")
    return times


def _trajectory_positions_for_command_space(
    payload: Mapping[str, object],
    *,
    times: np.ndarray,
    selected_env_ids: np.ndarray,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """把交互轨迹矩阵按 joint name 映射并补齐到 runtime command-space。"""

    if "positions" not in payload:
        raise ValueError("trajectory.positions is required")
    selected = np.asarray(selected_env_ids, dtype=int).reshape(-1)
    current = np.asarray(current_positions, dtype=float)
    command_names = tuple(str(name) for name in command_joint_names)
    if current.ndim != 2 or current.shape != (selected.size, len(command_names)):
        raise ValueError(
            "current_positions must match selected envs and command joints"
        )
    positions = _trajectory_position_batch(
        payload["positions"],
        env_count=selected.size,
        sample_count=int(times.size),
    )
    joint_names = optional_str_tuple(
        payload,
        "joint_names",
        label="trajectory.joint_names",
    )
    width = int(positions.shape[2])
    command_dim = len(command_names)
    if joint_names is None:
        if width > command_dim:
            raise ValueError(
                f"trajectory width {width} exceeds command width {command_dim}"
            )
        full = _fill_trajectory_missing_joints(
            positions,
            current_positions=current,
            command_dim=command_dim,
        )
        return full, command_names
    if len(joint_names) != width:
        raise ValueError(f"joint_names expected {width} names, got {len(joint_names)}")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names cannot contain duplicates")
    index_by_name = {name: index for index, name in enumerate(command_names)}
    unknown = [name for name in joint_names if name not in index_by_name]
    if unknown:
        if width == command_dim and are_generated_command_names(command_names):
            return positions, joint_names
        raise ValueError(f"unknown trajectory joint_names: {unknown}")
    full = np.repeat(current[:, None, :], int(times.size), axis=1)
    for source_index, name in enumerate(joint_names):
        full[:, :, index_by_name[name]] = positions[:, :, source_index]
    return full, command_names


def _trajectory_position_batch(
    values: object,
    *,
    env_count: int,
    sample_count: int,
) -> np.ndarray:
    """把 ``(T,D)`` 或 ``(E,T,D)`` positions 规范化为 ``(E,T,D)``。"""

    array = json_numeric_array(values, label="trajectory.positions")
    if array.ndim == 2:
        if array.shape[0] != int(sample_count):
            raise ValueError("trajectory.positions sample dimension must match times")
        array = np.repeat(array.reshape(1, *array.shape), int(env_count), axis=0)
    elif array.ndim == 3:
        if array.shape[1] != int(sample_count):
            raise ValueError("trajectory.positions sample dimension must match times")
        if array.shape[0] == 1 and int(env_count) != 1:
            array = np.repeat(array, int(env_count), axis=0)
        if array.shape[0] != int(env_count):
            raise ValueError(
                "trajectory.positions env dimension must be 1 or len(env_ids)"
            )
    else:
        raise ValueError("trajectory.positions must have shape (T,D) or (E,T,D)")
    if array.shape[2] < 1:
        raise ValueError("trajectory.positions joint dimension cannot be empty")
    return array.astype(float, copy=True)


def _fill_trajectory_missing_joints(
    positions: np.ndarray,
    *,
    current_positions: np.ndarray,
    command_dim: int,
) -> np.ndarray:
    """用当前 target 补齐未在轨迹中出现的 command joints。"""

    if positions.shape[2] == int(command_dim):
        return positions.astype(float, copy=True)
    full = np.repeat(
        np.asarray(current_positions, dtype=float)[:, None, :],
        positions.shape[1],
        axis=1,
    )
    full[:, :, : positions.shape[2]] = positions
    return full


__all__ = ["load_interactive_trajectory", "single_trajectory_robot_name"]
