"""tiled 协议与 runtime 共用的 env/robot selector 解析。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class _AllRobotsSelection:
    """内部 sentinel：消息显式指定 ``robot_ids='all'``。"""


ALL_ROBOTS = _AllRobotsSelection()
RobotSelection = tuple[str, ...] | _AllRobotsSelection | None


def message_env_ids(message: Mapping[str, object]) -> np.ndarray | None:
    """解析消息中的可选 ``env_ids``。"""

    if "env_ids" not in message:
        return None
    return _json_integer_array(message["env_ids"], "env_ids")


def message_required_env_ids(message: Mapping[str, object]) -> np.ndarray:
    """解析必须存在的 ``env_ids``。"""

    env_ids = message_env_ids(message)
    if env_ids is None:
        raise ValueError("env_ids is required")
    return env_ids


def message_env_id(message: Mapping[str, object]) -> int:
    """解析 snapshot 读取使用的单个 canonical ``env_id``。"""

    if "env_id" not in message:
        raise ValueError("env_id is required")
    return _json_integer(message["env_id"], "env_id")


def message_source_env_id(message: Mapping[str, object]) -> int:
    """解析 clone_state 的 source env id。"""

    if "source_env_id" not in message:
        raise ValueError("source_env_id is required")
    return _json_integer(message["source_env_id"], "source_env_id")


def message_target_env_ids(message: Mapping[str, object]) -> np.ndarray:
    """解析 clone_state 的 target env ids。"""

    if "target_env_ids" not in message:
        raise ValueError("target_env_ids is required")
    return _json_integer_array(message["target_env_ids"], "target_env_ids")


def _json_integer(value: object, label: str) -> int:
    """只接受 JSON 解码器产生的整数，拒绝 bool 与可转换标量。"""

    if type(value) is not int:
        raise ValueError(f"{label} must be a JSON integer")
    return value


def _json_integer_array(value: object, label: str) -> np.ndarray:
    """解析非空、无重复的 JSON integer 数组，不执行 runtime 范围校验。"""

    if not isinstance(value, list):
        raise ValueError(f"{label} must be a non-empty array of JSON integers")
    result = tuple(
        _json_integer(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{label} cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} cannot contain duplicates")
    return np.asarray(result, dtype=int)


def message_robot_names(
    message: Mapping[str, object],
    *,
    runtime: object | None = None,
) -> RobotSelection:
    """把公开 ``robot_id/robot_ids`` selector 解析成内部 label key。"""

    if "robot_id" in message and "robot_ids" in message:
        raise ValueError("robot_id and robot_ids cannot be combined")
    if "robot_id" in message:
        return (robot_name_for_id(runtime, robot_id(message["robot_id"])),)
    if "robot_ids" in message:
        values = message["robot_ids"]
        if values == "all":
            return ALL_ROBOTS
        if not isinstance(values, list) or not values:
            raise ValueError("robot_ids must be a non-empty list or 'all'")
        ids = tuple(robot_id(value) for value in values)
        if len(set(ids)) != len(ids):
            raise ValueError("robot_ids cannot contain duplicates")
        return tuple(robot_name_for_id(runtime, item) for item in ids)
    return None


def message_single_robot_name(
    message: Mapping[str, object],
    *,
    runtime: object | None = None,
) -> str | None:
    """解析只允许单机器人的消息 selector。"""

    if "robot_id" in message:
        if "robot_ids" in message:
            raise ValueError("robot_id cannot be combined with another selector")
        return robot_name_for_id(runtime, robot_id(message["robot_id"]))
    names = message_robot_names(message, runtime=runtime)
    if names is None:
        return None
    if isinstance(names, _AllRobotsSelection) or len(names) != 1:
        raise ValueError("exactly one robot is required")
    return names[0]


def robot_id(value: object) -> int:
    """校验公开 robot ID 为非负整数。"""

    if type(value) is not int or value < 0:
        raise ValueError("robot ID must be a non-negative integer")
    return value


def robot_name_for_id(runtime: object | None, robot_id_value: int) -> str:
    """通过 runtime registry 把 public robot ID 转成内部稳定 label。"""

    if runtime is None:
        raise ValueError("robot_id selection requires a runtime registry")
    scene = getattr(runtime, "scene", None)
    resolver = getattr(scene, "robot_label", None)
    if callable(resolver):
        return str(resolver(robot_id_value))
    names = tuple(getattr(runtime, "robot_names", ()))
    if robot_id_value < 0 or robot_id_value >= len(names):
        raise ValueError(
            f"unknown robot_id {robot_id_value}; available={list(range(len(names)))}"
        )
    return str(names[robot_id_value])


def selected_robot_names(
    articulation_views: Mapping[str, object],
    robot_filter: tuple[str, ...] | set[str] | _AllRobotsSelection | None,
) -> tuple[str, ...]:
    """在已有 batched articulation views 中选择机器人并校验缺失 label。"""

    if isinstance(robot_filter, _AllRobotsSelection):
        robot_filter = None
    if isinstance(robot_filter, tuple):
        robot_filter = set(robot_filter)
    names = tuple(articulation_views)
    if robot_filter is None:
        return names
    selected = tuple(name for name in names if name in robot_filter)
    missing = sorted(set(robot_filter) - set(names))
    if missing:
        raise ValueError(f"Unknown tiled robot names: {', '.join(missing)}")
    if not selected:
        raise ValueError("No tiled robots selected")
    return selected


__all__ = [
    "ALL_ROBOTS",
    "RobotSelection",
    "message_env_id",
    "message_env_ids",
    "message_required_env_ids",
    "message_robot_names",
    "message_single_robot_name",
    "message_source_env_id",
    "message_target_env_ids",
    "robot_id",
    "robot_name_for_id",
    "selected_robot_names",
]
