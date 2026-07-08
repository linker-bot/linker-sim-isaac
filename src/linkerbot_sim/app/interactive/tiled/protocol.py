"""JSON protocol parsing for tiled interactive commands."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from linkerbot_sim.app.interactive.tiled.command_utils import _optional_int, _optional_str
from linkerbot_sim.tiled import SUPPORTED_COMMAND_KINDS, TiledCommandAction


class _AllRobotsSelection:
    """内部 sentinel: 用户在消息里显式写了 ``robots:"all"``。"""


ALL_ROBOTS = _AllRobotsSelection()
RobotSelection = tuple[str, ...] | _AllRobotsSelection | None

CONTROL_MESSAGE_TYPES = frozenset(
    {
        "status",
        "reset",
        "get_state",
        "set_state",
        "load_trajectory",
        "step_trajectory",
        "trajectory_status",
        "clear_trajectory",
        "plan",
        "planner_status",
        "plan_status",
        "cancel_plan",
        "clear_completed",
        "quit",
    }
)
PLANNING_MESSAGE_TYPES = frozenset(
    {
        "plan",
        "plan_queue",
        "cspace_goal",
        "cspace_delta",
        "task_space_line",
        "task_space_arc",
        "specified_path",
    }
)
HAND_MESSAGE_TYPES = frozenset({"hand", "dual_hand"})
ACTION_ALIASES = {
    "joint_positions": "joint_position_target",
    "joint_deltas": "joint_delta_pos",
    "offset": "ee_delta_pos",
}


def parse_tiled_action(message: Mapping[str, object]) -> TiledCommandAction:
    """把一条 JSON object 解析为 ``TiledCommandAction``。"""

    action_message = _action_message(message)
    kind = str(action_message["kind"])
    values = _action_values(kind, action_message)
    decimation = _optional_int(action_message.get("decimation"))
    tcp_frame_name = _optional_str(action_message.get("tcp_frame_name"))
    interpolation = str(action_message.get("interpolation", "smoothstep"))
    pose_reference_frame = str(action_message.get("pose_reference_frame", "env"))
    return TiledCommandAction(
        kind=kind,
        values=values,
        decimation=decimation,
        interpolation=interpolation,
        tcp_frame_name=tcp_frame_name,
        pose_reference_frame=pose_reference_frame,
    )


def handle_tiled_interactive_message(
    message: Mapping[str, object],
    runtime: object,
) -> dict[str, object]:
    """执行一条 tiled 交互消息，并返回 JSON-compatible response。"""

    try:
        message_type = str(message.get("type", ""))
        if message_type == "status":
            return runtime.status()
        if message_type == "reset":
            return runtime.reset(env_ids=_message_env_ids(message))
        if message_type == "get_state":
            return runtime.get_state(
                env_ids=_message_env_ids(message),
                fields=_message_fields(message),
            )
        if message_type == "set_state":
            state = message.get("state")
            if not isinstance(state, Mapping):
                raise ValueError("set_state.state must be a JSON object")
            return runtime.set_state(state, env_ids=_message_env_ids(message))
        if message_type == "load_trajectory":
            return runtime.load_trajectory(
                message,
                env_ids=_message_env_ids(message),
                robot_name=_message_single_robot_name(message),
            )
        if message_type == "step_trajectory":
            return runtime.step_trajectory(
                env_ids=_message_env_ids(message),
                robot_names=_message_robot_names(message),
                decimation=_optional_int(message.get("decimation")),
            )
        if message_type == "trajectory_status":
            return runtime.trajectory_status(
                env_ids=_message_env_ids(message),
                robot_name=_message_single_robot_name(message),
            )
        if message_type == "clear_trajectory":
            return runtime.clear_trajectory(
                env_ids=_message_env_ids(message),
                robot_name=_message_single_robot_name(message),
            )
        if message_type in HAND_MESSAGE_TYPES:
            return runtime.submit_hand_motion(
                message,
                env_ids=_message_env_ids(message),
                robot_name=(
                    _message_single_robot_name(message)
                    if message_type == "hand"
                    else None
                ),
            )
        if message_type in PLANNING_MESSAGE_TYPES or "moves" in message:
            return runtime.submit_plan(
                message,
                env_ids=_message_env_ids(message),
                robot_name=_message_single_robot_name(message),
            )
        if message_type in {"planner_status", "plan_status"}:
            return runtime.planner_status(
                wait_timeout_s=float(message.get("wait_timeout_s", 0.0)),
            )
        if message_type == "cancel_plan":
            return runtime.cancel_plan(
                request_id=_optional_str(message.get("request_id")),
                env_ids=_message_env_ids(message),
                robot_name=_message_single_robot_name(message),
            )
        if message_type == "clear_completed":
            return runtime.clear_completed(
                request_ids=_message_request_ids(message),
            )
        if message_type == "quit":
            if runtime.quit_event is not None:
                runtime.quit_event.set()
            return {"event": "quit", "accepted": True}
        action = parse_tiled_action(message)
        return runtime.step_action(
            action,
            env_ids=_message_env_ids(message),
            robot_names=_message_robot_names(message),
        )
    except Exception as exc:
        return {"event": "rejected", "error": str(exc)}


def _message_env_ids(message: Mapping[str, object]) -> np.ndarray | None:
    """解析消息中的可选 env_ids。"""

    if "env_ids" not in message:
        return None
    return np.asarray(message["env_ids"], dtype=int)


def _message_fields(message: Mapping[str, object]) -> tuple[str, ...] | None:
    """解析 get_state.fields。"""

    if "fields" not in message:
        return None
    fields = message["fields"]
    if not isinstance(fields, list):
        raise ValueError("fields must be a list of strings")
    return tuple(str(item) for item in fields)


def _message_robot_names(message: Mapping[str, object]) -> RobotSelection:
    """解析 action 消息中的可选机器人选择。

    交互协议使用 ``robot`` 或 ``robots`` 作为面向用户的字段名。返回 ``None``
    表示消息未指定机器人；返回 ``ALL_ROBOTS`` 表示用户显式写了 ``robots:"all"``。
    """

    if "robot_names" in message:
        raise ValueError("robot_names is not supported; use robot or robots")
    if "robot" in message:
        value = str(message["robot"]).strip()
        if not value:
            raise ValueError("robot cannot be empty")
        return (value,)
    if "robots" not in message:
        return None
    value = message["robots"]
    if isinstance(value, str):
        robot_filter = _robot_filter(value)
        return ALL_ROBOTS if robot_filter is None else tuple(sorted(robot_filter))
    if not isinstance(value, list):
        raise ValueError("robots must be 'all', a comma-separated string, or a list")
    names = tuple(str(item) for item in value)
    if not names:
        raise ValueError("robots cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("robots cannot contain duplicates")
    return names


def _message_single_robot_name(message: Mapping[str, object]) -> str | None:
    """解析需要单机器人语义的消息。"""

    if "robot" in message:
        value = str(message["robot"]).strip()
        if not value:
            raise ValueError("robot cannot be empty")
        return value
    if "side" in message:
        value = str(message["side"]).strip()
        if not value:
            raise ValueError("side cannot be empty")
        return value
    names = _message_robot_names(message)
    if names is None:
        return None
    if names is ALL_ROBOTS:
        raise ValueError("exactly one robot is required")
    if len(names) != 1:
        raise ValueError("exactly one robot is required")
    return names[0]


def _message_request_ids(message: Mapping[str, object]) -> str | tuple[str, ...] | None:
    """解析 clear_completed 支持的 request_id/request_ids 字段。"""

    if "request_id" in message:
        return _optional_str(message.get("request_id"))
    if "request_ids" not in message:
        return None
    values = message["request_ids"]
    if not isinstance(values, list):
        raise ValueError("request_ids must be a list of strings")
    result = []
    for item in values:
        if item is None:
            raise ValueError("request_ids cannot contain null")
        result.append(_optional_str(item))
    return tuple(result)


def _robot_filter(value: str) -> set[str] | None:
    """解析 ``all`` 或逗号分隔的机器人逻辑名。"""

    if value.strip().lower() == "all":
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise ValueError("robots must be 'all' or a comma-separated name list")
    return result


def _selected_robot_names(
    articulation_views: Mapping[str, object],
    robot_filter: tuple[str, ...] | set[str] | str | _AllRobotsSelection | None,
) -> tuple[str, ...]:
    """在已有 batched articulation views 中选择机器人。"""

    if robot_filter is ALL_ROBOTS:
        robot_filter = None
    if isinstance(robot_filter, str):
        robot_filter = _robot_filter(robot_filter)
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


def _action_message(message: Mapping[str, object]) -> Mapping[str, object]:
    """提取 action payload，并拒绝旧 motion runtime 的命令类型。"""

    message_type = str(message.get("type", ""))
    if message_type == "step":
        action = message.get("action")
        if action is None and isinstance(message.get("kind"), str):
            return message
        if not isinstance(action, Mapping):
            raise ValueError("step.action must be a JSON object or step.kind must be set")
        return action
    if message_type in SUPPORTED_COMMAND_KINDS:
        return {**dict(message), "kind": message_type}
    kind = message.get("kind")
    if isinstance(kind, str) and kind in SUPPORTED_COMMAND_KINDS:
        return message
    if message_type in CONTROL_MESSAGE_TYPES:
        raise ValueError(f"{message_type!r} is a control message, not an action")
    raise ValueError(
        "unsupported tiled action; supported kinds are "
        + ", ".join(sorted(SUPPORTED_COMMAND_KINDS))
    )


def _action_values(
    kind: str,
    message: Mapping[str, object],
) -> np.ndarray | None:
    """解析 action 数值字段，支持适合交互输入的别名。"""

    if kind == "hold":
        return None
    if "values" in message:
        return np.asarray(message["values"], dtype=float)
    if kind == "joint_position_target" and "joint_positions" in message:
        return np.asarray(message["joint_positions"], dtype=float)
    if kind == "joint_delta_pos" and "joint_deltas" in message:
        return np.asarray(message["joint_deltas"], dtype=float)
    if kind == "ee_pose_target":
        return _pose_target_values(message)
    if kind == "ee_delta_pos" and "offset" in message:
        return np.asarray(message["offset"], dtype=float)
    if kind == "ee_delta_pose":
        return _delta_pose_values(message)
    for alias, alias_kind in ACTION_ALIASES.items():
        if alias in message and alias_kind != kind:
            raise ValueError(f"{alias!r} is an alias for {alias_kind!r}, not {kind!r}")
    raise ValueError(f"{kind}.values is required")


def _pose_target_values(message: Mapping[str, object]) -> np.ndarray:
    """解析 ee_pose_target 的 position + orientation 字段。"""

    if "position" not in message:
        raise ValueError("ee_pose_target.position is required")
    position = np.asarray(message["position"], dtype=float)
    orientation = np.asarray(
        message.get("orientation_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
        dtype=float,
    )
    position = position.reshape(1, -1) if position.ndim == 1 else position
    orientation = orientation.reshape(1, -1) if orientation.ndim == 1 else orientation
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("ee_pose_target.position must have shape (N, 3)")
    if orientation.ndim != 2 or orientation.shape[1] != 4:
        raise ValueError("orientation_quat_wxyz must have shape (N, 4)")
    if orientation.shape[0] == 1 and position.shape[0] != 1:
        orientation = np.repeat(orientation, position.shape[0], axis=0)
    if orientation.shape[0] != position.shape[0]:
        raise ValueError("position and orientation first dimensions must match")
    return np.concatenate([position, orientation], axis=1)


def _delta_pose_values(message: Mapping[str, object]) -> np.ndarray:
    """解析 ee_delta_pose 的交互字段。"""

    if "values" in message:
        return np.asarray(message["values"], dtype=float)
    if "offset" not in message:
        raise ValueError("ee_delta_pose.offset is required when values is omitted")
    offset = np.asarray(message["offset"], dtype=float)
    offset = offset.reshape(1, -1) if offset.ndim == 1 else offset
    if offset.ndim != 2 or offset.shape[1] != 3:
        raise ValueError("ee_delta_pose.offset must have shape (N, 3)")
    if "target_orientation_quat_wxyz" in message:
        orientation = np.asarray(message["target_orientation_quat_wxyz"], dtype=float)
        orientation = orientation.reshape(1, -1) if orientation.ndim == 1 else orientation
        if orientation.ndim != 2 or orientation.shape[1] != 4:
            raise ValueError("target_orientation_quat_wxyz must have shape (N, 4)")
        if orientation.shape[0] == 1 and offset.shape[0] != 1:
            orientation = np.repeat(orientation, offset.shape[0], axis=0)
        if orientation.shape[0] != offset.shape[0]:
            raise ValueError("offset and target orientation first dimensions must match")
        return np.concatenate([offset, orientation], axis=1)
    if "delta_rotvec" in message:
        rotvec = np.asarray(message["delta_rotvec"], dtype=float)
        rotvec = rotvec.reshape(1, -1) if rotvec.ndim == 1 else rotvec
        if rotvec.ndim != 2 or rotvec.shape[1] != 3:
            raise ValueError("delta_rotvec must have shape (N, 3)")
        if rotvec.shape[0] == 1 and offset.shape[0] != 1:
            rotvec = np.repeat(rotvec, offset.shape[0], axis=0)
        if rotvec.shape[0] != offset.shape[0]:
            raise ValueError("offset and delta_rotvec first dimensions must match")
        return np.concatenate([offset, rotvec], axis=1)
    return np.concatenate([offset, np.zeros_like(offset)], axis=1)
