"""canonical ``type='step'`` action JSON 到 ``TiledCommandAction``。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.message_utils import (
    json_number,
    json_numeric_array,
    optional_json_integer,
    optional_json_string,
    reject_unknown_fields,
)
from linkerbot_sim.configs.runtime import PlannerRequestDefaults, RuntimeCommandDefaults
from linkerbot_sim.planning.requests import resolve_orientation_mode
from linkerbot_sim.tiled.control.types import (
    SUPPORTED_COMMAND_KINDS,
    TiledCommandAction,
)


CONTROL_MESSAGE_TYPES = frozenset(
    {
        "status",
        "reset",
        "get_state",
        "set_state",
        "get_snapshot",
        "set_snapshot",
        "clone_state",
        "load_trajectory",
        "step_trajectory",
        "trajectory_status",
        "clear_trajectory",
        "plan",
        "planner_status",
        "cancel_plan",
        "clear_completed",
        "quit",
    }
)


def parse_tiled_action(
    message: Mapping[str, object],
    *,
    planner_defaults: PlannerRequestDefaults | None = None,
    command_defaults: RuntimeCommandDefaults | None = None,
) -> TiledCommandAction:
    """解析 canonical ``step`` 消息。"""

    planner_defaults = planner_defaults or PlannerRequestDefaults()
    defaults = command_defaults or RuntimeCommandDefaults()
    action_message = _action_message(message)
    kind = action_message["kind"]
    assert isinstance(kind, str)
    values = _action_values(kind, action_message)
    decimation = optional_json_integer(
        action_message,
        "decimation",
        label="step.decimation",
    )
    duration_s = _optional_json_float(action_message, "duration_s")
    if (
        kind == "ee_linear_path"
        and "duration_s" not in action_message
        and "decimation" not in action_message
    ):
        duration_s = float(planner_defaults.duration_s)
    sample_dt_s = _optional_json_float(action_message, "sample_dt_s")
    tcp_frame_name = optional_json_string(
        action_message,
        "tcp_frame_name",
        label="step.tcp_frame_name",
    )
    interpolation = (
        optional_json_string(
            action_message,
            "interpolation",
            label="step.interpolation",
        )
        or defaults.joint_interpolation
    )
    pose_reference_frame = (
        optional_json_string(
            action_message,
            "pose_reference_frame",
            label="step.pose_reference_frame",
        )
        or defaults.pose_frame
    )
    target_position = _optional_array(action_message, "target_position")
    target_offset = _optional_array(action_message, "target_offset")
    target_orientation_wxyz = _optional_array(
        action_message, "target_orientation_quat_wxyz"
    )
    orientation_mode = defaults.orientation_mode
    if kind == "ee_linear_path":
        requested_orientation_mode = optional_json_string(
            action_message,
            "orientation_mode",
            label="step.orientation_mode",
        )
        orientation_mode = resolve_orientation_mode(
            requested_mode=requested_orientation_mode,
            requested_mode_is_explicit="orientation_mode" in action_message,
            default_mode=defaults.orientation_mode,
            target_orientation_present=target_orientation_wxyz is not None,
        )
    return TiledCommandAction(
        kind=kind,
        values=values,
        decimation=decimation,
        duration_s=duration_s,
        sample_dt_s=sample_dt_s,
        interpolation=interpolation,
        tcp_frame_name=tcp_frame_name,
        pose_reference_frame=pose_reference_frame,
        target_position=target_position,
        target_offset=target_offset,
        orientation_mode=orientation_mode,
        target_orientation_wxyz=target_orientation_wxyz,
    )


def _action_message(message: Mapping[str, object]) -> Mapping[str, object]:
    """校验唯一的顶层 ``type='step'`` action 结构。"""

    message_type = message.get("type")
    if not isinstance(message_type, str):
        raise ValueError("tiled action type must be a string")
    if message_type != "step":
        if message_type in CONTROL_MESSAGE_TYPES:
            raise ValueError(f"{message_type!r} is a control message, not an action")
        raise ValueError("tiled actions require type='step'")
    kind = message.get("kind")
    if isinstance(kind, str) and kind in SUPPORTED_COMMAND_KINDS:
        _reject_unknown_action_fields(message, kind=kind)
        return message
    raise ValueError(
        "unsupported tiled action; supported kinds are "
        + ", ".join(sorted(SUPPORTED_COMMAND_KINDS))
    )


def _reject_unknown_action_fields(
    message: Mapping[str, object],
    *,
    kind: str,
) -> None:
    """按同步 action kind 拒绝未消费字段。"""

    common = {
        "type",
        "kind",
        "env_ids",
        "robot_id",
        "robot_ids",
        "decimation",
    }
    by_kind = {
        "hold": set(),
        "joint_position_target": {"values", "interpolation"},
        "joint_delta_pos": {"values", "interpolation"},
        "ee_pose_target": {
            "values",
            "interpolation",
            "tcp_frame_name",
            "pose_reference_frame",
        },
        "ee_delta_pos": {
            "values",
            "interpolation",
            "tcp_frame_name",
            "pose_reference_frame",
        },
        "ee_delta_pose": {
            "values",
            "interpolation",
            "tcp_frame_name",
            "pose_reference_frame",
        },
        "ee_linear_path": {
            "values",
            "interpolation",
            "tcp_frame_name",
            "pose_reference_frame",
            "duration_s",
            "sample_dt_s",
            "target_position",
            "target_offset",
            "orientation_mode",
            "target_orientation_quat_wxyz",
        },
    }
    reject_unknown_fields(
        message,
        common | by_kind[kind],
        label=f"tiled action kind {kind!r}",
    )


def _action_values(
    kind: str,
    message: Mapping[str, object],
) -> np.ndarray | None:
    """读取统一 ``values`` 字段；具体列宽由 action DTO 校验。"""

    if kind == "hold":
        if "values" in message:
            raise ValueError("hold does not accept values")
        return None
    linear_fields = {
        "target_position",
        "target_offset",
        "orientation_mode",
        "target_orientation_quat_wxyz",
        "sample_dt_s",
    }
    if kind != "ee_linear_path":
        unexpected = sorted(set(message) & linear_fields)
        if unexpected:
            raise ValueError(
                "linear path fields are only supported by ee_linear_path: "
                + ", ".join(unexpected)
            )
    if "values" in message:
        return json_numeric_array(message["values"], label=f"{kind}.values")
    if kind == "ee_linear_path" and (
        "target_position" in message or "target_offset" in message
    ):
        return None
    raise ValueError(f"{kind}.values is required")


def _optional_array(message: Mapping[str, object], key: str) -> np.ndarray | None:
    """读取可广播的 canonical 数组字段。"""

    if key not in message:
        return None
    return json_numeric_array(message[key], label=f"step.{key}")


def _optional_json_float(message: Mapping[str, object], key: str) -> float | None:
    """严格读取 JSON 数值，不接受 bool 或数字字符串。"""

    if key not in message:
        return None
    return json_number(message[key], label=f"step.{key}")


__all__ = ["parse_tiled_action"]
