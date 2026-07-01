"""Interactive motion JSON protocol.

This module parses plain JSON-compatible mappings into motion specs and control
commands. It intentionally does not import Isaac or create cuMotion contexts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from linkerbot_sim.app.motion.specs import (
    CSpaceDeltaPlanMoveSpec,
    CSpaceGoalPlanMoveSpec,
    CommandOverlaySpec,
    CumotionMoveSpec,
    DualHandMoveSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    MoveSpec,
    SpecifiedPathMoveSpec,
)
from linkerbot_sim.planning.requests import (
    IKRequest,
    TaskSpacePath,
    TcpArcSegment,
    TcpLineSegment,
)


InteractiveCommandKind = Literal[
    "moves",
    "hold",
    "status",
    "cancel",
    "cancel_current",
    "estop",
    "quit",
]


@dataclass(frozen=True)
class InteractiveMotionCommand:
    """Parsed interactive command."""

    kind: InteractiveCommandKind
    command_id: str | None = None
    moves: tuple[MoveSpec, ...] = ()
    duration_s: float | None = None
    cancel_id: str | None = None
    status_id: str | None = None


def parse_interactive_motion_message(
    message: Mapping[str, object],
    *,
    default_tcp_by_side: Mapping[str, str],
) -> InteractiveMotionCommand:
    """Parse one JSON object into an interactive command."""

    if "moves" in message:
        moves_value = message["moves"]
        if not isinstance(moves_value, Sequence) or isinstance(
            moves_value, (str, bytes)
        ):
            raise ValueError("moves must be a list")
        moves = tuple(
            _parse_move(_expect_mapping(item, "moves[]"), default_tcp_by_side)
            for item in moves_value
        )
        if not moves:
            raise ValueError("moves cannot be empty")
        return InteractiveMotionCommand(
            kind="moves",
            command_id=_optional_str(message.get("id")),
            moves=moves,
        )

    command_type = _required_str(message, "type")
    if command_type in {
        "hand",
        "dual_hand",
        "ik_pose",
        "ik_offset",
        "cspace_goal",
        "cspace_delta",
        "task_space_line",
        "task_space_arc",
    }:
        return InteractiveMotionCommand(
            kind="moves",
            command_id=_optional_str(message.get("id")),
            moves=(_parse_move(message, default_tcp_by_side),),
        )
    if command_type == "hold":
        return InteractiveMotionCommand(
            kind="hold",
            command_id=_optional_str(message.get("id")),
            duration_s=_optional_float(message.get("duration_s"), default=0.25),
        )
    if command_type == "status":
        return InteractiveMotionCommand(
            kind="status",
            status_id=_optional_str(message.get("id")),
        )
    if command_type == "cancel":
        cancel_id = _optional_str(message.get("id"))
        if cancel_id is None:
            raise ValueError("cancel id is required")
        return InteractiveMotionCommand(
            kind="cancel",
            cancel_id=cancel_id,
        )
    if command_type in {"cancel_current", "estop", "quit"}:
        return InteractiveMotionCommand(kind=command_type)
    raise ValueError(f"unsupported interactive command type: {command_type!r}")


def _parse_move(
    message: Mapping[str, object],
    default_tcp_by_side: Mapping[str, str],
) -> MoveSpec:
    move_type = _required_str(message, "type")
    if move_type == "hand":
        side = _required_side(message)
        move = HandMoveSpec(
            side=side,
            joint_positions=_required_joint_positions(message),
            duration_s=_required_float(message, "duration_s"),
            phase=_optional_str(message.get("phase")),
        )
        move.validate()
        return move
    if move_type == "dual_hand":
        duration_s = _required_float(message, "duration_s")
        move = DualHandMoveSpec(
            left=_parse_hand_child(message.get("left"), "left", duration_s=duration_s),
            right=_parse_hand_child(
                message.get("right"), "right", duration_s=duration_s
            ),
            duration_s=duration_s,
            phase=_optional_str(message.get("phase")),
        )
        move.validate()
        return move
    if move_type == "ik_pose":
        side = _required_side(message)
        tcp_frame_name = _tcp_frame_name(message, default_tcp_by_side, side)
        duration_s = _required_float(message, "duration_s")
        move = CumotionMoveSpec(
            request=IKRequest(
                target_position=_vector3(message, "position"),
                target_orientation=_optional_vector4(message, "orientation"),
                tcp_frame_name=tcp_frame_name,
                position_tolerance=_optional_float(
                    message.get("position_tolerance"), default=1.0e-4
                ),
                orientation_tolerance=_optional_float(
                    message.get("orientation_tolerance"), default=1.0e-3
                ),
                avoid_collisions=bool(message.get("avoid_collisions", False)),
            ),
            side=side,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
            execution="selected_side",
            phase=_optional_str(message.get("phase")),
            overlays=_parse_overlays(message.get("overlays"), duration_s=duration_s),
        )
        move.validate(require_side=True)
        return move
    if move_type == "ik_offset":
        side = _required_side(message)
        tcp_frame_name = _tcp_frame_name(message, default_tcp_by_side, side)
        duration_s = _required_float(message, "duration_s")
        move = IkOffsetMoveSpec(
            side=side,
            tcp_frame_name=tcp_frame_name,
            tcp_offset=tuple(_vector3(message, "offset").tolist()),
            duration_s=duration_s,
            phase=_optional_str(message.get("phase")),
            orientation_mode=str(message.get("orientation_mode", "current")),
            target_orientation=_optional_tuple4(message, "orientation"),
            overlays=_parse_overlays(message.get("overlays"), duration_s=duration_s),
        )
        move.validate(require_side=True)
        return move
    if move_type == "cspace_goal":
        side = _required_side(message)
        duration_s = _required_float(message, "duration_s")
        move = CSpaceGoalPlanMoveSpec(
            side=side,
            tcp_frame_name=_tcp_frame_name(message, default_tcp_by_side, side),
            joint_positions=tuple(_required_float_sequence(message, "joint_positions")),
            duration_s=duration_s,
            phase=_optional_str(message.get("phase")),
            overlays=_parse_overlays(message.get("overlays"), duration_s=duration_s),
        )
        move.validate(require_side=True)
        return move
    if move_type == "cspace_delta":
        side = _required_side(message)
        duration_s = _required_float(message, "duration_s")
        move = CSpaceDeltaPlanMoveSpec(
            side=side,
            tcp_frame_name=_tcp_frame_name(message, default_tcp_by_side, side),
            joint_deltas=tuple(_required_float_sequence(message, "joint_deltas")),
            duration_s=duration_s,
            phase=_optional_str(message.get("phase")),
            overlays=_parse_overlays(message.get("overlays"), duration_s=duration_s),
        )
        move.validate(require_side=True)
        return move
    if move_type == "task_space_line":
        side = _required_side(message)
        duration_s = _required_float(message, "duration_s")
        move = SpecifiedPathMoveSpec(
            side=side,
            tcp_frame_name=_tcp_frame_name(message, default_tcp_by_side, side),
            path=TaskSpacePath(segments=(_parse_tcp_line_segment(message),)),
            duration_s=duration_s,
            phase=_optional_str(message.get("phase")),
            overlays=_parse_overlays(message.get("overlays"), duration_s=duration_s),
        )
        move.validate(require_side=True)
        return move
    if move_type == "task_space_arc":
        side = _required_side(message)
        duration_s = _required_float(message, "duration_s")
        move = SpecifiedPathMoveSpec(
            side=side,
            tcp_frame_name=_tcp_frame_name(message, default_tcp_by_side, side),
            path=TaskSpacePath(segments=(_parse_tcp_arc_segment(message),)),
            duration_s=duration_s,
            phase=_optional_str(message.get("phase")),
            overlays=_parse_overlays(message.get("overlays"), duration_s=duration_s),
        )
        move.validate(require_side=True)
        return move
    raise ValueError(f"unsupported move type: {move_type!r}")


def _parse_tcp_line_segment(message: Mapping[str, object]) -> TcpLineSegment:
    return TcpLineSegment(
        target_position=_optional_vector3(message, "target_position"),
        target_offset=_optional_vector3(message, "target_offset"),
        orientation_mode=str(message.get("orientation_mode", "current")),
        target_orientation=_optional_vector4(
            message,
            "target_orientation"
            if "target_orientation" in message
            else "orientation",
        ),
    )


def _parse_tcp_arc_segment(message: Mapping[str, object]) -> TcpArcSegment:
    return TcpArcSegment(
        target_position=_optional_vector3(message, "target_position"),
        target_offset=_optional_vector3(message, "target_offset"),
        intermediate_position=_optional_vector3(message, "intermediate_position"),
        intermediate_offset=_optional_vector3(message, "intermediate_offset"),
        target_orientation=_optional_vector4(
            message,
            "target_orientation"
            if "target_orientation" in message
            else "orientation",
        ),
        arc_mode=str(message.get("arc_mode", "three_point")),
        constant_orientation=bool(message.get("constant_orientation", True)),
    )


def _parse_overlays(value: object, *, duration_s: float) -> tuple[CommandOverlaySpec, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("overlays must be a list")
    overlays = []
    for item in value:
        data = _expect_mapping(item, "overlays[]")
        overlay = CommandOverlaySpec(
            timing=str(data.get("timing", "sync")),
            left_hand=_parse_hand_child(
                data.get("left_hand"), "left", duration_s=duration_s
            ),
            right_hand=_parse_hand_child(
                data.get("right_hand"), "right", duration_s=duration_s
            ),
        )
        overlay.validate()
        overlays.append(overlay)
    return tuple(overlays)


def _parse_hand_child(
    value: object,
    side: str,
    *,
    duration_s: float | None,
) -> HandMoveSpec | None:
    if value is None:
        return None
    data = _expect_mapping(value, f"{side}_hand")
    hand = HandMoveSpec(
        side=side,
        joint_positions=_required_joint_positions(data),
        duration_s=_optional_float(data.get("duration_s"), default=duration_s),
        phase=_optional_str(data.get("phase")),
    )
    hand.validate()
    return hand


def _tcp_frame_name(
    message: Mapping[str, object],
    default_tcp_by_side: Mapping[str, str],
    side: str,
) -> str:
    value = _optional_str(message.get("tcp_frame_name"))
    if value:
        return value
    default = default_tcp_by_side.get(side)
    if not default:
        raise ValueError(f"default TCP frame is missing for side {side!r}")
    return str(default)


def _required_side(message: Mapping[str, object]) -> str:
    side = _required_str(message, "side").lower()
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return side


def _required_str(message: Mapping[str, object], key: str) -> str:
    value = message.get(key)
    if value is None or not str(value):
        raise ValueError(f"{key} is required")
    return str(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _required_float(message: Mapping[str, object], key: str) -> float:
    if key not in message:
        raise ValueError(f"{key} is required")
    return float(message[key])


def _optional_float(value: object, *, default: float | None) -> float | None:
    return default if value is None else float(value)


def _vector3(message: Mapping[str, object], key: str) -> np.ndarray:
    if key not in message:
        raise ValueError(f"{key} is required")
    return np.asarray(message[key], dtype=float).reshape(3)


def _optional_vector3(message: Mapping[str, object], key: str) -> np.ndarray | None:
    return None if key not in message else np.asarray(message[key], dtype=float).reshape(3)


def _optional_vector4(message: Mapping[str, object], key: str) -> np.ndarray | None:
    return None if key not in message else np.asarray(message[key], dtype=float).reshape(4)


def _optional_tuple4(
    message: Mapping[str, object],
    key: str,
) -> tuple[float, float, float, float] | None:
    value = _optional_vector4(message, key)
    if value is None:
        return None
    return tuple(float(item) for item in value.tolist())  # type: ignore[return-value]


def _required_float_sequence(
    message: Mapping[str, object],
    key: str,
) -> tuple[float, ...]:
    if key not in message:
        raise ValueError(f"{key} is required")
    value = message[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a list")
    return tuple(float(item) for item in value)


def _required_joint_positions(
    message: Mapping[str, object],
) -> Mapping[str, float] | tuple[float, ...]:
    if "joint_positions" not in message:
        raise ValueError("joint_positions is required")
    value = message["joint_positions"]
    if isinstance(value, Mapping):
        return {str(name): float(position) for name, position in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(item) for item in value)
    raise ValueError("joint_positions must be a mapping or list")


def _expect_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
