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
    RawJointSequenceMoveSpec,
    RawJointSequenceSideSpec,
    SpecifiedPathMoveSpec,
)
from linkerbot_sim.planning.requests import (
    IKRequest,
    TaskSpacePath,
    TcpArcSegment,
    TcpLineSegment,
)
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


InteractiveCommandKind = Literal[
    "moves",
    "hold",
    "reset",
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
    reset_id: str | None = None
    reset_mode: str = "runtime"
    reset_clear_queue: bool = True
    reset_hold_after_reset: bool = True


def parse_interactive_motion_message(
    message: Mapping[str, object],
    *,
    default_tcp_by_side: Mapping[str, str],
    default_side: str | None = None,
) -> InteractiveMotionCommand:
    """Parse one JSON object into an interactive command."""

    if "moves" in message:
        moves_value = message["moves"]
        if not isinstance(moves_value, Sequence) or isinstance(
            moves_value, (str, bytes)
        ):
            raise ValueError("moves must be a list")
        moves = tuple(
            _parse_move(
                _expect_mapping(item, "moves[]"),
                default_tcp_by_side,
                default_side=default_side,
            )
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
        "raw_joint_sequence",
    }:
        return InteractiveMotionCommand(
            kind="moves",
            command_id=_optional_str(message.get("id")),
            moves=(
                _parse_move(
                    message,
                    default_tcp_by_side,
                    default_side=default_side,
                ),
            ),
        )
    if command_type == "hold":
        return InteractiveMotionCommand(
            kind="hold",
            command_id=_optional_str(message.get("id")),
            duration_s=_optional_float(message.get("duration_s"), default=0.25),
        )
    if command_type == "reset":
        mode = str(message.get("mode", "runtime"))
        if mode != "runtime":
            raise ValueError("reset mode must be 'runtime'")
        return InteractiveMotionCommand(
            kind="reset",
            reset_id=_optional_str(message.get("id")),
            reset_mode=mode,
            reset_clear_queue=_optional_bool(
                message.get("clear_queue"), default=True, label="clear_queue"
            ),
            reset_hold_after_reset=_optional_bool(
                message.get("hold_after_reset"),
                default=True,
                label="hold_after_reset",
            ),
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
    *,
    default_side: str | None = None,
) -> MoveSpec:
    """解析单条 motion payload，并立即调用 spec.validate 做结构校验。"""

    move_type = _required_str(message, "type")
    if move_type == "hand":
        side = _side_for_message(message, default_side=default_side)
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
    if move_type == "raw_joint_sequence":
        move = _parse_raw_joint_sequence_move(message, default_side=default_side)
        move.validate()
        return move
    if move_type == "ik_pose":
        side = _side_for_message(message, default_side=default_side)
        tcp_frame_name = _tcp_frame_name(message, default_tcp_by_side, side)
        duration_s = _required_float(message, "duration_s")
        move = CumotionMoveSpec(
            request=IKRequest(
                target_position=_vector3(message, "position"),
                target_orientation=_optional_orientation_wxyz(message),
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
        side = _side_for_message(message, default_side=default_side)
        tcp_frame_name = _tcp_frame_name(message, default_tcp_by_side, side)
        duration_s = _required_float(message, "duration_s")
        move = IkOffsetMoveSpec(
            side=side,
            tcp_frame_name=tcp_frame_name,
            tcp_offset=tuple(_vector3(message, "offset").tolist()),
            duration_s=duration_s,
            phase=_optional_str(message.get("phase")),
            orientation_mode=str(message.get("orientation_mode", "current")),
            target_orientation=_optional_orientation_tuple_wxyz(message),
            overlays=_parse_overlays(message.get("overlays"), duration_s=duration_s),
        )
        move.validate(require_side=True)
        return move
    if move_type == "cspace_goal":
        side = _side_for_message(message, default_side=default_side)
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
        side = _side_for_message(message, default_side=default_side)
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
        side = _side_for_message(message, default_side=default_side)
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
        side = _side_for_message(message, default_side=default_side)
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


def _parse_raw_joint_sequence_move(
    message: Mapping[str, object],
    *,
    default_side: str | None = None,
) -> RawJointSequenceMoveSpec:
    """解析原始关节序列命令，支持左右子对象或单侧扁平写法。"""

    left = _parse_raw_joint_sequence_side(message.get("left"), "left")
    right = _parse_raw_joint_sequence_side(message.get("right"), "right")
    if "side" in message or "joint_positions" in message:
        side = _side_for_message(message, default_side=default_side)
        payload = RawJointSequenceSideSpec(
            joint_positions=_required_raw_joint_sequence_positions(message)
        )
        if side == "left":
            left = payload
        else:
            right = payload
    return RawJointSequenceMoveSpec(
        left=left,
        right=right,
        step_interval=_optional_int(message.get("step_interval"), default=1),
        phase=_optional_str(message.get("phase")),
    )


def _parse_raw_joint_sequence_side(
    value: object,
    label: str,
) -> RawJointSequenceSideSpec | None:
    """解析 raw_joint_sequence 的单侧矩阵或关节名到采样序列映射。"""

    if value is None:
        return None
    data = _expect_mapping(value, label)
    return RawJointSequenceSideSpec(
        joint_positions=_required_raw_joint_sequence_positions(data)
    )


def _parse_tcp_line_segment(message: Mapping[str, object]) -> TcpLineSegment:
    """解析 task_space_line 的线段终点、偏移和姿态策略。"""

    return TcpLineSegment(
        target_position=_optional_vector3(message, "target_position"),
        target_offset=_optional_vector3(message, "target_offset"),
        orientation_mode=str(message.get("orientation_mode", "current")),
        target_orientation=_optional_task_space_orientation_wxyz(message),
    )


def _parse_tcp_arc_segment(message: Mapping[str, object]) -> TcpArcSegment:
    """解析 task_space_arc 的三点圆弧/偏移圆弧参数。"""

    return TcpArcSegment(
        target_position=_optional_vector3(message, "target_position"),
        target_offset=_optional_vector3(message, "target_offset"),
        intermediate_position=_optional_vector3(message, "intermediate_position"),
        intermediate_offset=_optional_vector3(message, "intermediate_offset"),
        target_orientation=_optional_task_space_orientation_wxyz(message),
        arc_mode=str(message.get("arc_mode", "three_point")),
        constant_orientation=bool(message.get("constant_orientation", True)),
    )


def _parse_overlays(value: object, *, duration_s: float) -> tuple[CommandOverlaySpec, ...]:
    """解析与主臂运动同步或收尾执行的手部 overlay 命令。"""

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
    """解析嵌套的单手动作；缺省 duration 继承父级动作时长。"""

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
    """读取显式 tcp_frame_name，缺省时回退到该侧 runtime 默认 TCP。"""

    value = _optional_str(message.get("tcp_frame_name"))
    if value:
        return value
    default = default_tcp_by_side.get(side)
    if not default:
        raise ValueError(f"default TCP frame is missing for side {side!r}")
    return str(default)


def _required_side(message: Mapping[str, object]) -> str:
    """读取 side 字段并规范化为 left/right。"""

    return _normalize_side(_required_str(message, "side"), label="side")


def _side_for_message(
    message: Mapping[str, object],
    *,
    default_side: str | None,
) -> str:
    """读取 side；单臂 runtime 可传 default_side 让消息省略 side。"""

    if "side" in message:
        return _required_side(message)
    if default_side is None:
        return _required_side(message)
    return _normalize_side(default_side, label="default_side")


def _normalize_side(value: object, *, label: str) -> str:
    """把 side 字段规范化为 left/right。"""

    side = str(value).lower()
    if side not in {"left", "right"}:
        raise ValueError(f"{label} must be 'left' or 'right', got {side!r}")
    return side


def _required_str(message: Mapping[str, object], key: str) -> str:
    """读取必填字符串字段，空字符串按缺失处理。"""

    value = message.get(key)
    if value is None or not str(value):
        raise ValueError(f"{key} is required")
    return str(value)


def _optional_str(value: object) -> str | None:
    """把可选值转成字符串；None 或空字符串统一视为未提供。"""

    if value is None:
        return None
    text = str(value)
    return text if text else None


def _required_float(message: Mapping[str, object], key: str) -> float:
    """读取必填浮点字段，让 float(...) 负责兼容 int/str 数字。"""

    if key not in message:
        raise ValueError(f"{key} is required")
    return float(message[key])


def _optional_float(value: object, *, default: float | None) -> float | None:
    """读取可选浮点字段；未提供时返回调用方指定默认值。"""

    return default if value is None else float(value)


def _optional_int(value: object, *, default: int) -> int:
    """读取可选整数字段，拒绝 bool 和非整数浮点数。"""

    if value is None:
        return int(default)
    if isinstance(value, bool):
        raise ValueError("integer value is required")
    number = float(value)
    if not number.is_integer():
        raise ValueError("integer value is required")
    return int(number)


def _optional_bool(value: object, *, default: bool, label: str) -> bool:
    """读取可选布尔字段，拒绝字符串等隐式 truthy/falsy 值。"""

    if value is None:
        return bool(default)
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _vector3(message: Mapping[str, object], key: str) -> np.ndarray:
    """读取必填三维向量，并 reshape 成 shape=(3,) 的 numpy 数组。"""

    if key not in message:
        raise ValueError(f"{key} is required")
    return np.asarray(message[key], dtype=float).reshape(3)


def _optional_vector3(message: Mapping[str, object], key: str) -> np.ndarray | None:
    """读取可选三维向量；缺失时返回 None。"""

    return None if key not in message else np.asarray(message[key], dtype=float).reshape(3)


def _optional_vector4(message: Mapping[str, object], key: str) -> np.ndarray | None:
    """读取可选四维向量，通常用于四元数姿态。"""

    return None if key not in message else np.asarray(message[key], dtype=float).reshape(4)


def _optional_orientation_wxyz(
    message: Mapping[str, object],
    *,
    rpy_key: str = "orientation",
    quat_key: str = "orientation_quat_wxyz",
) -> np.ndarray | None:
    """读取交互协议姿态字段，并统一转换为 wxyz 四元数。

    默认姿态字段使用人类可读的 RPY 欧拉角；若调用方需要直接传四元数，应使用显式
    ``*_quat_wxyz`` 字段，避免同一个字段同时承担三维和四维语义。
    """

    has_rpy = rpy_key in message
    has_quat = quat_key in message
    if has_rpy and has_quat:
        raise ValueError(f"{rpy_key} and {quat_key} cannot both be set")
    if has_quat:
        return _optional_vector4(message, quat_key)
    if has_rpy:
        return rpy_xyz_to_quat_wxyz(_vector3(message, rpy_key))
    return None


def _optional_task_space_orientation_wxyz(
    message: Mapping[str, object],
) -> np.ndarray | None:
    """读取 task-space 目标姿态，支持 target_* 字段和通用 orientation 别名。"""

    provided = [
        key
        for key in (
            "target_orientation",
            "target_orientation_quat_wxyz",
            "orientation",
            "orientation_quat_wxyz",
        )
        if key in message
    ]
    if len(provided) > 1:
        raise ValueError(
            "only one orientation field can be set: "
            "target_orientation, target_orientation_quat_wxyz, "
            "orientation, orientation_quat_wxyz"
        )
    if "target_orientation" in message or "target_orientation_quat_wxyz" in message:
        return _optional_orientation_wxyz(
            message,
            rpy_key="target_orientation",
            quat_key="target_orientation_quat_wxyz",
        )
    return _optional_orientation_wxyz(message)


def _optional_orientation_tuple_wxyz(
    message: Mapping[str, object],
    *,
    rpy_key: str = "orientation",
    quat_key: str = "orientation_quat_wxyz",
) -> tuple[float, float, float, float] | None:
    """读取姿态字段并以不可变 wxyz tuple 形式保存到 motion spec。"""

    value = _optional_orientation_wxyz(
        message,
        rpy_key=rpy_key,
        quat_key=quat_key,
    )
    if value is None:
        return None
    return tuple(float(item) for item in value.tolist())  # type: ignore[return-value]


def _required_float_sequence(
    message: Mapping[str, object],
    key: str,
) -> tuple[float, ...]:
    """读取必填浮点序列，常用于 C-space goal/delta。"""

    if key not in message:
        raise ValueError(f"{key} is required")
    value = message[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a list")
    return tuple(float(item) for item in value)


def _required_joint_positions(
    message: Mapping[str, object],
) -> Mapping[str, float] | tuple[float, ...]:
    """读取单帧关节目标，支持按关节名映射或按命令顺序排列的列表。"""

    if "joint_positions" not in message:
        raise ValueError("joint_positions is required")
    value = message["joint_positions"]
    if isinstance(value, Mapping):
        return {str(name): float(position) for name, position in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(item) for item in value)
    raise ValueError("joint_positions must be a mapping or list")


def _required_raw_joint_sequence_positions(
    message: Mapping[str, object],
) -> Mapping[str, tuple[float, ...]] | tuple[tuple[float, ...], ...]:
    """读取多帧原始关节序列，支持列映射或二维矩阵。"""

    if "joint_positions" not in message:
        raise ValueError("joint_positions is required")
    value = message["joint_positions"]
    if isinstance(value, Mapping):
        return {
            str(name): _float_tuple(samples, label=f"joint_positions[{name!r}]")
            for name, samples in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = []
        for row in value:
            rows.append(_float_tuple(row, label="joint_positions[]"))
        return tuple(rows)
    raise ValueError("joint_positions must be a mapping or matrix")


def _float_tuple(value: object, *, label: str) -> tuple[float, ...]:
    """把 JSON list 转成 float tuple，并在报错中保留字段标签。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list")
    return tuple(float(item) for item in value)


def _expect_mapping(value: object, label: str) -> Mapping[str, object]:
    """断言某个 JSON 子节点是 object/mapping。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
