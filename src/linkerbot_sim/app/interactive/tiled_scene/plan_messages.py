"""canonical plan JSON 到冻结 request，以及 ready result 到 playback buffer。"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.message_utils import (
    command_rows_for_selected,
    json_number,
    json_numeric_array,
    optional_json_string,
    optional_str_tuple,
    reject_unknown_fields,
)
from linkerbot_sim.configs.runtime import (
    PlannerRequestDefaults,
    RuntimeCommandDefaults,
)
from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    TaskSpacePath,
    TcpLineSegment,
    resolve_orientation_mode,
)
from linkerbot_sim.tiled.planning.types import (
    TiledPlanningRequest,
    TiledPlanningResult,
    TiledPlanningSegment,
)
from linkerbot_sim.tiled.playback.buffer import TiledTrajectoryBuffer


def planning_request_from_message(
    message: Mapping[str, object],
    *,
    robot_name: str,
    env_ids: np.ndarray,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    default_sample_dt_s: float,
    default_tcp_frame_name: str | None = None,
    request_defaults: PlannerRequestDefaults | None = None,
    command_defaults: RuntimeCommandDefaults | None = None,
) -> TiledPlanningRequest:
    """把顶层 canonical ``plan`` 消息冻结成 worker-safe numpy request。"""

    payload = message
    request_defaults = request_defaults or PlannerRequestDefaults()
    command_defaults = command_defaults or RuntimeCommandDefaults()
    _validate_plan_fields(payload)
    current = np.asarray(current_positions, dtype=float)
    joint_names = tuple(str(name) for name in command_joint_names)
    request_id = (
        optional_json_string(payload, "request_id", label="plan.request_id")
        or f"plan-{uuid4().hex}"
    )
    duration_s = _json_float(
        payload,
        "duration_s",
        default=request_defaults.duration_s,
    )
    sample_dt_s = _json_float(
        payload,
        "sample_dt_s",
        default=default_sample_dt_s,
    )
    avoid_collisions = _strict_optional_bool(
        payload,
        "avoid_collisions",
        default=request_defaults.avoid_collisions,
    )
    segments = _planning_segments_from_payload(
        payload,
        current_positions=current,
        command_joint_names=joint_names,
        default_duration_s=duration_s,
        default_sample_dt_s=sample_dt_s,
        default_tcp_frame_name=default_tcp_frame_name,
        default_orientation_mode=command_defaults.orientation_mode,
        avoid_collisions=avoid_collisions,
    )
    single_goal = (
        segments[0].goal_positions
        if len(segments) == 1 and segments[0].goal_positions is not None
        else None
    )
    return TiledPlanningRequest(
        request_id=request_id,
        robot_name=str(robot_name),
        env_ids=tuple(int(env_id) for env_id in env_ids),
        current_positions=current,
        goal_positions=single_goal,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt_s=sample_dt_s,
        source=(
            optional_json_string(payload, "source", label="plan.source")
            or "interactive_plan"
        ),
        load_on_success=_strict_optional_bool(
            payload,
            "load_on_success",
            default=request_defaults.load_on_success,
        ),
        replace=_strict_optional_bool(
            payload,
            "replace",
            default=request_defaults.replace,
        ),
        avoid_collisions=avoid_collisions,
        segments=tuple(segments),
        metadata={
            "kind": _planning_kind(payload),
            "segment_kinds": tuple(segment.kind for segment in segments),
            "avoid_collisions": avoid_collisions,
        },
    )


def load_ready_planning_results(
    buffer: TiledTrajectoryBuffer,
    results: tuple[TiledPlanningResult, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """把成功且允许自动加载的 ready result 写入 trajectory buffer。"""

    loaded: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for result in results:
        if not result.success or not result.load_on_success:
            continue
        try:
            env_ids = buffer.load(
                robot_name=result.robot_name,
                env_ids=result.env_ids,
                times=result.times,
                positions=result.positions,
                joint_names=result.joint_names,
                request_id=result.request_id,
                source=result.source,
                replace=result.replace,
            )
        except ValueError as exc:
            error = str(exc)
            rejected.append(
                {
                    "request_id": result.request_id,
                    "robot": result.robot_name,
                    "env_ids": list(result.env_ids),
                    "error": error,
                    "code": (
                        "playback_capacity_exceeded"
                        if "playback capacity exceeded" in error
                        else "playback_load_rejected"
                    ),
                }
            )
            continue
        loaded.append(
            {
                "request_id": result.request_id,
                "robot": result.robot_name,
                "env_ids": list(env_ids),
            }
        )
    return loaded, rejected


def _planning_segments_from_payload(
    payload: Mapping[str, object],
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    default_duration_s: float,
    default_sample_dt_s: float,
    default_tcp_frame_name: str | None,
    default_orientation_mode: str,
    avoid_collisions: bool,
) -> tuple[TiledPlanningSegment, ...]:
    """把单条 ``plan`` 转换成 tiled planning segment。"""

    return (
        _planning_segment_from_payload(
            payload,
            current_positions=np.asarray(current_positions, dtype=float),
            command_joint_names=command_joint_names,
            default_duration_s=default_duration_s,
            default_sample_dt_s=default_sample_dt_s,
            default_tcp_frame_name=default_tcp_frame_name,
            default_orientation_mode=default_orientation_mode,
            avoid_collisions=avoid_collisions,
            label="plan",
        ),
    )


def _planning_segment_from_payload(
    payload: Mapping[str, object],
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    default_duration_s: float,
    default_sample_dt_s: float,
    default_tcp_frame_name: str | None,
    default_orientation_mode: str,
    avoid_collisions: bool,
    label: str,
) -> TiledPlanningSegment:
    """解析单个 tiled plan payload。"""

    kind = _planning_kind(payload)
    duration_s = _json_float(payload, "duration_s", default=default_duration_s)
    sample_dt_s = _json_float(
        payload,
        "sample_dt_s",
        default=default_sample_dt_s,
    )
    if _is_joint_planning_kind(kind):
        goal = _planning_goal_positions(
            payload,
            kind=kind,
            current_positions=current_positions,
            command_joint_names=command_joint_names,
        )
        return TiledPlanningSegment(
            kind=kind,
            duration_s=duration_s,
            sample_dt_s=sample_dt_s,
            goal_positions=goal,
            metadata={"source": label},
        )
    if kind == "linear_pose_path":
        path = TaskSpacePath(
            segments=(
                _tcp_line_segment_from_payload(
                    payload,
                    default_orientation_mode=default_orientation_mode,
                ),
            )
        )
        LinearPosePathRequest(
            current_q=np.asarray(current_positions[0], dtype=float),
            path=path,
            tcp_frame_name=_tcp_frame_name(payload, default_tcp_frame_name),
            duration_s=duration_s,
            sample_dt_s=sample_dt_s,
            avoid_collisions=avoid_collisions,
        ).validate_structure()
    else:
        raise ValueError(
            f"{label} unsupported tiled planning kind {kind!r}; "
            "supported kinds are joint_position_target, joint_delta_pos, "
            "linear_pose_path"
        )
    return TiledPlanningSegment(
        kind=kind,
        duration_s=duration_s,
        sample_dt_s=sample_dt_s,
        path=path,
        tcp_frame_name=_tcp_frame_name(payload, default_tcp_frame_name),
        metadata={"source": label},
    )


def _planning_kind(payload: Mapping[str, object]) -> str:
    """读取显式 plan kind，不根据目标字段猜测请求类型。"""

    message_type = payload.get("type")
    if not isinstance(message_type, str):
        raise ValueError("plan.type must be a string")
    if message_type != "plan":
        raise ValueError("tiled planning messages must use type='plan' and kind")
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("plan.kind must be a non-empty string")
    return kind


def _validate_plan_fields(payload: Mapping[str, object]) -> None:
    """按 plan kind 拒绝未消费字段和 Scene-only 请求策略。"""

    kind = _planning_kind(payload)
    common = {
        "type",
        "kind",
        "robot_id",
        "env_ids",
        "request_id",
        "duration_s",
        "sample_dt_s",
        "source",
        "load_on_success",
        "replace",
        "avoid_collisions",
    }
    by_kind = {
        "joint_position_target": {"joint_positions", "joint_names"},
        "joint_delta_pos": {"joint_deltas", "joint_names"},
        "linear_pose_path": {
            "target_position",
            "target_offset",
            "orientation_mode",
            "target_orientation_quat_wxyz",
            "tcp_frame_name",
        },
    }
    if kind not in by_kind:
        return
    reject_unknown_fields(
        payload,
        common | by_kind[kind],
        label=f"plan kind {kind!r}",
    )


def _is_joint_planning_kind(kind: str) -> bool:
    """返回 kind 是否可直接转换成 command-space joint goal。"""

    return kind in {"joint_position_target", "joint_delta_pos"}


def _planning_goal_positions(
    payload: Mapping[str, object],
    *,
    kind: str,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
) -> np.ndarray:
    """解析绝对关节目标或关节增量目标。"""

    joint_names = optional_str_tuple(
        payload,
        "joint_names",
        label="plan.joint_names",
    )
    if kind == "joint_position_target":
        if "joint_positions" not in payload:
            raise ValueError("plan.joint_positions is required")
        return command_rows_for_selected(
            payload["joint_positions"],
            current_positions=current_positions,
            command_joint_names=command_joint_names,
            joint_names=joint_names,
            base="current",
            label="plan.joint_positions",
        )
    if "joint_deltas" not in payload:
        raise ValueError("plan.joint_deltas is required")
    delta = command_rows_for_selected(
        payload["joint_deltas"],
        current_positions=current_positions,
        command_joint_names=command_joint_names,
        joint_names=joint_names,
        base="zero",
        label="plan.joint_deltas",
    )
    return np.asarray(current_positions, dtype=float) + delta


def _tcp_line_segment_from_payload(
    payload: Mapping[str, object],
    *,
    default_orientation_mode: str,
) -> TcpLineSegment:
    """解析 tiled plan 的 task-space line segment。"""

    target_orientation = _optional_task_space_orientation_wxyz(payload)
    requested_orientation_mode = optional_json_string(
        payload,
        "orientation_mode",
        label="plan.orientation_mode",
    )
    return TcpLineSegment(
        target_position=_optional_vector3(payload, "target_position"),
        target_offset=_optional_vector3(payload, "target_offset"),
        orientation_mode=resolve_orientation_mode(
            requested_mode=requested_orientation_mode,
            requested_mode_is_explicit="orientation_mode" in payload,
            default_mode=default_orientation_mode,
            target_orientation_present=target_orientation is not None,
        ),
        target_orientation=target_orientation,
    )


def _optional_vector3(payload: Mapping[str, object], key: str) -> np.ndarray | None:
    """读取可选 task-space 三维向量，并固定为 shape=(3,)。"""

    if key not in payload:
        return None
    return json_numeric_array(payload[key], label=f"plan.{key}").reshape(3)


def _optional_task_space_orientation_wxyz(
    payload: Mapping[str, object],
) -> np.ndarray | None:
    """读取 canonical wxyz 目标姿态。"""

    if "target_orientation_quat_wxyz" not in payload:
        return None
    return json_numeric_array(
        payload["target_orientation_quat_wxyz"],
        label="plan.target_orientation_quat_wxyz",
    ).reshape(4)


def _tcp_frame_name(
    payload: Mapping[str, object],
    default_tcp_frame_name: str | None,
) -> str | None:
    """按 payload 显式 frame 优先、runtime 默认其次解析 TCP frame。"""

    return (
        optional_json_string(
            payload,
            "tcp_frame_name",
            label="plan.tcp_frame_name",
        )
        or default_tcp_frame_name
    )


def _strict_optional_bool(
    payload: Mapping[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    """严格读取可选请求 boolean，并保留显式 false。"""

    if key not in payload:
        return bool(default)
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"plan.{key} must be a boolean")
    return value


def _json_float(
    payload: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    """严格读取 plan 数值，不接受 bool 或数字字符串。"""

    if key not in payload:
        return float(default)
    return json_number(payload[key], label=f"plan.{key}")


__all__ = ["load_ready_planning_results", "planning_request_from_message"]
