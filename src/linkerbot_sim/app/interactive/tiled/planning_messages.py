"""tiled interactive trajectory/planner message helpers.

本模块只处理“交互消息 -> 轨迹缓冲/planner request”的纯数据转换。它不访问 Isaac
World、stage 或 articulation view，因此 debug runtime 和 Isaac runtime 可以共用同一套
语义，避免后续某一边悄悄漂移。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import uuid4

import numpy as np

from linkerbot_sim.planning.requests import (
    CSpaceWaypointPath,
    TaskSpacePath,
    TcpArcSegment,
    TcpLineSegment,
)
from linkerbot_sim.tiled.planner_manager import (
    TiledPlanningRequest,
    TiledPlanningResult,
    TiledPlanningSegment,
)
from linkerbot_sim.tiled.trajectory import TiledTrajectoryBuffer, TiledTrajectoryOverlay
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


def load_interactive_trajectory(
    buffer: TiledTrajectoryBuffer,
    trajectory: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    robot_name: str,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
) -> dict[str, object]:
    """把交互 trajectory payload 规范化后载入 ``TiledTrajectoryBuffer``。"""

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    payload = _trajectory_payload(trajectory)
    times = _trajectory_times(payload)
    duration_s = max(0.0, float(times[-1]) - float(times[0]))
    full_positions, joint_names = _trajectory_positions_for_command_space(
        payload,
        times=times,
        selected_env_ids=selected,
        current_positions=current_positions,
        command_joint_names=command_joint_names,
    )
    overlays = _trajectory_overlays_from_payload(
        payload,
        current_positions=current_positions,
        command_joint_names=command_joint_names,
        default_duration_s=duration_s,
    )
    loaded = buffer.load(
        robot_name=robot_name,
        env_ids=selected,
        times=times,
        positions=full_positions,
        joint_names=joint_names,
        request_id=_optional_str(payload.get("request_id")),
        source=str(payload.get("source", "interactive")),
        replace=bool(payload.get("replace", True)),
        overlays=overlays,
        append=bool(payload.get("queue", False)),
    )
    return {
        "robot": str(robot_name),
        "env_ids": list(loaded),
        "samples": int(times.size),
        "joint_names": list(joint_names),
        "overlay_count": len(overlays),
    }


def planning_request_from_message(
    message: Mapping[str, object],
    *,
    robot_name: str,
    env_ids: np.ndarray,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    default_sample_dt_s: float,
    default_tcp_frame_name: str | None = None,
) -> TiledPlanningRequest:
    """把交互 ``plan`` 消息转换成后台 planner request。"""

    payload = _plan_payload(message)
    current = np.asarray(current_positions, dtype=float)
    joint_names = tuple(str(name) for name in command_joint_names)
    request_id = str(payload.get("request_id") or f"plan-{uuid4().hex}")
    duration_s = float(payload.get("duration_s", 1.0))
    sample_dt_s = float(payload.get("sample_dt_s", default_sample_dt_s))
    segments = _planning_segments_from_payload(
        payload,
        current_positions=current,
        command_joint_names=joint_names,
        default_duration_s=duration_s,
        default_sample_dt_s=sample_dt_s,
        default_tcp_frame_name=default_tcp_frame_name,
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
        source=str(payload.get("source", "interactive_plan")),
        load_on_success=bool(payload.get("load_on_success", True)),
        replace=bool(payload.get("replace", True)),
        segments=tuple(segments),
        trajectory_overlays=_trajectory_overlays_from_payload(
            payload,
            current_positions=current,
            command_joint_names=joint_names,
            default_duration_s=duration_s,
        ),
        metadata={
            "kind": _planning_kind(payload),
            "segment_kinds": tuple(segment.kind for segment in segments),
        },
    )


def load_ready_planning_results(
    buffer: TiledTrajectoryBuffer,
    results: tuple[TiledPlanningResult, ...],
) -> list[dict[str, object]]:
    """把成功的 ready planner result 写入 trajectory buffer。"""

    loaded: list[dict[str, object]] = []
    for result in results:
        if not result.success or not result.load_on_success:
            continue
        env_ids = buffer.load(
            robot_name=result.robot_name,
            env_ids=result.env_ids,
            times=result.times,
            positions=result.positions,
            joint_names=result.joint_names,
            request_id=result.request_id,
            source=result.source,
            replace=result.replace,
            overlays=result.trajectory_overlays,
        )
        loaded.append(
            {
                "request_id": result.request_id,
                "robot": result.robot_name,
                "env_ids": list(env_ids),
            }
        )
    return loaded


def load_interactive_hand_motion(
    buffer: TiledTrajectoryBuffer,
    hand_payload: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    robot_name: str,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    parent_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """把独立 hand motion 载入 trajectory buffer 队列。"""

    parent = {} if parent_payload is None else parent_payload
    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    duration_s = _required_hand_duration_s(hand_payload, parent)
    current = np.asarray(current_positions, dtype=float)
    if current.ndim != 2 or current.shape != (selected.size, len(command_joint_names)):
        raise ValueError(
            "current_positions must match selected envs and command joints"
        )
    overlay = _hand_overlay_from_payload(
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
    replace = bool(hand_payload.get("replace", parent.get("replace", False)))
    append = bool(hand_payload.get("queue", parent.get("queue", True))) and not replace
    loaded = buffer.load(
        robot_name=robot_name,
        env_ids=selected,
        times=times,
        positions=positions,
        joint_names=command_joint_names,
        request_id=_optional_str(
            hand_payload.get("request_id", parent.get("request_id"))
        ),
        source=str(
            hand_payload.get("source", parent.get("source", "interactive_hand"))
        ),
        replace=replace,
        overlays=(overlay,),
        append=append,
        dynamic_base=append,
    )
    return {
        "robot": str(robot_name),
        "env_ids": list(loaded),
        "duration_s": float(duration_s),
        "joint_names": list(command_joint_names),
        "overlay_count": 1,
        "queued": bool(append),
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


def _trajectory_payload(message: Mapping[str, object]) -> Mapping[str, object]:
    """提取轨迹 payload，支持顶层字段和 ``trajectory`` 子对象。"""

    payload = message.get("trajectory", message)
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory must be a JSON object")
    return payload


def _plan_payload(message: Mapping[str, object]) -> Mapping[str, object]:
    """提取 plan payload，支持顶层字段和 ``plan``/``request`` 子对象。"""

    payload = message.get("plan", message.get("request", message))
    if not isinstance(payload, Mapping):
        raise ValueError("plan payload must be a JSON object")
    return payload


def _planning_segments_from_payload(
    payload: Mapping[str, object],
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    default_duration_s: float,
    default_sample_dt_s: float,
    default_tcp_frame_name: str | None,
) -> tuple[TiledPlanningSegment, ...]:
    """把单条 ``plan`` 转换成 tiled planning segment。"""

    cursor = np.asarray(current_positions, dtype=float).copy()
    segment, _, _ = _planning_segment_from_payload(
        payload,
        current_positions=cursor,
        current_positions_known=True,
        command_joint_names=command_joint_names,
        default_duration_s=default_duration_s,
        default_sample_dt_s=default_sample_dt_s,
        default_tcp_frame_name=default_tcp_frame_name,
        label="plan",
    )
    return (segment,)


def _planning_segment_from_payload(
    payload: Mapping[str, object],
    *,
    current_positions: np.ndarray,
    current_positions_known: bool,
    command_joint_names: tuple[str, ...],
    default_duration_s: float,
    default_sample_dt_s: float,
    default_tcp_frame_name: str | None,
    label: str,
) -> tuple[TiledPlanningSegment, np.ndarray, bool]:
    """解析单个 tiled plan payload。"""

    kind = _planning_kind(payload)
    duration_s = float(payload.get("duration_s", default_duration_s))
    sample_dt_s = float(payload.get("sample_dt_s", default_sample_dt_s))
    if _is_joint_planning_kind(kind, payload):
        if _has_delta_goal(payload, kind) and not current_positions_known:
            raise ValueError(f"{label} joint delta cannot follow a path segment")
        goal = _planning_goal_positions(
            payload,
            kind=kind,
            current_positions=current_positions,
            command_joint_names=command_joint_names,
        )
        return (
            TiledPlanningSegment(
                kind=kind or "joint_position_target",
                duration_s=duration_s,
                sample_dt_s=sample_dt_s,
                goal_positions=goal,
                metadata={"source": label},
            ),
            goal.copy(),
            True,
        )
    if kind == "task_space_line":
        path = TaskSpacePath(segments=(_tcp_line_segment_from_payload(payload),))
    elif kind == "task_space_arc":
        path = TaskSpacePath(segments=(_tcp_arc_segment_from_payload(payload),))
    elif kind == "specified_path":
        path = _specified_path_from_payload(payload)
    else:
        raise ValueError(
            f"{label} unsupported tiled planning kind {kind!r}; "
            "supported kinds are joint_position_target, joint_delta_pos, "
            "task_space_line, task_space_arc, specified_path"
        )
    return (
        TiledPlanningSegment(
            kind=kind,
            duration_s=duration_s,
            sample_dt_s=sample_dt_s,
            path=path,
            tcp_frame_name=_tcp_frame_name(payload, default_tcp_frame_name),
            metadata={"source": label},
        ),
        current_positions,
        False,
    )


def _planning_kind(payload: Mapping[str, object]) -> str:
    """解析 tiled plan 的 kind。"""

    if "move_type" in payload:
        raise ValueError("plan.move_type is not supported; use plan.kind")
    message_type = str(payload.get("type", "")).strip()
    if message_type and message_type != "plan":
        raise ValueError("tiled planning messages must use type='plan' and kind")
    kind = str(payload.get("kind", "")).strip()
    if not kind:
        if "joint_deltas" in payload:
            return "joint_delta_pos"
        if any(key in payload for key in ("joint_positions", "values")):
            return "joint_position_target"
        raise ValueError("plan.kind is required")
    if kind in {"cspace_goal", "cspace_delta"}:
        raise ValueError(
            "tiled plan no longer accepts cspace_goal/cspace_delta; "
            "use kind joint_position_target or joint_delta_pos"
        )
    return kind


def _is_joint_planning_kind(kind: str, payload: Mapping[str, object]) -> bool:
    """判断 payload 是否为关节空间规划段。"""

    return kind in {"joint_position_target", "joint_delta_pos"} or any(
        key in payload for key in ("joint_positions", "joint_deltas", "values")
    )


def _has_delta_goal(payload: Mapping[str, object], kind: str) -> bool:
    """判断关节规划段是否使用相对增量。"""

    return kind == "joint_delta_pos" or "joint_deltas" in payload


def _planning_goal_positions(
    payload: Mapping[str, object],
    *,
    kind: str,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
) -> np.ndarray:
    """解析 plan 目标，支持绝对关节目标和关节增量目标。"""

    joint_names = _optional_str_tuple(payload.get("joint_names"))
    absolute_values = _first_present(payload, ("joint_positions",))
    delta_values = _first_present(payload, ("joint_deltas",))
    if (
        absolute_values is None
        and kind == "joint_position_target"
        and "values" in payload
    ):
        absolute_values = payload["values"]
    if delta_values is None and kind == "joint_delta_pos" and "values" in payload:
        delta_values = payload["values"]
    if absolute_values is not None and delta_values is not None:
        raise ValueError(
            "plan must specify either absolute joint target or joint delta"
        )
    if absolute_values is not None:
        return _command_rows_for_selected(
            absolute_values,
            current_positions=current_positions,
            command_joint_names=command_joint_names,
            joint_names=joint_names,
            base="current",
            label="plan.joint_positions",
        )
    if delta_values is not None:
        delta = _command_rows_for_selected(
            delta_values,
            current_positions=current_positions,
            command_joint_names=command_joint_names,
            joint_names=joint_names,
            base="zero",
            label="plan.joint_deltas",
        )
        return np.asarray(current_positions, dtype=float) + delta
    raise ValueError("plan requires joint_positions or joint_deltas")


def _tcp_line_segment_from_payload(payload: Mapping[str, object]) -> TcpLineSegment:
    """解析 tiled plan 的 task-space line segment。"""

    return TcpLineSegment(
        target_position=_optional_vector3(payload, "target_position"),
        target_offset=_optional_vector3(payload, "target_offset"),
        orientation_mode=str(payload.get("orientation_mode", "current")),
        target_orientation=_optional_task_space_orientation_wxyz(payload),
    )


def _tcp_arc_segment_from_payload(payload: Mapping[str, object]) -> TcpArcSegment:
    """解析 tiled plan 的 task-space arc segment。"""

    return TcpArcSegment(
        target_position=_optional_vector3(payload, "target_position"),
        target_offset=_optional_vector3(payload, "target_offset"),
        intermediate_position=_optional_vector3(payload, "intermediate_position"),
        intermediate_offset=_optional_vector3(payload, "intermediate_offset"),
        target_orientation=_optional_task_space_orientation_wxyz(payload),
        arc_mode=str(payload.get("arc_mode", "three_point")),
        constant_orientation=bool(payload.get("constant_orientation", True)),
    )


def _specified_path_from_payload(
    payload: Mapping[str, object],
) -> CSpaceWaypointPath | TaskSpacePath:
    """解析通用 specified_path payload。"""

    raw_path = payload.get("path", payload)
    path_payload = _expect_mapping(raw_path, "specified_path.path")
    path_type = str(path_payload.get("type", path_payload.get("kind", ""))).strip()
    if path_type in {"", "cspace_waypoints"} and "waypoints" in path_payload:
        waypoints = np.asarray(path_payload["waypoints"], dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 1:
            raise ValueError("specified_path waypoints must have shape (T,D), T>=2")
        return CSpaceWaypointPath(
            waypoints=tuple(
                waypoints[index].copy() for index in range(waypoints.shape[0])
            )
        )
    if path_type == "task_space_line":
        return TaskSpacePath(segments=(_tcp_line_segment_from_payload(path_payload),))
    if path_type == "task_space_arc":
        return TaskSpacePath(segments=(_tcp_arc_segment_from_payload(path_payload),))
    if path_type in {"task_space_segments", "task_space"}:
        raw_segments = path_payload.get("segments")
        if not isinstance(raw_segments, Sequence) or isinstance(
            raw_segments, (str, bytes)
        ):
            raise ValueError("specified_path.path.segments must be a list")
        segments = []
        for index, raw_segment in enumerate(raw_segments):
            segment_payload = _expect_mapping(
                raw_segment,
                f"specified_path.path.segments[{index}]",
            )
            segment_type = str(
                segment_payload.get("type", segment_payload.get("kind", ""))
            ).strip()
            if segment_type == "task_space_line":
                segments.append(_tcp_line_segment_from_payload(segment_payload))
            elif segment_type == "task_space_arc":
                segments.append(_tcp_arc_segment_from_payload(segment_payload))
            else:
                raise ValueError(
                    "specified_path task_space segment type must be "
                    "task_space_line or task_space_arc"
                )
        if not segments:
            raise ValueError("specified_path.path.segments cannot be empty")
        return TaskSpacePath(segments=tuple(segments))
    raise ValueError(
        "specified_path.path.type must be cspace_waypoints, task_space_line, "
        "task_space_arc, or task_space_segments"
    )


def _optional_vector3(payload: Mapping[str, object], key: str) -> np.ndarray | None:
    """解析可选 3D 向量。"""

    if key not in payload:
        return None
    return np.asarray(payload[key], dtype=float).reshape(3)


def _optional_task_space_orientation_wxyz(
    payload: Mapping[str, object],
) -> np.ndarray | None:
    """解析 task-space 目标姿态，兼容 RPY 和 wxyz 四元数输入。"""

    quat_keys = ("target_orientation_quat_wxyz", "orientation_quat_wxyz")
    rpy_keys = ("target_orientation", "orientation")
    quat_values = [key for key in quat_keys if key in payload]
    rpy_values = [key for key in rpy_keys if key in payload]
    if len(quat_values) + len(rpy_values) > 1:
        raise ValueError("task-space orientation must be specified only once")
    if quat_values:
        return np.asarray(payload[quat_values[0]], dtype=float).reshape(4)
    if rpy_values:
        return rpy_xyz_to_quat_wxyz(payload[rpy_values[0]])
    return None


def _tcp_frame_name(
    payload: Mapping[str, object],
    default_tcp_frame_name: str | None,
) -> str | None:
    """解析 task-space/specified-path 使用的 TCP frame。"""

    return _optional_str(payload.get("tcp_frame_name")) or default_tcp_frame_name


def _expect_mapping(value: object, label: str) -> Mapping[str, object]:
    """校验 JSON 子对象。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _command_rows_for_selected(
    values: object,
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    joint_names: tuple[str, ...] | None,
    base: str,
    label: str,
) -> np.ndarray:
    """把 selected-env 关节行补齐到完整 command-space。"""

    current = np.asarray(current_positions, dtype=float)
    if current.ndim != 2:
        raise ValueError("current_positions must have shape (E,D)")
    rows = _selected_variable_width_rows(
        values,
        selected_count=current.shape[0],
        label=label,
    )
    if rows.shape[1] > current.shape[1]:
        raise ValueError(
            f"{label} width {rows.shape[1]} exceeds command width {current.shape[1]}"
        )
    if base == "current":
        full = current.copy()
    elif base == "zero":
        full = np.zeros_like(current)
    else:
        raise ValueError("base must be 'current' or 'zero'")
    if joint_names is None:
        full[:, : rows.shape[1]] = rows
        return full
    if len(joint_names) != rows.shape[1]:
        raise ValueError(f"joint_names expected {rows.shape[1]} names")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names cannot contain duplicates")
    index_by_name = {name: index for index, name in enumerate(command_joint_names)}
    unknown = [name for name in joint_names if name not in index_by_name]
    if unknown:
        if rows.shape[1] == current.shape[1] and _are_generated_command_names(
            command_joint_names
        ):
            return rows
        raise ValueError(f"unknown plan joint_names: {unknown}")
    for source_index, name in enumerate(joint_names):
        full[:, index_by_name[name]] = rows[:, source_index]
    return full


def _selected_variable_width_rows(
    values: object,
    *,
    selected_count: int,
    label: str,
) -> np.ndarray:
    """把 ``(D,)``/``(E,D)`` 行规范化为 selected env 行数。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{label} must have shape (N,D)")
    if array.shape[0] == 1 and int(selected_count) != 1:
        array = np.repeat(array, int(selected_count), axis=0)
    if array.shape[0] != int(selected_count):
        raise ValueError(f"{label} first dimension must be 1 or len(env_ids)")
    return array.astype(float, copy=True)


def _first_present(
    payload: Mapping[str, object], keys: tuple[str, ...]
) -> object | None:
    """返回第一个存在的字段值。"""

    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _trajectory_times(payload: Mapping[str, object]) -> np.ndarray:
    """解析轨迹采样时间。"""

    if "times" not in payload:
        raise ValueError("trajectory.times is required")
    times = np.asarray(payload["times"], dtype=float).reshape(-1)
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
    """把交互轨迹矩阵补齐并映射到 runtime command-space。"""

    if "positions" in payload:
        raw_positions = payload["positions"]
    elif "joint_positions" in payload:
        raw_positions = payload["joint_positions"]
    else:
        raise ValueError("trajectory.positions is required")
    selected = np.asarray(selected_env_ids, dtype=int).reshape(-1)
    current = np.asarray(current_positions, dtype=float)
    command_names = tuple(str(name) for name in command_joint_names)
    if current.ndim != 2 or current.shape != (selected.size, len(command_names)):
        raise ValueError(
            "current_positions must match selected envs and command joints"
        )
    positions = _trajectory_position_batch(
        raw_positions,
        env_count=selected.size,
        sample_count=int(times.size),
    )
    joint_names = _optional_str_tuple(payload.get("joint_names"))
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
        if width == command_dim and _are_generated_command_names(command_names):
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

    array = np.asarray(values, dtype=float)
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


def _trajectory_overlays_from_payload(
    payload: Mapping[str, object],
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    default_duration_s: float,
) -> tuple[TiledTrajectoryOverlay, ...]:
    """解析 tiled hand overlays，并转换成 command-space 列覆盖。"""

    raw = payload.get("overlays")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("overlays must be a list")
    overlays: list[TiledTrajectoryOverlay] = []
    for index, item in enumerate(raw):
        data = _expect_mapping(item, f"overlays[{index}]")
        timing = str(data.get("timing", "sync"))
        if timing not in {"before", "sync", "after"}:
            raise ValueError("tiled hand overlay timing must be before, sync, or after")
        before_count = len(overlays)
        for hand_key in ("left_hand", "right_hand"):
            if data.get(hand_key) is None:
                continue
            hand = _expect_mapping(data.get(hand_key), f"overlays[{index}].{hand_key}")
            overlays.append(
                _hand_overlay_from_payload(
                    hand,
                    current_positions=current_positions,
                    command_joint_names=command_joint_names,
                    duration_s=_overlay_duration_s(
                        hand.get("duration_s", data.get("duration_s")),
                        default_duration_s=default_duration_s,
                    ),
                    timing=timing,
                    label=f"overlays[{index}].{hand_key}",
                )
            )
        if len(overlays) == before_count:
            raise ValueError(f"overlays[{index}] requires left_hand or right_hand")
    return tuple(overlays)


def _hand_overlay_from_payload(
    payload: Mapping[str, object],
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    duration_s: float | None,
    timing: str,
    label: str,
) -> TiledTrajectoryOverlay:
    """把单手 overlay mapping 转成 ``TiledTrajectoryOverlay``。"""

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
        joint_name = str(name)
        if joint_name not in index_by_name:
            raise ValueError(f"unknown tiled hand overlay joint name: {joint_name}")
        joint_indices.append(index_by_name[joint_name])
        target_columns.append(
            _overlay_target_column(
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
    return TiledTrajectoryOverlay(
        joint_indices=tuple(int(index) for index in indices),
        start_positions=starts,
        target_positions=targets,
        duration_s=duration_s,
        timing=timing,
    )


def _overlay_target_column(
    value: object,
    *,
    selected_count: int,
    label: str,
) -> np.ndarray:
    """解析 overlay 单个关节目标，支持标量或 selected-env 向量。"""

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(int(selected_count), float(array), dtype=float)
    if array.ndim == 1:
        if array.size != int(selected_count):
            raise ValueError(f"{label} must be scalar or have len(env_ids) values")
        return array.astype(float, copy=True)
    column = _selected_variable_width_rows(
        array,
        selected_count=selected_count,
        label=label,
    )
    if column.shape[1] != 1:
        raise ValueError(f"{label} must be a scalar or one value per selected env")
    return column[:, 0].copy()


def _overlay_duration_s(value: object, *, default_duration_s: float) -> float:
    """解析 overlay duration；缺省使用所属主轨迹时长。"""

    if value is None:
        return float(default_duration_s)
    duration = float(value)
    if duration < 0.0:
        raise ValueError("overlay duration_s cannot be negative")
    return duration


def _required_hand_duration_s(
    payload: Mapping[str, object],
    parent: Mapping[str, object],
) -> float:
    """解析独立 hand motion 时长。"""

    value = payload.get("duration_s", parent.get("duration_s"))
    if value is None:
        raise ValueError("hand duration_s is required")
    duration = float(value)
    if duration < 0.0:
        raise ValueError("hand duration_s cannot be negative")
    return duration


def _optional_str_tuple(value: object) -> tuple[str, ...] | None:
    """解析可选字符串列表。"""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("joint_names must be a list of strings")
    names = tuple(str(item) for item in value)
    if not names:
        raise ValueError("joint_names cannot be empty")
    return names


def _are_generated_command_names(names: tuple[str, ...]) -> bool:
    """判断是否为内部测试替身生成的 joint_0/joint_1/... 名称。"""

    return all(name == f"joint_{index}" for index, name in enumerate(names))


def _optional_str(value: object) -> str | None:
    """解析可选非空字符串。"""

    if value is None:
        return None
    result = str(value)
    if not result:
        raise ValueError("string value cannot be empty")
    return result
