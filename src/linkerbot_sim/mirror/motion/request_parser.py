"""Mirror v1 motion arguments 到 canonical timeline request 的严格解析器。

本模块是接口层与仿真主线程之间的纯数据边界：严格的 Mirror v1 ``arguments``
在这里转换为 ``RobotTimelineRequest``。解析过程不导入 Isaac 或 cuRobo，也不读取
articulation 状态，因此可以在 admission 后、主线程执行前安全完成。

运动请求只有两种入口形状：完整 ``plan_timeline``，或带 ``robot_id`` 的单 segment 简写。
二者最终都会生成同一个 ``RobotTimelineRequest``。robot ID 是会话内索引，``robot_label``
只作为可选一致性断言；坐标系、group 和 segment kind 都必须在请求中明确表达。

本模块采用严格字段白名单，并区分“字段缺失”和显式 ``null``。解析成功意味着结构、标量
类型和固定宽度向量均已通过协议校验，但不意味着目标在当前 runtime 中可达；机器人身份、
关节名称和规划可行性等依赖运行态的信息由主线程在执行边界继续校验。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from linkerbot_sim.configuration.modes.mirror import MirrorConfig
from linkerbot_sim.mirror.motion.timeline.requests import (
    JointGroupTrackRequest,
    RobotMotionUnitRequest,
    RobotTimelineRequest,
    RobotTrackRequest,
    TimelineSegmentRequest,
    joint_positions_mapping,
)
from linkerbot_sim.planning.requests import resolve_orientation_mode


_SINGLE_TRACK_WRAPPER_FIELDS = frozenset(
    {
        "type",
        "id",
        "robot_id",
        "robot_label",
        "group",
        "coordination",
        "force_collision_refresh",
    }
)


_OPERATION_KINDS = {
    "motion.plan_timeline": "plan_timeline",
    "motion.joint_goal": "joint_goal",
    "motion.joint_delta": "joint_delta",
    "motion.joint_trajectory": "joint_trajectory",
    "motion.joint_effort": "joint_effort",
    "motion.plan_cspace_goal": "plan_cspace_goal",
    "motion.plan_cspace_delta": "plan_cspace_delta",
    "motion.ik_pose": "ik_pose",
    "motion.ik_offset": "ik_offset",
    "motion.plan_linear_pose_path": "plan_linear_pose_path",
    "motion.hold": "hold",
}


@dataclass(frozen=True)
class MirrorPlannerDefaults:
    """从后端中立 planning profile 派生的请求缺省值。"""

    duration_s: float
    avoid_collisions: bool
    force_collision_refresh: bool
    coordination: str
    interpolation_dt_s: float
    timeout_s: float


@dataclass(frozen=True)
class MirrorCommandDefaults:
    """从 strict control profile 派生的轨迹语义缺省值。"""

    joint_interpolation: str
    pose_frame: str
    orientation_mode: str


def parse_mirror_motion_request(
    operation: str,
    arguments: Mapping[str, object],
    *,
    request_id: str,
    config: MirrorConfig,
    allow_effort: bool = False,
) -> RobotTimelineRequest:
    """把 Mirror v1 motion operation 解析为唯一 timeline request。

    v1 envelope 已在 interface 层校验；这里再次拒绝 ``type``/``id``，确保旧 flat
    envelope 不能通过 arguments 偷渡。归一化 mapping 只在函数内部存在，不构成 wire API。
    """

    try:
        command_type = _OPERATION_KINDS[operation]
    except KeyError as exc:
        raise ValueError(f"unsupported Mirror motion operation: {operation!r}") from exc
    if not isinstance(arguments, Mapping):
        raise ValueError("motion arguments must be a JSON object")
    forbidden = sorted({"type", "id"}.intersection(arguments))
    if forbidden:
        raise ValueError(
            f"motion arguments do not allow legacy envelope fields: {forbidden}"
        )
    message = {"type": command_type, "id": request_id, **dict(arguments)}
    configured_defaults = config.planning.request_defaults
    planner_defaults = MirrorPlannerDefaults(
        duration_s=configured_defaults.duration_s,
        avoid_collisions=configured_defaults.avoid_collisions,
        force_collision_refresh=configured_defaults.force_collision_refresh,
        coordination=configured_defaults.coordination,
        interpolation_dt_s=configured_defaults.sample_dt_s,
        timeout_s=configured_defaults.timeout_s,
    )
    command_defaults = MirrorCommandDefaults(
        joint_interpolation=config.control.joint_interpolation,
        pose_frame=config.control.pose_frame,
        orientation_mode=config.control.orientation_mode,
    )
    if type(allow_effort) is not bool:
        raise TypeError("allow_effort must be bool")
    if command_type == "plan_timeline":
        request = _parse_timeline_request(
            message,
            planner_defaults=planner_defaults,
            command_defaults=command_defaults,
        )
    else:
        request = _parse_single_track_timeline_request(
            message,
            planner_defaults=planner_defaults,
            command_defaults=command_defaults,
        )
    if not allow_effort:
        for track in request.tracks:
            for unit in track.units:
                for group_track in unit.group_tracks:
                    if any(
                        segment.kind == "joint_effort"
                        for segment in group_track.segments
                    ):
                        raise ValueError(
                            "joint_effort segments require linkerbot.mirror.v2"
                        )
    return request


def _parse_single_track_timeline_request(
    message: Mapping[str, object],
    *,
    planner_defaults: MirrorPlannerDefaults,
    command_defaults: MirrorCommandDefaults,
) -> RobotTimelineRequest:
    """把单 robot segment 简写包装成 track，并复用完整 timeline parser。

    包装只改变 envelope，不另建一套 segment 校验规则；因此简写与完整 timeline 对默认
    值、未知字段及错误路径保持同一语义。
    """

    kind = _required_str(message, "type")
    _reject_unknown_fields(
        message,
        set(_SINGLE_TRACK_WRAPPER_FIELDS) | (_all_segment_fields() - {"kind"}),
        kind or "single segment",
    )
    group = _optional_str(message, "group", label=f"{kind}.group") or "arm"
    segment = {
        key: value
        for key, value in message.items()
        if key not in _SINGLE_TRACK_WRAPPER_FIELDS or key == "force_collision_refresh"
    }
    segment["kind"] = kind
    track = {
        "robot_id": message.get("robot_id"),
        "group": group,
        "segments": [segment],
    }
    if "robot_label" in message:
        track["robot_label"] = message["robot_label"]
    timeline: dict[str, object] = {
        "type": "plan_timeline",
        "coordination": message.get(
            "coordination",
            planner_defaults.coordination,
        ),
        "force_collision_refresh": message.get(
            "force_collision_refresh",
            planner_defaults.force_collision_refresh,
        ),
        "tracks": [track],
    }
    if "id" in message:
        timeline["id"] = message["id"]
    return _parse_timeline_request(
        timeline,
        planner_defaults=planner_defaults,
        command_defaults=command_defaults,
    )


def _parse_timeline_request(
    message: Mapping[str, object],
    *,
    planner_defaults: MirrorPlannerDefaults,
    command_defaults: MirrorCommandDefaults,
) -> RobotTimelineRequest:
    """解析 timeline 顶层 coordination、collision refresh 和 robot tracks。

    输出由冻结的 request 对象和 tuple 组成，可在 transport 线程构造后安全交给主线程，
    不再依赖客户端随后可能修改的列表容器。
    """

    _reject_unknown_fields(
        message,
        {"type", "id", "tracks", "coordination", "force_collision_refresh"},
        "plan_timeline",
    )
    tracks_value = message.get("tracks")
    if not isinstance(tracks_value, list):
        raise ValueError("tracks must be a list")
    tracks = tuple(
        _parse_robot_track_request(
            _expect_mapping(value, f"tracks[{index}]"),
            index,
            planner_defaults=planner_defaults,
            command_defaults=command_defaults,
        )
        for index, value in enumerate(tracks_value)
    )
    return RobotTimelineRequest(
        tracks=tracks,
        coordination=(
            _optional_str(
                message,
                "coordination",
                label="plan_timeline.coordination",
            )
            or planner_defaults.coordination
        ),
        force_collision_refresh=_optional_bool(
            message,
            "force_collision_refresh",
            default=planner_defaults.force_collision_refresh,
            label="plan_timeline.force_collision_refresh",
        ),
        command_id=_optional_str(message, "id", label="plan_timeline.id"),
    )


def _parse_robot_track_request(
    message: Mapping[str, object],
    track_index: int,
    *,
    planner_defaults: MirrorPlannerDefaults,
    command_defaults: MirrorCommandDefaults,
) -> RobotTrackRequest:
    """解析一个 robot track，并强制 ``units`` 与 ``segments`` 二选一。"""

    robot_id = _required_robot_id(message, f"tracks[{track_index}].robot_id")
    has_units = "units" in message
    has_segments = "segments" in message
    allowed_fields = {"robot_id", "robot_label"}
    if has_units:
        allowed_fields.add("units")
    if has_segments:
        allowed_fields.update({"group", "segments"})
    _reject_unknown_fields(message, allowed_fields, f"tracks[{track_index}]")
    if has_units == has_segments:
        raise ValueError(
            f"tracks[{track_index}] requires exactly one of units or segments"
        )
    if has_segments:
        group = (
            _optional_str(
                message,
                "group",
                label=f"tracks[{track_index}].group",
            )
            or "arm"
        ).lower()
        group_track = JointGroupTrackRequest(
            group=group,
            segments=_parse_timeline_segments(
                message.get("segments"),
                f"tracks[{track_index}].segments",
                planner_defaults=planner_defaults,
                command_defaults=command_defaults,
            ),
        )
        units = (RobotMotionUnitRequest(group_tracks=(group_track,)),)
    else:
        values = message.get("units")
        if not isinstance(values, list):
            raise ValueError(f"tracks[{track_index}].units must be a list")
        units = tuple(
            _parse_motion_unit_request(
                _expect_mapping(value, f"tracks[{track_index}].units[{unit_index}]"),
                label=f"tracks[{track_index}].units[{unit_index}]",
                planner_defaults=planner_defaults,
                command_defaults=command_defaults,
            )
            for unit_index, value in enumerate(values)
        )
    return RobotTrackRequest(
        robot_id=robot_id,
        robot_label=_optional_str(
            message,
            "robot_label",
            label=f"tracks[{track_index}].robot_label",
        ),
        units=units,
    )


def _parse_motion_unit_request(
    message: Mapping[str, object],
    *,
    label: str,
    planner_defaults: MirrorPlannerDefaults,
    command_defaults: MirrorCommandDefaults,
) -> RobotMotionUnitRequest:
    """解析同一起点并行执行的 arm/hand ``group_tracks``。"""

    _reject_unknown_fields(message, {"group_tracks"}, label)
    values = message.get("group_tracks")
    if not isinstance(values, list):
        raise ValueError(f"{label}.group_tracks must be a list")
    tracks = []
    for index, value in enumerate(values):
        track = _expect_mapping(value, f"{label}.group_tracks[{index}]")
        _reject_unknown_fields(
            track,
            {"group", "segments"},
            f"{label}.group_tracks[{index}]",
        )
        tracks.append(
            JointGroupTrackRequest(
                group=_required_str(track, "group").lower(),
                segments=_parse_timeline_segments(
                    track.get("segments"),
                    f"{label}.group_tracks[{index}].segments",
                    planner_defaults=planner_defaults,
                    command_defaults=command_defaults,
                ),
            )
        )
    return RobotMotionUnitRequest(group_tracks=tuple(tracks))


def _parse_timeline_segments(
    value: object,
    label: str,
    *,
    planner_defaults: MirrorPlannerDefaults,
    command_defaults: MirrorCommandDefaults,
) -> tuple[TimelineSegmentRequest, ...]:
    """把 JSON segment array 转成不可变 request tuple，并保留精确错误路径。"""

    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(
        _parse_timeline_segment(
            _expect_mapping(item, f"{label}[{index}]"),
            label=f"{label}[{index}]",
            planner_defaults=planner_defaults,
            command_defaults=command_defaults,
        )
        for index, item in enumerate(value)
    )


def _parse_timeline_segment(
    message: Mapping[str, object],
    *,
    label: str,
    planner_defaults: MirrorPlannerDefaults,
    command_defaults: MirrorCommandDefaults,
) -> TimelineSegmentRequest:
    """解析一个 canonical segment，并校验当前 kind 的字段集合。"""

    kind = _required_str(message, "kind")
    planning_kind = kind in {
        "plan_cspace_goal",
        "plan_cspace_delta",
        "ik_pose",
        "ik_offset",
        "plan_linear_pose_path",
    }
    if "avoid_collisions" in message and not planning_kind:
        raise ValueError(
            f"{label}.avoid_collisions is only supported by planning kinds"
        )
    if "force_collision_refresh" in message and not planning_kind:
        raise ValueError(
            f"{label}.force_collision_refresh is only supported by planning kinds"
        )
    if "sample_dt_s" in message and not planning_kind:
        raise ValueError(f"{label}.sample_dt_s is only supported by planning kinds")
    if "interpolation" in message and kind not in {"joint_goal", "joint_delta"}:
        raise ValueError(
            f"{label}.interpolation is only supported by joint_goal/joint_delta"
        )
    if "orientation_mode" in message and kind != "plan_linear_pose_path":
        raise ValueError(
            f"{label}.orientation_mode is only supported by plan_linear_pose_path"
        )
    _reject_unknown_fields(message, _segment_allowed_fields(kind), label)
    joint_positions = None
    if "joint_positions" in message:
        joint_positions = joint_positions_mapping(
            message["joint_positions"], label=f"{label}.joint_positions"
        )
    elif "joint_deltas" in message:
        joint_positions = joint_positions_mapping(
            message["joint_deltas"], label=f"{label}.joint_deltas"
        )
    joint_efforts = None
    if "joint_efforts" in message:
        joint_efforts = joint_positions_mapping(
            message["joint_efforts"], label=f"{label}.joint_efforts"
        )
    target_position = None
    if "target_position" in message:
        target_position = tuple(_vector(message, "target_position", 3, label=label))
    orientation = _optional_vector4(
        message,
        "target_orientation_quat_wxyz",
        label=label,
    )
    offset = (
        None
        if "offset" not in message
        else tuple(_vector(message, "offset", 3, label=label))
    )
    reference_frame = _optional_str(
        message,
        "reference_frame",
        label=f"{label}.reference_frame",
    )
    offset_frame = _optional_str(
        message,
        "offset_frame",
        label=f"{label}.offset_frame",
    )
    if reference_frame is None and kind in {"ik_pose", "plan_linear_pose_path"}:
        reference_frame = command_defaults.pose_frame
    if offset_frame is None and kind in {"ik_offset", "plan_linear_pose_path"}:
        offset_frame = command_defaults.pose_frame
    _validate_reference_frame(reference_frame, f"{label}.reference_frame")
    _validate_reference_frame(offset_frame, f"{label}.offset_frame")
    metadata = message.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label}.metadata must be an object")
    orientation_mode = command_defaults.orientation_mode
    if kind == "plan_linear_pose_path":
        requested_orientation_mode = _optional_str(
            message,
            "orientation_mode",
            label=f"{label}.orientation_mode",
        )
        orientation_mode = resolve_orientation_mode(
            requested_mode=requested_orientation_mode,
            requested_mode_is_explicit="orientation_mode" in message,
            default_mode=command_defaults.orientation_mode,
            target_orientation_present=orientation is not None,
            label=f"{label}.orientation_mode",
            target_label=f"{label}.target_orientation_quat_wxyz",
        )
    return TimelineSegmentRequest(
        kind=kind,
        duration_s=_request_duration_s(
            message,
            kind=kind,
            planner_defaults=planner_defaults,
        ),
        sample_dt_s=(
            _optional_float(
                message,
                "sample_dt_s",
                label=f"{label}.sample_dt_s",
            )
            if "sample_dt_s" in message
            else (planner_defaults.interpolation_dt_s if planning_kind else None)
        ),
        timeout_s=(planner_defaults.timeout_s if planning_kind else None),
        joint_positions=joint_positions,
        joint_efforts=joint_efforts,
        times_s=_optional_float_list(
            message,
            "times_s",
            label=f"{label}.times_s",
        ),
        target_position=target_position,
        target_orientation_wxyz=(
            None if orientation is None else tuple(float(v) for v in orientation)
        ),
        offset=offset,
        tcp_frame_name=_optional_str(
            message,
            "tcp_frame_name",
            label=f"{label}.tcp_frame_name",
        ),
        reference_frame=reference_frame,
        offset_frame=offset_frame,
        interpolation=(
            _optional_str(
                message,
                "interpolation",
                label=f"{label}.interpolation",
            )
            or command_defaults.joint_interpolation
        ),
        orientation_mode=orientation_mode,
        avoid_collisions=_optional_bool(
            message,
            "avoid_collisions",
            default=(planner_defaults.avoid_collisions if planning_kind else False),
            label=f"{label}.avoid_collisions",
        ),
        force_collision_refresh=_optional_bool(
            message,
            "force_collision_refresh",
            default=(
                planner_defaults.force_collision_refresh if planning_kind else False
            ),
            label=f"{label}.force_collision_refresh",
        ),
        phase=_optional_str(message, "phase", label=f"{label}.phase"),
        metadata=metadata,
    )


def _request_duration_s(
    message: Mapping[str, object],
    *,
    kind: str,
    planner_defaults: MirrorPlannerDefaults,
) -> float:
    """请求显式 duration 优先；规划 kind 省略时使用 runtime 默认。"""

    if "duration_s" in message:
        return _required_float(message, "duration_s")
    if kind in {
        "plan_cspace_goal",
        "plan_cspace_delta",
        "ik_pose",
        "ik_offset",
        "plan_linear_pose_path",
    }:
        return float(planner_defaults.duration_s)
    raise ValueError("duration_s is required")


def _segment_allowed_fields(kind: str) -> set[str]:
    """返回一个 canonical Mirror segment kind 唯一允许消费的字段集合。"""

    common = {"kind", "duration_s", "phase", "metadata"}
    planning = {"sample_dt_s", "avoid_collisions", "force_collision_refresh"}
    by_kind = {
        "hold": set(),
        "joint_goal": {"joint_positions", "interpolation"},
        "joint_delta": {"joint_deltas", "interpolation"},
        "joint_trajectory": {"joint_positions", "times_s"},
        "joint_effort": {"joint_efforts"},
        "plan_cspace_goal": {"joint_positions"} | planning,
        "plan_cspace_delta": {"joint_deltas"} | planning,
        "ik_pose": {
            "target_position",
            "target_orientation_quat_wxyz",
            "tcp_frame_name",
            "reference_frame",
        }
        | planning,
        "ik_offset": {
            "offset",
            "target_orientation_quat_wxyz",
            "tcp_frame_name",
            "offset_frame",
        }
        | planning,
        "plan_linear_pose_path": {
            "target_position",
            "offset",
            "target_orientation_quat_wxyz",
            "orientation_mode",
            "tcp_frame_name",
            "reference_frame",
            "offset_frame",
        }
        | planning,
    }
    return common | by_kind.get(kind, set())


def _all_segment_fields() -> set[str]:
    """汇总全部已知 segment 字段，用于校验单 segment 简写 envelope。"""

    fields: set[str] = set()
    for kind in (
        "hold",
        "joint_goal",
        "joint_delta",
        "joint_trajectory",
        "joint_effort",
        "plan_cspace_goal",
        "plan_cspace_delta",
        "ik_pose",
        "ik_offset",
        "plan_linear_pose_path",
    ):
        fields.update(_segment_allowed_fields(kind))
    return fields


def _required_robot_id(message: Mapping[str, object], label: str) -> int:
    """读取非负 JSON 整数 robot ID，不接受 bool、浮点数或字符串转换。"""

    if "robot_id" not in message:
        raise ValueError(f"{label} is required")
    value = message["robot_id"]
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_reference_frame(value: str | None, label: str) -> None:
    """校验 Scene task-space frame；None 表示该 segment 不使用对应 frame。"""

    if value is not None and value not in {"world", "env", "robot_base", "tcp"}:
        raise ValueError(f"{label} must be one of: world, env, robot_base, tcp")


def _required_str(message: Mapping[str, object], key: str) -> str:
    """读取必填非空 JSON 字符串，不把其他标量转换成字符串。"""

    value = message.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(
    message: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> str | None:
    """读取可选非空 JSON 字符串；字段缺失与显式 ``null`` 不等价。"""

    if key not in message:
        return None
    value = message[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _reject_unknown_fields(
    message: Mapping[str, object],
    allowed: set[str],
    label: str,
) -> None:
    """拒绝当前 schema 未声明的字段。"""

    unknown = sorted(str(key) for key in message if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {unknown}")


def _required_float(message: Mapping[str, object], key: str) -> float:
    """读取必填有限 JSON number，不接受隐式类型转换。"""

    if key not in message:
        raise ValueError(f"{key} is required")
    return _json_float(message[key], label=key)


def _optional_float(
    message: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> float | None:
    """读取可选有限 JSON number；显式 ``null`` 不是省略。"""

    if key not in message:
        return None
    return _json_float(message[key], label=label)


def _optional_float_list(
    message: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> tuple[float, ...]:
    """读取可选 JSON number 数组，不把标量广播或包装成单元素数组。"""

    if key not in message:
        return ()
    value = message[key]
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(
        _json_float(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )


def _optional_bool(
    message: Mapping[str, object],
    key: str,
    *,
    default: bool,
    label: str,
) -> bool:
    """读取可选 JSON boolean；不接受 truthy 数字/字符串或显式 ``null``。"""

    if key not in message:
        return bool(default)
    value = message[key]
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _json_float(value: object, *, label: str) -> float:
    """校验一个 JSON number 并返回有限 float，显式排除 bool 子类。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(
    message: Mapping[str, object],
    key: str,
    width: int,
    *,
    label: str,
) -> tuple[float, ...]:
    """读取固定宽度 JSON number 数组，不借助 NumPy 接受字符串转换。"""

    if key not in message:
        raise ValueError(f"{key} is required")
    value = message[key]
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{label}.{key} must have shape ({width},)")
    return tuple(
        _json_float(item, label=f"{label}.{key}[{index}]")
        for index, item in enumerate(value)
    )


def _optional_vector4(
    message: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> tuple[float, ...] | None:
    """读取可选四维向量，通常用于四元数姿态。"""

    if key not in message:
        return None
    return _vector(message, key, 4, label=label)


def _expect_mapping(value: object, label: str) -> Mapping[str, object]:
    """断言某个 JSON 子节点是 object/mapping。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
