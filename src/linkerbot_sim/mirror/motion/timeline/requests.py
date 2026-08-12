"""Mirror 整数 physics tick timeline 的后端无关请求模型。

这些 dataclass 只保存协议语义，不持有 runtime、solver 或 numpy 轨迹缓存。层级固定为
timeline -> robot track -> motion unit -> joint-group track -> segment：同一 robot 的 unit
串行，不同 group track 在 unit 内共享起点，不同 robot track 从全局 tick 0 并行。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal


SegmentKind = Literal[
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
]


@dataclass(frozen=True)
class TimelineSegmentRequest:
    """一个尚未规划/采样的逻辑 segment。

    direct position kind 使用 ``joint_positions``，direct effort 使用 ``joint_efforts``；
    IK/path kind 使用 task-space 字段。duration
    仍以秒表示，只有 ``TimelinePlanningSession`` 知道 physics dt 并负责转换成整数 tick。
    """

    kind: SegmentKind
    duration_s: float
    joint_positions: object | None = None
    joint_efforts: object | None = None
    times_s: tuple[float, ...] = ()
    target_position: tuple[float, float, float] | None = None
    target_orientation_wxyz: tuple[float, float, float, float] | None = None
    offset: tuple[float, float, float] | None = None
    tcp_frame_name: str | None = None
    reference_frame: str | None = None
    offset_frame: str | None = None
    interpolation: Literal["linear", "smoothstep"] = "smoothstep"
    orientation_mode: Literal["free", "current", "target"] = "current"
    avoid_collisions: bool = False
    force_collision_refresh: bool = False
    phase: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    sample_dt_s: float | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        supported = {
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
        }
        if self.kind not in supported:
            raise ValueError(f"unsupported timeline segment kind: {self.kind!r}")
        duration = float(self.duration_s)
        if not isfinite(duration) or duration < 0:
            raise ValueError("segment duration_s must be finite and non-negative")
        sample_dt = None if self.sample_dt_s is None else float(self.sample_dt_s)
        timeout = None if self.timeout_s is None else float(self.timeout_s)
        planning_kinds = {
            "plan_cspace_goal",
            "plan_cspace_delta",
            "ik_pose",
            "ik_offset",
            "plan_linear_pose_path",
        }
        if sample_dt is not None:
            if self.kind not in planning_kinds:
                raise ValueError(
                    "segment sample_dt_s is only supported by planning kinds"
                )
            if not isfinite(sample_dt) or sample_dt <= 0:
                raise ValueError("segment sample_dt_s must be finite and positive")
        if timeout is not None:
            if self.kind not in planning_kinds:
                raise ValueError(
                    "segment timeout_s is only supported by planning kinds"
                )
            if not isfinite(timeout) or timeout <= 0:
                raise ValueError("segment timeout_s must be finite and positive")
        if self.kind == "joint_effort":
            if self.joint_efforts is None:
                raise ValueError("joint_effort requires joint_efforts")
            if self.joint_positions is not None:
                raise ValueError("joint_effort cannot define joint_positions")
        elif (
            self.kind not in {"hold"}
            and self.joint_positions is None
            and (self.kind not in {"ik_pose", "ik_offset", "plan_linear_pose_path"})
        ):
            raise ValueError(f"{self.kind} requires joint_positions")
        if self.kind != "joint_effort" and self.joint_efforts is not None:
            raise ValueError(f"{self.kind} cannot define joint_efforts")
        if self.kind == "ik_pose":
            if self.target_position is None:
                raise ValueError("ik_pose requires target_position")
            if self.reference_frame is None:
                raise ValueError("ik_pose requires reference_frame")
        if self.kind == "ik_offset":
            if self.offset is None:
                raise ValueError("ik_offset requires offset")
            if self.offset_frame is None:
                raise ValueError("ik_offset requires offset_frame")
        if self.kind == "plan_linear_pose_path":
            if (self.target_position is None) == (self.offset is None):
                raise ValueError(
                    "plan_linear_pose_path requires exactly one of "
                    "target_position or offset"
                )
            if self.target_position is not None and self.reference_frame is None:
                raise ValueError(
                    "plan_linear_pose_path target_position requires reference_frame"
                )
            if self.offset is not None and self.offset_frame is None:
                raise ValueError("plan_linear_pose_path offset requires offset_frame")
            if (
                self.orientation_mode == "target"
                and self.target_orientation_wxyz is None
            ):
                raise ValueError(
                    "plan_linear_pose_path orientation_mode='target' requires "
                    "target_orientation_quat_wxyz"
                )
            if (
                self.orientation_mode != "target"
                and self.target_orientation_wxyz is not None
            ):
                raise ValueError(
                    "plan_linear_pose_path target_orientation_quat_wxyz cannot be "
                    f"combined with orientation_mode={self.orientation_mode!r}"
                )
        if self.interpolation not in {"linear", "smoothstep"}:
            raise ValueError("interpolation must be linear or smoothstep")
        if self.orientation_mode not in {"free", "current", "target"}:
            raise ValueError("orientation_mode must be free, current, or target")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "sample_dt_s", sample_dt)
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "times_s", tuple(float(v) for v in self.times_s))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class JointGroupTrackRequest:
    """同一 articulation 内 arm 或 hand 的串行 segment 轨。"""

    group: Literal["arm", "hand"]
    segments: tuple[TimelineSegmentRequest, ...]

    def __post_init__(self) -> None:
        if self.group not in {"arm", "hand"}:
            raise ValueError("group must be 'arm' or 'hand'")
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("group track segments cannot be empty")
        object.__setattr__(self, "segments", segments)


@dataclass(frozen=True)
class RobotMotionUnitRequest:
    """共享起点并行执行的一组 arm/hand tracks，结束时间取最长子轨。"""

    group_tracks: tuple[JointGroupTrackRequest, ...]

    def __post_init__(self) -> None:
        tracks = tuple(self.group_tracks)
        if not tracks:
            raise ValueError("motion unit group_tracks cannot be empty")
        groups = [track.group for track in tracks]
        if len(groups) != len(set(groups)):
            raise ValueError("motion unit cannot repeat a joint group")
        object.__setattr__(self, "group_tracks", tracks)


@dataclass(frozen=True)
class RobotTrackRequest:
    """一个会话 robot ID 的串行 unit 列表。"""

    robot_id: int
    units: tuple[RobotMotionUnitRequest, ...]
    robot_label: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.robot_id, bool) or int(self.robot_id) < 0:
            raise ValueError("robot_id must be a non-negative integer")
        units = tuple(self.units)
        if not units:
            raise ValueError("robot track units cannot be empty")
        object.__setattr__(self, "robot_id", int(self.robot_id))
        object.__setattr__(self, "units", units)
        if self.robot_label is not None and not str(self.robot_label):
            raise ValueError("robot_label cannot be empty")


@dataclass(frozen=True)
class RobotTimelineRequest:
    """一次原子编译的多机器人请求；任一 track 失败则整次请求不执行。"""

    tracks: tuple[RobotTrackRequest, ...]
    coordination: Literal["independent", "static_others", "coupled"] = "independent"
    force_collision_refresh: bool = False
    command_id: str | None = None

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        if not tracks:
            raise ValueError("timeline tracks cannot be empty")
        ids = [track.robot_id for track in tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("timeline cannot contain duplicate robot IDs")
        if self.coordination not in {"independent", "static_others", "coupled"}:
            raise ValueError(
                "coordination must be independent, static_others, or coupled"
            )
        if self.coordination == "coupled":
            raise RuntimeError(
                "coordination='coupled' is unsupported because no coupled backend is configured"
            )
        object.__setattr__(self, "tracks", tracks)


def joint_positions_mapping(value: object, *, label: str) -> object:
    """冻结 JSON 中的 name mapping、flat vector 或 trajectory matrix。"""

    def json_float(item: object, item_label: str) -> float:
        """严格读取有限 JSON number，不接受 bool 或可转换字符串。"""

        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{item_label} must be a number")
        result = float(item)
        if not isfinite(result):
            raise ValueError(f"{item_label} must be finite")
        return result

    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{label} cannot be empty")
        frozen = {}
        for name, item in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{label} keys must be non-empty strings")
            if isinstance(item, list):
                samples = tuple(
                    json_float(sample, f"{label}[{name!r}][{index}]")
                    for index, sample in enumerate(item)
                )
                if not samples:
                    raise ValueError(f"{label}[{name!r}] cannot be empty")
                frozen[name] = samples
            else:
                frozen[name] = json_float(item, f"{label}[{name!r}]")
        return MappingProxyType(frozen)
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array or joint-name mapping")
    values = tuple(value)
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if isinstance(values[0], list):
        if any(not isinstance(row, list) for row in values):
            raise ValueError(f"{label} trajectory rows must all be arrays")
        rows = tuple(
            tuple(
                json_float(item, f"{label}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
            for row_index, row in enumerate(values)
        )
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError(f"{label} trajectory rows must have equal non-zero width")
        return rows
    return tuple(
        json_float(item, f"{label}[{index}]") for index, item in enumerate(values)
    )


__all__ = [
    "JointGroupTrackRequest",
    "RobotMotionUnitRequest",
    "RobotTimelineRequest",
    "RobotTrackRequest",
    "TimelineSegmentRequest",
    "joint_positions_mapping",
]
