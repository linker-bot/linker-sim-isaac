"""整数 physics tick timeline 的不可变模型、嵌套坐标和采样语义。

时间结构从外到内依次为 timeline、robot track、motion unit、joint-group track 和 segment；
每层 ``start_tick`` 都相对于直接父级。构造时把输入序列和数组复制、校验不重叠区间并冻结
metadata，执行器因此可以按 tick 读取而不必重复防御调用方修改。所有区间使用左闭右开
``[start_tick, end_tick)``，段间空隙通过最近一次 terminal sample 自动保持。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np


JointGroupName = Literal["arm", "hand"]
CoordinationPolicy = Literal["independent", "static_others", "coupled"]


@dataclass(frozen=True)
class SegmentSample:
    """一个 physics tick 的同组关节控制样本。

    三个向量必须与采样它的 segment ``joint_names`` 等长，``phase`` 随样本进入 observer 和
    logger；本轻量值对象由 segment 保证 shape，不单独重复校验。
    """

    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray
    phase: str


@dataclass(frozen=True)
class MotionSegment:
    """已经采样到执行 physics grid 上的关节轨迹片段。

    ``positions`` 行数就是实际占用 tick 数，不包含额外起始样本。velocities/efforts 缺省时
    生成同 shape 零矩阵；末行 velocity 固定为零，使随后自动 hold 不携带虚假的末端速度。
    ``requested_duration_s`` 仅记录量化前请求，不参与执行时长计算。
    """

    joint_names: tuple[str, ...]
    positions: np.ndarray
    start_tick: int = 0
    duration_ticks: int | None = None
    velocities: np.ndarray | None = None
    efforts: np.ndarray | None = None
    phase: str = "motion"
    requested_duration_s: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = _joint_names(self.joint_names, "MotionSegment.joint_names")
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != len(names):
            raise ValueError(
                "MotionSegment.positions must have shape (duration_ticks, joints)"
            )
        duration = (
            positions.shape[0]
            if self.duration_ticks is None
            else int(self.duration_ticks)
        )
        if duration < 0 or int(self.start_tick) < 0:
            raise ValueError("segment ticks cannot be negative")
        if positions.shape[0] != duration:
            raise ValueError("MotionSegment positions rows must equal duration_ticks")
        velocities = _sample_matrix_or_zeros(
            self.velocities, positions.shape, "MotionSegment.velocities"
        )
        efforts = _sample_matrix_or_zeros(
            self.efforts, positions.shape, "MotionSegment.efforts"
        )
        if duration:
            velocities[-1] = 0.0
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "efforts", efforts)
        object.__setattr__(self, "start_tick", int(self.start_tick))
        object.__setattr__(self, "duration_ticks", duration)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def end_tick(self) -> int:
        """返回父级局部坐标中的结束 tick（左闭右开）。"""

        return self.start_tick + int(self.duration_ticks)

    def sample(self, tick: int) -> SegmentSample:
        """按 segment 局部 tick 取样，并把越界 tick 限制到端点。"""

        if not self.duration_ticks:
            raise ValueError("a zero-tick motion segment cannot be sampled")
        index = min(max(int(tick) - self.start_tick, 0), self.duration_ticks - 1)
        return SegmentSample(
            positions=self.positions[index].copy(),
            velocities=self.velocities[index].copy(),
            efforts=self.efforts[index].copy(),
            phase=self.phase,
        )

    def terminal_sample(self) -> SegmentSample | None:
        """返回用于后续隐式 hold 的末端样本。"""

        if not self.duration_ticks:
            return None
        return SegmentSample(
            positions=self.positions[-1].copy(),
            velocities=np.zeros(len(self.joint_names), dtype=float),
            efforts=np.zeros(len(self.joint_names), dtype=float),
            phase=self.phase,
        )


@dataclass(frozen=True)
class HoldSegment:
    """显式保持段；它会占用父级 timeline 的实际 ticks。

    与 motion segment 不同，它只存一行位置并在每个 tick 返回副本，速度和力矩始终为零。
    零 tick hold 合法，可用于保留编排结构但不会被采样。
    """

    joint_names: tuple[str, ...]
    positions: np.ndarray
    duration_ticks: int
    start_tick: int = 0
    phase: str = "hold"
    requested_duration_s: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = _joint_names(self.joint_names, "HoldSegment.joint_names")
        positions = np.asarray(self.positions, dtype=float).reshape(-1)
        if positions.size != len(names):
            raise ValueError("HoldSegment positions must match joint_names")
        if int(self.start_tick) < 0 or int(self.duration_ticks) < 0:
            raise ValueError("segment ticks cannot be negative")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "start_tick", int(self.start_tick))
        object.__setattr__(self, "duration_ticks", int(self.duration_ticks))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def end_tick(self) -> int:
        """返回父级局部坐标中的结束 tick（左闭右开）。"""

        return self.start_tick + self.duration_ticks

    def sample(self, tick: int) -> SegmentSample:
        """返回固定 hold 样本；tick 仅用于统一 segment 采样接口。"""

        return self.terminal_sample()

    def terminal_sample(self) -> SegmentSample:
        """返回零 velocity/effort 的保持样本。"""

        return SegmentSample(
            positions=self.positions.copy(),
            velocities=np.zeros(len(self.joint_names), dtype=float),
            efforts=np.zeros(len(self.joint_names), dtype=float),
            phase=self.phase,
        )


TimelineSegment: TypeAlias = MotionSegment | HoldSegment


@dataclass(frozen=True)
class JointGroupTrack:
    """一个 motion unit 内 arm 或 hand 的串行 segment 轨。

    segments 可以使用不同但不重复的关节子集；采样结果按首次出现顺序建立联合关节空间。
    segment 之间的空隙从 baseline 或最近 terminal sample 保持，尾部持续保持到 track 结束。
    """

    group: JointGroupName
    segments: tuple[TimelineSegment, ...]
    start_tick: int = 0
    duration_ticks: int | None = None

    def __post_init__(self) -> None:
        if self.group not in {"arm", "hand"}:
            raise ValueError("joint group track must be 'arm' or 'hand'")
        if int(self.start_tick) < 0:
            raise ValueError("JointGroupTrack.start_tick cannot be negative")
        segments = tuple(self.segments)
        previous_end = 0
        for segment in segments:
            if segment.start_tick < previous_end:
                raise ValueError("joint group segments cannot overlap")
            previous_end = segment.end_tick
        natural = max((segment.end_tick for segment in segments), default=0)
        duration = natural if self.duration_ticks is None else int(self.duration_ticks)
        if duration < natural:
            raise ValueError("group duration cannot truncate a segment")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "start_tick", int(self.start_tick))
        object.__setattr__(self, "duration_ticks", duration)

    @property
    def end_tick(self) -> int:
        """返回 group track 在 unit 局部坐标中的结束 tick。"""

        return self.start_tick + int(self.duration_ticks)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """按 segment 首次出现顺序合并该 track 的唯一 joint names。"""

        names: list[str] = []
        for segment in self.segments:
            for name in segment.joint_names:
                if name not in names:
                    names.append(name)
        return tuple(names)

    def sample(
        self,
        unit_tick: int,
        baseline: Mapping[str, float],
    ) -> dict[str, tuple[float, float, float, str]]:
        """串行采样 segments；段间空隙和尾部自动保持最近状态。

        ``baseline`` 必须包含该 track 的全部 ``joint_names``，表示 unit 开始时的命令状态。
        返回值以 joint name 为键，tuple 依次为 position、velocity、effort 和 phase。
        """

        local_tick = int(unit_tick) - self.start_tick
        state = {
            name: (float(baseline[name]), 0.0, 0.0, "hold") for name in self.joint_names
        }
        if local_tick < 0:
            return state
        for segment in self.segments:
            if local_tick < segment.start_tick:
                return state
            if local_tick >= segment.end_tick:
                terminal = segment.terminal_sample()
                if terminal is not None:
                    _merge_sample(state, segment.joint_names, terminal)
                continue
            _merge_sample(state, segment.joint_names, segment.sample(local_tick))
            return state
        return state


@dataclass(frozen=True)
class RobotMotionUnit:
    """共享逻辑起点并行执行的 arm/hand tracks。

    同一 unit 每个 group 最多一条 track，且任意两条 track 不得写同名关节；这些不变量在
    构造和采样时各检查一次，避免后续合并结果依赖字典覆盖顺序。
    """

    group_tracks: tuple[JointGroupTrack, ...]
    start_tick: int = 0
    duration_ticks: int | None = None

    def __post_init__(self) -> None:
        tracks = tuple(self.group_tracks)
        groups = [track.group for track in tracks]
        if len(set(groups)) != len(groups):
            raise ValueError("a motion unit can contain at most one track per group")
        writers: set[str] = set()
        for track in tracks:
            overlap = writers & set(track.joint_names)
            if overlap:
                raise ValueError(
                    f"motion unit has multiple joint writers: {sorted(overlap)}"
                )
            writers.update(track.joint_names)
        natural = max((track.end_tick for track in tracks), default=0)
        duration = natural if self.duration_ticks is None else int(self.duration_ticks)
        if int(self.start_tick) < 0 or duration < natural:
            raise ValueError("motion unit ticks are invalid or truncate a group track")
        object.__setattr__(self, "group_tracks", tracks)
        object.__setattr__(self, "start_tick", int(self.start_tick))
        object.__setattr__(self, "duration_ticks", duration)

    @property
    def end_tick(self) -> int:
        """返回 unit 在 robot track 局部坐标中的结束 tick。"""

        return self.start_tick + int(self.duration_ticks)

    def sample(
        self,
        track_tick: int,
        command_state: Mapping[str, float],
    ) -> dict[str, tuple[float, float, float, str]]:
        """在同一 unit tick 并行采样各 group，并拒绝运行期重复 writer。"""

        unit_tick = int(track_tick) - self.start_tick
        result: dict[str, tuple[float, float, float, str]] = {}
        for group_track in self.group_tracks:
            sampled = group_track.sample(unit_tick, command_state)
            overlap = set(result) & set(sampled)
            if overlap:
                raise ValueError(
                    f"motion unit sampled multiple writers: {sorted(overlap)}"
                )
            result.update(sampled)
        return result


@dataclass(frozen=True)
class RobotTrack:
    """一个 session robot ID 的串行 motion units。

    units 按 tuple 顺序执行且区间不得重叠；空隙表示保持调用方已有 command state。
    ``robot_label`` 只用于诊断，执行身份以本 session 的非负 ``robot_id`` 为准。
    """

    robot_id: int
    units: tuple[RobotMotionUnit, ...]
    robot_label: str | None = None
    duration_ticks: int | None = None

    def __post_init__(self) -> None:
        if int(self.robot_id) < 0:
            raise ValueError("robot_id must be non-negative")
        units = tuple(self.units)
        previous_end = 0
        for unit in units:
            if unit.start_tick < previous_end:
                raise ValueError("robot motion units cannot overlap")
            previous_end = unit.end_tick
        natural = max((unit.end_tick for unit in units), default=0)
        duration = natural if self.duration_ticks is None else int(self.duration_ticks)
        if duration < natural:
            raise ValueError("robot track duration cannot truncate a unit")
        object.__setattr__(self, "robot_id", int(self.robot_id))
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "duration_ticks", duration)

    def active_unit(self, tick: int) -> RobotMotionUnit | None:
        """返回覆盖全局 track tick 的 unit。"""

        for unit in self.units:
            if unit.start_tick <= int(tick) < unit.end_tick:
                return unit
        return None


@dataclass(frozen=True)
class RobotTimeline:
    """共享同一 physics grid 的多机器人并行 timeline。

    每个 robot ID 最多一条 track，所有 track 使用同一正有限 ``physics_dt``。显式
    ``duration_ticks`` 可以延长尾部 hold，但不能截断任何 track。``scene_version`` 记录规划
    时的碰撞场景状态，供执行前检查规划结果是否过期，不是配置格式版本。
    """

    tracks: tuple[RobotTrack, ...]
    physics_dt: float
    duration_ticks: int | None = None
    coordination: CoordinationPolicy = "independent"
    scene_version: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dt = float(self.physics_dt)
        if not isfinite(dt) or dt <= 0:
            raise ValueError("physics_dt must be positive and finite")
        tracks = tuple(self.tracks)
        ids = [track.robot_id for track in tracks]
        if len(set(ids)) != len(ids):
            raise ValueError("a timeline cannot contain duplicate robot IDs")
        if self.coordination not in {"independent", "static_others", "coupled"}:
            raise ValueError("unknown coordination policy")
        natural = max((int(track.duration_ticks) for track in tracks), default=0)
        duration = natural if self.duration_ticks is None else int(self.duration_ticks)
        if duration < natural:
            raise ValueError("timeline duration cannot truncate a robot track")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "physics_dt", dt)
        object.__setattr__(self, "duration_ticks", duration)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        validate_collision_aware_coordination(self)

    @property
    def actual_duration_s(self) -> float:
        """返回整数 duration ticks 对应的实际 physics 时长。"""

        return int(self.duration_ticks) * self.physics_dt


def validate_collision_aware_coordination(timeline: RobotTimeline) -> None:
    """拒绝没有 coupled backend 支撑的跨机器人碰撞感知重叠区间。

    检查把各层局部 start tick 累加到 timeline 坐标。不同机器人的 motion segment 只要时间
    相交且任一段声明 collision-aware，就必须由 coupled planner 联合求解；当前没有该
    backend，因此也拒绝带实际 motion 的 ``coordination='coupled'``，不能伪装成已协调。
    """

    intervals = []
    for track in timeline.tracks:
        for unit in track.units:
            for group in unit.group_tracks:
                for segment in group.segments:
                    if not isinstance(segment, MotionSegment):
                        continue
                    start = unit.start_tick + group.start_tick + segment.start_tick
                    intervals.append(
                        (
                            track.robot_id,
                            start,
                            start + segment.duration_ticks,
                            bool(segment.metadata.get("collision_aware", False)),
                        )
                    )
    for index, first in enumerate(intervals):
        for second in intervals[index + 1 :]:
            if first[0] == second[0]:
                continue
            if (
                (first[3] or second[3])
                and max(first[1], second[1]) < min(first[2], second[2])
                and timeline.coordination != "coupled"
            ):
                raise RuntimeError(
                    "overlapping collision-aware robot motion requires "
                    "coordination='coupled', but no coupled backend is available"
                )
    if timeline.coordination == "coupled" and intervals:
        raise RuntimeError(
            "coordination='coupled' is not available without a coupled planning backend"
        )


def _merge_sample(
    target: dict[str, tuple[float, float, float, str]],
    names: Sequence[str],
    sample: SegmentSample,
) -> None:
    """把按列排列的 ``SegmentSample`` 写入 joint-name keyed 状态。"""

    for index, name in enumerate(names):
        target[name] = (
            float(sample.positions[index]),
            float(sample.velocities[index]),
            float(sample.efforts[index]),
            sample.phase,
        )


def _joint_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    """冻结并校验非空、无重复的 joint name sequence。"""

    names = tuple(str(value) for value in values)
    if not names:
        raise ValueError(f"{label} cannot be empty")
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{label} must contain unique non-empty names")
    return names


def _sample_matrix_or_zeros(values, shape, label: str) -> np.ndarray:
    """校验可选 sample matrix；省略时创建与 position 同 shape 的零矩阵。"""

    if values is None:
        return np.zeros(shape, dtype=float)
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"{label} expected shape {shape}, got {matrix.shape}")
    return matrix.copy()


__all__ = [
    "CoordinationPolicy",
    "HoldSegment",
    "JointGroupName",
    "JointGroupTrack",
    "MotionSegment",
    "RobotMotionUnit",
    "RobotTimeline",
    "RobotTrack",
    "SegmentSample",
    "TimelineSegment",
    "validate_collision_aware_coordination",
]
