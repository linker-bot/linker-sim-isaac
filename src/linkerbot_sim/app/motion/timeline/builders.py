"""Timeline tick 换算、可执行 segment/track 与 direct request 构造器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import ceil, isfinite

import numpy as np

from linkerbot_sim.app.motion.timeline.model import (
    HoldSegment,
    JointGroupName,
    JointGroupTrack,
    MotionSegment,
    RobotMotionUnit,
    RobotTrack,
    TimelineSegment,
)
from linkerbot_sim.app.motion.timeline.requests import (
    JointGroupTrackRequest,
    RobotMotionUnitRequest,
    TimelineSegmentRequest,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def duration_to_ticks(duration_s: float, physics_dt: float) -> int:
    """把非负秒数向上对齐到 physics grid，避免少执行一个 tick。"""

    duration = float(duration_s)
    dt = float(physics_dt)
    if not isfinite(duration) or duration < 0:
        raise ValueError("duration_s must be finite and non-negative")
    if not isfinite(dt) or dt <= 0:
        raise ValueError("physics_dt must be positive and finite")
    if duration == 0:
        return 0
    return max(1, int(ceil(duration / dt - 1.0e-12)))


def make_hold_segment(
    *,
    joint_names: Sequence[str],
    positions: Sequence[float],
    duration_s: float,
    physics_dt: float,
    start_tick: int = 0,
    phase: str = "hold",
) -> HoldSegment:
    """创建占用明确 ticks 的保持段。"""

    return HoldSegment(
        joint_names=tuple(joint_names),
        positions=np.asarray(positions, dtype=float),
        duration_ticks=duration_to_ticks(duration_s, physics_dt),
        start_tick=start_tick,
        phase=phase,
        requested_duration_s=float(duration_s),
    )


def make_goal_segment(
    *,
    joint_names: Sequence[str],
    start_positions: Sequence[float],
    target_positions: Sequence[float],
    duration_s: float,
    physics_dt: float,
    start_tick: int = 0,
    phase: str = "joint_goal",
    interpolation: str = "smoothstep",
    metadata: Mapping[str, object] | None = None,
) -> MotionSegment:
    """按 resolved interpolation 在整数 physics ticks 上构造 joint goal。"""

    names = tuple(joint_names)
    start = np.asarray(start_positions, dtype=float).reshape(-1)
    target = np.asarray(target_positions, dtype=float).reshape(-1)
    if start.size != len(names) or target.size != len(names):
        raise ValueError("goal positions must match joint_names")
    if interpolation not in {"linear", "smoothstep"}:
        raise ValueError("interpolation must be linear or smoothstep")
    ticks = duration_to_ticks(duration_s, physics_dt)
    changed = not np.allclose(start, target, rtol=0.0, atol=1.0e-12)
    if ticks == 0 and changed:
        raise ValueError("zero-duration non-trivial motion is not allowed")
    if ticks == 0:
        positions = np.empty((0, len(names)), dtype=float)
        velocities = positions.copy()
    else:
        alpha = np.arange(1, ticks + 1, dtype=float) / ticks
        if interpolation == "smoothstep":
            progress = alpha * alpha * (3.0 - 2.0 * alpha)
            rate = 6.0 * alpha * (1.0 - alpha) / (ticks * physics_dt)
        else:
            progress = alpha
            rate = np.full_like(alpha, 1.0 / (ticks * physics_dt))
        positions = start[None, :] + progress[:, None] * (target - start)[None, :]
        velocities = rate[:, None] * (target - start)[None, :]
    return MotionSegment(
        joint_names=names,
        positions=positions,
        velocities=velocities,
        start_tick=start_tick,
        phase=phase,
        requested_duration_s=float(duration_s),
        metadata={} if metadata is None else metadata,
    )


def make_trajectory_segment(
    trajectory: JointTrajectory,
    *,
    physics_dt: float,
    requested_duration_s: float = 0.0,
    start_positions: Sequence[float] | None = None,
    start_tick: int = 0,
    phase: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MotionSegment:
    """把连续时间 trajectory 重采样到 execution physics grid。"""

    lower, upper = trajectory.domain()
    planned_duration = max(0.0, upper - lower)
    ticks = max(
        duration_to_ticks(requested_duration_s, physics_dt),
        duration_to_ticks(planned_duration, physics_dt),
    )
    goal = np.asarray(trajectory.positions[-1], dtype=float)
    if ticks == 0:
        if start_positions is not None and not np.allclose(
            np.asarray(start_positions, dtype=float),
            goal,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("zero-duration non-trivial trajectory is not allowed")
        positions = np.empty((0, len(trajectory.joint_names)), dtype=float)
        velocities = positions.copy()
        efforts = positions.copy()
    elif len(trajectory) == 1 and start_positions is not None:
        return make_goal_segment(
            joint_names=trajectory.joint_names,
            start_positions=start_positions,
            target_positions=goal,
            duration_s=ticks * physics_dt,
            physics_dt=physics_dt,
            start_tick=start_tick,
            phase=phase or trajectory.phases[-1],
            metadata=metadata,
        )
    else:
        times = (
            lower + (np.arange(1, ticks + 1, dtype=float) / ticks) * planned_duration
        )
        samples = [trajectory.eval_all(float(time)) for time in times]
        positions = np.asarray([sample.position for sample in samples], dtype=float)
        scale = planned_duration / (ticks * physics_dt) if planned_duration > 0 else 0.0
        velocities = (
            np.asarray([sample.velocity for sample in samples], dtype=float) * scale
        )
        efforts = np.asarray([sample.effort for sample in samples], dtype=float)
    return MotionSegment(
        joint_names=trajectory.joint_names,
        positions=positions,
        velocities=velocities,
        efforts=efforts,
        start_tick=start_tick,
        phase=phase or trajectory.phases[-1],
        requested_duration_s=float(requested_duration_s),
        metadata={} if metadata is None else metadata,
    )


def sequential_group_track(
    group: JointGroupName,
    segments: Sequence[TimelineSegment],
) -> JointGroupTrack:
    """按输入顺序分配连续 segment start ticks，不重新计算 duration。"""

    cursor = 0
    assigned = []
    for segment in segments:
        assigned_segment = replace(segment, start_tick=cursor)
        assigned.append(assigned_segment)
        cursor += int(assigned_segment.duration_ticks)
    return JointGroupTrack(
        group=group,
        segments=tuple(assigned),
        duration_ticks=cursor,
    )


def sequential_robot_track(
    robot_id: int,
    units: Sequence[RobotMotionUnit],
    *,
    robot_label: str | None = None,
) -> RobotTrack:
    """按输入顺序分配连续 motion unit start ticks。"""

    cursor = 0
    assigned = []
    for unit in units:
        assigned_unit = replace(unit, start_tick=cursor)
        assigned.append(assigned_unit)
        cursor += int(assigned_unit.duration_ticks)
    return RobotTrack(
        robot_id=robot_id,
        robot_label=robot_label,
        units=tuple(assigned),
        duration_ticks=cursor,
    )


def _joint_goal(
    value: object,
    names: tuple[str, ...],
    current: np.ndarray,
) -> np.ndarray:
    """把 named/flat goal 规范化为 group joint order。"""

    if isinstance(value, Mapping):
        result = current.copy()
        index = {name: position for position, name in enumerate(names)}
        missing = sorted(set(value) - set(index))
        if missing:
            raise ValueError(f"joint goal contains joints outside the group: {missing}")
        for name, item in value.items():
            result[index[str(name)]] = float(item)
        return result
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != len(names):
        raise ValueError(
            f"joint goal expected {len(names)} values for {list(names)}, "
            f"got {result.size}"
        )
    return result


def _direct_trajectory(
    request: TimelineSegmentRequest,
    *,
    joint_names: tuple[str, ...],
    current: np.ndarray,
    physics_dt: float,
    phase: str,
) -> JointTrajectory:
    """把 named columns 或二维数组构造成 direct JointTrajectory。"""

    value = request.joint_positions
    if isinstance(value, Mapping):
        sample_count = None
        columns = {}
        for name, samples in value.items():
            if name not in joint_names:
                raise ValueError(f"trajectory joint {name!r} is outside the group")
            column = np.asarray(samples, dtype=float).reshape(-1)
            sample_count = column.size if sample_count is None else sample_count
            if column.size != sample_count:
                raise ValueError("named trajectory columns must have equal length")
            columns[str(name)] = column
        rows = np.repeat(current[None, :], int(sample_count), axis=0)
        index = {name: position for position, name in enumerate(joint_names)}
        for name, column in columns.items():
            rows[:, index[name]] = column
    else:
        rows = np.asarray(value, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != len(joint_names):
            raise ValueError(f"trajectory must have shape (N, {len(joint_names)})")
    if rows.shape[0] == 0:
        raise ValueError("trajectory cannot be empty")
    if request.times_s:
        sample_times = np.asarray(request.times_s, dtype=float)
        if sample_times.size != rows.shape[0]:
            raise ValueError("times_s must match trajectory sample count")
        if np.any(sample_times <= 0) or np.any(np.diff(sample_times) <= 0):
            raise ValueError("times_s must be positive and strictly increasing")
    else:
        duration = request.duration_s
        if duration <= 0:
            duration = rows.shape[0] * physics_dt
        sample_times = np.linspace(duration / rows.shape[0], duration, rows.shape[0])
    return JointTrajectory(
        times=np.concatenate(([0.0], sample_times)),
        positions=np.vstack((current, rows)),
        joint_names=joint_names,
        phases=tuple(phase for _ in range(rows.shape[0] + 1)),
    )


def _expand_full_command_unit(
    robot: object,
    unit: RobotMotionUnitRequest,
) -> RobotMotionUnitRequest:
    """把 arm_hand 全 command-space direct segment 拆成并行 arm/hand tracks。"""

    if robot.kind.value != "arm_hand" or len(unit.group_tracks) != 1:
        return unit
    track = unit.group_tracks[0]
    if len(track.segments) != 1:
        return unit
    segment = track.segments[0]
    if segment.kind not in {"joint_goal", "joint_delta", "joint_trajectory"}:
        return unit
    command_names = robot.joint_groups.command_joint_names
    split = _split_full_joint_value(
        segment.joint_positions,
        command_names=command_names,
        arm_names=robot.joint_groups.arm,
        hand_names=robot.joint_groups.hand,
    )
    if split is None:
        return unit
    arm_value, hand_value = split
    return RobotMotionUnitRequest(
        group_tracks=(
            JointGroupTrackRequest(
                "arm", (replace(segment, joint_positions=arm_value),)
            ),
            JointGroupTrackRequest(
                "hand", (replace(segment, joint_positions=hand_value),)
            ),
        )
    )


def _split_full_joint_value(
    value: object,
    *,
    command_names: tuple[str, ...],
    arm_names: tuple[str, ...],
    hand_names: tuple[str, ...],
) -> tuple[object, object] | None:
    """按 command joint names 拆分 flat/named/trajectory value。"""

    if isinstance(value, Mapping):
        keys = set(value)
        if keys <= set(arm_names) or keys <= set(hand_names):
            return None
        if not keys <= set(command_names):
            return None
        return (
            {name: value[name] for name in arm_names if name in value},
            {name: value[name] for name in hand_names if name in value},
        )
    array = np.asarray(value, dtype=float)
    if array.shape[-1] != len(command_names):
        return None
    index = {name: position for position, name in enumerate(command_names)}
    arm_index = [index[name] for name in arm_names]
    hand_index = [index[name] for name in hand_names]
    if array.ndim == 1:
        return tuple(array[indices].tolist() for indices in (arm_index, hand_index))
    if array.ndim == 2:
        return tuple(array[:, indices].tolist() for indices in (arm_index, hand_index))
    return None


__all__ = [
    "duration_to_ticks",
    "make_goal_segment",
    "make_hold_segment",
    "make_trajectory_segment",
    "sequential_group_track",
    "sequential_robot_track",
]
