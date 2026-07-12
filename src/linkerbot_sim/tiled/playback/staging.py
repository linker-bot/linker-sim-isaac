"""把主轨迹与部分关节 tracks 展开为 per-env before/main/after 播放队列。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.tiled.playback.models import (
    PlaybackJointTrack,
    _Playback,
    _PlaybackJointTrack,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def playback_sequences_for_envs(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    joint_names: tuple[str, ...],
    joint_tracks: Sequence[PlaybackJointTrack],
    request_id: str | None,
    source: str,
    dynamic_base: bool,
) -> tuple[tuple[_Playback, ...], ...]:
    """为每个 selected env 构造独立的 staged playback 队列。"""

    sample_times = np.asarray(times, dtype=float).reshape(-1)
    batch = np.asarray(positions, dtype=float)
    env_count = int(batch.shape[0])
    command_dim = int(batch.shape[2])
    _validate_joint_track_indices(joint_tracks, command_dim=command_dim)
    return tuple(
        _playback_sequence_for_env(
            row=row,
            times=sample_times,
            positions=batch[row],
            joint_names=joint_names,
            joint_tracks=joint_tracks,
            env_count=env_count,
            request_id=request_id,
            source=source,
            dynamic_base=dynamic_base,
        )
        for row in range(env_count)
    )


def _playback_sequence_for_env(
    *,
    row: int,
    times: np.ndarray,
    positions: np.ndarray,
    joint_names: tuple[str, ...],
    joint_tracks: Sequence[PlaybackJointTrack],
    env_count: int,
    request_id: str | None,
    source: str,
    dynamic_base: bool,
) -> tuple[_Playback, ...]:
    """构造单个 env 的 before/main/after playback 队列。"""

    main_duration_s = max(0.0, float(times[-1]) - float(times[0]))
    cursor = np.asarray(positions[0], dtype=float).reshape(-1).copy()
    cursor = _initial_cursor_with_joint_track_starts(
        cursor,
        joint_tracks=joint_tracks,
        row=row,
        env_count=env_count,
    )
    sequence: list[_Playback] = []

    before = _stage_playback(
        timing="before",
        cursor=cursor,
        joint_tracks=joint_tracks,
        row=row,
        env_count=env_count,
        joint_names=joint_names,
        default_duration_s=main_duration_s,
        request_id=request_id,
        source=source,
    )
    if before is not None:
        sequence.append(before)
        cursor = _cursor_after_timing(
            cursor,
            joint_tracks=joint_tracks,
            timing="before",
            row=row,
            env_count=env_count,
        )

    main_positions = np.asarray(positions, dtype=float).copy()
    sync_indices = _timing_indices(joint_tracks, "sync")
    for index in _timing_indices(joint_tracks, "before") - sync_indices:
        main_positions[:, index] = cursor[index]
    sync_track = _playback_joint_track_for_timing(
        joint_tracks,
        timing="sync",
        row=row,
        env_count=env_count,
        cursor=cursor,
        joint_names=joint_names,
        default_duration_s=main_duration_s,
    )
    main = JointTrajectory.from_samples(
        times=times,
        positions=main_positions,
        joint_names=joint_names,
    )
    sequence.append(
        _Playback.from_trajectory(
            main,
            request_id=request_id,
            source=source,
            joint_track=sync_track,
            stage="trajectory",
            dynamic_base=dynamic_base,
        )
    )
    cursor = np.asarray(main_positions[-1], dtype=float).reshape(-1).copy()
    cursor = _cursor_after_timing(
        cursor,
        joint_tracks=joint_tracks,
        timing="sync",
        row=row,
        env_count=env_count,
    )

    after = _stage_playback(
        timing="after",
        cursor=cursor,
        joint_tracks=joint_tracks,
        row=row,
        env_count=env_count,
        joint_names=joint_names,
        default_duration_s=main_duration_s,
        request_id=request_id,
        source=source,
    )
    if after is not None:
        sequence.append(after)
    return tuple(sequence)


def _stage_playback(
    *,
    timing: str,
    cursor: np.ndarray,
    joint_tracks: Sequence[PlaybackJointTrack],
    row: int,
    env_count: int,
    joint_names: tuple[str, ...],
    default_duration_s: float,
    request_id: str | None,
    source: str,
) -> _Playback | None:
    """构造只改变部分 command joints 的 before/after stage。"""

    joint_track = _playback_joint_track_for_timing(
        joint_tracks,
        timing=timing,
        row=row,
        env_count=env_count,
        cursor=cursor,
        joint_names=joint_names,
        default_duration_s=default_duration_s,
    )
    if joint_track is None:
        return None
    duration_s = (
        float(np.nanmax(joint_track.durations_s))
        if joint_track.durations_s.size
        else 0.0
    )
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        times = np.asarray([0.0], dtype=float)
        positions = cursor.reshape(1, -1)
    else:
        times = np.asarray([0.0, duration_s], dtype=float)
        positions = np.repeat(cursor.reshape(1, -1), 2, axis=0)
    trajectory = JointTrajectory.from_samples(
        times=times,
        positions=positions,
        joint_names=joint_names,
    )
    return _Playback.from_trajectory(
        trajectory,
        request_id=request_id,
        source=source,
        joint_track=joint_track,
        stage=timing,
    )


def _playback_joint_track_for_timing(
    joint_tracks: Sequence[PlaybackJointTrack],
    *,
    timing: str,
    row: int,
    env_count: int,
    cursor: np.ndarray,
    joint_names: tuple[str, ...],
    default_duration_s: float,
) -> _PlaybackJointTrack | None:
    """把同一 timing 的输入 tracks 合并成单 env sampler。"""

    merged_indices: list[int] = []
    merged_names: list[str] = []
    starts: list[float] = []
    targets: list[float] = []
    durations: list[float] = []
    seen: set[int] = set()
    for joint_track in joint_tracks:
        if joint_track.timing != timing:
            continue
        indices = tuple(int(index) for index in joint_track.joint_indices)
        duplicate = [index for index in indices if index in seen]
        if duplicate:
            raise ValueError(
                f"playback joint_indices duplicated across tracks: {duplicate}"
            )
        seen.update(indices)
        target = _joint_track_rows(
            joint_track.target_positions,
            env_count,
            "target_positions",
        )
        merged_indices.extend(indices)
        merged_names.extend(joint_names[index] for index in indices)
        starts.extend(float(cursor[index]) for index in indices)
        targets.extend(float(value) for value in target[row])
        duration = (
            float(default_duration_s)
            if joint_track.duration_s is None
            else float(joint_track.duration_s)
        )
        durations.extend(duration for _ in indices)
    if not merged_indices:
        return None
    return _PlaybackJointTrack(
        joint_indices=np.asarray(merged_indices, dtype=int),
        start_positions=np.asarray(starts, dtype=float),
        target_positions=np.asarray(targets, dtype=float),
        durations_s=np.asarray(durations, dtype=float),
        joint_names=tuple(merged_names),
    )


def _validate_joint_track_indices(
    joint_tracks: Sequence[PlaybackJointTrack],
    *,
    command_dim: int,
) -> None:
    """校验所有 sparse track column 都落在完整 command dimension 内。"""

    for joint_track in joint_tracks:
        out_of_range = [
            index for index in joint_track.joint_indices if index >= int(command_dim)
        ]
        if out_of_range:
            raise ValueError(f"playback joint_indices out of range: {out_of_range}")


def _initial_cursor_with_joint_track_starts(
    cursor: np.ndarray,
    *,
    joint_tracks: Sequence[PlaybackJointTrack],
    row: int,
    env_count: int,
) -> np.ndarray:
    """用 load 时的 track start targets 修正第一个 stage 的 cursor。"""

    result = np.asarray(cursor, dtype=float).reshape(-1).copy()
    initialized: set[int] = set()
    for joint_track in joint_tracks:
        start = _joint_track_rows(
            joint_track.start_positions,
            env_count,
            "start_positions",
        )
        for offset, index in enumerate(joint_track.joint_indices):
            if int(index) in initialized:
                continue
            result[int(index)] = float(start[row, offset])
            initialized.add(int(index))
    return result


def _cursor_after_timing(
    cursor: np.ndarray,
    *,
    joint_tracks: Sequence[PlaybackJointTrack],
    timing: str,
    row: int,
    env_count: int,
) -> np.ndarray:
    """把指定 timing stage 的 target 写入 cursor，作为后续 stage baseline。"""

    result = np.asarray(cursor, dtype=float).reshape(-1).copy()
    for joint_track in joint_tracks:
        if joint_track.timing != timing:
            continue
        target = _joint_track_rows(
            joint_track.target_positions,
            env_count,
            "target_positions",
        )
        for offset, index in enumerate(joint_track.joint_indices):
            result[int(index)] = float(target[row, offset])
    return result


def _timing_indices(
    joint_tracks: Sequence[PlaybackJointTrack],
    timing: str,
) -> set[int]:
    """收集指定 timing 下由 sparse tracks 占用的 command columns。"""

    return {
        int(index)
        for joint_track in joint_tracks
        if joint_track.timing == timing
        for index in joint_track.joint_indices
    }


def _joint_track_rows(
    values: np.ndarray,
    env_count: int,
    label: str,
) -> np.ndarray:
    """把单行 sparse target 广播到 selected env，或校验逐 env rows。"""

    array = np.asarray(values, dtype=float)
    if array.shape[0] == 1 and int(env_count) != 1:
        return np.repeat(array, int(env_count), axis=0)
    if array.shape[0] != int(env_count):
        raise ValueError(
            f"playback joint track {label} env dimension must be 1 or len(env_ids)"
        )
    return array.astype(float, copy=True)
