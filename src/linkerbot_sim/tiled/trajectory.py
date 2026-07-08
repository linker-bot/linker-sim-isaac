"""tiled 轨迹回放缓冲。

本模块只负责把已经规划好的 ``JointTrajectory`` 放进 per-env/per-robot 缓冲区，并在
同步 command boundary 采样成 batched joint target。它不创建 planner，不访问 Isaac
``World``，也不决定 ``world.step`` 的次数；这些职责分别属于 planner manager 和 runtime。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class TiledTrajectoryOverlay:
    """随 tiled 轨迹同步采样的 command-space 关节覆盖。

    ``joint_indices`` 指向轨迹/command-space 的列；``start_positions`` 和
    ``target_positions`` 支持 ``(K,)`` 广播到所有 selected env，或 ``(E,K)`` 为每个 env
    指定一行。``timing`` 可为 ``before``、``sync`` 或 ``after``；``duration_s`` 为
    ``None`` 时使用所在主轨迹时长。
    """

    joint_indices: tuple[int, ...]
    start_positions: np.ndarray
    target_positions: np.ndarray
    duration_s: float | None = None
    timing: str = "sync"

    def __post_init__(self) -> None:
        """校验 overlay 的列索引和起止矩阵形状。"""

        indices = tuple(int(index) for index in self.joint_indices)
        if not indices:
            raise ValueError("overlay joint_indices cannot be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("overlay joint_indices cannot contain duplicates")
        if any(index < 0 for index in indices):
            raise ValueError("overlay joint_indices cannot be negative")
        start = _overlay_matrix(self.start_positions, len(indices), "start_positions")
        target = _overlay_matrix(self.target_positions, len(indices), "target_positions")
        if start.shape[0] not in {1, target.shape[0]} and target.shape[0] != 1:
            raise ValueError("overlay start/target env dimensions are incompatible")
        if self.duration_s is not None and float(self.duration_s) < 0.0:
            raise ValueError("overlay duration_s cannot be negative")
        timing = str(self.timing)
        if timing not in {"before", "sync", "after"}:
            raise ValueError("overlay timing must be one of: before, sync, after")
        object.__setattr__(self, "joint_indices", indices)
        object.__setattr__(self, "start_positions", start)
        object.__setattr__(self, "target_positions", target)
        object.__setattr__(
            self,
            "duration_s",
            None if self.duration_s is None else float(self.duration_s),
        )
        object.__setattr__(self, "timing", timing)


@dataclass(frozen=True)
class TiledTrajectoryStepResult:
    """一次 tiled 轨迹采样结果。

    ``joint_positions`` 总是完整 batch，shape 为 ``(num_envs, command_dim)``。调用方可以
    直接把它作为 selected robot 的 batched target 写回 articulation view；没有 active
    trajectory 的 env 会保持传入的 ``current_positions``。
    """

    robot_name: str
    env_ids: tuple[int, ...]
    joint_positions: np.ndarray
    active_env_ids: tuple[int, ...]
    completed_env_ids: tuple[int, ...]
    idle_env_ids: tuple[int, ...]
    dt_s: float

    def to_json(self) -> dict[str, object]:
        """转换成交互响应可直接 ``json.dumps`` 的 payload。"""

        return {
            "robot": self.robot_name,
            "env_ids": list(self.env_ids),
            "active_env_ids": list(self.active_env_ids),
            "completed_env_ids": list(self.completed_env_ids),
            "idle_env_ids": list(self.idle_env_ids),
            "dt_s": self.dt_s,
        }


@dataclass(frozen=True)
class _PlaybackOverlay:
    """单个 env 的 overlay 采样状态。"""

    joint_indices: np.ndarray
    start_positions: np.ndarray
    target_positions: np.ndarray
    durations_s: np.ndarray
    joint_names: tuple[str, ...]

    def sample(
        self,
        positions: np.ndarray,
        *,
        elapsed_s: float,
        lower_s: float,
        upper_s: float,
        start_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """按当前轨迹时间覆盖指定 command-space 列。"""

        result = np.asarray(positions, dtype=float).reshape(-1).copy()
        start = (
            self.start_positions
            if start_positions is None
            else np.asarray(start_positions, dtype=float).reshape(-1)
        )
        domain_duration = max(0.0, float(upper_s) - float(lower_s))
        durations = np.where(np.isnan(self.durations_s), domain_duration, self.durations_s)
        elapsed = max(0.0, float(elapsed_s) - float(lower_s))
        alpha = np.ones_like(durations, dtype=float)
        positive = durations > 0.0
        alpha[positive] = np.clip(elapsed / durations[positive], 0.0, 1.0)
        result[self.joint_indices] = (
            start + alpha * (self.target_positions - start)
        )
        return result


@dataclass
class _Playback:
    """单个 env 上一条轨迹的播放游标。"""

    trajectory: JointTrajectory
    elapsed_s: float
    completed: bool
    request_id: str | None
    source: str
    overlay: _PlaybackOverlay | None = None
    stage: str = "trajectory"
    dynamic_base: bool = False
    dynamic_overlay_start: np.ndarray | None = None

    @classmethod
    def from_trajectory(
        cls,
        trajectory: JointTrajectory,
        *,
        request_id: str | None,
        source: str,
        overlay: _PlaybackOverlay | None = None,
        stage: str = "trajectory",
        dynamic_base: bool = False,
    ) -> "_Playback":
        """从轨迹起点创建播放游标。"""

        lower, upper = trajectory.domain()
        return cls(
            trajectory=trajectory,
            elapsed_s=float(lower),
            completed=bool(np.isclose(lower, upper)),
            request_id=request_id,
            source=str(source),
            overlay=overlay,
            stage=str(stage),
            dynamic_base=bool(dynamic_base),
        )

    def sample(
        self,
        dt_s: float,
        *,
        current_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """向前推进 ``dt_s`` 并返回当前应下发的关节目标。"""

        lower, upper = self.trajectory.domain()
        if not self.completed:
            # 这里的 elapsed 是轨迹局部时间，不是仿真全局时间。这样 planner 可以输出从
            # 0 开始的轨迹，也可以输出非零下界的轨迹，runtime 只需要告诉我们本次推进了多久。
            self.elapsed_s = min(float(upper), max(float(lower), self.elapsed_s + dt_s))
            self.completed = self.elapsed_s >= float(upper) - 1.0e-12
        if self.dynamic_base and current_positions is not None:
            sample = np.asarray(current_positions, dtype=float).reshape(-1).copy()
        else:
            sample = self.trajectory.eval(self.elapsed_s)
        if self.overlay is not None:
            if self.dynamic_base and self.dynamic_overlay_start is None:
                self.dynamic_overlay_start = sample[self.overlay.joint_indices].copy()
            return self.overlay.sample(
                sample,
                elapsed_s=self.elapsed_s,
                lower_s=float(lower),
                upper_s=float(upper),
                start_positions=self.dynamic_overlay_start,
            )
        return sample

    def status(self, env_id: int) -> dict[str, object]:
        """返回当前播放状态摘要。"""

        lower, upper = self.trajectory.domain()
        duration = max(0.0, float(upper) - float(lower))
        progress = 1.0 if duration <= 0.0 else (self.elapsed_s - float(lower)) / duration
        return {
            "env_id": int(env_id),
            "request_id": self.request_id,
            "source": self.source,
            "stage": self.stage,
            "completed": bool(self.completed),
            "elapsed_s": float(self.elapsed_s),
            "duration_s": duration,
            "progress": float(np.clip(progress, 0.0, 1.0)),
            "joint_names": list(self.trajectory.joint_names),
            "samples": int(len(self.trajectory)),
            "overlay_joint_names": (
                [] if self.overlay is None else list(self.overlay.joint_names)
            ),
        }


class TiledTrajectoryBuffer:
    """per-env/per-robot 轨迹缓冲。

    Buffer 的设计目标是让 planner 和 runtime 解耦：

    * planner 只把 ready trajectory 写入 buffer；
    * runtime 每个同步 command step 调用 ``step``；
    * 没有 ready trajectory 的 env 不阻塞，直接 hold 当前 target；
    * 每个 env 的轨迹可以长度不同，但采样结果始终合成同一个 batched target。
    """

    def __init__(self, *, num_envs: int) -> None:
        """创建空轨迹缓冲。"""

        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self._playbacks: dict[str, dict[int, list[_Playback]]] = {}

    def load(
        self,
        *,
        robot_name: str,
        env_ids: Sequence[int] | np.ndarray,
        times: Sequence[float] | np.ndarray,
        positions: Sequence[object] | np.ndarray,
        joint_names: Sequence[str],
        request_id: str | None = None,
        source: str = "manual",
        replace: bool = True,
        overlays: Sequence[TiledTrajectoryOverlay] = (),
        append: bool = False,
        dynamic_base: bool = False,
    ) -> tuple[int, ...]:
        """载入一组 selected env 的轨迹。

        ``positions`` 支持两种形状：

        * ``(T, D)``：同一条轨迹广播给所有 selected env；
        * ``(E, T, D)``：每个 selected env 一条轨迹，``E == len(env_ids)``。
        """

        robot = _robot_name(robot_name)
        selected = _normalize_env_ids(env_ids, self.num_envs)
        sample_times = _validate_times(times)
        position_batch = _normalize_positions(
            positions,
            env_count=selected.size,
            sample_count=sample_times.size,
            label="positions",
        )
        names = tuple(str(name) for name in joint_names)
        if len(names) != int(position_batch.shape[2]):
            raise ValueError(
                f"joint_names expected {position_batch.shape[2]} names, got {len(names)}"
            )
        playback_rows = _playback_sequences_for_envs(
            times=sample_times,
            positions=position_batch,
            joint_names=names,
            overlays=overlays,
            request_id=request_id,
            source=source,
            dynamic_base=dynamic_base,
        )
        robot_playbacks = self._playbacks.setdefault(robot, {})
        for env_id in selected:
            existing_queue = robot_playbacks.get(int(env_id), [])
            existing = _active_or_completed_playback(existing_queue)
            if (
                not bool(append)
                and existing is not None
                and not existing.completed
                and not bool(replace)
            ):
                raise ValueError(
                    f"trajectory for robot {robot!r} env {int(env_id)} is still active"
                )
        for row, env_id in enumerate(selected):
            sequence = list(playback_rows[row])
            if not sequence:
                raise ValueError("trajectory playback sequence cannot be empty")
            if bool(append) and not bool(replace):
                existing_queue = _drop_completed_tail(robot_playbacks.get(int(env_id), []))
                robot_playbacks[int(env_id)] = existing_queue + sequence
            else:
                robot_playbacks[int(env_id)] = sequence
        return tuple(int(env_id) for env_id in selected)

    def clear(
        self,
        *,
        robot_name: str | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> dict[str, list[int]]:
        """清理轨迹缓冲，返回每个机器人实际清理的 env id。"""

        selected = (
            None
            if env_ids is None
            else _normalize_env_ids(env_ids, self.num_envs)
        )
        robots = (
            tuple(self._playbacks)
            if robot_name is None
            else (_robot_name(robot_name),)
        )
        cleared: dict[str, list[int]] = {}
        for robot in robots:
            playbacks = self._playbacks.get(robot)
            if not playbacks:
                continue
            clear_ids = (
                tuple(playbacks)
                if selected is None
                else tuple(int(env_id) for env_id in selected if int(env_id) in playbacks)
            )
            for env_id in clear_ids:
                playbacks.pop(int(env_id), None)
            if clear_ids:
                cleared[robot] = [int(env_id) for env_id in clear_ids]
            if not playbacks:
                self._playbacks.pop(robot, None)
        return cleared

    def step(
        self,
        *,
        robot_name: str,
        current_positions: np.ndarray,
        dt_s: float,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> TiledTrajectoryStepResult:
        """采样 selected env 的 ready trajectory，并把其它 env 保持为当前 target。"""

        robot = _robot_name(robot_name)
        selected = (
            np.arange(self.num_envs, dtype=int)
            if env_ids is None
            else _normalize_env_ids(env_ids, self.num_envs)
        )
        if float(dt_s) < 0.0:
            raise ValueError("dt_s cannot be negative")
        targets = _current_positions(current_positions, self.num_envs)
        playbacks = self._playbacks.get(robot, {})
        active: list[int] = []
        completed: list[int] = []
        idle: list[int] = []
        for env_id in selected:
            queue = playbacks.get(int(env_id), [])
            playback = _active_or_completed_playback(queue)
            if playback is None:
                idle.append(int(env_id))
                continue
            sample = np.asarray(
                playback.sample(
                    float(dt_s),
                    current_positions=targets[int(env_id), :],
                ),
                dtype=float,
            ).reshape(-1)
            if sample.size != targets.shape[1]:
                raise ValueError(
                    "trajectory width "
                    f"{sample.size} does not match current width {targets.shape[1]}"
                )
            targets[int(env_id), :] = sample
            if playback.completed:
                completed.append(int(env_id))
                if len(queue) > 1:
                    queue.pop(0)
            else:
                active.append(int(env_id))
        return TiledTrajectoryStepResult(
            robot_name=robot,
            env_ids=tuple(int(env_id) for env_id in selected),
            joint_positions=targets,
            active_env_ids=tuple(active),
            completed_env_ids=tuple(completed),
            idle_env_ids=tuple(idle),
            dt_s=float(dt_s),
        )

    def status(
        self,
        *,
        robot_name: str | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> dict[str, object]:
        """返回当前缓冲状态。"""

        selected = (
            None
            if env_ids is None
            else set(int(env_id) for env_id in _normalize_env_ids(env_ids, self.num_envs))
        )
        robots = (
            tuple(self._playbacks)
            if robot_name is None
            else (_robot_name(robot_name),)
        )
        payload: dict[str, object] = {
            "num_envs": self.num_envs,
            "robots": {},
        }
        robot_payload = payload["robots"]
        assert isinstance(robot_payload, dict)
        for robot in robots:
            playbacks = self._playbacks.get(robot, {})
            entries = []
            for env_id in sorted(playbacks):
                if selected is not None and int(env_id) not in selected:
                    continue
                queue = playbacks[int(env_id)]
                playback = _active_or_completed_playback(queue)
                if playback is None:
                    continue
                item = playback.status(int(env_id))
                item["queue_length"] = len(queue)
                entries.append(item)
            robot_payload[robot] = {
                "count": len(entries),
                "active_env_ids": [
                    item["env_id"]
                    for item in entries
                    if not bool(item["completed"]) or int(item["queue_length"]) > 1
                ],
                "completed_env_ids": [
                    item["env_id"]
                    for item in entries
                    if bool(item["completed"]) and int(item["queue_length"]) <= 1
                ],
                "envs": entries,
            }
        return payload


def _robot_name(value: str) -> str:
    """规范化机器人名。"""

    name = str(value).strip()
    if not name:
        raise ValueError("robot_name cannot be empty")
    return name


def _normalize_env_ids(env_ids: Sequence[int] | np.ndarray, num_envs: int) -> np.ndarray:
    """把 env_ids 规范化为一维唯一 int 数组。"""

    array = np.asarray(env_ids, dtype=int)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError("env_ids must be a 1D array")
    if array.size == 0:
        raise ValueError("env_ids cannot be empty")
    if np.unique(array).size != array.size:
        raise ValueError("env_ids cannot contain duplicates")
    if np.any(array < 0) or np.any(array >= int(num_envs)):
        raise ValueError("env_ids contains out-of-range env id")
    return array.astype(int, copy=True)


def _validate_times(times: Sequence[float] | np.ndarray) -> np.ndarray:
    """校验轨迹采样时间。"""

    array = np.asarray(times, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("times cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("times must be finite")
    if array.size > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return array


def _normalize_positions(
    positions: Sequence[object] | np.ndarray,
    *,
    env_count: int,
    sample_count: int,
    label: str,
) -> np.ndarray:
    """把 ``(T,D)`` 或 ``(E,T,D)`` 轨迹矩阵规范化为 ``(E,T,D)``。"""

    array = np.asarray(positions, dtype=float)
    if array.ndim == 2:
        if array.shape[0] != int(sample_count):
            raise ValueError(f"{label} sample dimension must match times")
        array = np.repeat(array.reshape(1, *array.shape), int(env_count), axis=0)
    elif array.ndim == 3:
        if array.shape[1] != int(sample_count):
            raise ValueError(f"{label} sample dimension must match times")
        if array.shape[0] == 1 and int(env_count) != 1:
            array = np.repeat(array, int(env_count), axis=0)
        if array.shape[0] != int(env_count):
            raise ValueError(f"{label} env dimension must be 1 or len(env_ids)")
    else:
        raise ValueError(f"{label} must have shape (T, D) or (E, T, D)")
    if array.shape[2] < 1:
        raise ValueError(f"{label} joint dimension cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array.astype(float, copy=True)


def _overlay_matrix(values: np.ndarray, width: int, label: str) -> np.ndarray:
    """把 overlay 起点/终点矩阵规范化为 ``(E,K)``。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != int(width):
        raise ValueError(f"overlay {label} must have shape (K,) or (E,K)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"overlay {label} must be finite")
    return array.astype(float, copy=True)


def _playback_sequences_for_envs(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    joint_names: tuple[str, ...],
    overlays: Sequence[TiledTrajectoryOverlay],
    request_id: str | None,
    source: str,
    dynamic_base: bool,
) -> tuple[tuple[_Playback, ...], ...]:
    """为每个 selected env 构造 before/main/after playback 队列。"""

    sample_times = np.asarray(times, dtype=float).reshape(-1)
    batch = np.asarray(positions, dtype=float)
    env_count = int(batch.shape[0])
    command_dim = int(batch.shape[2])
    _validate_overlay_indices(overlays, command_dim=command_dim)
    return tuple(
        _playback_sequence_for_env(
            row=row,
            times=sample_times,
            positions=batch[row],
            joint_names=joint_names,
            overlays=overlays,
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
    overlays: Sequence[TiledTrajectoryOverlay],
    env_count: int,
    request_id: str | None,
    source: str,
    dynamic_base: bool,
) -> tuple[_Playback, ...]:
    """构造单个 env 的 playback 队列。"""

    main_duration_s = max(0.0, float(times[-1]) - float(times[0]))
    cursor = np.asarray(positions[0], dtype=float).reshape(-1).copy()
    cursor = _initial_cursor_with_overlay_starts(
        cursor,
        overlays=overlays,
        row=row,
        env_count=env_count,
    )
    sequence: list[_Playback] = []

    before = _stage_playback(
        timing="before",
        cursor=cursor,
        overlays=overlays,
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
            overlays=overlays,
            timing="before",
            row=row,
            env_count=env_count,
        )

    main_positions = np.asarray(positions, dtype=float).copy()
    sync_indices = _timing_indices(overlays, "sync")
    for index in _timing_indices(overlays, "before") - sync_indices:
        main_positions[:, index] = cursor[index]
    sync_overlay = _playback_overlay_for_timing(
        overlays,
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
            overlay=sync_overlay,
            stage="trajectory",
            dynamic_base=dynamic_base,
        )
    )
    cursor = np.asarray(main_positions[-1], dtype=float).reshape(-1).copy()
    cursor = _cursor_after_timing(
        cursor,
        overlays=overlays,
        timing="sync",
        row=row,
        env_count=env_count,
    )

    after = _stage_playback(
        timing="after",
        cursor=cursor,
        overlays=overlays,
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
    overlays: Sequence[TiledTrajectoryOverlay],
    row: int,
    env_count: int,
    joint_names: tuple[str, ...],
    default_duration_s: float,
    request_id: str | None,
    source: str,
) -> _Playback | None:
    """构造 before/after hand-only stage。"""

    overlay = _playback_overlay_for_timing(
        overlays,
        timing=timing,
        row=row,
        env_count=env_count,
        cursor=cursor,
        joint_names=joint_names,
        default_duration_s=default_duration_s,
    )
    if overlay is None:
        return None
    duration_s = float(np.nanmax(overlay.durations_s)) if overlay.durations_s.size else 0.0
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
        overlay=overlay,
        stage=timing,
    )


def _playback_overlay_for_timing(
    overlays: Sequence[TiledTrajectoryOverlay],
    *,
    timing: str,
    row: int,
    env_count: int,
    cursor: np.ndarray,
    joint_names: tuple[str, ...],
    default_duration_s: float,
) -> _PlaybackOverlay | None:
    """把指定 timing 的 overlay 合并成单个 env 的 sampler。"""

    merged_indices: list[int] = []
    merged_names: list[str] = []
    starts: list[float] = []
    targets: list[float] = []
    durations: list[float] = []
    seen: set[int] = set()
    for overlay in overlays:
        if overlay.timing != timing:
            continue
        indices = tuple(int(index) for index in overlay.joint_indices)
        duplicate = [index for index in indices if index in seen]
        if duplicate:
            raise ValueError(f"overlay joint_indices duplicated across overlays: {duplicate}")
        seen.update(indices)
        target = _overlay_rows(overlay.target_positions, env_count, "target_positions")
        merged_indices.extend(indices)
        merged_names.extend(joint_names[index] for index in indices)
        starts.extend(float(cursor[index]) for index in indices)
        targets.extend(float(value) for value in target[row])
        duration = (
            float(default_duration_s)
            if overlay.duration_s is None
            else float(overlay.duration_s)
        )
        durations.extend(duration for _ in indices)
    if not merged_indices:
        return None
    joint_indices = np.asarray(merged_indices, dtype=int)
    return _PlaybackOverlay(
        joint_indices=joint_indices,
        start_positions=np.asarray(starts, dtype=float),
        target_positions=np.asarray(targets, dtype=float),
        durations_s=np.asarray(durations, dtype=float),
        joint_names=tuple(merged_names),
    )


def _validate_overlay_indices(
    overlays: Sequence[TiledTrajectoryOverlay],
    *,
    command_dim: int,
) -> None:
    """校验 overlay 指向的 command-space 列存在。"""

    for overlay in overlays:
        out_of_range = [
            index for index in overlay.joint_indices if index >= int(command_dim)
        ]
        if out_of_range:
            raise ValueError(f"overlay joint_indices out of range: {out_of_range}")


def _initial_cursor_with_overlay_starts(
    cursor: np.ndarray,
    *,
    overlays: Sequence[TiledTrajectoryOverlay],
    row: int,
    env_count: int,
) -> np.ndarray:
    """用 overlay load 时的 current target 修正初始 cursor。"""

    result = np.asarray(cursor, dtype=float).reshape(-1).copy()
    initialized: set[int] = set()
    for overlay in overlays:
        start = _overlay_rows(overlay.start_positions, env_count, "start_positions")
        for offset, index in enumerate(overlay.joint_indices):
            if int(index) in initialized:
                continue
            result[int(index)] = float(start[row, offset])
            initialized.add(int(index))
    return result


def _cursor_after_timing(
    cursor: np.ndarray,
    *,
    overlays: Sequence[TiledTrajectoryOverlay],
    timing: str,
    row: int,
    env_count: int,
) -> np.ndarray:
    """把指定 timing 的 overlay 终点写入滚动 cursor。"""

    result = np.asarray(cursor, dtype=float).reshape(-1).copy()
    for overlay in overlays:
        if overlay.timing != timing:
            continue
        target = _overlay_rows(overlay.target_positions, env_count, "target_positions")
        for offset, index in enumerate(overlay.joint_indices):
            result[int(index)] = float(target[row, offset])
    return result


def _timing_indices(
    overlays: Sequence[TiledTrajectoryOverlay],
    timing: str,
) -> set[int]:
    """返回某个 timing 会覆盖的 command-space 列。"""

    return {
        int(index)
        for overlay in overlays
        if overlay.timing == timing
        for index in overlay.joint_indices
    }


def _active_or_completed_playback(queue: Sequence[_Playback]) -> _Playback | None:
    """返回当前队列头。"""

    return queue[0] if queue else None


def _drop_completed_tail(queue: Sequence[_Playback]) -> list[_Playback]:
    """追加新动作前丢弃没有后续意义的单个 completed 队列头。"""

    items = list(queue)
    if len(items) == 1 and items[0].completed:
        return []
    return items


def _overlay_rows(values: np.ndarray, env_count: int, label: str) -> np.ndarray:
    """把 overlay ``(1,K)`` 或 ``(E,K)`` 扩展到 selected env 数量。"""

    array = np.asarray(values, dtype=float)
    if array.shape[0] == 1 and int(env_count) != 1:
        return np.repeat(array, int(env_count), axis=0)
    if array.shape[0] != int(env_count):
        raise ValueError(f"overlay {label} env dimension must be 1 or len(env_ids)")
    return array.astype(float, copy=True)


def _current_positions(current_positions: np.ndarray, num_envs: int) -> np.ndarray:
    """校验并复制当前 batched 关节目标。"""

    array = np.asarray(current_positions, dtype=float)
    if array.ndim != 2 or array.shape[0] != int(num_envs) or array.shape[1] < 1:
        raise ValueError("current_positions must have shape (num_envs, command_dim)")
    return array.astype(float, copy=True)
