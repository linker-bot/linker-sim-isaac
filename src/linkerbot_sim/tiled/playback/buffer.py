"""按 env/robot 隔离的 tiled trajectory playback buffer。"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from linkerbot_sim.tiled.playback.models import (
    PlaybackJointTrack,
    TiledTrajectoryStepResult,
    _Playback,
)
from linkerbot_sim.tiled.playback.staging import playback_sequences_for_envs


class TiledTrajectoryBuffer:
    """把 ready trajectories 解耦为逐 env 队列和完整 batched command targets。

    Planner 只负责 load，runtime 在同步 command boundary 调用 ``step``。不同 env 可有
    不同队列长度；没有 active playback 的 env 始终保持传入的当前 target。
    """

    def __init__(
        self,
        *,
        num_envs: int,
        max_queue_depth_per_env: int = 32,
        max_samples_per_env: int = 100_000,
        max_duration_s_per_env: float = 3600.0,
        overflow_policy: str = "reject",
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if (
            isinstance(max_queue_depth_per_env, bool)
            or int(max_queue_depth_per_env) < 1
        ):
            raise ValueError("max_queue_depth_per_env must be positive")
        if isinstance(max_samples_per_env, bool) or int(max_samples_per_env) < 1:
            raise ValueError("max_samples_per_env must be positive")
        if isinstance(max_duration_s_per_env, bool) or (
            not math.isfinite(float(max_duration_s_per_env))
            or float(max_duration_s_per_env) <= 0.0
        ):
            raise ValueError("max_duration_s_per_env must be positive and finite")
        policy = str(overflow_policy).strip()
        if policy != "reject":
            raise ValueError("overflow_policy must be 'reject'")
        self.num_envs = int(num_envs)
        self.max_queue_depth_per_env = int(max_queue_depth_per_env)
        self.max_samples_per_env = int(max_samples_per_env)
        self.max_duration_s_per_env = float(max_duration_s_per_env)
        self.overflow_policy = policy
        self._playbacks: dict[str, dict[int, list[_Playback]]] = {}
        self._rejected_loads = 0
        self._rejected_loads_by_robot: dict[str, int] = {}

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
        joint_tracks: Sequence[PlaybackJointTrack] = (),
        append: bool = False,
        dynamic_base: bool = False,
    ) -> tuple[int, ...]:
        """载入 ``(T,D)`` 广播轨迹或 ``(E,T,D)`` per-env 轨迹。"""

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
                f"joint_names expected {position_batch.shape[2]} names, "
                f"got {len(names)}"
            )
        playback_rows = playback_sequences_for_envs(
            times=sample_times,
            positions=position_batch,
            joint_names=names,
            joint_tracks=joint_tracks,
            request_id=request_id,
            source=source,
            dynamic_base=dynamic_base,
        )
        # append 表示队列追加；即使 replace 仍为默认 True，也不能清空正在运行的队列。
        append_mode = bool(append)
        replace_mode = bool(replace)
        robot_playbacks = self._playbacks.get(robot, {})
        for env_id in selected:
            existing_queue = robot_playbacks.get(int(env_id), [])
            existing = _active_or_completed_playback(existing_queue)
            if (
                not append_mode
                and not replace_mode
                and existing is not None
                and not existing.completed
            ):
                raise ValueError(
                    f"trajectory for robot {robot!r} env {int(env_id)} is still active"
                )
        candidate_queues: dict[int, list[_Playback]] = {}
        for row, env_id in enumerate(selected):
            sequence = list(playback_rows[row])
            if not sequence:
                raise ValueError("trajectory playback sequence cannot be empty")
            if append_mode:
                existing_queue = _drop_completed_tail(
                    robot_playbacks.get(int(env_id), [])
                )
                candidate_queues[int(env_id)] = existing_queue + sequence
            else:
                candidate_queues[int(env_id)] = sequence
        violations = self._capacity_violations(candidate_queues)
        if violations:
            self._rejected_loads += 1
            self._rejected_loads_by_robot[robot] = (
                self._rejected_loads_by_robot.get(robot, 0) + 1
            )
            raise ValueError(
                "trajectory playback capacity exceeded; " + "; ".join(violations)
            )
        if robot not in self._playbacks:
            robot_playbacks = {}
            self._playbacks[robot] = robot_playbacks
        robot_playbacks.update(candidate_queues)
        return tuple(int(env_id) for env_id in selected)

    def _capacity_violations(self, queues: dict[int, list[_Playback]]) -> list[str]:
        """返回候选 queues 的全部容量违规，调用方据此执行原子拒绝。"""

        violations: list[str] = []
        for env_id, queue in queues.items():
            trajectories, samples, duration_s = _queue_capacity(queue)
            exceeded = []
            if trajectories > self.max_queue_depth_per_env:
                exceeded.append(
                    f"trajectories={trajectories}>{self.max_queue_depth_per_env}"
                )
            if samples > self.max_samples_per_env:
                exceeded.append(f"samples={samples}>{self.max_samples_per_env}")
            if duration_s > self.max_duration_s_per_env + 1.0e-12:
                exceeded.append(
                    f"duration_s={duration_s:g}>{self.max_duration_s_per_env:g}"
                )
            if exceeded:
                violations.append(f"env {env_id}: {', '.join(exceeded)}")
        return violations

    def clear(
        self,
        *,
        robot_name: str | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> dict[str, list[int]]:
        """清理 selected playback 队列并返回实际清理的 env IDs。"""

        selected = (
            None if env_ids is None else _normalize_env_ids(env_ids, self.num_envs)
        )
        robots = (
            tuple(self._playbacks) if robot_name is None else (_robot_name(robot_name),)
        )
        cleared: dict[str, list[int]] = {}
        for robot in robots:
            playbacks = self._playbacks.get(robot)
            if not playbacks:
                continue
            clear_ids = (
                tuple(playbacks)
                if selected is None
                else tuple(
                    int(env_id) for env_id in selected if int(env_id) in playbacks
                )
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
        """推进 selected env 队列头，并合成完整 batched target。"""

        robot = _robot_name(robot_name)
        selected = (
            np.arange(self.num_envs, dtype=int)
            if env_ids is None
            else _normalize_env_ids(env_ids, self.num_envs)
        )
        step_dt = float(dt_s)
        if not math.isfinite(step_dt) or step_dt < 0.0:
            raise ValueError("dt_s must be finite and non-negative")
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
                    step_dt,
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
            dt_s=step_dt,
        )

    def status(
        self,
        *,
        robot_name: str | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> dict[str, object]:
        """返回 robot/env 队列头、进度与 queue length。"""

        selected = (
            None
            if env_ids is None
            else {int(env_id) for env_id in _normalize_env_ids(env_ids, self.num_envs)}
        )
        robots = (
            tuple(self._playbacks) if robot_name is None else (_robot_name(robot_name),)
        )
        rejected_loads = (
            self._rejected_loads
            if robot_name is None
            else self._rejected_loads_by_robot.get(robots[0], 0)
        )
        payload: dict[str, object] = {
            "num_envs": self.num_envs,
            "limits": {
                "max_queue_depth_per_env": self.max_queue_depth_per_env,
                "max_samples_per_env": self.max_samples_per_env,
                "max_duration_s_per_env": self.max_duration_s_per_env,
                "overflow_policy": self.overflow_policy,
            },
            "queued_trajectories": 0,
            "queued_samples": 0,
            "queued_duration_s": 0.0,
            "rejected_loads": rejected_loads,
            "rejected_loads_scope": "buffer" if robot_name is None else "robot",
            "robots": {},
        }
        robot_payload = payload["robots"]
        assert isinstance(robot_payload, dict)
        total_trajectories = 0
        total_samples = 0
        total_duration_s = 0.0
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
                trajectories, samples, duration_s = _queue_capacity(queue)
                item["queued_trajectories"] = trajectories
                item["queued_samples"] = samples
                item["queued_duration_s"] = duration_s
                entries.append(item)
            queued_trajectories = sum(
                int(item["queued_trajectories"]) for item in entries
            )
            queued_samples = sum(int(item["queued_samples"]) for item in entries)
            queued_duration_s = sum(
                float(item["queued_duration_s"]) for item in entries
            )
            robot_payload[robot] = {
                "count": len(entries),
                "queued_trajectories": queued_trajectories,
                "queued_samples": queued_samples,
                "queued_duration_s": queued_duration_s,
                "rejected_loads": self._rejected_loads_by_robot.get(robot, 0),
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
            total_trajectories += queued_trajectories
            total_samples += queued_samples
            total_duration_s += queued_duration_s
        payload["queued_trajectories"] = total_trajectories
        payload["queued_samples"] = total_samples
        payload["queued_duration_s"] = total_duration_s
        return payload


def _robot_name(value: str) -> str:
    """规范化内部 robot label，并拒绝空名称。"""

    name = str(value).strip()
    if not name:
        raise ValueError("robot_name cannot be empty")
    return name


def _normalize_env_ids(
    env_ids: Sequence[int] | np.ndarray,
    num_envs: int,
) -> np.ndarray:
    """规范化、去重并校验 buffer selector env IDs。"""

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
    """冻结非空有限且严格递增的 trajectory time vector。"""

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
    """把 ``(T,D)`` 或 ``(E,T,D)`` 规范化为 ``(E,T,D)``。"""

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


def _active_or_completed_playback(
    queue: Sequence[_Playback],
) -> _Playback | None:
    """返回 queue head；刚完成的 head 在一次 status/step 周期内仍可见。"""

    return queue[0] if queue else None


def _drop_completed_tail(queue: Sequence[_Playback]) -> list[_Playback]:
    """移除没有后继项的 completed head，保留有 queued successor 的切换语义。"""

    items = list(queue)
    if len(items) == 1 and items[0].completed:
        return []
    return items


def _queue_capacity(queue: Sequence[_Playback]) -> tuple[int, int, float]:
    """返回 staged trajectory 数、样本总数与 domain duration 总和。"""

    samples = 0
    duration_s = 0.0
    for playback in queue:
        lower, upper = playback.trajectory.domain()
        samples += len(playback.trajectory)
        duration_s += max(0.0, float(upper) - float(lower))
    return len(queue), samples, duration_s


def _current_positions(current_positions: np.ndarray, num_envs: int) -> np.ndarray:
    """校验并复制完整 `(num_envs, command_dim)` command baseline。"""

    array = np.asarray(current_positions, dtype=float)
    if array.ndim != 2 or array.shape[0] != int(num_envs) or array.shape[1] < 1:
        raise ValueError("current_positions must have shape (num_envs, command_dim)")
    if not np.all(np.isfinite(array)):
        raise ValueError("current_positions must contain finite values")
    return array.astype(float, copy=True)


__all__ = ["TiledTrajectoryBuffer"]
