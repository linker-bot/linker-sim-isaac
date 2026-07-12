"""Tiled trajectory playback 的输入 track、游标和 step result 模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linkerbot_sim.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class PlaybackJointTrack:
    """随主轨迹或独立 stage 采样的部分 command-space 关节轨。

    ``joint_indices`` 指向完整 command-space 列；起止位置可用 ``(K,)`` 广播到所有
    selected env，也可用 ``(E,K)`` 逐 env 指定。``timing`` 决定该轨在主轨迹之前、
    同步或之后执行。
    """

    joint_indices: tuple[int, ...]
    start_positions: np.ndarray
    target_positions: np.ndarray
    duration_s: float | None = None
    timing: str = "sync"

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.joint_indices)
        if not indices:
            raise ValueError("playback joint_indices cannot be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("playback joint_indices cannot contain duplicates")
        if any(index < 0 for index in indices):
            raise ValueError("playback joint_indices cannot be negative")
        start = _joint_track_matrix(
            self.start_positions, len(indices), "start_positions"
        )
        target = _joint_track_matrix(
            self.target_positions, len(indices), "target_positions"
        )
        if start.shape[0] not in {1, target.shape[0]} and target.shape[0] != 1:
            raise ValueError(
                "playback joint track start/target env dimensions are incompatible"
            )
        if self.duration_s is not None and (
            not np.isfinite(float(self.duration_s)) or float(self.duration_s) < 0.0
        ):
            raise ValueError(
                "playback joint track duration_s must be finite and non-negative"
            )
        timing = str(self.timing)
        if timing not in {"before", "sync", "after"}:
            raise ValueError(
                "playback joint track timing must be one of: before, sync, after"
            )
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
    """一次 tiled 轨迹采样得到的完整 batched command target。"""

    robot_name: str
    env_ids: tuple[int, ...]
    joint_positions: np.ndarray
    active_env_ids: tuple[int, ...]
    completed_env_ids: tuple[int, ...]
    idle_env_ids: tuple[int, ...]
    dt_s: float

    def __post_init__(self) -> None:
        """拒绝非有限 playback 输出，避免异常 target 进入 action 边界。"""

        positions = np.asarray(self.joint_positions, dtype=float)
        if positions.ndim != 2 or not np.all(np.isfinite(positions)):
            raise ValueError("joint_positions must be a finite 2D array")
        dt_s = float(self.dt_s)
        if not np.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be finite and non-negative")
        object.__setattr__(self, "joint_positions", positions.copy())
        object.__setattr__(self, "dt_s", dt_s)

    def to_json(self) -> dict[str, object]:
        """返回不包含大型 joint matrix 的 playback step 摘要。"""

        return {
            "robot": self.robot_name,
            "env_ids": list(self.env_ids),
            "active_env_ids": list(self.active_env_ids),
            "completed_env_ids": list(self.completed_env_ids),
            "idle_env_ids": list(self.idle_env_ids),
            "dt_s": self.dt_s,
        }


@dataclass(frozen=True)
class _PlaybackJointTrack:
    """已经绑定到单个 env 与 command columns 的部分关节采样器。"""

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
        """在指定 stage 时间域内插值 sparse joint columns，并覆盖完整 command row。"""

        result = np.asarray(positions, dtype=float).reshape(-1).copy()
        start = (
            self.start_positions
            if start_positions is None
            else np.asarray(start_positions, dtype=float).reshape(-1)
        )
        domain_duration = max(0.0, float(upper_s) - float(lower_s))
        durations = np.where(
            np.isnan(self.durations_s), domain_duration, self.durations_s
        )
        elapsed = max(0.0, float(elapsed_s) - float(lower_s))
        alpha = np.ones_like(durations, dtype=float)
        positive = durations > 0.0
        alpha[positive] = np.clip(elapsed / durations[positive], 0.0, 1.0)
        result[self.joint_indices] = start + alpha * (self.target_positions - start)
        return result


@dataclass
class _Playback:
    """单个 env 上一条 staged trajectory 的可变播放游标。"""

    trajectory: JointTrajectory
    elapsed_s: float
    completed: bool
    request_id: str | None
    source: str
    joint_track: _PlaybackJointTrack | None = None
    stage: str = "trajectory"
    dynamic_base: bool = False
    dynamic_track_start: np.ndarray | None = None

    @classmethod
    def from_trajectory(
        cls,
        trajectory: JointTrajectory,
        *,
        request_id: str | None,
        source: str,
        joint_track: _PlaybackJointTrack | None = None,
        stage: str = "trajectory",
        dynamic_base: bool = False,
    ) -> "_Playback":
        """从 immutable trajectory 创建位于 domain 起点的可变播放游标。"""

        lower, upper = trajectory.domain()
        return cls(
            trajectory=trajectory,
            elapsed_s=float(lower),
            completed=bool(np.isclose(lower, upper)),
            request_id=request_id,
            source=str(source),
            joint_track=joint_track,
            stage=str(stage),
            dynamic_base=bool(dynamic_base),
        )

    def sample(
        self,
        dt_s: float,
        *,
        current_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """推进局部时间，并以需要时的动态 command state 为基线采样。"""

        lower, upper = self.trajectory.domain()
        if not self.completed:
            self.elapsed_s = min(float(upper), max(float(lower), self.elapsed_s + dt_s))
            self.completed = self.elapsed_s >= float(upper) - 1.0e-12
        if self.dynamic_base and current_positions is not None:
            sample = np.asarray(current_positions, dtype=float).reshape(-1).copy()
        else:
            sample = self.trajectory.eval(self.elapsed_s)
        if self.joint_track is not None:
            if self.dynamic_base and self.dynamic_track_start is None:
                self.dynamic_track_start = sample[self.joint_track.joint_indices].copy()
            return self.joint_track.sample(
                sample,
                elapsed_s=self.elapsed_s,
                lower_s=float(lower),
                upper_s=float(upper),
                start_positions=self.dynamic_track_start,
            )
        return sample

    def status(self, env_id: int) -> dict[str, object]:
        """返回单 env playback 的进度、来源、stage 和 sparse joint 摘要。"""

        lower, upper = self.trajectory.domain()
        duration = max(0.0, float(upper) - float(lower))
        progress = (
            1.0 if duration <= 0.0 else (self.elapsed_s - float(lower)) / duration
        )
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
            "joint_track_names": (
                [] if self.joint_track is None else list(self.joint_track.joint_names)
            ),
        }


def _joint_track_matrix(values: np.ndarray, width: int, label: str) -> np.ndarray:
    """把 sparse track `(K,)`/`(E,K)` 输入规范为有限二维矩阵。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != int(width):
        raise ValueError(f"playback joint track {label} must have shape (K,) or (E,K)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"playback joint track {label} must be finite")
    return array.astype(float, copy=True)
