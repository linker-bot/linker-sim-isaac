"""tiled 规划请求、结果与后端协议；不包含线程或求解器实现。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TiledPlanningSegment:
    """一次 tiled 规划请求中的一个逻辑运动段。"""

    kind: str
    duration_s: float | None = None
    sample_dt_s: float | None = None
    goal_positions: np.ndarray | None = None
    path: object | None = None
    tcp_frame_name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("planning segment kind cannot be empty")
        if self.duration_s is not None and (
            not np.isfinite(float(self.duration_s)) or float(self.duration_s) <= 0.0
        ):
            raise ValueError("planning segment duration_s must be finite and positive")
        if self.sample_dt_s is not None and (
            not np.isfinite(float(self.sample_dt_s)) or float(self.sample_dt_s) <= 0.0
        ):
            raise ValueError("planning segment sample_dt_s must be finite and positive")
        goal = None
        if self.goal_positions is not None:
            goal = np.asarray(self.goal_positions, dtype=float)
            if goal.ndim != 2:
                raise ValueError(
                    "planning segment goal_positions must have shape (E, D)"
                )
            if not np.all(np.isfinite(goal)):
                raise ValueError(
                    "planning segment goal_positions must contain finite values"
                )
        tcp_frame_name = (
            None if self.tcp_frame_name is None else str(self.tcp_frame_name).strip()
        )
        if tcp_frame_name == "":
            raise ValueError("planning segment tcp_frame_name cannot be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "goal_positions", None if goal is None else goal.copy()
        )
        object.__setattr__(self, "tcp_frame_name", tcp_frame_name)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TiledPlanningRequest:
    """主线程冻结后交给 planner worker 的 tiled request。"""

    request_id: str
    robot_name: str
    env_ids: tuple[int, ...]
    current_positions: np.ndarray
    joint_names: tuple[str, ...]
    sample_dt_s: float
    goal_positions: np.ndarray | None = None
    duration_s: float = 1.0
    source: str = "interactive"
    load_on_success: bool = True
    replace: bool = True
    avoid_collisions: bool = False
    segments: tuple[TiledPlanningSegment, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        env_ids = tuple(int(env_id) for env_id in self.env_ids)
        if not env_ids:
            raise ValueError("env_ids cannot be empty")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("env_ids cannot contain duplicates")
        current = np.asarray(self.current_positions, dtype=float)
        if current.ndim != 2:
            raise ValueError("current_positions must have shape (E, D)")
        if current.shape[0] != len(env_ids):
            raise ValueError("current_positions first dimension must match env_ids")
        if current.shape[1] != len(self.joint_names):
            raise ValueError("joint_names length must match command dimension")
        if not np.all(np.isfinite(current)):
            raise ValueError("current_positions must contain finite values")
        if not np.isfinite(float(self.duration_s)) or float(self.duration_s) <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if not np.isfinite(float(self.sample_dt_s)) or float(self.sample_dt_s) <= 0.0:
            raise ValueError("sample_dt_s must be finite and positive")
        goal = (
            None
            if self.goal_positions is None
            else np.asarray(self.goal_positions, dtype=float)
        )
        segments = tuple(self.segments)
        if not segments and goal is None:
            raise ValueError("planning request requires goal_positions or segments")
        if goal is not None and goal.shape != current.shape:
            raise ValueError("goal_positions must match current_positions shape")
        if goal is not None and not np.all(np.isfinite(goal)):
            raise ValueError("goal_positions must contain finite values")
        for index, segment in enumerate(segments):
            if (
                segment.goal_positions is not None
                and segment.goal_positions.shape != current.shape
            ):
                raise ValueError(
                    f"segments[{index}].goal_positions must match current_positions shape"
                )
        object.__setattr__(self, "env_ids", env_ids)
        object.__setattr__(self, "current_positions", current.copy())
        object.__setattr__(
            self, "goal_positions", None if goal is None else goal.copy()
        )
        object.__setattr__(
            self, "joint_names", tuple(str(name) for name in self.joint_names)
        )
        object.__setattr__(self, "sample_dt_s", float(self.sample_dt_s))
        object.__setattr__(self, "avoid_collisions", bool(self.avoid_collisions))
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TiledPlanningResult:
    """一次异步规划的 batched trajectory 或失败状态。"""

    request_id: str
    robot_name: str
    env_ids: tuple[int, ...]
    success: bool
    status: str
    message: str
    times: np.ndarray
    positions: np.ndarray
    joint_names: tuple[str, ...]
    source: str = "planner"
    load_on_success: bool = True
    replace: bool = True

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        positions = np.asarray(self.positions, dtype=float)
        if not np.all(np.isfinite(times)):
            raise ValueError("planning result times must contain finite values")
        if not np.all(np.isfinite(positions)):
            raise ValueError("planning result positions must contain finite values")
        if self.success:
            if times.size == 0:
                raise ValueError("successful planning result requires times")
            if positions.ndim != 3:
                raise ValueError("successful planning result positions must be (E,T,D)")
            if positions.shape[0] != len(self.env_ids):
                raise ValueError("positions first dimension must match env_ids")
            if positions.shape[1] != times.size:
                raise ValueError("positions sample dimension must match times")
            if positions.shape[2] != len(self.joint_names):
                raise ValueError("joint_names length must match positions width")
            if times.size > 1 and np.any(np.diff(times) <= 0.0):
                raise ValueError("planning result times must be strictly increasing")
        object.__setattr__(
            self, "env_ids", tuple(int(env_id) for env_id in self.env_ids)
        )
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(
            self, "joint_names", tuple(str(name) for name in self.joint_names)
        )

    @classmethod
    def failed(
        cls,
        request: TiledPlanningRequest,
        *,
        status: str,
        message: str,
    ) -> "TiledPlanningResult":
        """构造保留 request identity 和 playback metadata 的失败结果。"""

        return cls(
            request_id=request.request_id,
            robot_name=request.robot_name,
            env_ids=request.env_ids,
            success=False,
            status=str(status),
            message=str(message),
            times=np.asarray([], dtype=float),
            positions=np.empty((0, 0, 0), dtype=float),
            joint_names=request.joint_names,
            source=request.source,
            load_on_success=False,
            replace=request.replace,
        )

    def to_json(self) -> dict[str, object]:
        """返回不包含大型 trajectory 数组的状态摘要。"""

        return {
            "request_id": self.request_id,
            "robot": self.robot_name,
            "env_ids": list(self.env_ids),
            "success": bool(self.success),
            "status": self.status,
            "message": self.message,
            "samples": int(self.times.size),
            "joint_names": list(self.joint_names),
            "source": self.source,
            "load_on_success": bool(self.load_on_success),
        }


class TiledPlannerBackend(Protocol):
    """异步 manager 可调用的 planner backend 协议。"""

    def plan(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """执行一次规划并返回统一结果。"""


__all__ = [
    "TiledPlannerBackend",
    "TiledPlanningRequest",
    "TiledPlanningResult",
    "TiledPlanningSegment",
]
