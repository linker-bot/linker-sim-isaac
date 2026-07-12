"""cuRobo joint batch core 的纯数组输入输出。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CuroboBatchJointProblem:
    """按 batch row 排列的一次同构 command-space 关节规划。"""

    current_positions: np.ndarray
    goal_positions: np.ndarray
    command_joint_names: tuple[str, ...]
    duration_s: float
    sample_dt_s: float
    avoid_collisions: bool = False

    def __post_init__(self) -> None:
        current = np.asarray(self.current_positions, dtype=float)
        goal = np.asarray(self.goal_positions, dtype=float)
        names = tuple(str(name) for name in self.command_joint_names)
        if current.ndim != 2 or current.shape[0] < 1:
            raise ValueError("current_positions must have shape (B, C) with B >= 1")
        if goal.shape != current.shape:
            raise ValueError("goal_positions must match current_positions")
        if len(names) != current.shape[1]:
            raise ValueError("command_joint_names must match command dimension")
        if float(self.duration_s) <= 0.0:
            raise ValueError("duration_s must be positive")
        if float(self.sample_dt_s) <= 0.0:
            raise ValueError("sample_dt_s must be positive")
        object.__setattr__(self, "current_positions", current.copy())
        object.__setattr__(self, "goal_positions", goal.copy())
        object.__setattr__(self, "command_joint_names", names)
        object.__setattr__(self, "duration_s", float(self.duration_s))
        object.__setattr__(self, "sample_dt_s", float(self.sample_dt_s))
        object.__setattr__(self, "avoid_collisions", bool(self.avoid_collisions))


@dataclass(frozen=True)
class CuroboBatchJointResult:
    """按输入 row 顺序返回的 success、时间轴和 command-space 轨迹。"""

    success: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    status: tuple[str, ...]
    message: str = ""

    def __post_init__(self) -> None:
        success = np.asarray(self.success, dtype=bool).reshape(-1)
        times = np.asarray(self.times, dtype=float).reshape(-1)
        positions = np.asarray(self.positions, dtype=float)
        status = tuple(str(value) for value in self.status)
        if positions.ndim != 3:
            raise ValueError("positions must have shape (B, T, C)")
        if positions.shape[0] != success.size:
            raise ValueError("positions batch dimension must match success")
        if positions.shape[1] != times.size:
            raise ValueError("positions sample dimension must match times")
        if len(status) != success.size:
            raise ValueError("status must have one entry per batch row")
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "message", str(self.message))

    @property
    def all_succeeded(self) -> bool:
        """返回全部真实 batch rows 是否成功。"""

        return bool(np.all(self.success))

    @classmethod
    def failed(
        cls,
        problem: CuroboBatchJointProblem,
        *,
        status: str,
        message: str,
        success: np.ndarray | None = None,
    ) -> "CuroboBatchJointResult":
        """按 problem row 数创建无 trajectory 的失败结果，可保留部分成功 mask。"""

        rows, width = problem.current_positions.shape
        row_success = (
            np.zeros(rows, dtype=bool)
            if success is None
            else np.asarray(success, dtype=bool).reshape(rows)
        )
        row_status = tuple("SUCCESS" if ok else str(status) for ok in row_success)
        return cls(
            success=row_success,
            times=np.asarray([], dtype=float),
            positions=np.empty((rows, 0, width), dtype=float),
            status=row_status,
            message=message,
        )


__all__ = ["CuroboBatchJointProblem", "CuroboBatchJointResult"]
