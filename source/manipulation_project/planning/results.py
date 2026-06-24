"""运动解算结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlanningDiagnostics:
    """规划和解算诊断信息。"""

    status: str = ""
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IKResult:
    """逆运动学结果。"""

    joint_positions: np.ndarray
    success: bool
    position_error: float
    orientation_error: float | None = None
    message: str = ""
    status: str = ""
    num_solutions: int = 1


@dataclass(frozen=True)
class MotionResult:
    """路径级运动规划结果。"""

    joint_path: np.ndarray | None
    trajectory: object | None
    success: bool
    status: str
    diagnostics: PlanningDiagnostics = field(default_factory=PlanningDiagnostics)
