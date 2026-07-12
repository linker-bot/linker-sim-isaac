"""实现共享 planner contract 的确定性 joint-space linear backend。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.planning.backend import PlanningRequest
from linkerbot_sim.planning.requests import LinearPosePathRequest
from linkerbot_sim.planning.results import MotionResult, PlanningDiagnostics
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.retiming import trajectory_sample_times


class LinearPlannerBackend:
    """不依赖 robot model，生成可直接执行的 joint-space interpolation。

    适用于调用方明确选择直线 joint interpolation 作为执行策略的场景。它不提供 IK、
    collision checking、joint-limit validation 或 velocity/acceleration constrained
    optimization。它只在调用方显式选择时使用，不会自动替代 cuRobo。
    """

    backend_name = "linear"

    def __init__(
        self,
        joint_names: Sequence[str],
        *,
        default_duration_s: float = 1.0,
        default_sample_dt_s: float | None = None,
    ) -> None:
        names = tuple(str(name) for name in joint_names)
        if not names or any(not name for name in names):
            raise ValueError("linear planner requires non-empty joint_names")
        if len(set(names)) != len(names):
            raise ValueError("linear planner joint_names cannot contain duplicates")
        if float(default_duration_s) <= 0.0:
            raise ValueError("linear planner default_duration_s must be positive")
        if default_sample_dt_s is not None and float(default_sample_dt_s) <= 0.0:
            raise ValueError("linear planner default_sample_dt_s must be positive")
        self._joint_names = names
        self._default_duration_s = float(default_duration_s)
        self._default_sample_dt_s = (
            None if default_sample_dt_s is None else float(default_sample_dt_s)
        )

    def joint_names(self) -> tuple[str, ...]:
        """返回构造 backend 时冻结的 command joint order。"""

        return self._joint_names

    def plan(self, request: PlanningRequest) -> MotionResult:
        """只处理 joint goal，并在 canonical time grid 上生成 linear trajectory。"""

        if isinstance(request, LinearPosePathRequest):
            request.validate_structure()
            return _failed(
                "UNSUPPORTED",
                "linear planner does not support task-space linear pose paths",
            )

        request.validate_structure()
        current = np.asarray(request.current_q, dtype=float).reshape(-1)
        if current.size != len(self._joint_names):
            raise ValueError(
                "linear planner current_q width does not match joint_names: "
                f"{current.size} != {len(self._joint_names)}"
            )
        if request.goal_q is None:
            return _failed(
                "UNSUPPORTED",
                "linear planner only supports joint-space goals",
            )
        if request.avoid_collisions:
            return _failed(
                "COLLISION_UNSUPPORTED",
                "linear planner cannot satisfy avoid_collisions=True",
            )

        goal = np.asarray(request.goal_q, dtype=float).reshape(-1)
        duration_s = (
            self._default_duration_s
            if request.duration_s is None
            else float(request.duration_s)
        )
        sample_dt_s = (
            self._default_sample_dt_s
            if request.sample_dt_s is None
            else float(request.sample_dt_s)
        )
        if sample_dt_s is None:
            raise ValueError(
                "linear planner sample_dt_s is required; inject the runtime physics dt "
                "as default_sample_dt_s or set it on the request"
            )
        times = trajectory_sample_times(
            duration_s=duration_s,
            sample_dt_s=sample_dt_s,
            include_start=True,
        )
        progress = (times / float(times[-1])).reshape(-1, 1)
        positions = current.reshape(1, -1) + progress * (goal - current).reshape(1, -1)
        trajectory = joint_trajectory_from_positions(
            times=times,
            positions=positions,
            joint_names=self._joint_names,
            phase="linear_joint_plan",
        )
        return MotionResult(
            path=positions.copy(),
            trajectory=trajectory,
            success=True,
            status="SUCCESS",
            diagnostics=PlanningDiagnostics(
                status="SUCCESS",
                message="linear joint trajectory generated",
                metrics={
                    "duration_s": float(times[-1]),
                    "sample_dt_s": sample_dt_s,
                    "samples": float(times.size),
                },
            ),
        )


def _failed(status: str, message: str) -> MotionResult:
    """构造不携带 trajectory 的 backend-neutral failure result。"""

    return MotionResult(
        path=None,
        trajectory=None,
        success=False,
        status=str(status),
        diagnostics=PlanningDiagnostics(status=str(status), message=str(message)),
    )


__all__ = ["LinearPlannerBackend"]
