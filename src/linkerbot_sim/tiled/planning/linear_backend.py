"""把共享 scalar linear planner contract 适配到 tiled batched requests。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.planning.linear_backend import LinearPlannerBackend
from linkerbot_sim.planning.requests import MotionRequest
from linkerbot_sim.tiled.planning.types import (
    TiledPlanningRequest,
    TiledPlanningResult,
    TiledPlanningSegment,
)
from linkerbot_sim.trajectories.types import JointTrajectory


class LinearJointPlannerBackend:
    """逐 env 调用 canonical scalar linear planner，并堆叠为 tiled result。"""

    def plan(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """按 env row 串行规划 joint segments，并要求所有 row 使用相同 time grid。"""

        segments = _request_segments(request)
        if isinstance(segments, TiledPlanningResult):
            return segments
        backend = LinearPlannerBackend(request.joint_names)
        row_times: np.ndarray | None = None
        rows: list[np.ndarray] = []
        for row in range(len(request.env_ids)):
            planned = _plan_linear_row(
                backend,
                request=request,
                segments=segments,
                row=row,
            )
            if isinstance(planned, TiledPlanningResult):
                return planned
            times, positions = planned
            if row_times is None:
                row_times = times
            elif row_times.shape != times.shape or not np.allclose(row_times, times):
                return TiledPlanningResult.failed(
                    request,
                    status="INVALID_TIMES",
                    message="linear planner produced inconsistent row time grids",
                )
            rows.append(positions)
        assert row_times is not None
        return TiledPlanningResult(
            request_id=request.request_id,
            robot_name=request.robot_name,
            env_ids=request.env_ids,
            success=True,
            status="SUCCESS",
            message="linear joint trajectory generated",
            times=row_times,
            positions=np.stack(rows, axis=0),
            joint_names=request.joint_names,
            source=request.source,
            load_on_success=request.load_on_success,
            replace=request.replace,
        )


def _request_segments(
    request: TiledPlanningRequest,
) -> tuple[TiledPlanningSegment, ...] | TiledPlanningResult:
    """读取显式 segments，或把 single joint goal 规范为一个 segment。"""

    if request.segments:
        return request.segments
    if request.goal_positions is None:
        return TiledPlanningResult.failed(
            request,
            status="UNSUPPORTED",
            message="linear planner requires joint-space goal_positions",
        )
    return (
        TiledPlanningSegment(
            kind="joint_position_target",
            duration_s=request.duration_s,
            sample_dt_s=request.sample_dt_s,
            goal_positions=request.goal_positions,
        ),
    )


def _plan_linear_row(
    backend: LinearPlannerBackend,
    *,
    request: TiledPlanningRequest,
    segments: tuple[TiledPlanningSegment, ...],
    row: int,
) -> tuple[np.ndarray, np.ndarray] | TiledPlanningResult:
    """串行拼接一个 env row 的 joint segments，并去掉段边界重复起点。"""

    current = request.current_positions[row].copy()
    time_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    elapsed_s = 0.0
    for index, segment in enumerate(segments):
        if segment.goal_positions is None:
            return TiledPlanningResult.failed(
                request,
                status="UNSUPPORTED",
                message=(
                    "linear planner only supports joint-space segments; "
                    f"segment {index} kind={segment.kind!r} has no goal_positions"
                ),
            )
        result = backend.plan(
            MotionRequest(
                current_q=current,
                goal_q=segment.goal_positions[row],
                duration_s=_segment_duration_s(request, segment),
                sample_dt_s=_segment_sample_dt_s(request, segment),
                avoid_collisions=request.avoid_collisions,
            )
        )
        if not result.success:
            return TiledPlanningResult.failed(
                request,
                status=result.status,
                message=result.diagnostics.message,
            )
        trajectory = result.trajectory
        if not isinstance(trajectory, JointTrajectory):  # pragma: no cover
            raise TypeError("linear planner did not return a JointTrajectory")
        local_times = np.asarray(trajectory.times, dtype=float)
        local_positions = np.asarray(trajectory.positions, dtype=float)
        if position_parts:
            time_parts.append(elapsed_s + local_times[1:])
            position_parts.append(local_positions[1:, :])
        else:
            time_parts.append(elapsed_s + local_times)
            position_parts.append(local_positions)
        elapsed_s += float(local_times[-1])
        current = local_positions[-1].copy()
    return np.concatenate(time_parts), np.concatenate(position_parts, axis=0)


def _segment_duration_s(
    request: TiledPlanningRequest, segment: TiledPlanningSegment
) -> float:
    """返回 segment override duration，省略时继承 request。"""

    return float(
        request.duration_s if segment.duration_s is None else segment.duration_s
    )


def _segment_sample_dt_s(
    request: TiledPlanningRequest, segment: TiledPlanningSegment
) -> float:
    """返回 segment override sample period，省略时继承 request。"""

    return float(
        request.sample_dt_s if segment.sample_dt_s is None else segment.sample_dt_s
    )


__all__ = ["LinearJointPlannerBackend"]
