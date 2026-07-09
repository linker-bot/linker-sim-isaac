"""cuMotion adapter for tiled asynchronous planning.

本模块位于 backend 层，只负责把已有 cuMotion ``MotionPlanner`` facade 接入 tiled
``TiledPlannerManager``。tiled 核心层不直接依赖 cuMotion，便于 debug runtime、线性
    planner 和真实 Isaac/cumotion 后端保持清晰边界。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from linkerbot_sim.tiled.planner_manager import (
    TiledPlanningRequest,
    TiledPlanningResult,
)
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.types import JointTrajectory


class CuMotionJointPlannerBackend:
    """把已有 cuMotion ``MotionPlanner`` facade 接到 tiled async manager。

    ``planner_factory`` 必须返回当前 worker 独享的 planner/context 组合，避免多个线程共享带
    内部状态的 cuMotion planner 实例。该 adapter 支持 tiled 的关节目标段和
    specified-path/task-space 段；两者都会被重采样成统一 ``(E,T,D)`` 关节轨迹。
    """

    def __init__(
        self,
        planner_factory: Callable[[str], object],
    ) -> None:
        """保存 planner factory。"""

        self._planner_factory = planner_factory

    def plan(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """逐 env 调用独立 cuMotion planner，并重采样到共同时间网格。"""

        try:
            from linkerbot_sim.planning.requests import MotionRequest, SpecifiedPathRequest
        except Exception as exc:  # pragma: no cover - 只有缺依赖环境会进入
            return TiledPlanningResult.failed(
                request,
                status="IMPORT_FAILED",
                message=str(exc),
            )
        if not request.segments:
            if request.goal_positions is None:
                return TiledPlanningResult.failed(
                    request,
                    status="UNSUPPORTED",
                    message="cuMotion tiled planner requires goal_positions or segments",
                )
            segments = (
                _RuntimeSegment(
                    kind="joint_position_target",
                    duration_s=float(request.duration_s),
                    sample_dt_s=float(request.sample_dt_s),
                    goal_positions=request.goal_positions,
                    path=None,
                    tcp_frame_name=None,
                ),
            )
        else:
            segments = tuple(
                _RuntimeSegment(
                    kind=segment.kind,
                    duration_s=(
                        float(request.duration_s)
                        if segment.duration_s is None
                        else float(segment.duration_s)
                    ),
                    sample_dt_s=(
                        float(request.sample_dt_s)
                        if segment.sample_dt_s is None
                        else float(segment.sample_dt_s)
                    ),
                    goal_positions=segment.goal_positions,
                    path=segment.path,
                    tcp_frame_name=segment.tcp_frame_name,
                )
                for segment in request.segments
            )
        times = _segment_sample_times(segments)
        planned_rows: list[np.ndarray] = []
        for row in range(len(request.env_ids)):
            planner = self._planner_factory(request.robot_name)
            env_result = _plan_env_segments(
                planner,
                request=request,
                segments=segments,
                row=row,
                motion_request_type=MotionRequest,
                specified_path_request_type=SpecifiedPathRequest,
            )
            if isinstance(env_result, TiledPlanningResult):
                return env_result
            row_times, row_positions = env_result
            if row_times.shape != times.shape or not np.allclose(row_times, times):
                return TiledPlanningResult.failed(
                    request,
                    status="INVALID_TIMES",
                    message="cuMotion tiled planner produced inconsistent segment times",
                )
            planned_rows.append(row_positions)
        return TiledPlanningResult(
            request_id=request.request_id,
            robot_name=request.robot_name,
            env_ids=request.env_ids,
            success=True,
            status="SUCCESS",
            message="cuMotion tiled trajectory generated",
            times=times,
            positions=np.stack(planned_rows, axis=0),
            joint_names=request.joint_names,
            source=request.source,
            load_on_success=request.load_on_success,
            replace=request.replace,
            trajectory_overlays=request.trajectory_overlays,
        )


class _RuntimeSegment:
    """cuMotion backend 内部使用的规划段。"""

    def __init__(
        self,
        *,
        kind: str,
        duration_s: float,
        sample_dt_s: float,
        goal_positions: np.ndarray | None,
        path: object | None,
        tcp_frame_name: str | None,
    ) -> None:
        """保存单段规划运行时参数，供统一采样时间轴使用。"""

        self.kind = kind
        self.duration_s = duration_s
        self.sample_dt_s = sample_dt_s
        self.goal_positions = goal_positions
        self.path = path
        self.tcp_frame_name = tcp_frame_name


def _segment_sample_times(segments: tuple[_RuntimeSegment, ...]) -> np.ndarray:
    """按多段 request 生成共同采样时间网格。"""

    parts: list[np.ndarray] = []
    elapsed_s = 0.0
    for segment in segments:
        steps = max(1, int(np.ceil(segment.duration_s / segment.sample_dt_s)))
        local = np.linspace(0.0, segment.duration_s, steps + 1)
        if parts:
            parts.append(elapsed_s + local[1:])
        else:
            parts.append(elapsed_s + local)
        elapsed_s += segment.duration_s
    return np.concatenate(parts, axis=0)


def _plan_env_segments(
    planner: object,
    *,
    request: TiledPlanningRequest,
    segments: tuple[_RuntimeSegment, ...],
    row: int,
    motion_request_type: type,
    specified_path_request_type: type,
) -> tuple[np.ndarray, np.ndarray] | TiledPlanningResult:
    """按 env 顺序执行多段 planner，并拼成单行轨迹。"""

    current_q = request.current_positions[row].copy()
    row_times: list[np.ndarray] = []
    row_positions: list[np.ndarray] = []
    elapsed_s = 0.0
    planner_joint_names = tuple(planner.joint_names())
    for index, segment in enumerate(segments):
        local_times = _segment_sample_times((segment,))
        if segment.goal_positions is not None:
            backend_request = motion_request_type(
                current_q=current_q,
                goal_q=segment.goal_positions[row],
                tcp_frame_name=segment.tcp_frame_name,
                duration_s=segment.duration_s,
            )
        elif segment.path is not None:
            backend_request = specified_path_request_type(
                current_q=current_q,
                path=segment.path,
                tcp_frame_name=segment.tcp_frame_name,
                duration_s=segment.duration_s,
            )
        else:
            return TiledPlanningResult.failed(
                request,
                status="UNSUPPORTED",
                message=f"segment {index} kind={segment.kind!r} has no goal or path",
            )
        motion_result = planner.plan(backend_request)
        if not getattr(motion_result, "success", False):
            return TiledPlanningResult.failed(
                request,
                status=str(getattr(motion_result, "status", "FAILED")),
                message=_motion_result_message(motion_result),
            )
        trajectory = _motion_result_to_joint_trajectory(
            motion_result,
            planner_joint_names=planner_joint_names,
            sample_dt_s=segment.sample_dt_s,
        )
        local_positions = _trajectory_positions_at_times(
            trajectory,
            times=local_times,
            target_joint_names=request.joint_names,
        )
        if row_positions:
            row_times.append(elapsed_s + local_times[1:])
            row_positions.append(local_positions[1:, :])
        else:
            row_times.append(elapsed_s + local_times)
            row_positions.append(local_positions)
        elapsed_s += segment.duration_s
        current_q = np.asarray(trajectory.eval(float(local_times[-1])), dtype=float).reshape(-1)
    return np.concatenate(row_times, axis=0), np.concatenate(row_positions, axis=0)


def _motion_result_message(result: object) -> str:
    """从 MotionResult-like 对象提取可读 message。"""

    diagnostics = getattr(result, "diagnostics", None)
    message = getattr(diagnostics, "message", "") if diagnostics is not None else ""
    return str(message or getattr(result, "status", "planning failed"))


def _motion_result_to_joint_trajectory(
    result: object,
    *,
    planner_joint_names: tuple[str, ...],
    sample_dt_s: float,
) -> JointTrajectory:
    """把 MotionResult-like 对象转换成项目 ``JointTrajectory``。"""

    trajectory = getattr(result, "trajectory", None)
    if isinstance(trajectory, JointTrajectory):
        return trajectory
    if trajectory is not None:
        from linkerbot_sim.backends.cumotion.trajectory_sampler import (
            joint_trajectory_from_cumotion,
        )

        return joint_trajectory_from_cumotion(
            trajectory,
            joint_names=planner_joint_names,
            sample_dt=float(sample_dt_s),
        )
    path = getattr(result, "path", None)
    if path is None:
        raise ValueError("successful MotionResult has neither trajectory nor path")
    positions = np.asarray(path, dtype=float)
    if positions.ndim != 2:
        raise ValueError("MotionResult.path must have shape (T,D)")
    times = np.linspace(
        0.0,
        max(float(sample_dt_s), float(sample_dt_s) * (len(positions) - 1)),
        len(positions),
    )
    return joint_trajectory_from_positions(
        times=times,
        positions=positions,
        joint_names=planner_joint_names,
        phase="cumotion_path",
    )


def _trajectory_positions_at_times(
    trajectory: JointTrajectory,
    *,
    times: np.ndarray,
    target_joint_names: tuple[str, ...],
) -> np.ndarray:
    """按目标 joint_names 顺序重采样单条轨迹。"""

    index_by_name = {name: index for index, name in enumerate(trajectory.joint_names)}
    missing = [name for name in target_joint_names if name not in index_by_name]
    if missing:
        raise ValueError(f"trajectory is missing joint_names: {missing}")
    rows = []
    for time_s in np.asarray(times, dtype=float).reshape(-1):
        sample = trajectory.eval(float(time_s))
        rows.append([sample[index_by_name[name]] for name in target_joint_names])
    return np.asarray(rows, dtype=float)


__all__ = ["CuMotionJointPlannerBackend"]
