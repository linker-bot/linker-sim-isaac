"""tiled 异步规划与 cuRobo 单条/批量求解能力的组合层。

tiled request 使用 controller command-space，并携带 request/env identity 与回放元数据；
cuRobo batch core 只处理按行排列的数组问题。本模块是两种语义唯一的组合位置：把同构
joint-space requests 拼成纯数组 batch，按 row slices 恢复 tiled 结果；task-space path 则保留
逐 request、逐 env 的 ``CuroboMotionPlanner`` 路径。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from linkerbot_sim.backends.curobo.batch.joint_planner import (
    CuroboBatchJointPlanner,
)
from linkerbot_sim.backends.curobo.batch.types import CuroboBatchJointProblem
from linkerbot_sim.backends.curobo.joint_mapping import CuroboJointMapping
from linkerbot_sim.planning.backend import PlannerBackend
from linkerbot_sim.tiled.planning.batching import (
    PlanningBatchLayout,
    normalize_joint_batch_mode,
    planning_batch_layout,
)
from linkerbot_sim.tiled.planning.types import (
    TiledPlanningRequest,
    TiledPlanningResult,
)
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.retiming import (
    retime_joint_trajectory,
    trajectory_sample_times,
)
from linkerbot_sim.trajectories.types import JointTrajectory


class TiledCuroboPlanningBackend:
    """把 cuRobo 单条与 batch planner 接到 tiled async manager。

    ``planner_factory`` 必须为每次 worker 调用返回可独占使用的 planner facade。真实 cuRobo
    planner 内部可能持有 CUDA graph、缓存和临时 tensor；不要在多个 planner worker 线程中
    共享同一个实例。

    joint-space 目标会优先走 cuRobo ``BatchMotionPlanner.plan_cspace``，让多个 env 在一次
    GPU 调用中并行规划；task-space linear path 仍使用单 env facade，因为它需要先按路径几何
    离散并通过 sequential IK 做 waypoint conversion。
    """

    def __init__(
        self,
        planner_factory: Callable[[str], PlannerBackend],
        *,
        joint_batch_mode: str = "auto",
    ) -> None:
        """保存 planner factory。"""

        self._planner_factory = planner_factory
        self._joint_batch_mode = normalize_joint_batch_mode(joint_batch_mode)

    def plan(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """规划一条 tiled request，并返回 ``(E,T,D)`` command-space 轨迹。"""

        planner = self._planner_factory(request.robot_name)
        _require_planner_backend(planner)
        try:
            return self._plan_with_planner(planner, request=request)
        finally:
            _close_planner(planner)

    def plan_many(
        self,
        requests: Sequence[TiledPlanningRequest],
    ) -> tuple[TiledPlanningResult, ...]:
        """按 FIFO 顺序批量规划同构 joint-space requests。

        ``planning_batch_layout`` 只记录每条 request 占用的数组行区间；这里直接构造
        ``CuroboBatchJointProblem``，不会创建替代 request 或虚构 env ID。task-space 或不兼容
        请求保持逐 request fallback。
        """

        batch = tuple(requests)
        if not batch:
            return ()
        layout = planning_batch_layout(batch)
        if layout is None:
            return tuple(self.plan(request) for request in batch)
        planner = self._planner_factory(batch[0].robot_name)
        _require_planner_backend(planner)
        try:
            if self._joint_batch_mode != "per_env":
                batch_results = _plan_joint_batch_if_supported(
                    planner,
                    layout=layout,
                )
                if batch_results is not None:
                    return batch_results
                if self._joint_batch_mode == "batch_only":
                    return _batch_unavailable_results(layout)
            return tuple(
                self._plan_per_env_with_planner(planner, request=request)
                for request in layout.requests
            )
        finally:
            _close_planner(planner)

    def _plan_with_planner(
        self,
        planner: object,
        *,
        request: TiledPlanningRequest,
    ) -> TiledPlanningResult:
        """在一个 worker 独占的 planner 生命周期内执行单条请求。"""

        layout = planning_batch_layout((request,))
        if self._joint_batch_mode != "per_env" and layout is not None:
            batch_results = _plan_joint_batch_if_supported(planner, layout=layout)
            if batch_results is not None:
                return batch_results[0]
        if self._joint_batch_mode == "batch_only":
            return _batch_unavailable_result(request)
        return self._plan_per_env_with_planner(planner, request=request)

    @staticmethod
    def _plan_per_env_with_planner(
        planner: object,
        *,
        request: TiledPlanningRequest,
    ) -> TiledPlanningResult:
        """复用既有 planner，逐 env 执行 joint 或 task-space segments。"""

        try:
            from linkerbot_sim.planning.requests import (
                LinearPosePathRequest,
                MotionRequest,
            )
        except Exception as exc:  # pragma: no cover - 防御极端 import 环境
            return TiledPlanningResult.failed(
                request,
                status="IMPORT_FAILED",
                message=str(exc),
            )
        segments = _runtime_segments_from_request(request)
        if isinstance(segments, TiledPlanningResult):
            return segments
        times = _segment_sample_times(segments)
        planned_rows: list[np.ndarray] = []
        for row in range(len(request.env_ids)):
            env_result = _plan_env_segments(
                planner,
                request=request,
                segments=segments,
                row=row,
                motion_request_type=MotionRequest,
                linear_pose_path_request_type=LinearPosePathRequest,
            )
            if isinstance(env_result, TiledPlanningResult):
                return env_result
            row_times, row_positions = env_result
            if row_times.shape != times.shape or not np.allclose(row_times, times):
                return TiledPlanningResult.failed(
                    request,
                    status="INVALID_TIMES",
                    message="cuRobo tiled planner produced inconsistent segment times",
                )
            planned_rows.append(row_positions)
        return TiledPlanningResult(
            request_id=request.request_id,
            robot_name=request.robot_name,
            env_ids=request.env_ids,
            success=True,
            status="SUCCESS",
            message="cuRobo tiled trajectory generated",
            times=times,
            positions=np.stack(planned_rows, axis=0),
            joint_names=request.joint_names,
            source=request.source,
            load_on_success=request.load_on_success,
            replace=request.replace,
        )


def _close_planner(planner: object) -> None:
    """关闭 planner facade 或其 context，避免 tiled worker 泄漏 cuRobo CUDA 资源。"""

    close = getattr(planner, "close", None)
    if callable(close):
        close()
        return
    context = getattr(planner, "context", None)
    context_close = getattr(context, "close", None)
    if callable(context_close):
        context_close()


def _require_planner_backend(planner: object) -> None:
    """校验 factory 返回对象实现 shared ``PlannerBackend`` runtime protocol。"""

    if not isinstance(planner, PlannerBackend):
        raise TypeError("tiled planner factory must return a PlannerBackend")


def _batch_unavailable_result(
    request: TiledPlanningRequest,
) -> TiledPlanningResult:
    """构造 ``joint_batch_mode=batch_only`` 无可用 batch core 时的结果。"""

    return TiledPlanningResult.failed(
        request,
        status="BATCH_UNAVAILABLE",
        message=(
            "cuRobo tiled planner joint_batch_mode=batch_only requires a "
            "batch-capable joint-space request"
        ),
    )


def _batch_unavailable_results(
    layout: PlanningBatchLayout,
) -> tuple[TiledPlanningResult, ...]:
    """为 layout 中每条原始请求构造 batch 不可用结果。"""

    return tuple(_batch_unavailable_result(request) for request in layout.requests)


def _failed_batch_results(
    layout: PlanningBatchLayout,
    *,
    status: str,
    message: str,
) -> tuple[TiledPlanningResult, ...]:
    """按当前 all-or-nothing 语义把 batch 失败传播到每条原始请求。"""

    return tuple(
        TiledPlanningResult.failed(request, status=status, message=message)
        for request in layout.requests
    )


def _plan_joint_batch_if_supported(
    planner: object,
    *,
    layout: PlanningBatchLayout,
) -> tuple[TiledPlanningResult, ...] | None:
    """直接把 tiled request rows 交给纯数组 cuRobo batch core。

    返回 ``None`` 仅表示 planner 没有 batch 能力，调用方可以按配置回落到逐 env planner。
    求解失败仍返回与输入 request 数量相同的 tiled results。
    """

    context = getattr(planner, "context", None)
    batch_planner = getattr(context, "batch_motion_planner", None)
    if context is None or batch_planner is None:
        return None

    segments_by_request: list[tuple[_RuntimeSegment, ...]] = []
    for request in layout.requests:
        segments = _runtime_segments_from_request(request)
        if isinstance(segments, TiledPlanningResult):
            return _failed_batch_results(
                layout,
                status=segments.status,
                message=segments.message,
            )
        if any(segment.goal_positions is None for segment in segments):
            return None
        segments_by_request.append(segments)

    template_segments = segments_by_request[0]
    expected_times = _segment_sample_times(template_segments)
    first_request = layout.requests[0]
    core = CuroboBatchJointPlanner(
        context,
        batch_planner=batch_planner,
        planner_joint_names=tuple(str(name) for name in planner.joint_names()),
    )
    current_positions = np.vstack(
        [request.current_positions for request in layout.requests]
    )
    time_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    elapsed_s = 0.0
    for segment_index, template in enumerate(template_segments):
        goals = [
            segments[segment_index].goal_positions for segments in segments_by_request
        ]
        if any(goal is None for goal in goals):  # pragma: no cover - 已由上方校验
            return None
        goal_positions = np.vstack([goal for goal in goals if goal is not None])
        core_result = core.plan(
            CuroboBatchJointProblem(
                current_positions=current_positions,
                goal_positions=goal_positions,
                command_joint_names=first_request.joint_names,
                duration_s=template.duration_s,
                sample_dt_s=template.sample_dt_s,
                avoid_collisions=first_request.avoid_collisions,
            )
        )
        if not core_result.all_succeeded:
            status = next(
                (
                    row_status
                    for ok, row_status in zip(
                        core_result.success,
                        core_result.status,
                        strict=True,
                    )
                    if not ok
                ),
                "FAILED",
            )
            return _failed_batch_results(
                layout,
                status=status,
                message=core_result.message,
            )
        if position_parts:
            time_parts.append(elapsed_s + core_result.times[1:])
            position_parts.append(core_result.positions[:, 1:, :])
        else:
            time_parts.append(elapsed_s + core_result.times)
            position_parts.append(core_result.positions)
        elapsed_s += template.duration_s
        current_positions = core_result.positions[:, -1, :].copy()

    produced_times = np.concatenate(time_parts, axis=0)
    if produced_times.shape != expected_times.shape or not np.allclose(
        produced_times, expected_times
    ):
        return _failed_batch_results(
            layout,
            status="INVALID_TIMES",
            message="cuRobo batch tiled planner produced inconsistent segment times",
        )
    return _split_batch_trajectory(
        layout,
        times=produced_times,
        positions=np.concatenate(position_parts, axis=1),
        message="cuRobo BatchMotionPlanner tiled trajectory generated",
    )


def _split_batch_trajectory(
    layout: PlanningBatchLayout,
    *,
    times: np.ndarray,
    positions: np.ndarray,
    message: str,
) -> tuple[TiledPlanningResult, ...]:
    """按 layout 的连续 row slices 恢复真实 request/env identity 与回放元数据。"""

    if positions.shape[0] != layout.problem_count:
        return _failed_batch_results(
            layout,
            status="INVALID_BATCH_RESULT",
            message=(
                "cuRobo batch result row count "
                f"{positions.shape[0]} != {layout.problem_count}"
            ),
        )
    return tuple(
        TiledPlanningResult(
            request_id=request.request_id,
            robot_name=request.robot_name,
            env_ids=request.env_ids,
            success=True,
            status="SUCCESS",
            message=message,
            times=times,
            positions=positions[row_slice].copy(),
            joint_names=request.joint_names,
            source=request.source,
            load_on_success=request.load_on_success,
            replace=request.replace,
        )
        for request, row_slice in zip(
            layout.requests,
            layout.row_slices,
            strict=True,
        )
    )


class _RuntimeSegment:
    """cuRobo tiled backend 内部使用的单段规划参数。"""

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
        """保存已经从 request 继承默认值后的 segment 参数。"""

        self.kind = kind
        self.duration_s = duration_s
        self.sample_dt_s = sample_dt_s
        self.goal_positions = goal_positions
        self.path = path
        self.tcp_frame_name = tcp_frame_name


def _runtime_segments_from_request(
    request: TiledPlanningRequest,
) -> tuple[_RuntimeSegment, ...] | TiledPlanningResult:
    """把 tiled request 规整为统一 segment 列表。"""

    if not request.segments:
        if request.goal_positions is None:
            return TiledPlanningResult.failed(
                request,
                status="UNSUPPORTED",
                message="cuRobo tiled planner requires goal_positions or segments",
            )
        return (
            _RuntimeSegment(
                kind="joint_position_target",
                duration_s=float(request.duration_s),
                sample_dt_s=float(request.sample_dt_s),
                goal_positions=request.goal_positions,
                path=None,
                tcp_frame_name=None,
            ),
        )
    return tuple(
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


def _segment_sample_times(segments: tuple[_RuntimeSegment, ...]) -> np.ndarray:
    """按多段 request 生成共同采样时间网格。"""

    parts: list[np.ndarray] = []
    elapsed_s = 0.0
    for segment in segments:
        local = trajectory_sample_times(
            duration_s=segment.duration_s,
            sample_dt_s=segment.sample_dt_s,
            include_start=True,
        )
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
    linear_pose_path_request_type: type,
) -> tuple[np.ndarray, np.ndarray] | TiledPlanningResult:
    """按 env 顺序执行多段 planner，并拼成一条 command-space 轨迹。"""

    current_command = request.current_positions[row].copy()
    planner_joint_names = tuple(str(name) for name in planner.joint_names())
    mapping = _command_to_curobo_mapping(
        planner_joint_names=planner_joint_names,
        command_joint_names=request.joint_names,
    )
    row_times: list[np.ndarray] = []
    row_positions: list[np.ndarray] = []
    elapsed_s = 0.0
    for index, segment in enumerate(segments):
        local_times = _segment_sample_times((segment,))
        current_cspace = mapping.command_to_cspace(current_command.reshape(1, -1))[0]
        base_command = _base_command_positions_for_segment(
            current_command=current_command,
            segment=segment,
            row=row,
            local_times=local_times,
        )
        if segment.goal_positions is not None:
            goal_command = np.asarray(segment.goal_positions[row], dtype=float)
            goal_cspace = mapping.command_to_cspace(goal_command.reshape(1, -1))[0]
            backend_request = motion_request_type(
                current_q=current_cspace,
                goal_q=goal_cspace,
                tcp_frame_name=segment.tcp_frame_name,
                duration_s=segment.duration_s,
                sample_dt_s=segment.sample_dt_s,
                avoid_collisions=request.avoid_collisions,
            )
        elif segment.path is not None:
            backend_request = linear_pose_path_request_type(
                current_q=current_cspace,
                path=segment.path,
                tcp_frame_name=segment.tcp_frame_name,
                duration_s=segment.duration_s,
                sample_dt_s=segment.sample_dt_s,
                avoid_collisions=request.avoid_collisions,
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
        retimed_trajectory = retime_joint_trajectory(
            trajectory,
            duration_s=segment.duration_s,
            sample_dt_s=segment.sample_dt_s,
            start_position=current_cspace,
            include_start=True,
        )
        if retimed_trajectory.times.shape != local_times.shape or not np.allclose(
            retimed_trajectory.times, local_times
        ):
            return TiledPlanningResult.failed(
                request,
                status="INVALID_TIMES",
                message="cuRobo tiled planner produced a non-canonical time grid",
            )
        cspace_positions = _trajectory_positions_in_joint_order(
            retimed_trajectory,
            target_joint_names=planner_joint_names,
        )
        command_positions = mapping.cspace_to_command(
            cspace_positions,
            base_command_positions=base_command,
        )
        if row_positions:
            row_times.append(elapsed_s + local_times[1:])
            row_positions.append(command_positions[1:, :])
        else:
            row_times.append(elapsed_s + local_times)
            row_positions.append(command_positions)
        elapsed_s += segment.duration_s
        current_command = command_positions[-1].copy()
    return np.concatenate(row_times, axis=0), np.concatenate(row_positions, axis=0)


def _command_to_curobo_mapping(
    *,
    planner_joint_names: tuple[str, ...],
    command_joint_names: tuple[str, ...],
) -> CuroboJointMapping:
    """创建 command-space 到 cuRobo C-space 的名称映射。"""

    try:
        return CuroboJointMapping.from_joint_names(
            cspace_joint_names=planner_joint_names,
            command_joint_names=command_joint_names,
        )
    except ValueError as exc:
        raise ValueError(
            "tiled command joint_names do not contain all cuRobo planner joints"
        ) from exc


def _base_command_positions_for_segment(
    *,
    current_command: np.ndarray,
    segment: _RuntimeSegment,
    row: int,
    local_times: np.ndarray,
) -> np.ndarray:
    """生成回填 cuRobo C-space 结果时使用的 command-space 基底轨迹。

    对关节目标段，非 C-space 列按 current->goal 线性插值，保证手部等未规划 DOF 仍能跟随
    tiled request；对 task-space/path 段，未规划 DOF 保持当前值。
    """

    current = np.asarray(current_command, dtype=float).reshape(1, -1)
    if segment.goal_positions is None:
        return np.repeat(current, local_times.size, axis=0)
    goal = np.asarray(segment.goal_positions[row], dtype=float).reshape(1, -1)
    alpha = (np.asarray(local_times, dtype=float) / segment.duration_s).reshape(-1, 1)
    return current + (goal - current) * alpha


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
    path = getattr(result, "path", None)
    if path is None:
        raise ValueError(
            "successful cuRobo MotionResult has neither trajectory nor path"
        )
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
        phase="curobo_path",
    )


def _trajectory_positions_in_joint_order(
    trajectory: JointTrajectory,
    *,
    target_joint_names: tuple[str, ...],
) -> np.ndarray:
    """按目标 joint_names 顺序读取已经完成重定时的轨迹位置。"""

    index_by_name = {name: index for index, name in enumerate(trajectory.joint_names)}
    missing = [name for name in target_joint_names if name not in index_by_name]
    if missing:
        raise ValueError(f"trajectory is missing joint_names: {missing}")
    indices = [index_by_name[name] for name in target_joint_names]
    return trajectory.positions[:, indices]


__all__ = ["TiledCuroboPlanningBackend"]
