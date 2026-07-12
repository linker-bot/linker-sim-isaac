"""把 cuRobo MotionGen 适配为 backend-neutral contract 的 motion planner facade。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.backends.curobo.collision_capability import (
    collision_capability_message,
    context_supports_collision_queries,
)
from linkerbot_sim.backends.curobo.linear_pose_path import plan_linear_pose_path
from linkerbot_sim.backends.curobo.tool_pose import (
    goal_tool_pose_from_single_tcp_target,
    update_active_tool_pose_criteria,
)
from linkerbot_sim.backends.curobo.trajectory_adapter import (
    joint_trajectory_from_curobo,
)
from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    MotionRequest,
)
from linkerbot_sim.planning.results import MotionResult, PlanningDiagnostics
from linkerbot_sim.trajectories.retiming import retime_joint_trajectory


class CuroboMotionPlanner:
    """把项目 ``MotionRequest`` 转成 cuRobo ``MotionPlanner`` 调用。"""

    backend_name = "curobo"

    def __init__(self, context, *, tcp_frame_name: str | None = None) -> None:
        """保存 context 和默认 TCP frame。"""

        self.context = context
        # tiled joint-space batch planning 只需要 facade 的 context 和 joint_names。
        # 单问题 MotionPlanner warmup 显存开销较大，因此延迟到真正执行 ``plan()`` 时再创建。
        self._planner = None
        self.tcp_frame_name = str(tcp_frame_name or context.default_tcp_frame)

    @property
    def planner(self):
        """按需取得 context 的单问题 MotionPlanner。"""

        if self._planner is None:
            self._planner = self.context.motion_planner
        return self._planner

    def joint_names(self) -> list[str]:
        """返回 planner 使用的 C-space 关节名。"""

        return self.context.joint_names()

    def close(self) -> None:
        """释放底层 ``CuroboContext`` 持有的 CUDA graph / solver 资源。"""

        close = getattr(self.context, "close", None)
        if callable(close):
            close()
        self._planner = None

    def plan(
        self,
        request: MotionRequest | LinearPosePathRequest,
    ) -> MotionResult:
        """执行一次 cuRobo 规划。

        cuRobo 新接口只公开目标式 ``MotionRequest`` 和线性位姿路径
        ``LinearPosePathRequest``。
        """

        if isinstance(request, LinearPosePathRequest):
            return plan_linear_pose_path(
                self.context,
                request,
                tcp_frame_name=self.tcp_frame_name,
            )
        request.validate_structure()
        if request.avoid_collisions and not context_supports_collision_queries(
            self.context,
            consumer="planner",
        ):
            return _collision_unsupported_motion_result(self.context)
        current_state = self.context.joint_state_from_positions(
            np.asarray(request.current_q, dtype=float).reshape(1, -1)
        )
        if request.goal_q is not None:
            goal_state = self.context.joint_state_from_positions(
                np.asarray(request.goal_q, dtype=float).reshape(1, -1)
            )
            result = self.planner.plan_cspace(goal_state, current_state)
        else:
            assert request.goal_pose is not None
            frame_name = str(request.tcp_frame_name or self.tcp_frame_name)
            current_q = np.asarray(request.current_q, dtype=float).reshape(1, -1)
            goal = goal_tool_pose_from_single_tcp_target(
                self.context,
                tcp_frame_name=frame_name,
                target_position=request.goal_pose.position,
                target_orientation=request.goal_pose.orientation,
                seed=current_q,
            )
            update_active_tool_pose_criteria(
                self.context,
                self.planner,
                active_tool_frame=frame_name,
                orientation_free=request.goal_pose.orientation is None,
            )
            result = self.planner.plan_pose(goal, current_state)
        return _motion_result_from_curobo(
            result,
            joint_names=tuple(self.joint_names()),
            duration_s=request.duration_s,
            sample_dt_s=request.sample_dt_s,
            start_q=np.asarray(request.current_q, dtype=float).reshape(-1),
        )


def _collision_unsupported_motion_result(context=None) -> MotionResult:
    """构造缺少完整 cuRobo 碰撞模型时的规划失败结果。"""

    message = (
        "cuRobo collision-aware motion planning cannot satisfy "
        "avoid_collisions=True: "
        + collision_capability_message(context, consumer="planner")
    )
    return MotionResult(
        path=None,
        trajectory=None,
        success=False,
        status="COLLISION_UNSUPPORTED",
        diagnostics=PlanningDiagnostics(
            status="COLLISION_UNSUPPORTED",
            message=message,
        ),
    )


def _motion_result_from_curobo(
    result,
    *,
    joint_names: tuple[str, ...],
    duration_s: float | None = None,
    sample_dt_s: float | None = None,
    start_q: np.ndarray | None = None,
) -> MotionResult:
    """把 cuRobo result-like 对象转换成项目 ``MotionResult``。"""

    if result is None:
        return MotionResult(
            path=None,
            trajectory=None,
            success=False,
            status="NO_RESULT",
            diagnostics=PlanningDiagnostics(
                status="NO_RESULT",
                message="cuRobo planner returned None",
            ),
        )
    success = _result_success(result)
    status = "SUCCESS" if success else "FAILED"
    trajectory = None
    path = None
    if success:
        trajectory = joint_trajectory_from_curobo(
            result,
            joint_names=joint_names,
            sample_dt=sample_dt_s,
        )
        trajectory = retime_joint_trajectory(
            trajectory,
            duration_s=duration_s,
            sample_dt_s=sample_dt_s,
            start_position=start_q,
            phase="curobo_motion_plan",
        )
        path = trajectory.positions
    metrics = {"total_time": float(getattr(result, "total_time", 0.0) or 0.0)}
    if path is not None:
        metrics.update(_path_metrics(path))
    return MotionResult(
        path=path,
        trajectory=trajectory,
        success=success,
        status=status,
        diagnostics=PlanningDiagnostics(
            status=status,
            message=str(getattr(result, "status", status)),
            metrics=metrics,
        ),
    )


def _result_success(result) -> bool:
    """读取 cuRobo result 的 success。"""

    value = getattr(result, "success", False)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=bool)
    return bool(array.any())


def _path_metrics(path: np.ndarray) -> dict[str, float]:
    """从 trajectory matrix 计算可序列化的 joint-space path metrics。"""

    positions = np.asarray(path, dtype=float)
    if positions.ndim != 2:
        return {}
    if positions.shape[0] <= 1:
        path_length = 0.0
    else:
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    return {
        "num_waypoints": float(positions.shape[0]),
        "trajectory_samples": float(positions.shape[0]),
        "path_length": path_length,
    }
