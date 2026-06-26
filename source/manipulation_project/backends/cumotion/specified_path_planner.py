"""specified-path cuMotion pipeline。

specified path 的核心语义是“路径几何由调用方指定”，而不是让 graph planner 或 optimizer 自己
搜索到目标。本模块支持 ``CSpaceWaypointPath``：调用方直接给出一组 C-space waypoint，
后端可选用 ``CSpaceTrajectoryGenerator`` 做时间参数化。

``TaskSpacePath`` 和 ``CompositePath`` 在本模块中作为未实现的请求类型处理，并明确抛出
``NotImplementedError``。
"""

from __future__ import annotations

import numpy as np

from manipulation_project.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from manipulation_project.backends.cumotion.motion_planner_utils import (
    path_length,
    validate_cspace_width,
)
from manipulation_project.backends.cumotion.trajectory_generation import (
    generate_cspace_trajectory,
)
from manipulation_project.planning.requests import (
    CSpaceWaypointPath,
    CompositePath,
    SpecifiedPathRequest,
    TaskSpacePath,
)
from manipulation_project.planning.results import MotionResult, PlanningDiagnostics


def plan_specified_path(
    context,
    request: SpecifiedPathRequest,
    config: MotionPlannerBackendConfig,
    *,
    tcp_frame_name: str,
) -> MotionResult:
    """执行调用方指定路径 pipeline。

    指定路径 pipeline 不读取环境 obstacle，也不做避障搜索；调用方需要保证路径几何安全，
    或在其它地方做后验碰撞检查。
    """

    request.validate_structure()
    current = np.asarray(request.current_q, dtype=float).reshape(-1)
    validate_cspace_width(context, current, "current_q")
    # C-space waypoint 路径不使用 TCP frame；这里解析 frame 只是保持请求边界一致。
    _frame_name = str(request.tcp_frame_name or tcp_frame_name)

    if isinstance(request.path, CSpaceWaypointPath):
        joint_path = _cspace_waypoint_path(context, request.path)
    elif isinstance(request.path, TaskSpacePath):
        # 不静默 fallback 到逐点 IK，避免调用方以为自己使用了官方 PathSpec conversion。
        raise NotImplementedError(
            "specified_path.task_space_segments is not implemented; "
            "use CSpaceWaypointPath or tcp_line helper"
        )
    elif isinstance(request.path, CompositePath):
        raise NotImplementedError(
            "specified_path.composite is not implemented"
        )
    else:
        raise ValueError(f"Unsupported specified path type: {type(request.path).__name__}")

    trajectory = generate_cspace_trajectory(
        context,
        joint_path,
        config.trajectory_generation,
        duration_s=request.duration_s,
    )
    metrics = {
        "num_waypoints": float(joint_path.shape[0]),
        "num_collision_objects": 0.0,
        "path_length": float(path_length(joint_path)),
    }
    diagnostics = PlanningDiagnostics(
        status="SUCCESS",
        message=(
            "pipeline=specified_path family=cspace_waypoints "
            f"collision_check={config.specified_path.validate_collision_after_generation}"
        ),
        metrics=metrics,
    )
    return MotionResult(
        joint_path=joint_path,
        trajectory=trajectory,
        success=True,
        status=diagnostics.status,
        diagnostics=diagnostics,
    )


def _cspace_waypoint_path(context, path: CSpaceWaypointPath) -> np.ndarray:
    """校验并堆叠调用方指定的 C-space waypoints。"""

    waypoints = [
        np.asarray(waypoint, dtype=float).reshape(-1) for waypoint in path.waypoints
    ]
    if len(waypoints) < 2:
        raise ValueError("CSpaceWaypointPath requires at least 2 waypoints")
    for index, waypoint in enumerate(waypoints):
        validate_cspace_width(context, waypoint, f"path.waypoints[{index}]")
    return np.vstack([waypoint.reshape(1, -1) for waypoint in waypoints])
