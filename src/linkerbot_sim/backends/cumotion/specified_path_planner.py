"""specified-path cuMotion pipeline。

specified path 的核心语义是“路径几何由调用方指定”，而不是让 graph planner 或 optimizer 自己
搜索到目标。后端会把指定路径转成 C-space waypoint path，再可选用
``CSpaceTrajectoryGenerator`` 做时间参数化。

``CSpaceWaypointPath``、``TaskSpacePath`` 和 ``CompositePath`` 都通过 cuMotion 官方
PathSpec/conversion API 生成 C-space waypoint path。
"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from linkerbot_sim.backends.cumotion.motion_planner_utils import (
    path_length,
    validate_cspace_width,
)
from linkerbot_sim.backends.cumotion.path_spec_adapter import (
    composite_path_to_joint_path,
    cspace_waypoints_to_joint_path,
    task_space_path_to_joint_path,
)
from linkerbot_sim.backends.cumotion.trajectory_generation import (
    generate_cspace_trajectory,
)
from linkerbot_sim.planning.requests import (
    CSpaceWaypointPath,
    CompositePath,
    SpecifiedPathRequest,
    TaskSpacePath,
)
from linkerbot_sim.planning.results import MotionResult, PlanningDiagnostics


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
    # Task-space 和 composite 路径需要控制 frame；C-space waypoint 路径不读取该 frame，但仍在
    # 入口处统一解析，保证 diagnostics 和请求边界行为一致。
    frame_name = str(request.tcp_frame_name or tcp_frame_name)

    if isinstance(request.path, CSpaceWaypointPath):
        family = "cspace_waypoints"
        # C-space waypoint 路径走官方 CSpacePathSpec -> LinearCSpacePath，而不是直接 vstack。
        # 这样三类 specified-path 都经过 cuMotion 官方 path API，行为更接近真实运行环境。
        joint_path = cspace_waypoints_to_joint_path(context, request, config)
    elif isinstance(request.path, TaskSpacePath):
        family = "task_space_segments"
        # TaskSpacePath 明确使用 TaskSpacePathSpec + convert_task_space_path_spec_to_cspace。
        joint_path = task_space_path_to_joint_path(
            context,
            request,
            config,
            tcp_frame_name=frame_name,
        )
    elif isinstance(request.path, CompositePath):
        family = "composite"
        # CompositePath 由 adapter 拼成 CompositePathSpec，再由 cuMotion 统一转换成 C-space。
        joint_path = composite_path_to_joint_path(
            context,
            request,
            config,
            tcp_frame_name=frame_name,
        )
    else:
        raise ValueError(f"Unsupported specified path type: {type(request.path).__name__}")

    # Task-space conversion 和 C-space path 生成都发生在规划阶段。执行层不会每个 physics step
    # 重新做 IK 或 path conversion，而是只播放这里已经生成好的 cuMotion trajectory。
    trajectory = generate_cspace_trajectory(
        context,
        joint_path,
        config.trajectory_generation,
        duration_s=request.duration_s,
    )
    if trajectory is None:
        raise RuntimeError(
            "specified_path requires at least two waypoints to generate trajectory"
        )
    metrics = {
        "num_waypoints": float(joint_path.shape[0]),
        # specified_path 本身是 collision-unaware path generation；当前实现不读取环境 world。
        # validate_collision_after_generation 只进入 diagnostics，真正后验碰撞检查留给后续实现。
        "num_collision_objects": 0.0,
        "path_length": float(path_length(joint_path)),
    }
    diagnostics = PlanningDiagnostics(
        status="SUCCESS",
        message=(
            f"pipeline=specified_path family={family} path_conversion=official "
            f"collision_check={config.specified_path.validate_collision_after_generation}"
        ),
        metrics=metrics,
    )
    return MotionResult(
        path=joint_path,
        trajectory=trajectory,
        success=True,
        status=diagnostics.status,
        diagnostics=diagnostics,
    )
