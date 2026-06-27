"""cuMotion ``TrajectoryOptimizer`` pipeline。

该模块实现 motion planner facade 的默认 pipeline。与 graph search 不同，optimizer 直接输出
cuMotion ``Trajectory``，因此成功结果默认不构造离散 ``path``。调用方如果需要完整 DOF
轨迹，应通过 ``trajectory_adapter`` 采样 optimizer 返回的 trajectory，再由任务层按关节名
回填到完整 articulation。

当前接入范围：

* C-space 终端目标：``MotionRequest.goal_q``。
* TCP 位置目标：``goal_pose.position`` 且不约束姿态。
* TCP 位姿目标：``goal_pose.position + orientation``，姿态为终端完整姿态约束。

goalset、路径直线约束、轴约束等更细的官方能力暂未封装。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipulation_project.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from manipulation_project.backends.cumotion.motion_planner_utils import (
    apply_config_params,
    attr,
    collision_world_for_pipeline,
    status_name,
    validate_cspace_width,
)
from manipulation_project.backends.cumotion.pose_adapter import (
    rotation_from_quat_wxyz,
)
from manipulation_project.planning.requests import MotionRequest
from manipulation_project.planning.results import MotionResult, PlanningDiagnostics


def plan_trajectory_optimization(
    context,
    request: MotionRequest,
    config: MotionPlannerBackendConfig,
    *,
    tcp_frame_name: str,
) -> MotionResult:
    """对目标式 ``MotionRequest`` 执行 trajectory optimization。"""

    request.validate_structure()
    optimizer_config = config.trajectory_optimization
    frame_name = str(request.tcp_frame_name or tcp_frame_name)
    current = np.asarray(request.current_q, dtype=float).reshape(-1)
    validate_cspace_width(context, current, "current_q")

    # optimizer 默认考虑 context 当前环境；关闭时使用空 world view，仅影响本次后端 config。
    collision_world = collision_world_for_pipeline(
        context,
        use_environment_obstacles=optimizer_config.use_environment_obstacles,
    )
    backend_config = _trajectory_optimizer_config(
        context,
        frame_name=frame_name,
        world_view=collision_world.world_view,
        config=config,
    )
    optimizer = context.cumotion.create_trajectory_optimizer(backend_config)
    results = _plan_to_request_target(context, optimizer, current, request)
    return _motion_result(
        results,
        num_collision_objects=len(collision_world.handles),
        frame_name=frame_name,
    )


def _plan_to_request_target(context, optimizer, current: np.ndarray, request):
    """根据目标类型构造 optimizer target 并调用对应 plan 方法。"""

    optimizer_type = context.cumotion.TrajectoryOptimizer
    if request.goal_q is not None:
        goal = np.asarray(request.goal_q, dtype=float).reshape(-1)
        validate_cspace_width(context, goal, "goal_q")
        target = _construct_cspace_target(optimizer_type, goal)
        return optimizer.plan_to_cspace_target(current, target)
    if request.goal_pose is None:
        raise ValueError("Exactly one of goal_q or goal_pose must be provided")
    # task-space target 由 translation constraint 和 orientation constraint 组成。
    # 无姿态目标时显式使用 none()，表示只约束 TCP 终点位置。
    translation_constraint = optimizer_type.TranslationConstraint.target(
        np.asarray(request.goal_pose.position, dtype=float).reshape(3)
    )
    if request.goal_pose.orientation is None:
        orientation_constraint = optimizer_type.OrientationConstraint.none()
    else:
        orientation_constraint = optimizer_type.OrientationConstraint.terminal_target(
            rotation_from_quat_wxyz(context.cumotion, request.goal_pose.orientation)
        )
    target = _construct_task_space_target(
        optimizer_type,
        translation_constraint,
        orientation_constraint,
    )
    return optimizer.plan_to_task_space_target(current, target)


def _construct_cspace_target(optimizer_type, goal: np.ndarray):
    """构造 ``TrajectoryOptimizer.CSpaceTarget``。

    cuMotion 的 pybind 构造签名在官方 HTML 文档中不总是完整展开，不同版本也可能存在轻微
    差异。这里按“终端目标 + 无路径平移约束 + 无路径姿态约束”的语义尝试常见构造形态。
    如果后续真实环境暴露了更明确的签名，只需要集中调整这个函数。
    """

    cspace_target_type = optimizer_type.CSpaceTarget
    translation_constraint = cspace_target_type.TranslationPathConstraint.none()
    orientation_constraint = cspace_target_type.OrientationPathConstraint.none()
    try:
        return cspace_target_type(goal, translation_constraint, orientation_constraint)
    except TypeError:
        try:
            return cspace_target_type(goal)
        except TypeError:
            return cspace_target_type(cspace_position_terminal_target=goal)


def _construct_task_space_target(
    optimizer_type,
    translation_constraint,
    orientation_constraint,
):
    """构造 ``TrajectoryOptimizer.TaskSpaceTarget``。

    与 C-space target 一样，这里兼容位置参数和关键字参数两种 pybind 暴露形态，避免把版本差异
    扩散到主规划逻辑。
    """

    task_space_target_type = optimizer_type.TaskSpaceTarget
    try:
        return task_space_target_type(translation_constraint, orientation_constraint)
    except TypeError:
        return task_space_target_type(
            translation_constraint=translation_constraint,
            orientation_constraint=orientation_constraint,
        )


def _motion_result(results, *, num_collision_objects: int, frame_name: str):
    """把 optimizer results 转成 ``MotionResult``。

    optimizer 成功时 ``trajectory`` 是主产物；``path`` 保持为 ``None``，避免把采样后的
    诊断路径误认为 optimizer 原生路径输出。
    """

    status_value = attr(results, "status", default=None)
    status = status_name(status_value)
    success = status == "SUCCESS"
    trajectory = attr(results, "trajectory", default=None) if success else None
    target_index = attr(results, "target_index", default=-1)
    diagnostics = PlanningDiagnostics(
        status=status or ("SUCCESS" if success else "FAILED"),
        message=f"pipeline=trajectory_optimization frame={frame_name}",
        metrics={
            "num_waypoints": 0.0,
            "num_collision_objects": float(num_collision_objects),
            "target_index": float(target_index if target_index is not None else -1),
        },
    )
    return MotionResult(
        path=None,
        trajectory=trajectory,
        success=success,
        status=diagnostics.status,
        diagnostics=diagnostics,
    )


def _trajectory_optimizer_config(
    context,
    *,
    frame_name: str,
    world_view,
    config: MotionPlannerBackendConfig,
):
    """创建 ``TrajectoryOptimizerConfig`` 并应用参数覆盖。"""

    optimizer_config = config.trajectory_optimization
    if optimizer_config.config_path:
        backend_config = context.cumotion.create_trajectory_optimizer_config_from_file(
            Path(optimizer_config.config_path),
            context.robot_description,
            frame_name,
            world_view,
        )
    else:
        backend_config = context.cumotion.create_default_trajectory_optimizer_config(
            context.robot_description,
            frame_name,
            world_view,
        )
    apply_config_params(
        backend_config,
        optimizer_config.params,
        context.cumotion.TrajectoryOptimizerConfig.ParamValue,
    )
    return backend_config
