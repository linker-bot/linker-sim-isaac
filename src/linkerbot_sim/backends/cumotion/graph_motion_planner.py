"""graph-based cuMotion ``MotionPlanner`` pipeline。

用 cuMotion ``MotionPlanner`` 从当前 C-space 搜索到目标 C-space / TCP 位置 / TCP 位姿，
得到 sparse ``path`` 或 ``interpolated_path``，再按 ``trajectory_generation``
配置强制生成时间参数化 ``Trajectory``。

graph search 的默认碰撞语义是使用 ``CuMotionContext`` 当前同步的环境 obstacle；如需调试
忽略环境障碍，应通过 ``GraphSearchConfig.use_environment_obstacles=False`` 显式选择。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from linkerbot_sim.backends.cumotion.motion_planner_utils import (
    apply_config_params,
    attr,
    collision_world_for_pipeline,
    path_length,
    result_path_samples,
    stack_path,
    validate_cspace_width,
)
from linkerbot_sim.backends.cumotion.pose_adapter import (
    pose_from_position_quat_wxyz,
)
from linkerbot_sim.backends.cumotion.trajectory_generation import (
    generate_cspace_trajectory,
)
from linkerbot_sim.planning.requests import MotionRequest
from linkerbot_sim.planning.results import MotionResult, PlanningDiagnostics


_PlannerTargetType = Literal["cspace", "translation", "pose"]


def plan_graph_search(
    context,
    request: MotionRequest,
    config: MotionPlannerBackendConfig,
    *,
    tcp_frame_name: str,
) -> MotionResult:
    """执行 graph planner，并把结果转换成统一 ``MotionResult``。"""

    # 验证 request 是否满足指定结构
    request.validate_structure()

    graph_config = config.graph_search
    frame_name = str(request.tcp_frame_name or tcp_frame_name)
    current = np.asarray(request.current_q, dtype=float).reshape(-1)

    # 验证 request 的 current_q 是否满足机械臂关节个数
    validate_cspace_width(context, current, "current_q")

    # graph_search 默认使用 context 当前环境；关闭时只切到空 world view，不会修改 context
    # 中已经同步好的真实环境。
    collision_world = collision_world_for_pipeline(
        context,
        use_environment_obstacles=graph_config.use_environment_obstacles,
    )
    planner_config = _motion_planner_config(
        context,
        frame_name=frame_name,
        world_view=collision_world.world_view,
        config=config,
    )
    planner = context.cumotion.create_motion_planner(planner_config)
    results, target_type = _plan_to_request_target(
        context,
        planner,
        current,
        request,
        generate_interpolated_path=graph_config.generate_interpolated_path,
    )
    return _motion_result(
        context,
        results,
        config,
        target_type=target_type,
        frame_name=frame_name,
        num_collision_objects=len(collision_world.handles),
        duration_s=request.duration_s,
    )


def _plan_to_request_target(
    context,
    planner,
    current: np.ndarray,
    request: MotionRequest,
    *,
    generate_interpolated_path: bool,
) -> tuple[object, _PlannerTargetType]:
    """根据 ``MotionRequest`` 的目标类型选择 graph planner API。"""

    # 有 target angles 时先考虑使用 target angles
    if request.goal_q is not None:
        goal = np.asarray(request.goal_q, dtype=float).reshape(-1)
        validate_cspace_width(context, goal, "goal_q")
        return (
            planner.plan_to_cspace_target(
                current, goal, bool(generate_interpolated_path)
            ),
            "cspace",
        )

    # 没有 target angles 也没有 target pose 则报错
    if request.goal_pose is None:
        raise ValueError("Exactly one of goal_q or goal_pose must be provided")

    # 没有 target angles, 有 position, 没有 orientation
    if request.goal_pose.orientation is None:
        translation = np.asarray(request.goal_pose.position, dtype=float).reshape(3)
        return (
            planner.plan_to_translation_target(
                current, translation, bool(generate_interpolated_path)
            ),
            "translation",
        )
    # 没有 target angles, 有 position 以及 orientation
    pose_target = pose_from_position_quat_wxyz(
        context.cumotion, request.goal_pose.position, request.goal_pose.orientation
    )
    return (
        planner.plan_to_pose_target(
            current, pose_target, bool(generate_interpolated_path)
        ),
        "pose",
    )


def _motion_result(
    context,
    results,
    config: MotionPlannerBackendConfig,
    *,
    target_type: _PlannerTargetType,
    frame_name: str,
    num_collision_objects: int,
    duration_s: float | None,
) -> MotionResult:
    """把 ``MotionPlanner.Results`` 标准化成项目 ``MotionResult``。"""

    # graph planner 的原生产物是离散 C-space path；有些 cuMotion 版本还会同时给出
    # interpolated_path。这里先保留 path，是为了让上层能记录 waypoint 数量、路径长度和
    # 末端构型；真正执行时不允许直接播放这个离散 path，必须在下面强制生成 trajectory。
    # 是否优先使用 interpolated_path 只由 graph_search.generate_interpolated_path 控制。
    path_samples = result_path_samples(
        results,
        prefer_interpolated=config.graph_search.generate_interpolated_path,
    )
    joint_path = stack_path(path_samples)
    # path_found 为真但没有实际 path 时仍视为失败，避免上层拿到 success=True 却没有可执行数据。
    success = bool(attr(results, "path_found", default=False)) and joint_path is not None
    trajectory = None
    if success:
        # 项目侧现在把 trajectory generation 作为 graph_search 成功结果的强约束：
        # 只返回 path 但没有时间轴/速度/加速度的结果不能进入 execution 层。这样可以避免
        # 动作脚本层各自用简单插值兜底，导致同一个后端结果在不同动作里有不同执行语义。
        trajectory = generate_cspace_trajectory(
            context,
            joint_path,
            config.trajectory_generation,
            duration_s=duration_s,
        )
        if trajectory is None:
            success = False
    metrics = {
        "num_waypoints": float(0 if joint_path is None else joint_path.shape[0]),
        "num_collision_objects": float(num_collision_objects),
        "path_length": float(path_length(joint_path)),
    }
    diagnostics = PlanningDiagnostics(
        status="SUCCESS" if success else "FAILED",
        message=f"pipeline=graph_search target={target_type} frame={frame_name}",
        metrics=metrics,
    )
    return MotionResult(
        path=joint_path if success else None,
        trajectory=trajectory if success else None,
        success=success,
        status=diagnostics.status,
        diagnostics=diagnostics,
    )


def _motion_planner_config(
    context,
    *,
    frame_name: str,
    world_view,
    config: MotionPlannerBackendConfig,
):
    """创建 graph ``MotionPlannerConfig`` 并写入参数覆盖。"""

    config_path = config.graph_search.motion_planner_config_path
    if config_path:
        planner_config = context.cumotion.create_motion_planner_config_from_file(
            Path(config_path),
            context.robot_description,
            frame_name,
            world_view,
        )
    else:
        planner_config = context.cumotion.create_default_motion_planner_config(
            context.robot_description,
            frame_name,
            world_view,
        )
    apply_config_params(
        planner_config,
        config.graph_search.motion_planner_params,
        context.cumotion.MotionPlannerConfig.ParamValue,
    )
    return planner_config
