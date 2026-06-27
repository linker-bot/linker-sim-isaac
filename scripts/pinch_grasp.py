#!/usr/bin/env python3
"""运行 AR5 + LinkerHand L6 的绳端夹捏抓取动作脚本。

本文件是一个完整可运行动作入口：抓取动作参数直接写在脚本内，不再通过
外部 trajectory YAML 或旧任务包间接提供。外部 YAML 只保留机器人、控制器、环境、绳体、
日志和 cuMotion profile 这些可复用系统配置。

执行流程：
    1. 读取系统配置并启动 Isaac Sim。
    2. 导入 capsule rope 和 AR5+L6 组合机器人。
    3. 创建 JointController 和可选日志器。
    4. 根据内置闭合手型计算 pinch TCP。
    5. 规划 approach、grasp、lift、wiggle 等阶段并按 physics dt 播放。

数组/坐标约定：
    动作内部大多使用 Isaac articulation 完整 DOF 顺序；只有进入 cuMotion 时才切到
    ``context.joint_names()`` 对应的 C-space 顺序。笛卡尔目标使用当前 demo 的 world/base 对齐
    坐标，单位 m；关节角和 RPY 使用 rad。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.app.launch import launch_simulation_app
from linkerbot_sim.assets.robot_loader import (
    RobotAssetConfig,
    import_robot_asset,
)
from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    apply_solver_iteration_overrides,
)
from linkerbot_sim.assets.usd_overrides import (
    apply_robot_usd_overrides,
    disable_robot_gravity,
)
from linkerbot_sim.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)
from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
    SpecifiedPathConfig,
)
from linkerbot_sim.backends.cumotion.tcp_context import make_cumotion_context
from linkerbot_sim.backends.cumotion.trajectory_sampler import (
    joint_trajectory_from_cumotion,
)
from linkerbot_sim.controllers.config import (
    joint_control_settings,
    load_controller_profiles,
    physx_override_configs,
)
from linkerbot_sim.controllers.joint_controller import JointController
from linkerbot_sim.envs.scene_builder import build_world, configure_visuals
from linkerbot_sim.execution.runtime import ExecutionRuntime, ExecutionStep
from linkerbot_sim.execution.steps import (
    FullJointTrajectoryStep,
    HoldJointTargetStep,
    SmoothJointTargetStep,
)
from linkerbot_sim.logging.config import (
    joint_logging_config_from_mapping,
    override_logging_config,
)
from linkerbot_sim.logging.joint_logger import JointTrackingLogger
from linkerbot_sim.objects.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
    endpoint_center,
)
from linkerbot_sim.planning.requests import (
    IKRequest,
    MotionRequest,
    SpecifiedPathRequest,
    TaskSpacePath,
    TcpLineSegment,
)
from linkerbot_sim.robots.joint_groups import target_vector_from_mapping
from linkerbot_sim.robots.mimic import (
    expand_targets_with_mjcf_equalities,
    mjcf_equality_follower_joint_names,
)
from linkerbot_sim.tcp.pinch_tcp import DEFAULT_PINCH_TCP_FRAME, make_pinch_tcp
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.types import JointTrajectory
from linkerbot_sim.utils.config import deep_merge, load_yaml
from linkerbot_sim.utils.paths import repo_path
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


_CUMOTION_TRAJECTORY_SAMPLE_STEP_MULTIPLE = 1
_CUMOTION_TRAJECTORY_MIN_SAMPLES = 50


DEFAULT_PRE_PINCH_HAND_TARGETS = {
    "L6V1_L_hand_thumb_cmc_roll": 0.95,
    "L6V1_L_hand_thumb_cmc_pitch": 0.28,
    "L6V1_L_hand_index_mcp_pitch": 0.25,
    "L6V1_L_hand_middle_mcp_pitch": 0.15,
    "L6V1_L_hand_ring_mcp_pitch": 0.15,
    "L6V1_L_hand_pinky_mcp_pitch": 0.12,
}

DEFAULT_CLOSED_PINCH_HAND_TARGETS = {
    "L6V1_L_hand_thumb_cmc_roll": 0.95,
    "L6V1_L_hand_thumb_cmc_pitch": 0.7,
    "L6V1_L_hand_index_mcp_pitch": 0.85,
    "L6V1_L_hand_middle_mcp_pitch": 0.45,
    "L6V1_L_hand_ring_mcp_pitch": 0.4,
    "L6V1_L_hand_pinky_mcp_pitch": 0.35,
}


DEFAULT_ENDPOINT = "left"
TARGET_WORLD_OFFSET = (0.02, 0.0, 0.03)
TARGET_RPY = (0.0, 2.007128639793479, -1.5707963267948966)
USE_ORIENTATION = True
APPROACH_DISTANCE = 0.10
LIFT_HEIGHT = 0.4
PREP_DURATION = 1.0
MOVE_DURATION = 3.0
APPROACH_DURATION = 1.2
CLOSE_DURATION = 1.0
LIFT_DURATION = 2.0
WIGGLE_CYCLES = 2
WIGGLE_AMPLITUDE = 0.2
WIGGLE_DURATION = 2.0
WIGGLE_AXIS = (1.0, 0.0, 0.0)
FINAL_HOLD_DURATION = 5.0
POST_JOINT_SWEEP_DURATION = 2.0
POST_JOINT_SWEEP_TARGETS = (2.1, -2.1)
TCP_FRAME_NAME = DEFAULT_PINCH_TCP_FRAME

SHORT_SMOKE_DURATION = 0.02
SHORT_SMOKE_LIFT_HEIGHT = 0.05


def build_planned_joint_motion_trajectory(
    *,
    motion_planner,
    dof_names: list[str],
    arm_indices: np.ndarray,
    start_all: np.ndarray,
    target_all: np.ndarray,
    duration_s: float,
    phase: str,
    physics_dt: float | None = None,
) -> JointTrajectory:
    """用 cuMotion 规划一段完整 DOF 的关节角到关节角运动。

    ``motion_planner`` 只处理 cuMotion C-space 关节；本函数负责把完整 articulation
    DOF 中的机械臂关节切出来调用 ``MotionRequest(goal_q=...)``，再把返回的 C-space
    trajectory 嵌回完整 DOF 轨迹。非 C-space DOF 会按 start/target 线性插值，用于夹爪和
    机械臂同时变化的阶段。
    """

    start = np.asarray(start_all, dtype=float).reshape(-1)
    target = np.asarray(target_all, dtype=float).reshape(-1)
    arm_indices = np.asarray(arm_indices, dtype=int).reshape(-1)
    start_q = start[arm_indices]
    target_q = target[arm_indices]
    result = motion_planner.plan(
        MotionRequest(
            current_q=start_q,
            goal_q=target_q,
            duration_s=duration_s,
        )
    )
    if not result.success:
        raise RuntimeError(
            f"cuMotion joint motion planning failed for {phase}: status={result.status}"
        )

    if result.trajectory is None:
        raise RuntimeError(
            f"cuMotion joint motion planning returned no trajectory for {phase}: "
            f"status={result.status}"
        )
    return _full_trajectory_from_cumotion_trajectory(
        result.trajectory,
        motion_planner=motion_planner,
        dof_names=dof_names,
        arm_indices=arm_indices,
        start_all=start,
        target_all=target,
        requested_duration_s=duration_s,
        phase=phase,
        physics_dt=physics_dt,
    )


def build_specified_tcp_line_trajectory(
    *,
    context: CuMotionContext,
    tcp_frame_name: str,
    dof_names: list[str],
    arm_indices: np.ndarray,
    start_all: np.ndarray,
    target_position: np.ndarray,
    duration_s: float,
    phase: str,
    physics_dt: float | None = None,
    base_config: MotionPlannerBackendConfig | None = None,
) -> JointTrajectory:
    """用 specified_path 的 ``TcpLineSegment`` 构建完整 DOF TCP 直线轨迹。

    该函数是动作脚本到新 specified-path 接口的直接调用点：不再经过旧的 ``tcp_line.py`` 逐点 IK
    helper，也不保留 ``TcpLineRequest`` 兼容层。cuMotion 只返回 C-space 路径/轨迹；这里负责把
    机械臂 C-space 列写回完整 articulation DOF。
    """

    start = np.asarray(start_all, dtype=float).reshape(-1)
    arm_indices = np.asarray(arm_indices, dtype=int).reshape(-1)
    current_q = start[arm_indices]
    target_position = np.asarray(target_position, dtype=float).reshape(3)

    specified_planner = context.make_motion_planner(
        tcp_frame_name=tcp_frame_name,
        config=_specified_path_config_from_base(base_config),
    )
    result = specified_planner.plan(
        SpecifiedPathRequest(
            current_q=current_q,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=target_position,
                        orientation_mode="current",
                    ),
                )
            ),
        )
    )
    if not result.success:
        raise RuntimeError(
            f"cuMotion specified TCP line planning failed for {phase}: "
            f"status={result.status}"
        )

    if result.trajectory is None:
        raise RuntimeError(
            f"cuMotion specified TCP line returned no trajectory for {phase}: "
            f"status={result.status}"
        )
    if result.path is None or np.asarray(result.path).shape[0] == 0:
        raise RuntimeError(
            f"cuMotion specified TCP line returned trajectory without path "
            f"for {phase}: status={result.status}"
        )
    target_all = start.copy()
    target_all[arm_indices] = np.asarray(result.path, dtype=float)[-1]
    return _full_trajectory_from_cumotion_trajectory(
        result.trajectory,
        motion_planner=specified_planner,
        dof_names=dof_names,
        arm_indices=arm_indices,
        start_all=start,
        target_all=target_all,
        requested_duration_s=duration_s,
        phase=phase,
        physics_dt=physics_dt,
    )


def _specified_path_config_from_base(
    base_config: MotionPlannerBackendConfig | None,
) -> MotionPlannerBackendConfig:
    """从动作主 planner 配置派生 task-space specified-path 配置。"""

    base = base_config or MotionPlannerBackendConfig()
    return MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        graph_search=base.graph_search,
        trajectory_generation=base.trajectory_generation,
        trajectory_optimization=base.trajectory_optimization,
        specified_path=SpecifiedPathConfig(
            family="task_space_segments",
            validate_collision_after_generation=(
                base.specified_path.validate_collision_after_generation
            ),
            cspace_waypoints=base.specified_path.cspace_waypoints,
            task_space_segments=base.specified_path.task_space_segments,
            composite=base.specified_path.composite,
        ),
    )


def _full_trajectory_from_cumotion_trajectory(
    cumotion_trajectory,
    *,
    motion_planner,
    dof_names: list[str],
    arm_indices: np.ndarray,
    start_all: np.ndarray,
    target_all: np.ndarray,
    requested_duration_s: float,
    phase: str,
    physics_dt: float | None,
) -> JointTrajectory:
    """把 cuMotion time-parameterized C-space trajectory 嵌回完整 DOF 轨迹。

    cuMotion ``MotionPlanner`` 负责找路径，``CSpaceTrajectoryGenerator`` 负责按照机器人
    约束生成带时间、速度、加速度和 jerk 的轨迹。本函数只做两个项目侧适配：
    1. 如果配置阶段时长大于 cuMotion 生成时长，则等比例拉长时间轴并缩放导数，保持运动更慢
       且不超过 cuMotion 速度/加速度规划；不会把轨迹压短到小于 cuMotion 生成时长。
    2. cuMotion 只包含机械臂 C-space，本函数把这些列写回完整 articulation DOF；非 C-space
       DOF 继续按起终点做线性补齐，用于手部或其它未纳入 cuMotion 模型的关节。
    """

    cspace_trajectory = _sample_and_retime_cumotion_trajectory(
        cumotion_trajectory,
        joint_names=tuple(motion_planner.joint_names()),
        requested_duration_s=requested_duration_s,
        phase=phase,
        physics_dt=physics_dt,
    )
    arm_indices = np.asarray(arm_indices, dtype=int).reshape(-1)
    if cspace_trajectory.positions.shape[1] != arm_indices.size:
        raise ValueError(
            "cuMotion trajectory dof mismatch: "
            f"trajectory has {cspace_trajectory.positions.shape[1]} columns, "
            f"arm_indices has {arm_indices.size}"
        )

    duration_s = float(cspace_trajectory.times[-1])
    full_positions = _full_positions_from_cspace_path(
        cspace_trajectory.positions,
        times=cspace_trajectory.times,
        duration_s=duration_s,
        start_all=start_all,
        target_all=target_all,
        arm_indices=arm_indices,
    )

    # 先用完整位置矩阵做一次有限差分，给非 C-space DOF 生成一致的速度/加速度诊断；随后用
    # cuMotion 的导数覆盖机械臂列，确保执行层能拿到后端真实的加减速规划结果。
    baseline = joint_trajectory_from_positions(
        times=cspace_trajectory.times,
        positions=full_positions,
        joint_names=tuple(dof_names),
        phase=phase,
    )
    velocities = baseline.velocities.copy()
    accelerations = baseline.accelerations.copy()
    jerks = baseline.jerks.copy()
    velocities[:, arm_indices] = cspace_trajectory.velocities
    accelerations[:, arm_indices] = cspace_trajectory.accelerations
    jerks[:, arm_indices] = cspace_trajectory.jerks
    return JointTrajectory.from_samples(
        times=cspace_trajectory.times,
        positions=full_positions,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        phases=tuple(phase for _ in range(cspace_trajectory.times.size)),
        joint_names=tuple(dof_names),
    )


def _sample_and_retime_cumotion_trajectory(
    cumotion_trajectory,
    *,
    joint_names: tuple[str, ...],
    requested_duration_s: float,
    phase: str,
    physics_dt: float | None,
) -> JointTrajectory:
    """采样 cuMotion trajectory，并按配置时长做只拉长不压短的时间缩放。"""

    sample_dt = _cumotion_trajectory_sample_dt(
        cumotion_trajectory, physics_dt=physics_dt
    )
    trajectory = joint_trajectory_from_cumotion(
        cumotion_trajectory,
        joint_names=joint_names,
        sample_dt=sample_dt,
        phase=phase,
    )
    relative_times = trajectory.times - float(trajectory.times[0])
    generated_duration = float(relative_times[-1])
    if generated_duration <= 1.0e-12:
        return JointTrajectory.from_samples(
            times=np.asarray([0.0], dtype=float),
            positions=trajectory.positions[-1:].copy(),
            velocities=np.zeros_like(trajectory.positions[-1:]),
            accelerations=np.zeros_like(trajectory.positions[-1:]),
            jerks=np.zeros_like(trajectory.positions[-1:]),
            phases=(phase,),
            joint_names=joint_names,
        )

    # requested_duration_s 是 demo 配置里的阶段时长。若它更长，只会降低速度/加速度/jerk；
    # 若它更短，则保留 cuMotion generator 给出的时长，避免把已规划好的加速度约束压坏。
    target_duration = max(float(requested_duration_s), generated_duration)
    scale = target_duration / generated_duration
    return JointTrajectory.from_samples(
        times=relative_times * scale,
        positions=trajectory.positions.copy(),
        velocities=trajectory.velocities / scale,
        accelerations=trajectory.accelerations / (scale * scale),
        jerks=trajectory.jerks / (scale * scale * scale),
        phases=tuple(phase for _ in range(trajectory.times.size)),
        joint_names=joint_names,
    )


def _cumotion_trajectory_sample_dt(
    cumotion_trajectory, *, physics_dt: float | None
) -> float:
    """根据 physics step 和 cuMotion 轨迹时域选择采样周期。"""

    lower, upper = _trajectory_domain_bounds(cumotion_trajectory)
    span = max(0.0, upper - lower)
    if physics_dt is not None:
        step_dt = float(physics_dt)
        base_dt = step_dt * float(_CUMOTION_TRAJECTORY_SAMPLE_STEP_MULTIPLE)
    else:
        # 测试或离线构建没有 Isaac world 时退回 100 Hz，但运行路径应优先从
        # world.get_physics_dt() 传入 physics_dt，避免采样点与物理步长错位。
        base_dt = 0.01
    if span <= 0.0:
        return base_dt
    return min(
        base_dt,
        max(span / float(_CUMOTION_TRAJECTORY_MIN_SAMPLES), 1.0e-4),
    )


def _trajectory_domain_bounds(trajectory) -> tuple[float, float]:
    """读取 cuMotion trajectory domain 的上下界。"""

    domain = trajectory.domain()
    lower = float(
        getattr(domain, "lower", domain[0] if isinstance(domain, tuple) else 0.0)
    )
    upper = float(
        getattr(domain, "upper", domain[1] if isinstance(domain, tuple) else lower)
    )
    return lower, upper


def _full_positions_from_cspace_path(
    cspace_path: np.ndarray,
    *,
    times: np.ndarray,
    duration_s: float,
    start_all: np.ndarray,
    target_all: np.ndarray,
    arm_indices: np.ndarray,
) -> np.ndarray:
    """把 C-space path 嵌回完整 DOF 位置矩阵。"""

    start = np.asarray(start_all, dtype=float).reshape(-1)
    target = np.asarray(target_all, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    if duration_s <= 0:
        alpha = np.ones((times.size, 1), dtype=float)
    else:
        alpha = (times / float(duration_s)).reshape(-1, 1)
    full_positions = start.reshape(1, -1) + alpha * (target - start).reshape(1, -1)
    full_positions[:, np.asarray(arm_indices, dtype=int).reshape(-1)] = cspace_path
    full_positions[0] = start
    full_positions[-1] = target
    return full_positions


def grasp_target_position(
    rope_config: CapsuleRopeConfig, *, endpoint: str, lift_height: float = 0.0
) -> np.ndarray:
    """计算夹捏 TCP 的世界坐标目标位置。

    参数:
        rope_config: rope 对象配置，提供端点 cuboid 的几何位置。
        endpoint: 抓取左端或右端。
        lift_height: 额外 z 方向抬升高度，单位 m。
    返回:
        shape 为 ``(3,)`` 的世界坐标位置数组，单位 m。
    """

    # endpoint_center 给出端块几何中心，target_world_offset 用于把 TCP 对准更适合夹捏的点，
    # lift_height 只在 z 方向叠加，保持抓取水平位置不变。
    return (
        np.asarray(endpoint_center(rope_config, endpoint), dtype=float)
        + np.asarray(TARGET_WORLD_OFFSET, dtype=float)
        + np.asarray([0.0, 0.0, lift_height], dtype=float)
    )


def plan_pinch_grasp(
    robot,
    *,
    rope_config: CapsuleRopeConfig,
    mjcf_path: str | Path,
    cumotion_config: CuMotionConfig,
    motion_planner_config: MotionPlannerBackendConfig,
    endpoint: str,
    short_smoke: bool,
    physics_dt: float | None = None,
) -> dict[str, object]:
    """规划夹捏抓取各阶段的 IK 解和完整 DOF 目标。"""

    mjcf_path = Path(mjcf_path)
    tcp_frame_name = TCP_FRAME_NAME
    lift_height = SHORT_SMOKE_LIFT_HEIGHT if short_smoke else LIFT_HEIGHT
    move_duration = SHORT_SMOKE_DURATION if short_smoke else MOVE_DURATION
    approach_duration = SHORT_SMOKE_DURATION if short_smoke else APPROACH_DURATION
    lift_duration = SHORT_SMOKE_DURATION if short_smoke else LIFT_DURATION
    wiggle_cycles = 0 if short_smoke else WIGGLE_CYCLES
    wiggle_duration = SHORT_SMOKE_DURATION if short_smoke else WIGGLE_DURATION
    post_joint_sweep_duration = (
        SHORT_SMOKE_DURATION if short_smoke else POST_JOINT_SWEEP_DURATION
    )
    post_joint_sweep_target_values = () if short_smoke else POST_JOINT_SWEEP_TARGETS

    # 先用闭合手型计算 thumb/index 的几何夹捏中心。这里需要展开 mimic follower，
    # 否则 MJCF 运动链里从动关节会停在 0，TCP 会偏离实际闭合指尖中心。
    closed_geometry_targets = expand_targets_with_mjcf_equalities(
        DEFAULT_CLOSED_PINCH_HAND_TARGETS, mjcf_path
    )
    tcp = make_pinch_tcp(
        mjcf_path,
        closed_geometry_targets,
        parent_frame=cumotion_config.flange_frame,
        frame_name=tcp_frame_name,
    )

    # 三个核心笛卡尔目标：接近点、真正抓取点、抬升点。wiggle 目标在抬升点附近
    # 沿配置轴线来回偏移，用来验证抓取是否稳定。
    pinch_world = grasp_target_position(rope_config, endpoint=endpoint)
    approach_world = pinch_world + np.asarray(
        [0.0, 0.0, APPROACH_DISTANCE], dtype=float
    )
    lifted_world = grasp_target_position(
        rope_config, endpoint=endpoint, lift_height=lift_height
    )
    wiggle_axis = np.asarray(WIGGLE_AXIS, dtype=float)
    wiggle_axis = wiggle_axis / np.linalg.norm(wiggle_axis)
    wiggle_worlds: list[np.ndarray] = []
    for _cycle_index in range(wiggle_cycles):
        wiggle_worlds.append(lifted_world - wiggle_axis * WIGGLE_AMPLITUDE)
        wiggle_worlds.append(lifted_world + wiggle_axis * WIGGLE_AMPLITUDE)

    target_orientation = rpy_xyz_to_quat_wxyz(TARGET_RPY)
    ik_orientation = target_orientation if USE_ORIENTATION else None
    # IK 后端只认识机器人描述里的 frame。cuMotion backend 负责把 pinch TCP 装配进
    # 临时 URDF/context，动作脚本只保留 TCP 几何和完整 DOF 映射逻辑。
    with make_cumotion_context(cumotion_config, tcp=tcp) as context:
        ik_defaults = context.config.kinematics.ik
        ik_joint_names = context.joint_names()
        dof_names = list(robot.dof_names)
        dof_index_by_name = {name: index for index, name in enumerate(dof_names)}
        # cuMotion 模型和 Isaac articulation 可能来自不同资产文件。这里按名称检查能尽早
        # 发现 URDF/MJCF 关节名不一致，而不是在写目标数组时静默错位。
        missing_ik_joints = [
            name for name in ik_joint_names if name not in dof_index_by_name
        ]
        if missing_ik_joints:
            raise ValueError(
                f"cuMotion joints not found in articulation: {missing_ik_joints}"
            )
        arm_indices = np.asarray(
            [dof_index_by_name[name] for name in ik_joint_names], dtype=int
        )
        current_cspace = np.asarray(robot.get_joint_positions(), dtype=float).reshape(
            -1
        )[arm_indices]
        solver = context.make_inverse_kinematics(tcp_frame_name=tcp_frame_name)
        motion_planner = context.make_motion_planner(
            tcp_frame_name=tcp_frame_name,
            config=motion_planner_config,
        )
        # 第一次 IK 用当前 articulation C-space 热启动，后续阶段用上一阶段解热启动，
        # 保持关节轨迹连续，也减少求解器跳解概率。
        approach = solver.solve(
            IKRequest(
                target_position=approach_world,
                target_orientation=ik_orientation,
                warm_start_ik_cspace_seed=current_cspace,
                position_tolerance=ik_defaults.position_tolerance,
                orientation_tolerance=ik_defaults.orientation_tolerance,
            )
        )
        initial_all = np.asarray(robot.get_joint_positions(), dtype=float)
        pre_pinch_all = target_vector_from_mapping(
            dof_names, DEFAULT_PRE_PINCH_HAND_TARGETS, base=initial_all
        )
        approach_all = pre_pinch_all.copy()
        approach_all[arm_indices] = np.asarray(approach.joint_positions, dtype=float)
        # approach_all 是接近点的完整姿态；从这里开始构建一条短 TCP 直线下沉轨迹，
        # 比直接 IK 到抓取点再关节插值更接近“沿竖直方向靠近端块”的动作意图。TCP 直线现在
        # 直接使用 specified_path 的 TaskSpacePath/TcpLineSegment，不再经过旧逐点 IK helper。
        grasp_line_trajectory = build_specified_tcp_line_trajectory(
            context=context,
            tcp_frame_name=tcp_frame_name,
            dof_names=dof_names,
            arm_indices=arm_indices,
            start_all=approach_all,
            target_position=pinch_world,
            duration_s=approach_duration,
            phase="approach_box",
            physics_dt=physics_dt,
            base_config=motion_planner_config,
        )
        grasp_joint_positions = np.asarray(
            grasp_line_trajectory.positions[-1], dtype=float
        )[arm_indices]
        lift = solver.solve(
            IKRequest(
                target_position=lifted_world,
                target_orientation=ik_orientation,
                warm_start_ik_cspace_seed=grasp_joint_positions,
                position_tolerance=ik_defaults.position_tolerance,
                orientation_tolerance=ik_defaults.orientation_tolerance,
            )
        )
        # wiggle 阶段每个目标都用上一目标热启动，减少在冗余机械臂上突然换解的概率。
        wiggles = []
        warm = lift.joint_positions
        for target in wiggle_worlds:
            result = solver.solve(
                IKRequest(
                    target_position=target,
                    target_orientation=ik_orientation,
                    warm_start_ik_cspace_seed=warm,
                    position_tolerance=ik_defaults.position_tolerance,
                    orientation_tolerance=ik_defaults.orientation_tolerance,
                )
            )
            wiggles.append((target, result))
            warm = result.joint_positions

        # 把 cuMotion IK 解写回完整 articulation 目标。手部关节用稀疏映射覆盖，其它 DOF
        # 沿用上一阶段目标，保证未参与阶段切换的关节不被意外归零。
        grasp_open_all = np.asarray(
            grasp_line_trajectory.positions[-1], dtype=float
        ).copy()
        grasp_closed_all = target_vector_from_mapping(
            dof_names, DEFAULT_CLOSED_PINCH_HAND_TARGETS, base=grasp_open_all
        )
        lifted_all = grasp_closed_all.copy()
        lifted_all[arm_indices] = np.asarray(lift.joint_positions, dtype=float)

        wiggle_all_targets = []
        for _world, result in wiggles:
            wiggle_all = grasp_closed_all.copy()
            wiggle_all[arm_indices] = np.asarray(result.joint_positions, dtype=float)
            wiggle_all_targets.append(wiggle_all)

        # 末尾扫动第 1 个机械臂关节是 scripted demo 的额外扰动，用于观察夹持是否稳固。
        post_joint_sweep_targets = []
        for joint_1_target in post_joint_sweep_target_values:
            sweep_all = lifted_all.copy()
            sweep_all[arm_indices[0]] = float(joint_1_target)
            post_joint_sweep_targets.append(sweep_all)

        # 下面这些阶段都是“机械臂关节角 -> 机械臂关节角”的运动：cuMotion MotionPlanner
        # 在 C-space 中生成路径，再把路径嵌回完整 DOF 轨迹播放。手指开合阶段使用
        # smoothstep，因为手部 DOF 不属于 cuMotion 机器人描述的 C-space。
        move_to_approach_trajectory = build_planned_joint_motion_trajectory(
            motion_planner=motion_planner,
            dof_names=dof_names,
            arm_indices=arm_indices,
            start_all=pre_pinch_all,
            target_all=approach_all,
            duration_s=move_duration,
            phase="move_to_approach",
            physics_dt=physics_dt,
        )
        lift_trajectory = build_planned_joint_motion_trajectory(
            motion_planner=motion_planner,
            dof_names=dof_names,
            arm_indices=arm_indices,
            start_all=grasp_closed_all,
            target_all=lifted_all,
            duration_s=lift_duration,
            phase="lift",
            physics_dt=physics_dt,
        )
        wiggle_trajectories = []
        previous_target = lifted_all
        for index, wiggle_all in enumerate(wiggle_all_targets, start=1):
            trajectory = build_planned_joint_motion_trajectory(
                motion_planner=motion_planner,
                dof_names=dof_names,
                arm_indices=arm_indices,
                start_all=previous_target,
                target_all=wiggle_all,
                duration_s=wiggle_duration,
                phase=f"wiggle_{index}",
                physics_dt=physics_dt,
            )
            wiggle_trajectories.append(trajectory)
            previous_target = wiggle_all
        wiggle_return_trajectory = None
        if wiggle_all_targets:
            wiggle_return_trajectory = build_planned_joint_motion_trajectory(
                motion_planner=motion_planner,
                dof_names=dof_names,
                arm_indices=arm_indices,
                start_all=previous_target,
                target_all=lifted_all,
                duration_s=wiggle_duration,
                phase="wiggle_return_center",
                physics_dt=physics_dt,
            )
        post_joint_sweep_trajectories = []
        previous_target = lifted_all
        for index, sweep_all in enumerate(post_joint_sweep_targets, start=1):
            trajectory = build_planned_joint_motion_trajectory(
                motion_planner=motion_planner,
                dof_names=dof_names,
                arm_indices=arm_indices,
                start_all=previous_target,
                target_all=sweep_all,
                duration_s=post_joint_sweep_duration,
                phase=f"post_joint_1_sweep_{index}",
                physics_dt=physics_dt,
            )
            post_joint_sweep_trajectories.append(trajectory)
            previous_target = sweep_all

    return {
        "initial_all": initial_all,
        "pre_pinch_all": pre_pinch_all,
        "move_to_approach_trajectory": move_to_approach_trajectory,
        "approach_line_trajectory": grasp_line_trajectory,
        "grasp_open_all": grasp_open_all,
        "grasp_closed_all": grasp_closed_all,
        "lifted_all": lifted_all,
        "lift_trajectory": lift_trajectory,
        "wiggle_trajectories": wiggle_trajectories,
        "wiggle_return_trajectory": wiggle_return_trajectory,
        "post_joint_sweep_trajectories": post_joint_sweep_trajectories,
        "ik": {
            "pinch_world": pinch_world,
            "approach_world": approach_world,
            "lifted_world": lifted_world,
            "tcp_xyz": tcp.xyz,
            "approach_success": approach.success,
            "approach_error": approach.position_error,
            "lift_success": lift.success,
            "lift_error": lift.position_error,
            "wiggles": [
                (world_target, result.success, result.position_error)
                for world_target, result in wiggles
            ],
        },
    }


def pinch_grasp_execution_steps(
    plan: dict[str, object], *, short_smoke: bool
) -> list[ExecutionStep]:
    """把抓取 plan 拆成可顺序执行的执行步骤列表。"""

    prep_duration = SHORT_SMOKE_DURATION if short_smoke else PREP_DURATION
    close_duration = SHORT_SMOKE_DURATION if short_smoke else CLOSE_DURATION
    final_hold_duration = SHORT_SMOKE_DURATION if short_smoke else FINAL_HOLD_DURATION
    steps: list[ExecutionStep] = [
        SmoothJointTargetStep(
            start_all=plan["initial_all"],
            target_all=plan["pre_pinch_all"],
            duration=prep_duration,
            phase="pre_pinch",
        ),
        FullJointTrajectoryStep(
            trajectory=plan["move_to_approach_trajectory"],
            phase="move_to_approach",
        ),
        FullJointTrajectoryStep(
            trajectory=plan["approach_line_trajectory"],
            phase="approach_box",
        ),
        SmoothJointTargetStep(
            start_all=plan["grasp_open_all"],
            target_all=plan["grasp_closed_all"],
            duration=close_duration,
            phase="close_fingers",
        ),
        FullJointTrajectoryStep(
            trajectory=plan["lift_trajectory"],
            phase="lift",
        ),
    ]
    for index, trajectory in enumerate(plan["wiggle_trajectories"], start=1):
        steps.append(
            FullJointTrajectoryStep(trajectory=trajectory, phase=f"wiggle_{index}")
        )
    if plan["wiggle_return_trajectory"] is not None:
        steps.append(
            FullJointTrajectoryStep(
                trajectory=plan["wiggle_return_trajectory"],
                phase="wiggle_return_center",
            )
        )
    steps.append(
        HoldJointTargetStep(
            target_all=plan["lifted_all"],
            duration=final_hold_duration,
            phase="final",
        )
    )
    for index, trajectory in enumerate(plan["post_joint_sweep_trajectories"], start=1):
        steps.append(
            FullJointTrajectoryStep(
                trajectory=trajectory,
                phase=f"post_joint_1_sweep_{index}",
            )
        )
    return steps


def run_pinch_grasp_action(
    *,
    robot,
    world,
    articulation_action_type,
    controller,
    simulation_app,
    render: bool,
    rope_config: CapsuleRopeConfig,
    mjcf_path: str | Path,
    cumotion_config: CuMotionConfig,
    motion_planner_config: MotionPlannerBackendConfig,
    endpoint: str,
    short_smoke: bool,
    drive_logger=None,
) -> dict[str, object]:
    """规划并执行完整夹捏抓取脚本。"""

    # 先规划再构造 runtime，确保 IK/目标生成失败时不会推进 world，也不会写入半段日志。
    # cuMotion 连续轨迹采样使用 physics_dt 对齐物理步长，避免固定 0.01s 与仿真频率错位。
    plan = plan_pinch_grasp(
        robot,
        rope_config=rope_config,
        mjcf_path=mjcf_path,
        cumotion_config=cumotion_config,
        motion_planner_config=motion_planner_config,
        endpoint=endpoint,
        short_smoke=short_smoke,
        physics_dt=float(world.get_physics_dt()),
    )
    runtime = ExecutionRuntime(
        articulation=robot,
        simulation_world=world,
        articulation_action_type=articulation_action_type,
        joint_controller=controller,
        simulation_app=simulation_app,
        render_enabled=render,
        drive_logger=drive_logger,
    )
    step = 0
    for execution_step in pinch_grasp_execution_steps(plan, short_smoke=short_smoke):
        step = execution_step.run(runtime, step)
    return {"steps": step, "ik": plan["ik"]}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    各配置文件默认指向仓库内的标准抓绳 demo：
    - robot config 选择 AR5V2_L + L6V1_L 组合机器人；
    - controller config 目录提供按部件分组的位置、速度和 effort 控制参数；
    - env config 提供物理步频、重力和 solver iteration；
    - rope config 提供 capsule rope 资产路径和 prim 路径；
    - pinch grasp 动作参数固定在本脚本内，命令行只覆盖少量常用开关。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config", type=Path, default=Path("configs/robots/ar5v2_l6v1_l.yaml")
    )
    parser.add_argument(
        "--controller-config", type=Path, default=Path("configs/controllers")
    )
    parser.add_argument(
        "--env-config", type=Path, default=Path("configs/envs/rope_scene.yaml")
    )
    parser.add_argument(
        "--rope-config", type=Path, default=Path("configs/objects/capsule_rope.yaml")
    )
    parser.add_argument(
        "--cumotion-config",
        type=Path,
        default=Path("configs/cumotion/default.yaml"),
        help=(
            "cuMotion profile YAML. Its cumotion section is used as robot-level "
            "defaults, and cumotion.motion_planner is used as action planner defaults."
        ),
    )
    parser.add_argument(
        "--logging-config",
        type=Path,
        default=Path("configs/logging/default_logger.yaml"),
    )
    parser.add_argument(
        "--log", type=Path, default=None, help="覆盖关节跟踪 CSV 输出路径"
    )
    parser.add_argument(
        "--log-interval-steps", type=int, default=None, help="覆盖日志采样步长"
    )
    parser.add_argument(
        "--log-measured-effort",
        action="store_true",
        help="记录 PhysX measured joint effort",
    )
    parser.add_argument(
        "--log-applied-effort",
        action="store_true",
        help="记录 Isaac applied joint effort",
    )
    parser.add_argument(
        "--log-action-effort",
        action="store_true",
        help="记录控制器实际下发的 effort action",
    )
    parser.add_argument(
        "--no-log-effort-command",
        action="store_true",
        help="不记录语义 effort command 列",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="最终目标保持到窗口关闭")
    parser.add_argument(
        "--no-grasp", action="store_true", help="只导入机器人和绳体，并短暂保持初始姿态"
    )
    parser.add_argument(
        "--short-smoke",
        action="store_true",
        help="覆盖阶段时长，用于快速 headless smoke",
    )
    parser.add_argument("--endpoint", choices=("left", "right"), default=None)
    parser.add_argument(
        "--control-mode", choices=("position", "velocity", "effort"), default="position"
    )
    parser.add_argument("--physics-frequency", type=float, default=None)
    parser.add_argument("--render-frequency", type=float, default=None)
    parser.add_argument("--gravity-z", type=float, default=None)
    parser.add_argument("--enable-robot-gravity", action="store_true")
    return parser.parse_args()


def solver_settings(env_config: dict) -> SolverIterationConfig | None:
    """从环境配置构造机器人 solver iteration 覆盖设置。

    返回:
        配置中存在 ``solver`` 时返回 ``SolverIterationConfig``；不存在时返回
        ``None``，表示不主动覆盖 PhysX 默认 solver 设置。
    """

    solver = env_config.get("solver")
    if solver is None:
        return None
    return SolverIterationConfig(
        solver_type=str(solver.get("type", "TGS")),
        arm_position_iterations=int(solver.get("arm_position_iterations", 32)),
        arm_velocity_iterations=int(solver.get("arm_velocity_iterations", 4)),
        hand_position_iterations=int(solver.get("hand_position_iterations", 32)),
        hand_velocity_iterations=int(solver.get("hand_velocity_iterations", 4)),
        apply_scope=str(solver.get("apply_scope", "arm_hand")),
    )


def merged_robot_config_with_cumotion_profile(
    robot_config: dict, cumotion_profile: dict
) -> dict:
    """把 cuMotion profile 的后端默认值合入 robot config。

    profile 只提供通用 ``cumotion`` 默认参数；具体机器人 YAML 仍负责覆盖
    ``xrdf_path``、``urdf_path``、``flange_frame`` 等资产相关字段。
    """

    profile_cumotion = cumotion_profile.get("cumotion")
    if profile_cumotion is None:
        return dict(robot_config)
    if not isinstance(profile_cumotion, dict):
        raise ValueError("cuMotion profile key 'cumotion' must be a mapping")
    return deep_merge({"cumotion": profile_cumotion}, robot_config)


def robot_cumotion_config(robot_config: dict) -> CuMotionConfig:
    """读取 cuMotion 机器人模型配置。

    机器人配置必须通过 ``cumotion`` 段显式描述 cuMotion 资源和默认求解器参数。
    """

    return CuMotionConfig.from_mapping(robot_config)


def motion_planner_config_from_profile(
    cumotion_profile: dict,
) -> MotionPlannerBackendConfig:
    """从 cuMotion profile 里取出 motion planner 配置。"""

    profile_cumotion = cumotion_profile.get("cumotion")
    if profile_cumotion is None:
        return MotionPlannerBackendConfig.from_mapping(None)
    if not isinstance(profile_cumotion, dict):
        raise ValueError("cuMotion profile key 'cumotion' must be a mapping")
    profile_motion_planner = profile_cumotion.get("motion_planner")
    if profile_motion_planner is not None and not isinstance(
        profile_motion_planner, dict
    ):
        raise ValueError(
            "cuMotion profile key 'cumotion.motion_planner' must be a mapping"
        )
    return MotionPlannerBackendConfig.from_mapping(profile_motion_planner)


def hold_initial_pose(
    robot,
    world,
    articulation_action_type,
    controller,
    simulation_app,
    render: bool,
    logger,
) -> None:
    """保持当前姿态几步，用于 import smoke。

    ``--no-grasp`` 会走这个分支。它不执行抓取动作，只把当前机器人关节位置作为目标反复下发，
    用于确认机器人资产、驱动参数、mimic follower 和日志系统是否能正常初始化。
    如果同时传入 ``--hold`` 和 ``--gui``，会持续保持到 Isaac 窗口关闭。
    """

    full_target = np.asarray(robot.get_joint_positions(), dtype=float)
    full_velocity = np.zeros(robot.num_dof, dtype=float)
    step = 0
    while step < 3 or (simulation_app is not None and simulation_app.is_running()):
        targets = controller.targets_from_full_state(full_target, full_velocity)
        controller.apply_targets(articulation_action_type, targets)
        world.step(render=render)
        if logger is not None:
            driven_indices = controller.driven_indices
            if logger.should_write(step):
                log_values = logger.collect_step_values(
                    robot, controller, targets, driven_indices
                )
                logger.write(
                    step=step,
                    time_s=(step + 1) * float(world.get_physics_dt()),
                    phase="initial_hold",
                    drive_update=True,
                    **log_values,
                )
        step += 1
        if simulation_app is None and step >= 3:
            break


def main() -> None:
    """脚本主入口。"""

    # Isaac/Kit 日志很多，开启行缓冲可以保证 RUN_PINCH_GRASP_* 状态行尽快刷出，
    # 方便 live log、调试脚本和外部监控程序读取。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    # 先加载所有 YAML 配置。这里还没有启动 Isaac Sim，尽量把纯 Python 的配置错误提前暴露，
    # 避免启动 GUI 后才因为路径或字段缺失失败。
    cumotion_profile = load_yaml(args.cumotion_config)
    robot_config = merged_robot_config_with_cumotion_profile(
        load_yaml(args.robot_config), cumotion_profile
    )
    controller_profiles = load_controller_profiles(args.controller_config)
    env_config = load_yaml(args.env_config)
    rope_config_data = load_yaml(args.rope_config)
    logging_config = joint_logging_config_from_mapping(load_yaml(args.logging_config))
    logging_config = override_logging_config(
        logging_config,
        joint_tracking_path=args.log,
        interval_steps=args.log_interval_steps,
        log_measured_effort=True if args.log_measured_effort else None,
        log_applied_effort=True if args.log_applied_effort else None,
        log_action_effort=True if args.log_action_effort else None,
        log_command_effort=False if args.no_log_effort_command else None,
    )

    # RobotAssetConfig 只描述“如何把资产导入 stage”，例如 asset_type、asset_path、prim_path。
    # controlled_joints 则描述控制器主动下发目标的关节集合，mimic follower 会在运行时自动补齐。
    robot_asset = RobotAssetConfig.from_mapping(robot_config)
    controlled_joints = list(robot_config.get("controlled_joints", ["all"]))
    robot_cumotion = robot_cumotion_config(robot_config)

    rope_config = CapsuleRopeConfig.from_mapping(rope_config_data)
    motion_planner_config = motion_planner_config_from_profile(cumotion_profile)
    endpoint = args.endpoint or DEFAULT_ENDPOINT

    # 物理步频决定接触稳定性和控制刷新上限；渲染步频只影响 GUI/相机刷新。
    # headless 模式下 rendering_dt 之后会直接跟 physics_dt 对齐，避免无意义的渲染节拍。
    if "env" not in env_config:
        raise ValueError("Environment config must contain top-level env section")
    env = env_config["env"]
    physics_frequency = float(
        args.physics_frequency
        if args.physics_frequency is not None
        else env.get("physics_frequency", 600.0)
    )
    render_frequency = float(
        args.render_frequency
        if args.render_frequency is not None
        else env.get("render_frequency", 100.0)
    )
    gravity_z = float(
        args.gravity_z if args.gravity_z is not None else env.get("gravity_z", -9.81)
    )
    if physics_frequency <= 0 or render_frequency <= 0:
        raise ValueError("physics and render frequencies must be positive")

    # Isaac Sim 首次启动时需要接受 EULA；这里设置默认值，避免 headless 运行卡在交互确认。
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    simulation_app = launch_simulation_app(gui=args.gui)
    try:
        # Isaac 相关 import 必须放在 SimulationApp 启动之后，否则部分扩展和 USD context 尚未初始化。
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        import omni.usd

        # 创建 World。physics_dt 控制 PhysX step，rendering_dt 控制 GUI 刷新间隔。
        physics_dt = 1.0 / physics_frequency
        rendering_dt = 1.0 / render_frequency if args.gui else physics_dt
        world = build_world(
            physics_dt=physics_dt, rendering_dt=rendering_dt, gravity_z=gravity_z
        )
        if args.gui:
            configure_visuals()

        # 绳体对象以 USD reference 的方式挂到 stage 中。这里返回的 rope_model 主要用于打印诊断，
        # 真实碰撞体和关节由 USD/PhysX 在 stage 中维护。
        stage = omni.usd.get_context().get_stage()
        rope_model = add_capsule_rope_reference(stage, rope_config)
        print(
            "RUN_PINCH_GRASP_ROPE "
            f"asset={rope_config.asset_file()} prim_path={rope_config.prim_path} "
            f"segments={rope_config.segments} shape={rope_config.shape} "
            f"bodies={len(rope_model['bodies'])} joints={len(rope_model['joints'])}",
            flush=True,
        )

        # 导入机器人资产。MJCF/URDF 导入后会生成 stage prim；后续控制和 PhysX 覆盖都基于该 prim。
        articulation_path, asset_path, imported_root_path = import_robot_asset(
            robot_asset
        )
        mjcf_path = asset_path if robot_asset.asset_type == "mjcf" else None

        # 对刚导入的 USD prim 做运行时覆盖：关节 drive 初值、摩擦、最大力、碰撞近似等。
        # 这些覆盖不会修改原始资产文件，只影响当前 stage。
        apply_robot_usd_overrides(
            imported_root_path,
            physx_override_configs(controller_profiles),
            driven_joint_names=controlled_joints,
            mjcf_path=mjcf_path,
        )

        # 根据 env.solver 覆盖 articulation/rigid body 的 PhysX solver iteration。
        # 抓绳接触和灵巧手多关节链都对 solver iteration 较敏感。
        solver_config = solver_settings(env_config)
        solver_counts = (
            apply_solver_iteration_overrides(stage, articulation_path, solver_config)
            if solver_config is not None
            else {"configured": 0}
        )

        # 默认关闭机器人刚体重力，让关节主要按控制器命令运动。
        # 如果需要测试真实重力下的下垂、显式控制或力控行为，可以传 --enable-robot-gravity。
        if not args.enable_robot_gravity:
            disabled = disable_robot_gravity(imported_root_path)
            print(
                f"RUN_PINCH_GRASP_GRAVITY robot_gravity=false disabled_rigid_bodies={len(disabled)}",
                flush=True,
            )
        else:
            print("RUN_PINCH_GRASP_GRAVITY robot_gravity=true", flush=True)
        print(f"RUN_PINCH_GRASP_SOLVER {solver_counts}", flush=True)

        # 将导入后的 articulation 包装为 Isaac Sim SingleArticulation，并 reset world 以初始化 handles。
        robot = world.scene.add(
            SingleArticulation(prim_path=articulation_path, name=robot_asset.name)
        )
        world.reset()
        world.get_physics_context().set_gravity(gravity_z)
        if not args.enable_robot_gravity:
            robot.disable_gravity()
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))

        # JointController 负责：
        # - 按 --control-mode 选择 position、velocity 或 effort 主动关节控制配置；
        # - 为 implicit 控制写入 Isaac drive gain，为 explicit 控制计算 effort action；
        # - 始终用 follower 独立 position drive 跟随 master 实际状态。
        controller = JointController(
            robot,
            joint_names=controlled_joints,
            settings=joint_control_settings(
                controller_profiles, mode=args.control_mode
            ),
            mjcf_path=mjcf_path,
        )
        controller.configure_runtime()

        # L6 手的 DIP 等 follower 关节由 MJCF equality 描述。运行时根据实际 master 关节状态
        # 更新 follower 目标，避免 follower 跟随“命令目标”而不是“实际主动关节”导致超前。
        mimic_names = mjcf_equality_follower_joint_names(mjcf_path)

        # 日志只记录实际受驱动的 DOF，即主动关节 + mimic follower。flush_interval_steps 控制
        # CSV 刷盘频率，避免每个 physics step 都 flush 造成 I/O 开销过大。
        driven_joint_names = [
            list(robot.dof_names)[int(index)] for index in controller.driven_indices
        ]
        flush_interval_steps = logging_config.flush_interval_steps(
            float(world.get_physics_dt())
        )
        log_path = (
            None
            if not logging_config.enabled or logging_config.joint_tracking_path is None
            else repo_path(logging_config.joint_tracking_path)
        )
        logger = JointTrackingLogger(
            log_path,
            driven_joint_names,
            flush_interval_steps=flush_interval_steps,
            config=logging_config,
        )
        print(
            "RUN_PINCH_GRASP_IMPORTED "
            f"asset={asset_path} prim_path={articulation_path} num_dof={robot.num_dof} "
            f"control_mode={args.control_mode} mimic_joint_names={sorted(mimic_names)} "
            f"follower_relations={controller.follower_mapper.relations}",
            flush=True,
        )
        print(
            "RUN_PINCH_GRASP_DOF_NAMES " + ", ".join(list(robot.dof_names)), flush=True
        )

        try:
            if args.no_grasp:
                # 仅做导入和控制器 smoke test，不构造 pinch TCP，也不调用 cuMotion。
                hold_initial_pose(
                    robot,
                    world,
                    ArticulationAction,
                    controller,
                    simulation_app if args.hold else None,
                    args.gui,
                    logger,
                )
                print("RUN_PINCH_GRASP_HOLD_OK", flush=True)
            else:
                result = run_pinch_grasp_action(
                    robot=robot,
                    world=world,
                    articulation_action_type=ArticulationAction,
                    controller=controller,
                    simulation_app=simulation_app,
                    render=args.gui,
                    rope_config=rope_config,
                    mjcf_path=asset_path,
                    cumotion_config=robot_cumotion,
                    motion_planner_config=motion_planner_config,
                    endpoint=endpoint,
                    short_smoke=args.short_smoke,
                    drive_logger=logger,
                )
                print(
                    "RUN_PINCH_GRASP_OK "
                    f"steps={result['steps']} ik={result['ik']} log={log_path}",
                    flush=True,
                )
        finally:
            # 无论动作成功、失败还是用户 Ctrl+C，都尽量关闭 CSV 文件，避免最后几行日志丢失。
            logger.close()
    finally:
        # 必须关闭 SimulationApp，否则 Kit/Isaac 进程和扩展资源可能残留。
        simulation_app.close()


if __name__ == "__main__":
    main()
