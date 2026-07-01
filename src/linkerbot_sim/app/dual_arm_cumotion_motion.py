"""双臂 cuMotion 通用动作执行封装。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from linkerbot_sim.app.cumotion_motion_specs import (
    CSpaceDeltaPlanMoveSpec,
    CumotionMoveSpec,
    DualArmTcpSpec,
    IkOffsetMoveSpec,
    MoveSpec,
    SpecifiedPathMoveSpec,
    default_move_phase,
    normalize_move_sequence,
    side_tcp_frame_name,
    specified_path_planner_config,
    tcp_transforms_from_dual_spec,
)
from linkerbot_sim.app.cumotion_motion_runtime import (
    command_values_by_name,
    cspace_goal_to_command_vector,
    cspace_linear_trajectory,
    cspace_trajectory_from_motion_result,
    current_command_from_runtime,
    cumotion_boundary,
    duration_for_move,
    explicit_tcp_frame_name,
    phase_for_move,
    solve_ik_request,
)
from linkerbot_sim.app.dual_robot_runtime import DualRobotAppRuntime
from linkerbot_sim.assets.robot_loader import dual_robot_root_poses_from_env_config
from linkerbot_sim.backends.cumotion.profile_config import (
    merged_robot_config_with_cumotion_profile,
    motion_planner_config_from_profile,
    robot_cumotion_config,
)
from linkerbot_sim.backends.cumotion.tcp_context import make_cumotion_context
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.execution.dual_runtime import DualRobotRuntime, RobotSideRuntime
from linkerbot_sim.execution.dual_steps import (
    DualCommandPositionTargetStep,
    DualCommandPositionTrajectoryStep,
)
from linkerbot_sim.planning.dual_arm_cspace_partition import (
    DualArmJointPartitions,
    selected_side_goal,
    split_dual_arm_trajectory_to_commands,
)
from linkerbot_sim.planning.requests import IKRequest, MotionRequest, SpecifiedPathRequest
from linkerbot_sim.trajectories.types import JointTrajectory
from linkerbot_sim.utils.config import load_yaml


DEFAULT_HOLD_REFRESH_DURATION_S = 0.25


@dataclass(frozen=True)
class DualArmCuMotionSummary:
    """双臂 cuMotion 配置摘要，供脚本打印和 dry-run 检查。"""

    cumotion_profile: str
    dual_arm_profile: str
    planning_pipeline: str
    left_tcp: str
    right_tcp: str


@dataclass(frozen=True)
class IkMotionGoal:
    """一次双臂 context 中的 IK 目标摘要。"""

    side: str
    tcp_frame_name: str
    goal_q: np.ndarray
    target_position: np.ndarray
    position_error: float
    orientation_error: float | None


@dataclass(frozen=True)
class _DualMoveExecutionResult:
    step: int
    cspace_q: np.ndarray
    left_command: np.ndarray
    right_command: np.ndarray


def dual_arm_cumotion_summary(
    *,
    cumotion_profile: str,
    dual_arm_profile: str,
    tcp: DualArmTcpSpec | None = None,
) -> DualArmCuMotionSummary:
    """加载 profile 并返回不创建后端 context 的 cuMotion 摘要。"""

    cumotion_config = load_profile_yaml("cumotion", cumotion_profile)
    motion_config = motion_planner_config_from_profile(cumotion_config)
    dual_arm = load_dual_arm_semantic_config(dual_arm_profile)
    return DualArmCuMotionSummary(
        cumotion_profile=cumotion_profile,
        dual_arm_profile=dual_arm_profile,
        planning_pipeline=motion_config.planning_pipeline,
        left_tcp=(
            tcp.left.frame_name
            if tcp is not None
            else _side_tcp_frame_name(dual_arm, "left")
        ),
        right_tcp=(
            tcp.right.frame_name
            if tcp is not None
            else _side_tcp_frame_name(dual_arm, "right")
        ),
    )


def run_dual_arm_cumotion_motion(
    runtime: DualRobotAppRuntime,
    *,
    tcp: DualArmTcpSpec,
    moves: Sequence[MoveSpec],
    start_step: int = 0,
    cumotion_profile: str = "default",
    dual_arm_profile: str = "ar5v2_l6v1_dual",
) -> int:
    """执行客户传入的双臂 cuMotion move 序列，并返回累计 physics step。"""

    tcp.validate()
    move_specs = normalize_move_sequence(moves)

    execution = runtime.execution
    physics_dt = float(execution.simulation_world.get_physics_dt())
    sample_dt = max(physics_dt, 1.0e-6)
    left_command = current_command(execution.left)
    right_command = current_command(execution.right)

    cumotion_profile_data = load_profile_yaml("cumotion", cumotion_profile)
    robot_config = merged_robot_config_with_cumotion_profile(
        runtime.robot_config,
        cumotion_profile_data,
    )
    cumotion_config = robot_cumotion_config(
        robot_config,
        dual_root_poses=dual_robot_root_poses_from_env_config(runtime.env_config),
    )
    motion_planner_config = motion_planner_config_from_profile(cumotion_profile_data)
    dual_arm = load_dual_arm_semantic_config(dual_arm_profile)
    tcp_transforms = tcp_transforms_from_dual_spec(tcp)
    tcp_parent_frames = {
        tcp.left.frame_name: _side_flange_frame(dual_arm, "left"),
        tcp.right.frame_name: _side_flange_frame(dual_arm, "right"),
    }

    step = int(start_step)
    with make_cumotion_context(
        cumotion_config,
        tcp=tcp_transforms,
        tcp_parent_frames=tcp_parent_frames,
    ) as context:
        context.clear_collision_world()
        joint_names = tuple(context.joint_names())
        current_q = current_dual_cspace_command(execution, joint_names)
        partitions = DualArmJointPartitions.from_joint_names(
            joint_names,
            left_joint_names=_side_arm_joints(dual_arm, "left"),
            right_joint_names=_side_arm_joints(dual_arm, "right"),
        )

        for move_index, move in enumerate(move_specs, start=1):
            result = _run_dual_move(
                move,
                move_index=move_index,
                context=context,
                partitions=partitions,
                current_q=current_q,
                joint_names=joint_names,
                runtime=execution,
                left_command=left_command,
                right_command=right_command,
                step=step,
                sample_dt=sample_dt,
                motion_planner_config=motion_planner_config,
                tcp=tcp,
            )
            step = result.step
            current_q = result.cspace_q
            left_command = result.left_command
            right_command = result.right_command

    return step


def hold_dual_current_pose(
    runtime: DualRobotRuntime,
    *,
    step: int,
    simulation_app,
    refresh_duration_s: float = DEFAULT_HOLD_REFRESH_DURATION_S,
) -> int:
    """GUI 调试时在完整动作结束后保持当前左右 command target。"""

    while simulation_app.is_running():
        current_left = current_command(runtime.left)
        current_right = current_command(runtime.right)
        step = DualCommandPositionTargetStep(
            left_start_command=current_left,
            right_start_command=current_right,
            left_target_command=current_left,
            right_target_command=current_right,
            duration=refresh_duration_s,
            phase="dual_hold",
        ).run(runtime, step)
    return step


def current_command(side_runtime: RobotSideRuntime) -> np.ndarray:
    """读取 articulation 当前关节位置，并投影到 controller command-space。"""

    return current_command_from_runtime(side_runtime)


def current_dual_cspace_command(
    runtime: DualRobotRuntime,
    joint_names: Sequence[str],
) -> np.ndarray:
    """按 cuMotion C-space 关节名拼出当前双臂关节向量。"""

    return dual_cspace_vector_from_side_commands(
        joint_names=joint_names,
        left_command_joint_names=runtime.left.joint_controller.command_joint_names,
        right_command_joint_names=runtime.right.joint_controller.command_joint_names,
        left_command=current_command(runtime.left),
        right_command=current_command(runtime.right),
    )


def dual_cspace_vector_from_side_commands(
    *,
    joint_names: Sequence[str],
    left_command_joint_names: Sequence[str],
    right_command_joint_names: Sequence[str],
    left_command: np.ndarray,
    right_command: np.ndarray,
) -> np.ndarray:
    """把左右 controller command-space 向量按 C-space 关节名重排。"""

    values_by_name = command_values_by_name(
        left_command_joint_names,
        left_command,
        label="left",
    )
    values_by_name.update(
        command_values_by_name(
            right_command_joint_names,
            right_command,
            label="right",
        )
    )
    missing = [str(name) for name in joint_names if str(name) not in values_by_name]
    if missing:
        raise ValueError(
            f"cuMotion C-space joints are missing from dual command-space: {missing}"
        )
    return np.asarray([values_by_name[str(name)] for name in joint_names], dtype=float)


def dual_cspace_goal_to_command(
    *,
    side_runtime: RobotSideRuntime,
    base_command: np.ndarray,
    joint_names: Sequence[str],
    goal_q: np.ndarray,
) -> np.ndarray:
    """把融合 C-space goal 中属于某侧的机械臂关节写回该侧 command-space。"""

    return cspace_goal_to_command_vector(
        command_joint_names=side_runtime.joint_controller.command_joint_names,
        base_command=base_command,
        joint_names=joint_names,
        goal_q=goal_q,
    )


def load_dual_arm_semantic_config(profile: str) -> Mapping[str, object]:
    """读取 ``configs/dual_arm/<profile>.yaml`` 的 ``dual_arm`` 分组。"""

    config = load_yaml(Path("configs") / "dual_arm" / f"{profile}.yaml")
    dual_arm = config.get("dual_arm")
    if not isinstance(dual_arm, Mapping):
        raise ValueError(f"dual arm profile {profile!r} must contain dual_arm mapping")
    for side in ("left", "right"):
        if not isinstance(dual_arm.get(side), Mapping):
            raise ValueError(f"dual arm profile {profile!r} missing {side} mapping")
    return dual_arm


def solve_side_ik_goal(
    *,
    context,
    partitions: DualArmJointPartitions,
    current_q: np.ndarray,
    side: str,
    tcp_frame_name: str,
    offset: np.ndarray,
) -> IkMotionGoal:
    """用当前 TCP 位姿加偏移构造单侧 IK goal，另一侧 C-space 保持不动。"""

    current = np.asarray(current_q, dtype=float).reshape(-1)
    fk = context.make_forward_kinematics()
    current_pose = fk.compute_pose(current, tcp_frame_name)
    target_position = (
        np.asarray(current_pose.position, dtype=float).reshape(3)
        + np.asarray(offset, dtype=float).reshape(3)
    )
    ik_defaults = context.config.kinematics.ik
    request = IKRequest(
        target_position=target_position,
        target_orientation=current_pose.orientation,
        tcp_frame_name=tcp_frame_name,
        warm_start_ik_cspace_seed=current,
        position_tolerance=ik_defaults.position_tolerance,
        orientation_tolerance=ik_defaults.orientation_tolerance,
        avoid_collisions=False,
    )
    result = solve_ik_request(
        context,
        request,
        tcp_frame_name=tcp_frame_name,
        label="dual-arm",
    )
    goal_q = selected_side_goal(
        base_q=current,
        solved_q=result.joint_positions,
        partitions=partitions,
        active_side=side,
    )
    return IkMotionGoal(
        side=side,
        tcp_frame_name=tcp_frame_name,
        goal_q=goal_q,
        target_position=target_position,
        position_error=float(result.position_error),
        orientation_error=None
        if result.orientation_error is None
        else float(result.orientation_error),
    )


def side_joint_delta_goal(
    *,
    base_q: np.ndarray,
    partitions: DualArmJointPartitions,
    side: str,
    deltas: Sequence[float],
) -> np.ndarray:
    """在融合 C-space 中只对指定侧机械臂叠加关节扰动。"""

    goal = np.asarray(base_q, dtype=float).reshape(-1).copy()
    active = partitions.active_indices(side)
    for index, delta in zip(active, deltas):
        goal[int(index)] += float(delta)
    return goal


def dual_cspace_linear_trajectory(
    *,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """为 IK 动作构造一条简单 C-space 插值轨迹。"""

    return cspace_linear_trajectory(
        start_q=start_q,
        goal_q=goal_q,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )


def dual_cspace_trajectory_from_motion_result(
    result,
    *,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """把 cuMotion MotionResult 转成可拆分的双臂 C-space 轨迹。"""

    return cspace_trajectory_from_motion_result(
        result,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )


def selected_side_dual_trajectory(
    *,
    trajectory: JointTrajectory,
    base_q: np.ndarray,
    partitions: DualArmJointPartitions,
    active_side: str,
) -> JointTrajectory:
    """只保留选定侧 C-space 轨迹，另一侧保持 ``base_q``。"""

    active = set(int(index) for index in partitions.active_indices(active_side))
    all_indices = set(range(len(partitions.joint_names)))
    inactive = np.asarray(sorted(all_indices - active), dtype=int)
    base = np.asarray(base_q, dtype=float).reshape(-1)
    if base.size != len(partitions.joint_names):
        raise ValueError(
            f"base_q expected {len(partitions.joint_names)} values, got {base.size}"
        )
    positions = trajectory.positions.copy()
    velocities = trajectory.velocities.copy()
    accelerations = trajectory.accelerations.copy()
    jerks = trajectory.jerks.copy()
    efforts = trajectory.efforts.copy()
    if inactive.size:
        positions[:, inactive] = base[inactive]
        velocities[:, inactive] = 0.0
        accelerations[:, inactive] = 0.0
        jerks[:, inactive] = 0.0
        efforts[:, inactive] = 0.0
    return JointTrajectory.from_samples(
        times=trajectory.times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        efforts=efforts,
        phases=trajectory.phases,
        joint_names=trajectory.joint_names,
    )


def _run_dual_move(
    move: MoveSpec,
    *,
    move_index: int,
    context,
    partitions: DualArmJointPartitions,
    current_q: np.ndarray,
    joint_names: Sequence[str],
    runtime: DualRobotRuntime,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    motion_planner_config,
    tcp: DualArmTcpSpec,
) -> _DualMoveExecutionResult:
    execution_mode = _dual_execution_mode(move)
    side = None if execution_mode == "dual_cspace" else _move_side_required(move)
    tcp_frame_name = _move_tcp_frame_name(move, tcp=tcp, side=side)
    duration_s = duration_for_move(move)
    phase = phase_for_move(
        move,
        side=side,
        dual_cspace=execution_mode == "dual_cspace",
    )

    if isinstance(move, IkOffsetMoveSpec):
        move.validate(require_side=True)
        ik_goal = solve_side_ik_goal(
            context=context,
            partitions=partitions,
            current_q=current_q,
            side=side,
            tcp_frame_name=tcp_frame_name,
            offset=np.asarray(move.tcp_offset, dtype=float),
        )
        trajectory = dual_cspace_linear_trajectory(
            start_q=current_q,
            goal_q=ik_goal.goal_q,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
        )
        result = _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
        )
        print(
            "DUAL_ARM_CUMOTION_IK_OK "
            f"move={move_index} side={ik_goal.side} tcp={ik_goal.tcp_frame_name} "
            f"phase={phase} position_error={ik_goal.position_error:.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return result

    if isinstance(move, CumotionMoveSpec) and isinstance(move.request, IKRequest):
        move.validate(require_side=execution_mode != "dual_cspace")
        request = _ik_request_with_runtime_defaults(
            move.request,
            current_q=current_q,
            tcp_frame_name=tcp_frame_name,
        )
        ik_result = solve_ik_request(
            context,
            request,
            tcp_frame_name=tcp_frame_name,
            label="dual-arm",
        )
        goal_q = (
            np.asarray(ik_result.joint_positions, dtype=float).reshape(-1)
            if execution_mode == "dual_cspace"
            else selected_side_goal(
                base_q=current_q,
                solved_q=ik_result.joint_positions,
                partitions=partitions,
                active_side=side,
            )
        )
        trajectory = dual_cspace_linear_trajectory(
            start_q=current_q,
            goal_q=goal_q,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
        )
        result = _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
        )
        print(
            "DUAL_ARM_CUMOTION_IK_OK "
            f"move={move_index} side={side} tcp={tcp_frame_name} phase={phase} "
            f"position_error={float(ik_result.position_error):.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return result

    if isinstance(move, CSpaceDeltaPlanMoveSpec):
        move.validate(require_side=True)
        goal_q = side_joint_delta_goal(
            base_q=current_q,
            partitions=partitions,
            side=side,
            deltas=move.joint_deltas,
        )
        request = MotionRequest(
            current_q=current_q,
            goal_q=goal_q,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        trajectory = _plan_dual_motion_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=motion_planner_config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
            side=side,
            execution_mode=execution_mode,
        )
        trajectory = selected_side_dual_trajectory(
            trajectory=trajectory,
            base_q=current_q,
            partitions=partitions,
            active_side=side,
        )
        return _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
        )

    if isinstance(move, SpecifiedPathMoveSpec):
        move.validate(require_side=True)
        request = SpecifiedPathRequest(
            current_q=current_q,
            path=move.path,
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
        )
        specified_config = specified_path_planner_config(
            motion_planner_config,
            path=move.path,
        )
        trajectory = _plan_dual_motion_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=specified_config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
            side=side,
            execution_mode=execution_mode,
        )
        trajectory = selected_side_dual_trajectory(
            trajectory=trajectory,
            base_q=current_q,
            partitions=partitions,
            active_side=side,
        )
        return _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
        )

    if isinstance(move, CumotionMoveSpec):
        move.validate(require_side=execution_mode != "dual_cspace")
        request = _planning_request_with_runtime_defaults(
            move.request,
            duration_s=duration_s,
            tcp_frame_name=tcp_frame_name,
        )
        config = (
            specified_path_planner_config(motion_planner_config, path=request.path)
            if isinstance(request, SpecifiedPathRequest)
            else motion_planner_config
        )
        trajectory = _plan_dual_motion_trajectory(
            context=context,
            request=request,
            tcp_frame_name=tcp_frame_name,
            config=config,
            joint_names=joint_names,
            duration_s=duration_s,
            sample_dt=sample_dt,
            phase=phase,
            move_index=move_index,
            side=side,
            execution_mode=execution_mode,
        )
        if execution_mode != "dual_cspace":
            trajectory = selected_side_dual_trajectory(
                trajectory=trajectory,
                base_q=current_q,
                partitions=partitions,
                active_side=side,
            )
        return _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
        )

    raise TypeError(f"unsupported move spec type: {type(move).__name__}")


def _plan_dual_motion_trajectory(
    *,
    context,
    request: MotionRequest | SpecifiedPathRequest,
    tcp_frame_name: str,
    config,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
    move_index: int,
    side: str | None,
    execution_mode: str,
) -> JointTrajectory:
    planner = context.make_motion_planner(
        tcp_frame_name=tcp_frame_name,
        config=config,
    )
    print(
        "DUAL_ARM_CUMOTION_PLAN_START "
        f"move={move_index} side={side} tcp={tcp_frame_name} phase={phase} "
        f"type={type(request).__name__} execution={execution_mode} "
        f"pipeline={config.planning_pipeline}",
        flush=True,
    )
    result = cumotion_boundary("motion planner", planner.plan, request)
    if not result.success:
        raise RuntimeError(
            "cuMotion dual-arm path planning failed: "
            f"phase={phase} status={result.status} "
            f"message={result.diagnostics.message}"
        )
    trajectory = dual_cspace_trajectory_from_motion_result(
        result,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )
    metrics = result.diagnostics.metrics
    print(
        "DUAL_ARM_CUMOTION_PLAN_OK "
        f"move={move_index} side={side} tcp={tcp_frame_name} phase={phase} "
        f"pipeline={config.planning_pipeline} status={result.status} "
        f"samples={len(trajectory)} "
        f"path_waypoints={int(metrics.get('num_waypoints', 0.0))} "
        f"path_length={metrics.get('path_length', 0.0):.6g}",
        flush=True,
    )
    return trajectory


def _execute_dual_cspace_trajectory(
    *,
    runtime: DualRobotRuntime,
    partitions: DualArmJointPartitions,
    trajectory: JointTrajectory,
    joint_names: Sequence[str],
    left_start_command: np.ndarray,
    right_start_command: np.ndarray,
    step: int,
    phase: str,
) -> _DualMoveExecutionResult:
    if len(trajectory) == 0:
        raise ValueError(f"trajectory for {phase!r} cannot be empty")
    goal_q = np.asarray(trajectory.positions[-1], dtype=float).reshape(-1)
    left_target = dual_cspace_goal_to_command(
        side_runtime=runtime.left,
        base_command=left_start_command,
        joint_names=joint_names,
        goal_q=goal_q,
    )
    right_target = dual_cspace_goal_to_command(
        side_runtime=runtime.right,
        base_command=right_start_command,
        joint_names=joint_names,
        goal_q=goal_q,
    )
    left_trajectory, right_trajectory = split_dual_arm_trajectory_to_commands(
        dual_arm_trajectory=trajectory,
        partitions=partitions,
        left_command_joint_names=runtime.left.joint_controller.command_joint_names,
        right_command_joint_names=runtime.right.joint_controller.command_joint_names,
        left_start_command=left_start_command,
        right_start_command=right_start_command,
        left_target_command=left_target,
        right_target_command=right_target,
        phase=phase,
    )
    step = DualCommandPositionTrajectoryStep(
        left_trajectory=left_trajectory,
        right_trajectory=right_trajectory,
        phase=phase,
    ).run(runtime, step)
    return _DualMoveExecutionResult(
        step=step,
        cspace_q=goal_q,
        left_command=np.asarray(left_trajectory.positions[-1], dtype=float),
        right_command=np.asarray(right_trajectory.positions[-1], dtype=float),
    )


def _ik_request_with_runtime_defaults(
    request: IKRequest,
    *,
    current_q: np.ndarray,
    tcp_frame_name: str,
) -> IKRequest:
    return replace(
        request,
        tcp_frame_name=request.tcp_frame_name or tcp_frame_name,
        warm_start_ik_cspace_seed=(
            request.warm_start_ik_cspace_seed
            if request.warm_start_ik_cspace_seed is not None
            else np.asarray(current_q, dtype=float).reshape(-1)
        ),
    )


def _planning_request_with_runtime_defaults(
    request: MotionRequest | SpecifiedPathRequest,
    *,
    duration_s: float,
    tcp_frame_name: str,
) -> MotionRequest | SpecifiedPathRequest:
    return replace(
        request,
        tcp_frame_name=request.tcp_frame_name or tcp_frame_name,
        duration_s=request.duration_s if request.duration_s is not None else duration_s,
    )


def _dual_execution_mode(move: MoveSpec) -> str:
    if isinstance(move, CumotionMoveSpec) and move.execution == "dual_cspace":
        return "dual_cspace"
    return "selected_side"


def _move_side_required(move: MoveSpec) -> str:
    side = getattr(move, "side", None)
    if side is None:
        raise ValueError(f"{type(move).__name__} requires side in dual-arm runtime")
    return _normalize_side(side)


def _move_tcp_frame_name(
    move: MoveSpec,
    *,
    tcp: DualArmTcpSpec,
    side: str | None,
) -> str:
    value = explicit_tcp_frame_name(move)
    if value is not None:
        return value
    if side is not None:
        return side_tcp_frame_name(tcp, side)
    raise ValueError(
        f"{type(move).__name__} requires tcp_frame_name when side is not set"
    )


def _side_dual_arm_config(
    dual_arm: Mapping[str, object], side: str
) -> Mapping[str, object]:
    side_config = dual_arm.get(side)
    if not isinstance(side_config, Mapping):
        raise ValueError(f"dual_arm.{side} must be a mapping")
    required = {"arm_joints"}
    missing = sorted(required - set(side_config))
    if missing:
        raise ValueError(f"dual_arm.{side} missing required keys: {missing}")
    return side_config


def _side_arm_joints(dual_arm: Mapping[str, object], side: str) -> tuple[str, ...]:
    value = _side_dual_arm_config(dual_arm, side)["arm_joints"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"dual_arm.{side}.arm_joints must be a sequence")
    joints = tuple(str(name) for name in value)
    if not joints:
        raise ValueError(f"dual_arm.{side}.arm_joints cannot be empty")
    return joints


def _side_tcp_frame_name(dual_arm: Mapping[str, object], side: str) -> str:
    side_config = _side_dual_arm_config(dual_arm, side)
    if "tcp_frame" not in side_config:
        raise ValueError(f"dual_arm.{side}.tcp_frame cannot be empty")
    value = str(side_config["tcp_frame"])
    if not value:
        raise ValueError(f"dual_arm.{side}.tcp_frame cannot be empty")
    return value


def _side_flange_frame(dual_arm: Mapping[str, object], side: str) -> str:
    side_config = _side_dual_arm_config(dual_arm, side)
    if "flange_frame" not in side_config:
        raise ValueError(f"dual_arm.{side}.flange_frame cannot be empty")
    value = str(side_config["flange_frame"])
    if not value:
        raise ValueError(f"dual_arm.{side}.flange_frame cannot be empty")
    return value


def _normalize_side(side: str) -> str:
    normalized = str(side).lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return normalized
