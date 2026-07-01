"""双臂 cuMotion 通用动作执行封装。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from linkerbot_sim.app.motion.specs import (
    CSpaceDeltaPlanMoveSpec,
    CSpaceGoalPlanMoveSpec,
    CommandOverlaySpec,
    CumotionMoveSpec,
    DualHandMoveSpec,
    DualArmTcpSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    MoveSpec,
    RawJointSequenceMoveSpec,
    RawJointSequenceSideSpec,
    SpecifiedPathMoveSpec,
    default_move_phase,
    normalize_move_sequence,
    side_tcp_frame_name,
    specified_path_planner_config,
    tcp_transforms_from_dual_spec,
)
from linkerbot_sim.app.motion.runtime import (
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
from linkerbot_sim.app.runtime.dual_robot import DualRobotAppRuntime
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
    DualCommandExecutionInterrupted,
    DualCommandPositionTargetStep,
    DualCommandPositionTrajectoryStep,
    DualRawCommandTargetSequenceStep,
)
from linkerbot_sim.planning.dual_arm_cspace_partition import (
    DualArmJointPartitions,
    selected_side_goal,
    split_dual_arm_trajectory_to_commands,
)
from linkerbot_sim.planning.requests import IKRequest, MotionRequest, SpecifiedPathRequest
from linkerbot_sim.robots.classification import component_for_name
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
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
    """单个双臂 move 执行后的滚动状态，用于串联后续 move。"""

    step: int
    cspace_q: np.ndarray
    left_command: np.ndarray
    right_command: np.ndarray


class DualArmCuMotionExecutionSession:
    """Long-lived dual-arm cuMotion execution context."""

    def __init__(
        self,
        runtime: DualRobotAppRuntime,
        *,
        tcp: DualArmTcpSpec,
        cumotion_profile: str = "default",
        dual_arm_profile: str = "ar5v2_l6v1_dual",
    ) -> None:
        """加载双臂 cuMotion context、规划配置和左右 C-space 分区。"""

        tcp.validate()
        self.runtime = runtime
        self.execution = runtime.execution
        self.tcp = tcp
        self.cumotion_profile = str(cumotion_profile)
        self.dual_arm_profile = str(dual_arm_profile)

        physics_dt = float(self.execution.simulation_world.get_physics_dt())
        self.sample_dt = max(physics_dt, 1.0e-6)

        cumotion_profile_data = load_profile_yaml("cumotion", self.cumotion_profile)
        robot_config = merged_robot_config_with_cumotion_profile(
            runtime.robot_config,
            cumotion_profile_data,
        )
        cumotion_config = robot_cumotion_config(
            robot_config,
            dual_root_poses=dual_robot_root_poses_from_env_config(
                runtime.env_config
            ),
        )
        self.motion_planner_config = motion_planner_config_from_profile(
            cumotion_profile_data
        )
        self.dual_arm = load_dual_arm_semantic_config(self.dual_arm_profile)
        tcp_transforms = tcp_transforms_from_dual_spec(tcp)
        tcp_parent_frames = {
            tcp.left.frame_name: _side_flange_frame(self.dual_arm, "left"),
            tcp.right.frame_name: _side_flange_frame(self.dual_arm, "right"),
        }
        self._context_manager = make_cumotion_context(
            cumotion_config,
            tcp=tcp_transforms,
            tcp_parent_frames=tcp_parent_frames,
        )
        self.context = self._context_manager.__enter__()
        self._closed = False
        try:
            self.context.clear_collision_world()
            self.joint_names = tuple(self.context.joint_names())
            self.partitions = DualArmJointPartitions.from_joint_names(
                self.joint_names,
                left_joint_names=_side_arm_joints(self.dual_arm, "left"),
                right_joint_names=_side_arm_joints(self.dual_arm, "right"),
            )
            self.step = 0
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """释放长期持有的 cuMotion context manager。"""

        if self._closed:
            return
        self._closed = True
        self._context_manager.__exit__(None, None, None)

    def __enter__(self) -> "DualArmCuMotionExecutionSession":
        """让 session 可用于 with 语句，复用已加载的 cuMotion context。"""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """退出 with 语句时关闭 cuMotion context。"""

        self.close()

    def execute_moves(
        self,
        moves: Sequence[MoveSpec],
        *,
        start_step: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        """Execute a move sequence using the already loaded cuMotion context."""

        move_specs = normalize_move_sequence(moves)
        step = self.step if start_step is None else int(start_step)
        current_q, left_command, right_command = self.refresh_current_state()

        for move_index, move in enumerate(move_specs, start=1):
            _raise_if_requested_stop(should_stop)
            result = _run_dual_move(
                move,
                move_index=move_index,
                context=self.context,
                partitions=self.partitions,
                current_q=current_q,
                joint_names=self.joint_names,
                runtime=self.execution,
                left_command=left_command,
                right_command=right_command,
                step=step,
                sample_dt=self.sample_dt,
                motion_planner_config=self.motion_planner_config,
                tcp=self.tcp,
                should_stop=should_stop,
            )
            step = result.step
            current_q = result.cspace_q
            left_command = result.left_command
            right_command = result.right_command

        self.step = step
        return step

    def refresh_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read current command-space and merged C-space vectors from Isaac."""

        left_command = current_command(self.execution.left)
        right_command = current_command(self.execution.right)
        current_q = dual_cspace_vector_from_side_commands(
            joint_names=self.joint_names,
            left_command_joint_names=(
                self.execution.left.joint_controller.command_joint_names
            ),
            right_command_joint_names=(
                self.execution.right.joint_controller.command_joint_names
            ),
            left_command=left_command,
            right_command=right_command,
        )
        return current_q, left_command, right_command


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

    with DualArmCuMotionExecutionSession(
        runtime,
        tcp=tcp,
        cumotion_profile=cumotion_profile,
        dual_arm_profile=dual_arm_profile,
    ) as session:
        return session.execute_moves(moves, start_step=start_step)


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
    orientation_mode: str = "current",
    target_orientation: np.ndarray | None = None,
) -> IkMotionGoal:
    """用当前 TCP 位姿加偏移构造单侧 IK goal，另一侧 C-space 保持不动。"""

    current = np.asarray(current_q, dtype=float).reshape(-1)
    fk = context.make_forward_kinematics()
    current_pose = fk.compute_pose(current, tcp_frame_name)
    target_position = (
        np.asarray(current_pose.position, dtype=float).reshape(3)
        + np.asarray(offset, dtype=float).reshape(3)
    )
    target_orientation_value = _target_orientation_for_mode(
        mode=orientation_mode,
        current_orientation=current_pose.orientation,
        target_orientation=target_orientation,
    )
    ik_defaults = context.config.kinematics.ik
    request = IKRequest(
        target_position=target_position,
        target_orientation=target_orientation_value,
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


def side_joint_absolute_goal(
    *,
    base_q: np.ndarray,
    partitions: DualArmJointPartitions,
    side: str,
    joint_positions: Sequence[float],
) -> np.ndarray:
    """在融合 C-space 中只对指定侧机械臂写入绝对关节角目标。"""

    goal = np.asarray(base_q, dtype=float).reshape(-1).copy()
    active = partitions.active_indices(side)
    for index, position in zip(active, joint_positions):
        goal[int(index)] = float(position)
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
    should_stop: Callable[[], bool] | None = None,
) -> _DualMoveExecutionResult:
    """分派并执行一个双臂 move，统一维护 C-space 和左右 command-space 状态。"""

    if isinstance(move, HandMoveSpec):
        move.validate()
        return _execute_hand_move(
            move,
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            should_stop=should_stop,
        )

    if isinstance(move, DualHandMoveSpec):
        move.validate()
        return _execute_dual_hand_move(
            move,
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            should_stop=should_stop,
        )

    if isinstance(move, RawJointSequenceMoveSpec):
        move.validate()
        return _execute_raw_joint_sequence_move(
            move,
            runtime=runtime,
            joint_names=joint_names,
            left_command=left_command,
            right_command=right_command,
            step=step,
            should_stop=should_stop,
        )

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
        step, left_command, right_command = _execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        ik_goal = solve_side_ik_goal(
            context=context,
            partitions=partitions,
            current_q=current_q,
            side=side,
            tcp_frame_name=tcp_frame_name,
            offset=np.asarray(move.tcp_offset, dtype=float),
            orientation_mode=move.orientation_mode,
            target_orientation=(
                None
                if move.target_orientation is None
                else np.asarray(move.target_orientation, dtype=float)
            ),
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
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        print(
            "DUAL_ARM_CUMOTION_IK_OK "
            f"move={move_index} side={ik_goal.side} tcp={ik_goal.tcp_frame_name} "
            f"phase={phase} position_error={ik_goal.position_error:.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return _finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=ik_goal.goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CumotionMoveSpec) and isinstance(move.request, IKRequest):
        move.validate(require_side=execution_mode != "dual_cspace")
        step, left_command, right_command = _execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
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
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        print(
            "DUAL_ARM_CUMOTION_IK_OK "
            f"move={move_index} side={side} tcp={tcp_frame_name} phase={phase} "
            f"position_error={float(ik_result.position_error):.6g} "
            f"samples={len(trajectory)}",
            flush=True,
        )
        return _finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CSpaceGoalPlanMoveSpec):
        move.validate(require_side=True)
        step, left_command, right_command = _execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        goal_q = side_joint_absolute_goal(
            base_q=current_q,
            partitions=partitions,
            side=side,
            joint_positions=move.joint_positions,
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
        result = _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return _finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CSpaceDeltaPlanMoveSpec):
        move.validate(require_side=True)
        step, left_command, right_command = _execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
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
        result = _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return _finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=goal_q,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, SpecifiedPathMoveSpec):
        move.validate(require_side=True)
        step, left_command, right_command = _execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
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
        result = _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return _finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=np.asarray(trajectory.positions[-1], dtype=float),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )

    if isinstance(move, CumotionMoveSpec):
        move.validate(require_side=execution_mode != "dual_cspace")
        step, left_command, right_command = _execute_overlays(
            tuple(move.overlays),
            timing="before",
            runtime=runtime,
            current_q=current_q,
            left_command=left_command,
            right_command=right_command,
            step=step,
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
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
        result = _execute_dual_cspace_trajectory(
            runtime=runtime,
            partitions=partitions,
            trajectory=trajectory,
            joint_names=joint_names,
            left_start_command=left_command,
            right_start_command=right_command,
            step=step,
            phase=phase,
            overlays=tuple(move.overlays),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
        )
        return _finish_arm_move_with_after_overlays(
            result,
            overlays=tuple(move.overlays),
            runtime=runtime,
            current_q=np.asarray(trajectory.positions[-1], dtype=float),
            sample_dt=sample_dt,
            default_duration_s=duration_s,
            should_stop=should_stop,
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
    """调用双臂 cuMotion planner，并把结果转成融合 C-space 轨迹。"""

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
    overlays: Sequence[CommandOverlaySpec] = (),
    sample_dt: float = 1.0 / 60.0,
    default_duration_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> _DualMoveExecutionResult:
    """把融合 C-space 轨迹拆成左右 command-space 轨迹并执行。"""

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
    left_trajectory, right_trajectory = _apply_sync_overlays_to_trajectories(
        tuple(overlays),
        left_trajectory=left_trajectory,
        right_trajectory=right_trajectory,
        runtime=runtime,
        left_start_command=left_start_command,
        right_start_command=right_start_command,
        sample_dt=sample_dt,
        default_duration_s=default_duration_s,
        phase=phase,
    )
    step = DualCommandPositionTrajectoryStep(
        left_trajectory=left_trajectory,
        right_trajectory=right_trajectory,
        phase=phase,
        should_stop=should_stop,
    ).run(runtime, step)
    return _DualMoveExecutionResult(
        step=step,
        cspace_q=goal_q,
        left_command=np.asarray(left_trajectory.positions[-1], dtype=float),
        right_command=np.asarray(right_trajectory.positions[-1], dtype=float),
    )


def _execute_hand_move(
    move: HandMoveSpec,
    *,
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    should_stop: Callable[[], bool] | None = None,
) -> _DualMoveExecutionResult:
    """把单手 move 包装成 DualHandMoveSpec，复用双手执行路径。"""

    dual = DualHandMoveSpec(
        left=move if _normalize_side(move.side) == "left" else None,
        right=move if _normalize_side(move.side) == "right" else None,
        duration_s=move.duration_s,
        phase=move.phase,
    )
    return _execute_dual_hand_move(
        dual,
        runtime=runtime,
        current_q=current_q,
        left_command=left_command,
        right_command=right_command,
        step=step,
        sample_dt=sample_dt,
        should_stop=should_stop,
    )


def _execute_dual_hand_move(
    move: DualHandMoveSpec,
    *,
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    should_stop: Callable[[], bool] | None = None,
) -> _DualMoveExecutionResult:
    """执行左右手 command-space 线性插值动作，不改变机械臂 C-space。"""

    duration_s = _dual_hand_duration(move)
    phase = phase_for_move(move)
    left_target = (
        _hand_target_command(
            runtime.left,
            start_command=left_command,
            hand=move.left,
        )
        if move.left is not None
        else left_command
    )
    right_target = (
        _hand_target_command(
            runtime.right,
            start_command=right_command,
            hand=move.right,
        )
        if move.right is not None
        else right_command
    )
    left_trajectory = _command_linear_trajectory(
        command_joint_names=runtime.left.joint_controller.command_joint_names,
        start_command=left_command,
        target_command=left_target,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    ) if move.left is not None else None
    right_trajectory = _command_linear_trajectory(
        command_joint_names=runtime.right.joint_controller.command_joint_names,
        start_command=right_command,
        target_command=right_target,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    ) if move.right is not None else None
    step = DualCommandPositionTrajectoryStep(
        left_trajectory=left_trajectory,
        right_trajectory=right_trajectory,
        phase=phase,
        should_stop=should_stop,
    ).run(runtime, step)
    return _DualMoveExecutionResult(
        step=step,
        cspace_q=np.asarray(current_q, dtype=float).reshape(-1),
        left_command=np.asarray(left_target, dtype=float).reshape(-1),
        right_command=np.asarray(right_target, dtype=float).reshape(-1),
    )


def _execute_raw_joint_sequence_move(
    move: RawJointSequenceMoveSpec,
    *,
    runtime: DualRobotRuntime,
    joint_names: Sequence[str],
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    should_stop: Callable[[], bool] | None = None,
) -> _DualMoveExecutionResult:
    """按 physics step 执行调用方提供的原始左右 command target 序列。"""

    phase = move.phase or "raw_joint_sequence"
    left_positions = _raw_sequence_side_matrix(
        move.left,
        side_runtime=runtime.left,
        current_command=left_command,
        label="left",
    )
    right_positions = _raw_sequence_side_matrix(
        move.right,
        side_runtime=runtime.right,
        current_command=right_command,
        label="right",
    )
    _validate_raw_side_sample_counts(left_positions, right_positions)
    step = DualRawCommandTargetSequenceStep(
        left_positions=left_positions,
        right_positions=right_positions,
        step_interval=int(move.step_interval),
        phase=phase,
        should_stop=should_stop,
    ).run(runtime, step)
    next_left = (
        np.asarray(left_positions[-1], dtype=float).reshape(-1)
        if left_positions is not None
        else np.asarray(left_command, dtype=float).reshape(-1)
    )
    next_right = (
        np.asarray(right_positions[-1], dtype=float).reshape(-1)
        if right_positions is not None
        else np.asarray(right_command, dtype=float).reshape(-1)
    )
    next_q = dual_cspace_vector_from_side_commands(
        joint_names=joint_names,
        left_command_joint_names=runtime.left.joint_controller.command_joint_names,
        right_command_joint_names=runtime.right.joint_controller.command_joint_names,
        left_command=next_left,
        right_command=next_right,
    )
    return _DualMoveExecutionResult(
        step=step,
        cspace_q=next_q,
        left_command=next_left,
        right_command=next_right,
    )


def _execute_overlays(
    overlays: Sequence[CommandOverlaySpec],
    *,
    timing: str,
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    left_command: np.ndarray,
    right_command: np.ndarray,
    step: int,
    sample_dt: float,
    default_duration_s: float | None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """执行指定 timing 的手部 overlays，并返回更新后的 step 与左右 command。"""

    current_left = np.asarray(left_command, dtype=float).reshape(-1)
    current_right = np.asarray(right_command, dtype=float).reshape(-1)
    for overlay in overlays:
        if overlay.timing != timing:
            continue
        move = _dual_hand_move_from_overlay(
            overlay,
            duration_s=default_duration_s,
            phase=f"overlay_{timing}",
        )
        result = _execute_dual_hand_move(
            move,
            runtime=runtime,
            current_q=current_q,
            left_command=current_left,
            right_command=current_right,
            step=step,
            sample_dt=sample_dt,
            should_stop=should_stop,
        )
        step = result.step
        current_left = result.left_command
        current_right = result.right_command
    return step, current_left, current_right


def _finish_arm_move_with_after_overlays(
    result: _DualMoveExecutionResult,
    *,
    overlays: Sequence[CommandOverlaySpec],
    runtime: DualRobotRuntime,
    current_q: np.ndarray,
    sample_dt: float,
    default_duration_s: float | None,
    should_stop: Callable[[], bool] | None = None,
) -> _DualMoveExecutionResult:
    """主臂动作完成后执行 after overlays，并把结果合并为新的滚动状态。"""

    step, left_command, right_command = _execute_overlays(
        overlays,
        timing="after",
        runtime=runtime,
        current_q=current_q,
        left_command=result.left_command,
        right_command=result.right_command,
        step=result.step,
        sample_dt=sample_dt,
        default_duration_s=default_duration_s,
        should_stop=should_stop,
    )
    return _DualMoveExecutionResult(
        step=step,
        cspace_q=np.asarray(current_q, dtype=float).reshape(-1),
        left_command=left_command,
        right_command=right_command,
    )


def _apply_sync_overlays_to_trajectories(
    overlays: Sequence[CommandOverlaySpec],
    *,
    left_trajectory: JointTrajectory,
    right_trajectory: JointTrajectory,
    runtime: DualRobotRuntime,
    left_start_command: np.ndarray,
    right_start_command: np.ndarray,
    sample_dt: float,
    default_duration_s: float | None,
    phase: str,
) -> tuple[JointTrajectory, JointTrajectory]:
    """把 sync overlays 的手部轨迹合并进主臂左右 command 轨迹。"""

    left = left_trajectory
    right = right_trajectory
    left_target = np.asarray(left.positions[-1], dtype=float).reshape(-1)
    right_target = np.asarray(right.positions[-1], dtype=float).reshape(-1)
    for overlay in overlays:
        if overlay.timing != "sync":
            continue
        if overlay.left_hand is not None:
            left_target = _hand_target_command(
                runtime.left,
                start_command=left_target,
                hand=_hand_with_default_duration(
                    overlay.left_hand,
                    default_duration_s,
                ),
            )
            left = _overlay_hand_trajectory(
                base=left,
                side_runtime=runtime.left,
                start_command=left_start_command,
                target_command=left_target,
                hand=overlay.left_hand,
                sample_dt=sample_dt,
                default_duration_s=default_duration_s,
                phase=phase,
            )
        if overlay.right_hand is not None:
            right_target = _hand_target_command(
                runtime.right,
                start_command=right_target,
                hand=_hand_with_default_duration(
                    overlay.right_hand,
                    default_duration_s,
                ),
            )
            right = _overlay_hand_trajectory(
                base=right,
                side_runtime=runtime.right,
                start_command=right_start_command,
                target_command=right_target,
                hand=overlay.right_hand,
                sample_dt=sample_dt,
                default_duration_s=default_duration_s,
                phase=phase,
            )
    return left, right


def _overlay_hand_trajectory(
    *,
    base: JointTrajectory,
    side_runtime: RobotSideRuntime,
    start_command: np.ndarray,
    target_command: np.ndarray,
    hand: HandMoveSpec,
    sample_dt: float,
    default_duration_s: float | None,
    phase: str,
) -> JointTrajectory:
    """为一侧手部 overlay 生成 command 轨迹，并覆盖 base 中的手部列。"""

    duration_s = (
        float(hand.duration_s)
        if hand.duration_s is not None
        else float(default_duration_s or base.times[-1])
    )
    hand_trajectory = _command_linear_trajectory(
        command_joint_names=side_runtime.joint_controller.command_joint_names,
        start_command=start_command,
        target_command=target_command,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )
    return _merge_hand_columns(
        base=base,
        overlay=hand_trajectory,
        command_joint_names=side_runtime.joint_controller.command_joint_names,
        hand_joint_names=_hand_command_joint_names(
            side_runtime.joint_controller.command_joint_names
        ),
    )


def _merge_hand_columns(
    *,
    base: JointTrajectory,
    overlay: JointTrajectory,
    command_joint_names: Sequence[str],
    hand_joint_names: Sequence[str],
) -> JointTrajectory:
    """按 base 时间轴重采样 overlay，并只替换手部 command 列。"""

    names = tuple(str(name) for name in command_joint_names)
    hand_indices = [
        index for index, name in enumerate(names) if name in set(hand_joint_names)
    ]
    if not hand_indices:
        return base
    positions = base.positions.copy()
    velocities = base.velocities.copy()
    accelerations = base.accelerations.copy()
    jerks = base.jerks.copy()
    efforts = base.efforts.copy()
    for index in hand_indices:
        for matrix, overlay_matrix in (
            (positions, overlay.positions),
            (velocities, overlay.velocities),
            (accelerations, overlay.accelerations),
            (jerks, overlay.jerks),
            (efforts, overlay.efforts),
        ):
            matrix[:, index] = np.asarray(
                [
                    np.interp(t, overlay.times, overlay_matrix[:, index])
                    for t in base.times
                ],
                dtype=float,
            )
    return JointTrajectory.from_samples(
        times=base.times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        efforts=efforts,
        phases=base.phases,
        joint_names=base.joint_names,
    )


def _hand_target_command(
    side_runtime: RobotSideRuntime,
    *,
    start_command: np.ndarray,
    hand: HandMoveSpec,
) -> np.ndarray:
    """把手部目标写入一侧 command 向量，支持按关节名或按手部顺序输入。"""

    if _normalize_side(hand.side) != side_runtime.side:
        raise ValueError(
            f"hand target side {hand.side!r} does not match runtime side "
            f"{side_runtime.side!r}"
        )
    target = np.asarray(start_command, dtype=float).reshape(-1).copy()
    command_names = tuple(
        str(name) for name in side_runtime.joint_controller.command_joint_names
    )
    if target.size != len(command_names):
        raise ValueError(
            f"{side_runtime.side} command shape mismatch: "
            f"{len(command_names)} names, {target.size} values"
        )
    if isinstance(hand.joint_positions, Mapping):
        index_by_name = {name: index for index, name in enumerate(command_names)}
        for name, value in hand.joint_positions.items():
            key = str(name)
            if key not in index_by_name:
                raise ValueError(
                    f"{side_runtime.side} hand joint {key!r} is not in command-space"
                )
            target[index_by_name[key]] = float(value)
        return target
    hand_names = _hand_command_joint_names(command_names)
    values = np.asarray(hand.joint_positions, dtype=float).reshape(-1)
    if values.size > len(hand_names):
        raise ValueError(
            f"{side_runtime.side} hand target has {values.size} values, "
            f"but only {len(hand_names)} hand command joints exist"
        )
    index_by_name = {name: index for index, name in enumerate(command_names)}
    for name, value in zip(hand_names, values):
        target[index_by_name[name]] = float(value)
    return target


def _raw_sequence_side_matrix(
    side: RawJointSequenceSideSpec | None,
    *,
    side_runtime: RobotSideRuntime,
    current_command: np.ndarray,
    label: str,
) -> np.ndarray | None:
    """把单侧 raw sequence 规范化为 shape=(samples, command_dof) 的矩阵。"""

    if side is None:
        return None
    command_names = tuple(
        str(name) for name in side_runtime.joint_controller.command_joint_names
    )
    base = np.asarray(current_command, dtype=float).reshape(-1)
    if base.size != len(command_names):
        raise ValueError(
            f"{label} command shape mismatch: {len(command_names)} names, {base.size} values"
        )
    positions = side.joint_positions
    if isinstance(positions, Mapping):
        matrix = _raw_mapping_sequence_matrix(
            positions,
            command_joint_names=command_names,
            base_command=base,
            label=label,
        )
    else:
        matrix = np.asarray(positions, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"{label} raw joint sequence must have shape (N, dof)")
        if matrix.shape[1] != len(command_names):
            raise ValueError(
                f"{label} raw joint sequence expected {len(command_names)} columns, "
                f"got {matrix.shape[1]}"
            )
    if matrix.shape[0] == 0:
        raise ValueError(f"{label} raw joint sequence cannot be empty")
    return matrix


def _raw_mapping_sequence_matrix(
    positions: Mapping[str, Sequence[float]],
    *,
    command_joint_names: Sequence[str],
    base_command: np.ndarray,
    label: str,
) -> np.ndarray:
    """把按关节名给出的 raw sequence 展开成完整 command-space 矩阵。"""

    if not positions:
        raise ValueError(f"{label} raw joint sequence mapping cannot be empty")
    index_by_name = {
        str(name): index for index, name in enumerate(command_joint_names)
    }
    sample_count: int | None = None
    columns: dict[int, np.ndarray] = {}
    for name, values in positions.items():
        key = str(name)
        if key not in index_by_name:
            raise ValueError(f"{label} raw joint {key!r} is not in command-space")
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{label} raw joint {key!r} samples cannot be empty")
        if sample_count is None:
            sample_count = int(array.size)
        elif array.size != sample_count:
            raise ValueError(f"{label} raw joint sequence sample counts must match")
        columns[index_by_name[key]] = array
    assert sample_count is not None
    matrix = np.tile(np.asarray(base_command, dtype=float).reshape(1, -1), (sample_count, 1))
    for index, values in columns.items():
        matrix[:, index] = values
    return matrix


def _validate_raw_side_sample_counts(
    left_positions: np.ndarray | None,
    right_positions: np.ndarray | None,
) -> None:
    """校验 raw sequence 左右两侧样本数一致，避免一侧提前结束。"""

    counts = [
        int(matrix.shape[0])
        for matrix in (left_positions, right_positions)
        if matrix is not None
    ]
    if not counts:
        raise ValueError("RawJointSequenceMoveSpec requires at least one side")
    if len(set(counts)) != 1:
        raise ValueError("left/right raw joint sequence sample counts must match")


def _hand_command_joint_names(command_joint_names: Sequence[str]) -> tuple[str, ...]:
    """从一侧 command-space 名称中过滤出 hand 组件关节。"""

    return tuple(
        str(name)
        for name in command_joint_names
        if component_for_name(str(name)) == "hand"
    )


def _command_linear_trajectory(
    *,
    command_joint_names: Sequence[str],
    start_command: np.ndarray,
    target_command: np.ndarray,
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """在 command-space 中生成一条线性插值轨迹。"""

    start = np.asarray(start_command, dtype=float).reshape(-1)
    target = np.asarray(target_command, dtype=float).reshape(-1)
    if start.size != target.size:
        raise ValueError(
            f"command trajectory shape mismatch: {start.size} vs {target.size}"
        )
    times = _trajectory_sample_times(duration_s=duration_s, sample_dt=sample_dt)
    duration = max(float(duration_s), float(sample_dt))
    alpha = (times / duration).reshape(-1, 1)
    positions = start.reshape(1, -1) + alpha * (target - start).reshape(1, -1)
    positions[-1] = target
    return joint_trajectory_from_positions(
        times=times,
        positions=positions,
        joint_names=tuple(str(name) for name in command_joint_names),
        phase=phase,
    )


def _trajectory_sample_times(*, duration_s: float, sample_dt: float) -> np.ndarray:
    """按仿真采样周期生成轨迹采样时间，并保证至少有一个样本。"""

    duration = max(float(duration_s), float(sample_dt))
    dt = max(float(sample_dt), 1.0e-6)
    steps = max(1, int(np.ceil(duration / dt)))
    return np.asarray(
        [min(duration, (index + 1) * dt) for index in range(steps)],
        dtype=float,
    )


def _dual_hand_duration(move: DualHandMoveSpec) -> float:
    """解析双手动作时长；父级缺省时取左右子手动作的最大时长。"""

    if move.duration_s is not None:
        return float(move.duration_s)
    durations = [
        float(hand.duration_s)
        for hand in (move.left, move.right)
        if hand is not None and hand.duration_s is not None
    ]
    if durations:
        return max(durations)
    raise ValueError("DualHandMoveSpec requires duration_s")


def _dual_hand_move_from_overlay(
    overlay: CommandOverlaySpec,
    *,
    duration_s: float | None,
    phase: str,
) -> DualHandMoveSpec:
    """把 overlay 转成可直接执行的 DualHandMoveSpec。"""

    return DualHandMoveSpec(
        left=_hand_with_default_duration(overlay.left_hand, duration_s),
        right=_hand_with_default_duration(overlay.right_hand, duration_s),
        duration_s=duration_s,
        phase=phase,
    )


def _hand_with_default_duration(
    hand: HandMoveSpec | None,
    duration_s: float | None,
) -> HandMoveSpec | None:
    """给 overlay 手部动作补默认时长；显式时长优先。"""

    if hand is None or hand.duration_s is not None:
        return hand
    if duration_s is None:
        raise ValueError("hand overlay duration_s is required")
    return replace(hand, duration_s=float(duration_s))


def _target_orientation_for_mode(
    *,
    mode: str,
    current_orientation: np.ndarray,
    target_orientation: np.ndarray | None,
) -> np.ndarray | None:
    """根据 orientation_mode 选择 IK 目标姿态。"""

    if mode == "none":
        return None
    if mode == "current":
        return np.asarray(current_orientation, dtype=float).reshape(4)
    if mode == "target":
        if target_orientation is None:
            raise ValueError("target_orientation is required for orientation_mode='target'")
        return np.asarray(target_orientation, dtype=float).reshape(4)
    raise ValueError("orientation_mode must be one of: current, target, none")


def _raise_if_requested_stop(should_stop: Callable[[], bool] | None) -> None:
    """在可中断边界检查外部停止请求，并抛出统一中断异常。"""

    if should_stop is not None and should_stop():
        raise DualCommandExecutionInterrupted("dual command execution interrupted")


def _ik_request_with_runtime_defaults(
    request: IKRequest,
    *,
    current_q: np.ndarray,
    tcp_frame_name: str,
) -> IKRequest:
    """给 IK request 补运行期默认 TCP 和 warm-start C-space seed。"""

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
    """给规划 request 补运行期默认 TCP 和 duration。"""

    return replace(
        request,
        tcp_frame_name=request.tcp_frame_name or tcp_frame_name,
        duration_s=request.duration_s if request.duration_s is not None else duration_s,
    )


def _dual_execution_mode(move: MoveSpec) -> str:
    """把 move 映射为双臂执行模式：selected_side 或 dual_cspace。"""

    if isinstance(move, CumotionMoveSpec) and move.execution == "dual_cspace":
        return "dual_cspace"
    return "selected_side"


def _move_side_required(move: MoveSpec) -> str:
    """读取双臂 selected-side move 的 side 字段。"""

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
    """解析 move 使用的 TCP frame；selected-side 模式可从左右默认 TCP 推导。"""

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
    """读取 dual_arm 语义配置中某一侧的配置块，并校验必需字段。"""

    side_config = dual_arm.get(side)
    if not isinstance(side_config, Mapping):
        raise ValueError(f"dual_arm.{side} must be a mapping")
    required = {"arm_joints"}
    missing = sorted(required - set(side_config))
    if missing:
        raise ValueError(f"dual_arm.{side} missing required keys: {missing}")
    return side_config


def _side_arm_joints(dual_arm: Mapping[str, object], side: str) -> tuple[str, ...]:
    """读取某一侧机械臂在融合 C-space 中的关节顺序。"""

    value = _side_dual_arm_config(dual_arm, side)["arm_joints"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"dual_arm.{side}.arm_joints must be a sequence")
    joints = tuple(str(name) for name in value)
    if not joints:
        raise ValueError(f"dual_arm.{side}.arm_joints cannot be empty")
    return joints


def _side_tcp_frame_name(dual_arm: Mapping[str, object], side: str) -> str:
    """读取某一侧动作层默认 TCP frame 名称。"""

    side_config = _side_dual_arm_config(dual_arm, side)
    if "tcp_frame" not in side_config:
        raise ValueError(f"dual_arm.{side}.tcp_frame cannot be empty")
    value = str(side_config["tcp_frame"])
    if not value:
        raise ValueError(f"dual_arm.{side}.tcp_frame cannot be empty")
    return value


def _side_flange_frame(dual_arm: Mapping[str, object], side: str) -> str:
    """读取某一侧机械臂法兰 frame，用于绑定临时 TCP link。"""

    side_config = _side_dual_arm_config(dual_arm, side)
    if "flange_frame" not in side_config:
        raise ValueError(f"dual_arm.{side}.flange_frame cannot be empty")
    value = str(side_config["flange_frame"])
    if not value:
        raise ValueError(f"dual_arm.{side}.flange_frame cannot be empty")
    return value


def _normalize_side(side: str) -> str:
    """把 side 规范化为 left/right。"""

    normalized = str(side).lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return normalized
