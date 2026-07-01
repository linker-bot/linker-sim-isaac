"""双臂 C-space、IK goal 和 selected-side 轨迹辅助函数。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.app.motion.runtime import (
    command_values_by_name,
    cspace_goal_to_command_vector,
    cspace_linear_trajectory,
    cspace_trajectory_from_motion_result,
    current_command_from_runtime,
    solve_ik_request,
)
from linkerbot_sim.execution.dual_runtime import DualRobotRuntime, RobotSideRuntime
from linkerbot_sim.planning.dual_arm_cspace_partition import (
    DualArmJointPartitions,
    selected_side_goal,
)
from linkerbot_sim.planning.requests import IKRequest
from linkerbot_sim.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class IkMotionGoal:
    """一次双臂 context 中的 IK 目标摘要。"""

    side: str
    tcp_frame_name: str
    goal_q: np.ndarray
    target_position: np.ndarray
    position_error: float
    orientation_error: float | None


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
