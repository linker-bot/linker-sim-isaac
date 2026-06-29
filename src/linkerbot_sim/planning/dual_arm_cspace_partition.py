"""双臂融合 C-space 的分区和轨迹拆分工具。

第一版双臂策略是“同一个 14-DOF cuMotion context 中预置 left/right TCP，每次运动选择
其中一侧作为任务 TCP”。本模块只处理 C-space 向量和关节名索引，不做 IK、不调用 cuMotion，
不接触 Isaac，也不解析 YAML 配置。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.trajectories.command_trajectory import (
    command_trajectory_from_arm_trajectory,
)
from linkerbot_sim.trajectories.types import JointTrajectory


@dataclass(frozen=True)
class DualArmJointPartitions:
    """左右臂在融合 cuMotion C-space 中的索引分组。"""

    joint_names: tuple[str, ...]
    left_indices: np.ndarray
    right_indices: np.ndarray

    @classmethod
    def from_joint_names(
        cls,
        joint_names: Sequence[str],
        *,
        left_joint_names: Sequence[str],
        right_joint_names: Sequence[str],
    ) -> "DualArmJointPartitions":
        """按关节名构造左右索引分组。"""

        names = tuple(str(name) for name in joint_names)
        index_by_name = {name: index for index, name in enumerate(names)}
        if len(index_by_name) != len(names):
            raise ValueError("joint_names contains duplicates")
        left = _indices_for_names(index_by_name, left_joint_names, "left")
        right = _indices_for_names(index_by_name, right_joint_names, "right")
        overlap = sorted(set(left.tolist()) & set(right.tolist()))
        if overlap:
            raise ValueError(f"left/right joint partitions overlap: {overlap}")
        return cls(joint_names=names, left_indices=left, right_indices=right)

    def active_indices(self, side: str) -> np.ndarray:
        """返回本次选定侧的 C-space 索引。"""

        normalized = _normalize_side(side)
        return self.left_indices if normalized == "left" else self.right_indices


def selected_side_goal(
    *,
    base_q: np.ndarray,
    solved_q: np.ndarray,
    partitions: DualArmJointPartitions,
    active_side: str,
) -> np.ndarray:
    """把单 TCP IK 解中的选定侧关节写回 14-DOF 目标，另一侧保持。"""

    base = _as_cspace_vector(base_q, len(partitions.joint_names), "base_q")
    solved = _as_cspace_vector(solved_q, len(partitions.joint_names), "solved_q")
    goal = base.copy()
    goal[partitions.active_indices(active_side)] = solved[
        partitions.active_indices(active_side)
    ]
    return goal


def split_dual_arm_trajectory_to_commands(
    *,
    dual_arm_trajectory: JointTrajectory,
    partitions: DualArmJointPartitions,
    left_command_joint_names: Sequence[str],
    right_command_joint_names: Sequence[str],
    left_start_command: np.ndarray,
    right_start_command: np.ndarray,
    left_target_command: np.ndarray,
    right_target_command: np.ndarray,
    phase: str | None = None,
) -> tuple[JointTrajectory, JointTrajectory]:
    """把融合 14-DOF arm 轨迹拆成左右 controller command-space 轨迹。"""

    left_arm = _arm_subtrajectory(
        dual_arm_trajectory, partitions.left_indices, "left"
    )
    right_arm = _arm_subtrajectory(
        dual_arm_trajectory, partitions.right_indices, "right"
    )
    left_indices = _command_indices_for_arm_joints(
        left_command_joint_names, left_arm.joint_names, "left"
    )
    right_indices = _command_indices_for_arm_joints(
        right_command_joint_names, right_arm.joint_names, "right"
    )
    return (
        command_trajectory_from_arm_trajectory(
            arm_trajectory=left_arm,
            command_joint_names=left_command_joint_names,
            arm_command_indices=left_indices,
            start_command=left_start_command,
            target_command=left_target_command,
            phase=phase,
        ),
        command_trajectory_from_arm_trajectory(
            arm_trajectory=right_arm,
            command_joint_names=right_command_joint_names,
            arm_command_indices=right_indices,
            start_command=right_start_command,
            target_command=right_target_command,
            phase=phase,
        ),
    )


def _arm_subtrajectory(
    trajectory: JointTrajectory, indices: np.ndarray, side: str
) -> JointTrajectory:
    index_array = np.asarray(indices, dtype=int).reshape(-1)
    return JointTrajectory.from_samples(
        times=trajectory.times,
        positions=trajectory.positions[:, index_array],
        velocities=trajectory.velocities[:, index_array],
        accelerations=trajectory.accelerations[:, index_array],
        jerks=trajectory.jerks[:, index_array],
        efforts=trajectory.efforts[:, index_array],
        phases=trajectory.phases,
        joint_names=tuple(trajectory.joint_names[int(index)] for index in index_array),
    )


def _command_indices_for_arm_joints(
    command_joint_names: Sequence[str],
    arm_joint_names: Sequence[str],
    side: str,
) -> np.ndarray:
    command_index_by_name = {
        str(name): index for index, name in enumerate(command_joint_names)
    }
    missing = [
        str(name) for name in arm_joint_names if str(name) not in command_index_by_name
    ]
    if missing:
        raise ValueError(
            f"{side} arm joints are missing from command-space joints: {missing}"
        )
    return np.asarray(
        [command_index_by_name[str(name)] for name in arm_joint_names], dtype=int
    )


def _indices_for_names(
    index_by_name: Mapping[str, int], names: Sequence[str], label: str
) -> np.ndarray:
    missing = [str(name) for name in names if str(name) not in index_by_name]
    if missing:
        raise ValueError(f"{label} joint names not found in C-space: {missing}")
    return np.asarray([index_by_name[str(name)] for name in names], dtype=int)


def _normalize_side(side: str) -> str:
    normalized = str(side).lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"active_side must be 'left' or 'right', got {side!r}")
    return normalized


def _as_cspace_vector(values: np.ndarray, expected_size: int, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != expected_size:
        raise ValueError(f"{label} expected {expected_size} values, got {vector.size}")
    return vector
