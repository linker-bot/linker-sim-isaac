"""单臂 C-space 与 command-space 投影辅助函数。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.app.motion.runtime import (
    cspace_goal_to_command_vector,
    cspace_linear_trajectory as _shared_cspace_linear_trajectory,
    cspace_vector_from_command,
    current_command_from_runtime,
)
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.trajectories.types import JointTrajectory


def current_command(runtime: ExecutionRuntime) -> np.ndarray:
    """读取 articulation 当前关节位置，并投影到 controller command-space。"""

    return current_command_from_runtime(runtime)


def current_cspace_command(
    runtime: ExecutionRuntime,
    *,
    joint_names: Sequence[str],
    current_command_values: np.ndarray | None = None,
) -> np.ndarray:
    """按 cuMotion C-space 关节名拼出当前单臂关节向量。"""

    command = (
        current_command(runtime)
        if current_command_values is None
        else np.asarray(current_command_values, dtype=float).reshape(-1)
    )
    return cspace_vector_from_command(
        joint_names=joint_names,
        command_joint_names=runtime.joint_controller.command_joint_names,
        command=command,
        label="command",
    )


def cspace_goal_to_command(
    *,
    runtime: ExecutionRuntime,
    base_command: np.ndarray,
    joint_names: Sequence[str],
    goal_q: np.ndarray,
) -> np.ndarray:
    """把 C-space goal 中的机械臂关节写回单臂 command-space。"""

    return cspace_goal_to_command_vector(
        command_joint_names=runtime.joint_controller.command_joint_names,
        base_command=base_command,
        joint_names=joint_names,
        goal_q=goal_q,
    )


def cspace_linear_trajectory(
    *,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """为单臂 IK 动作构造一条简单 C-space 插值轨迹。"""

    return _shared_cspace_linear_trajectory(
        start_q=start_q,
        goal_q=goal_q,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt,
        phase=phase,
    )
