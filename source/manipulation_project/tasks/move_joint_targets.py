"""关节目标任务辅助函数。

该模块负责把 YAML 中的稀疏关节目标转换成控制器可执行的采样轨迹。实际仿真下发
由 ``manipulation_project.execution`` 层负责。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipulation_project.execution.joint_trajectory_executor import execute_joint_trajectory
from manipulation_project.robots.joint_groups import target_vector_from_mapping
from manipulation_project.trajectories.joint_trajectory_builder import build_joint_target_trajectory


# 兼容旧脚本/外部调用：执行器已经迁移到 execution 层。
execute_joint_target_trajectory = execute_joint_trajectory


@dataclass(frozen=True)
class MoveJointTargetsConfig:
    """固定时长关节目标移动的配置。

    输入字段:
        targets: 稀疏的 ``关节名 -> 目标位置(rad)`` 映射。
        duration_s: 从当前姿态移动到目标姿态的持续时间，单位秒。
        interpolation: 插值函数名称，例如 ``smoothstep``。
        sample_hz: 采样频率，单位 Hz，用于生成离散轨迹点。
        phase: 写入每个采样行的阶段名。
    输出:
        dataclass 本身不执行计算，作为构建轨迹时的输入参数集合。
    """

    targets: dict[str, float]
    duration_s: float
    interpolation: str = "smoothstep"
    sample_hz: float = 200.0
    phase: str = "joint_target"


def build_command_trajectory_from_sparse_targets(
    *,
    dof_names: list[str],
    command_indices: np.ndarray,
    current_positions: np.ndarray,
    config: MoveJointTargetsConfig,
):
    """从稀疏关节目标构建命令空间轨迹。

    参数:
        dof_names: articulation 的完整 DOF 名称列表，顺序必须与 Isaac 数组一致。
        command_indices: 控制器实际接收命令的 DOF 索引。
        current_positions: 当前完整 DOF 位置数组，单位 rad。
        config: 关节目标任务配置。
    返回:
        ``JointTrajectory``，其 ``joint_names`` 和 ``joint_positions`` 只包含
        ``command_indices`` 对应的命令关节。
    """

    full_target = target_vector_from_mapping(dof_names, config.targets, base=current_positions)
    start = np.asarray(current_positions, dtype=float).reshape(-1)[command_indices]
    target = full_target[command_indices]
    command_names = [dof_names[int(index)] for index in command_indices]
    return build_joint_target_trajectory(
        start,
        target,
        joint_names=command_names,
        duration_s=config.duration_s,
        sample_dt=1.0 / config.sample_hz,
        interpolation=config.interpolation,
        phase=config.phase,
    )
