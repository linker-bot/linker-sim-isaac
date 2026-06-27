"""把后端关节轨迹组合成 controller command-space 轨迹。

本模块位于 planning 层，而不是 cuMotion backend 层。cuMotion backend 只应该知道机械臂
C-space；controller command-space 可能同时包含机械臂主动关节、手部主动关节或其它由动作
脚本控制的主动 DOF。把二者合成是动作规划边界的事，不应让后端了解灵巧手或 mimic。

重要边界：
    * 输入的 ``arm_trajectory`` 必须已经是离散采样后的项目 ``JointTrajectory``。
    * 本模块不调用 ``trajectory.eval_all(...)``，也不接触 cuMotion trajectory function。
    * 本模块不 retime、不拉长、不压短轨迹；输出时间网格严格沿用 ``arm_trajectory.times``。
    * mimic follower 不出现在 command-space 中，后续由 ``JointController`` 每帧展开。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def command_trajectory_from_arm_trajectory(
    *,
    arm_trajectory: JointTrajectory,
    command_joint_names: Sequence[str],
    arm_command_indices: Sequence[int],
    start_command: np.ndarray,
    target_command: np.ndarray,
    phase: str | None = None,
) -> JointTrajectory:
    """把机械臂轨迹嵌入 controller command-space。

    ``arm_trajectory`` 的列顺序来自规划后端的机械臂 C-space；``arm_command_indices`` 指明这些
    列应该写入 command-space 的哪些列。非机械臂 command 列按 ``start_command`` 到
    ``target_command`` 在线性补齐，常用于让手部主动关节在机械臂运动期间保持或缓慢变化。
    """

    arm_indices = np.asarray(arm_command_indices, dtype=int).reshape(-1)
    start = np.asarray(start_command, dtype=float).reshape(-1)
    target = np.asarray(target_command, dtype=float).reshape(-1)
    names = tuple(str(name) for name in command_joint_names)
    if len(names) != start.size or target.size != start.size:
        raise ValueError(
            "command trajectory shape mismatch: "
            f"{len(names)} names, start {start.size}, target {target.size}"
        )
    if arm_trajectory.positions.shape[1] != arm_indices.size:
        raise ValueError(
            "arm trajectory width mismatch: "
            f"trajectory has {arm_trajectory.positions.shape[1]} columns, "
            f"arm_command_indices has {arm_indices.size}"
        )

    command_positions = _command_positions_from_arm_positions(
        arm_positions=arm_trajectory.positions,
        times=arm_trajectory.times,
        start_command=start,
        target_command=target,
        arm_command_indices=arm_indices,
    )

    # 非机械臂列只需要给执行器和日志一个一致的导数估计；机械臂列必须保留后端 trajectory
    # generator 给出的真实 velocity/acceleration/jerk，避免有限差分覆盖 cuMotion 的时间参数化。
    phase_name = phase if phase is not None else _phase_from_trajectory(arm_trajectory)
    baseline = joint_trajectory_from_positions(
        times=arm_trajectory.times,
        positions=command_positions,
        joint_names=names,
        phase=phase_name,
    )
    velocities = baseline.velocities.copy()
    accelerations = baseline.accelerations.copy()
    jerks = baseline.jerks.copy()
    velocities[:, arm_indices] = arm_trajectory.velocities
    accelerations[:, arm_indices] = arm_trajectory.accelerations
    jerks[:, arm_indices] = arm_trajectory.jerks

    return JointTrajectory.from_samples(
        times=arm_trajectory.times,
        positions=command_positions,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        efforts=baseline.efforts,
        phases=tuple(phase_name for _ in range(len(arm_trajectory))),
        joint_names=names,
    )


def _command_positions_from_arm_positions(
    *,
    arm_positions: np.ndarray,
    times: np.ndarray,
    start_command: np.ndarray,
    target_command: np.ndarray,
    arm_command_indices: np.ndarray,
) -> np.ndarray:
    """生成 command-space 位置矩阵，并把机械臂列替换为后端采样结果。"""

    arm_positions = np.asarray(arm_positions, dtype=float)
    times = np.asarray(times, dtype=float).reshape(-1)
    start = np.asarray(start_command, dtype=float).reshape(-1)
    target = np.asarray(target_command, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("arm trajectory must contain at least one sample")

    # sampler 已经去掉首样本，因此 times[0] 通常是第一个 physics step 的目标时刻。
    # 这里仍以 0 作为隐含起点来计算非机械臂列插值比例：第一个样本是“从当前状态迈出的第一步”，
    # 最后一个样本严格落到 target。这个计算只补齐手部等 command 列，不改变轨迹时间网格。
    duration_s = float(times[-1])
    if duration_s <= 0.0:
        alpha = np.ones((times.size, 1), dtype=float)
    else:
        alpha = (times / duration_s).reshape(-1, 1)
    command_positions = start.reshape(1, -1) + alpha * (target - start).reshape(1, -1)
    command_positions[-1] = target
    command_positions[:, arm_command_indices] = arm_positions
    return command_positions


def _phase_from_trajectory(trajectory: JointTrajectory) -> str:
    """从输入轨迹取一个稳定 phase 名称。"""

    return str(trajectory.phases[0]) if trajectory.phases else "trajectory"
