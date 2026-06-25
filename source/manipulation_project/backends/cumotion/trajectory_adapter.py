"""cuMotion 轨迹到项目轨迹容器的适配。

cuMotion 返回的 trajectory 对象由后端定义，本模块只读取公共的 ``domain`` 与 ``eval_all``
风格接口，并转换成项目统一的 ``JointTrajectory``。输出矩阵列顺序由调用方传入的
``joint_names`` 决定，通常应与 cuMotion C-space 名称完全一致。

职责边界:
    * 不运行 cuMotion 规划，只采样已有后端轨迹对象。
    * 不把 C-space 轨迹扩展成 Isaac 完整 DOF；控制器/任务层负责名称映射。
    * 不改变单位；时间单位 s，关节位置 rad，速度 rad/s。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manipulation_project.trajectories.types import JointTrajectory


def joint_trajectory_from_cumotion(
    trajectory,
    *,
    joint_names: Sequence[str],
    sample_dt: float | None = None,
    times: Sequence[float] | None = None,
    phase: str = "cumotion",
) -> JointTrajectory:
    """把 cuMotion ``Trajectory`` 采样为项目 ``JointTrajectory``。

    cuMotion 轨迹通过 ``domain()`` 暴露时间域，通过 ``eval_all(t)`` 返回 position、
    velocity、acceleration 和 jerk。这里把这些值保存成矩阵，供控制器和日志使用。
    """

    # 支持显式 times 和 sample_dt 两种入口：测试可传固定 times，运行时可按控制频率采样。
    sample_times = _sample_times(trajectory, sample_dt=sample_dt, times=times)
    positions = []
    velocities = []
    accelerations = []
    jerks = []
    for time_s in sample_times:
        # cuMotion 状态字段在不同版本中可能是属性也可能是零参方法，统一经 ``_attr`` 读取。
        state = trajectory.eval_all(float(time_s))
        positions.append(np.asarray(_attr(state, "position"), dtype=float).reshape(-1))
        velocities.append(np.asarray(_attr(state, "velocity"), dtype=float).reshape(-1))
        accelerations.append(np.asarray(_attr(state, "acceleration"), dtype=float).reshape(-1))
        jerks.append(np.asarray(_attr(state, "jerk"), dtype=float).reshape(-1))
    return JointTrajectory.from_samples(
        times=sample_times,
        positions=np.vstack(positions),
        velocities=np.vstack(velocities),
        accelerations=np.vstack(accelerations),
        jerks=np.vstack(jerks),
        phases=tuple(phase for _ in range(sample_times.size)),
        joint_names=tuple(joint_names),
    )


def _sample_times(trajectory, *, sample_dt: float | None, times: Sequence[float] | None) -> np.ndarray:
    """根据显式时间序列或采样周期生成轨迹采样时刻。"""

    if times is not None:
        array = np.asarray(times, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError("times cannot be empty")
        return array
    if sample_dt is None or sample_dt <= 0:
        raise ValueError("sample_dt must be positive when times is not provided")
    # domain 兼容对象属性 lower/upper 和 tuple 两种表示；生成的最后一个采样点强制不超过 upper。
    domain = trajectory.domain()
    lower = float(getattr(domain, "lower", domain[0] if isinstance(domain, tuple) else 0.0))
    upper = float(getattr(domain, "upper", domain[1] if isinstance(domain, tuple) else lower))
    steps = max(1, int(np.ceil((upper - lower) / float(sample_dt))))
    return np.asarray([min(upper, lower + index * float(sample_dt)) for index in range(steps + 1)], dtype=float)


def _attr(state, name: str):
    """兼容 cuMotion 状态字段是属性或零参方法两种形式。"""

    value = getattr(state, name)
    return value() if callable(value) else value
