"""cuMotion 轨迹采样到项目轨迹容器。

cuMotion 返回的 trajectory 对象由后端定义，本模块只读取公共的 ``domain`` 与 ``eval_all``
风格接口，并转换成项目统一的 ``JointTrajectory``。输出矩阵列顺序由调用方传入的
``joint_names`` 决定，通常应与 cuMotion C-space 名称完全一致。

职责边界:
    * 不运行 cuMotion 规划，只采样已有后端轨迹对象。
    * 不把 C-space 轨迹扩展成 Isaac 完整 DOF；控制器/动作脚本层负责名称映射。
    * 不改变单位；时间单位 s，关节位置 rad，速度 rad/s。
    * 输出给 execution 的采样矩阵不包含首样本。首样本通常就是当前状态，过滤动作放在
      cuMotion 轨迹函数采样边界，避免 execution 或动作脚本重复切片。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.trajectories.types import JointTrajectory


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

    本模块和 ``src/linkerbot_sim/trajectories`` 的职责不同：这里只理解 cuMotion
    对象如何采样；``trajectories`` 只定义项目内部矩阵轨迹容器和从位置矩阵构造导数的工具。
    """

    # 支持显式 times 和 sample_dt 两种入口：测试可传固定 times，运行时可按控制频率采样。
    sample_times = _sample_times(trajectory, sample_dt=sample_dt, times=times)
    positions = []
    velocities = []
    accelerations = []
    jerks = []
    for time_s in sample_times:
        # cuMotion 1.1 的真实 pybind 接口返回 ``(position, velocity, acceleration, jerk)``
        # 四元组；部分测试替身或旧封装会返回带同名属性/方法的对象。统一在这里拆解，避免
        # 动作脚本层关心后端对象的具体 Python 形态。
        position, velocity, acceleration, jerk = _trajectory_state_values(
            trajectory.eval_all(float(time_s))
        )
        positions.append(np.asarray(position, dtype=float).reshape(-1))
        velocities.append(np.asarray(velocity, dtype=float).reshape(-1))
        accelerations.append(np.asarray(acceleration, dtype=float).reshape(-1))
        jerks.append(np.asarray(jerk, dtype=float).reshape(-1))
    return JointTrajectory.from_samples(
        times=sample_times,
        positions=np.vstack(positions),
        velocities=np.vstack(velocities),
        accelerations=np.vstack(accelerations),
        jerks=np.vstack(jerks),
        phases=tuple(phase for _ in range(sample_times.size)),
        joint_names=tuple(joint_names),
    )


def _sample_times(
    trajectory, *, sample_dt: float | None, times: Sequence[float] | None
) -> np.ndarray:
    """根据显式时间序列或采样周期生成轨迹采样时刻。

    返回值会自动移除时间域下界对应的首样本。execution 语义是一行推进一个 physics
    step，因此第 0 行应该是“下一帧目标”，而不是当前状态。
    """

    domain = trajectory.domain()
    lower = float(
        getattr(domain, "lower", domain[0] if isinstance(domain, tuple) else 0.0)
    )
    upper = float(
        getattr(domain, "upper", domain[1] if isinstance(domain, tuple) else lower)
    )
    if times is not None:
        array = np.asarray(times, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError("times cannot be empty")
        return _drop_initial_sample(array, lower=lower, upper=upper)
    if sample_dt is None or sample_dt <= 0:
        raise ValueError("sample_dt must be positive when times is not provided")
    # domain 兼容对象属性 lower/upper 和 tuple 两种表示；采样从第一个 step 后开始，
    # 最后一个采样点强制不超过 upper。这样输出矩阵天然是一行一个执行 physics step。
    steps = max(1, int(np.ceil((upper - lower) / float(sample_dt))))
    return np.asarray(
        [min(upper, lower + index * float(sample_dt)) for index in range(1, steps + 1)],
        dtype=float,
    )


def _drop_initial_sample(
    times: np.ndarray, *, lower: float, upper: float
) -> np.ndarray:
    """从显式采样时间里移除首样本，并保证退化轨迹仍至少有一个目标点。"""

    array = np.asarray(times, dtype=float).reshape(-1)
    if array.size == 1:
        return array
    # 显式 times 常来自测试或上层 helper。只删除等于 domain lower 的首个样本，不改动其它
    # 时间点，避免意外改变调用方已经 materialize 好的 physics-step 网格。
    if np.isclose(array[0], float(lower), rtol=0.0, atol=1.0e-12):
        array = array[1:]
    if array.size == 0:
        return np.asarray([float(upper)], dtype=float)
    return array


def _attr(state, name: str):
    """兼容 cuMotion 状态字段是属性或零参方法两种形式。"""

    value = getattr(state, name)
    return value() if callable(value) else value


def _trajectory_state_values(state):
    """返回 ``position, velocity, acceleration, jerk`` 四个轨迹状态数组。"""

    if isinstance(state, tuple):
        if len(state) != 4:
            raise ValueError(
                "trajectory.eval_all tuple must contain position, velocity, acceleration, jerk"
            )
        return state
    return (
        _attr(state, "position"),
        _attr(state, "velocity"),
        _attr(state, "acceleration"),
        _attr(state, "jerk"),
    )
