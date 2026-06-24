"""cuMotion 轨迹到项目轨迹容器的适配。"""

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

    sample_times = _sample_times(trajectory, sample_dt=sample_dt, times=times)
    positions = []
    velocities = []
    accelerations = []
    jerks = []
    for time_s in sample_times:
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
    if times is not None:
        array = np.asarray(times, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError("times cannot be empty")
        return array
    if sample_dt is None or sample_dt <= 0:
        raise ValueError("sample_dt must be positive when times is not provided")
    domain = trajectory.domain()
    lower = float(getattr(domain, "lower", domain[0] if isinstance(domain, tuple) else 0.0))
    upper = float(getattr(domain, "upper", domain[1] if isinstance(domain, tuple) else lower))
    steps = max(1, int(np.ceil((upper - lower) / float(sample_dt))))
    return np.asarray([min(upper, lower + index * float(sample_dt)) for index in range(steps + 1)], dtype=float)


def _attr(state, name: str):
    value = getattr(state, name)
    return value() if callable(value) else value
