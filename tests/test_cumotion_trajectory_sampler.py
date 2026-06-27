from __future__ import annotations

import numpy as np

from manipulation_project.backends.cumotion.trajectory_sampler import (
    joint_trajectory_from_cumotion,
)


class _State:
    def __init__(self, value: float) -> None:
        self.position = np.asarray([value, value + 1.0])
        self.velocity = np.asarray([2.0 * value, 2.0 * value])
        self.acceleration = np.asarray([0.5, 0.5])
        self.jerk = np.asarray([0.0, 0.0])


class _Trajectory:
    def __init__(self, *, tuple_state: bool = False) -> None:
        self.tuple_state = tuple_state

    def domain(self):
        return (0.0, 1.0)

    def eval_all(self, time_s: float):
        state = _State(time_s)
        if self.tuple_state:
            return (state.position, state.velocity, state.acceleration, state.jerk)
        return state


def test_joint_trajectory_from_cumotion_samples_eval_all() -> None:
    trajectory = joint_trajectory_from_cumotion(
        _Trajectory(),
        joint_names=("j1", "j2"),
        sample_dt=0.5,
        phase="planned",
    )

    assert len(trajectory) == 3
    np.testing.assert_allclose(trajectory.times, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(trajectory.positions[-1], [1.0, 2.0])
    np.testing.assert_allclose(trajectory.velocities[1], [1.0, 1.0])
    assert trajectory.phases[0] == "planned"


def test_joint_trajectory_from_cumotion_accepts_real_tuple_eval_all() -> None:
    trajectory = joint_trajectory_from_cumotion(
        _Trajectory(tuple_state=True),
        joint_names=("j1", "j2"),
        times=[0.0, 1.0],
    )

    np.testing.assert_allclose(trajectory.positions, [[0.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(trajectory.velocities, [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_allclose(trajectory.accelerations[-1], [0.5, 0.5])
