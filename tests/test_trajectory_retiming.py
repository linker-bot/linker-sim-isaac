from __future__ import annotations

import numpy as np

from linkerbot_sim.trajectories.retiming import (
    retime_joint_trajectory,
    trajectory_sample_times,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def test_retime_joint_trajectory_recomputes_derivatives_on_target_grid() -> None:
    trajectory = JointTrajectory.from_samples(
        times=np.asarray([0.0, 0.1]),
        positions=np.asarray([[0.0], [1.0]]),
        velocities=np.asarray([[100.0], [100.0]]),
        accelerations=np.asarray([[50.0], [50.0]]),
        joint_names=("j0",),
    )

    retimed = retime_joint_trajectory(
        trajectory,
        duration_s=2.0,
        sample_dt_s=0.5,
        phase="retimed",
    )

    np.testing.assert_allclose(retimed.times, [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(retimed.positions[:, 0], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(retimed.velocities[:, 0], [0.5, 0.5, 0.5, 0.5])
    assert not np.allclose(retimed.velocities[:, 0], 100.0)
    assert retimed.phases == ("retimed", "retimed", "retimed", "retimed")


def test_retime_joint_trajectory_anchors_current_q_and_uses_path_progress() -> None:
    trajectory = JointTrajectory.from_samples(
        times=np.asarray([0.0, 1.0, 2.0, 3.0]),
        positions=np.asarray([[10.0], [10.0], [10.0], [20.0]]),
        velocities=np.asarray([[100.0], [100.0], [100.0], [100.0]]),
        joint_names=("j0",),
    )

    retimed = retime_joint_trajectory(
        trajectory,
        duration_s=2.0,
        sample_dt_s=0.5,
        start_position=np.asarray([0.0]),
        phase="anchored",
    )

    np.testing.assert_allclose(retimed.times, [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(retimed.positions[:, 0], [5.0, 10.0, 15.0, 20.0])
    np.testing.assert_allclose(retimed.velocities[:, 0], [10.0, 10.0, 10.0, 10.0])


def test_retime_joint_trajectory_interpolates_multidimensional_path_by_distance() -> (
    None
):
    trajectory = JointTrajectory.from_samples(
        times=np.asarray([0.0, 1.0, 2.0]),
        positions=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 3.0]]),
        joint_names=("j0", "j1"),
    )

    retimed = retime_joint_trajectory(
        trajectory,
        duration_s=4.0,
        sample_dt_s=1.0,
    )

    np.testing.assert_allclose(
        retimed.positions,
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
    )


def test_retime_joint_trajectory_accepts_repeated_terminal_waypoint() -> None:
    trajectory = JointTrajectory.from_samples(
        times=np.asarray([0.0, 1.0, 2.0]),
        positions=np.asarray([[0.0], [1.0], [1.0]]),
        joint_names=("j0",),
    )

    retimed = retime_joint_trajectory(
        trajectory,
        duration_s=2.0,
        sample_dt_s=0.5,
    )

    np.testing.assert_allclose(retimed.positions[:, 0], [0.25, 0.5, 0.75, 1.0])


def test_include_start_uses_same_samples_as_single_execution_grid() -> None:
    source = JointTrajectory.from_samples(
        times=np.asarray([0.0, 0.1, 5.0]),
        positions=np.asarray([[0.0], [0.1], [1.0]]),
        joint_names=("j0",),
    )

    single = retime_joint_trajectory(
        source,
        duration_s=0.2,
        sample_dt_s=0.05,
        start_position=np.asarray([0.0]),
    )
    anchored = retime_joint_trajectory(
        source,
        duration_s=0.2,
        sample_dt_s=0.05,
        start_position=np.asarray([0.0]),
        include_start=True,
    )

    np.testing.assert_allclose(anchored.times, [0.0, *single.times])
    np.testing.assert_allclose(anchored.positions[0], [0.0])
    np.testing.assert_allclose(anchored.positions[1:], single.positions)


def test_sample_times_keep_tick_grid_and_exact_non_divisible_endpoint() -> None:
    np.testing.assert_allclose(
        trajectory_sample_times(duration_s=0.11, sample_dt_s=0.05),
        [0.05, 0.1, 0.11],
    )
    np.testing.assert_allclose(
        trajectory_sample_times(
            duration_s=0.11,
            sample_dt_s=0.05,
            include_start=True,
        ),
        [0.0, 0.05, 0.1, 0.11],
    )
