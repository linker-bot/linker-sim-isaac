from __future__ import annotations

import numpy as np

from manipulation_project.trajectories.cartesian_waypoints import (
    sample_cartesian_pose_line,
)
from manipulation_project.utils.timing import sample_times


def test_sample_cartesian_pose_line_interpolates_position_and_orientation() -> None:
    times = sample_times(1.0, 2.0)
    waypoints = sample_cartesian_pose_line(
        times=times,
        start_position=[0.0, 0.0, 0.0],
        target_position=[0.0, 0.0, 1.0],
        start_orientation=[1.0, 0.0, 0.0, 0.0],
        target_orientation=[0.0, 0.0, 0.0, 1.0],
    )

    assert len(waypoints) == 3
    np.testing.assert_allclose(
        [waypoint.time_s for waypoint in waypoints], [0.0, 0.5, 1.0]
    )
    np.testing.assert_allclose(
        [waypoint.position[2] for waypoint in waypoints], [0.0, 0.5, 1.0]
    )
    np.testing.assert_allclose(
        waypoints[1].orientation, [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )
    np.testing.assert_allclose(
        waypoints[2].orientation, [0.0, 0.0, 0.0, 1.0], atol=1.0e-12
    )


def test_sample_cartesian_pose_line_allows_position_only_waypoints() -> None:
    waypoints = sample_cartesian_pose_line(
        times=np.asarray([0.0, 1.0], dtype=float),
        start_position=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
    )

    assert waypoints[0].orientation is None
    assert waypoints[1].orientation is None
