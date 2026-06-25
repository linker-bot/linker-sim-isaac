from __future__ import annotations

import numpy as np

from manipulation_project.trajectories.interpolation import (
    linear,
    smootherstep,
    smoothstep,
)
from manipulation_project.trajectories.joint_trajectory_builder import (
    build_joint_target_trajectory,
    joint_trajectory_from_positions,
)
from manipulation_project.utils.rotations import rpy_xyz_to_quat_wxyz


def test_smoothstep_endpoints() -> None:
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smootherstep(0.0) == 0.0
    assert smootherstep(1.0) == 1.0
    assert linear(0.25) == 0.25


def test_joint_target_trajectory_reaches_target() -> None:
    trajectory = build_joint_target_trajectory(
        [0.0, 1.0],
        [1.0, 3.0],
        joint_names=["j1", "j2"],
        duration_s=1.0,
        sample_dt=0.25,
    )
    assert len(trajectory) == 5
    assert trajectory.times.shape == (5,)
    assert trajectory.positions.shape == (5, 2)
    assert trajectory.velocities.shape == (5, 2)
    assert trajectory.accelerations.shape == (5, 2)
    assert trajectory.jerks.shape == (5, 2)
    np.testing.assert_allclose(trajectory.positions[0], [0.0, 1.0])
    np.testing.assert_allclose(trajectory.positions[-1], [1.0, 3.0])
    np.testing.assert_allclose(trajectory.eval(1.0), [1.0, 3.0])
    sample = trajectory.eval_all(0.5)
    assert sample.position.shape == (2,)
    assert sample.velocity.shape == (2,)


def test_rpy_to_quaternion_is_unit_length() -> None:
    quat = rpy_xyz_to_quat_wxyz([0.0, 2.007128639793479, -np.pi / 2.0])
    assert quat.shape == (4,)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_joint_trajectory_from_positions_differentiates_samples() -> None:
    trajectory = joint_trajectory_from_positions(
        times=np.asarray([0.0, 0.5, 1.0], dtype=float),
        positions=np.asarray([[0.0], [1.0], [3.0]], dtype=float),
        joint_names=["j1"],
        phase="sampled",
    )

    np.testing.assert_allclose(trajectory.velocities[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(trajectory.accelerations[:, 0], [0.0, 4.0, 4.0])
    assert trajectory.phases == ("sampled", "sampled", "sampled")
