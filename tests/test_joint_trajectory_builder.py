from __future__ import annotations

import numpy as np

from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)


def test_joint_trajectory_from_positions_differentiates_samples() -> None:
    trajectory = joint_trajectory_from_positions(
        times=np.asarray([0.0, 0.5, 1.0], dtype=float),
        positions=np.asarray([[0.0], [1.0], [3.0]], dtype=float),
        joint_names=["j1"],
        phase="sampled",
    )

    np.testing.assert_allclose(trajectory.velocities[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(trajectory.accelerations[:, 0], [0.0, 4.0, 4.0])
    np.testing.assert_allclose(trajectory.eval(1.0), [3.0])
    sample = trajectory.eval_all(0.5)
    assert sample.position.shape == (1,)
    assert sample.velocity.shape == (1,)
    assert trajectory.phases == ("sampled", "sampled", "sampled")
