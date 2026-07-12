from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.backends.curobo.trajectory_adapter import (
    joint_trajectory_from_curobo,
)


def test_joint_trajectory_from_curobo_uses_interpolated_trajectory() -> None:
    trajectory = SimpleNamespace(
        position=np.asarray([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]]),
        velocity=np.asarray([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]]),
        acceleration=np.asarray([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]),
    )
    result = SimpleNamespace(
        interpolated_trajectory=trajectory,
        interpolated_trajectory_dt=np.asarray([0.05]),
    )

    joint_trajectory = joint_trajectory_from_curobo(
        result,
        joint_names=("j0", "j1"),
    )

    np.testing.assert_allclose(joint_trajectory.times, [0.0, 0.05, 0.1])
    np.testing.assert_allclose(joint_trajectory.positions[-1], [2.0, 4.0])
    np.testing.assert_allclose(joint_trajectory.velocities[1], [11.0, 21.0])


def test_joint_trajectory_from_curobo_accepts_singleton_batch_and_seed_dims() -> None:
    trajectory = SimpleNamespace(
        position=np.asarray([[[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]]]),
        velocity=np.asarray([[[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]]]),
        acceleration=np.asarray([[[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]]),
    )
    result = SimpleNamespace(
        interpolated_trajectory=trajectory,
        interpolated_trajectory_dt=np.asarray([0.05]),
    )

    joint_trajectory = joint_trajectory_from_curobo(
        result,
        joint_names=("j0", "j1"),
    )

    np.testing.assert_allclose(joint_trajectory.times, [0.0, 0.05, 0.1])
    np.testing.assert_allclose(joint_trajectory.positions[-1], [2.0, 4.0])
    np.testing.assert_allclose(joint_trajectory.velocities[1], [11.0, 21.0])


def test_result_dt_takes_precedence_over_explicit_sample_dt() -> None:
    result = SimpleNamespace(
        interpolated_trajectory=SimpleNamespace(
            position=np.asarray([[0.0], [0.5], [1.0]])
        ),
        interpolated_trajectory_dt=np.asarray([0.025]),
    )

    trajectory = joint_trajectory_from_curobo(
        result,
        joint_names=("j0",),
        sample_dt=0.1,
    )

    np.testing.assert_allclose(trajectory.times, [0.0, 0.025, 0.05])


def test_joint_trajectory_from_curobo_falls_back_to_finite_difference() -> None:
    trajectory = SimpleNamespace(
        position=np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    )

    joint_trajectory = joint_trajectory_from_curobo(
        trajectory,
        joint_names=("j0", "j1"),
        sample_dt=0.1,
    )

    np.testing.assert_allclose(joint_trajectory.times, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(joint_trajectory.positions[:, 0], [0.0, 1.0, 2.0])
    assert joint_trajectory.velocities.shape == (3, 2)


def test_joint_trajectory_from_curobo_rejects_missing_dt_source() -> None:
    trajectory = SimpleNamespace(
        position=np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    )

    with pytest.raises(ValueError, match="(?i)(sample_)?dt"):
        joint_trajectory_from_curobo(
            trajectory,
            joint_names=("j0", "j1"),
        )
