from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.utils.math_utils import make_rpy_transform
from linkerbot_sim.utils.rotations import (
    quat_wxyz_to_matrix,
    rpy_xyz_to_quat_wxyz,
)


def test_rpy_to_quaternion_is_unit_length() -> None:
    quat = rpy_xyz_to_quat_wxyz([0.0, 2.007128639793479, -np.pi / 2.0])
    assert quat.shape == (4,)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_rpy_to_quaternion_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="rpy_rad must contain 3 finite values"):
        rpy_xyz_to_quat_wxyz([0.0, np.nan, 0.0])


def test_quaternion_matrix_uses_wxyz_and_normalizes_input() -> None:
    half_turn = np.sqrt(0.5)
    expected = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    np.testing.assert_allclose(
        quat_wxyz_to_matrix([2.0 * half_turn, 0.0, 0.0, 2.0 * half_turn]),
        expected,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(quat_wxyz_to_matrix([0.0, 0.0, 0.0, 0.0]), np.eye(3))


def test_make_rpy_transform_combines_scipy_rotation_and_translation() -> None:
    transform = make_rpy_transform((1.0, 2.0, 3.0), (0.0, 0.0, np.pi / 2.0))

    np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        transform[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1.0e-12,
    )
