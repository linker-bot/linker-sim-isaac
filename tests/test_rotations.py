from __future__ import annotations

import numpy as np

from manipulation_project.utils.rotations import rpy_xyz_to_quat_wxyz


def test_rpy_to_quaternion_is_unit_length() -> None:
    quat = rpy_xyz_to_quat_wxyz([0.0, 2.007128639793479, -np.pi / 2.0])
    assert quat.shape == (4,)
    assert np.isclose(np.linalg.norm(quat), 1.0)
