from __future__ import annotations

import numpy as np

from linkerbot_sim.backends.cumotion.forward_kinematics import (
    CuMotionForwardKinematics,
)


class _Rotation:
    def __init__(self, matrix, quaternion) -> None:
        self._matrix = np.asarray(matrix, dtype=float)
        self._quaternion = tuple(float(value) for value in quaternion)

    def matrix(self):
        return self._matrix

    def w(self) -> float:
        return self._quaternion[0]

    def x(self) -> float:
        return self._quaternion[1]

    def y(self) -> float:
        return self._quaternion[2]

    def z(self) -> float:
        return self._quaternion[3]


class _Pose:
    def __init__(self, matrix, quaternion, translation=(0.1, 0.2, 0.3)) -> None:
        self.translation = np.asarray(translation, dtype=float)
        self.rotation = _Rotation(matrix, quaternion)


class _Kinematics:
    def __init__(self, pose) -> None:
        self._pose = pose

    def pose(self, joint_positions, frame_name: str):
        assert frame_name == "tool"
        np.testing.assert_allclose(joint_positions, [1.0, 2.0])
        return self._pose


class _Context:
    def __init__(self, pose) -> None:
        self.kinematics = _Kinematics(pose)

    def joint_names(self) -> list[str]:
        return ["j1", "j2"]

    def frame_names(self) -> list[str]:
        return ["tool"]


def test_compute_pose_returns_position_and_orientation() -> None:
    matrix = np.diag([1.0, -1.0, -1.0])
    fk = CuMotionForwardKinematics(_Context(_Pose(matrix, (0.0, 1.0, 0.0, 0.0))))

    pose = fk.compute_pose([1.0, 2.0], "tool")

    np.testing.assert_allclose(pose.position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose.rotation_matrix, matrix)
    np.testing.assert_allclose(np.abs(pose.orientation), [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(fk.compute_position([1.0, 2.0], "tool"), pose.position)
    np.testing.assert_allclose(
        fk.compute_orientation([1.0, 2.0], "tool"), pose.orientation
    )
