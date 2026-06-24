"""项目姿态数据与 cuMotion Pose3/Rotation3 的互转。"""

from __future__ import annotations

import numpy as np

from manipulation_project.utils.math_utils import quat_wxyz_to_matrix


def target_pose(cumotion, position, orientation=None):
    """构造 cuMotion ``Pose3``。"""

    translation = np.asarray(position, dtype=float).reshape(3)
    if orientation is None:
        return cumotion.Pose3.from_translation(translation)
    rotation_matrix = quat_wxyz_to_matrix(orientation)
    return cumotion.Pose3(cumotion.Rotation3.from_matrix(rotation_matrix), translation)


def pose_from_matrix(cumotion, matrix):
    """从 4x4 矩阵构造 cuMotion ``Pose3``。"""

    pose = np.asarray(matrix, dtype=float).reshape(4, 4)
    return cumotion.Pose3(cumotion.Rotation3.from_matrix(pose[:3, :3]), pose[:3, 3])
