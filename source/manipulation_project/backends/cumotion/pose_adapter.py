"""项目姿态数据与 cuMotion Pose3/Rotation3 的互转。

项目内部和配置边界使用 ``wxyz`` 四元数，而 cuMotion ``Rotation3`` 更常通过旋转矩阵
构造。这里集中处理格式转换，避免 IK/FK/碰撞世界适配代码各自重复并引入顺序错误。
"""

from __future__ import annotations

import numpy as np

from manipulation_project.utils.math_utils import quat_wxyz_to_matrix


def target_pose(cumotion, position, orientation=None):
    """构造 cuMotion ``Pose3``。

    ``orientation`` 为 ``None`` 时只约束平移，适合只关心 TCP 位置的任务；否则按 wxyz
    四元数转换为旋转矩阵再交给 cuMotion。
    """

    translation = np.asarray(position, dtype=float).reshape(3)
    if orientation is None:
        return cumotion.Pose3.from_translation(translation)
    rotation_matrix = quat_wxyz_to_matrix(orientation)
    return cumotion.Pose3(cumotion.Rotation3.from_matrix(rotation_matrix), translation)


def pose_from_matrix(cumotion, matrix):
    """从 4x4 齐次矩阵构造 cuMotion ``Pose3``。

    矩阵的前三列表示旋转，最后一列前三项表示平移，单位与调用方的碰撞/目标坐标系一致。
    """

    pose = np.asarray(matrix, dtype=float).reshape(4, 4)
    return cumotion.Pose3(cumotion.Rotation3.from_matrix(pose[:3, :3]), pose[:3, 3])
