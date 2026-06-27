"""项目姿态数据到 cuMotion Pose3/Rotation3 的边界适配。

项目内部和配置边界使用 ``wxyz`` 四元数；cuMotion ``Rotation3`` 构造函数同样接受
``w, x, y, z`` 分量，因此目标姿态可以直接从四元数构造。本模块只负责创建 cuMotion
原生对象，不承担通用姿态数学；通用四元数/矩阵转换放在 ``utils`` 中。
"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.utils.rotations import normalize_quat_wxyz_or_identity


def rotation_from_quat_wxyz(cumotion, quaternion):
    """从项目统一的 wxyz 四元数构造 cuMotion ``Rotation3``。

    输入会先归一化；零四元数按单位旋转处理，四元数存在小于 0 的情况下使用恒等变换。
    """

    w, x, y, z = normalize_quat_wxyz_or_identity(quaternion)
    return cumotion.Rotation3(float(w), float(x), float(y), float(z))


def pose_from_position_quat_wxyz(cumotion, position, orientation=None):
    """从位置和可选 wxyz 四元数构造 cuMotion ``Pose3``。

    ``orientation`` 为 ``None`` 时只约束平移，适合只关心 TCP 位置的任务；否则按 wxyz
    四元数直接构造 cuMotion ``Rotation3``。
    """

    translation = np.asarray(position, dtype=float).reshape(3)
    if orientation is None:
        return cumotion.Pose3.from_translation(translation)
    return cumotion.Pose3(rotation_from_quat_wxyz(cumotion, orientation), translation)


def pose_from_matrix(cumotion, matrix):
    """从 4x4 齐次矩阵构造 cuMotion ``Pose3``。

    矩阵的前三列表示旋转，最后一列前三项表示平移，单位与调用方的碰撞/目标坐标系一致。
    """

    pose = np.asarray(matrix, dtype=float).reshape(4, 4)
    return cumotion.Pose3(cumotion.Rotation3.from_matrix(pose[:3, :3]), pose[:3, 3])


def rotation_from_axis_angle(cumotion, axis, angle):
    """从轴角构造 cuMotion ``Rotation3``。"""

    axis_array = np.asarray(axis, dtype=float).reshape(3)
    return cumotion.Rotation3.from_axis_angle(axis_array, float(angle))


def rotation_from_scaled_axis(cumotion, scaled_axis):
    """从 scaled-axis 向量构造 cuMotion ``Rotation3``。"""

    scaled_axis_array = np.asarray(scaled_axis, dtype=float).reshape(3)
    return cumotion.Rotation3.from_scaled_axis(scaled_axis_array)


def pose_from_rotation_translation(cumotion, rotation, translation):
    """从 cuMotion ``Rotation3`` 和 3D 平移构造 ``Pose3``。"""

    return cumotion.Pose3(rotation, np.asarray(translation, dtype=float).reshape(3))
