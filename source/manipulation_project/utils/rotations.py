"""命令行和配置输入常用的旋转转换。

项目配置里常用固定轴 XYZ RPY（roll、pitch、yaw）描述目标姿态，也就是外旋
XYZ 顺序；外部配置输入单位通常为 degree。SciPy 中小写 ``"xyz"`` 对应该约定，
等价于旧实现的 ``Rz @ Ry @ Rx``。
四元数对外统一使用 wxyz 顺序。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rpy_xyz_to_matrix(rpy_rad) -> np.ndarray:
    """把固定轴 XYZ 顺序（外旋 XYZ 顺序）的 RPY 弧度值转换为旋转矩阵。

    参数:
        rpy_rad: 长度 3 的 ``(roll, pitch, yaw)``，单位 rad。
    返回:
        shape ``(3, 3)`` 的旋转矩阵。
    """

    return Rotation.from_euler("xyz", np.asarray(rpy_rad, dtype=float).reshape(3)).as_matrix()


def rpy_xyz_deg_to_matrix(rpy_deg) -> np.ndarray:
    """把固定轴 XYZ 顺序（外旋 XYZ 顺序）的 RPY 角度值转换为旋转矩阵。

    参数:
        rpy_deg: 长度 3 的 ``(roll, pitch, yaw)``，单位 degree。
    返回:
        shape ``(3, 3)`` 的旋转矩阵。
    """

    return Rotation.from_euler("xyz", np.asarray(rpy_deg, dtype=float).reshape(3), degrees=True).as_matrix()


def matrix_to_quat_wxyz(matrix) -> np.ndarray:
    """把旋转矩阵转换为 wxyz 四元数。

    参数:
        matrix: shape ``(3, 3)`` 的旋转矩阵。
    返回:
        shape ``(4,)`` 的四元数，顺序为 ``[w, x, y, z]``。
    """

    x, y, z, w = Rotation.from_matrix(np.asarray(matrix, dtype=float).reshape(3, 3)).as_quat()
    return np.asarray([w, x, y, z], dtype=float)


def rpy_xyz_deg_to_quat_wxyz(rpy_deg) -> np.ndarray:
    """把固定轴 XYZ 顺序（外旋 XYZ 顺序）的 RPY 角度值转换为 wxyz 四元数。

    参数:
        rpy_deg: 长度 3 的 ``(roll, pitch, yaw)``，单位 degree。
    返回:
        shape ``(4,)`` 的 wxyz 四元数。
    """

    return matrix_to_quat_wxyz(rpy_xyz_deg_to_matrix(rpy_deg))
