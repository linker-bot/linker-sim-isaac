"""命令行和配置输入常用的旋转转换。

项目配置里常用固定轴 XYZ RPY（roll、pitch、yaw）描述目标姿态，也就是外旋 XYZ 顺序；
外部配置输入单位统一为 rad。SciPy 中小写 ``"xyz"`` 对应该约定，等价于旧实现的
``Rz @ Ry @ Rx``。四元数对外统一使用 wxyz 顺序。

职责边界:
    * 在配置友好的 RPY、项目边界的 wxyz 四元数、旋转矩阵之间转换。
    * 不处理坐标系命名或机器人 frame 变换；调用方必须明确姿态属于哪个 frame。

SciPy ``Rotation`` 的四元数接口使用 xyzw，本模块集中做顺序转换，避免任务、TCP 和
Foxglove 日志中出现 wxyz/xyzw 混用。所有函数都返回新的 ndarray，不修改输入。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _finite_vector(value, *, size: int, label: str) -> np.ndarray:
    """把冷路径旋转输入规范为有限一维向量，统一边界错误。"""

    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    return result


def rpy_xyz_to_matrix(rpy_rad) -> np.ndarray:
    """把固定轴 XYZ 顺序（外旋 XYZ 顺序）的 RPY 弧度值转换为旋转矩阵。

    参数:
        rpy_rad: 长度 3 的 ``(roll, pitch, yaw)``，单位 rad。
    返回:
        shape ``(3, 3)`` 的旋转矩阵。
    """

    return Rotation.from_euler(
        "xyz", _finite_vector(rpy_rad, size=3, label="rpy_rad")
    ).as_matrix()


def rpy_xyz_to_quat_wxyz(rpy_rad) -> np.ndarray:
    """把固定轴 XYZ 顺序（外旋 XYZ 顺序）的 RPY 弧度值转换为 wxyz 四元数。

    参数:
        rpy_rad: 长度 3 的 ``(roll, pitch, yaw)``，单位 rad。
    返回:
        shape ``(4,)`` 的 wxyz 四元数。
    """

    xyzw = Rotation.from_euler(
        "xyz", _finite_vector(rpy_rad, size=3, label="rpy_rad")
    ).as_quat()
    return xyzw[[3, 0, 1, 2]]


def matrix_to_quat_wxyz(matrix) -> np.ndarray:
    """把旋转矩阵转换为 wxyz 四元数。

    参数:
        matrix: shape ``(3, 3)`` 的旋转矩阵。
    返回:
        shape ``(4,)`` 的四元数，顺序为 ``[w, x, y, z]``。
    """

    x, y, z, w = Rotation.from_matrix(
        np.asarray(matrix, dtype=float).reshape(3, 3)
    ).as_quat()
    return np.asarray([w, x, y, z], dtype=float)


def normalize_quat_wxyz_or_identity(quat, *, label: str = "quat_wxyz") -> np.ndarray:
    """归一化 wxyz 四元数；零四元数按单位旋转处理。

    该函数适合底层适配层保持容错行为：零四元数通常表示配置未写或后端缺省值。
    """

    quat_wxyz = np.asarray(quat, dtype=float).reshape(-1)
    if quat_wxyz.size != 4:
        raise ValueError(f"{label} expected 4 quaternion values, got {quat_wxyz.size}")
    norm = float(np.linalg.norm(quat_wxyz))
    if norm <= 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat_wxyz / norm


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    """把 wxyz 四元数转换为旋转矩阵；零四元数按单位旋转处理。"""

    w, x, y, z = normalize_quat_wxyz_or_identity(quat)
    return Rotation.from_quat([x, y, z, w]).as_matrix()
