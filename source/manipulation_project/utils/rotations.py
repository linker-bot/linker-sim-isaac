"""命令行和配置输入常用的旋转转换。

项目配置里常用固定轴 XYZ RPY（roll、pitch、yaw）描述目标姿态，也就是外旋
XYZ 顺序；外部配置输入单位通常为 degree。SciPy 中小写 ``"xyz"`` 对应该约定，
等价于旧实现的 ``Rz @ Ry @ Rx``。
四元数对外统一使用 wxyz 顺序。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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


def normalize_quat_wxyz(quat, *, label: str = "quat_wxyz") -> np.ndarray:
    """归一化 wxyz 四元数。"""

    quat_wxyz = np.asarray(quat, dtype=float).reshape(-1)
    if quat_wxyz.size != 4:
        raise ValueError(f"{label} expected 4 quaternion values, got {quat_wxyz.size}")
    norm = float(np.linalg.norm(quat_wxyz))
    if norm <= 0.0:
        raise ValueError(f"{label} must be non-zero")
    return quat_wxyz / norm


def quat_wxyz_to_xyzw(quat) -> np.ndarray:
    """把 wxyz 四元数转换为 SciPy 使用的 xyzw 顺序。"""

    quat_wxyz = normalize_quat_wxyz(quat)
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)


def quat_xyzw_to_wxyz(quat) -> np.ndarray:
    """把 SciPy 使用的 xyzw 四元数转换为项目使用的 wxyz 顺序。"""

    quat_xyzw = np.asarray(quat, dtype=float).reshape(-1)
    if quat_xyzw.size != 4:
        raise ValueError(f"quat_xyzw expected 4 quaternion values, got {quat_xyzw.size}")
    return normalize_quat_wxyz([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def slerp_quat_wxyz(start_orientation, target_orientation, times) -> list[np.ndarray | None]:
    """按时间对 wxyz 起止四元数做球面线性插值。

    若任一端点为 ``None``，返回全 ``None``，供只约束位置的 IK waypoint 使用。
    """

    sample_times = np.asarray(times, dtype=float).reshape(-1)
    if sample_times.size == 0:
        raise ValueError("times must contain at least one sample")
    if start_orientation is None or target_orientation is None:
        return [None for _ in range(sample_times.size)]

    start = normalize_quat_wxyz(start_orientation, label="start_orientation")
    target = normalize_quat_wxyz(target_orientation, label="target_orientation")
    if np.dot(start, target) < 0.0:
        target = -target

    duration = float(sample_times[-1])
    if duration <= 0:
        return [target.copy()]

    key_rots = Rotation.from_quat([quat_wxyz_to_xyzw(start), quat_wxyz_to_xyzw(target)])
    sampled = Slerp([0.0, duration], key_rots)(sample_times).as_quat()
    return [quat_xyzw_to_wxyz(quat) for quat in sampled]
