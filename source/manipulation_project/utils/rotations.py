"""命令行和配置输入常用的旋转转换。

项目配置里常用固定轴 XYZ RPY（roll、pitch、yaw）描述目标姿态，也就是外旋 XYZ 顺序；
外部配置输入单位统一为 rad。SciPy 中小写 ``"xyz"`` 对应该约定，等价于旧实现的
``Rz @ Ry @ Rx``。四元数对外统一使用 wxyz 顺序。

职责边界:
    * 在配置友好的 RPY、项目边界的 wxyz 四元数、SciPy 的 xyzw 四元数之间转换。
    * 提供 TCP waypoint 姿态插值用的 slerp helper。
    * 不处理坐标系命名或机器人 frame 变换；调用方必须明确姿态属于哪个 frame。

SciPy ``Rotation`` 的四元数接口使用 xyzw，本模块集中做顺序转换，避免任务、TCP 和
Foxglove 日志中出现 wxyz/xyzw 混用。所有函数都返回新的 ndarray，不修改输入。
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

    return Rotation.from_euler(
        "xyz", np.asarray(rpy_rad, dtype=float).reshape(3)
    ).as_matrix()


def rpy_xyz_to_quat_wxyz(rpy_rad) -> np.ndarray:
    """把固定轴 XYZ 顺序（外旋 XYZ 顺序）的 RPY 弧度值转换为 wxyz 四元数。

    参数:
        rpy_rad: 长度 3 的 ``(roll, pitch, yaw)``，单位 rad。
    返回:
        shape ``(4,)`` 的 wxyz 四元数。
    """

    return matrix_to_quat_wxyz(rpy_xyz_to_matrix(rpy_rad))


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


def normalize_quat_wxyz(quat, *, label: str = "quat_wxyz") -> np.ndarray:
    """归一化 wxyz 四元数。

    参数:
        quat: 任意可转换为长度 4 数组的四元数，顺序为 ``[w, x, y, z]``。
        label: 报错信息中使用的字段名。
    返回:
        单位长度 wxyz 四元数副本；零四元数或长度错误会抛出 ``ValueError``。
    """

    quat_wxyz = np.asarray(quat, dtype=float).reshape(-1)
    if quat_wxyz.size != 4:
        raise ValueError(f"{label} expected 4 quaternion values, got {quat_wxyz.size}")
    norm = float(np.linalg.norm(quat_wxyz))
    if norm <= 0.0:
        raise ValueError(f"{label} must be non-zero")
    return quat_wxyz / norm


def quat_wxyz_to_xyzw(quat) -> np.ndarray:
    """把 wxyz 四元数转换为 SciPy 使用的 xyzw 顺序。

    输入会先归一化，避免姿态插值或矩阵转换时把配置里的数值误差继续放大。
    """

    quat_wxyz = normalize_quat_wxyz(quat)
    return np.asarray(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float
    )


def quat_xyzw_to_wxyz(quat) -> np.ndarray:
    """把 SciPy 使用的 xyzw 四元数转换为项目使用的 wxyz 顺序。

    参数:
        quat: SciPy ``Rotation.as_quat`` 风格的 ``[x, y, z, w]``。
    返回:
        归一化后的项目标准 ``[w, x, y, z]`` 四元数。
    """

    quat_xyzw = np.asarray(quat, dtype=float).reshape(-1)
    if quat_xyzw.size != 4:
        raise ValueError(
            f"quat_xyzw expected 4 quaternion values, got {quat_xyzw.size}"
        )
    return normalize_quat_wxyz([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def slerp_quat_wxyz(
    start_orientation, target_orientation, times
) -> list[np.ndarray | None]:
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
    # 四元数 q 和 -q 表示同一姿态。若点积为负，翻转目标可让 Slerp 走较短弧线，避免
    # 中间姿态绕远路旋转。
    if np.dot(start, target) < 0.0:
        target = -target

    duration = float(sample_times[-1])
    if duration <= 0:
        # 退化时间域只需要返回终点姿态，和位置采样中的“0 时长直接到目标”保持一致。
        return [target.copy()]

    key_rots = Rotation.from_quat([quat_wxyz_to_xyzw(start), quat_wxyz_to_xyzw(target)])
    sampled = Slerp([0.0, duration], key_rots)(sample_times).as_quat()
    return [quat_xyzw_to_wxyz(quat) for quat in sampled]
