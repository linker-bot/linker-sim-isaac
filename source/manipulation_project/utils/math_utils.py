"""轨迹、TCP 和 IK 共享的小型数学工具。

这里保留的是项目内高频使用的简单转换，复杂旋转计算交给 SciPy ``Rotation`` 完成。约定：
项目外部姿态常用 wxyz 四元数，SciPy 内部接口需要 xyzw，因此函数内部会转换。

职责边界:
    * 做局部数值转换、形状校验和简单齐次矩阵构造。
    * 不读取配置，不访问 Isaac/Omni，也不解释机器人关节语义。
    * 不捕获异常；调用方可以根据任务语义把 ``ValueError`` 包装成配置错误或规划失败。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def as_vector(values, *, length: int | None = None, label: str = "vector") -> np.ndarray:
    """把输入转换为一维 float ndarray，并可选校验长度。

    参数:
        values: 标量/序列/ndarray 输入。
        length: 期望长度；为 ``None`` 时不检查。
        label: 报错信息中的变量名。
    返回:
        shape ``(N,)`` 的 float ndarray。
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    if length is not None and array.size != length:
        raise ValueError(f"{label} expected {length} values, got {array.size}")
    return array


def expand_scalar_or_vector(values, length: int, label: str) -> np.ndarray:
    """把标量扩展为定长向量，或校验已有向量长度。

    参数:
        values: 单个数或长度为 ``length`` 的序列。
        length: 目标向量长度。
        label: 报错信息中的变量名。
    返回:
        shape ``(length,)`` 的 float ndarray。
    """

    # 控制器配置经常允许“一个标量应用到整组关节”或“逐关节给一个向量”两种写法；
    # 这里统一展开，调用方后续只处理定长数组。
    array = as_vector(values, label=label)
    if array.size == 1:
        return np.full(length, float(array[0]), dtype=float)
    if array.size != length:
        raise ValueError(f"{label} expected 1 or {length} values, got {array.size}")
    return array.astype(float)


def clamp01(value: float) -> float:
    """把标量截断到 ``[0, 1]``。

    参数:
        value: 任意可转换为 float 的数值。
    返回:
        截断后的 float。
    """

    return min(1.0, max(0.0, float(value)))


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    """把 wxyz 四元数转换为旋转矩阵。

    参数:
        quat: 长度 4 的四元数，顺序为 ``[w, x, y, z]``。
    返回:
        shape ``(3, 3)`` 的旋转矩阵；零范数输入返回单位矩阵。
    """

    quat_wxyz = as_vector(quat, length=4, label="quat_wxyz")
    norm = float(np.linalg.norm(quat_wxyz))
    if norm <= 0.0:
        # 零四元数通常来自缺省/坏配置；返回单位旋转比让 SciPy 抛晦涩异常更适合作为底层工具。
        return np.eye(3, dtype=float)
    w, x, y, z = quat_wxyz / norm
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def axis_angle_to_matrix(axis, angle: float) -> np.ndarray:
    """把轴角旋转转换为旋转矩阵。

    参数:
        axis: 长度 3 的旋转轴，不要求预归一化。
        angle: 绕轴旋转角，单位 rad。
    返回:
        shape ``(3, 3)`` 的旋转矩阵；零轴返回单位矩阵。
    """

    axis_array = as_vector(axis, length=3, label="axis")
    norm = float(np.linalg.norm(axis_array))
    if norm <= 0.0:
        # 零轴没有明确旋转方向；按“无旋转”处理，方便 MJCF 缺省轴或异常小轴值保持稳定。
        return np.eye(3, dtype=float)
    return Rotation.from_rotvec((axis_array / norm) * float(angle)).as_matrix()


def make_transform(position=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    """构造齐次变换矩阵。

    参数:
        position: 长度 3 的平移，单位由调用场景决定，通常为 m。
        rotation: 可选 shape ``(3, 3)`` 旋转矩阵；为空时使用单位旋转。
    返回:
        shape ``(4, 4)`` 的齐次变换矩阵。
    """

    # 使用齐次矩阵统一表示平移和旋转，便于 MJCF body chain 正运动学逐段右乘。
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = as_vector(position, length=3, label="position")
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    return transform
