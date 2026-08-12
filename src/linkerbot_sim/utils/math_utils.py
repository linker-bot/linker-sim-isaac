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

from linkerbot_sim.utils.rotations import rpy_xyz_to_matrix


def as_vector(
    values, *, length: int | None = None, label: str = "vector"
) -> np.ndarray:
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


def make_rpy_transform(position=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """用 XYZ RPY 和平移构造齐次变换矩阵。"""

    return make_transform(position, rpy_xyz_to_matrix(rpy))
