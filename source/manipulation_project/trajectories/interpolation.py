"""插值和时间缩放函数。

这些函数都接收归一化时间 ``alpha``，并返回归一化进度 ``scale``。调用方再用
``start + scale * (target - start)`` 生成具体位置。
"""

from __future__ import annotations

from manipulation_project.utils.math_utils import clamp01


def linear(alpha: float) -> float:
    """线性时间缩放。

    参数:
        alpha: 归一化时间，通常在 ``[0, 1]``。
    返回:
        截断到 ``[0, 1]`` 的线性进度。
    """

    return clamp01(alpha)


def smoothstep(alpha: float) -> float:
    """三次 smoothstep，端点速度为 0。

    参数:
        alpha: 归一化时间，通常在 ``[0, 1]``。
    返回:
        三次平滑进度，范围 ``[0, 1]``。
    """

    a = clamp01(alpha)
    return a * a * (3.0 - 2.0 * a)


def smootherstep(alpha: float) -> float:
    """五次 smootherstep，端点速度和加速度都为 0。

    参数:
        alpha: 归一化时间，通常在 ``[0, 1]``。
    返回:
        五次平滑进度，范围 ``[0, 1]``。
    """

    a = clamp01(alpha)
    return a * a * a * (10.0 - 15.0 * a + 6.0 * a * a)


def interpolation_fn(name: str):
    """按名称返回插值函数。

    参数:
        name: 插值名称，支持 ``linear/lin``、``smoothstep/cubic``、
        ``smootherstep/quintic/quintic_smoothstep``。
    返回:
        可调用对象，签名为 ``fn(alpha: float) -> float``。
    """

    normalized = name.strip().lower()
    if normalized in {"linear", "lin"}:
        return linear
    if normalized in {"smoothstep", "cubic"}:
        return smoothstep
    if normalized in {"smootherstep", "quintic", "quintic_smoothstep"}:
        return smootherstep
    raise ValueError(f"Unsupported interpolation: {name}")
