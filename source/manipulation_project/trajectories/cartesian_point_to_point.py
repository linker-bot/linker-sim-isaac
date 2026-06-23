"""笛卡尔点到点轨迹采样。

这里只生成位置采样，不处理姿态插值，也不做 IK；任务层可以把这些采样点逐个交给
IK 求解器或其它控制器。
"""

from __future__ import annotations

import numpy as np

from manipulation_project.trajectories.interpolation import interpolation_fn


def sample_cartesian_point_to_point(
    start_position,
    target_position,
    *,
    duration_s: float,
    sample_dt: float,
    interpolation: str = "smoothstep",
) -> list[tuple[float, np.ndarray]]:
    """在两个笛卡尔点之间采样位置。

    参数:
        start_position: 起点位置，长度 3，单位 m。
        target_position: 终点位置，长度 3，单位 m。
        duration_s: 运动持续时间，单位 s；为 0 时只返回终点。
        sample_dt: 采样时间间隔，单位 s。
        interpolation: 时间缩放函数名称。
    返回:
        列表元素为 ``(time_s, position)``，position 是 shape ``(3,)`` 的位置数组。
    """

    start = np.asarray(start_position, dtype=float).reshape(3)
    target = np.asarray(target_position, dtype=float).reshape(3)
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")
    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive")
    if duration_s == 0:
        return [(0.0, target.copy())]
    steps = max(1, int(np.ceil(duration_s / sample_dt)))
    fn = interpolation_fn(interpolation)
    return [
        (min(duration_s, step * sample_dt), start + fn(min(duration_s, step * sample_dt) / duration_s) * (target - start))
        for step in range(steps + 1)
    ]
