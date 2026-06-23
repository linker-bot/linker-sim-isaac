"""笛卡尔直线轨迹辅助函数。

当前实现把直线段视为两个点之间的线性空间插值，时间缩放由
``sample_cartesian_point_to_point`` 提供。
"""

from __future__ import annotations

from manipulation_project.trajectories.cartesian_point_to_point import sample_cartesian_point_to_point


def sample_cartesian_line(*args, **kwargs):
    """采样笛卡尔直线段。

    参数:
        与 ``sample_cartesian_point_to_point`` 完全相同。
    返回:
        ``list[(time_s, position)]``，其中 position 为 shape ``(3,)``、单位 m。
    """

    return sample_cartesian_point_to_point(*args, **kwargs)
