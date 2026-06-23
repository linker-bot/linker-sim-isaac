"""仿真循环计时辅助函数。

用于把“控制器更新频率”转换为“每隔多少个 physics step 更新一次命令”。
"""

from __future__ import annotations


def interval_steps(update_frequency_hz: float, physics_dt: float) -> int:
    """计算低频更新间隔对应的 physics step 数。

    参数:
        update_frequency_hz: 目标更新频率，单位 Hz。
        physics_dt: 仿真物理步长，单位 s。
    返回:
        至少为 1 的整数 step 间隔。
    """

    if update_frequency_hz <= 0:
        raise ValueError("update_frequency_hz must be positive")
    if physics_dt <= 0:
        raise ValueError("physics_dt must be positive")
    return max(1, int(round(1.0 / (update_frequency_hz * physics_dt))))
