"""时间参数化辅助数据结构。

目前只封装固定频率采样网格；后续若加入速度/加速度约束，可以在这里扩展更复杂的
时间参数化策略。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeGrid:
    """固定频率采样网格。

    输入字段:
        duration_s: 总持续时间，单位 s。
        sample_dt: 相邻采样点时间间隔，单位 s。
    输出:
        ``sample_hz`` 属性返回对应采样频率。
    """

    duration_s: float
    sample_dt: float

    @property
    def sample_hz(self) -> float:
        """返回采样频率。

        参数:
            无。
        返回:
            ``1.0 / sample_dt``，单位 Hz。
        """

        return 1.0 / self.sample_dt
