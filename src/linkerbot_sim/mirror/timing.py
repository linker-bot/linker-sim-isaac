"""Mirror physics tick 的可选墙钟同步。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
import time


@dataclass
class WallClockStepSynchronizer:
    """限制连续 physics tick 的墙钟速率，并在落后时重定位 deadline。"""

    enabled: bool
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _next_step_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled 必须是 boolean")
        if not callable(self.monotonic) or not callable(self.sleep):
            raise TypeError("monotonic 和 sleep 必须可调用")

    def rebase(self) -> None:
        """让下一步立即执行，并从该步重新建立墙钟 deadline。"""

        self._next_step_at = None

    def before_step(self, physics_dt_s: float) -> None:
        """在 physics step 前等待剩余 tick 时间；落后时不补跑历史 deadline。"""

        dt = float(physics_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("physics_dt_s 必须是有限正数")
        if not self.enabled:
            return

        now = float(self.monotonic())
        if not math.isfinite(now):
            raise RuntimeError("monotonic 返回了非有限时间")
        deadline = self._next_step_at
        if deadline is None or now >= deadline:
            # 第一 tick 立即执行；若 owner 已落后，也只从当前时间建立下一 deadline，
            # 避免连续无等待地补跑积压的仿真步。
            self._next_step_at = now + dt
            return

        self.sleep(deadline - now)
        self._next_step_at = deadline + dt


__all__ = ["WallClockStepSynchronizer"]
