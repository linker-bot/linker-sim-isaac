"""Mirror 主线程事件循环。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import time

from linkerbot_sim.mirror.interface.transport import MirrorTransportHub
from linkerbot_sim.mirror.runtime import MirrorCloseReport, MirrorRuntime


@dataclass(frozen=True)
class MirrorRunResult:
    iterations: int
    physics_steps: int
    close_report: MirrorCloseReport | None


def _app_running(runtime: MirrorRuntime) -> bool:
    callback = getattr(runtime.session.app, "is_running", None)
    return True if not callable(callback) else bool(callback())


def run_mirror(
    runtime: MirrorRuntime,
    *,
    endpoints: Sequence[object] = (),
    poll_timeout_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_ready: Callable[[], None] | None = None,
    max_iterations: int | None = None,
    close_on_exit: bool = True,
) -> MirrorRunResult:
    """在 session owner thread 消费 admission，并推进单场景物理/渲染。"""

    if not isinstance(runtime, MirrorRuntime):
        raise TypeError("run_mirror must be given a MirrorRuntime")
    timeout = (
        runtime.config.control.interface.queue_poll_timeout_s
        if poll_timeout_s is None
        else float(poll_timeout_s)
    )
    if timeout <= 0.0:
        raise ValueError("poll_timeout_s must be > 0")
    if runtime.config.control.sync_simulation_to_wall_clock:
        # queue wait 也是墙钟时间；同步模式下最多等待一个 physics tick，避免轮询等待
        # 与随后 paced idle batch 叠加，使仿真长期慢于真实时间。
        timeout = min(timeout, runtime.physics_dt_s)
    hub = MirrorTransportHub(tuple(endpoints)) if endpoints else None
    if hub is not None:
        runtime.attach_ingress(hub)
    iterations = 0
    physics_steps = 0
    render_period = 1.0 / runtime.config.scene.render_frequency_hz
    next_render_at = time.monotonic() + render_period
    close_report: MirrorCloseReport | None = None
    try:
        # endpoint 全部完成 bind/ready 后才宣布可用；start 中途失败由 hub 逆序回滚，随后
        # finally 仍会按 runtime owner graph 关闭 outputs、session 和 App。
        if hub is not None:
            hub.start()
        if on_ready is not None:
            on_ready()
        while not runtime.controller.quit_requested and _app_running(runtime):
            if should_stop is not None and should_stop():
                break
            if max_iterations is not None and iterations >= max_iterations:
                break
            response = runtime.controller.process_next(timeout_s=timeout)
            if response is None:
                if runtime.controller.admission.status().estopped:
                    # estop 后冻结物理；只允许 ingress 提交 reset/status/quit。
                    iterations += 1
                    continue
                policy = runtime.config.control.idle_physics_policy
                if policy == "hold_step":
                    # duration 是仿真时间而不是 wall-clock sleep。按 physics dt 量化为整数
                    # tick，每一 tick 都走同一 post-step output 边界。
                    for _ in range(_idle_hold_step_count(runtime)):
                        runtime.step(render=runtime.rendering is not None)
                        physics_steps += 1
                elif (
                    runtime.rendering is not None and time.monotonic() >= next_render_at
                ):
                    runtime.render()
                    next_render_at = time.monotonic() + render_period
            iterations += 1
    finally:
        if close_on_exit:
            close_report = runtime.close()
    return MirrorRunResult(
        iterations=iterations,
        physics_steps=physics_steps,
        close_report=close_report,
    )


def _idle_hold_step_count(runtime: MirrorRuntime) -> int:
    """把 strict idle duration 量化为至少一个 physics tick。"""

    get_dt = getattr(runtime.physics_runtime, "get_physics_dt", None)
    if not callable(get_dt):
        # 轻量 embedded/test runtime 可以只实现 step；此时一次循环仍推进恰好一步。
        return 1
    physics_dt = runtime.physics_dt_s
    duration = float(runtime.config.control.idle_step_duration_s)
    return max(1, int(math.ceil(duration / physics_dt - 1.0e-12)))


__all__ = ["MirrorRunResult", "run_mirror"]
