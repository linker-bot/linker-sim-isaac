from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from linkerbot_sim.app.interactive.tiled_scene import transport
from linkerbot_sim.app.interactive.tiled_scene.runtime import stepping


class _EmptyRequestQueue:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def get(self, *, timeout: float):
        self.timeouts.append(timeout)
        raise queue.Empty


def test_tiled_scene_hold_step_advances_one_configured_idle_chunk(monkeypatch) -> None:
    monotonic_values = iter((0.0, 0.0, 0.0, 0.025, 0.03))
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(monotonic_values))

    class Runtime:
        idle_period_s = 0.01

        def __init__(self) -> None:
            self.quit_event = threading.Event()
            self.session = SimpleNamespace(
                world=SimpleNamespace(get_physics_dt=lambda: 0.01)
            )
            self.idle_steps = 0

        def idle_step(self) -> None:
            self.idle_steps += 1
            if self.idle_steps == 3:
                self.quit_event.set()

    runtime = Runtime()
    request_queue = _EmptyRequestQueue()

    transport.run_interactive_loop(
        runtime,
        telemetry=None,
        request_queue=request_queue,  # type: ignore[arg-type]
        telemetry_rate_hz=0.0,
        idle_physics_policy="hold_step",
        idle_step_duration_s=0.025,
        queue_poll_timeout_s=0.1,
    )

    assert request_queue.timeouts == pytest.approx([0.025])
    assert runtime.idle_steps == 3


def test_idle_chunk_rejects_unbounded_physics_tick_count() -> None:
    runtime = SimpleNamespace(
        session=SimpleNamespace(world=SimpleNamespace(get_physics_dt=lambda: 1.0e-6))
    )

    with pytest.raises(ValueError, match="too many physics ticks"):
        transport._runtime_idle_step_count(runtime, 10.01)


def test_hold_step_requires_positive_idle_duration() -> None:
    runtime = SimpleNamespace(
        quit_event=threading.Event(),
        idle_step=lambda: None,
    )

    with pytest.raises(ValueError, match="positive idle_step_duration_s"):
        transport.run_interactive_loop(
            runtime,
            telemetry=None,
            request_queue=queue.Queue(),
            telemetry_rate_hz=0.0,
            idle_physics_policy="hold_step",
        )


def test_idle_step_count_requires_runtime_physics_dt() -> None:
    runtime = SimpleNamespace(session=SimpleNamespace(world=SimpleNamespace()))

    with pytest.raises(
        ValueError, match="requires runtime.session.world.get_physics_dt"
    ):
        transport._runtime_idle_step_count(runtime, 0.1)


def test_runtime_idle_period_requires_runtime_dt() -> None:
    runtime = SimpleNamespace(session=SimpleNamespace(world=SimpleNamespace()))

    with pytest.raises(ValueError, match="runtime idle period is unavailable"):
        stepping.idle_period_s(runtime)
