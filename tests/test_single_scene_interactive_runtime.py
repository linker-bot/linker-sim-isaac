from __future__ import annotations

import sys
from io import StringIO
from types import SimpleNamespace

import pytest

from linkerbot_sim.app.interactive.policies import InteractiveRuntimePolicy
from linkerbot_sim.app.interactive.single_scene import cli as single_scene_cli
from linkerbot_sim.app.interactive.single_scene import runtime as single_scene_runtime
from linkerbot_sim.app.interactive.single_scene.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.single_scene.transports import (
    start_interactive_transports,
)


class _IdleOnceApp:
    def __init__(self) -> None:
        self.is_running_calls = 0

    def is_running(self) -> bool:
        self.is_running_calls += 1
        return self.is_running_calls == 1


class _EmptyMotionQueue:
    def __init__(self) -> None:
        self.pending_timeouts: list[float | None] = []
        self.status_provider = None

    def set_status_provider(self, provider) -> None:
        self.status_provider = provider

    def quit_requested(self) -> bool:
        return False

    def consume_snapshot_request(self):
        return None

    def consume_reset_request(self):
        return None

    def next_pending(self, *, timeout_s: float | None = None):
        self.pending_timeouts.append(timeout_s)
        return None

    def estop_requested(self) -> bool:
        return False

    def should_stop_current(self) -> bool:
        return False

    def request_quit(self) -> None:
        return None


def test_single_scene_interactive_queue_timeout_runs_one_idle_hold_chunk(
    monkeypatch,
) -> None:
    motion_queue = _EmptyMotionQueue()
    planner = object()
    transport = SimpleNamespace(
        stop=lambda **_kwargs: SimpleNamespace(stopped=True, live_resources=())
    )
    hold_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        single_scene_runtime,
        "InteractiveMotionQueue",
        lambda **_kwargs: motion_queue,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "TimelinePlanningSession",
        lambda _runtime, *, planner_backend: planner,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_state_stream",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_transports",
        lambda **_kwargs: transport,
    )

    def fake_hold_all(
        runtime,
        *,
        planner,
        step: int,
        duration_s: float,
        should_stop,
    ) -> int:
        hold_calls.append(
            {
                "runtime": runtime,
                "planner": planner,
                "step": step,
                "duration_s": duration_s,
                "should_stop": should_stop,
            }
        )
        return step + 1

    monkeypatch.setattr(single_scene_runtime, "_hold_all", fake_hold_all)
    runtime = SimpleNamespace(
        session=SimpleNamespace(app=_IdleOnceApp()),
        status=lambda: {},
        status_prefix="TEST_SINGLE_SCENE_INTERACTIVE",
    )

    final_step = single_scene_runtime.run_single_scene_interactive_motion(
        runtime,
        stdin_enabled=False,
        start_step=7,
        planner_backend="linear",
    )

    assert motion_queue.pending_timeouts == [0.05]
    assert len(hold_calls) == 1
    assert hold_calls[0]["runtime"] is runtime
    assert hold_calls[0]["planner"] is planner
    assert hold_calls[0]["step"] == 7
    assert hold_calls[0]["duration_s"] == pytest.approx(0.05)
    assert hold_calls[0]["should_stop"] == motion_queue.should_stop_current
    assert final_step == 8


def test_single_scene_cli_forwards_explicit_liveness_and_idle_policies(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single-scene-interactive",
            "--stdin-eof-policy",
            "keep_alive",
            "--idle-physics-policy",
            "hold_step",
            "--state-include-efforts",
            "--foxglove-joint-effort-field",
            "measured",
        ],
    )
    args = single_scene_cli.parse_args()
    runtime = SimpleNamespace(close=lambda: None)
    runtime_kwargs: dict[str, object] = {}
    runner_kwargs: dict[str, object] = {}

    def fake_create_single_scene_runtime(**kwargs):
        runtime_kwargs.update(kwargs)
        return runtime

    def fake_run_single_scene_interactive_motion(_runtime, **kwargs) -> int:
        assert _runtime is runtime
        runner_kwargs.update(kwargs)
        return 13

    monkeypatch.setattr(
        single_scene_cli,
        "create_single_scene_runtime",
        fake_create_single_scene_runtime,
    )
    monkeypatch.setattr(
        single_scene_cli,
        "run_single_scene_interactive_motion",
        fake_run_single_scene_interactive_motion,
    )

    assert single_scene_cli.run_interactive_mode(args) == 13
    assert runtime_kwargs["hold_app"] is True
    assert runtime_kwargs["simulation_app"].gui is False
    assert runtime_kwargs["camera_output_settings"].queue_size == 128
    assert runtime_kwargs["shutdown_settings"].transport_timeout_s == 2.0
    assert runtime_kwargs["controller_bundle"] == "default"
    assert runner_kwargs.get("policy") == InteractiveRuntimePolicy(
        stdin_eof_policy="keep_alive",
        idle_physics_policy="hold_step",
    )
    assert runner_kwargs["interactive_settings"].command_history_capacity == 256
    assert runner_kwargs["execution_settings"].idle_step_duration_s == pytest.approx(
        0.05
    )
    assert runner_kwargs["shutdown_settings"].transport_timeout_s == 2.0
    state_stream = runner_kwargs["state_stream_config"]
    assert state_stream.buffer_size == 1
    assert state_stream.drop_policy == "latest"
    assert state_stream.on_error == "stop"
    assert state_stream.include_joint_states is True
    assert state_stream.include_state_json is True
    assert state_stream.include_scene_markers is False
    assert state_stream.include_efforts is True
    assert state_stream.foxglove_joint_effort_field == "measured"
    assert state_stream.topics.joint_states == "/joint_states"
    assert state_stream.topics.scene == "/scene"
    assert state_stream.topics.state == "/linkerbot/state"
    assert state_stream.mcap_existing_file_policy == "error"


def test_single_scene_interactive_explicit_no_hold_skips_idle_advancement(
    monkeypatch,
) -> None:
    motion_queue = _EmptyMotionQueue()
    transport = SimpleNamespace(
        stop=lambda **_kwargs: SimpleNamespace(stopped=True, live_resources=())
    )
    hold_calls: list[object] = []
    monkeypatch.setattr(
        single_scene_runtime,
        "InteractiveMotionQueue",
        lambda **_kwargs: motion_queue,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "TimelinePlanningSession",
        lambda _runtime, *, planner_backend: object(),
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_state_stream",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_transports",
        lambda **_kwargs: transport,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "_hold_all",
        lambda *_args, **_kwargs: hold_calls.append(object()),
    )
    runtime = SimpleNamespace(
        session=SimpleNamespace(app=_IdleOnceApp()),
        status=lambda: {},
        status_prefix="TEST_SINGLE_SCENE_INTERACTIVE",
    )

    final_step = single_scene_runtime.run_single_scene_interactive_motion(
        runtime,
        stdin_enabled=False,
        start_step=7,
        planner_backend="linear",
        policy=InteractiveRuntimePolicy(
            stdin_eof_policy="exit",
            idle_physics_policy="pause",
        ),
    )

    assert motion_queue.pending_timeouts == [0.05]
    assert hold_calls == []
    assert final_step == 7


def test_single_scene_shutdown_stops_ingress_before_state_publisher(
    monkeypatch,
) -> None:
    order: list[str] = []
    motion_queue = _EmptyMotionQueue()
    motion_queue.request_quit = lambda: order.append("queue_quit")
    transport = SimpleNamespace(
        stop=lambda **_kwargs: (
            order.append("transport")
            or SimpleNamespace(stopped=True, live_resources=())
        )
    )
    state_stream = SimpleNamespace(
        close=lambda: order.append("state_stream") or True,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "InteractiveMotionQueue",
        lambda **_kwargs: motion_queue,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "TimelinePlanningSession",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_state_stream",
        lambda *_args, **_kwargs: state_stream,
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_transports",
        lambda **_kwargs: transport,
    )
    runtime = SimpleNamespace(
        session=SimpleNamespace(app=SimpleNamespace(is_running=lambda: False)),
        status=lambda: {},
        status_prefix=None,
    )

    single_scene_runtime.run_single_scene_interactive_motion(
        runtime,
        stdin_enabled=False,
        policy=InteractiveRuntimePolicy(
            stdin_eof_policy="exit",
            idle_physics_policy="pause",
        ),
    )

    assert order == ["queue_quit", "transport", "state_stream"]


def test_single_scene_snapshot_response_identifies_isaac_backend(monkeypatch) -> None:
    snapshot = SimpleNamespace(as_dict=lambda: {"schema": "linkerbot.snapshot"})
    monkeypatch.setattr(
        single_scene_runtime,
        "get_single_scene_snapshot",
        lambda _runtime: snapshot,
    )

    response = single_scene_runtime._handle_snapshot_request(
        object(),
        SimpleNamespace(kind="get_snapshot"),
    )

    assert response == {
        "event": "snapshot",
        "accepted": True,
        "backend": "isaac",
        "snapshot": {"schema": "linkerbot.snapshot"},
    }


@pytest.mark.parametrize(
    ("stdin_eof_policy", "quit_requested"),
    (("exit", True), ("keep_alive", False)),
)
def test_single_scene_hold_controls_stdin_eof_liveness(
    monkeypatch,
    stdin_eof_policy: str,
    quit_requested: bool,
) -> None:
    queue = InteractiveMotionQueue()
    monkeypatch.setattr(sys, "stdin", StringIO(""))

    transports = start_interactive_transports(
        queue=queue,
        stdin_eof_policy=stdin_eof_policy,
    )
    try:
        reader = transports.stdin_reader
        assert reader is not None
        reader.thread.join(timeout=1.0)
        assert not reader.is_alive()
        assert queue.quit_requested() is quit_requested
    finally:
        transports.stop()


def test_single_scene_state_publisher_keeps_process_alive_after_stdin_eof(
    monkeypatch,
) -> None:
    queue = InteractiveMotionQueue()
    monkeypatch.setattr(sys, "stdin", StringIO(""))

    transports = start_interactive_transports(
        queue=queue,
        stdin_eof_policy="exit",
        keepalive_consumer_active=True,
    )
    try:
        reader = transports.stdin_reader
        assert reader is not None
        reader.thread.join(timeout=1.0)
        assert not reader.is_alive()
        assert queue.quit_requested() is False
    finally:
        transports.stop()
