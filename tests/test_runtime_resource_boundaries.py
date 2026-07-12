from __future__ import annotations

import asyncio
from collections.abc import Mapping
import io
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.app.interactive.tiled_scene.transport import (
    BoundedInteractiveRequestQueue,
    SharedTransportAdmission,
    WebSocketServerHandle,
    _InteractiveControl,
    _InteractiveRequest,
    _enqueue_interactive_request,
    _read_bounded_line,
    combined_transport_status,
    run_interactive_loop,
    start_stdin_jsonl_reader,
    start_tcp_jsonl_server,
    start_websocket_server,
    stop_tcp_jsonl_server,
)
from linkerbot_sim.app.interactive.stdin_reader import (
    StdinJsonlFrame,
    start_interruptible_stdin_jsonl_reader,
)
from linkerbot_sim.app.interactive.single_scene.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.single_scene.transports import (
    start_interactive_transports,
)
from linkerbot_sim.app.runtime import (
    single_scene_runtime as single_scene_runtime_module,
)
from linkerbot_sim.app.runtime.single_scene_runtime import SingleSceneRuntime
from linkerbot_sim.app.interactive.single_scene.state_stream import (
    InteractiveStateStreamConfig,
    _foxglove_sinks,
    start_interactive_state_stream,
)
from linkerbot_sim.sensors.camera.frame import CameraFrame
from linkerbot_sim.sensors.camera.recorder import CameraFramePublisher
from linkerbot_sim.telemetry.foxglove_state import StatePublisher
from linkerbot_sim.telemetry.foxglove import FoxgloveTopicConfig
from linkerbot_sim.telemetry.state_snapshot import StateSnapshot, StateStream
from linkerbot_sim.tiled.planning.linear_backend import LinearJointPlannerBackend
from linkerbot_sim.tiled.planning.manager import TiledPlannerManager
from linkerbot_sim.tiled.planning.types import TiledPlanningRequest


def _planning_request(request_id: str) -> TiledPlanningRequest:
    return TiledPlanningRequest(
        request_id=request_id,
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0]]),
        goal_positions=np.asarray([[1.0]]),
        joint_names=("j0",),
        duration_s=0.1,
        sample_dt_s=0.05,
    )


def test_single_scene_status_includes_camera_and_telemetry_metrics() -> None:
    camera_status = {
        "queue_depth": 3,
        "queue_capacity": 8,
        "dropped_frames": 5,
    }
    runtime = SimpleNamespace(
        robot_registry=SimpleNamespace(robots_by_id={}),
        planning_registry=SimpleNamespace(metrics=lambda: {}),
        collision_registry=SimpleNamespace(metrics=lambda: {}),
        object_state_views={},
        config_fingerprint="unit-scene",
        camera_output=SimpleNamespace(
            publisher=SimpleNamespace(status=lambda: camera_status)
        ),
        telemetry_status_provider=lambda: {
            "dropped_snapshots": 4,
            "error_count": 2,
            "last_published_sequence": 19,
        },
    )

    status = SingleSceneRuntime.status(runtime)

    assert status["camera_output"] == camera_status
    assert status["telemetry"] == {
        "dropped_snapshots": 4,
        "error_count": 2,
        "last_published_sequence": 19,
    }


def test_single_scene_runtime_timeout_retains_owner_and_retries_before_closing_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RetryCamera:
        def __init__(self) -> None:
            self.attempts = 0
            self.publisher = SimpleNamespace(status=lambda: {"thread_alive": True})

        def close(self) -> bool:
            self.attempts += 1
            calls.append("camera")
            return self.attempts > 1

    runtime = SingleSceneRuntime(
        session=SimpleNamespace(app=object()),
        env_config={},
        robot_registry=SimpleNamespace(robots_by_id={}),
        planning_registry=SimpleNamespace(close=lambda: calls.append("planning")),
        collision_registry=SimpleNamespace(),
        camera_output=RetryCamera(),
        loggers=(SimpleNamespace(close=lambda: calls.append("logger")),),
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.runtime.single_scene_runtime.close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )

    first = runtime.close()
    assert first.stopped is False
    assert first.live_resources == ("camera_output",)
    assert runtime._closed is False
    assert calls == ["planning", "camera", "logger"]

    second = runtime.close()
    assert second.stopped is True
    assert runtime._closed is True
    assert calls == ["planning", "camera", "logger", "camera", "simulation_app"]


def test_single_scene_runtime_retries_retained_state_stream_before_dependencies() -> (
    None
):
    calls: list[str] = []

    class RetryStateStream:
        def __init__(self) -> None:
            self.attempts = 0

        def close(self) -> bool:
            self.attempts += 1
            calls.append("state_stream")
            return self.attempts > 1

    runtime = SingleSceneRuntime(
        session=SimpleNamespace(app=SimpleNamespace(close=lambda: None)),
        env_config={},
        robot_registry=SimpleNamespace(robots_by_id={}),
        planning_registry=SimpleNamespace(close=lambda: calls.append("planning")),
        collision_registry=SimpleNamespace(),
    )
    runtime.retain_shutdown_resource("state_stream", RetryStateStream())

    assert runtime.close().live_resources == ("state_stream",)
    assert calls == ["state_stream"]
    assert runtime.close().stopped is True
    assert calls == ["state_stream", "state_stream", "planning"]


def test_single_scene_runtime_retained_close_error_does_not_skip_remaining_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RetriableResource:
        def __init__(self) -> None:
            self.attempts = 0

        def close(self) -> bool:
            self.attempts += 1
            calls.append("failing_resource")
            if self.attempts == 1:
                raise RuntimeError("retained close failed")
            return True

    failing = RetriableResource()
    runtime = SingleSceneRuntime(
        session=SimpleNamespace(app=object()),
        env_config={},
        robot_registry=SimpleNamespace(robots_by_id={}),
        planning_registry=SimpleNamespace(close=lambda: calls.append("planning")),
        collision_registry=SimpleNamespace(),
        loggers=(SimpleNamespace(close=lambda: calls.append("logger")),),
    )
    runtime.retain_shutdown_resource("failing", failing)
    runtime.retain_shutdown_resource(
        "stable",
        SimpleNamespace(close=lambda: calls.append("stable_resource") or True),
    )
    monkeypatch.setattr(
        single_scene_runtime_module,
        "close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )

    with pytest.raises(RuntimeError, match="retained close failed"):
        runtime.close()

    assert calls == [
        "failing_resource",
        "stable_resource",
        "planning",
        "logger",
    ]
    assert tuple(runtime.shutdown_resources) == ("failing",)

    assert runtime.close().stopped is True
    assert calls[-2:] == ["failing_resource", "simulation_app"]


def test_state_stream_start_failure_restores_runtime_and_closes_every_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls: list[str] = []

    class Sink:
        def __init__(self, name: str, *, fail_close: bool) -> None:
            self.name = name
            self.fail_close = fail_close

        def publish(self, _snapshot: StateSnapshot) -> None:
            pass

        def close(self) -> None:
            close_calls.append(self.name)
            if self.fail_close:
                raise RuntimeError("sink cleanup failed")

    class FailingThread:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("state thread start failed")

    previous_observer = object()

    def previous_status() -> dict[str, object]:
        return {"owner": "previous"}

    runtime = SimpleNamespace(
        robots_by_id={},
        robot_registry=object(),
        object_handles=(),
        session=SimpleNamespace(stage=None),
        state_observer=previous_observer,
        telemetry_status_provider=previous_status,
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.single_scene.state_stream._foxglove_sinks",
        lambda _config: [
            Sink("failing", fail_close=True),
            Sink("stable", fail_close=False),
        ],
    )
    monkeypatch.setattr(
        "linkerbot_sim.telemetry.foxglove_state.Thread",
        FailingThread,
    )

    with pytest.raises(RuntimeError, match="state thread start failed"):
        start_interactive_state_stream(
            runtime,
            config=InteractiveStateStreamConfig(foxglove_live_port=8765),
        )

    assert close_calls == ["failing", "stable"]
    assert runtime.state_observer is previous_observer
    assert runtime.telemetry_status_provider is previous_status


def test_single_scene_runtime_logger_failure_does_not_skip_peers_or_close_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RetriableLogger:
        def __init__(self, name: str, *, fail_once: bool) -> None:
            self.name = name
            self.fail_once = fail_once
            self.attempts = 0

        def close(self) -> None:
            self.attempts += 1
            calls.append(self.name)
            if self.fail_once and self.attempts == 1:
                raise RuntimeError(f"{self.name} close failed")

    failing = RetriableLogger("failing_logger", fail_once=True)
    stable = RetriableLogger("stable_logger", fail_once=False)
    runtime = SingleSceneRuntime(
        session=SimpleNamespace(app=object()),
        env_config={},
        robot_registry=SimpleNamespace(robots_by_id={}),
        planning_registry=SimpleNamespace(close=lambda: calls.append("planning")),
        collision_registry=SimpleNamespace(),
        loggers=(failing, stable),
    )
    monkeypatch.setattr(
        single_scene_runtime_module,
        "close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )

    with pytest.raises(RuntimeError, match="failing_logger close failed"):
        runtime.close()
    assert calls == [
        "planning",
        "failing_logger",
        "stable_logger",
    ]
    assert runtime._closed is False

    assert runtime.close().stopped is True
    assert calls == [
        "planning",
        "failing_logger",
        "stable_logger",
        "failing_logger",
        "simulation_app",
    ]


def test_single_scene_runtime_planning_failure_closes_peers_but_keeps_app_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RetriablePlanning:
        def __init__(self) -> None:
            self.attempts = 0

        def close(self) -> None:
            self.attempts += 1
            calls.append("planning")
            if self.attempts == 1:
                raise RuntimeError("planning close failed")

    runtime = SingleSceneRuntime(
        session=SimpleNamespace(app=object()),
        env_config={},
        robot_registry=SimpleNamespace(robots_by_id={}),
        planning_registry=RetriablePlanning(),
        collision_registry=SimpleNamespace(),
        camera_output=SimpleNamespace(close=lambda: calls.append("camera") or True),
        loggers=(SimpleNamespace(close=lambda: calls.append("logger")),),
    )
    monkeypatch.setattr(
        single_scene_runtime_module,
        "close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )

    with pytest.raises(RuntimeError, match="planning close failed"):
        runtime.close()

    assert calls == ["planning", "camera", "logger"]
    assert runtime._app_closed is False

    assert runtime.close().stopped is True
    assert calls == ["planning", "camera", "logger", "planning", "simulation_app"]


def test_failed_scene_creation_cleanup_attempts_every_owned_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    monkeypatch.setattr(
        single_scene_runtime_module,
        "close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )

    single_scene_runtime_module._cleanup_failed_single_scene_runtime(
        planning_registry=Resource("planning", fail=True),
        camera_output=Resource("camera", fail=True),
        loggers=(
            Resource("failing_logger", fail=True),
            Resource("stable_logger", fail=False),
        ),
        app=object(),
    )

    assert calls == [
        "planning",
        "camera",
        "failing_logger",
        "stable_logger",
        "simulation_app",
    ]


def test_state_stream_rejects_wrong_runtime_before_opening_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.single_scene.state_stream.FoxgloveStateSink.open_live",
        lambda **_kwargs: opened.append("live"),
    )

    with pytest.raises(ValueError, match="requires SingleSceneRuntime"):
        start_interactive_state_stream(
            object(),
            config=InteractiveStateStreamConfig(foxglove_live_port=8765),
        )

    assert opened == []


def test_state_stream_sink_factory_rolls_back_first_sink_when_second_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = SimpleNamespace(close_attempted=False)

    def fail_first_close() -> None:
        first.close_attempted = True
        raise RuntimeError("live close failed")

    first.close = fail_first_close
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.single_scene.state_stream.FoxgloveStateSink.open_live",
        lambda **_kwargs: first,
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.single_scene.state_stream.FoxgloveStateSink.open_mcap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mcap failed")),
    )

    with pytest.raises(RuntimeError, match="mcap failed"):
        _foxglove_sinks(
            InteractiveStateStreamConfig(
                foxglove_live_port=8765,
                foxglove_mcap_path="state.mcap",
            )
        )

    assert first.close_attempted is True


def test_state_stream_mcap_preflight_fails_before_live_sink_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.mcap"
    target.write_bytes(b"existing")
    opened: list[str] = []
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.single_scene.state_stream.FoxgloveStateSink.open_live",
        lambda **_kwargs: opened.append("live"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        _foxglove_sinks(
            InteractiveStateStreamConfig(
                foxglove_live_port=8765,
                foxglove_mcap_path=target,
                mcap_existing_file_policy="error",
            )
        )

    assert opened == []
    assert target.read_bytes() == b"existing"


def test_state_stream_mcap_resume_is_rejected_before_any_sink_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.single_scene.state_stream.FoxgloveStateSink.open_live",
        lambda **_kwargs: opened.append("live"),
    )

    with pytest.raises(ValueError, match="cannot append"):
        _foxglove_sinks(
            InteractiveStateStreamConfig(
                foxglove_live_port=8765,
                foxglove_mcap_path=tmp_path / "state.mcap",
                mcap_existing_file_policy="resume",
            )
        )

    assert opened == []


def test_state_stream_forwards_runtime_topics_and_modalities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sink = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.single_scene.state_stream.FoxgloveStateSink.open_live",
        lambda **kwargs: calls.append(kwargs) or sink,
    )
    topics = FoxgloveTopicConfig(
        joint_states="/instance/joints",
        scene="/instance/markers",
        state="/instance/state",
    )

    assert _foxglove_sinks(
        InteractiveStateStreamConfig(
            foxglove_live_port=8765,
            include_joint_states=False,
            include_state_json=True,
            include_scene_markers=False,
            topics=topics,
        )
    ) == [sink]

    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 8765,
            "joint_effort_field": "none",
            "publish_joint_states": False,
            "publish_state_json": True,
            "publish_scene_markers": False,
            "topics": topics,
        }
    ]


def test_interactive_request_queue_rejects_n_plus_one_without_blocking() -> None:
    requests = BoundedInteractiveRequestQueue(capacity=1)
    first = _InteractiveRequest(line="{}", source="test")
    second = _InteractiveRequest(line="{}", source="test")

    assert _enqueue_interactive_request(requests, first) is None
    rejection = _enqueue_interactive_request(requests, second)

    assert rejection == {
        "event": "rejected",
        "error": "interactive request queue is full",
        "code": "request_queue_full",
        "capacity": 1,
    }
    assert requests.get_nowait() is first
    assert requests.status() == {
        "request_queue_depth": 0,
        "request_queue_capacity": 1,
        "pending_requests": 0,
        "running_requests": 1,
        "active_requests": 1,
        "outstanding_requests": 1,
        "rejected_requests": 1,
    }
    assert _enqueue_interactive_request(requests, second)["code"] == (
        "request_queue_full"
    )
    requests.task_done()
    assert _enqueue_interactive_request(requests, second) is None
    assert requests.get_nowait() is second
    requests.task_done()
    assert requests.status()["active_requests"] == 0
    assert requests.status()["rejected_requests"] == 2


def test_interactive_loop_adds_transport_status_and_publishes_response() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.quit_event = threading.Event()

        def status(self) -> dict[str, object]:
            return {"event": "status", "num_envs": 1}

        def quit(self) -> dict[str, object]:
            self.quit_event.set()
            return {"event": "quit", "accepted": True}

    runtime = Runtime()
    requests = BoundedInteractiveRequestQueue(capacity=2)
    response_queue: queue.Queue[dict[str, object]] = queue.Queue()
    requests.put_nowait(
        _InteractiveRequest(
            line='{ "type": "status" }',
            source="test",
            response_queue=response_queue,
        )
    )
    requests.put_nowait(_InteractiveRequest(line='{ "type": "quit" }', source="test"))
    events: list[Mapping[str, object]] = []

    run_interactive_loop(
        runtime,
        telemetry=None,
        request_queue=requests,
        telemetry_rate_hz=0.0,
        queue_poll_timeout_s=0.01,
        event_publisher=lambda event: not events.append(dict(event)),
        transport_status_provider=lambda: combined_transport_status(requests),
    )

    response = response_queue.get_nowait()
    assert response["transport"]["request_queue_capacity"] == 2
    assert response["transport"]["running_requests"] == 1
    assert response["transport"]["active_requests"] == 2
    assert [event["event"] for event in events] == ["status", "quit"]
    assert requests.status()["active_requests"] == 0


def test_stdin_eof_control_survives_full_data_queue(monkeypatch) -> None:
    requests = BoundedInteractiveRequestQueue(capacity=1)
    data_request = _InteractiveRequest(line="{}", source="existing")
    requests.put_nowait(data_request)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    reader = start_stdin_jsonl_reader(requests, quit_on_eof=True)
    reader.thread.join(timeout=1.0)

    assert not reader.is_alive()
    assert requests.qsize() == 2
    assert requests.full() is True
    assert requests.get_nowait() is data_request
    requests.task_done()
    assert requests.full() is False
    control = requests.get_nowait()
    assert isinstance(control, _InteractiveControl)
    assert control.kind == "stdin_eof"
    requests.task_done()
    requests.join()
    assert requests.unfinished_tasks == 0


def test_stdin_fallback_rejects_oversized_line_with_bounded_read(
    monkeypatch, capsys
) -> None:
    requests = BoundedInteractiveRequestQueue(capacity=1)
    admission = SharedTransportAdmission(max_connections=1)
    monkeypatch.setattr(sys, "stdin", io.StringIO("x" * 100 + "\n"))

    reader = start_stdin_jsonl_reader(
        requests,
        quit_on_eof=True,
        max_message_bytes=8,
        admission=admission,
    )
    reader.thread.join(timeout=1.0)

    assert not reader.is_alive()
    assert "message_too_large" in capsys.readouterr().out
    assert admission.status()["messages_received"] == 1
    assert admission.status()["messages_rejected"] == 1
    assert admission.status()["oversized_messages"] == 1
    control = requests.get_nowait()
    assert isinstance(control, _InteractiveControl)
    requests.task_done()


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n"])
def test_tiled_scene_jsonl_limit_counts_payload_without_line_ending(
    terminator: bytes,
) -> None:
    at_limit = io.BytesIO(b"x" * 8 + terminator)
    above_limit = io.BytesIO(b"x" * 9 + terminator)

    assert _read_bounded_line(at_limit, max_message_bytes=8) == (b"x" * 8, False)
    assert _read_bounded_line(above_limit, max_message_bytes=8) == (
        b"oversized",
        True,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("x" * 8, StdinJsonlFrame(text="x" * 8)),
        ("x" * 9, StdinJsonlFrame(text=None, oversized=True)),
    ],
)
def test_stdin_reader_uses_payload_byte_boundary(
    payload: str,
    expected: StdinJsonlFrame,
) -> None:
    frames: list[StdinJsonlFrame] = []
    eof = threading.Event()

    reader = start_interruptible_stdin_jsonl_reader(
        stream=io.StringIO(payload + "\n"),
        max_message_bytes=8,
        on_frame=frames.append,
        on_eof=eof.set,
        thread_name="test-stdin-jsonl-reader",
    )
    reader.thread.join(timeout=1.0)

    assert not reader.is_alive()
    assert eof.is_set()
    assert frames == [expected]


def test_stdin_reader_supports_bytes_io_and_rejects_invalid_utf8() -> None:
    frames: list[StdinJsonlFrame] = []
    eof = threading.Event()

    reader = start_interruptible_stdin_jsonl_reader(
        stream=io.BytesIO(b"ok\n\xff\n"),
        max_message_bytes=8,
        on_frame=frames.append,
        on_eof=eof.set,
        thread_name="test-binary-stdin-jsonl-reader",
    )
    reader.thread.join(timeout=1.0)

    assert not reader.is_alive()
    assert eof.is_set()
    assert frames[0] == StdinJsonlFrame(text="ok")
    assert frames[1].text is None
    assert frames[1].oversized is False
    assert isinstance(frames[1].decode_error, UnicodeDecodeError)


def test_stdin_reader_discards_one_oversized_frame_across_read_chunks() -> None:
    frames: list[StdinJsonlFrame] = []

    reader = start_interruptible_stdin_jsonl_reader(
        stream=io.BytesIO(b"x" * 40 + b"\nnext\n"),
        max_message_bytes=8,
        on_frame=frames.append,
        on_eof=lambda: None,
        thread_name="test-chunked-stdin-jsonl-reader",
    )
    reader.thread.join(timeout=1.0)

    assert not reader.is_alive()
    assert frames == [
        StdinJsonlFrame(text=None, oversized=True),
        StdinJsonlFrame(text="next"),
    ]


def test_stdin_reader_stop_interrupts_blocked_pipe_without_closing_source() -> None:
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", closefd=True)
    try:
        reader = start_interruptible_stdin_jsonl_reader(
            stream=stream,
            max_message_bytes=8,
            on_frame=lambda _frame: None,
            on_eof=lambda: None,
            thread_name="test-blocked-stdin-jsonl-reader",
        )

        assert reader.stop(timeout_s=1.0) is True
        assert not reader.is_alive()
        assert stream.closed is False
    finally:
        stream.close()
        os.close(write_fd)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10"])
def test_transport_start_boundaries_reject_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        start_interactive_transports(
            queue=InteractiveMotionQueue(),
            stdin_enabled=False,
            tcp_jsonl_host=host,
            tcp_jsonl_port=0,
        )

    requests: queue.Queue[object] = queue.Queue()
    with pytest.raises(ValueError, match="loopback"):
        start_tcp_jsonl_server(
            requests,
            quit_event=None,
            host=host,
            port=0,
        )
    with pytest.raises(ValueError, match="loopback"):
        start_websocket_server(
            requests,
            quit_event=None,
            host=host,
            port=0,
        )


def test_tcp_jsonl_rejects_oversized_message_before_queueing() -> None:
    requests: queue.Queue[object] = queue.Queue(maxsize=1)
    server = start_tcp_jsonl_server(
        requests,
        quit_event=None,
        host="127.0.0.1",
        port=0,
        max_message_bytes=8,
        server_poll_interval_s=0.01,
        response_poll_interval_s=0.01,
    )
    try:
        with socket.create_connection(server.server_address, timeout=1.0) as client:
            client.sendall(b'{"message":"too long"}\n')
            response = json.loads(client.makefile("rb").readline())
        assert response["event"] == "rejected"
        assert response["code"] == "message_too_large"
        assert requests.empty()
        metrics = server.status()
        assert metrics["messages_received"] == 1
        assert metrics["messages_rejected"] == 1
        assert metrics["oversized_messages"] == 1
    finally:
        status = stop_tcp_jsonl_server(server, timeout_s=1.0)
        assert status["server_closed"] is True
        assert status["serve_thread_alive"] is False


def test_tcp_jsonl_rejects_connection_above_limit() -> None:
    requests: queue.Queue[object] = queue.Queue(maxsize=1)
    server = start_tcp_jsonl_server(
        requests,
        quit_event=None,
        host="127.0.0.1",
        port=0,
        max_connections=1,
        server_poll_interval_s=0.01,
    )
    first = socket.create_connection(server.server_address, timeout=1.0)
    try:
        deadline = time.monotonic() + 1.0
        while server.status()["active_connections"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        with socket.create_connection(server.server_address, timeout=1.0) as second:
            response = json.loads(second.makefile("rb").readline())
        assert response["code"] == "connection_limit"
        assert server.status()["rejected_connections"] == 1
    finally:
        status = stop_tcp_jsonl_server(server, timeout_s=1.0)
        assert status["live_socket_count"] == 0
        assert status["live_handler_thread_count"] == 0
        first.close()


def test_websocket_start_timeout_stops_unreturned_server_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads: list[threading.Thread] = []

    async def serve_without_becoming_ready(self: WebSocketServerHandle) -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._stop_event = stop_event
        threads.append(threading.current_thread())
        await stop_event.wait()
        assert self._stop_requested.is_set()

    monkeypatch.setattr(WebSocketServerHandle, "_serve", serve_without_becoming_ready)

    with pytest.raises(
        TimeoutError,
        match="WebSocket server did not start before timeout",
    ):
        start_websocket_server(
            queue.Queue(),
            quit_event=None,
            host="127.0.0.1",
            port=0,
            startup_timeout_s=0.25,
        )

    assert len(threads) == 1
    assert not threads[0].is_alive()


def test_websocket_stop_before_loop_prevents_late_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import websockets

    entered_thread = threading.Event()
    release_thread = threading.Event()
    bind_attempted = threading.Event()
    threads: list[threading.Thread] = []
    original_run = WebSocketServerHandle._run

    def delayed_run(self: WebSocketServerHandle) -> None:
        threads.append(threading.current_thread())
        entered_thread.set()
        assert release_thread.wait(timeout=1.0)
        original_run(self)

    def unexpected_serve(*_args: object, **_kwargs: object) -> object:
        bind_attempted.set()
        raise AssertionError("server attempted to bind after startup cleanup")

    monkeypatch.setattr(WebSocketServerHandle, "_run", delayed_run)
    monkeypatch.setattr(websockets, "serve", unexpected_serve)

    with pytest.raises(
        TimeoutError,
        match="WebSocket server did not start before timeout",
    ) as caught:
        start_websocket_server(
            queue.Queue(),
            quit_event=None,
            host="127.0.0.1",
            port=0,
            startup_timeout_s=0.05,
        )

    assert entered_thread.is_set()
    assert caught.value.__notes__ == [
        "WebSocket startup cleanup timed out; the server thread is still alive"
    ]
    release_thread.set()
    threads[0].join(timeout=1.0)
    assert not threads[0].is_alive()
    assert not bind_attempted.is_set()


def test_websocket_start_cleanup_error_does_not_replace_start_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_error = RuntimeError("startup exploded")

    def fail_start(
        _self: WebSocketServerHandle,
        *,
        startup_timeout_s: float,
    ) -> None:
        assert startup_timeout_s == pytest.approx(0.25)
        raise start_error

    def fail_stop(
        _self: WebSocketServerHandle,
        *,
        timeout_s: float,
    ) -> dict[str, object]:
        assert timeout_s == pytest.approx(0.25)
        raise OSError("cleanup exploded")

    monkeypatch.setattr(WebSocketServerHandle, "start", fail_start)
    monkeypatch.setattr(WebSocketServerHandle, "stop", fail_stop)

    with pytest.raises(RuntimeError, match="startup exploded") as caught:
        start_websocket_server(
            queue.Queue(),
            quit_event=None,
            host="127.0.0.1",
            port=0,
            startup_timeout_s=0.25,
        )

    assert caught.value is start_error
    assert caught.value.__notes__ == [
        "WebSocket startup cleanup failed: OSError: cleanup exploded"
    ]


def test_websocket_adapter_queues_request_and_returns_main_thread_response() -> None:
    requests = BoundedInteractiveRequestQueue(capacity=1)
    server = start_websocket_server(
        requests,
        quit_event=None,
        host="127.0.0.1",
        port=0,
        server_poll_interval_s=0.01,
        response_poll_interval_s=0.01,
    )

    def _respond() -> None:
        request = requests.get(timeout=1.0)
        request.response_queue.put({"event": "status", "source": request.source})
        requests.task_done()

    responder = threading.Thread(target=_respond)
    responder.start()

    async def _client() -> dict[str, object]:
        import websockets

        async with websockets.connect(
            f"ws://127.0.0.1:{server.bound_port}"
        ) as websocket:
            await websocket.send('{"type":"status"}')
            return json.loads(await websocket.recv())

    try:
        assert asyncio.run(_client()) == {
            "event": "status",
            "source": "websocket",
        }
        responder.join(timeout=1.0)
        assert not responder.is_alive()
        assert requests.status()["active_requests"] == 0
    finally:
        assert server.stop(timeout_s=1.0)["thread_alive"] is False


def test_websocket_adapter_rejects_full_request_and_event_queues() -> None:
    requests: queue.Queue[object] = queue.Queue(maxsize=1)
    requests.put_nowait(_InteractiveRequest(line="{}", source="existing"))
    server = start_websocket_server(
        requests,
        quit_event=None,
        host="127.0.0.1",
        port=0,
        event_queue_capacity=1,
        server_poll_interval_s=0.01,
    )

    async def _client() -> dict[str, object]:
        import websockets

        async with websockets.connect(
            f"ws://127.0.0.1:{server.bound_port}"
        ) as websocket:
            await websocket.send("{}")
            return json.loads(await websocket.recv())

    try:
        assert asyncio.run(_client())["code"] == "request_queue_full"
        assert server.publish_event({"event": "first"}) is True
        assert server.publish_event({"event": "second"}) is False
        assert server.status()["rejected_events"] == 1
    finally:
        server.stop(timeout_s=1.0)


def test_websocket_protocol_closes_huge_message_with_1009_and_metrics() -> None:
    requests = BoundedInteractiveRequestQueue(capacity=1)
    admission = SharedTransportAdmission(max_connections=1)
    server = start_websocket_server(
        requests,
        quit_event=None,
        host="127.0.0.1",
        port=0,
        max_message_bytes=32,
        admission=admission,
        server_poll_interval_s=0.01,
    )

    async def _client() -> int:
        import websockets
        from websockets.exceptions import ConnectionClosedError

        async with websockets.connect(
            f"ws://127.0.0.1:{server.bound_port}"
        ) as websocket:
            await websocket.send("x" * 100_000)
            with pytest.raises(ConnectionClosedError) as caught:
                await websocket.recv()
            return int(caught.value.code)

    try:
        assert asyncio.run(_client()) == 1009
        deadline = time.monotonic() + 1.0
        while admission.status()["oversized_messages"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert admission.status()["messages_received"] == 1
        assert admission.status()["messages_rejected"] == 1
        assert requests.empty()
    finally:
        server.stop(timeout_s=1.0)


def test_websocket_stop_is_idempotent_after_quit_event_closed_loop() -> None:
    requests = BoundedInteractiveRequestQueue(capacity=1)
    quit_event = threading.Event()
    server = start_websocket_server(
        requests,
        quit_event=quit_event,
        host="127.0.0.1",
        port=0,
        server_poll_interval_s=0.01,
    )

    quit_event.set()
    assert server.thread is not None
    server.thread.join(timeout=1.0)
    assert not server.thread.is_alive()

    status = server.stop(timeout_s=1.0)
    assert status["thread_alive"] is False
    assert status["shutdown_timed_out"] is False


def test_tcp_and_websocket_share_process_connection_limit() -> None:
    requests = BoundedInteractiveRequestQueue(capacity=1)
    admission = SharedTransportAdmission(max_connections=1)
    tcp_server = start_tcp_jsonl_server(
        requests,
        quit_event=None,
        host="127.0.0.1",
        port=0,
        admission=admission,
        server_poll_interval_s=0.01,
    )
    websocket_server = start_websocket_server(
        requests,
        quit_event=None,
        host="127.0.0.1",
        port=0,
        admission=admission,
        server_poll_interval_s=0.01,
    )
    tcp_client = socket.create_connection(tcp_server.server_address, timeout=1.0)

    async def _client() -> dict[str, object]:
        import websockets

        async with websockets.connect(
            f"ws://127.0.0.1:{websocket_server.bound_port}"
        ) as websocket:
            return json.loads(await websocket.recv())

    try:
        deadline = time.monotonic() + 1.0
        while admission.status()["active_connections"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert asyncio.run(_client())["code"] == "connection_limit"
        status = combined_transport_status(
            requests,
            tcp_server=tcp_server,
            websocket_server=websocket_server,
            admission=admission,
        )
        assert status["admission"]["active_connections"] == 1
        assert status["admission"]["rejected_connections"] == 1
        assert status["admission"]["max_connections"] == 1
    finally:
        websocket_server.stop(timeout_s=1.0)
        stop_tcp_jsonl_server(tcp_server, timeout_s=1.0)
        tcp_client.close()


def test_planner_status_counts_rejections_evictions_and_workers() -> None:
    manager = TiledPlannerManager(
        max_workers=1,
        max_pending_requests=1,
        max_completed_results=1,
    )
    try:
        manager.submit(_planning_request("first"))
        with pytest.raises(RuntimeError, match="too many pending"):
            manager.submit(_planning_request("rejected"))
        manager.collect_ready(timeout_s=1.0)
        manager.submit(_planning_request("second"))
        manager.collect_ready(timeout_s=1.0)

        status = manager.status()
        assert status["max_workers"] == 1
        assert status["rejected_requests"] == 1
        assert status["evicted_completed_results"] == 1
        assert [item["request_id"] for item in status["completed"]] == ["second"]
    finally:
        manager.shutdown()


def test_planner_shutdown_timeout_preserves_live_future_diagnostics() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBackend:
        def plan(self, request):
            started.set()
            release.wait(timeout=2.0)
            return LinearJointPlannerBackend().plan(request)

    manager = TiledPlannerManager(
        backend=BlockingBackend(),
        max_workers=1,
        shutdown_timeout_s=0.0,
    )
    manager.submit(_planning_request("live"))
    manager.collect_ready()
    assert started.wait(timeout=1.0)

    shutdown = manager.shutdown()

    assert shutdown == {
        "shutdown_timed_out": True,
        "live_request_ids": ["live"],
    }
    assert manager.status()["live_request_ids"] == ["live"]
    assert manager.status()["shutdown_timed_out"] is True
    release.set()
    deadline = time.monotonic() + 1.0
    while manager.status()["live_request_ids"]:
        assert time.monotonic() < deadline
        manager.collect_ready()
        time.sleep(0.001)
    assert manager.shutdown(wait_timeout_s=0.1)["live_request_ids"] == []


class _BlockingSink:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def publish(self, _value: object) -> None:
        self.entered.set()
        self.release.wait(timeout=2.0)

    def close(self) -> None:
        self.closed = True


def test_state_publisher_timeout_retains_live_thread_and_open_sink() -> None:
    stream = StateStream()
    sink = _BlockingSink()
    publisher = StatePublisher(stream=stream, sink=sink, shutdown_timeout_s=0.0)
    publisher.start()
    stream.publish(StateSnapshot(step=0, time_s=0.0, robots=()))
    assert sink.entered.wait(timeout=1.0)

    assert publisher.close() is False
    assert publisher.thread is not None
    assert publisher.status()["thread_alive"] is True
    assert publisher.status()["shutdown_timed_out"] is True
    assert sink.closed is False

    sink.release.set()
    assert publisher.close(timeout_s=1.0) is True
    assert publisher.thread is None
    assert sink.closed is True


def test_camera_publisher_timeout_retains_live_thread_and_open_sink() -> None:
    sink = _BlockingSink()
    publisher = CameraFramePublisher(
        sink=sink,
        max_queue_size=1,
        shutdown_timeout_s=0.0,
    )
    publisher.start()
    publisher.publish(
        CameraFrame(
            camera_name="camera",
            modality="depth",
            frame_index=0,
            simulation_step=0,
            time_s=0.0,
            data=np.zeros((1, 1), dtype=np.float32),
        )
    )
    assert sink.entered.wait(timeout=1.0)

    assert publisher.close() is False
    assert publisher.thread is not None
    assert publisher.status()["thread_alive"] is True
    assert publisher.status()["shutdown_timed_out"] is True
    assert sink.closed is False

    sink.release.set()
    assert publisher.close(timeout_s=1.0) is True
    assert publisher.thread is None
    assert sink.closed is True
