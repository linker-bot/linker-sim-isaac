from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from io import StringIO
from queue import Queue
from threading import Event, Thread
from time import monotonic, sleep

import pytest

import linkerbot_sim.app.interactive.single_scene.transports as transports_module
from linkerbot_sim.app.interactive.single_scene.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.single_scene.transports import (
    InteractiveTransportHandles,
    _TransportMetrics,
    _enqueue_transport_event,
    handle_interactive_message,
    start_interactive_transports,
)


def _timeline_message(command_id: str) -> dict[str, object]:
    return {
        "type": "plan_timeline",
        "id": command_id,
        "tracks": [
            {
                "robot_id": 0,
                "segments": [{"kind": "hold", "duration_s": 0.1}],
            }
        ],
    }


def _wait_for(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def _unused_tcp_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_single_scene_request_capacity_counts_pending_and_running_at_n_plus_one() -> (
    None
):
    queue = InteractiveMotionQueue(request_capacity=1)

    accepted = handle_interactive_message(
        message=_timeline_message("first"),
        queue=queue,
    )
    assert accepted["event"] == "accepted"
    assert queue.status()["queue"]["depth"] == 1

    running = queue.next_pending(timeout_s=0.0)
    assert running is not None
    assert running.command_id == "first"
    rejected = handle_interactive_message(
        message=_timeline_message("second"),
        queue=queue,
    )
    assert rejected == {
        "event": "rejected",
        "accepted": False,
        "code": "request_queue_full",
        "reason": "request_queue_full",
        "error": "request_queue is full (depth=1, capacity=1)",
        "depth": 1,
        "capacity": 1,
        "id": "second",
    }
    status = queue.status()["queue"]
    assert status["active"]["depth"] == 1
    assert status["pending"]["depth"] == 0
    assert status["running"]["depth"] == 1
    assert status["rejected"] == 1

    queue.mark_done("first")
    assert (
        handle_interactive_message(
            message=_timeline_message("second"),
            queue=queue,
        )["event"]
        == "accepted"
    )


def test_terminal_history_evicts_only_terminal_commands() -> None:
    queue = InteractiveMotionQueue(
        request_capacity=2,
        terminal_history_capacity=1,
    )
    for command_id in ("first", "second"):
        assert (
            handle_interactive_message(
                message=_timeline_message(command_id),
                queue=queue,
            )["event"]
            == "accepted"
        )

    first = queue.next_pending(timeout_s=0.0)
    assert first is not None
    queue.mark_done(first.command_id)
    assert {item["id"] for item in queue.status()["commands"]} == {
        "first",
        "second",
    }

    second = queue.next_pending(timeout_s=0.0)
    assert second is not None
    queue.mark_done(second.command_id)
    status = queue.status()
    assert [item["id"] for item in status["commands"]] == ["second"]
    assert status["queue"]["terminal_history"] == {
        "depth": 1,
        "capacity": 1,
        "evicted": 1,
    }


def test_command_registry_stays_bounded_under_sustained_submissions() -> None:
    queue = InteractiveMotionQueue(
        request_capacity=1,
        terminal_history_capacity=3,
    )

    for index in range(1_000):
        command_id = f"command-{index}"
        assert (
            handle_interactive_message(
                message=_timeline_message(command_id),
                queue=queue,
            )["event"]
            == "accepted"
        )
        command = queue.next_pending(timeout_s=0.0)
        assert command is not None
        queue.mark_done(command.command_id)

    status = queue.status()
    assert len(status["commands"]) == 3
    assert status["queue"]["depth"] == 0
    assert status["queue"]["terminal_history"] == {
        "depth": 3,
        "capacity": 3,
        "evicted": 997,
    }


def test_snapshot_capacity_rejects_n_plus_one_and_reports_metrics() -> None:
    queue = InteractiveMotionQueue(snapshot_request_capacity=1)
    result: dict[str, object] = {}

    def request_first() -> None:
        result.update(
            queue.request_snapshot(
                kind="get_snapshot",
                snapshot_id="first",
                timeout_s=1.0,
            )
        )

    thread = Thread(target=request_first)
    thread.start()
    _wait_for(lambda: queue.status()["queue"]["snapshot_requests"]["depth"] == 1)

    rejected = queue.request_snapshot(
        kind="get_snapshot",
        snapshot_id="second",
        timeout_s=1.0,
    )
    assert rejected["reason"] == "snapshot_request_queue_full"
    status = queue.status()["queue"]["snapshot_requests"]
    assert status["depth"] == 1
    assert status["capacity"] == 1
    assert status["rejected"] == 1

    request = queue.consume_snapshot_request()
    assert request is not None
    assert queue.begin_snapshot_request(request)
    status = queue.status()["queue"]["snapshot_requests"]
    assert status["depth"] == 1
    assert status["pending"] == 0
    assert status["running"] == 1
    rejected_while_running = queue.request_snapshot(
        kind="get_snapshot",
        snapshot_id="third",
        timeout_s=1.0,
    )
    assert rejected_while_running["reason"] == "snapshot_request_queue_full"
    queue.mark_snapshot_done(request, {"event": "snapshot"})
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert result == {"event": "snapshot", "id": "first"}
    assert queue.status()["queue"]["snapshot_requests"]["depth"] == 0


def test_snapshot_timeout_after_dequeue_cannot_begin_execution() -> None:
    queue = InteractiveMotionQueue(snapshot_request_capacity=1)
    result: dict[str, object] = {}

    thread = Thread(
        target=lambda: result.update(
            queue.request_snapshot(
                kind="get_snapshot",
                snapshot_id="late",
                timeout_s=0.03,
            )
        )
    )
    thread.start()
    _wait_for(lambda: queue.status()["queue"]["snapshot_requests"]["depth"] == 1)
    request = queue.consume_snapshot_request()
    assert request is not None
    thread.join(timeout=1.0)

    assert result["event"] == "snapshot_timeout"
    assert queue.begin_snapshot_request(request) is False
    queue.mark_snapshot_done(request, {"event": "snapshot"})
    assert request.done is False
    assert queue.status()["queue"]["snapshot_requests"]["timed_out"] == 1


def test_executing_snapshot_waits_for_terminal_result_past_admission_deadline() -> None:
    queue = InteractiveMotionQueue(snapshot_request_capacity=1)
    result: dict[str, object] = {}
    thread = Thread(
        target=lambda: result.update(
            queue.request_snapshot(
                kind="get_snapshot",
                snapshot_id="executing",
                timeout_s=0.02,
            )
        )
    )
    thread.start()
    _wait_for(lambda: queue.status()["queue"]["snapshot_requests"]["depth"] == 1)
    request = queue.consume_snapshot_request()
    assert request is not None
    assert queue.begin_snapshot_request(request)

    sleep(0.04)
    assert thread.is_alive()
    queue.mark_snapshot_done(
        request,
        {"event": "snapshot", "accepted": True},
    )
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == {
        "event": "snapshot",
        "accepted": True,
        "id": "executing",
    }
    status = queue.status()["queue"]["snapshot_requests"]
    assert status["depth"] == 0
    assert status["timed_out"] == 0


def test_timed_out_queued_snapshot_is_not_consumed() -> None:
    queue = InteractiveMotionQueue(
        snapshot_request_capacity=1,
        snapshot_timeout_s=0.01,
    )

    response = queue.request_snapshot(
        kind="get_snapshot",
        snapshot_id="expired",
    )

    assert response["event"] == "snapshot_timeout"
    assert queue.consume_snapshot_request() is None
    assert queue.status()["queue"]["snapshot_requests"]["timeout_s"] == 0.01


def test_reset_request_is_not_silently_overwritten() -> None:
    queue = InteractiveMotionQueue()
    first = handle_interactive_message(
        message={"type": "reset", "id": "first"},
        queue=queue,
    )
    second = handle_interactive_message(
        message={"type": "reset", "id": "second"},
        queue=queue,
    )

    assert first["accepted"] is True
    assert second["event"] == "rejected"
    assert second["reason"] == "reset_request_in_progress"
    request = queue.consume_reset_request()
    assert request is not None
    assert request.reset_id == "first"
    assert queue.status()["queue"]["reset_requests"]["rejected"] == 1

    queue.mark_reset_done("first")
    assert (
        handle_interactive_message(
            message={"type": "reset", "id": "third"},
            queue=queue,
        )["accepted"]
        is True
    )


def test_tcp_limits_have_deterministic_responses_and_metrics() -> None:
    queue = InteractiveMotionQueue()
    message = json.dumps({"type": "status"}, separators=(",", ":"))
    handles = start_interactive_transports(
        queue=queue,
        stdin_enabled=False,
        tcp_jsonl_host="127.0.0.1",
        tcp_jsonl_port=0,
        max_message_bytes=len(message.encode("utf-8")),
        max_connections=1,
        server_poll_interval_s=0.01,
        shutdown_timeout_s=1.0,
    )
    assert handles.tcp_server is not None
    host, port = handles.tcp_server.server_address
    try:
        with socket.create_connection((host, port), timeout=1.0) as client:
            with client.makefile("rwb") as client_file:
                client_file.write((message + "\n").encode("utf-8"))
                client_file.flush()
                assert json.loads(client_file.readline())["event"] == "status"

                oversized = message + " "
                client_file.write((oversized + "\n").encode("utf-8"))
                client_file.flush()
                response = json.loads(client_file.readline())
                assert response["event"] == "rejected"
                assert response["reason"] == "message_too_large"

        _wait_for(lambda: handles.status()["connections"]["depth"] == 0)
        status = handles.status()
        assert status["messages"]["received"] == 2
        assert status["messages"]["rejected"] == 1
        assert status["messages"]["oversized"] == 1
    finally:
        report = handles.stop()
    assert report.stopped
    assert report.live_resources == ()


def test_stdin_reader_applies_message_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue = InteractiveMotionQueue()
    monkeypatch.setattr(sys, "stdin", StringIO('{"message":"too long"}\n'))
    handles = start_interactive_transports(
        queue=queue,
        max_message_bytes=8,
        shutdown_timeout_s=1.0,
    )

    reader = handles.stdin_reader
    assert reader is not None
    reader.thread.join(timeout=1.0)

    assert not reader.is_alive()
    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert responses[0]["reason"] == "message_too_large"
    assert queue.quit_requested()
    assert handles.status()["messages"]["oversized"] == 1
    assert handles.stop().stopped


def test_tcp_bind_failure_does_not_start_unowned_stdin_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdin_started = Event()
    monkeypatch.setattr(
        transports_module,
        "start_interruptible_stdin_jsonl_reader",
        lambda **_kwargs: stdin_started.set(),
    )
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])
        with pytest.raises(OSError):
            start_interactive_transports(
                queue=InteractiveMotionQueue(),
                tcp_jsonl_host="127.0.0.1",
                tcp_jsonl_port=port,
                shutdown_timeout_s=1.0,
            )

    assert not stdin_started.is_set()


def test_websocket_bind_failure_rolls_back_started_tcp_server() -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        websocket_port = int(occupied.getsockname()[1])
        with pytest.raises(
            RuntimeError,
            match="WebSocket server failed during startup",
        ):
            start_interactive_transports(
                queue=InteractiveMotionQueue(),
                stdin_enabled=False,
                tcp_jsonl_host="127.0.0.1",
                tcp_jsonl_port=0,
                websocket_host="127.0.0.1",
                websocket_port=websocket_port,
                server_poll_interval_s=0.01,
                shutdown_timeout_s=1.0,
            )

    assert not any(
        thread.is_alive()
        and thread.name
        in {"interactive-motion-tcp-jsonl", "interactive-motion-websocket"}
        for thread in threading.enumerate()
    )


def test_tcp_connection_limit_rejects_n_plus_one() -> None:
    queue = InteractiveMotionQueue()
    handles = start_interactive_transports(
        queue=queue,
        stdin_enabled=False,
        tcp_jsonl_host="127.0.0.1",
        tcp_jsonl_port=0,
        max_connections=1,
        server_poll_interval_s=0.01,
        shutdown_timeout_s=1.0,
    )
    assert handles.tcp_server is not None
    host, port = handles.tcp_server.server_address
    first = socket.create_connection((host, port), timeout=1.0)
    second: socket.socket | None = None
    try:
        _wait_for(lambda: handles.status()["connections"]["depth"] == 1)
        second = socket.create_connection((host, port), timeout=1.0)
        with second.makefile("rb") as second_file:
            response = json.loads(second_file.readline())
        assert response["event"] == "rejected"
        assert response["reason"] == "max_connections"
        _wait_for(lambda: handles.status()["connections"]["rejected"] == 1)
    finally:
        first.close()
        if second is not None:
            second.close()
        report = handles.stop()
    assert report.stopped


def test_websocket_message_and_connection_limits_are_observable() -> None:
    import websockets

    queue = InteractiveMotionQueue()
    message = json.dumps({"type": "status"}, separators=(",", ":"))
    port = _unused_tcp_port()
    handles = start_interactive_transports(
        queue=queue,
        stdin_enabled=False,
        websocket_host="127.0.0.1",
        websocket_port=port,
        max_message_bytes=len(message.encode("utf-8")),
        max_connections=1,
        server_poll_interval_s=0.01,
        response_poll_interval_s=0.01,
        shutdown_timeout_s=1.0,
    )

    async def connect():
        deadline = monotonic() + 1.0
        while True:
            try:
                return await websockets.connect(f"ws://127.0.0.1:{port}")
            except OSError:
                if monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.005)

    async def exercise() -> None:
        first = await connect()
        try:
            await first.send(message)
            assert json.loads(await first.recv())["event"] == "status"

            await first.send(message + " ")
            oversized = json.loads(await first.recv())
            assert oversized["event"] == "rejected"
            assert oversized["reason"] == "message_too_large"

            second = await connect()
            try:
                rejected = json.loads(await second.recv())
                assert rejected["event"] == "rejected"
                assert rejected["reason"] == "max_connections"
            finally:
                await second.close()

            await first.send(message + "  ")
            with pytest.raises(websockets.exceptions.ConnectionClosedError) as exc:
                await first.recv()
            assert exc.value.code == 1009
        finally:
            await first.close()

    try:
        asyncio.run(exercise())
        _wait_for(lambda: handles.status()["connections"]["depth"] == 0)
        status = handles.status()
        assert status["connections"]["rejected"] == 1
        assert status["messages"]["oversized"] == 2
    finally:
        report = handles.stop()
    assert report.stopped


def test_event_queue_reject_new_is_bounded_and_observable() -> None:
    metrics = _TransportMetrics(
        max_message_bytes=64,
        max_connections=1,
        event_queue_capacity=1,
    )
    event_queue: Queue[dict[str, object]] = Queue(maxsize=1)

    assert _enqueue_transport_event(event_queue, {"event": "first"}, metrics=metrics)
    assert not _enqueue_transport_event(
        event_queue,
        {"event": "second"},
        metrics=metrics,
    )
    assert metrics.status()["events"] == {
        "depth": 1,
        "capacity": 1,
        "aggregate_capacity": 0,
        "rejected": 1,
        "discarded": 0,
    }


def test_stop_reports_threads_still_live_at_deadline() -> None:
    release = Event()
    thread = Thread(
        target=release.wait,
        daemon=True,
        name="blocked-test-transport",
    )
    thread.start()
    handles = InteractiveTransportHandles(
        threads=(thread,),
        stop_event=Event(),
        shutdown_timeout_s=0.0,
    )
    try:
        report = handles.stop()
        assert report.stopped is False
        assert report.live_resources == ("blocked-test-transport",)
        assert handles.threads == (thread,)
    finally:
        release.set()
        thread.join(timeout=1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_message_bytes", 0),
        ("max_connections", 0),
        ("event_queue_capacity", 0),
        ("server_poll_interval_s", 0.0),
        ("response_poll_interval_s", 0.0),
        ("shutdown_timeout_s", -0.1),
    ),
)
def test_transport_limits_reject_zero_or_negative_values(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValueError, match=field):
        start_interactive_transports(
            queue=InteractiveMotionQueue(),
            stdin_enabled=False,
            **{field: value},
        )
