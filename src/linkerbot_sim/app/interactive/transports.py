"""Interactive motion transports for stdin, TCP JSONL, and WebSocket JSON."""

from __future__ import annotations

import asyncio
import json
import socketserver
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread

from linkerbot_sim.app.interactive.protocol import (
    InteractiveMotionCommand,
    parse_interactive_motion_message,
)
from linkerbot_sim.app.interactive.queue import InteractiveMotionQueue


@dataclass
class InteractiveTransportHandles:
    """Started background transports."""

    threads: tuple[Thread, ...]
    stop_event: Event
    tcp_server: socketserver.ThreadingTCPServer | None = None

    def stop(self) -> None:
        self.stop_event.set()
        if self.tcp_server is not None:
            self.tcp_server.shutdown()
            self.tcp_server.server_close()


def start_interactive_transports(
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    stdin_enabled: bool = True,
    tcp_jsonl_host: str | None = None,
    tcp_jsonl_port: int | None = None,
    websocket_host: str | None = None,
    websocket_port: int | None = None,
) -> InteractiveTransportHandles:
    """Start requested transports in background threads."""

    stop_event = Event()
    threads: list[Thread] = []
    tcp_server = None
    if stdin_enabled:
        thread = Thread(
            target=_stdin_jsonl_reader,
            kwargs={
                "queue": queue,
                "default_tcp_by_side": default_tcp_by_side,
                "stop_event": stop_event,
                "quit_on_eof": tcp_jsonl_port is None and websocket_port is None,
            },
            daemon=True,
            name="interactive-motion-stdin",
        )
        thread.start()
        threads.append(thread)
    if tcp_jsonl_port is not None:
        tcp_server = _start_tcp_jsonl_server(
            queue=queue,
            default_tcp_by_side=default_tcp_by_side,
            host=tcp_jsonl_host or "127.0.0.1",
            port=int(tcp_jsonl_port),
        )
        thread = Thread(
            target=tcp_server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
            name="interactive-motion-tcp-jsonl",
        )
        thread.start()
        threads.append(thread)
    if websocket_port is not None:
        thread = Thread(
            target=_run_websocket_server,
            kwargs={
                "queue": queue,
                "default_tcp_by_side": default_tcp_by_side,
                "host": websocket_host or "127.0.0.1",
                "port": int(websocket_port),
                "stop_event": stop_event,
            },
            daemon=True,
            name="interactive-motion-websocket",
        )
        thread.start()
        threads.append(thread)
    return InteractiveTransportHandles(
        threads=tuple(threads),
        stop_event=stop_event,
        tcp_server=tcp_server,
    )


def handle_interactive_message(
    *,
    message: Mapping[str, object],
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
) -> dict[str, object]:
    """Parse and apply one transport message."""

    command = parse_interactive_motion_message(
        message,
        default_tcp_by_side=default_tcp_by_side,
    )
    return _apply_command(command, queue)


def _apply_command(
    command: InteractiveMotionCommand,
    queue: InteractiveMotionQueue,
) -> dict[str, object]:
    if command.kind == "moves":
        queued = queue.submit(command)
        return {
            "event": "accepted",
            "id": queued.command_id,
            "state": queued.state,
            "queue_index": queue.pending_index(queued.command_id),
        }
    if command.kind == "status":
        return queue.status(command.status_id)
    if command.kind == "cancel":
        cancelled = queue.cancel(command.cancel_id or "")
        return {
            "event": "cancel",
            "id": command.cancel_id,
            "accepted": cancelled,
        }
    if command.kind == "cancel_current":
        return {
            "event": "cancel_current",
            "accepted": queue.request_cancel_current(),
        }
    if command.kind == "estop":
        queue.request_estop()
        return {"event": "estop", "accepted": True}
    if command.kind == "quit":
        queue.request_quit()
        return {"event": "quit", "accepted": True}
    if command.kind == "hold":
        queued = queue.submit(command)
        return {"event": "accepted", "id": queued.command_id, "state": queued.state}
    raise ValueError(f"unsupported command kind: {command.kind!r}")


def _stdin_jsonl_reader(
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    stop_event: Event,
    quit_on_eof: bool,
) -> None:
    while not stop_event.is_set():
        line = sys.stdin.readline()
        if line == "":
            if quit_on_eof:
                queue.request_quit()
            return
        response = _handle_json_line(
            line,
            queue=queue,
            default_tcp_by_side=default_tcp_by_side,
        )
        print(json.dumps(response, ensure_ascii=False), flush=True)


def _start_tcp_jsonl_server(
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    host: str,
    port: int,
) -> socketserver.ThreadingTCPServer:
    class _Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            while True:
                line = self.rfile.readline()
                if line == b"":
                    return
                response = _handle_json_line(
                    line.decode("utf-8"),
                    queue=queue,
                    default_tcp_by_side=default_tcp_by_side,
                )
                self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return _Server((host, port), _Handler)


def _handle_json_line(
    line: str,
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
) -> dict[str, object]:
    try:
        message = json.loads(line)
        if not isinstance(message, Mapping):
            raise ValueError("message must be a JSON object")
        return handle_interactive_message(
            message=message,
            queue=queue,
            default_tcp_by_side=default_tcp_by_side,
        )
    except Exception as exc:
        return {"event": "rejected", "error": str(exc)}


def _run_websocket_server(
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    host: str,
    port: int,
    stop_event: Event,
) -> None:
    try:
        import websockets
    except ImportError as exc:
        print(
            "DUAL_ARM_INTERACTIVE_WEBSOCKET_UNAVAILABLE "
            "error=websockets_package_not_installed",
            flush=True,
        )
        raise RuntimeError("websockets package is required for WebSocket transport") from exc

    async def handler(websocket) -> None:
        event_queue: Queue[dict[str, object]] = Queue()

        def listener(event: dict[str, object]) -> None:
            event_queue.put(event)

        queue.add_listener(listener)

        async def sender() -> None:
            while not stop_event.is_set():
                event = await asyncio.to_thread(event_queue.get)
                await websocket.send(json.dumps(event))

        sender_task = asyncio.create_task(sender())
        try:
            async for text in websocket:
                try:
                    message = json.loads(text)
                    if not isinstance(message, Mapping):
                        raise ValueError("message must be a JSON object")
                    response = handle_interactive_message(
                        message=message,
                        queue=queue,
                        default_tcp_by_side=default_tcp_by_side,
                    )
                except Exception as exc:
                    response = {"event": "rejected", "error": str(exc)}
                await websocket.send(json.dumps(response))
        finally:
            sender_task.cancel()
            queue.remove_listener(listener)

    async def main() -> None:
        async with websockets.serve(handler, host, port):
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

    asyncio.run(main())
