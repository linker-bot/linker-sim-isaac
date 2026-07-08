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
        """停止所有后台传输；TCP server 需要额外 shutdown 才能跳出 serve_forever。"""

        self.stop_event.set()
        if self.tcp_server is not None:
            self.tcp_server.shutdown()
            self.tcp_server.server_close()


def start_interactive_transports(
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    default_side: str | None = None,
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
                "default_side": default_side,
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
            default_side=default_side,
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
                "default_side": default_side,
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
    default_side: str | None = None,
) -> dict[str, object]:
    """Parse and apply one transport message."""

    command = parse_interactive_motion_message(
        message,
        default_tcp_by_side=default_tcp_by_side,
        default_side=default_side,
    )
    return _apply_command(command, queue)


def _apply_command(
    command: InteractiveMotionCommand,
    queue: InteractiveMotionQueue,
) -> dict[str, object]:
    """把已解析命令应用到队列，并返回可直接发给客户端的响应。"""

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
    if command.kind == "reset":
        request = queue.request_reset(
            reset_id=command.reset_id,
            mode=command.reset_mode,
            clear_queue=command.reset_clear_queue,
            hold_after_reset=command.reset_hold_after_reset,
        )
        return {"event": "reset", "accepted": True, **request.snapshot()}
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
    default_side: str | None,
    stop_event: Event,
    quit_on_eof: bool,
) -> None:
    """从 stdin 按 JSONL 读取命令；常用于管道或本地调试。"""

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
            default_side=default_side,
        )
        print(json.dumps(response, ensure_ascii=False), flush=True)


def _start_tcp_jsonl_server(
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    default_side: str | None,
    host: str,
    port: int,
) -> socketserver.ThreadingTCPServer:
    """启动一条 TCP JSONL 服务；每行一个 JSON 请求，每行一个 JSON 响应。"""

    class _Handler(socketserver.StreamRequestHandler):
        """TCP 连接处理器；一个连接内可连续发送多行 JSON。"""

        def handle(self) -> None:
            """循环读取客户端 JSONL，并把每条响应写回同一连接。"""

            while True:
                line = self.rfile.readline()
                if line == b"":
                    return
                response = _handle_json_line(
                    line.decode("utf-8"),
                    queue=queue,
                    default_tcp_by_side=default_tcp_by_side,
                    default_side=default_side,
                )
                self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))

    class _Server(socketserver.ThreadingTCPServer):
        """允许端口快速复用、连接线程后台化的 JSONL server。"""

        allow_reuse_address = True
        daemon_threads = True

    return _Server((host, port), _Handler)


def _handle_json_line(
    line: str,
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    default_side: str | None,
) -> dict[str, object]:
    """解析一行 JSONL 并执行；所有异常都会转成 rejected 响应。"""

    try:
        message = json.loads(line)
        if not isinstance(message, Mapping):
            raise ValueError("message must be a JSON object")
        return handle_interactive_message(
            message=message,
            queue=queue,
            default_tcp_by_side=default_tcp_by_side,
            default_side=default_side,
        )
    except Exception as exc:
        return {"event": "rejected", "error": str(exc)}


def _run_websocket_server(
    *,
    queue: InteractiveMotionQueue,
    default_tcp_by_side: Mapping[str, str],
    default_side: str | None,
    host: str,
    port: int,
    stop_event: Event,
) -> None:
    """运行 WebSocket 传输，支持请求响应和队列事件异步推送。"""

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
        """服务单个 WebSocket 客户端，并把队列事件转发给该连接。"""

        event_queue: Queue[dict[str, object]] = Queue()

        def listener(event: dict[str, object]) -> None:
            """把同步队列事件转存到线程安全队列，供 async sender 消费。"""

            event_queue.put(event)

        queue.add_listener(listener)

        async def sender() -> None:
            """后台发送队列事件，避免阻塞主消息接收循环。"""

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
                        default_side=default_side,
                    )
                except Exception as exc:
                    response = {"event": "rejected", "error": str(exc)}
                await websocket.send(json.dumps(response))
        finally:
            sender_task.cancel()
            queue.remove_listener(listener)

    async def main() -> None:
        """创建 WebSocket server，并轮询 stop_event 决定退出。"""

        async with websockets.serve(handler, host, port):
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

    asyncio.run(main())
