"""stdin/TCP JSONL transports and tiled interactive main loop."""

from __future__ import annotations

import json
import queue
import socketserver
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

from linkerbot_sim.app.interactive.tiled.protocol import handle_tiled_interactive_message
from linkerbot_sim.app.interactive.tiled.telemetry_publish import (
    _publish_response_telemetry,
    _publish_state_telemetry,
)
from linkerbot_sim.telemetry.tiled import TiledInteractiveTelemetrySink


@dataclass(frozen=True)
class _InteractiveRequest:
    """transport 线程交给主仿真线程处理的一条 JSONL 请求。"""

    line: str
    source: str
    response_queue: "queue.Queue[dict[str, object]] | None" = None
    echo_response: bool = False


@dataclass(frozen=True)
class _InteractiveControl:
    """transport 生命周期事件，例如 stdin EOF。"""

    kind: str


def start_stdin_jsonl_reader(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    *,
    quit_on_eof: bool,
) -> threading.Thread:
    """启动 stdin JSONL reader 线程。"""

    def _reader() -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                if quit_on_eof:
                    request_queue.put(_InteractiveControl(kind="stdin_eof"))
                return
            request_queue.put(
                _InteractiveRequest(
                    line=line,
                    source="stdin",
                    echo_response=True,
                )
            )

    thread = threading.Thread(
        target=_reader,
        daemon=True,
        name="tiled-interactive-stdin-jsonl",
    )
    thread.start()
    return thread


def run_interactive_loop(
    runtime: object,
    *,
    telemetry: TiledInteractiveTelemetrySink | None,
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    telemetry_rate_hz: float,
) -> None:
    """运行真正的 tiled interactive 主循环。"""

    telemetry_period_s = _telemetry_period_s(telemetry, telemetry_rate_hz)
    idle_period_s = _runtime_idle_period_s(runtime, telemetry_period_s)
    now = time.monotonic()
    next_telemetry_at = now + telemetry_period_s if telemetry_period_s is not None else now
    next_idle_at = now + idle_period_s if idle_period_s is not None else now
    _publish_state_telemetry(telemetry, runtime, event="state")
    while runtime.quit_event is None or not runtime.quit_event.is_set():
        timeout = _interactive_queue_timeout(
            telemetry_period_s=telemetry_period_s,
            next_telemetry_at=next_telemetry_at,
            idle_period_s=idle_period_s,
            next_idle_at=next_idle_at,
        )
        try:
            item = request_queue.get(timeout=timeout)
        except queue.Empty:
            item = None

        now = time.monotonic()
        if item is None and idle_period_s is not None and now >= next_idle_at:
            _runtime_idle_step(runtime)
            next_idle_at = now + idle_period_s

        if isinstance(item, _InteractiveControl):
            if item.kind == "stdin_eof" and runtime.quit_event is not None:
                runtime.quit_event.set()
            continue
        if isinstance(item, _InteractiveRequest):
            response = _handle_json_line(item.line, runtime)
            _publish_response_telemetry(telemetry, runtime, response)
            if item.echo_response:
                print(json.dumps(response, ensure_ascii=False), flush=True)
            if item.response_queue is not None:
                item.response_queue.put(response)

        if telemetry_period_s is None:
            continue
        now = time.monotonic()
        if now >= next_telemetry_at:
            _publish_state_telemetry(telemetry, runtime, event="state")
            next_telemetry_at = now + telemetry_period_s


def start_tcp_jsonl_server(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    *,
    quit_event: threading.Event | None,
    host: str,
    port: int,
) -> socketserver.ThreadingTCPServer:
    """启动 TCP JSONL server；每行请求对应一行响应。"""

    class _Handler(socketserver.StreamRequestHandler):
        """处理单个 TCP JSONL 客户端连接。"""

        def handle(self) -> None:
            """循环读取 JSONL 请求并写回响应。"""

            while quit_event is None or not quit_event.is_set():
                line = self.rfile.readline()
                if line == b"":
                    return
                response_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
                request_queue.put(
                    _InteractiveRequest(
                        line=line.decode("utf-8"),
                        source="tcp_jsonl",
                        response_queue=response_queue,
                    )
                )
                response = response_queue.get()
                self.wfile.write(
                    (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
                )

    class _Server(socketserver.ThreadingTCPServer):
        """允许快速复用端口，并让连接处理线程后台化。"""

        allow_reuse_address = True
        daemon_threads = True

    server = _Server((host, int(port)), _Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        daemon=True,
        name="tiled-interactive-tcp-jsonl",
    )
    thread.start()
    return server


def _handle_json_line(line: str, runtime: object) -> dict[str, object]:
    """解析一行 JSONL 并转成 rejected/response。"""

    try:
        message = json.loads(line)
        if not isinstance(message, Mapping):
            raise ValueError("message must be a JSON object")
        return handle_tiled_interactive_message(message, runtime)
    except Exception as exc:
        return {"event": "rejected", "error": str(exc)}


def _quit_on_stdin_eof(
    *,
    hold: bool,
    tcp_jsonl_port: int | None,
    telemetry: TiledInteractiveTelemetrySink | None,
) -> bool:
    """判断 stdin EOF 是否应该结束进程。"""

    return not bool(hold) and tcp_jsonl_port is None and telemetry is None


def _telemetry_period_s(
    telemetry: TiledInteractiveTelemetrySink | None,
    telemetry_rate_hz: float,
) -> float | None:
    """把 telemetry 频率转换成主循环 timeout 周期。"""

    if telemetry is None:
        return None
    rate = float(telemetry_rate_hz)
    if rate <= 0.0:
        return None
    return 1.0 / rate


def _interactive_queue_timeout(
    *,
    telemetry_period_s: float | None,
    next_telemetry_at: float,
    idle_period_s: float | None = None,
    next_idle_at: float | None = None,
) -> float:
    """返回主循环等待请求队列的 timeout。"""

    deadlines = [time.monotonic() + 0.1]
    if telemetry_period_s is not None:
        deadlines.append(next_telemetry_at)
    if idle_period_s is not None and next_idle_at is not None:
        deadlines.append(next_idle_at)
    return max(0.0, min(deadlines) - time.monotonic())


def _runtime_idle_period_s(
    runtime: object,
    telemetry_period_s: float | None,
) -> float | None:
    """判断主循环是否需要空闲刷新，以及刷新周期。"""

    idle_step = getattr(runtime, "idle_step", None)
    if not callable(idle_step):
        return None
    if not bool(getattr(runtime, "render", False)) and telemetry_period_s is None:
        return None
    configured = getattr(runtime, "idle_period_s", None)
    if configured is not None:
        try:
            value = float(configured)
            if value > 0.0:
                return value
        except (TypeError, ValueError):
            pass
    if telemetry_period_s is not None and telemetry_period_s > 0.0:
        return float(telemetry_period_s)
    return 1.0 / 60.0


def _runtime_idle_step(runtime: object) -> None:
    """执行一次 runtime 空闲刷新；异常交给主循环外层日志捕获。"""

    idle_step = getattr(runtime, "idle_step", None)
    if callable(idle_step):
        idle_step()
