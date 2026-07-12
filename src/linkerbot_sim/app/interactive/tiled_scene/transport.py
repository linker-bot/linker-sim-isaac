"""TiledSceneRuntime 的 stdin/TCP JSONL transport 与主线程事件循环。

transport 线程只生产 ``_InteractiveRequest``；主循环串行调用 runtime、发布 telemetry
并回填 response queue。这样所有 Isaac/PhysX 操作都留在创建 runtime 的主线程。

数据请求的 admission 从入队持续到主线程完成响应，而不是在 dequeue 时释放；慢 runtime
因而会对所有入口形成明确背压。stdin EOF 等生命周期 control 可绕过数据容量，保证队列
满载时仍能驱动关闭。TCP 使用连接线程，WebSocket 在专用线程中运行 asyncio loop，两者
通过单元素 response queue 与同步主循环交接，不能直接调用 runtime。
"""

from __future__ import annotations

import asyncio
import math
import queue
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from linkerbot_sim.app.interactive.policies import (
    IdlePhysicsPolicy,
    StdinEofPolicy,
)
from linkerbot_sim.utils.json import (
    strict_json_dumps,
    strict_json_loads,
)
from linkerbot_sim.app.interactive.stdin_reader import (
    InterruptibleStdinJsonlReader,
    StdinJsonlFrame,
    start_interruptible_stdin_jsonl_reader,
)
from linkerbot_sim.app.interactive.tiled_scene.protocol import (
    handle_tiled_interactive_message,
)
from linkerbot_sim.app.interactive.tiled_scene.telemetry_publish import (
    _publish_response_telemetry,
    _publish_state_telemetry,
)
from linkerbot_sim.telemetry.tiled.sink import TiledInteractiveTelemetrySink
from linkerbot_sim.utils.config import require_loopback_host


_CONTROL_QUEUE_RETRY_TIMEOUT_S = 0.1
_MAX_IDLE_STEPS_PER_CHUNK = 10_000


@dataclass(frozen=True)
class _InteractiveRequest:
    """transport 线程交给主仿真线程处理的一条完整 JSONL 请求。

    ``response_queue`` 的唯一生产者是主线程，对应连接线程/协程是消费者；stdin 没有响应
    队列，而是通过 ``echo_response`` 要求主线程在执行后写 stdout。
    """

    line: str
    source: str
    response_queue: "queue.Queue[dict[str, object]] | None" = None
    echo_response: bool = False


@dataclass(frozen=True)
class _InteractiveControl:
    """不占用数据 admission 的 transport 生命周期事件，例如 stdin EOF。"""

    kind: str


class SharedTransportAdmission:
    """TCP/WebSocket 共用的进程级连接 admission 与分 transport 消息计数。

    一个锁同时保护总连接上限和各入口计数，防止 TCP 与 WebSocket 并发接入时分别通过
    上限检查。消息计数是诊断数据；请求容量由 ``BoundedInteractiveRequestQueue`` 独立
    管理，连接已接纳不代表其下一条消息一定能进入主线程队列。
    """

    def __init__(self, *, max_connections: int) -> None:
        """创建共享 admission；``max_connections`` 是全部远端 transport 的总额。"""

        self.max_connections = _positive_int(max_connections, label="max_connections")
        self._lock = threading.Lock()
        self._active_by_transport: dict[str, int] = {}
        self._rejected_connections_by_transport: dict[str, int] = {}
        self._messages_received_by_transport: dict[str, int] = {}
        self._messages_rejected_by_transport: dict[str, int] = {}
        self._oversized_messages_by_transport: dict[str, int] = {}

    def try_acquire(self, transport: str) -> bool:
        """原子预留一个连接槽位，达到总上限时按 transport 记录拒绝。"""

        with self._lock:
            active = sum(self._active_by_transport.values())
            if active >= self.max_connections:
                self._increment(self._rejected_connections_by_transport, transport)
                return False
            self._increment(self._active_by_transport, transport)
            return True

    def release(self, transport: str) -> None:
        """释放指定 transport 先前取得的槽位；不平衡释放视为生命周期错误。"""

        with self._lock:
            active = self._active_by_transport.get(transport, 0)
            if active < 1:
                raise RuntimeError(f"unbalanced transport release: {transport}")
            self._active_by_transport[transport] = active - 1

    def record_message_received(self, transport: str) -> None:
        """记录一条已完成 framing 的入站消息。"""

        with self._lock:
            self._increment(self._messages_received_by_transport, transport)

    def record_message_rejected(
        self, transport: str, *, oversized: bool = False
    ) -> None:
        """记录协议/admission 拒绝，并可同时标记消息超过字节上限。"""

        with self._lock:
            self._increment(self._messages_rejected_by_transport, transport)
            if oversized:
                self._increment(self._oversized_messages_by_transport, transport)

    def status(self) -> dict[str, object]:
        """返回总量与分 transport 的一致、JSON-compatible 指标快照。"""

        with self._lock:
            active_by_transport = dict(self._active_by_transport)
            rejected_connections = dict(self._rejected_connections_by_transport)
            received = dict(self._messages_received_by_transport)
            rejected = dict(self._messages_rejected_by_transport)
            oversized = dict(self._oversized_messages_by_transport)
        transports = sorted(
            set(active_by_transport)
            | set(rejected_connections)
            | set(received)
            | set(rejected)
            | set(oversized)
        )
        return {
            "active_connections": sum(active_by_transport.values()),
            "max_connections": self.max_connections,
            "rejected_connections": sum(rejected_connections.values()),
            "messages_received": sum(received.values()),
            "messages_rejected": sum(rejected.values()),
            "oversized_messages": sum(oversized.values()),
            "by_transport": {
                name: {
                    "active_connections": active_by_transport.get(name, 0),
                    "rejected_connections": rejected_connections.get(name, 0),
                    "messages_received": received.get(name, 0),
                    "messages_rejected": rejected.get(name, 0),
                    "oversized_messages": oversized.get(name, 0),
                }
                for name in transports
            },
        }

    def transport_status(self, transport: str) -> dict[str, int]:
        """返回单个 transport 的指标视图，同时保留进程级连接容量。"""

        status = self.status()
        by_transport = cast(Mapping[str, Mapping[str, int]], status["by_transport"])
        values = by_transport.get(transport, {})
        return {
            "active_connections": int(values.get("active_connections", 0)),
            "max_connections": self.max_connections,
            "rejected_connections": int(values.get("rejected_connections", 0)),
            "messages_received": int(values.get("messages_received", 0)),
            "messages_rejected": int(values.get("messages_rejected", 0)),
            "oversized_messages": int(values.get("oversized_messages", 0)),
        }

    @staticmethod
    def _increment(values: dict[str, int], key: str) -> None:
        values[key] = values.get(key, 0) + 1


class BoundedInteractiveRequestQueue(
    queue.Queue[_InteractiveRequest | _InteractiveControl]
):
    """按 outstanding 请求而非容器长度限流的主线程请求队列。

    ``_InteractiveRequest`` 从 ``put`` 到匹配的 ``task_done`` 始终占用容量；被主线程
    ``get`` 后只是从 pending 转为 running。``_InteractiveControl`` 不占数据容量，因此
    可在满载时入队驱动 EOF/关闭，但仍计入标准 Queue 的 unfinished task 配对。

    ``get`` 与 ``task_done`` 的请求种类通过 thread-local FIFO 配对，所以消费线程必须在
    同一线程按取得顺序调用 ``task_done``；当前设计中唯一消费者就是 tiled 主循环。
    """

    def __init__(self, *, capacity: int) -> None:
        """初始化请求容量、状态计数和每个消费线程的 task 配对上下文。"""

        self.capacity = _positive_int(capacity, label="capacity")
        super().__init__(maxsize=self.capacity)
        self._rejected_requests = 0
        self._pending_requests = 0
        self._running_requests = 0
        self._outstanding_requests = 0
        self._task_context = threading.local()
        self._records_rejections_internally = True

    def put(
        self,
        item: _InteractiveRequest | _InteractiveControl,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """加入请求或 control；仅数据请求申请 outstanding admission。

        容量等待、计数增加和实际入队在 Queue 自带的 ``not_full`` 条件锁内完成，多个
        transport 生产者不会同时越过容量。非阻塞拒绝在同一临界区累计。
        """

        claimed = isinstance(item, _InteractiveRequest)
        with self.not_full:
            if claimed:
                try:
                    self._wait_for_capacity(block=block, timeout=timeout)
                except queue.Full:
                    self._rejected_requests += 1
                    raise
                self._outstanding_requests += 1
                self._pending_requests += 1
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> _InteractiveRequest | _InteractiveControl:
        """取得下一项，并把数据请求从 pending 原子切换为 running。

        dequeue 不释放 admission；调用方处理完 runtime、telemetry 与 response 交付后才
        能调用 ``task_done``。control 会记录为非 claimed，以保持 unfinished task 平衡。
        """

        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise queue.Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("timeout must be a non-negative number")
            else:
                end_time = time.monotonic() + timeout
                while not self._qsize():
                    remaining = end_time - time.monotonic()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            claimed = isinstance(item, _InteractiveRequest)
            if claimed:
                self._pending_requests -= 1
                self._running_requests += 1
            self.not_full.notify()
        task_kinds = getattr(self._task_context, "claimed", None)
        if task_kinds is None:
            task_kinds = []
            self._task_context.claimed = task_kinds
        task_kinds.append(claimed)
        return item

    def task_done(self) -> None:
        """完成与本消费线程最早一次 ``get`` 的配对，并释放请求 admission。

        对数据请求而言，此调用必须晚于 response 写入；否则新请求可能被接纳，而旧请求
        的不可逆 runtime 工作或响应交付仍未完成。
        """

        task_kinds = getattr(self._task_context, "claimed", None)
        if not task_kinds:
            raise ValueError("task_done() called without a matching get()")
        claimed = bool(task_kinds.pop(0))
        with self.all_tasks_done:
            unfinished = self.unfinished_tasks - 1
            if unfinished < 0:
                raise ValueError("task_done() called too many times")
            self.unfinished_tasks = unfinished
            if claimed:
                self._running_requests -= 1
                self._outstanding_requests -= 1
                self.not_full.notify_all()
            if unfinished == 0:
                self.all_tasks_done.notify_all()

    def full(self) -> bool:
        """返回 outstanding 数据请求是否已占满 admission，而非底层 deque 是否满。"""

        with self.mutex:
            return self._outstanding_requests >= self.capacity

    def record_rejection(self) -> None:
        """为不经本类 ``put`` 计数路径的外部拒绝累计一次诊断计数。"""

        with self.mutex:
            self._rejected_requests += 1

    def status(self) -> dict[str, int]:
        """返回 pending/running/outstanding 深度和累计容量拒绝。"""

        with self.mutex:
            return {
                "request_queue_depth": self._pending_requests,
                "request_queue_capacity": self.capacity,
                "pending_requests": self._pending_requests,
                "running_requests": self._running_requests,
                "active_requests": self._outstanding_requests,
                "outstanding_requests": self._outstanding_requests,
                "rejected_requests": self._rejected_requests,
            }

    def _wait_for_capacity(self, *, block: bool, timeout: float | None) -> None:
        if not block:
            if self._outstanding_requests >= self.capacity:
                raise queue.Full
            return
        if timeout is None:
            while self._outstanding_requests >= self.capacity:
                self.not_full.wait()
            return
        if timeout < 0:
            raise ValueError("timeout must be a non-negative number")
        end_time = time.monotonic() + timeout
        while self._outstanding_requests >= self.capacity:
            remaining = end_time - time.monotonic()
            if remaining <= 0.0:
                raise queue.Full
            self.not_full.wait(remaining)


def interactive_request_queue_status(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
) -> dict[str, int]:
    """返回标准 Queue 或 admission-aware Queue 的统一容量诊断。

    标准 Queue 无法区分已经 dequeue 的运行中请求，因此 fallback 只报告容器深度；生产
    路径使用 ``BoundedInteractiveRequestQueue`` 时则能准确报告完整生命周期。
    """

    status = getattr(request_queue, "status", None)
    if callable(status):
        provider = cast(Callable[[], Mapping[str, int]], status)
        return dict(provider())
    return {
        "request_queue_depth": request_queue.qsize(),
        "request_queue_capacity": request_queue.maxsize,
        "pending_requests": request_queue.qsize(),
        "running_requests": 0,
        "active_requests": request_queue.qsize(),
        "outstanding_requests": request_queue.qsize(),
        "rejected_requests": 0,
    }


def start_stdin_jsonl_reader(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    *,
    quit_on_eof: bool,
    max_message_bytes: int = 1_048_576,
    admission: SharedTransportAdmission | None = None,
) -> InterruptibleStdinJsonlReader:
    """启动并返回拥有自身 fd 副本的可中断 stdin JSONL reader。

    frame 回调只校验边界并尝试入队，不调用 runtime。自然 EOF 根据策略投递可靠 control；
    调用方必须保留返回句柄，以便关闭时唤醒并 join reader 线程。
    """

    max_message_bytes = _positive_int(max_message_bytes, label="max_message_bytes")
    shutdown_event = threading.Event()

    def handle_eof() -> None:
        """按 EOF 策略可靠投递控制消息，不直接触碰 runtime。"""

        if quit_on_eof:
            _put_control_reliably(
                request_queue,
                _InteractiveControl(kind="stdin_eof"),
                stop_event=shutdown_event,
            )

    return start_interruptible_stdin_jsonl_reader(
        stream=sys.stdin,
        max_message_bytes=max_message_bytes,
        on_frame=lambda frame: _handle_tiled_stdin_frame(
            frame,
            request_queue=request_queue,
            max_message_bytes=max_message_bytes,
            admission=admission,
            shutdown_event=shutdown_event,
        ),
        on_eof=handle_eof,
        thread_name="tiled-interactive-stdin-jsonl",
        shutdown_event=shutdown_event,
    )


def _handle_tiled_stdin_frame(
    frame: StdinJsonlFrame,
    *,
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    max_message_bytes: int,
    admission: SharedTransportAdmission | None,
    shutdown_event: threading.Event,
) -> None:
    """校验一条已 framing 的 stdin 消息，只把完整 UTF-8 文本交给主线程队列。"""

    if shutdown_event.is_set():
        return
    if admission is not None:
        admission.record_message_received("stdin")
    if frame.oversized or frame.decode_error is not None:
        if admission is not None:
            admission.record_message_rejected("stdin", oversized=frame.oversized)
        rejection = _rejected_response(
            (
                "message exceeds max_message_bytes"
                if frame.oversized
                else f"message must be UTF-8: {frame.decode_error}"
            ),
            code=("message_too_large" if frame.oversized else "invalid_encoding"),
            limit=max_message_bytes,
        )
        print(strict_json_dumps(rejection, ensure_ascii=False), flush=True)
        return
    assert frame.text is not None
    request = _InteractiveRequest(
        line=frame.text,
        source="stdin",
        echo_response=True,
    )
    if shutdown_event.is_set():
        return
    rejection = _enqueue_interactive_request(request_queue, request)
    if rejection is not None:
        if admission is not None:
            admission.record_message_rejected("stdin")
        print(strict_json_dumps(rejection, ensure_ascii=False), flush=True)


def run_interactive_loop(
    runtime: object,
    *,
    telemetry: TiledInteractiveTelemetrySink | None,
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    telemetry_rate_hz: float,
    idle_physics_policy: IdlePhysicsPolicy = "pause",
    idle_step_duration_s: float | None = None,
    queue_poll_timeout_s: float = 0.1,
    event_publisher: Callable[[Mapping[str, object]], bool] | None = None,
    transport_status_provider: Callable[[], Mapping[str, object]] | None = None,
) -> None:
    """按显式空闲策略运行 tiled 主线程事件循环。

    本函数必须在创建 runtime 的线程调用，并是 ``request_queue`` 的唯一消费者。每次只
    串行执行一条 JSON 请求；response telemetry、WebSocket 事件和连接专属 response 都
    在 ``task_done`` 释放 admission 前完成。队列空闲时，timeout 同时受 telemetry deadline、
    idle physics deadline 与退出轮询上限约束，任何周期任务都不会永久阻塞关闭。
    """
    queue_poll_timeout_s = _positive_float(
        queue_poll_timeout_s, label="queue_poll_timeout_s"
    )
    if idle_step_duration_s is not None:
        idle_step_duration_s = _positive_float(
            idle_step_duration_s, label="idle_step_duration_s"
        )

    telemetry_period_s = _telemetry_period_s(telemetry, telemetry_rate_hz)
    idle_period_s = _runtime_idle_period_s(
        runtime,
        idle_physics_policy=idle_physics_policy,
        idle_step_duration_s=idle_step_duration_s,
    )
    idle_step_count = 0
    if idle_period_s is not None:
        assert idle_step_duration_s is not None
        idle_step_count = _runtime_idle_step_count(runtime, idle_step_duration_s)
    now = time.monotonic()
    next_telemetry_at = (
        now + telemetry_period_s if telemetry_period_s is not None else now
    )
    next_idle_at = now + idle_period_s if idle_period_s is not None else now
    active_telemetry = telemetry if telemetry_period_s is not None else None
    _publish_state_telemetry(active_telemetry, runtime, event="state")
    quit_event = getattr(runtime, "quit_event", None)
    while quit_event is None or not quit_event.is_set():
        timeout = _interactive_queue_timeout(
            telemetry_period_s=telemetry_period_s,
            next_telemetry_at=next_telemetry_at,
            idle_period_s=idle_period_s,
            next_idle_at=next_idle_at,
            queue_poll_timeout_s=queue_poll_timeout_s,
        )
        try:
            item = request_queue.get(timeout=timeout)
        except queue.Empty:
            item = None

        now = time.monotonic()
        if item is None and idle_period_s is not None and now >= next_idle_at:
            _runtime_idle_step(runtime, count=idle_step_count)
            next_idle_at = time.monotonic() + idle_period_s

        if isinstance(item, _InteractiveControl):
            try:
                if item.kind == "stdin_eof" and quit_event is not None:
                    quit_event.set()
            finally:
                request_queue.task_done()
            continue
        if isinstance(item, _InteractiveRequest):
            try:
                response = _handle_json_line(item.line, runtime)
                if (
                    response.get("event") == "status"
                    and transport_status_provider is not None
                ):
                    response = {
                        **response,
                        "transport": dict(transport_status_provider()),
                    }
                _publish_response_telemetry(active_telemetry, runtime, response)
                if event_publisher is not None:
                    event_publisher(response)
                if item.echo_response:
                    print(strict_json_dumps(response, ensure_ascii=False), flush=True)
                if item.response_queue is not None:
                    item.response_queue.put(response)
            finally:
                request_queue.task_done()

        if telemetry_period_s is None:
            continue
        now = time.monotonic()
        if now >= next_telemetry_at:
            _publish_state_telemetry(active_telemetry, runtime, event="state")
            next_telemetry_at = now + telemetry_period_s


def start_tcp_jsonl_server(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    *,
    quit_event: threading.Event | None,
    host: str,
    port: int,
    max_message_bytes: int = 1_048_576,
    max_connections: int = 16,
    server_poll_interval_s: float = 0.1,
    response_poll_interval_s: float = 0.5,
    admission: SharedTransportAdmission | None = None,
) -> socketserver.ThreadingTCPServer:
    """启动 TCP JSONL server；同一连接中每行请求严格对应一行响应。

    handler 线程只负责有界读取、入队和等待单元素 response queue。连接 admission 在创建
    handler 前取得，在 handler 的 ``finally`` 中释放；返回 server 后由调用方负责调用
    ``stop_tcp_jsonl_server``，以关闭监听 socket、活跃连接及后台线程。
    """

    host = require_loopback_host(host, label="host")
    max_message_bytes = _positive_int(max_message_bytes, label="max_message_bytes")
    max_connections = _positive_int(max_connections, label="max_connections")
    server_poll_interval_s = _positive_float(
        server_poll_interval_s, label="server_poll_interval_s"
    )
    response_poll_interval_s = _positive_float(
        response_poll_interval_s, label="response_poll_interval_s"
    )
    shared_admission = admission or SharedTransportAdmission(
        max_connections=max_connections
    )

    class _Handler(socketserver.StreamRequestHandler):
        """处理单个 TCP JSONL 客户端连接。"""

        def handle(self) -> None:
            """循环读取 JSONL 请求并写回响应。"""

            server = cast(Any, self.server)
            while not server._stopping_event.is_set() and (
                quit_event is None or not quit_event.is_set()
            ):
                line, oversized = _read_bounded_line(
                    self.rfile, max_message_bytes=max_message_bytes
                )
                if line == b"":
                    return
                shared_admission.record_message_received("tcp_jsonl")
                if oversized:
                    shared_admission.record_message_rejected(
                        "tcp_jsonl", oversized=True
                    )
                    self._write_response(
                        _rejected_response(
                            "message exceeds max_message_bytes",
                            code="message_too_large",
                            limit=max_message_bytes,
                        )
                    )
                    continue
                try:
                    decoded = line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    shared_admission.record_message_rejected("tcp_jsonl")
                    self._write_response(
                        _rejected_response(
                            f"message must be UTF-8: {exc}",
                            code="invalid_encoding",
                        )
                    )
                    continue
                response_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
                rejection = _enqueue_interactive_request(
                    request_queue,
                    _InteractiveRequest(
                        line=decoded,
                        source="tcp_jsonl",
                        response_queue=response_queue,
                    ),
                )
                if rejection is not None:
                    shared_admission.record_message_rejected("tcp_jsonl")
                    self._write_response(rejection)
                    continue
                # 主循环是该 response 的唯一生产者。用有超时的 get 轮询，让主循环退出或
                # quit_event 置位时，连接线程也能及时收敛，而不是永久阻塞在 get() 上。
                response: dict[str, object] | None = None
                while not server._stopping_event.is_set() and (
                    quit_event is None or not quit_event.is_set()
                ):
                    try:
                        response = response_queue.get(timeout=response_poll_interval_s)
                        break
                    except queue.Empty:
                        continue
                if response is None:
                    return
                if response.get("event") == "rejected":
                    shared_admission.record_message_rejected("tcp_jsonl")
                self._write_response(response)

        def _write_response(self, response: Mapping[str, object]) -> None:
            """写一条 JSONL 响应。"""

            self.wfile.write(
                (strict_json_dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
            )
            self.wfile.flush()

    class _Server(socketserver.ThreadingTCPServer):
        """允许快速复用端口，并让连接处理线程后台化。"""

        allow_reuse_address = True
        daemon_threads = True

        def __init__(self, server_address, handler_class) -> None:
            self._admission = shared_admission
            self._resource_lock = threading.Lock()
            self._active_sockets: set[socket.socket] = set()
            self._handler_threads: set[threading.Thread] = set()
            self._stopping_event = threading.Event()
            super().__init__(server_address, handler_class)

        def process_request(self, request, client_address) -> None:
            """在派生 handler 线程前取得共享连接 admission 并登记活跃 socket。"""

            if not self._admission.try_acquire("tcp_jsonl"):
                _reject_connection(
                    request,
                    _rejected_response(
                        "maximum concurrent connections reached",
                        code="connection_limit",
                        limit=self._admission.max_connections,
                    ),
                )
                self.shutdown_request(request)
                return
            if isinstance(request, socket.socket):
                with self._resource_lock:
                    self._active_sockets.add(request)
            try:
                super().process_request(request, client_address)
            except BaseException:
                self._release_connection(request)
                raise

        def process_request_thread(self, request, client_address) -> None:
            """登记当前 handler，并在所有退出路径恰好一次释放连接资源。"""

            thread = threading.current_thread()
            with self._resource_lock:
                self._handler_threads.add(thread)
            try:
                super().process_request_thread(request, client_address)
            finally:
                with self._resource_lock:
                    self._handler_threads.discard(thread)
                self._release_connection(request)

        def _release_connection(self, request: object) -> None:
            released = False
            if isinstance(request, socket.socket):
                with self._resource_lock:
                    if request in self._active_sockets:
                        self._active_sockets.discard(request)
                        released = True
            if released:
                self._admission.release("tcp_jsonl")

        def close_active_connections(self) -> None:
            """标记 server 停止并关闭活跃 socket，以解除 handler 的读/响应等待。"""

            self._stopping_event.set()
            with self._resource_lock:
                sockets = tuple(self._active_sockets)
            for active_socket in sockets:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    active_socket.close()
                except OSError:
                    pass

        def handler_threads(self) -> tuple[threading.Thread, ...]:
            """返回活跃 handler 线程快照，供有界关闭在锁外逐个 join。"""

            with self._resource_lock:
                return tuple(self._handler_threads)

        def status(self) -> dict[str, object]:
            """返回连接、handler、请求队列和停止阶段的诊断快照。"""

            with self._resource_lock:
                sockets = tuple(self._active_sockets)
                handler_threads = tuple(self._handler_threads)
            return {
                **self._admission.transport_status("tcp_jsonl"),
                "max_message_bytes": max_message_bytes,
                "live_socket_count": len(sockets),
                "live_handler_thread_count": sum(
                    1 for thread in handler_threads if thread.is_alive()
                ),
                "live_handler_threads": sorted(
                    thread.name for thread in handler_threads if thread.is_alive()
                ),
                "stopping": self._stopping_event.is_set(),
                **interactive_request_queue_status(request_queue),
            }

    if ":" in host:
        _Server.address_family = socket.AF_INET6
    server = _Server((host, int(port)), _Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": server_poll_interval_s},
        daemon=True,
        name="tiled-interactive-tcp-jsonl",
    )
    thread.start()
    setattr(server, "_serve_thread", thread)
    setattr(server, "_shutdown_thread", None)
    setattr(server, "_server_closed", False)
    setattr(server, "_admission", shared_admission)
    return server


def stop_tcp_jsonl_server(
    server: socketserver.ThreadingTCPServer,
    *,
    timeout_s: float = 2.0,
) -> dict[str, object]:
    """在统一 deadline 内停止 TCP serve loop、连接和 handler 线程。

    先关闭活跃连接以解除读阻塞，再用独立 shutdown 线程唤醒 ``serve_forever``；仅当
    shutdown 确认完成后才关闭 server socket。超时时不会丢弃任何线程句柄，返回的 status
    会明确报告仍存活的 serve/shutdown/handler 资源，后续可以再次调用本函数收敛。
    """

    timeout = _non_negative_float(timeout_s, label="timeout_s")
    deadline = time.monotonic() + timeout
    close_connections = getattr(server, "close_active_connections", None)
    if callable(close_connections):
        close_connections()
    shutdown_thread = getattr(server, "_shutdown_thread", None)
    if shutdown_thread is None:
        shutdown_thread = threading.Thread(
            target=server.shutdown,
            daemon=True,
            name="tiled-interactive-tcp-jsonl-shutdown",
        )
        setattr(server, "_shutdown_thread", shutdown_thread)
        shutdown_thread.start()
    shutdown_thread.join(timeout=max(0.0, deadline - time.monotonic()))
    if not shutdown_thread.is_alive() and not getattr(server, "_server_closed", False):
        server.server_close()
        setattr(server, "_server_closed", True)
    serve_thread = getattr(server, "_serve_thread", None)
    if serve_thread is not None:
        serve_thread.join(timeout=max(0.0, deadline - time.monotonic()))
    handler_threads_provider = getattr(server, "handler_threads", None)
    handler_threads = (
        cast(Callable[[], tuple[threading.Thread, ...]], handler_threads_provider)()
        if callable(handler_threads_provider)
        else ()
    )
    for handler_thread in handler_threads:
        handler_thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return tcp_jsonl_server_status(server)


def tcp_jsonl_server_status(
    server: socketserver.ThreadingTCPServer,
) -> dict[str, object]:
    """返回 TCP server、连接与关闭线程的生命周期诊断，不改变其状态。"""

    serve_thread = getattr(server, "_serve_thread", None)
    shutdown_thread = getattr(server, "_shutdown_thread", None)
    connection_status = getattr(server, "status", None)
    details: Mapping[str, object] = (
        cast(Callable[[], Mapping[str, object]], connection_status)()
        if callable(connection_status)
        else {}
    )
    return {
        **details,
        "serve_thread_alive": (serve_thread is not None and serve_thread.is_alive()),
        "shutdown_thread_alive": (
            shutdown_thread is not None and shutdown_thread.is_alive()
        ),
        "shutdown_timed_out": (
            shutdown_thread is not None and shutdown_thread.is_alive()
        ),
        "server_closed": bool(getattr(server, "_server_closed", False)),
    }


def combined_transport_status(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    *,
    tcp_server: socketserver.ThreadingTCPServer | None = None,
    websocket_server: "WebSocketServerHandle | None" = None,
    admission: SharedTransportAdmission | None = None,
) -> dict[str, object]:
    """组合 request queue、TCP、WebSocket 与共享 admission 的 status payload。

    该函数只读取快照，不拥有或关闭传入资源；未显式提供 admission 时，会从现有 server
    句柄中发现同一个共享实例，避免把分 transport 容量误报为独立总额。
    """

    status: dict[str, object] = {
        **interactive_request_queue_status(request_queue),
    }
    if tcp_server is not None:
        status["tcp_jsonl"] = tcp_jsonl_server_status(tcp_server)
    if websocket_server is not None:
        status["websocket"] = websocket_server.status()
    if admission is None and tcp_server is not None:
        admission = getattr(tcp_server, "_admission", None)
    if admission is None and websocket_server is not None:
        admission = websocket_server.admission
    if admission is not None:
        status["admission"] = admission.status()
    return status


class WebSocketServerHandle:
    """拥有 WebSocket asyncio loop 与线程的生命周期句柄。

    该 server 只处理 I/O：入站消息进入同步 request queue，主线程通过单元素 response
    queue 回应。主动事件先进入一个进程内有界 outbound queue，再由 loop 广播给当时所有
    客户端；队列满时采用 reject-new 并计数，不阻塞仿真主线程。
    """

    def __init__(
        self,
        request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
        *,
        quit_event: threading.Event | None,
        host: str,
        port: int,
        max_message_bytes: int,
        max_connections: int,
        event_queue_capacity: int,
        server_poll_interval_s: float,
        response_poll_interval_s: float,
        admission: SharedTransportAdmission,
    ) -> None:
        """保存已校验设置并创建跨线程就绪、停止和 outbound 队列原语。"""

        self.request_queue = request_queue
        self.quit_event = quit_event
        self.host = require_loopback_host(host, label="host")
        self.port = int(port)
        self.max_message_bytes = max_message_bytes
        self.admission = admission
        self.max_connections = admission.max_connections
        self.server_poll_interval_s = server_poll_interval_s
        self.response_poll_interval_s = response_poll_interval_s
        self.event_queue: queue.Queue[Mapping[str, object]] = queue.Queue(
            maxsize=event_queue_capacity
        )
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._clients: set[Any] = set()
        self.thread: threading.Thread | None = None
        self.last_error: BaseException | None = None
        self.rejected_events = 0
        self.bound_port: int | None = None
        self.shutdown_timed_out = False

    def start(self, *, startup_timeout_s: float = 5.0) -> None:
        """启动专用 asyncio 线程，并等待 server bind 成功或确定失败。

        方法对已启动句柄幂等；超时表示就绪状态未知，调用方仍应通过 ``stop`` 回收已创建
        的线程，而不能假定后台启动已自动取消。
        """

        if self.thread is not None:
            return
        timeout = _positive_float(startup_timeout_s, label="startup_timeout_s")
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="tiled-interactive-websocket",
        )
        self.thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("WebSocket server did not start before timeout")
        if self.last_error is not None:
            raise RuntimeError(f"WebSocket server failed: {self.last_error}")

    def publish_event(self, event: Mapping[str, object]) -> bool:
        """复制并非阻塞加入 outbound event queue；队列满时明确返回 ``False``。

        这是仿真主线程到 asyncio 线程的单向边界。返回成功仅表示事件已排队，不保证每个
        客户端最终收到；发送期间断开的客户端会从活动集合移除。
        """

        try:
            self.event_queue.put_nowait(dict(event))
        except queue.Full:
            with self._lock:
                self.rejected_events += 1
            return False
        return True

    def stop(self, *, timeout_s: float = 2.0) -> dict[str, object]:
        """线程安全地通知 asyncio server 停止，并有界 join 其线程。

        ``call_soon_threadsafe`` 是同步关闭方进入 loop 的唯一入口。loop 可能恰好被 quit
        watcher 关闭，此竞态表示目标已经达成；join 超时则保留 live thread 并写入诊断。
        """

        timeout = _non_negative_float(timeout_s, label="timeout_s")
        self._stop_requested.set()
        with self._lock:
            loop = self._loop
            stop_event = self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(stop_event.set)
            except RuntimeError:
                # quit watcher 可能在 is_closed() 与 call_soon_threadsafe() 之间关闭 loop；
                # 该竞态下 server 已经停止，无需再次投递。
                pass
        thread = self.thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                self.shutdown_timed_out = True
        return self.status()

    def status(self) -> dict[str, object]:
        """返回 bind、连接、队列、线程及最近错误的 JSON-compatible 诊断快照。"""

        thread = self.thread
        with self._lock:
            return {
                "host": self.host,
                "port": self.bound_port,
                **self.admission.transport_status("websocket"),
                "max_message_bytes": self.max_message_bytes,
                "event_queue_depth": self.event_queue.qsize(),
                "event_queue_capacity": self.event_queue.maxsize,
                "rejected_events": self.rejected_events,
                **interactive_request_queue_status(self.request_queue),
                "thread_alive": thread is not None and thread.is_alive(),
                "shutdown_timed_out": self.shutdown_timed_out,
                "last_error": (
                    None
                    if self.last_error is None
                    else f"{type(self.last_error).__name__}: {self.last_error}"
                ),
            }

    def _run(self) -> None:
        """在线程入口拥有 asyncio loop，并把启动/运行异常暴露给同步调用方。"""

        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self.last_error = exc
            self._ready.set()
        finally:
            with self._lock:
                self._loop = None
                self._stop_event = None

    async def _serve(self) -> None:
        """绑定 server，启动事件广播与 quit watcher，并按 asyncio stop event 收敛。"""

        import websockets

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            # ``stop`` 可能在本线程取得 asyncio loop 前发生；持久 stop request 防止
            # 已超时、无人持有的 helper 随后继续 bind 并成为孤立 server。
            if self._stop_requested.is_set():
                return
            async with websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                max_size=self.max_message_bytes + 1,
            ) as server:
                sockets = tuple(server.sockets or ())
                if sockets:
                    self.bound_port = int(sockets[0].getsockname()[1])
                self._ready.set()
                event_task = asyncio.create_task(self._publish_events())
                quit_task = asyncio.create_task(self._watch_quit_event())
                try:
                    await self._stop_event.wait()
                finally:
                    event_task.cancel()
                    quit_task.cancel()
                    await asyncio.gather(event_task, quit_task, return_exceptions=True)
        finally:
            self._ready.set()

    async def _handle_client(self, websocket: Any) -> None:
        """处理单个客户端的 admission、顺序请求/响应和连接释放。

        一个连接一次只等待一个主线程响应，因此客户端请求顺序与响应顺序一致。连接关闭
        只停止等待方，不撤销已被主线程接纳的 runtime 请求；该请求仍会执行并释放队列
        admission，只是无人再消费其单元素 response queue。
        """

        admitted = self.admission.try_acquire("websocket")
        if not admitted:
            await websocket.send(
                strict_json_dumps(
                    _rejected_response(
                        "maximum concurrent connections reached",
                        code="connection_limit",
                        limit=self.admission.max_connections,
                    ),
                    ensure_ascii=False,
                )
            )
            await websocket.close(code=1013, reason="connection limit")
            return
        self._clients.add(websocket)
        try:
            try:
                async for message in websocket:
                    self.admission.record_message_received("websocket")
                    if not isinstance(message, str):
                        self.admission.record_message_rejected("websocket")
                        await self._send_rejection(
                            websocket,
                            "binary WebSocket messages are unsupported",
                            code="invalid_message_type",
                        )
                        continue
                    if len(message.encode("utf-8")) > self.max_message_bytes:
                        self.admission.record_message_rejected(
                            "websocket", oversized=True
                        )
                        await self._send_rejection(
                            websocket,
                            "message exceeds max_message_bytes",
                            code="message_too_large",
                            limit=self.max_message_bytes,
                        )
                        continue
                    response_queue: queue.Queue[dict[str, object]] = queue.Queue(
                        maxsize=1
                    )
                    rejection = _enqueue_interactive_request(
                        self.request_queue,
                        _InteractiveRequest(
                            line=message,
                            source="websocket",
                            response_queue=response_queue,
                        ),
                    )
                    if rejection is not None:
                        self.admission.record_message_rejected("websocket")
                        await websocket.send(
                            strict_json_dumps(rejection, ensure_ascii=False)
                        )
                        continue
                    response = await self._wait_for_response(response_queue, websocket)
                    if response is None:
                        return
                    if response.get("event") == "rejected":
                        self.admission.record_message_rejected("websocket")
                    await websocket.send(
                        strict_json_dumps(response, ensure_ascii=False)
                    )
            except Exception as exc:
                if _websocket_close_code(exc) == 1009:
                    self.admission.record_message_received("websocket")
                    self.admission.record_message_rejected("websocket", oversized=True)
                elif not websocket.closed:
                    raise
        finally:
            self._clients.discard(websocket)
            self.admission.release("websocket")

    async def _wait_for_response(
        self,
        response_queue: "queue.Queue[dict[str, object]]",
        websocket: Any,
    ) -> dict[str, object] | None:
        """轮询跨线程 response queue，同时允许连接关闭或 runtime quit 中断等待。"""

        while not websocket.closed:
            if self.quit_event is not None and self.quit_event.is_set():
                return None
            try:
                return response_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(self.response_poll_interval_s)
        return None

    async def _send_rejection(
        self,
        websocket: Any,
        error: str,
        *,
        code: str,
        **details: object,
    ) -> None:
        """把 transport 边界错误编码成统一 rejection 并发送给当前客户端。"""

        await websocket.send(
            strict_json_dumps(
                _rejected_response(error, code=code, **details),
                ensure_ascii=False,
            )
        )

    async def _publish_events(self) -> None:
        """从全局 outbound 队列取事件，并广播给发送时仍连接的客户端。"""

        while True:
            if not self._clients:
                await asyncio.sleep(self.server_poll_interval_s)
                continue
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(self.server_poll_interval_s)
                continue
            payload = strict_json_dumps(dict(event), ensure_ascii=False)
            clients = tuple(self._clients)
            results = await asyncio.gather(
                *(client.send(payload) for client in clients),
                return_exceptions=True,
            )
            for client, result in zip(clients, results, strict=True):
                if isinstance(result, BaseException):
                    self._clients.discard(client)

    async def _watch_quit_event(self) -> None:
        """把线程侧 runtime quit event 桥接为 asyncio stop event。"""

        while True:
            if self.quit_event is not None and self.quit_event.is_set():
                if self._stop_event is not None:
                    self._stop_event.set()
                return
            await asyncio.sleep(self.server_poll_interval_s)


def start_websocket_server(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    *,
    quit_event: threading.Event | None,
    host: str,
    port: int,
    max_message_bytes: int = 1_048_576,
    max_connections: int = 16,
    event_queue_capacity: int = 256,
    server_poll_interval_s: float = 0.1,
    response_poll_interval_s: float = 0.5,
    startup_timeout_s: float = 5.0,
    admission: SharedTransportAdmission | None = None,
) -> WebSocketServerHandle:
    """启动 WebSocket JSON adapter，并返回必须显式关闭的拥有型句柄。

    所有请求只进入主线程 queue；若 TCP 与 WebSocket 同时启用，调用方应传入同一个
    ``SharedTransportAdmission``，使 ``max_connections`` 表示进程级总额。启动确认
    失败时，本 helper 会先有界停止尚未返回的句柄，再重新抛出原始启动异常。
    """

    host = require_loopback_host(host, label="host")
    validated_max_connections = _positive_int(max_connections, label="max_connections")
    admission = admission or SharedTransportAdmission(
        max_connections=validated_max_connections
    )
    handle = WebSocketServerHandle(
        request_queue,
        quit_event=quit_event,
        host=host,
        port=port,
        max_message_bytes=_positive_int(max_message_bytes, label="max_message_bytes"),
        max_connections=validated_max_connections,
        event_queue_capacity=_positive_int(
            event_queue_capacity, label="event_queue_capacity"
        ),
        server_poll_interval_s=_positive_float(
            server_poll_interval_s, label="server_poll_interval_s"
        ),
        response_poll_interval_s=_positive_float(
            response_poll_interval_s, label="response_poll_interval_s"
        ),
        admission=admission,
    )
    validated_startup_timeout_s = _positive_float(
        startup_timeout_s,
        label="startup_timeout_s",
    )
    try:
        handle.start(startup_timeout_s=validated_startup_timeout_s)
    except BaseException as startup_error:
        try:
            cleanup_status = handle.stop(timeout_s=validated_startup_timeout_s)
        except BaseException as cleanup_error:
            startup_error.add_note(
                "WebSocket startup cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        else:
            if cleanup_status.get("thread_alive"):
                startup_error.add_note(
                    "WebSocket startup cleanup timed out; "
                    "the server thread is still alive"
                )
        raise
    return handle


def _enqueue_interactive_request(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    request: _InteractiveRequest,
) -> dict[str, object] | None:
    """非阻塞投递请求；满队列返回可直接发给客户端的拒绝响应。"""

    try:
        request_queue.put_nowait(request)
    except queue.Full:
        _record_queue_rejection(request_queue)
        return _rejected_response(
            "interactive request queue is full",
            code="request_queue_full",
            capacity=request_queue.maxsize,
        )
    return None


def _record_queue_rejection(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
) -> None:
    """为标准/自定义 Queue 兼容地记录一次外部 admission 拒绝，避免重复计数。"""

    if getattr(request_queue, "_records_rejections_internally", False):
        return
    record = getattr(request_queue, "record_rejection", None)
    if callable(record):
        record()


def _rejected_response(
    error: str,
    *,
    code: str,
    **details: object,
) -> dict[str, object]:
    """构造 transport 共享的机器可读拒绝响应。"""

    return {"event": "rejected", "error": error, "code": code, **details}


def _read_bounded_line(
    stream: object,
    *,
    max_message_bytes: int,
) -> tuple[bytes, bool]:
    """读取一行并按不含 CRLF 的 payload 字节数执行容量限制。"""

    readline = getattr(stream, "readline")
    chunk = readline(max_message_bytes + 3)
    if chunk == b"":
        return b"", False
    complete_line = chunk.endswith(b"\n")
    payload = chunk.removesuffix(b"\n").removesuffix(b"\r")
    oversized = len(payload) > max_message_bytes
    if oversized and not complete_line:
        while chunk and not chunk.endswith(b"\n"):
            chunk = readline(max_message_bytes + 3)
    return (b"oversized", True) if oversized else (payload, False)


def _put_control_reliably(
    request_queue: "queue.Queue[_InteractiveRequest | _InteractiveControl]",
    control: _InteractiveControl,
    *,
    stop_event: threading.Event | None = None,
) -> bool:
    """control 不得因 data 满而丢失；标准 Queue 满时等待消费者释放。"""

    while stop_event is None or not stop_event.is_set():
        try:
            request_queue.put(control, timeout=_CONTROL_QUEUE_RETRY_TIMEOUT_S)
            return True
        except queue.Full:
            continue
    return False


def _reject_connection(sock: object, response: Mapping[str, object]) -> None:
    """在 handler 创建前尽力向被拒 TCP socket 写回一条响应。"""

    if not isinstance(sock, socket.socket):
        return
    try:
        sock.sendall(
            (strict_json_dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
        )
    except OSError:
        pass


def _websocket_close_code(exc: BaseException) -> int | None:
    """兼容 websockets 的 received/sent close frame 与最终聚合 code。"""

    for value in (getattr(exc, "sent", None), getattr(exc, "rcvd", None), exc):
        code = getattr(value, "code", None)
        if isinstance(code, int) and code != 1006:
            return code
    return None


def _positive_int(value: object, *, label: str) -> int:
    """校验严格正整数设置，不接受 bool 或隐式数值转换。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _positive_float(value: object, *, label: str) -> float:
    """校验有限正数设置，不接受 bool。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return parsed


def _non_negative_float(value: object, *, label: str) -> float:
    """校验有限非负数设置，不接受 bool。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return parsed


def _handle_json_line(
    line: str,
    runtime: object,
) -> dict[str, object]:
    """解析一行 JSONL 并转成 rejected/response。"""

    try:
        message = strict_json_loads(line)
        if not isinstance(message, Mapping):
            raise ValueError("message must be a JSON object")
        return handle_tiled_interactive_message(message, runtime)
    except Exception as exc:
        return {"event": "rejected", "error": str(exc)}


def _quit_on_stdin_eof(
    *,
    stdin_eof_policy: StdinEofPolicy = "exit",
    tcp_jsonl_port: int | None,
    telemetry: TiledInteractiveTelemetrySink | None,
    keepalive_consumer_active: bool = False,
) -> bool:
    """按显式 stdin 策略判断 EOF 是否应结束进程。"""

    return (
        stdin_eof_policy == "exit"
        and tcp_jsonl_port is None
        and telemetry is None
        and not keepalive_consumer_active
    )


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
    queue_poll_timeout_s: float = 0.1,
) -> float:
    """返回主循环等待请求队列的 timeout。"""

    deadlines = [time.monotonic() + float(queue_poll_timeout_s)]
    if telemetry_period_s is not None:
        deadlines.append(next_telemetry_at)
    if idle_period_s is not None and next_idle_at is not None:
        deadlines.append(next_idle_at)
    return max(0.0, min(deadlines) - time.monotonic())


def _runtime_idle_period_s(
    runtime: object,
    *,
    idle_physics_policy: IdlePhysicsPolicy,
    idle_step_duration_s: float | None = None,
) -> float | None:
    """判断主循环是否需要空闲刷新，以及刷新周期。"""

    if idle_physics_policy != "hold_step":
        return None
    idle_step = getattr(runtime, "idle_step", None)
    if not callable(idle_step):
        raise ValueError("hold_step requires runtime.idle_step()")
    if idle_step_duration_s is None:
        raise ValueError("hold_step requires a positive idle_step_duration_s")
    return _positive_float(idle_step_duration_s, label="idle_step_duration_s")


def _runtime_idle_step_count(
    runtime: object,
    idle_step_duration_s: float,
) -> int:
    """把配置的 idle chunk 对齐到 runtime physics grid。"""

    duration_s = _positive_float(idle_step_duration_s, label="idle_step_duration_s")
    session = getattr(runtime, "session", None)
    world = getattr(session, "world", None)
    physics_dt_getter = getattr(world, "get_physics_dt", None)
    if not callable(physics_dt_getter):
        raise ValueError(
            "idle_step_duration_s requires runtime.session.world.get_physics_dt()"
        )
    try:
        physics_dt = _positive_float(physics_dt_getter(), label="runtime physics dt")
    except Exception as exc:
        raise ValueError(
            "idle_step_duration_s requires a positive finite runtime physics dt"
        ) from exc
    requested_steps = duration_s / physics_dt
    if not math.isfinite(requested_steps):
        raise ValueError("idle_step_duration_s resolves to a non-finite tick count")
    count = max(1, int(math.ceil(requested_steps - 1.0e-12)))
    if count > _MAX_IDLE_STEPS_PER_CHUNK:
        raise ValueError(
            "idle_step_duration_s resolves to too many physics ticks; "
            f"maximum is {_MAX_IDLE_STEPS_PER_CHUNK}"
        )
    return count


def _runtime_idle_step(runtime: object, *, count: int = 1) -> None:
    """执行一个 runtime 空闲 chunk；异常交给主循环外层日志捕获。"""

    idle_step = getattr(runtime, "idle_step", None)
    if callable(idle_step):
        for _ in range(count):
            idle_step()
