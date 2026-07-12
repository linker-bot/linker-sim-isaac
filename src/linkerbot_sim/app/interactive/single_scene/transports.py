"""Single Scene 交互模式的 stdin、TCP JSONL 与 WebSocket transport。

所有 transport 都把 JSON 转成纯数据命令后写入同一个 ``InteractiveMotionQueue``。
后台线程不访问 Isaac；WebSocket 额外订阅队列事件，TCP/stdin 则保持一问一答语义。

TCP 连接各自由 socketserver 线程处理，WebSocket server 在专用线程中运行 asyncio loop，
stdin 由可中断 reader 线程拥有。三者只共享线程安全队列、停止事件和指标；runtime 状态
始终由仿真主线程独占。消息长度在 JSON 解码前限制，TCP 与 WebSocket 共用连接 admission，
避免启用多个入口后分别越过进程级资源上限。
"""

from __future__ import annotations

import asyncio
import socket
import socketserver
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import cast

from linkerbot_sim.app.interactive.policies import StdinEofPolicy
from linkerbot_sim.utils.json import (
    strict_json_dumps,
    strict_json_loads,
)
from linkerbot_sim.app.interactive.single_scene.protocol import (
    InteractiveMotionCommand,
    parse_interactive_motion_message,
)
from linkerbot_sim.app.interactive.single_scene.queue import (
    InteractiveMotionQueue,
    InteractiveQueueFullError,
    InteractiveRequestConflictError,
    SnapshotRequestKind,
)
from linkerbot_sim.app.interactive.stdin_reader import (
    InterruptibleStdinJsonlReader,
    StdinJsonlFrame,
    start_interruptible_stdin_jsonl_reader,
)
from linkerbot_sim.utils.config import require_loopback_host


@dataclass(frozen=True)
class TransportShutdownReport:
    """在统一关闭 deadline 内停止 transport 资源的结果。

    ``stopped=False`` 不会隐藏仍存活的线程或连接；调用方可通过 ``live_resources`` 诊断
    哪个回调或 socket 未在期限内收敛，并可保留 handles 后续再次停止。
    """

    stopped: bool
    live_resources: tuple[str, ...]


class _TransportMetrics:
    """供 TCP、WebSocket 与 status 响应共享的线程安全计数器。

    计数是进程期累计值，depth 则反映瞬时占用。所有复合读写都在同一把锁内完成，保证
    status 中每个子结构来自一致快照；指标不参与业务控制，实际 admission 仍在对应的
    ``try_acquire``/``release`` 临界区完成。
    """

    def __init__(
        self,
        *,
        max_message_bytes: int,
        max_connections: int,
        event_queue_capacity: int,
    ) -> None:
        self.max_message_bytes = max_message_bytes
        self.max_connections = max_connections
        self.event_queue_capacity = event_queue_capacity
        self._lock = Lock()
        self._active_connections = 0
        self._rejected_connections = 0
        self._messages_received = 0
        self._messages_rejected = 0
        self._oversized_messages = 0
        self._event_depth = 0
        self._events_rejected = 0
        self._events_discarded = 0
        self._shutdown_stopped: bool | None = None
        self._live_resources: tuple[str, ...] = ()

    def try_acquire_connection(self) -> bool:
        """原子预留一个 TCP/WebSocket 共用连接槽位；达到上限时记录拒绝。"""

        with self._lock:
            if self._active_connections >= self.max_connections:
                self._rejected_connections += 1
                return False
            self._active_connections += 1
            return True

    def release_connection(self) -> None:
        """释放先前预留的连接槽位；重复清理不会把诊断深度减成负数。"""

        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    def record_message(self, *, rejected: bool, oversized: bool = False) -> None:
        """记录一条已经完成 framing 的入站消息及其最终接纳结果。"""

        with self._lock:
            self._messages_received += 1
            if rejected:
                self._messages_rejected += 1
            if oversized:
                self._oversized_messages += 1

    def record_event_enqueued(self) -> None:
        """增加所有 WebSocket 客户端 outbound 事件队列的聚合深度。"""

        with self._lock:
            self._event_depth += 1

    def record_event_dequeued(self) -> None:
        """sender 取得一个 outbound 事件后减少聚合深度。"""

        with self._lock:
            self._event_depth = max(0, self._event_depth - 1)

    def record_event_rejected(self) -> None:
        """记录因某个客户端队列已满而按 reject-new 策略丢弃的新事件。"""

        with self._lock:
            self._events_rejected += 1

    def discard_events(self, count: int) -> None:
        """连接断开时从聚合深度移除该客户端尚未发送的事件。"""

        if count <= 0:
            return
        with self._lock:
            self._event_depth = max(0, self._event_depth - count)
            self._events_discarded += count

    def record_shutdown(self, report: TransportShutdownReport) -> None:
        """保存最近一次关闭结果，同时保留未退出资源的诊断名称。"""

        with self._lock:
            self._shutdown_stopped = report.stopped
            self._live_resources = report.live_resources

    def active_connection_count(self) -> int:
        """返回当前连接深度，供关闭报告判断是否仍有 handler 存活。"""

        with self._lock:
            return self._active_connections

    def status(self) -> dict[str, object]:
        """返回 JSON-compatible 的实时深度、容量和累计计数快照。"""

        with self._lock:
            active_connections = self._active_connections
            return {
                "connections": {
                    "depth": active_connections,
                    "capacity": self.max_connections,
                    "rejected": self._rejected_connections,
                },
                "messages": {
                    "received": self._messages_received,
                    "rejected": self._messages_rejected,
                    "oversized": self._oversized_messages,
                    "max_message_bytes": self.max_message_bytes,
                },
                "events": {
                    "depth": self._event_depth,
                    "capacity": self.event_queue_capacity,
                    "aggregate_capacity": (
                        self.event_queue_capacity * active_connections
                    ),
                    "rejected": self._events_rejected,
                    "discarded": self._events_discarded,
                },
                "shutdown": {
                    "stopped": self._shutdown_stopped,
                    "live_resources": list(self._live_resources),
                },
            }


@dataclass
class InteractiveTransportHandles:
    """统一拥有已启动后台线程、停止事件和 server/reader 资源的关闭句柄。

    启动函数把所有成功创建的资源移交给本对象；即使后续某个 transport 启动失败，也会
    构造临时 handles 执行同一套回滚。``stop`` 由内部锁串行化，可安全地被 finally 与
    外部关闭路径重复调用。
    """

    threads: tuple[Thread, ...]
    stop_event: Event
    tcp_server: socketserver.ThreadingTCPServer | None = None
    stdin_reader: InterruptibleStdinJsonlReader | None = field(
        default=None,
        repr=False,
    )
    shutdown_timeout_s: float = 2.0
    _metrics: _TransportMetrics | None = field(default=None, repr=False)
    _stop_lock: Lock = field(default_factory=Lock, repr=False)
    _tcp_shutdown_thread: Thread | None = field(default=None, repr=False)
    last_shutdown_report: TransportShutdownReport | None = None

    def status(self) -> dict[str, object]:
        """返回可嵌入交互队列 status payload 的 transport 指标。"""

        return {} if self._metrics is None else self._metrics.status()

    def stop(self, *, timeout_s: float | None = None) -> TransportShutdownReport:
        """按统一 deadline 停止并 join 全部 transport 资源。

        顺序是：先设置共享停止事件并唤醒 stdin；再关闭活跃 TCP socket 以解除
        ``readline``；随后在独立线程调用可能阻塞的 server shutdown；最后用同一个剩余
        时间预算 join 所有线程。超时只产生报告，不会遗失尚存活资源的句柄。
        """

        timeout = self.shutdown_timeout_s if timeout_s is None else float(timeout_s)
        if timeout < 0.0:
            raise ValueError("transport shutdown timeout_s must be >= 0")
        with self._stop_lock:
            deadline = time.monotonic() + timeout
            self.stop_event.set()
            stdin_reader = self.stdin_reader
            if stdin_reader is not None:
                stdin_stopped = stdin_reader.stop(
                    timeout_s=max(0.0, deadline - time.monotonic())
                )
                if stdin_stopped:
                    self.stdin_reader = None
            shutdown_thread = self._tcp_shutdown_thread
            if self.tcp_server is not None:
                close_active = getattr(self.tcp_server, "close_active_requests", None)
                if callable(close_active):
                    close_active()

                if shutdown_thread is None:

                    def stop_tcp_server() -> None:
                        """唤醒 ``serve_forever``，退出后关闭其监听 socket。"""

                        assert self.tcp_server is not None
                        self.tcp_server.shutdown()
                        self.tcp_server.server_close()

                    shutdown_thread = Thread(
                        target=stop_tcp_server,
                        daemon=True,
                        name="interactive-motion-tcp-shutdown",
                    )
                    self._tcp_shutdown_thread = shutdown_thread
                    shutdown_thread.start()

            resources = list(self.threads)
            if shutdown_thread is not None:
                resources.append(shutdown_thread)
            for thread in resources:
                remaining = max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)

            live_resources = [thread.name for thread in resources if thread.is_alive()]
            if self.stdin_reader is not None and self.stdin_reader.is_alive():
                live_resources.append(self.stdin_reader.name)
            if self._metrics is not None:
                connection_depth = self._metrics.active_connection_count()
                if connection_depth:
                    live_resources.append(f"interactive-connections:{connection_depth}")
            report = TransportShutdownReport(
                stopped=not live_resources,
                live_resources=tuple(live_resources),
            )
            self.last_shutdown_report = report
            if self._metrics is not None:
                self._metrics.record_shutdown(report)
            return report


def start_interactive_transports(
    *,
    queue: InteractiveMotionQueue,
    stdin_enabled: bool = True,
    tcp_jsonl_host: str | None = None,
    tcp_jsonl_port: int | None = None,
    websocket_host: str | None = None,
    websocket_port: int | None = None,
    stdin_eof_policy: StdinEofPolicy = "exit",
    keepalive_consumer_active: bool = False,
    max_message_bytes: int = 1_048_576,
    max_connections: int = 16,
    event_queue_capacity: int = 256,
    startup_timeout_s: float = 5.0,
    server_poll_interval_s: float = 0.1,
    response_poll_interval_s: float = 0.5,
    shutdown_timeout_s: float = 2.0,
) -> InteractiveTransportHandles:
    """按配置启动 transport 后台线程，并返回统一关闭句柄。

    ``exit`` 只在没有远程 transport 时把 stdin EOF 转成 quit；``keep_alive`` 始终保持
    进程。TCP/WebSocket 服务不会因为本地 stdin 关闭而终止。

    远端端点会在 stdin reader 启动前完成 bind/就绪确认；因此同步 bind 失败或 WebSocket
    启动超时可以完整回滚，不会遗留一个阻塞在进程输入上的无主线程。成功返回后调用方
    必须保存 handles，并在 runtime 停止前调用 ``stop``。
    """

    if tcp_jsonl_host is not None:
        tcp_jsonl_host = require_loopback_host(
            tcp_jsonl_host,
            label="tcp_jsonl_host",
        )
    if websocket_host is not None:
        websocket_host = require_loopback_host(
            websocket_host,
            label="websocket_host",
        )
    max_message_bytes = _positive_int(max_message_bytes, "max_message_bytes")
    max_connections = _positive_int(max_connections, "max_connections")
    event_queue_capacity = _positive_int(event_queue_capacity, "event_queue_capacity")
    startup_timeout_s = _positive_float(startup_timeout_s, "startup_timeout_s")
    server_poll_interval_s = _positive_float(
        server_poll_interval_s, "server_poll_interval_s"
    )
    response_poll_interval_s = _positive_float(
        response_poll_interval_s, "response_poll_interval_s"
    )
    shutdown_timeout_s = _nonnegative_float(shutdown_timeout_s, "shutdown_timeout_s")
    metrics = _TransportMetrics(
        max_message_bytes=max_message_bytes,
        max_connections=max_connections,
        event_queue_capacity=event_queue_capacity,
    )
    queue.set_transport_status_provider(metrics.status)
    stop_event = Event()
    threads: list[Thread] = []
    tcp_server = None
    stdin_reader = None
    try:
        # 先绑定远端端点再启动 stdin，避免同步 bind 错误遗留一个阻塞在进程输入上的
        # 无主 reader。
        if tcp_jsonl_port is not None:
            tcp_server = _start_tcp_jsonl_server(
                queue=queue,
                host=tcp_jsonl_host or "127.0.0.1",
                port=int(tcp_jsonl_port),
                max_message_bytes=max_message_bytes,
                metrics=metrics,
            )
            thread = Thread(
                target=tcp_server.serve_forever,
                kwargs={"poll_interval": server_poll_interval_s},
                daemon=True,
                name="interactive-motion-tcp-jsonl",
            )
            try:
                thread.start()
            except BaseException:
                tcp_server.server_close()
                tcp_server = None
                raise
            threads.append(thread)
        if websocket_port is not None:
            websocket_ready = Event()
            websocket_errors: list[BaseException] = []

            def run_websocket() -> None:
                """在线程中运行 WebSocket loop，并把启动异常交回所有者线程。"""

                try:
                    _run_websocket_server(
                        queue=queue,
                        host=websocket_host or "127.0.0.1",
                        port=int(websocket_port),
                        stop_event=stop_event,
                        max_message_bytes=max_message_bytes,
                        event_queue_capacity=event_queue_capacity,
                        server_poll_interval_s=server_poll_interval_s,
                        response_poll_interval_s=response_poll_interval_s,
                        metrics=metrics,
                        startup_event=websocket_ready,
                    )
                except BaseException as exc:
                    websocket_errors.append(exc)
                    websocket_ready.set()

            thread = Thread(
                target=run_websocket,
                daemon=True,
                name="interactive-motion-websocket",
            )
            thread.start()
            threads.append(thread)
            if not websocket_ready.wait(timeout=startup_timeout_s):
                raise TimeoutError(
                    "interactive WebSocket server did not bind before startup timeout"
                )
            if websocket_errors:
                raise RuntimeError(
                    "interactive WebSocket server failed during startup"
                ) from websocket_errors[0]
        if stdin_enabled:
            quit_on_eof = (
                stdin_eof_policy == "exit"
                and tcp_jsonl_port is None
                and websocket_port is None
                and not keepalive_consumer_active
            )
            stdin_reader = start_interruptible_stdin_jsonl_reader(
                stream=sys.stdin,
                max_message_bytes=max_message_bytes,
                on_frame=lambda frame: _handle_stdin_frame(
                    frame,
                    queue=queue,
                    stop_event=stop_event,
                    max_message_bytes=max_message_bytes,
                    metrics=metrics,
                ),
                on_eof=lambda: _handle_stdin_eof(
                    queue=queue,
                    stop_event=stop_event,
                    quit_on_eof=quit_on_eof,
                ),
                thread_name="interactive-motion-stdin",
                shutdown_event=stop_event,
            )
        return InteractiveTransportHandles(
            threads=tuple(threads),
            stop_event=stop_event,
            tcp_server=tcp_server,
            stdin_reader=stdin_reader,
            shutdown_timeout_s=shutdown_timeout_s,
            _metrics=metrics,
        )
    except BaseException:
        rollback = InteractiveTransportHandles(
            threads=tuple(threads),
            stop_event=stop_event,
            tcp_server=tcp_server,
            stdin_reader=stdin_reader,
            shutdown_timeout_s=shutdown_timeout_s,
            _metrics=metrics,
        )
        report = rollback.stop(timeout_s=shutdown_timeout_s)
        queue.set_transport_status_provider(None)
        if not report.stopped:
            print(
                "SINGLE_SCENE_INTERACTIVE_STARTUP_ROLLBACK_TIMEOUT "
                f"live_resources={list(report.live_resources)}",
                flush=True,
            )
        raise


def handle_interactive_message(
    *,
    message: Mapping[str, object],
    queue: InteractiveMotionQueue,
) -> dict[str, object]:
    """解析一条 transport message，并把 canonical command 应用到共享队列。

    解析错误由外层消息边界转成 rejected；这里仅把 admission/单例冲突转换成稳定响应，
    使所有 transport 对同一条合法但暂不可接纳的命令返回相同字段。
    """

    command = parse_interactive_motion_message(
        message,
        planner_defaults=getattr(queue, "planner_request_defaults", None),
        command_defaults=getattr(queue, "command_defaults", None),
    )
    try:
        return _apply_command(command, queue)
    except (InteractiveQueueFullError, InteractiveRequestConflictError) as exc:
        return exc.response(request_id=_interactive_command_id(command))


def _apply_command(
    command: InteractiveMotionCommand,
    queue: InteractiveMotionQueue,
) -> dict[str, object]:
    """把已解析命令应用到队列，并返回可直接发给客户端的响应。"""

    if command.kind == "timeline":
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
            clear_queue=command.reset_clear_queue,
            hold_after_reset=command.reset_hold_after_reset,
        )
        return {"event": "reset", "accepted": True, **request.snapshot()}
    if command.kind in {"get_snapshot", "set_snapshot"}:
        # snapshot 命令需要读写 runtime/PhysX 状态，必须经由 queue 交给仿真主线程；
        # request_snapshot 会在 transport 线程里等待主线程返回 response。
        return queue.request_snapshot(
            kind=cast(SnapshotRequestKind, command.kind),
            snapshot_id=command.command_id,
            snapshot=command.snapshot,
            label_map=command.snapshot_label_map,
            strict=command.snapshot_strict,
        )
    if command.kind == "estop":
        queue.request_estop()
        return {"event": "estop", "accepted": True}
    if command.kind == "quit":
        queue.request_quit()
        return {"event": "quit", "accepted": True}
    raise ValueError(f"unsupported command kind: {command.kind!r}")


def _handle_stdin_frame(
    frame: StdinJsonlFrame,
    *,
    queue: InteractiveMotionQueue,
    stop_event: Event,
    max_message_bytes: int,
    metrics: _TransportMetrics,
) -> None:
    """处理一条已经有界 framing 的 stdin 记录，不再访问原始 stream。

    reader 线程是本函数的调用者；输出一行响应后立即 flush，保证管道调用方不会因 Python
    缓冲等待。停止事件先于 frame 处理检查，关闭竞态中不会继续接纳新命令。
    """

    if stop_event.is_set():
        return
    if frame.oversized:
        metrics.record_message(rejected=True, oversized=True)
        response = _message_too_large_response(max_message_bytes)
    elif frame.decode_error is not None:
        metrics.record_message(rejected=True)
        response = {
            "event": "rejected",
            "accepted": False,
            "reason": "invalid_encoding",
            "error": f"message must be UTF-8: {frame.decode_error}",
        }
    else:
        assert frame.text is not None
        response = _handle_json_line(
            frame.text,
            queue=queue,
            max_message_bytes=max_message_bytes,
            metrics=metrics,
        )
    print(strict_json_dumps(response, ensure_ascii=False), flush=True)


def _handle_stdin_eof(
    *,
    queue: InteractiveMotionQueue,
    stop_event: Event,
    quit_on_eof: bool,
) -> None:
    """仅在自然 EOF 赢得关闭竞态时应用已配置的 EOF 退出策略。"""

    if quit_on_eof and not stop_event.is_set():
        queue.request_quit()


def _start_tcp_jsonl_server(
    *,
    queue: InteractiveMotionQueue,
    host: str,
    port: int,
    max_message_bytes: int,
    metrics: _TransportMetrics,
) -> socketserver.ThreadingTCPServer:
    """启动一条 TCP JSONL 服务；每行一个 JSON 请求，每行一个 JSON 响应。"""

    class _Handler(socketserver.StreamRequestHandler):
        """TCP 连接处理器；一个连接内可连续发送多行 JSON。"""

        def handle(self) -> None:
            """循环读取客户端 JSONL，并把每条响应写回同一连接。"""

            while True:
                line, oversized = _read_bounded_jsonl(
                    self.rfile,
                    max_message_bytes=max_message_bytes,
                )
                if line is None and not oversized:
                    return
                if oversized:
                    response = _message_too_large_response(max_message_bytes)
                    metrics.record_message(rejected=True, oversized=True)
                else:
                    assert line is not None
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        response = _rejected_response(exc)
                        metrics.record_message(rejected=True)
                    else:
                        response = _handle_json_line(
                            text,
                            queue=queue,
                            metrics=metrics,
                        )
                self.wfile.write(
                    (strict_json_dumps(response, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                )

    class _Server(socketserver.ThreadingTCPServer):
        """允许端口快速复用、连接线程后台化的 JSONL server。"""

        allow_reuse_address = True
        daemon_threads = True

        def __init__(self, server_address, request_handler_class) -> None:
            self._active_requests: set[socket.socket] = set()
            self._active_requests_lock = Lock()
            super().__init__(server_address, request_handler_class)

        def verify_request(self, request, client_address) -> bool:
            """在创建 handler 线程前拒绝超过 TCP/WebSocket 共用上限的连接。"""

            del client_address
            request_socket = cast(socket.socket, request)
            if not metrics.try_acquire_connection():
                response = {
                    "event": "rejected",
                    "accepted": False,
                    "code": "connection_limit",
                    "reason": "max_connections",
                    "error": "interactive transport connection limit reached",
                    "capacity": metrics.max_connections,
                }
                try:
                    request_socket.sendall(
                        (strict_json_dumps(response, ensure_ascii=False) + "\n").encode(
                            "utf-8"
                        )
                    )
                except OSError:
                    pass
                return False
            with self._active_requests_lock:
                self._active_requests.add(request_socket)
            return True

        def process_request_thread(self, request, client_address) -> None:
            """在 handler 的所有退出路径完成后释放连接 admission。"""

            try:
                super().process_request_thread(request, client_address)
            finally:
                with self._active_requests_lock:
                    self._active_requests.discard(cast(socket.socket, request))
                metrics.release_connection()

        def close_active_requests(self) -> None:
            """关闭活跃 socket，解除有界关闭期间阻塞在 ``readline`` 的 handlers。"""

            with self._active_requests_lock:
                requests = tuple(self._active_requests)
            for request in requests:
                try:
                    request.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    request.close()
                except OSError:
                    pass

    if ":" in host:
        _Server.address_family = socket.AF_INET6
    return _Server((host, port), _Handler)


def _handle_json_line(
    line: str,
    *,
    queue: InteractiveMotionQueue,
    max_message_bytes: int | None = None,
    metrics: _TransportMetrics | None = None,
) -> dict[str, object]:
    """解析一行 JSONL 并执行；所有异常都会转成 rejected 响应。"""

    if max_message_bytes is not None and len(line.encode("utf-8")) > max_message_bytes:
        response = _message_too_large_response(max_message_bytes)
        if metrics is not None:
            metrics.record_message(rejected=True, oversized=True)
        return response
    try:
        message = strict_json_loads(line)
        if not isinstance(message, Mapping):
            raise ValueError("message must be a JSON object")
        response = handle_interactive_message(
            message=message,
            queue=queue,
        )
    except Exception as exc:
        response = _rejected_response(exc)
    if metrics is not None:
        metrics.record_message(rejected=response.get("event") == "rejected")
    return response


def _run_websocket_server(
    *,
    queue: InteractiveMotionQueue,
    host: str,
    port: int,
    stop_event: Event,
    max_message_bytes: int,
    event_queue_capacity: int,
    server_poll_interval_s: float,
    response_poll_interval_s: float,
    metrics: _TransportMetrics,
    startup_event: Event,
) -> None:
    """运行 WebSocket 传输，支持请求响应和队列事件异步推送。"""

    try:
        import websockets
    except ImportError as exc:
        print(
            "SINGLE_SCENE_INTERACTIVE_WEBSOCKET_UNAVAILABLE "
            "error=websockets_package_not_installed",
            flush=True,
        )
        raise RuntimeError(
            "websockets package is required for WebSocket transport"
        ) from exc

    async def handler(websocket) -> None:
        """服务单个 WebSocket 客户端，并把队列事件转发给该连接。

        request response 与主动事件共享 ``send_lock``，确保同一连接上两个协程不会并发
        调用 ``send``。断开时先注销同步 listener，再取消 sender，最后释放未发送事件和
        连接 admission，关闭后的队列事件因而不会继续流入孤立队列。
        """

        if not metrics.try_acquire_connection():
            response = {
                "event": "rejected",
                "accepted": False,
                "code": "connection_limit",
                "reason": "max_connections",
                "error": "interactive transport connection limit reached",
                "capacity": metrics.max_connections,
            }
            await websocket.send(strict_json_dumps(response, ensure_ascii=False))
            await websocket.close(code=1013, reason="max_connections")
            return

        event_queue: Queue[dict[str, object]] = Queue(maxsize=event_queue_capacity)

        def listener(event: dict[str, object]) -> None:
            """把同步队列事件非阻塞转存到线程安全队列，供 async sender 消费。"""

            _enqueue_transport_event(event_queue, event, metrics=metrics)

        queue.add_listener(listener)
        send_lock = asyncio.Lock()

        async def sender() -> None:
            """后台发送队列事件，避免同步 listener 或消息接收循环被网络背压阻塞。"""

            while not stop_event.is_set():
                try:
                    event = await asyncio.to_thread(
                        event_queue.get,
                        True,
                        response_poll_interval_s,
                    )
                except Empty:
                    continue
                metrics.record_event_dequeued()
                async with send_lock:
                    await websocket.send(strict_json_dumps(event, ensure_ascii=False))

        sender_task = asyncio.create_task(sender())
        try:
            async for text in websocket:
                if isinstance(text, bytes):
                    message_size = len(text)
                    if message_size > max_message_bytes:
                        response = _message_too_large_response(max_message_bytes)
                        metrics.record_message(rejected=True, oversized=True)
                    else:
                        try:
                            decoded_text = text.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            response = _rejected_response(exc)
                            metrics.record_message(rejected=True)
                        else:
                            response = await asyncio.to_thread(
                                _handle_json_line,
                                decoded_text,
                                queue=queue,
                                max_message_bytes=max_message_bytes,
                                metrics=metrics,
                            )
                else:
                    response = await asyncio.to_thread(
                        _handle_json_line,
                        text,
                        queue=queue,
                        max_message_bytes=max_message_bytes,
                        metrics=metrics,
                    )
                async with send_lock:
                    await websocket.send(
                        strict_json_dumps(response, ensure_ascii=False)
                    )
        except websockets.exceptions.ConnectionClosedError as exc:
            close_codes = {
                close.code for close in (exc.rcvd, exc.sent) if close is not None
            }
            if 1009 in close_codes:
                metrics.record_message(rejected=True, oversized=True)
        finally:
            queue.remove_listener(listener)
            sender_task.cancel()
            await asyncio.gather(sender_task, return_exceptions=True)
            metrics.discard_events(event_queue.qsize())
            metrics.release_connection()

    async def main() -> None:
        """创建 WebSocket server，报告 bind 就绪，并轮询线程停止事件退出。"""

        async with websockets.serve(
            handler,
            host,
            port,
            compression=None,
            max_size=max_message_bytes + 1,
            max_queue=1,
        ):
            startup_event.set()
            while not stop_event.is_set():
                await asyncio.sleep(server_poll_interval_s)

    asyncio.run(main())


def _interactive_command_id(command: InteractiveMotionCommand) -> str | None:
    """返回被拒命令在其具体 kind 中使用的协议 ID。"""

    if command.kind == "reset":
        return command.reset_id
    if command.kind == "cancel":
        return command.cancel_id
    return command.command_id


def _enqueue_transport_event(
    event_queue: Queue[dict[str, object]],
    event: dict[str, object],
    *,
    metrics: _TransportMetrics,
) -> bool:
    """非阻塞加入 WebSocket 事件，队列满时按确定的 reject-new 策略计数。"""

    try:
        event_queue.put_nowait(event)
    except Full:
        metrics.record_event_rejected()
        return False
    metrics.record_event_enqueued()
    return True


def _read_bounded_jsonl(
    reader,
    *,
    max_message_bytes: int,
) -> tuple[bytes | None, bool]:
    """读取一条 JSONL，且单次保留不超过 payload 上限加 framing 字节。

    超长且尚未遇到换行时，剩余部分会分块 drain 到消息边界；下一次读取因而从完整的新
    消息开始，不会把被拒记录的尾部误当成另一条请求。
    """

    chunk = reader.readline(max_message_bytes + 2)
    if chunk == b"":
        return None, False
    payload = chunk.removesuffix(b"\n").removesuffix(b"\r")
    complete_line = chunk.endswith(b"\n")
    oversized = len(payload) > max_message_bytes or (
        not complete_line and len(chunk) > max_message_bytes
    )
    if oversized and not complete_line:
        _drain_jsonl_line(reader, chunk_size=max_message_bytes + 2)
    return (None, True) if oversized else (payload, False)


def _drain_jsonl_line(reader, *, chunk_size: int) -> None:
    """以有界 chunk 丢弃超长 JSONL frame 的剩余部分，直到 EOF 或换行。"""

    while True:
        chunk = reader.readline(chunk_size)
        if chunk == b"" or chunk.endswith(b"\n"):
            return


def _message_too_large_response(max_message_bytes: int) -> dict[str, object]:
    """返回所有 transport 共用的 canonical 消息越界响应。"""

    return {
        "event": "rejected",
        "accepted": False,
        "code": "message_too_large",
        "reason": "message_too_large",
        "error": f"message exceeds max_message_bytes={max_message_bytes}",
        "capacity": max_message_bytes,
    }


def _rejected_response(exc: Exception) -> dict[str, object]:
    """把格式、编码或协议异常转换为稳定的 malformed-input 响应。"""

    return {
        "event": "rejected",
        "accepted": False,
        "code": "invalid_message",
        "reason": "invalid_message",
        "error": str(exc),
    }


def _positive_int(value: int, name: str) -> int:
    """校验正整数 transport 设置，并显式拒绝 bool。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    """校验有限正数轮询间隔。"""

    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    """校验有限非负关闭 timeout。"""

    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result
