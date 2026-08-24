"""Mirror v1 的 stdin、TCP JSONL 与 WebSocket ingress。

后台线程只解析/提交纯 JSON 并等待所属 response；它们从不访问 USD、Isaac 或 planner。
所有 endpoint 共享同一个 controller/admission handler。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from ipaddress import ip_address
import select
import socket
import sys
from threading import Event, Lock, Thread, current_thread
from typing import TextIO

from linkerbot_sim.mirror.lifecycle import close_result_stopped

from .protocol import MirrorResponse, decode_request, encode_response


JsonHandler = Callable[[str], str]


def _positive_int(value: object, *, label: str) -> int:
    """拒绝 bool/零值，避免容量配置在运行时退化为无界或全拒绝。"""

    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_timeout(value: object, *, label: str) -> float:
    """解析严格正 deadline；NaN 也会因比较为假而被拒绝。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite positive number")
    parsed = float(value)
    if not parsed > 0.0 or parsed == float("inf"):
        raise ValueError(f"{label} must be a finite positive number")
    return parsed


def _transport_failure(*, code: str, message: str) -> str:
    """为 framing/admission 前失败生成稳定的 Mirror v1 response。"""

    return encode_response(
        MirrorResponse.failure(
            "invalid-request",
            code=code,
            message=message,
        )
    )


def make_json_handler(
    submit_and_wait: Callable[..., MirrorResponse],
    *,
    timeout_s: float,
) -> JsonHandler:
    """把 controller 的 typed API 适配为 transport 共用的文本 handler。"""

    def handle(payload: str) -> str:
        try:
            request = decode_request(payload)
        except Exception as exc:
            # admission 前无法可靠取得 request_id，使用固定诊断 ID；连接仍可继续处理下一条。
            return encode_response(
                MirrorResponse.failure(
                    "invalid-request",
                    code="invalid_request",
                    message=str(exc),
                )
            )
        try:
            response = submit_and_wait(request, timeout_s=timeout_s)
        except TimeoutError as exc:
            response = MirrorResponse.failure(
                request.request_id,
                code="response_timeout",
                message=str(exc),
                protocol=request.protocol,
            )
        return encode_response(response)

    return handle


@dataclass(frozen=True)
class TransportCloseReport:
    stopped: bool
    live_resources: tuple[str, ...] = ()


def _loopback_host(value: str) -> str:
    if value.casefold() == "localhost":
        return value
    try:
        address = ip_address(value)
    except ValueError as exc:
        raise ValueError(
            "Mirror transport host must be a numeric loopback address or localhost"
        ) from exc
    if not address.is_loopback:
        raise ValueError("Mirror transport only allows a loopback host")
    return value


class StdinJsonlTransport:
    """一行一请求的 stdin ingress；输出响应写到显式 stream。"""

    resource_name = "stdin_jsonl"

    def __init__(
        self,
        handler: JsonHandler,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        eof_requests_quit: Callable[[], None] | None = None,
        max_message_bytes: int = 1_048_576,
        poll_interval_s: float = 0.1,
        shutdown_timeout_s: float = 2.0,
    ) -> None:
        self._handler = handler
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout
        self._eof_requests_quit = eof_requests_quit
        self.max_message_bytes = _positive_int(
            max_message_bytes, label="max_message_bytes"
        )
        self.poll_interval_s = _positive_timeout(
            poll_interval_s, label="stdin poll_interval_s"
        )
        self.shutdown_timeout_s = _positive_timeout(
            shutdown_timeout_s, label="stdin shutdown_timeout_s"
        )
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("stdin transport is already started")
        self._thread = Thread(
            target=self._run,
            name="mirror-stdin-jsonl",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            line = self._readline()
            if line is None:
                continue
            if line == "":
                if self._eof_requests_quit is not None:
                    self._eof_requests_quit()
                return
            if not line.strip():
                continue
            size = len(line.encode("utf-8"))
            response = (
                _transport_failure(
                    code="message_too_large",
                    message=(
                        "stdin JSONL message exceeds max_message_bytes="
                        f"{self.max_message_bytes}"
                    ),
                )
                if size > self.max_message_bytes
                else self._handler(line)
            )
            self._output.write(response + "\n")
            self._output.flush()

    def _readline(self) -> str | None:
        """对真实 stdin 使用可中断 poll；StringIO 等测试流保持同步读取。

        直接在线程里阻塞 ``readline`` 会使 runtime 无法在 EOF 之前完成逆序关闭。Linux
        文件描述符先经 ``select`` 等待，close 设置 stop event 后最多一个 poll 周期即可退出；
        没有 ``fileno`` 的内存流不会阻塞，可以直接读取。
        """

        try:
            descriptor = self._input.fileno()
        except (AttributeError, OSError, ValueError):
            return self._input.readline()
        ready, _, _ = select.select((descriptor,), (), (), self.poll_interval_s)
        return self._input.readline() if ready else None

    def close(self, *, timeout_s: float | None = None) -> TransportCloseReport:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return TransportCloseReport(stopped=True)
        timeout = (
            self.shutdown_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        )
        thread.join(timeout)
        return TransportCloseReport(
            stopped=not thread.is_alive(),
            live_resources=(() if not thread.is_alive() else (thread.name,)),
        )


class TcpJsonlTransport:
    """loopback TCP JSONL server；每条连接内响应与请求严格同序。"""

    resource_name = "tcp_jsonl"

    def __init__(
        self,
        handler: JsonHandler,
        *,
        host: str,
        port: int,
        max_message_bytes: int = 1_048_576,
        max_connections: int = 16,
        startup_timeout_s: float = 5.0,
        shutdown_timeout_s: float = 2.0,
    ) -> None:
        self._handler = handler
        self.host = _loopback_host(host)
        if type(port) is not int or not 0 <= port <= 65_535:
            raise ValueError("TCP port must be in [0, 65535]")
        self.port = port
        self.max_message_bytes = _positive_int(
            max_message_bytes, label="max_message_bytes"
        )
        self.max_connections = _positive_int(max_connections, label="max_connections")
        self.startup_timeout_s = _positive_timeout(
            startup_timeout_s, label="TCP startup_timeout_s"
        )
        self.shutdown_timeout_s = _positive_timeout(
            shutdown_timeout_s, label="TCP shutdown_timeout_s"
        )
        self._stop = Event()
        self._ready = Event()
        self._listener: socket.socket | None = None
        self._thread: Thread | None = None
        self._connections: set[socket.socket] = set()
        self._workers: set[Thread] = set()
        self._lock = Lock()
        self._start_error: BaseException | None = None
        self._bound_port: int | None = None

    @property
    def bound_port(self) -> int:
        return self.port if self._bound_port is None else self._bound_port

    def start(self, *, timeout_s: float | None = None) -> None:
        if self._thread is not None:
            raise RuntimeError("TCP transport is already started")
        self._thread = Thread(
            target=self._serve,
            name="mirror-tcp-jsonl",
            daemon=True,
        )
        self._thread.start()
        timeout = (
            self.startup_timeout_s
            if timeout_s is None
            else _positive_timeout(timeout_s, label="TCP start timeout_s")
        )
        if not self._ready.wait(timeout):
            raise TimeoutError("TCP transport startup timed out")
        if self._start_error is not None:
            raise RuntimeError(
                f"TCP transport failed to start: {self._start_error}"
            ) from self._start_error

    def _serve(self) -> None:
        try:
            family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
            listener = socket.socket(family, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(self.max_connections)
            listener.settimeout(0.2)
            self._listener = listener
            self._bound_port = int(listener.getsockname()[1])
        except BaseException as exc:
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                try:
                    connection, _address = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                with self._lock:
                    active = len(self._connections)
                    if active >= self.max_connections:
                        self._reject_connection_capacity(connection)
                        continue
                    self._connections.add(connection)
                    worker = Thread(
                        target=self._serve_connection,
                        args=(connection,),
                        name=f"mirror-tcp-client-{active}",
                        daemon=True,
                    )
                    self._workers.add(worker)
                worker.start()
        finally:
            # close 可能在 listener 赋值前发生；由 owner thread 与 serve thread 双方幂等
            # close，确保启动 timeout 竞态也不会遗留监听 socket。
            listener.close()

    def _reject_connection_capacity(self, connection: socket.socket) -> None:
        """在关闭第 N+1 条连接前发送可诊断的协议失败。"""

        try:
            connection.settimeout(0.2)
            response = _transport_failure(
                code="connection_capacity_exceeded",
                message=f"TCP max_connections={self.max_connections} reached",
            )
            connection.sendall((response + "\n").encode("utf-8"))
        except OSError:
            pass
        finally:
            connection.close()

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(0.5)
            reader = connection.makefile("rb")
            writer = connection.makefile("wb")
            with reader, writer:
                while not self._stop.is_set():
                    try:
                        line = reader.readline(self.max_message_bytes + 2)
                    except TimeoutError:
                        continue
                    if line == b"":
                        return
                    frame = line[:-1] if line.endswith(b"\n") else line
                    if len(frame) > self.max_message_bytes:
                        # ``readline(limit)`` 可能停在超长记录中间；先丢弃本记录余部，下一次
                        # 循环才能从完整 JSONL frame 开始。
                        while line and not line.endswith(b"\n"):
                            line = reader.readline(self.max_message_bytes + 2)
                        response = _transport_failure(
                            code="message_too_large",
                            message=(
                                "TCP JSONL message exceeds max_message_bytes="
                                f"{self.max_message_bytes}"
                            ),
                        )
                    else:
                        try:
                            payload = line.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            response = _transport_failure(
                                code="invalid_encoding",
                                message=f"TCP JSONL must be UTF-8: {exc}",
                            )
                        else:
                            response = self._handler(payload)
                    writer.write((response + "\n").encode("utf-8"))
                    writer.flush()
        finally:
            with self._lock:
                self._connections.discard(connection)
                self._workers.discard(current_thread())
            connection.close()

    def close(self, *, timeout_s: float | None = None) -> TransportCloseReport:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            listener.close()
        with self._lock:
            connections = tuple(self._connections)
            workers = tuple(self._workers)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        timeout = (
            self.shutdown_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        )
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        for worker in workers:
            worker.join(timeout)
        live = [item.name for item in workers if item.is_alive()]
        if thread is not None and thread.is_alive():
            live.append(thread.name)
        return TransportCloseReport(stopped=not live, live_resources=tuple(live))


class WebSocketTransport:
    """使用可选 ``websockets.sync`` 的 loopback text-frame server。"""

    resource_name = "websocket"

    def __init__(
        self,
        handler: JsonHandler,
        *,
        host: str,
        port: int,
        max_message_bytes: int = 1_048_576,
        max_connections: int = 16,
        startup_timeout_s: float = 5.0,
        shutdown_timeout_s: float = 2.0,
    ) -> None:
        self._handler = handler
        self.host = _loopback_host(host)
        if type(port) is not int or not 0 <= port <= 65_535:
            raise ValueError("WebSocket port must be in [0, 65535]")
        self.port = port
        self.max_message_bytes = _positive_int(
            max_message_bytes, label="max_message_bytes"
        )
        self.max_connections = _positive_int(max_connections, label="max_connections")
        self.startup_timeout_s = _positive_timeout(
            startup_timeout_s, label="WebSocket startup_timeout_s"
        )
        self.shutdown_timeout_s = _positive_timeout(
            shutdown_timeout_s, label="WebSocket shutdown_timeout_s"
        )
        self._server: object | None = None
        self._thread: Thread | None = None
        self._ready = Event()
        self._stop = Event()
        self._connections = 0
        self._lock = Lock()
        self._start_error: BaseException | None = None

    @property
    def bound_port(self) -> int:
        """返回实际监听端口；port=0 的测试/embedded 调用也可发现地址。"""

        socket_object = getattr(self._server, "socket", None)
        if socket_object is None:
            return self.port
        return int(socket_object.getsockname()[1])

    def start(self, *, timeout_s: float | None = None) -> None:
        if self._thread is not None:
            raise RuntimeError("WebSocket transport is already started")
        self._thread = Thread(
            target=self._serve,
            name="mirror-websocket",
            daemon=True,
        )
        self._thread.start()
        timeout = (
            self.startup_timeout_s
            if timeout_s is None
            else _positive_timeout(timeout_s, label="WebSocket start timeout_s")
        )
        if not self._ready.wait(timeout):
            raise TimeoutError("WebSocket transport startup timed out")
        if self._start_error is not None:
            raise RuntimeError(
                f"WebSocket transport failed to start: {self._start_error}"
            ) from self._start_error

    def _serve(self) -> None:
        try:
            from websockets.sync.server import serve

            def websocket_handler(connection: object) -> None:
                with self._lock:
                    if self._connections >= self.max_connections:
                        response = _transport_failure(
                            code="connection_capacity_exceeded",
                            message=(
                                f"WebSocket max_connections={self.max_connections} reached"
                            ),
                        )
                        connection.send(response)
                        connection.close(
                            code=1013, reason="connection capacity exceeded"
                        )
                        return
                    self._connections += 1
                try:
                    for message in connection:  # type: ignore[operator]
                        if not isinstance(message, str):
                            connection.close(code=1003, reason="text frames only")
                            return
                        if len(message.encode("utf-8")) > self.max_message_bytes:
                            connection.send(
                                _transport_failure(
                                    code="message_too_large",
                                    message=(
                                        "WebSocket message exceeds max_message_bytes="
                                        f"{self.max_message_bytes}"
                                    ),
                                )
                            )
                            continue
                        connection.send(self._handler(message))
                finally:
                    with self._lock:
                        self._connections -= 1

            server = serve(
                websocket_handler,
                self.host,
                self.port,
                max_size=self.max_message_bytes,
            )
            self._server = server
            self._ready.set()
            # close 可能在可选依赖导入或 bind 尚未完成时先发生。赋值后再次检查 stop，
            # 避免 startup timeout 回滚漏掉刚创建的 server。
            if self._stop.is_set():
                server.shutdown()
            else:
                server.serve_forever()
        except BaseException as exc:
            self._start_error = exc
            self._ready.set()

    def close(self, *, timeout_s: float | None = None) -> TransportCloseReport:
        self._stop.set()
        shutdown = getattr(self._server, "shutdown", None)
        if callable(shutdown):
            shutdown()
        thread = self._thread
        if thread is not None:
            timeout = (
                self.shutdown_timeout_s
                if timeout_s is None
                else max(0.0, float(timeout_s))
            )
            thread.join(timeout)
        live = () if thread is None or not thread.is_alive() else (thread.name,)
        return TransportCloseReport(stopped=not live, live_resources=live)


@dataclass
class MirrorTransportHub:
    """三种 ingress 的共同 owner；部分启动失败时逆序回滚。"""

    endpoints: tuple[object, ...]
    _started: list[object] = field(default_factory=list, init=False, repr=False)

    def start(self) -> None:
        try:
            for endpoint in self.endpoints:
                start = getattr(endpoint, "start", None)
                if not callable(start):
                    raise TypeError(f"transport {type(endpoint).__name__} is missing start")
                # 先登记再启动：start 在创建部分线程/socket 后抛错时，rollback 仍能关闭它。
                self._started.append(endpoint)
                start()
        except BaseException:
            self.close()
            raise

    def close(self) -> TransportCloseReport:
        live: list[str] = []
        for endpoint in reversed(tuple(self._started)):
            close = getattr(endpoint, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
            except BaseException as exc:
                # 一个 endpoint 的 close 失败不能阻止更早启动的 endpoint 释放。保留它在
                # _started 中供 MirrorRuntime 下一次 close 重试，并把异常写入资源诊断。
                live.append(f"{type(endpoint).__name__}: {type(exc).__name__}: {exc}")
                continue
            stopped = close_result_stopped(result)
            resources = tuple(getattr(result, "live_resources", ()))
            if stopped:
                self._started = [item for item in self._started if item is not endpoint]
            else:
                live.extend(resources or (type(endpoint).__name__,))
        return TransportCloseReport(stopped=not live, live_resources=tuple(live))


__all__ = [
    "MirrorTransportHub",
    "StdinJsonlTransport",
    "TcpJsonlTransport",
    "TransportCloseReport",
    "WebSocketTransport",
    "make_json_handler",
]
