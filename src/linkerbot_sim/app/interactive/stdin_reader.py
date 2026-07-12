"""可中断且有消息长度上限的进程标准输入 JSONL reader。

生产环境的 stdin 通常暴露文件描述符。本模块复制该描述符，并在工作线程中同时等待输入
和私有唤醒管道；关闭方因此可以唤醒并回收线程，而无需关闭属于整个进程的
``sys.stdin``。测试使用的有限内存流没有可轮询的描述符，此时改用有界同步读取，但仍
复用同一个增量 framer，保证三条输入路径具有一致的换行、UTF-8 和长度边界。

工作线程拥有复制出来的 fd 和唤醒管道，调用方仍拥有原始 stream。frame/EOF 回调也在
工作线程执行，因此回调只能把消息交给线程安全队列，不能直接操作仿真 runtime。
"""

from __future__ import annotations

import os
import selectors
from collections.abc import Callable
from dataclasses import dataclass
from io import BufferedIOBase, RawIOBase
from threading import Event, Lock, Thread
from typing import BinaryIO, TextIO, cast


@dataclass(frozen=True)
class StdinJsonlFrame:
    """一条完成 framing 的 stdin 记录，携带 UTF-8 文本或互斥的 framing 错误。

    ``oversized`` 表示整条记录已被有界丢弃，``decode_error`` 表示长度合法但不是有效
    UTF-8；这两类错误都不把不完整文本暴露给协议解析器。
    """

    text: str | None
    oversized: bool = False
    decode_error: UnicodeDecodeError | None = None


class InterruptibleStdinJsonlReader:
    """拥有 stdin 描述符副本、唤醒管道和唯一 reader 线程的生命周期句柄。

    构造完成即启动线程。``stop`` 可从其他线程调用，并负责发出停止信号和 join；对象
    不会关闭传入的原始 stream。若回调自身阻塞，join 可能超时，此时返回 ``False``，
    fd 仍由尚未退出的工作线程持有，不能被关闭方抢先回收。
    """

    def __init__(
        self,
        *,
        stream: object,
        max_message_bytes: int,
        on_frame: Callable[[StdinJsonlFrame], None],
        on_eof: Callable[[], None],
        thread_name: str,
        shutdown_event: Event | None = None,
    ) -> None:
        """取得输入资源并立即启动 reader 线程。

        ``shutdown_event`` 可与更大的 transport 生命周期共享；省略时由本实例私有持有。
        构造中任何 fd/pipe/thread 启动失败都会在异常向外传播前回收已经取得的资源。
        """

        if (
            isinstance(max_message_bytes, bool)
            or not isinstance(max_message_bytes, int)
            or max_message_bytes < 1
        ):
            raise ValueError("max_message_bytes must be a positive integer")
        self.max_message_bytes = max_message_bytes
        self._stream = stream
        self._on_frame = on_frame
        self._on_eof = on_eof
        self._stop_event = shutdown_event or Event()
        self._resource_lock = Lock()
        self._input_fd: int | None = None
        self._wakeup_read_fd: int | None = None
        self._wakeup_write_fd: int | None = None
        self._binary_fallback: object | None = None

        source = getattr(stream, "buffer", stream)
        try:
            source_fd = int(source.fileno())
        except (AttributeError, OSError, TypeError, ValueError):
            # Text wrappers expose a binary ``buffer``.  ``BytesIO`` has no
            # wrapper, so its identity alone cannot distinguish it from
            # ``StringIO``; the standard binary base classes do.
            if source is not stream or isinstance(source, (BufferedIOBase, RawIOBase)):
                self._binary_fallback = source
        else:
            self._input_fd = os.dup(source_fd)
            try:
                self._wakeup_read_fd, self._wakeup_write_fd = os.pipe()
            except BaseException:
                os.close(self._input_fd)
                self._input_fd = None
                raise

        self.thread = Thread(
            target=self._run,
            daemon=True,
            name=thread_name,
        )
        try:
            self.thread.start()
        except BaseException:
            self._close_owned_fds()
            raise

    @property
    def name(self) -> str:
        """返回所拥有工作线程的诊断名称。"""

        return self.thread.name

    def is_alive(self) -> bool:
        """返回所拥有的 reader 线程是否仍在运行。"""

        return self.thread.is_alive()

    def stop(self, *, timeout_s: float | None = None) -> bool:
        """唤醒并 join 工作线程；回调仍未退出时返回 ``False``。

        停止事件先于管道写入设置，确保输入与停止同时就绪时停止优先。只有确认线程退出
        后，调用线程才补充关闭残留 fd；正常退出路径则由工作线程的 ``finally`` 关闭。
        """

        if timeout_s is not None and timeout_s < 0.0:
            raise ValueError("stdin shutdown timeout_s must be >= 0")
        self._stop_event.set()
        with self._resource_lock:
            wakeup_write_fd = self._wakeup_write_fd
        if wakeup_write_fd is not None:
            try:
                os.write(wakeup_write_fd, b"\0")
            except OSError:
                # 自然 EOF 可能已让工作线程并发退出并关闭管道；此时停止目标已经达成。
                pass
        self.thread.join(timeout=timeout_s)
        stopped = not self.thread.is_alive()
        if stopped:
            self._close_owned_fds()
        return stopped

    def _run(self) -> None:
        try:
            if self._input_fd is not None:
                self._run_fd()
            elif self._binary_fallback is not None:
                self._run_binary_fallback(self._binary_fallback)
            else:
                self._run_text_fallback(self._stream)
        finally:
            self._close_owned_fds()

    def _run_fd(self) -> None:
        """等待输入或停止信号，并在两者同时就绪时让停止取得优先级。"""

        assert self._input_fd is not None
        assert self._wakeup_read_fd is not None
        framer = _BoundedJsonlFramer(self.max_message_bytes)
        # ``SelectSelector`` 也接受普通文件；CLI 从重定向文件而非终端/管道读取时仍能
        # 使用同一套可中断路径。
        with selectors.SelectSelector() as selector:
            selector.register(self._input_fd, selectors.EVENT_READ, "stdin")
            selector.register(self._wakeup_read_fd, selectors.EVENT_READ, "stop")
            while True:
                ready = selector.select()
                if self._stop_event.is_set() or any(
                    key.data == "stop" for key, _events in ready
                ):
                    return
                for key, _events in ready:
                    if key.data != "stdin":
                        continue
                    payload = os.read(self._input_fd, 65_536)
                    if not payload:
                        self._emit_frames(framer.finish())
                        if not self._stop_event.is_set():
                            self._on_eof()
                        return
                    self._emit_frames(framer.feed(payload))
                    if self._stop_event.is_set():
                        return

    def _run_binary_fallback(self, stream: object) -> None:
        """读取不暴露 ``fileno`` 的有限二进制 file-like stream。"""

        framer = _BoundedJsonlFramer(self.max_message_bytes)
        readline = getattr(stream, "readline")
        while not self._stop_event.is_set():
            chunk = readline(self.max_message_bytes + 3)
            if chunk == b"":
                self._emit_frames(framer.finish())
                if not self._stop_event.is_set():
                    self._on_eof()
                return
            self._emit_frames(framer.feed(bytes(chunk)))

    def _run_text_fallback(self, stream: object) -> None:
        """以有界 chunk 读取 ``StringIO`` 等有限文本 stream。"""

        framer = _BoundedJsonlFramer(self.max_message_bytes)
        readline = getattr(stream, "readline")
        while not self._stop_event.is_set():
            chunk = readline(self.max_message_bytes + 3)
            if chunk == "":
                self._emit_frames(framer.finish())
                if not self._stop_event.is_set():
                    self._on_eof()
                return
            framer_input = cast(str, chunk).encode("utf-8")
            self._emit_frames(framer.feed(framer_input))

    def _emit_frames(self, frames: tuple[StdinJsonlFrame, ...]) -> None:
        for frame in frames:
            if self._stop_event.is_set():
                return
            self._on_frame(frame)

    def _close_owned_fds(self) -> None:
        with self._resource_lock:
            fds = (
                self._input_fd,
                self._wakeup_read_fd,
                self._wakeup_write_fd,
            )
            self._input_fd = None
            self._wakeup_read_fd = None
            self._wakeup_write_fd = None
        for fd in fds:
            if fd is None:
                continue
            try:
                os.close(fd)
            except OSError:
                pass


class _BoundedJsonlFramer:
    """增量切分 JSONL，内存中最多保留上限加一个字节。

    多出的一个字节只用于判定一条记录确实越界。一旦越界，framer 进入 discard 状态，
    在下一个换行前不再保留该记录的任何后续字节；这样攻击者无法通过无换行输入让内存
    随消息长度增长。换行到达后状态复位，下一条记录仍可正常解析。
    """

    def __init__(self, max_message_bytes: int) -> None:
        self.max_message_bytes = max_message_bytes
        self._payload = bytearray()
        self._discarding_oversized = False

    def feed(self, data: bytes) -> tuple[StdinJsonlFrame, ...]:
        """接收任意字节片段，并返回其中所有已遇到换行边界的完整 frame。"""

        frames: list[StdinJsonlFrame] = []
        cursor = 0
        while cursor < len(data):
            newline = data.find(b"\n", cursor)
            if newline < 0:
                self._append(data[cursor:])
                break
            self._append(data[cursor:newline])
            frames.append(self._finish_line())
            cursor = newline + 1
        return tuple(frames)

    def finish(self) -> tuple[StdinJsonlFrame, ...]:
        """在 EOF 时提交最后一条无换行记录；没有残留数据则不产生 frame。"""

        if not self._payload and not self._discarding_oversized:
            return ()
        return (self._finish_line(),)

    def _append(self, fragment: bytes) -> None:
        if self._discarding_oversized:
            return
        remaining = self.max_message_bytes + 1 - len(self._payload)
        if len(fragment) > remaining:
            self._payload.clear()
            self._discarding_oversized = True
            return
        self._payload.extend(fragment)

    def _finish_line(self) -> StdinJsonlFrame:
        oversized = self._discarding_oversized
        payload = bytes(self._payload)
        if payload.endswith(b"\r"):
            payload = payload[:-1]
        if len(payload) > self.max_message_bytes:
            oversized = True
        self._payload.clear()
        self._discarding_oversized = False
        if oversized:
            return StdinJsonlFrame(text=None, oversized=True)
        try:
            return StdinJsonlFrame(text=payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            return StdinJsonlFrame(text=None, decode_error=exc)


def start_interruptible_stdin_jsonl_reader(
    *,
    stream: BinaryIO | TextIO | object,
    max_message_bytes: int,
    on_frame: Callable[[StdinJsonlFrame], None],
    on_eof: Callable[[], None],
    thread_name: str,
    shutdown_event: Event | None = None,
) -> InterruptibleStdinJsonlReader:
    """启动并返回拥有独立资源的可中断 stdin JSONL reader。

    ``on_frame`` 与 ``on_eof`` 均在新建线程中调用；调用方应在停止整个 transport 时保存
    返回句柄并调用 ``stop``，以便有界等待该线程和它拥有的 fd。
    """

    return InterruptibleStdinJsonlReader(
        stream=stream,
        max_message_bytes=max_message_bytes,
        on_frame=on_frame,
        on_eof=on_eof,
        thread_name=thread_name,
        shutdown_event=shutdown_event,
    )


__all__ = [
    "InterruptibleStdinJsonlReader",
    "StdinJsonlFrame",
    "start_interruptible_stdin_jsonl_reader",
]
