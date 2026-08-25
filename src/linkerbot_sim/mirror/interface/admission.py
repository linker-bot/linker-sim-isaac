"""三种 ingress 共用的有界 admission/response 队列。"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Condition
import time

from .protocol import MirrorRequest, MirrorResponse


class MirrorAdmissionError(RuntimeError):
    code = "admission_error"


class DuplicateRequestError(MirrorAdmissionError):
    code = "duplicate_request_id"


class AdmissionCapacityError(MirrorAdmissionError):
    code = "queue_capacity_exceeded"


class AdmissionClosedError(MirrorAdmissionError):
    code = "admission_closed"


class RuntimeEstoppedError(MirrorAdmissionError):
    code = "runtime_estopped"


@dataclass(frozen=True)
class AdmissionStatus:
    pending: int
    active_request_id: str | None
    terminal: int
    capacity: int
    closed: bool
    estopped: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "pending": self.pending,
            "active_request_id": self.active_request_id,
            "terminal": self.terminal,
            "capacity": self.capacity,
            "closed": self.closed,
            "estopped": self.estopped,
        }


class MirrorAdmissionQueue:
    """有界、单主线程消费、并发 ingress 可安全提交的队列。

    相同 request_id 在 pending/active/terminal 三种状态下都拒绝。terminal history 至少与
    admission capacity 同大，保证已接收请求的响应不会在等待方读取前立即被逐出。
    """

    def __init__(self, *, capacity: int = 256, terminal_capacity: int = 512) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(terminal_capacity) is not int or terminal_capacity < capacity:
            raise ValueError("terminal_capacity must be an integer >= capacity")
        self.capacity = capacity
        self.terminal_capacity = terminal_capacity
        self._condition = Condition()
        self._pending: deque[MirrorRequest] = deque()
        self._active: MirrorRequest | None = None
        self._terminal: OrderedDict[str, MirrorResponse] = OrderedDict()
        self._reserved_immediate: set[str] = set()
        self._cancelled: set[str] = set()
        self._closed = False
        self._estopped = False

    def submit(self, request: MirrorRequest) -> None:
        with self._condition:
            if self._closed:
                raise AdmissionClosedError("Mirror admission is closed")
            if self._estopped and (
                request.operation.startswith("motion.")
                or request.operation
                in {
                    "state.set",
                    "control.set_mode",
                    "control.set_hybrid_parameters",
                    "control.tare_wrench",
                }
            ):
                raise RuntimeEstoppedError(
                    "Mirror is estopped; new motion, state.set and control.set_mode are rejected until reset"
                )
            if self._known(request.request_id):
                raise DuplicateRequestError(
                    f"duplicate request_id: {request.request_id!r}"
                )
            if len(self._pending) + int(self._active is not None) >= self.capacity:
                raise AdmissionCapacityError("Mirror admission queue is full")
            self._pending.append(request)
            self._condition.notify_all()

    def reserve_immediate(self, request_id: str) -> None:
        """为线程安全控制操作原子保留 ID，保持与普通请求相同的去重语义。"""

        with self._condition:
            if self._closed:
                raise AdmissionClosedError("Mirror admission is closed")
            if self._known(request_id):
                raise DuplicateRequestError(f"duplicate request_id: {request_id!r}")
            self._reserved_immediate.add(request_id)

    def complete_immediate(self, response: MirrorResponse) -> None:
        with self._condition:
            if response.request_id not in self._reserved_immediate:
                raise RuntimeError("immediate response has no matching reservation")
            self._reserved_immediate.remove(response.request_id)
            self._store_terminal(response)
            self._condition.notify_all()

    def claim(self, *, timeout_s: float | None = None) -> MirrorRequest | None:
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        with self._condition:
            while not self._pending and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if self._active is not None:
                raise RuntimeError(
                    "Mirror admission only allows a single main-thread active request"
                )
            if not self._pending:
                return None
            self._active = self._pending.popleft()
            return self._active

    def complete(self, response: MirrorResponse) -> None:
        with self._condition:
            if self._active is None or self._active.request_id != response.request_id:
                raise RuntimeError(
                    "response does not belong to the current active request"
                )
            self._active = None
            self._cancelled.discard(response.request_id)
            self._store_terminal(response)
            self._condition.notify_all()

    def wait_response(self, request_id: str, *, timeout_s: float) -> MirrorResponse:
        """等待 terminal response，并在 deadline 到期时原子终止 admission。

        pending 请求会在持有 condition 锁时从队列移除，因此 owner thread 不可能在
        timeout 返回后再 claim 它。已经被 owner claim 的 active 请求不能在 ingress
        线程强制回滚；这里只设置 cooperative-cancel，执行器会在下一个取消边界停止。
        """

        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while request_id not in self._terminal:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._expire_response_wait(request_id)
                    raise TimeoutError(
                        f"timed out waiting for Mirror response: {request_id!r}"
                    )
                self._condition.wait(remaining)
            return self._terminal[request_id]

    def _expire_response_wait(self, request_id: str) -> None:
        """在 condition 锁内终止 pending 请求或取消 active 请求。"""

        if self._active is not None and self._active.request_id == request_id:
            self._cancelled.add(request_id)
            self._condition.notify_all()
            return
        for request in tuple(self._pending):
            if request.request_id != request_id:
                continue
            self._pending.remove(request)
            self._store_terminal(
                MirrorResponse.failure(
                    request_id,
                    code="response_timeout",
                    message=f"timed out waiting for Mirror response: {request_id!r}",
                    protocol=request.protocol,
                )
            )
            self._condition.notify_all()
            return

    def cancel(self, request_id: str) -> bool:
        with self._condition:
            if self._active is not None and self._active.request_id == request_id:
                self._cancelled.add(request_id)
                return True
            for request in tuple(self._pending):
                if request.request_id != request_id:
                    continue
                self._pending.remove(request)
                self._store_terminal(
                    MirrorResponse.failure(
                        request_id,
                        code="cancelled",
                        message="the request was cancelled before execution",
                        protocol=request.protocol,
                    )
                )
                self._condition.notify_all()
                return True
            return False

    def cancel_current(self) -> bool:
        with self._condition:
            if self._active is None:
                return False
            self._cancelled.add(self._active.request_id)
            return True

    def cancel_pending(self, *, code: str, message: str) -> tuple[str, ...]:
        """清空尚未执行的请求，但不改变 active/estop 状态。"""

        with self._condition:
            cancelled: list[str] = []
            while self._pending:
                request = self._pending.popleft()
                cancelled.append(request.request_id)
                self._store_terminal(
                    MirrorResponse.failure(
                        request.request_id,
                        code=code,
                        message=message,
                        protocol=request.protocol,
                    )
                )
            self._condition.notify_all()
            return tuple(cancelled)

    def should_cancel(self, request_id: str) -> bool:
        with self._condition:
            return request_id in self._cancelled or self._estopped or self._closed

    def estop(self) -> tuple[str, ...]:
        with self._condition:
            self._estopped = True
            cancelled = []
            while self._pending:
                request = self._pending.popleft()
                cancelled.append(request.request_id)
                self._store_terminal(
                    MirrorResponse.failure(
                        request.request_id,
                        code="estopped",
                        message="the request was cleared by runtime.estop",
                        protocol=request.protocol,
                    )
                )
            if self._active is not None:
                self._cancelled.add(self._active.request_id)
                cancelled.append(self._active.request_id)
            self._condition.notify_all()
            return tuple(cancelled)

    def clear_estop(self) -> None:
        with self._condition:
            self._estopped = False

    def status(self) -> AdmissionStatus:
        with self._condition:
            return AdmissionStatus(
                pending=len(self._pending),
                active_request_id=(
                    None if self._active is None else self._active.request_id
                ),
                terminal=len(self._terminal),
                capacity=self.capacity,
                closed=self._closed,
                estopped=self._estopped,
            )

    def close(self) -> bool:
        with self._condition:
            if self._closed:
                return True
            self._closed = True
            while self._pending:
                request = self._pending.popleft()
                self._store_terminal(
                    MirrorResponse.failure(
                        request.request_id,
                        code="runtime_closing",
                        message="Mirror is shutting down",
                        protocol=request.protocol,
                    )
                )
            if self._active is not None:
                self._cancelled.add(self._active.request_id)
            self._condition.notify_all()
            return self._active is None

    def _known(self, request_id: str) -> bool:
        return bool(
            request_id in self._terminal
            or request_id in self._reserved_immediate
            or (self._active is not None and self._active.request_id == request_id)
            or any(item.request_id == request_id for item in self._pending)
        )

    def _store_terminal(self, response: MirrorResponse) -> None:
        self._terminal[response.request_id] = response
        self._terminal.move_to_end(response.request_id)
        while len(self._terminal) > self.terminal_capacity:
            self._terminal.popitem(last=False)


__all__ = [
    "AdmissionCapacityError",
    "AdmissionClosedError",
    "AdmissionStatus",
    "DuplicateRequestError",
    "MirrorAdmissionError",
    "MirrorAdmissionQueue",
    "RuntimeEstoppedError",
]
