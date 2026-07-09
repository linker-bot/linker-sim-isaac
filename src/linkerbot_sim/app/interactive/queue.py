"""Thread-safe queue for interactive motion commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Condition, Event, Lock
from typing import Literal

from linkerbot_sim.app.motion.specs import MoveSpec
from linkerbot_sim.app.interactive.protocol import InteractiveMotionCommand


CommandState = Literal["pending", "running", "done", "failed", "cancelled"]
SnapshotRequestKind = Literal["get_snapshot", "set_snapshot"]


@dataclass(frozen=True)
class ResetRequest:
    """One requested simulation reset operation."""

    reset_id: str
    mode: str = "runtime"
    clear_queue: bool = True
    hold_after_reset: bool = True

    def snapshot(self) -> dict[str, object]:
        """返回可序列化 reset 请求。"""

        return {
            "id": self.reset_id,
            "mode": self.mode,
            "clear_queue": self.clear_queue,
            "hold_after_reset": self.hold_after_reset,
        }


@dataclass
class SnapshotRequest:
    """一次待仿真主线程处理的 snapshot 请求。

    transport 线程只负责解析 JSON 和等待结果，不能直接读写 Isaac/PhysX 状态；所以
    get/set snapshot 都会被封装成这个 request，由 single/dual 的主仿真循环取出执行。
    """

    snapshot_id: str
    kind: SnapshotRequestKind
    snapshot: Mapping[str, object] | None = None
    robot_map: Mapping[str, str] | None = None
    strict: bool = True
    response: dict[str, object] | None = None
    done: bool = False

    def snapshot_info(self) -> dict[str, object]:
        """返回可序列化 snapshot 请求摘要，用于事件广播。"""

        return {
            "id": self.snapshot_id,
            "kind": self.kind,
            "strict": bool(self.strict),
        }


@dataclass
class QueuedMotionCommand:
    """One queued motion command and its execution state."""

    command_id: str
    moves: tuple[MoveSpec, ...]
    duration_s: float | None = None
    state: CommandState = "pending"
    error: str | None = None
    steps: int | None = None

    def snapshot(self) -> dict[str, object]:
        """返回可序列化的命令状态快照，供 status 响应和 WebSocket 事件使用。"""

        return {
            "id": self.command_id,
            "state": self.state,
            "duration_s": self.duration_s,
            "error": self.error,
            "steps": self.steps,
        }


class InteractiveMotionQueue:
    """Queue plus command-state registry for interactive execution."""

    def __init__(self) -> None:
        """初始化命令注册表、执行状态和跨线程通知原语。"""

        self._condition = Condition()
        self._lock = Lock()
        self._next_id = 1
        self._commands: dict[str, QueuedMotionCommand] = {}
        self._order: list[str] = []
        self._current_id: str | None = None
        self._next_reset_id = 1
        self._reset_request: ResetRequest | None = None
        self._resetting = False
        self._last_reset: dict[str, object] | None = None
        self._next_snapshot_id = 1
        self._snapshot_requests: list[SnapshotRequest] = []
        self._cancel_current = Event()
        self._estop = Event()
        self._quit = Event()
        self._listeners: list[Callable[[dict[str, object]], None]] = []

    def add_listener(self, listener: Callable[[dict[str, object]], None]) -> None:
        """注册事件监听器，通常用于 WebSocket 侧推送队列状态变化。"""

        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[dict[str, object]], None]) -> None:
        """移除之前注册的事件监听器；使用身份比较避免误删同形函数。"""

        with self._lock:
            self._listeners = [
                existing for existing in self._listeners if existing is not listener
            ]

    def submit(self, command: InteractiveMotionCommand) -> QueuedMotionCommand:
        """把 moves/hold 命令放入待执行队列，并发出 accepted 事件。"""

        if command.kind not in {"moves", "hold"}:
            raise ValueError(f"cannot submit non-move command {command.kind!r}")
        command_id = command.command_id or self._new_command_id()
        with self._condition:
            if command_id in self._commands:
                raise ValueError(f"duplicate command id: {command_id}")
            queued = QueuedMotionCommand(
                command_id=command_id,
                moves=tuple(command.moves),
                duration_s=command.duration_s,
            )
            self._commands[command_id] = queued
            self._order.append(command_id)
            queue_index = self._pending_index_locked(command_id)
            self._condition.notify_all()
        self.emit(
            {
                "event": "accepted",
                "id": command_id,
                "state": "pending",
                "queue_index": queue_index,
            }
        )
        return queued

    def next_pending(self, *, timeout_s: float | None = None) -> QueuedMotionCommand | None:
        """阻塞等待下一个 pending 命令，并原子地把它切换为 running。"""

        with self._condition:
            if not self._condition.wait_for(
                lambda: (
                    self._quit.is_set()
                    or self._reset_request is not None
                    or self._next_pending_locked() is not None
                ),
                timeout=timeout_s,
            ):
                return None
            if self._quit.is_set() or self._reset_request is not None:
                return None
            queued = self._next_pending_locked()
            if queued is None:
                return None
            queued.state = "running"
            self._current_id = queued.command_id
            self._cancel_current.clear()
        self.emit({"event": "running", "id": queued.command_id, "state": "running"})
        return queued

    def mark_done(self, command_id: str, *, steps: int | None = None) -> None:
        """把命令标记为成功完成，并记录执行结束时的仿真步数。"""

        self._mark_terminal(command_id, "done", steps=steps)

    def mark_failed(self, command_id: str, error: str) -> None:
        """把命令标记为失败，并把异常文本暴露给交互客户端。"""

        self._mark_terminal(command_id, "failed", error=error)

    def mark_cancelled(
        self,
        command_id: str,
        error: str | None = None,
        *,
        steps: int | None = None,
    ) -> None:
        """把命令标记为取消；running 命令通常由执行循环捕获中断后调用。"""

        self._mark_terminal(command_id, "cancelled", error=error, steps=steps)

    def cancel(self, command_id: str) -> bool:
        """取消指定命令；pending 直接终止，running 通过 stop flag 请求执行循环中断。"""

        with self._condition:
            queued = self._commands.get(command_id)
            if queued is None:
                return False
            if queued.state == "pending":
                queued.state = "cancelled"
                self._condition.notify_all()
                should_emit = True
            elif queued.state == "running":
                self._cancel_current.set()
                should_emit = False
            else:
                return False
        if should_emit:
            self.emit({"event": "cancelled", "id": command_id, "state": "cancelled"})
        return True

    def request_reset(
        self,
        *,
        reset_id: str | None = None,
        mode: str = "runtime",
        clear_queue: bool = True,
        hold_after_reset: bool = True,
    ) -> ResetRequest:
        """请求主循环在安全边界执行 reset。"""

        if mode != "runtime":
            raise ValueError("reset mode must be 'runtime'")
        request = ResetRequest(
            reset_id=reset_id or self._new_reset_id(),
            mode=mode,
            clear_queue=bool(clear_queue),
            hold_after_reset=bool(hold_after_reset),
        )
        cancelled_ids: list[str] = []
        with self._condition:
            self._reset_request = request
            self._resetting = True
            self._last_reset = {
                "id": request.reset_id,
                "state": "requested",
                "mode": request.mode,
            }
            if request.clear_queue:
                for command in self._commands.values():
                    if command.state == "pending":
                        command.state = "cancelled"
                        cancelled_ids.append(command.command_id)
            if self._current_id is not None:
                self._cancel_current.set()
            self._condition.notify_all()
        for command_id in cancelled_ids:
            self.emit({"event": "cancelled", "id": command_id, "state": "cancelled"})
        self.emit({"event": "reset_requested", **request.snapshot()})
        return request

    def request_snapshot(
        self,
        *,
        kind: SnapshotRequestKind,
        snapshot_id: str | None = None,
        snapshot: Mapping[str, object] | None = None,
        robot_map: Mapping[str, str] | None = None,
        strict: bool = True,
        timeout_s: float = 30.0,
    ) -> dict[str, object]:
        """请求仿真主线程执行 snapshot get/set，并阻塞等待响应。

        这里使用 ``Condition`` 把 transport 线程和仿真主循环串起来：transport 创建请求后
        等待 ``mark_snapshot_done/failed`` 唤醒；主循环在安全的 physics 边界消费请求。
        """

        if kind == "set_snapshot" and snapshot is None:
            raise ValueError("set_snapshot requires snapshot")
        request = SnapshotRequest(
            snapshot_id=snapshot_id or self._new_snapshot_id(),
            kind=kind,
            snapshot=snapshot,
            robot_map=robot_map,
            strict=bool(strict),
        )
        with self._condition:
            # snapshot 请求单独排队，不与 motion command 共用 pending 列表；这样它不会被
            # 长时间动作队列阻塞，也不会改变 motion command 的状态机。
            self._snapshot_requests.append(request)
            self._condition.notify_all()
        self.emit({"event": "snapshot_requested", **request.snapshot_info()})
        with self._condition:
            completed = self._condition.wait_for(
                lambda: request.done or self._quit.is_set(),
                timeout=float(timeout_s),
            )
            if not completed or request.response is None:
                return {
                    "event": "snapshot_timeout",
                    "accepted": False,
                    "id": request.snapshot_id,
                }
            return dict(request.response)

    def consume_snapshot_request(self) -> SnapshotRequest | None:
        """取出 pending snapshot 请求；只有仿真主循环应调用。

        返回后请求对象仍由等待中的 transport 持有引用，所以主循环只需要在同一个对象上
        写入 response/done 并 notify 即可。
        """

        with self._condition:
            if not self._snapshot_requests:
                return None
            return self._snapshot_requests.pop(0)

    def mark_snapshot_done(
        self,
        request: SnapshotRequest,
        response: Mapping[str, object],
    ) -> None:
        """标记 snapshot 请求成功完成并唤醒等待的 transport。"""

        payload = dict(response)
        payload.setdefault("id", request.snapshot_id)
        with self._condition:
            # response 存在 request 对象上，等待方醒来后直接返回这份 JSON-compatible dict。
            request.response = payload
            request.done = True
            self._condition.notify_all()
        self.emit({"event": "snapshot_done", "id": request.snapshot_id})

    def mark_snapshot_failed(self, request: SnapshotRequest, error: str) -> None:
        """标记 snapshot 请求失败，并用统一 rejected-like payload 唤醒等待方。"""

        response = {
            "event": "snapshot_failed",
            "accepted": False,
            "id": request.snapshot_id,
            "error": error,
        }
        with self._condition:
            request.response = response
            request.done = True
            self._condition.notify_all()
        self.emit(response)

    def consume_reset_request(self) -> ResetRequest | None:
        """取出 pending reset 请求；只有主循环应调用。"""

        with self._condition:
            request = self._reset_request
            self._reset_request = None
            return request

    def mark_reset_done(self, reset_id: str, *, step: int | None = None) -> None:
        """标记 reset 成功完成。"""

        with self._condition:
            self._resetting = False
            self._last_reset = {
                "id": reset_id,
                "state": "done",
                "step": step,
            }
            self._cancel_current.clear()
            self._condition.notify_all()
        event: dict[str, object] = {
            "event": "reset_done",
            "id": reset_id,
            "state": "done",
        }
        if step is not None:
            event["step"] = step
        self.emit(event)

    def mark_reset_failed(self, reset_id: str, error: str) -> None:
        """标记 reset 失败。"""

        with self._condition:
            self._resetting = False
            self._last_reset = {
                "id": reset_id,
                "state": "failed",
                "error": error,
            }
            self._condition.notify_all()
        self.emit(
            {
                "event": "reset_failed",
                "id": reset_id,
                "state": "failed",
                "error": error,
            }
        )

    def request_cancel_current(self) -> bool:
        """请求中断当前 running 命令；没有当前命令时返回 False。"""

        with self._condition:
            if self._current_id is None:
                return False
            self._cancel_current.set()
            return True

    def request_estop(self) -> None:
        """触发急停：取消所有 pending 命令，并要求当前命令尽快停止。"""

        self._estop.set()
        self._cancel_current.set()
        with self._condition:
            for command in self._commands.values():
                if command.state == "pending":
                    command.state = "cancelled"
            self._condition.notify_all()
        self.emit({"event": "estop", "state": "cancelled"})

    def request_quit(self) -> None:
        """请求交互循环退出，并唤醒可能正在等待队列的执行线程。"""

        self._quit.set()
        with self._condition:
            self._condition.notify_all()

    def should_stop_current(self) -> bool:
        """返回当前执行命令是否应停止；执行 stepper 会周期性轮询该函数。"""

        return self._cancel_current.is_set() or self._estop.is_set() or self._quit.is_set()

    def estop_requested(self) -> bool:
        """返回是否已收到急停请求。"""

        return self._estop.is_set()

    def quit_requested(self) -> bool:
        """返回是否已收到退出请求。"""

        return self._quit.is_set()

    def status(self, command_id: str | None = None) -> dict[str, object]:
        """生成 status 响应；不传 id 时返回整个队列和当前急停状态。"""

        with self._condition:
            if command_id is not None:
                command = self._commands.get(command_id)
                return {
                    "event": "status",
                    "commands": [] if command is None else [command.snapshot()],
                }
            return {
                "event": "status",
                "commands": [
                    self._commands[command_id].snapshot()
                    for command_id in self._order
                ],
                "current_id": self._current_id,
                "estop": self._estop.is_set(),
                "resetting": self._resetting,
                "last_reset": self._last_reset,
            }

    def pending_index(self, command_id: str) -> int | None:
        """返回命令在 pending 子队列中的位置；非 pending 或不存在时返回 None。"""

        with self._condition:
            return self._pending_index_locked(command_id)

    def _pending_index_locked(self, command_id: str) -> int | None:
        """在持有 condition 锁时计算 pending 队列索引。"""

        pending = [
            queued.command_id
            for queued in self._commands.values()
            if queued.state == "pending"
        ]
        try:
            return pending.index(command_id)
        except ValueError:
            return None

    def emit(self, event: dict[str, object]) -> None:
        """向所有监听器广播事件；复制事件字典，避免监听器修改共享对象。"""

        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(dict(event))

    def _mark_terminal(
        self,
        command_id: str,
        state: CommandState,
        *,
        error: str | None = None,
        steps: int | None = None,
    ) -> None:
        """统一处理 done/failed/cancelled 三种终态切换和事件广播。"""

        with self._condition:
            queued = self._commands.get(command_id)
            if queued is None:
                return
            queued.state = state
            queued.error = error
            queued.steps = steps
            if self._current_id == command_id:
                self._current_id = None
                self._cancel_current.clear()
            self._condition.notify_all()
        event = {"event": state, "id": command_id, "state": state}
        if error is not None:
            event["error"] = error
        if steps is not None:
            event["steps"] = steps
        self.emit(event)

    def _next_pending_locked(self) -> QueuedMotionCommand | None:
        """在持有 condition 锁时按提交顺序找到下一个 pending 命令。"""

        for command_id in self._order:
            queued = self._commands[command_id]
            if queued.state == "pending":
                return queued
        return None

    def _new_command_id(self) -> str:
        """生成进程内递增的默认命令 id。"""

        with self._lock:
            command_id = f"cmd-{self._next_id}"
            self._next_id += 1
            return command_id

    def _new_reset_id(self) -> str:
        """生成进程内递增的 reset id。"""

        with self._lock:
            reset_id = f"reset-{self._next_reset_id}"
            self._next_reset_id += 1
            return reset_id

    def _new_snapshot_id(self) -> str:
        """生成进程内递增的 snapshot 请求 id。"""

        with self._lock:
            snapshot_id = f"snapshot-{self._next_snapshot_id}"
            self._next_snapshot_id += 1
            return snapshot_id
