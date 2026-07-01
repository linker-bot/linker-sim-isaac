"""Thread-safe queue for interactive motion commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Event, Lock
from typing import Literal

from linkerbot_sim.app.motion.specs import MoveSpec
from linkerbot_sim.app.interactive.protocol import InteractiveMotionCommand


CommandState = Literal["pending", "running", "done", "failed", "cancelled"]


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
                lambda: self._quit.is_set() or self._next_pending_locked() is not None,
                timeout=timeout_s,
            ):
                return None
            if self._quit.is_set():
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
