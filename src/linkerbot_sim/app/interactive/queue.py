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
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[dict[str, object]], None]) -> None:
        with self._lock:
            self._listeners = [
                existing for existing in self._listeners if existing is not listener
            ]

    def submit(self, command: InteractiveMotionCommand) -> QueuedMotionCommand:
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
        self._mark_terminal(command_id, "done", steps=steps)

    def mark_failed(self, command_id: str, error: str) -> None:
        self._mark_terminal(command_id, "failed", error=error)

    def mark_cancelled(
        self,
        command_id: str,
        error: str | None = None,
        *,
        steps: int | None = None,
    ) -> None:
        self._mark_terminal(command_id, "cancelled", error=error, steps=steps)

    def cancel(self, command_id: str) -> bool:
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
        with self._condition:
            if self._current_id is None:
                return False
            self._cancel_current.set()
            return True

    def request_estop(self) -> None:
        self._estop.set()
        self._cancel_current.set()
        with self._condition:
            for command in self._commands.values():
                if command.state == "pending":
                    command.state = "cancelled"
            self._condition.notify_all()
        self.emit({"event": "estop", "state": "cancelled"})

    def request_quit(self) -> None:
        self._quit.set()
        with self._condition:
            self._condition.notify_all()

    def should_stop_current(self) -> bool:
        return self._cancel_current.is_set() or self._estop.is_set() or self._quit.is_set()

    def estop_requested(self) -> bool:
        return self._estop.is_set()

    def quit_requested(self) -> bool:
        return self._quit.is_set()

    def status(self, command_id: str | None = None) -> dict[str, object]:
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
        with self._condition:
            return self._pending_index_locked(command_id)

    def _pending_index_locked(self, command_id: str) -> int | None:
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
        for command_id in self._order:
            queued = self._commands[command_id]
            if queued.state == "pending":
                return queued
        return None

    def _new_command_id(self) -> str:
        with self._lock:
            command_id = f"cmd-{self._next_id}"
            self._next_id += 1
            return command_id
