"""交互运动命令的线程安全队列与生命周期注册表。

transport 线程只负责提交请求和等待结果，仿真主线程负责消费命令并读写 Isaac/PhysX。
本模块用 ``Condition``、``Event`` 和不可变 request DTO 维护这条边界，同时集中生成
WebSocket 事件和 ``status`` 所需的可序列化状态。

容量按“已 admission 且尚未进入终态”的请求计数，而不是按容器长度计数：motion 请求在
``pending`` 和 ``running`` 阶段都占用槽位，snapshot 请求直到执行结束或确认取消后才
释放槽位。这样 transport 收到 ``accepted`` 后，主线程一定为该请求预留了执行资源。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from threading import Condition, Event, Lock
from time import monotonic
from typing import Literal

from linkerbot_sim.app.motion.timeline.requests import RobotTimelineRequest
from linkerbot_sim.app.interactive.single_scene.protocol import InteractiveMotionCommand


CommandState = Literal["pending", "running", "done", "failed", "cancelled"]
SnapshotRequestKind = Literal["get_snapshot", "set_snapshot"]
TERMINAL_COMMAND_STATES = frozenset({"done", "failed", "cancelled"})


class InteractiveQueueFullError(RuntimeError):
    """有界交互请求队列无法为新请求预留 admission 槽位。"""

    def __init__(self, resource: str, *, depth: int, capacity: int) -> None:
        """保存拒绝发生时的资源名称、占用深度和容量，供稳定响应使用。"""

        self.resource = resource
        self.depth = int(depth)
        self.capacity = int(capacity)
        super().__init__(
            f"{resource} is full (depth={self.depth}, capacity={self.capacity})"
        )

    def response(self, *, request_id: str | None = None) -> dict[str, object]:
        """返回跨 transport 一致的容量拒绝响应，并保留拒绝时的深度快照。"""

        response: dict[str, object] = {
            "event": "rejected",
            "accepted": False,
            "code": f"{self.resource}_full",
            "reason": f"{self.resource}_full",
            "error": str(self),
            "depth": self.depth,
            "capacity": self.capacity,
        }
        if request_id is not None:
            response["id"] = request_id
        return response


class InteractiveRequestConflictError(RuntimeError):
    """只能单例执行的控制请求已经处于 pending 或 running 状态。"""

    def __init__(self, resource: str) -> None:
        """记录发生单例冲突的资源名称。"""

        self.resource = resource
        super().__init__(f"{resource} is already pending or running")

    def response(self, *, request_id: str | None = None) -> dict[str, object]:
        """返回跨 transport 一致的单例请求冲突响应。"""

        response: dict[str, object] = {
            "event": "rejected",
            "accepted": False,
            "code": f"{self.resource}_in_progress",
            "reason": f"{self.resource}_in_progress",
            "error": str(self),
        }
        if request_id is not None:
            response["id"] = request_id
        return response


@dataclass(frozen=True)
class ResetRequest:
    """一次待主线程执行的 simulation reset 请求。"""

    reset_id: str
    clear_queue: bool = True
    hold_after_reset: bool = True

    def snapshot(self) -> dict[str, object]:
        """返回可序列化 reset 请求。"""

        return {
            "id": self.reset_id,
            "clear_queue": self.clear_queue,
            "hold_after_reset": self.hold_after_reset,
        }


@dataclass
class SnapshotRequest:
    """一次待仿真主线程处理的 snapshot 请求。

    transport 线程只负责解析 JSON 和等待结果，不能直接读写 Isaac/PhysX 状态；所以
    get/set snapshot 都会被封装成这个 request，由 Single Scene 主仿真循环取出执行。

    ``claimed`` 仅表示请求已离开 deque；``executing`` 必须由主线程紧贴 runtime 访问前
    原子设置。这个两阶段交接让 transport 超时可以取消“已取出但尚未执行”的请求，同时
    又不会把已经改变 runtime 状态的请求错误报告为未接纳。
    """

    snapshot_id: str
    kind: SnapshotRequestKind
    snapshot: Mapping[str, object] | None = None
    label_map: Mapping[str, str] | None = None
    strict: bool = True
    response: dict[str, object] | None = None
    done: bool = False
    claimed: bool = False
    executing: bool = False
    cancelled: bool = False
    admission_released: bool = False

    def snapshot_info(self) -> dict[str, object]:
        """返回可序列化 snapshot 请求摘要，用于事件广播。"""

        return {
            "id": self.snapshot_id,
            "kind": self.kind,
            "strict": bool(self.strict),
        }


@dataclass
class QueuedMotionCommand:
    """一条 timeline 命令及其从 pending 到终态的执行状态。"""

    command_id: str
    timeline: RobotTimelineRequest
    state: CommandState = "pending"
    error: str | None = None
    steps: int | None = None

    def snapshot(self) -> dict[str, object]:
        """返回可序列化的命令状态快照，供 status 响应和 WebSocket 事件使用。"""

        return {
            "id": self.command_id,
            "state": self.state,
            "error": self.error,
            "steps": self.steps,
            "command_kind": "timeline",
        }


class InteractiveMotionQueue:
    """交互执行队列、命令状态注册表和跨线程通知中心。

    ``submit``/查询接口可由 transport 线程调用；``next_pending``、reset 和 snapshot
    消费接口由仿真主线程调用。所有共享状态都在内部锁保护下更新，事件监听器则在锁外
    通知，避免 listener 回调反向进入队列时发生死锁。motion、reset、snapshot 三类请求
    共用唤醒条件但维护独立状态机；主线程始终是 runtime 副作用的唯一所有者。
    """

    def __init__(
        self,
        *,
        request_capacity: int | None = None,
        terminal_history_capacity: int | None = None,
        snapshot_request_capacity: int | None = None,
        snapshot_timeout_s: float = 30.0,
        planner_request_defaults: object | None = None,
        command_defaults: object | None = None,
    ) -> None:
        """初始化命令注册表、admission 计数和跨线程通知原语。

        ``None`` 容量表示不限制；终态历史容量只约束诊断记录，不影响已经完成的执行。
        snapshot timeout 约束的是开始执行前的等待阶段，执行一旦开始就必须等待真实结果。
        """

        self._request_capacity = _optional_capacity(
            request_capacity,
            name="request_capacity",
            allow_zero=False,
        )
        self._terminal_history_capacity = _optional_capacity(
            terminal_history_capacity,
            name="terminal_history_capacity",
            allow_zero=True,
        )
        self._snapshot_request_capacity = _optional_capacity(
            snapshot_request_capacity,
            name="snapshot_request_capacity",
            allow_zero=False,
        )
        self._snapshot_timeout_s = _positive_timeout(
            snapshot_timeout_s,
            name="snapshot_timeout_s",
        )
        self.planner_request_defaults = planner_request_defaults
        self.command_defaults = command_defaults
        self._condition = Condition()
        self._lock = Lock()
        self._next_id = 1
        self._commands: dict[str, QueuedMotionCommand] = {}
        self._order: list[str] = []
        self._terminal_order: deque[str] = deque()
        self._current_id: str | None = None
        self._request_rejected = 0
        self._terminal_evicted = 0
        self._next_reset_id = 1
        self._reset_request: ResetRequest | None = None
        self._resetting = False
        self._last_reset: dict[str, object] | None = None
        self._reset_rejected = 0
        self._next_snapshot_id = 1
        self._snapshot_requests: deque[SnapshotRequest] = deque()
        self._snapshot_outstanding = 0
        self._snapshot_rejected = 0
        self._snapshot_timed_out = 0
        self._cancel_current = Event()
        self._estop = Event()
        self._quit = Event()
        self._listeners: list[Callable[[dict[str, object]], None]] = []
        self._status_provider: Callable[[], Mapping[str, object]] | None = None
        self._transport_status_provider: Callable[[], Mapping[str, object]] | None = (
            None
        )

    def set_status_provider(
        self, provider: Callable[[], Mapping[str, object]] | None
    ) -> None:
        """注册 runtime discovery provider，并把其结果合并到每次 status 响应。"""

        with self._lock:
            self._status_provider = provider

    def set_transport_status_provider(
        self, provider: Callable[[], Mapping[str, object]] | None
    ) -> None:
        """注册 transport 指标 provider，供统一 status 响应读取。"""

        with self._lock:
            self._transport_status_provider = provider

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
        """为 timeline 命令预留槽位、登记为 pending，并发出 accepted 事件。

        ID 去重和容量检查与写入在同一把 condition 锁内完成，因此并发 transport 不会
        同时越过上限。事件在解锁后广播，监听器看到 accepted 时命令已经可被主线程读取。
        """

        if command.kind != "timeline":
            raise ValueError(f"cannot submit non-move command {command.kind!r}")
        if command.timeline is None:
            raise ValueError("timeline command is missing timeline data")
        command_id = command.command_id or self._new_command_id()
        with self._condition:
            if command_id in self._commands:
                raise ValueError(f"duplicate command id: {command_id}")
            active_depth = self._active_request_depth_locked()
            if (
                self._request_capacity is not None
                and active_depth >= self._request_capacity
            ):
                self._request_rejected += 1
                raise InteractiveQueueFullError(
                    "request_queue",
                    depth=active_depth,
                    capacity=self._request_capacity,
                )
            queued = QueuedMotionCommand(
                command_id=command_id,
                timeline=command.timeline,
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

    def next_pending(
        self, *, timeout_s: float | None = None
    ) -> QueuedMotionCommand | None:
        """阻塞等待下一个 pending 命令，并原子地把它切换为 running。

        reset 与 quit 会打断等待；同一时刻只允许一个 ``_current_id``。返回对象后，调用
        方即成为该命令的执行所有者，并必须最终调用一个 ``mark_*`` 终态接口释放槽位。
        """

        with self._condition:
            if not self._condition.wait_for(
                lambda: (
                    self._quit.is_set()
                    or self._reset_request is not None
                    or (
                        self._current_id is None
                        and self._next_pending_locked() is not None
                    )
                ),
                timeout=timeout_s,
            ):
                return None
            if (
                self._quit.is_set()
                or self._reset_request is not None
                or self._current_id is not None
            ):
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
        """取消指定命令；pending 立即终止，running 只发出协作式停止请求。

        running 命令可能已产生物理副作用，因此本方法不会抢先写入 cancelled 终态；执行
        所有者观察 stop flag、停止在安全边界后，再通过 ``mark_cancelled`` 确认终态。
        """

        with self._condition:
            queued = self._commands.get(command_id)
            if queued is None:
                return False
            if queued.state == "pending":
                queued.state = "cancelled"
                self._record_terminal_locked(command_id)
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
        clear_queue: bool = True,
        hold_after_reset: bool = True,
    ) -> ResetRequest:
        """请求主循环在安全边界执行 reset。

        reset 是单例控制操作。接纳后可同步取消所有 pending motion，并向 running motion
        发出停止信号；真正的 runtime reset 及最终成功/失败状态仍只由主线程写入。
        """

        request = ResetRequest(
            reset_id=reset_id or self._new_reset_id(),
            clear_queue=bool(clear_queue),
            hold_after_reset=bool(hold_after_reset),
        )
        cancelled_ids: list[str] = []
        with self._condition:
            if self._resetting:
                self._reset_rejected += 1
                raise InteractiveRequestConflictError("reset_request")
            self._reset_request = request
            self._resetting = True
            self._last_reset = {
                "id": request.reset_id,
                "state": "requested",
            }
            if request.clear_queue:
                for command in self._commands.values():
                    if command.state == "pending":
                        command.state = "cancelled"
                        cancelled_ids.append(command.command_id)
                for command_id in cancelled_ids:
                    self._record_terminal_locked(command_id)
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
        label_map: Mapping[str, str] | None = None,
        strict: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        """请求仿真主线程执行 snapshot get/set，并阻塞等待响应。

        这里使用 ``Condition`` 把 transport 线程和仿真主循环串起来：transport 创建请求后
        等待 ``mark_snapshot_done/failed`` 唤醒；主循环在安全的 physics 边界消费请求。
        """

        if kind == "set_snapshot" and snapshot is None:
            raise ValueError("set_snapshot requires snapshot")
        wait_timeout_s = (
            self._snapshot_timeout_s
            if timeout_s is None
            else _positive_timeout(timeout_s, name="timeout_s")
        )
        request = SnapshotRequest(
            snapshot_id=snapshot_id or self._new_snapshot_id(),
            kind=kind,
            snapshot=snapshot,
            label_map=label_map,
            strict=bool(strict),
        )
        with self._condition:
            # snapshot 请求单独排队，不与 motion command 共用 pending 列表；这样它不会被
            # 长时间动作队列阻塞，也不会改变 motion command 的状态机。
            snapshot_depth = self._snapshot_outstanding
            if (
                self._snapshot_request_capacity is not None
                and snapshot_depth >= self._snapshot_request_capacity
            ):
                self._snapshot_rejected += 1
                error = InteractiveQueueFullError(
                    "snapshot_request_queue",
                    depth=snapshot_depth,
                    capacity=self._snapshot_request_capacity,
                )
                return error.response(request_id=request.snapshot_id)
            self._snapshot_requests.append(request)
            self._snapshot_outstanding += 1
            self._condition.notify_all()
        self.emit({"event": "snapshot_requested", **request.snapshot_info()})
        with self._condition:
            deadline = monotonic() + wait_timeout_s
            while True:
                if request.done and request.response is not None:
                    return dict(request.response)
                if self._quit.is_set():
                    if request.executing:
                        return {
                            "event": "snapshot_running",
                            "accepted": True,
                            "state": "running",
                            "id": request.snapshot_id,
                        }
                    request.cancelled = True
                    try:
                        self._snapshot_requests.remove(request)
                    except ValueError:
                        pass
                    self._release_snapshot_admission_locked(request)
                    return {
                        "event": "snapshot_cancelled",
                        "accepted": False,
                        "reason": "shutdown",
                        "id": request.snapshot_id,
                    }
                remaining = deadline - monotonic()
                if remaining > 0.0:
                    self._condition.wait(timeout=remaining)
                    continue
                if request.executing:
                    # runtime 状态可能已经改变。执行一旦开始，admission deadline 就不能
                    # 再诚实地把请求报告为拒绝，只能等待执行方给出确定结果。
                    self._condition.wait_for(
                        lambda: request.done or self._quit.is_set()
                    )
                    continue
                request.cancelled = True
                try:
                    self._snapshot_requests.remove(request)
                except ValueError:
                    pass
                if not request.executing:
                    self._release_snapshot_admission_locked(request)
                self._snapshot_timed_out += 1
                return {
                    "event": "snapshot_timeout",
                    "accepted": False,
                    "id": request.snapshot_id,
                }

    def consume_snapshot_request(self) -> SnapshotRequest | None:
        """取出 pending snapshot 请求；只有仿真主循环应调用。

        返回后请求对象仍由等待中的 transport 持有引用，所以主循环只需要在同一个对象上
        写入 response/done 并 notify 即可。
        """

        with self._condition:
            while self._snapshot_requests:
                request = self._snapshot_requests.popleft()
                if request.cancelled:
                    continue
                request.claimed = True
                return request
            return None

    def begin_snapshot_request(self, request: SnapshotRequest) -> bool:
        """在 transport 尚未超时取消时，原子授权 snapshot 开始执行。

        仿真线程必须在接触 runtime 状态前立即调用本方法。它封闭了 dequeue 与 timeout
        之间的竞态：即使请求已经离开 deque，只要 transport 在此期间超时，它便不能在
        稍后重新开始。返回 ``True`` 后执行已不可逆地接纳，调用方必须标记完成或失败。
        """

        with self._condition:
            if (
                request.cancelled
                or request.done
                or request.executing
                or not request.claimed
            ):
                return False
            request.executing = True
            return True

    def mark_snapshot_done(
        self,
        request: SnapshotRequest,
        response: Mapping[str, object],
    ) -> None:
        """标记 snapshot 请求成功完成并唤醒等待的 transport。"""

        payload = dict(response)
        payload.setdefault("id", request.snapshot_id)
        with self._condition:
            if request.cancelled:
                request.executing = False
                self._release_snapshot_admission_locked(request)
                self._condition.notify_all()
                return
            # response 存在 request 对象上，等待方醒来后直接返回这份 JSON-compatible dict。
            request.response = payload
            request.done = True
            request.executing = False
            self._release_snapshot_admission_locked(request)
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
            if request.cancelled:
                request.executing = False
                self._release_snapshot_admission_locked(request)
                self._condition.notify_all()
                return
            request.response = response
            request.done = True
            request.executing = False
            self._release_snapshot_admission_locked(request)
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
            cancelled_ids: list[str] = []
            for command in self._commands.values():
                if command.state == "pending":
                    command.state = "cancelled"
                    cancelled_ids.append(command.command_id)
            for command_id in cancelled_ids:
                self._record_terminal_locked(command_id)
            self._condition.notify_all()
        self.emit({"event": "estop", "state": "cancelled"})

    def request_quit(self) -> None:
        """请求交互循环退出，并唤醒可能正在等待队列的执行线程。"""

        self._quit.set()
        with self._condition:
            self._condition.notify_all()

    def should_stop_current(self) -> bool:
        """返回当前执行命令是否应停止；执行 stepper 会周期性轮询该函数。"""

        return (
            self._cancel_current.is_set() or self._estop.is_set() or self._quit.is_set()
        )

    def estop_requested(self) -> bool:
        """返回是否已收到急停请求。"""

        return self._estop.is_set()

    def quit_requested(self) -> bool:
        """返回是否已收到退出请求。"""

        return self._quit.is_set()

    def status(self, command_id: str | None = None) -> dict[str, object]:
        """生成 status 响应；不传 id 时返回整个队列和当前急停状态。"""

        with self._condition:
            queue_status = self._queue_status_locked()
            if command_id is not None:
                command = self._commands.get(command_id)
                result = {
                    "event": "status",
                    "commands": [] if command is None else [command.snapshot()],
                    "queue": queue_status,
                }
            else:
                result = {
                    "event": "status",
                    "commands": [
                        self._commands[item].snapshot() for item in self._order
                    ],
                    "current_id": self._current_id,
                    "estop": self._estop.is_set(),
                    "resetting": self._resetting,
                    "last_reset": self._last_reset,
                    "queue": queue_status,
                }
        with self._lock:
            provider = self._status_provider
            transport_provider = self._transport_status_provider
        if provider is not None:
            result.update(dict(provider()))
        result["queue"] = queue_status
        if transport_provider is not None:
            result["transport"] = dict(transport_provider())
        return result

    def pending_index(self, command_id: str) -> int | None:
        """返回命令在 pending 子队列中的位置；非 pending 或不存在时返回 None。"""

        with self._condition:
            return self._pending_index_locked(command_id)

    def _pending_index_locked(self, command_id: str) -> int | None:
        """在持有 condition 锁时计算 pending 队列索引。"""

        pending = [
            item for item in self._order if self._commands[item].state == "pending"
        ]
        try:
            return pending.index(command_id)
        except ValueError:
            return None

    def emit(self, event: dict[str, object]) -> None:
        """在队列锁外同步广播事件，并为每个监听器复制一份字典。

        listener 负责自行快速返回或转存到异步队列；这里不吞掉异常，以免 transport 事件
        链路静默失效。复制只隔离顶层键，不承诺递归深拷贝嵌套 payload。
        """

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
            if queued.state in TERMINAL_COMMAND_STATES:
                return
            queued.state = state
            queued.error = error
            queued.steps = steps
            if self._current_id == command_id:
                self._current_id = None
                self._cancel_current.clear()
            self._record_terminal_locked(command_id)
            self._condition.notify_all()
        event: dict[str, object] = {
            "event": state,
            "id": command_id,
            "state": state,
        }
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

    def _pending_depth_locked(self) -> int:
        """返回已经接纳但尚未开始运行的命令数；调用方必须持有 condition 锁。"""

        return sum(
            self._commands[command_id].state == "pending" for command_id in self._order
        )

    def _active_request_depth_locked(self) -> int:
        """返回仍占用 admission 槽位的 pending 与 running 请求总数。"""

        return sum(
            self._commands[command_id].state in {"pending", "running"}
            for command_id in self._order
        )

    def _record_terminal_locked(self, command_id: str) -> None:
        """记录一次终态切换，并只淘汰超出容量的终态历史。"""

        self._terminal_order.append(command_id)
        capacity = self._terminal_history_capacity
        if capacity is None:
            return
        while len(self._terminal_order) > capacity:
            evicted_id = self._terminal_order.popleft()
            command = self._commands.get(evicted_id)
            if command is None or command.state not in TERMINAL_COMMAND_STATES:
                continue
            del self._commands[evicted_id]
            self._order.remove(evicted_id)
            self._terminal_evicted += 1

    def _release_snapshot_admission_locked(self, request: SnapshotRequest) -> None:
        """在请求确定不会再执行后，恰好一次地释放 snapshot admission 槽位。"""

        if request.admission_released:
            return
        request.admission_released = True
        self._snapshot_outstanding -= 1
        if self._snapshot_outstanding < 0:
            raise RuntimeError("snapshot admission accounting became negative")

    def _queue_status_locked(self) -> dict[str, object]:
        """返回各有界队列的实时深度、容量及进程期累计拒绝/淘汰计数。"""

        active_depth = self._active_request_depth_locked()
        pending_depth = self._pending_depth_locked()
        running_depth = active_depth - pending_depth
        terminal_depth = len(self._terminal_order)
        snapshot_pending_depth = len(self._snapshot_requests)
        snapshot_depth = self._snapshot_outstanding
        snapshot_running_depth = snapshot_depth - snapshot_pending_depth
        return {
            "depth": active_depth,
            "capacity": self._request_capacity,
            "rejected": self._request_rejected,
            "evicted": self._terminal_evicted,
            "active": {
                "depth": active_depth,
                "capacity": self._request_capacity,
                "rejected": self._request_rejected,
            },
            "pending": {
                "depth": pending_depth,
                "capacity": self._request_capacity,
            },
            "running": {
                "depth": running_depth,
                "capacity": self._request_capacity,
            },
            "terminal_history": {
                "depth": terminal_depth,
                "capacity": self._terminal_history_capacity,
                "evicted": self._terminal_evicted,
            },
            "snapshot_requests": {
                "depth": snapshot_depth,
                "capacity": self._snapshot_request_capacity,
                "pending": snapshot_pending_depth,
                "running": snapshot_running_depth,
                "timeout_s": self._snapshot_timeout_s,
                "rejected": self._snapshot_rejected,
                "timed_out": self._snapshot_timed_out,
            },
            "reset_requests": {
                "depth": int(self._resetting),
                "capacity": 1,
                "rejected": self._reset_rejected,
            },
        }

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


def _optional_capacity(
    value: int | None,
    *,
    name: str,
    allow_zero: bool,
) -> int | None:
    """校验可选队列容量；拒绝 Python 中虽是整数子类但语义不明确的布尔值。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _positive_timeout(value: float, *, name: str) -> float:
    """校验 timeout 是有限正数，避免无限值破坏 deadline 计算。"""

    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result
