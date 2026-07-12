"""快照恢复使用的补偿事务，以及回滚不完整后的 runtime fail-stop 状态。

这里的“事务”不是数据库或 PhysX 原生事务：调用方仍然逐项写 articulation、对象位姿和
控制缓存，本模块只负责记录与这些写操作一一对应的补偿动作。调用方必须在实际写入之前
注册回滚动作；后续任一步骤失败时，补偿动作按写入顺序的逆序执行，从而尽量恢复写入前
状态。

如果某个补偿动作失败，或失败发生在已标记为不可逆的步骤之后，runtime 的状态就不能再
被证明是一致的。此时会记录首个 fatal reason 并触发退出信号，后续状态写入统一拒绝；
可靠的恢复方式是销毁并重建 runtime，而不是继续在未知状态上操作。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager


class RuntimeMutationRejected(RuntimeError):
    """runtime 曾发生不完整回滚，因此拒绝继续修改其状态。"""


class RollbackStepError(RuntimeError):
    """一个带名称的补偿步骤执行失败。

    ``step`` 标明失败的恢复单元，``cause`` 保留原始异常，便于上层日志同时呈现恢复阶段
    和底层错误，而不丢失异常链。
    """

    def __init__(self, step: str, cause: BaseException) -> None:
        self.step = str(step)
        self.cause = cause
        super().__init__(f"{self.step}: {type(cause).__name__}: {cause}")


class SnapshotRollbackError(RuntimeError):
    """快照写入失败，且至少一个尽力回滚步骤也失败。

    该异常表示原始写入异常已经不再是唯一问题：runtime 可能同时包含新旧状态，调用方
    不应捕获后继续写入。``cause`` 是触发事务回滚的异常，``rollback_errors`` 列出所有
    失败的补偿动作。
    """

    def __init__(
        self,
        operation: str,
        cause: BaseException,
        rollback_errors: Sequence[BaseException],
    ) -> None:
        self.operation = str(operation)
        self.cause = cause
        self.rollback_errors = tuple(rollback_errors)
        details = "; ".join(
            f"{type(error).__name__}: {error}" for error in self.rollback_errors
        )
        super().__init__(
            f"{self.operation} failed and rollback was incomplete: {details}"
        )


class MutationTransaction:
    """按照前向写入顺序收集补偿动作。

    本对象只管理补偿动作的生命周期，不主动执行任何 runtime 写入。典型调用顺序是
    ``add_rollback``、执行对应写入，最后由 ``mutation_transaction`` 在成功时提交或在
    失败时逆序回滚。
    """

    def __init__(self) -> None:
        self._rollback_actions: list[tuple[str, Callable[[], None]]] = []
        self._irreversible_steps: list[str] = []

    def add_rollback(self, step: str, action: Callable[[], None]) -> None:
        """为即将执行的写入注册补偿动作。

        必须在前向写入之前调用，这样即使 setter 写到一半抛出异常，该步骤仍会参与回滚。
        ``step`` 仅用于诊断，``action`` 应恢复注册时捕获的旧值且尽量保持幂等。
        """

        if not callable(action):
            raise TypeError("rollback action must be callable")
        self._rollback_actions.append((str(step), action))

    def mark_irreversible(self, step: str) -> None:
        """声明一个无法由补偿动作完整重建的提交步骤已经开始。

        例如清空轨迹缓冲或重置 observer 缓存会丢失内部历史。若其后再发生异常，即便所有
        可逆字段均成功恢复，也必须将 runtime 标记为 fatal，避免把“值相同”误判为“状态
        完全等价”。
        """

        self._irreversible_steps.append(str(step))

    @property
    def irreversible_steps(self) -> tuple[str, ...]:
        """返回事务内已经尝试过的不可逆步骤名称快照。"""

        return tuple(self._irreversible_steps)

    def rollback(self) -> tuple[BaseException, ...]:
        """逆序执行所有补偿动作，并汇总而不是隐藏回滚异常。

        一个补偿动作失败后仍继续执行更早的动作，以最大化可恢复状态；返回值为空表示所有
        已登记动作均成功。无论结果如何都会清空动作列表，避免同一事务被重复回滚。
        """

        errors: list[BaseException] = []
        # 后写入的状态通常依赖先写入的状态，因此补偿必须严格遵循 LIFO 顺序。
        for step, action in reversed(self._rollback_actions):
            try:
                action()
            except BaseException as exc:
                errors.append(RollbackStepError(step, exc))
        self._rollback_actions.clear()
        return tuple(errors)

    def commit(self) -> None:
        """完整写入成功后丢弃补偿动作和不可逆步骤记录。"""

        self._rollback_actions.clear()
        self._irreversible_steps.clear()


def runtime_fatal_error(runtime: object) -> str | None:
    """返回 runtime 首次进入 fail-stop 的原因；尚可写时返回 ``None``。"""

    value = getattr(runtime, "fatal_error", None)
    return None if value is None else str(value)


def require_runtime_mutable(runtime: object, *, operation: str) -> None:
    """在写入入口拒绝已经 fail-stop 的 runtime。

    所有状态修改 API 都应在读取/校验后、首次写入前调用此守卫。fatal 状态不会自动清除，
    因为仅靠本地字段无法确认 PhysX、控制缓存和规划器状态重新一致。
    """

    reason = runtime_fatal_error(runtime)
    if reason is not None:
        raise RuntimeMutationRejected(
            f"{operation} rejected: runtime requires rebuild after fatal mutation "
            f"failure: {reason}"
        )


def mark_runtime_fatal(
    runtime: object,
    *,
    operation: str,
    cause: BaseException,
    rollback_errors: Sequence[BaseException] = (),
    irreversible_steps: Sequence[str] = (),
) -> str:
    """记录永久 runtime 故障，并尽力触发可用的退出钩子。

    只保留首次 fatal reason，防止后续清理异常覆盖最接近根因的信息。退出信号按
    ``quit_event.set``、``request_quit``、session app ``close`` 的可用性尝试；信号本身
    的异常不会替代原始事务异常。
    """

    # 首个 fatal reason 是诊断根因；重复标记只再次尝试退出，不改写原因。
    existing = runtime_fatal_error(runtime)
    if existing is None:
        message = f"{operation}: {type(cause).__name__}: {cause}"
        if rollback_errors:
            details = "; ".join(
                f"{type(error).__name__}: {error}" for error in rollback_errors
            )
            message = f"{message}; rollback_errors=[{details}]"
        if irreversible_steps:
            message = (
                f"{message}; irreversible_steps="
                f"{list(str(step) for step in irreversible_steps)}"
            )
        setattr(runtime, "fatal_error", message)
    else:
        message = existing

    quit_signaled = False
    quit_event = getattr(runtime, "quit_event", None)
    set_quit = getattr(quit_event, "set", None)
    if callable(set_quit):
        quit_signaled = _best_effort_call(set_quit) or quit_signaled
    request_quit = getattr(runtime, "request_quit", None)
    if callable(request_quit):
        quit_signaled = _best_effort_call(request_quit) or quit_signaled
    if not quit_signaled:
        session = getattr(runtime, "session", None)
        app = getattr(session, "app", None)
        close_app = getattr(app, "close", None)
        if callable(close_app):
            _best_effort_call(close_app)
    return message


@contextmanager
def mutation_transaction(
    runtime: object,
    *,
    operation: str,
) -> Iterator[MutationTransaction]:
    """为一组 runtime 写入建立补偿事务边界。

    正常退出时调用 ``commit``，仅清除补偿记录；异常退出时先逆序回滚，再重新抛出原始
    异常。若回滚本身失败，则抛出 ``SnapshotRollbackError`` 并将原始异常作为 cause；若
    已执行不可逆步骤，即使可逆回滚成功也会把 runtime 标记为 fatal。

    注意该上下文管理器提供的是“失败时尽力恢复”的原子性：只有全部补偿成功且没有不可逆
    步骤时，调用方才能把失败后的 runtime 视为写入前状态。
    """

    require_runtime_mutable(runtime, operation=operation)
    transaction = MutationTransaction()
    try:
        yield transaction
    except BaseException as cause:
        irreversible_steps = transaction.irreversible_steps
        rollback_errors = transaction.rollback()
        if rollback_errors or irreversible_steps:
            mark_runtime_fatal(
                runtime,
                operation=operation,
                cause=cause,
                rollback_errors=rollback_errors,
                irreversible_steps=irreversible_steps,
            )
        if rollback_errors:
            raise SnapshotRollbackError(
                operation,
                cause,
                rollback_errors,
            ) from cause
        raise
    else:
        transaction.commit()


def _best_effort_call(callback: Callable[[], object]) -> bool:
    """调用 fail-stop 信号，同时避免信号异常覆盖事务异常。"""

    try:
        callback()
    except BaseException:
        return False
    return True


__all__ = [
    "MutationTransaction",
    "RollbackStepError",
    "RuntimeMutationRejected",
    "SnapshotRollbackError",
    "mark_runtime_fatal",
    "mutation_transaction",
    "require_runtime_mutable",
    "runtime_fatal_error",
]
