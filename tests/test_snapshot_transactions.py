from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from linkerbot_sim.snapshots.transactions import (
    RuntimeMutationRejected,
    SnapshotRollbackError,
    mutation_transaction,
)


def test_mutation_transaction_rolls_back_in_reverse_and_rethrows_cause() -> None:
    runtime = SimpleNamespace(quit_event=threading.Event())
    rollback_order: list[str] = []
    cause = RuntimeError("forward write failed")

    with pytest.raises(RuntimeError) as exc_info:
        with mutation_transaction(runtime, operation="test write") as transaction:
            transaction.add_rollback("first", lambda: rollback_order.append("first"))
            transaction.add_rollback("second", lambda: rollback_order.append("second"))
            raise cause

    assert exc_info.value is cause
    assert rollback_order == ["second", "first"]
    assert not hasattr(runtime, "fatal_error")
    assert not runtime.quit_event.is_set()

    with mutation_transaction(runtime, operation="next write"):
        pass


def test_incomplete_rollback_preserves_cause_and_permanently_fail_stops() -> None:
    runtime = SimpleNamespace(quit_event=threading.Event())
    rollback_order: list[str] = []
    cause = RuntimeError("forward write failed")

    def failed_rollback() -> None:
        rollback_order.append("second")
        raise OSError("rollback setter failed")

    with pytest.raises(SnapshotRollbackError) as exc_info:
        with mutation_transaction(runtime, operation="test write") as transaction:
            transaction.add_rollback("first", lambda: rollback_order.append("first"))
            transaction.add_rollback("second", failed_rollback)
            raise cause

    assert exc_info.value.cause is cause
    assert exc_info.value.__cause__ is cause
    assert rollback_order == ["second", "first"]
    assert "forward write failed" in runtime.fatal_error
    assert "rollback setter failed" in runtime.fatal_error
    assert runtime.quit_event.is_set()

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        with mutation_transaction(runtime, operation="next write"):
            raise AssertionError("fatal gate did not reject before mutation")


def test_failure_after_irreversible_commit_action_fail_stops_after_rollback() -> None:
    runtime = SimpleNamespace(quit_event=threading.Event())
    rolled_back: list[str] = []
    cause = RuntimeError("planner cancellation failed")

    with pytest.raises(RuntimeError) as exc_info:
        with mutation_transaction(runtime, operation="state write") as transaction:
            transaction.add_rollback(
                "physical state",
                lambda: rolled_back.append("physical state"),
            )
            transaction.mark_irreversible("trajectory buffer clear")
            raise cause

    assert exc_info.value is cause
    assert rolled_back == ["physical state"]
    assert "irreversible_steps=['trajectory buffer clear']" in runtime.fatal_error
    assert runtime.quit_event.is_set()
