from __future__ import annotations

from dataclasses import replace
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.mirror.interface.state_stream import (
    InteractiveStateStreamHandle,
)
from linkerbot_sim.telemetry.foxglove_state import (
    CompositeStateSink,
    FoxgloveStateSink,
    StatePublisher,
)
from linkerbot_sim.telemetry.state_snapshot import (
    ObjectPoseSnapshot,
    RobotJointStateSnapshot,
    StateSnapshot,
    StateStream,
)


class _FakeFoxgloveLogger:
    def __init__(self) -> None:
        self.joint_states = []
        self.state_json = []
        self.hybrid_control_json = []
        self.scene_spheres = []
        self.closed = False

    def log_joint_state(self, **kwargs) -> None:
        self.joint_states.append(kwargs)

    def log_state_json(self, state, *, time_s=None) -> None:
        self.state_json.append((state, time_s))

    def log_hybrid_control_json(self, diagnostics, *, time_s=None) -> None:
        self.hybrid_control_json.append((diagnostics, time_s))

    def log_scene_spheres(self, **kwargs) -> None:
        self.scene_spheres.append(kwargs)

    def close(self) -> None:
        self.closed = True


def _snapshot() -> StateSnapshot:
    return StateSnapshot(
        step=7,
        time_s=0.8,
        phase="push",
        robots=(
            RobotJointStateSnapshot(
                robot_id=0,
                label="robot_a",
                joint_names=("j0", "j1"),
                positions_rad=np.asarray([0.1, 0.2]),
                velocities_rad_s=np.asarray([1.0, 2.0]),
                accelerations_rad_s2=np.asarray([10.0, 20.0]),
                commanded_efforts=np.asarray([0.3, 0.4]),
                measured_efforts=np.asarray([0.5, 0.6]),
                applied_efforts=np.asarray([0.7, 0.8]),
            ),
            RobotJointStateSnapshot(
                robot_id=1,
                label="robot_b",
                joint_names=("j0",),
                positions_rad=np.asarray([-0.1]),
                velocities_rad_s=np.asarray([-1.0]),
                accelerations_rad_s2=np.asarray([-10.0]),
                commanded_efforts=np.asarray([-0.3]),
                measured_efforts=np.asarray([-0.5]),
                applied_efforts=np.asarray([-0.7]),
            ),
        ),
        objects=(
            ObjectPoseSnapshot(
                name="Tblock",
                prim_path="/World/TBlock",
                position_m=np.asarray([0.1, 0.2, 0.3]),
                orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
        ),
    )


def test_foxglove_state_sink_publishes_joint_state_and_full_json() -> None:
    logger = _FakeFoxgloveLogger()
    sink = FoxgloveStateSink(logger, joint_effort_field="measured")

    sink.publish(_snapshot())

    joint_state = logger.joint_states[0]
    assert joint_state["joint_names"] == [
        "robot_a/j0",
        "robot_a/j1",
        "robot_b/j0",
    ]
    np.testing.assert_allclose(joint_state["positions"], [0.1, 0.2, -0.1])
    np.testing.assert_allclose(joint_state["efforts"], [0.5, 0.6, -0.5])
    state_json, time_s = logger.state_json[0]
    assert state_json["phase"] == "push"
    assert state_json["objects"]["Tblock"]["prim_path"] == "/World/TBlock"
    assert time_s == 0.8
    np.testing.assert_allclose(logger.scene_spheres[0]["positions"], [[0.1, 0.2, 0.3]])


def test_foxglove_state_sink_can_omit_joint_effort_field() -> None:
    logger = _FakeFoxgloveLogger()
    sink = FoxgloveStateSink(logger, joint_effort_field="none")

    sink.publish(_snapshot())

    assert logger.joint_states[0]["efforts"] is None
    sink.close()
    assert logger.closed is True


def test_foxglove_state_sink_honors_modality_switches() -> None:
    logger = _FakeFoxgloveLogger()
    sink = FoxgloveStateSink(
        logger,
        publish_joint_states=False,
        publish_state_json=False,
        publish_scene_markers=False,
    )

    sink.publish(_snapshot())

    assert logger.joint_states == []
    assert logger.state_json == []
    assert logger.hybrid_control_json == []
    assert logger.scene_spheres == []


def test_foxglove_state_sink_publishes_dedicated_hybrid_payload() -> None:
    logger = _FakeFoxgloveLogger()
    sink = FoxgloveStateSink(
        logger,
        publish_joint_states=False,
        publish_state_json=False,
        publish_scene_markers=False,
        publish_hybrid_control=True,
    )
    active = replace(
        _snapshot(),
        hybrid_control={"active": True, "request_id": "hybrid-1"},
    )

    sink.publish(active)
    sink.publish(_snapshot())

    assert logger.hybrid_control_json == [
        ({"active": True, "request_id": "hybrid-1"}, 0.8),
        ({"active": False}, 0.8),
    ]


def test_state_publisher_continues_after_error_and_reports_metrics() -> None:
    failed = threading.Event()
    published = threading.Event()

    class FlakySink:
        def __init__(self) -> None:
            self.calls = 0

        def publish(self, _snapshot: StateSnapshot) -> None:
            self.calls += 1
            if self.calls == 1:
                failed.set()
                raise RuntimeError("first publish failed")
            published.set()

        def close(self) -> None:
            return None

    stream = StateStream(capacity=2, drop_policy="drop_oldest")
    publisher = StatePublisher(
        stream=stream,
        sink=FlakySink(),
        on_error="continue",
        worker_poll_interval_s=0.01,
    )
    publisher.start()
    stream.publish(_snapshot())
    assert failed.wait(timeout=1.0)
    stream.publish(_snapshot())
    assert published.wait(timeout=1.0)

    status = publisher.status()
    assert status["error_count"] == 1
    assert status["last_published_sequence"] == 2
    assert status["dropped_snapshots"] == 0
    assert "RuntimeError: first publish failed" == status["last_error"]
    assert publisher.close(timeout_s=1.0) is True


def test_state_publisher_close_drains_admitted_snapshots_before_sink_close() -> None:
    entered = threading.Event()
    release = threading.Event()
    close_started = threading.Event()

    class SlowSink:
        def __init__(self) -> None:
            self.published: list[StateSnapshot] = []
            self.closed = False

        def publish(self, snapshot: StateSnapshot) -> None:
            if not self.published:
                entered.set()
                release.wait(timeout=2.0)
            self.published.append(snapshot)

        def close(self) -> None:
            self.closed = True

    snapshots = (_snapshot(), _snapshot(), _snapshot())
    stream = StateStream(capacity=3, drop_policy="drop_oldest")
    sink = SlowSink()
    publisher = StatePublisher(
        stream=stream,
        sink=sink,
        shutdown_timeout_s=1.0,
        worker_poll_interval_s=0.01,
    )
    publisher.start()
    stream.publish(snapshots[0])
    assert entered.wait(timeout=1.0)
    stream.publish(snapshots[1])
    stream.publish(snapshots[2])
    close_result: list[bool] = []

    def close_publisher() -> None:
        close_started.set()
        close_result.append(publisher.close())

    closer = threading.Thread(target=close_publisher)
    closer.start()
    assert close_started.wait(timeout=1.0)
    closer.join(timeout=0.05)
    assert closer.is_alive()
    release.set()
    closer.join(timeout=1.0)

    assert not closer.is_alive()
    assert close_result == [True]
    assert sink.published == list(snapshots)
    assert sink.closed is True
    status = publisher.status()
    assert status["buffer_depth"] == 0
    assert status["dropped_snapshots"] == 0
    assert status["last_published_sequence"] == 3


def test_state_stream_handle_keeps_timeout_status_until_successful_retry() -> None:
    previous_observer = object()

    def previous_status() -> dict[str, object]:
        return {"owner": "previous"}

    class RetriablePublisher:
        def __init__(self) -> None:
            self.close_results = [False, True]

        def close(self) -> bool:
            return self.close_results.pop(0)

        def status(self) -> dict[str, object]:
            return {"thread_alive": bool(self.close_results)}

    publisher = RetriablePublisher()
    runtime = SimpleNamespace(
        state_observer=object(),
        telemetry_status_provider=lambda: {"owner": "active"},
    )
    handle = InteractiveStateStreamHandle(
        runtime=runtime,
        previous_observer=previous_observer,
        previous_status_provider=previous_status,
        publisher=publisher,  # type: ignore[arg-type]
    )

    assert handle.close() is False
    assert runtime.state_observer is previous_observer
    assert runtime.telemetry_status_provider() == {"thread_alive": True}

    assert handle.close() is True
    assert runtime.telemetry_status_provider is previous_status


def test_composite_state_sink_closes_every_sink_and_retries_only_failure() -> None:
    close_calls: list[str] = []

    class RetriableSink:
        def __init__(self, name: str, *, fail_once: bool) -> None:
            self.name = name
            self.fail_once = fail_once
            self.attempts = 0

        def publish(self, _snapshot: StateSnapshot) -> None:
            pass

        def close(self) -> None:
            self.attempts += 1
            close_calls.append(self.name)
            if self.fail_once and self.attempts == 1:
                raise RuntimeError(f"{self.name} close failed")

    failing = RetriableSink("failing", fail_once=True)
    stable = RetriableSink("stable", fail_once=False)
    composite = CompositeStateSink((failing, stable))

    with np.testing.assert_raises_regex(RuntimeError, "failing close failed"):
        composite.close()
    assert close_calls == ["failing", "stable"]

    composite.close()
    assert close_calls == ["failing", "stable", "failing"]


def test_state_publisher_records_composite_close_failure_in_status() -> None:
    class RetriableSink:
        def __init__(self, *, fail_once: bool) -> None:
            self.fail_once = fail_once
            self.attempts = 0

        def publish(self, _snapshot: StateSnapshot) -> None:
            pass

        def close(self) -> None:
            self.attempts += 1
            if self.fail_once and self.attempts == 1:
                raise RuntimeError("state sink close failed")

    failing = RetriableSink(fail_once=True)
    stable = RetriableSink(fail_once=False)
    publisher = StatePublisher(
        stream=StateStream(),
        sink=CompositeStateSink((failing, stable)),
    )

    assert publisher.close() is False
    status = publisher.status()
    assert status["error_count"] == 1
    assert status["last_error"] == "RuntimeError: state sink close failed"
    assert status["sink_closed"] is False
    assert failing.attempts == 1
    assert stable.attempts == 1

    assert publisher.close() is True
    assert publisher.status()["sink_closed"] is True
    assert failing.attempts == 2
    assert stable.attempts == 1


def test_state_publisher_start_failure_clears_unstarted_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingThread:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("state publisher start failed")

    monkeypatch.setattr(
        "linkerbot_sim.telemetry.foxglove_state.Thread",
        FailingThread,
    )
    publisher = StatePublisher(
        stream=StateStream(),
        sink=SimpleNamespace(publish=lambda _snapshot: None, close=lambda: None),
    )

    with np.testing.assert_raises_regex(RuntimeError, "state publisher start failed"):
        publisher.start()

    assert publisher.thread is None


def test_state_publisher_timeout_status_clears_after_successful_retry() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingSink:
        def publish(self, _snapshot: StateSnapshot) -> None:
            entered.set()
            release.wait(timeout=2.0)

        def close(self) -> None:
            pass

    stream = StateStream()
    publisher = StatePublisher(
        stream=stream,
        sink=BlockingSink(),
        worker_poll_interval_s=0.01,
    )
    publisher.start()
    stream.publish(_snapshot())
    assert entered.wait(timeout=1.0)

    assert publisher.close(timeout_s=0.01) is False
    assert publisher.status()["shutdown_timed_out"] is True

    release.set()
    assert publisher.close(timeout_s=1.0) is True
    assert publisher.status()["shutdown_timed_out"] is False
