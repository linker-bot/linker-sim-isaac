from __future__ import annotations

from collections.abc import Mapping
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import pytest

from linkerbot_sim.configuration.catalog import load_mirror_config
from linkerbot_sim.mirror.controller import MirrorController
from linkerbot_sim.mirror.controller import _OUT_OF_BAND_OPERATIONS
from linkerbot_sim.mirror.control_mode import MirrorControlModeService
from linkerbot_sim.mirror.interface.admission import (
    AdmissionCapacityError,
    DuplicateRequestError,
    MirrorAdmissionQueue,
)
from linkerbot_sim.mirror.interface.protocol import (
    MIRROR_PROTOCOL,
    MIRROR_PROTOCOL_V2,
    MIRROR_PROTOCOL_V3,
    MirrorRequest,
    MirrorResponse,
)
from linkerbot_sim.mirror.hybrid_parameters import HybridParameterService
from linkerbot_sim.mirror.motion import MirrorMotionOwner
from linkerbot_sim.mirror.reset import MirrorResetService
from linkerbot_sim.mirror.snapshot import MirrorSnapshotService
from linkerbot_sim.mirror.state import MirrorStateService


def _request(
    envelope_id: str,
    operation: str,
    *,
    protocol: str = MIRROR_PROTOCOL,
    **arguments: object,
) -> MirrorRequest:
    return MirrorRequest(
        protocol=protocol,
        request_id=envelope_id,
        operation=operation,
        arguments=arguments,
    )


class _Motion:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], str, str]] = []

    def execute(
        self,
        operation: str,
        arguments: Mapping[str, object],
        *,
        request_id: str,
        should_cancel: object,
        protocol: str,
    ) -> dict[str, object]:
        self.calls.append((operation, dict(arguments), request_id, protocol))
        return {"operation": operation, "request_id": request_id}

    def close(self) -> bool:
        return True

    def tare_wrench(
        self,
        arguments: Mapping[str, object],
        *,
        request_id: str,
        should_cancel: object,
    ) -> dict[str, object]:
        del should_cancel
        self.calls.append(
            ("control.tare_wrench", dict(arguments), request_id, MIRROR_PROTOCOL_V3)
        )
        return {"event": "wrench_tared", "tare_generation": 1}


def _controller() -> tuple[MirrorController, _Motion]:
    motion = _Motion()
    state = {"value": 1}
    controller = MirrorController(
        admission=MirrorAdmissionQueue(capacity=2, terminal_capacity=4),
        motion=MirrorMotionOwner(motion),
        state=MirrorStateService(
            getter=lambda: state,
            setter=lambda value, strict=True: state.update(value),
        ),
        snapshots=MirrorSnapshotService(
            capture=lambda: {"schema": "test", "value": state["value"]},
            restore=lambda value, **_kwargs: state.update(value),
        ),
        reset_service=MirrorResetService(
            lambda hold_after_reset=True: {"hold": hold_after_reset}
        ),
    )
    controller.bind_status_provider(lambda: {"mode": "mirror"})
    control_mode = MirrorControlModeService(initial_mode="position", bindings=())
    control_mode.bind_runtime(SimpleNamespace(fatal_error=None))
    controller.control_mode = control_mode
    return controller, motion


def test_admission_is_bounded_and_rejects_duplicate_ids() -> None:
    queue = MirrorAdmissionQueue(capacity=1, terminal_capacity=1)
    request = _request("r-1", "runtime.status")
    queue.submit(request)

    with pytest.raises(DuplicateRequestError):
        queue.submit(request)
    with pytest.raises(AdmissionCapacityError):
        queue.submit(_request("r-2", "runtime.status"))


def test_controller_dispatches_motion_reset_snapshot_and_status() -> None:
    controller, motion = _controller()

    motion_response = controller.dispatch(
        _request("m-1", "motion.joint_delta", robot_id=0, positions=[0.1])
    )
    reset_response = controller.dispatch(
        _request("reset-1", "runtime.reset", hold_after_reset=False)
    )
    snapshot_response = controller.dispatch(_request("s-1", "snapshot.get"))
    status_response = controller.dispatch(_request("status-1", "runtime.status"))

    assert motion_response.ok is True
    assert motion.calls == [
        (
            "motion.joint_delta",
            {"robot_id": 0, "positions": [0.1]},
            "m-1",
            MIRROR_PROTOCOL,
        )
    ]
    assert reset_response.result["reset"] == {"hold": False}  # type: ignore[index]
    assert snapshot_response.result == {"schema": "test", "value": 1}
    assert status_response.result["mode"] == "mirror"  # type: ignore[index]
    assert "queue" in status_response.result  # type: ignore[operator]


def test_v2_control_get_set_are_queued_and_echo_protocol() -> None:
    controller, _motion = _controller()
    assert "control.set_mode" not in _OUT_OF_BAND_OPERATIONS

    get_response = controller.dispatch(
        _request("mode-get", "control.get_mode", protocol=MIRROR_PROTOCOL_V2)
    )
    set_response = controller.dispatch(
        _request(
            "mode-set",
            "control.set_mode",
            protocol=MIRROR_PROTOCOL_V2,
            mode="velocity",
            expected_generation=0,
        )
    )

    assert get_response.protocol == MIRROR_PROTOCOL_V2
    assert get_response.result["active_mode"] == "position"  # type: ignore[index]
    assert set_response.protocol == MIRROR_PROTOCOL_V2
    assert set_response.result == {
        "previous_mode": "position",
        "active_mode": "velocity",
        "generation": 1,
        "changed": True,
    }


def test_v3_hybrid_parameters_tare_and_motion_share_queued_dispatch() -> None:
    controller, motion = _controller()
    settings = load_mirror_config("physx_cpu_hybrid").hybrid_control
    assert settings is not None
    controller.hybrid_parameters = HybridParameterService(settings)

    initial = controller.dispatch(
        _request(
            "hybrid-get",
            "control.get_hybrid_parameters",
            protocol=MIRROR_PROTOCOL_V3,
        )
    )
    updated = controller.dispatch(
        _request(
            "hybrid-set",
            "control.set_hybrid_parameters",
            protocol=MIRROR_PROTOCOL_V3,
            expected_generation=0,
            posture_stiffness=4.0,
        )
    )
    tared = controller.dispatch(
        _request(
            "hybrid-tare",
            "control.tare_wrench",
            protocol=MIRROR_PROTOCOL_V3,
            robot_id=0,
            tcp_frame_name="tcp",
            reference_frame="world",
        )
    )
    moved = controller.dispatch(
        _request(
            "hybrid-motion",
            "motion.hybrid_force_position",
            protocol=MIRROR_PROTOCOL_V3,
            opaque="backend-parser-owned",
        )
    )

    assert initial.protocol == MIRROR_PROTOCOL_V3
    assert initial.result["generation"] == 0  # type: ignore[index]
    assert updated.result["generation"] == 1  # type: ignore[index]
    assert tared.result == {"event": "wrench_tared", "tare_generation": 1}
    assert moved.ok is True
    assert motion.calls[-2:] == [
        (
            "control.tare_wrench",
            {"robot_id": 0, "tcp_frame_name": "tcp", "reference_frame": "world"},
            "hybrid-tare",
            MIRROR_PROTOCOL_V3,
        ),
        (
            "motion.hybrid_force_position",
            {"opaque": "backend-parser-owned"},
            "hybrid-motion",
            MIRROR_PROTOCOL_V3,
        ),
    ]


@pytest.mark.parametrize("invalid", [True, -1, 1.0, "1", None])
def test_control_set_rejects_invalid_expected_generation(invalid: object) -> None:
    controller, _motion = _controller()
    response = controller.dispatch(
        _request(
            "mode-invalid",
            "control.set_mode",
            protocol=MIRROR_PROTOCOL_V2,
            mode="velocity",
            expected_generation=invalid,
        )
    )

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "invalid_arguments"


def test_v2_synthetic_cancel_estop_and_close_responses_echo_protocol() -> None:
    queue = MirrorAdmissionQueue(capacity=4, terminal_capacity=8)
    cancel_request = _request(
        "cancelled-v2", "control.get_mode", protocol=MIRROR_PROTOCOL_V2
    )
    estop_request = _request(
        "estopped-v2", "control.get_mode", protocol=MIRROR_PROTOCOL_V2
    )
    close_request = _request(
        "closed-v2", "control.get_mode", protocol=MIRROR_PROTOCOL_V2
    )

    queue.submit(cancel_request)
    assert queue.cancel(cancel_request.request_id)
    assert queue.wait_response("cancelled-v2", timeout_s=0.1).protocol == (
        MIRROR_PROTOCOL_V2
    )
    queue.submit(estop_request)
    queue.estop()
    assert queue.wait_response("estopped-v2", timeout_s=0.1).protocol == (
        MIRROR_PROTOCOL_V2
    )
    queue.clear_estop()
    queue.submit(close_request)
    queue.close()
    assert queue.wait_response("closed-v2", timeout_s=0.1).protocol == (
        MIRROR_PROTOCOL_V2
    )


def test_controller_dispatches_state_get_and_strict_state_set() -> None:
    controller, _motion = _controller()
    setter_calls: list[tuple[dict[str, object], bool]] = []

    def set_state(value: Mapping[str, object], *, strict: bool = True) -> object:
        setter_calls.append((dict(value), strict))
        return {"accepted": True, "strict": strict}

    controller.state.setter = set_state

    before = controller.dispatch(_request("state-get-1", "state.get"))
    updated = controller.dispatch(
        _request(
            "state-set-1",
            "state.set",
            state={"value": 2, "nested": {"positions": [0.1, -0.2]}},
            strict=False,
        )
    )

    assert before.ok is True
    assert before.result == {"value": 1}
    assert updated.ok is True
    assert updated.result == {"accepted": True, "strict": False}
    assert setter_calls == [
        (
            {"value": 2, "nested": {"positions": [0.1, -0.2]}},
            False,
        )
    ]


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("state.get", {"unexpected": True}),
        ("state.set", {}),
        ("state.set", {"state": []}),
        ("state.set", {"state": {}, "strict": 1}),
        ("state.set", {"state": {}, "unexpected": True}),
    ],
)
def test_state_operations_reject_invalid_arguments_before_adapter_access(
    operation: str,
    arguments: dict[str, object],
) -> None:
    controller, _motion = _controller()
    adapter_calls: list[str] = []
    controller.state.getter = lambda: adapter_calls.append("get") or {"value": 1}
    controller.state.setter = lambda _value, strict=True: adapter_calls.append(
        f"set:{strict}"
    )

    response = controller.dispatch(_request("state-invalid", operation, **arguments))

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "invalid_arguments"
    assert adapter_calls == []


def test_queued_state_operations_only_touch_adapter_on_consumer_thread() -> None:
    controller, _motion = _controller()
    consumer_thread_id = get_ident()
    adapter_threads: list[tuple[str, int]] = []
    controller.state.getter = lambda: (
        adapter_threads.append(("get", get_ident())) or {"value": 1}
    )
    controller.state.setter = lambda _value, strict=True: (
        adapter_threads.append((f"set:{strict}", get_ident())) or {"accepted": True}
    )

    for request in (
        _request("thread-state-get", "state.get"),
        _request("thread-state-set", "state.set", state={"value": 2}),
    ):
        ingress = Thread(target=controller.admission.submit, args=(request,))
        ingress.start()
        ingress.join(timeout=1.0)
        assert not ingress.is_alive()
        response = controller.process_next(timeout_s=0.1)
        assert response is not None and response.ok is True

    assert adapter_threads == [
        ("get", consumer_thread_id),
        ("set:True", consumer_thread_id),
    ]


def test_out_of_band_cancel_is_immediate_and_request_id_is_still_unique() -> None:
    controller, _motion = _controller()
    pending = _request("pending", "snapshot.get")
    controller.admission.submit(pending)
    cancel = _request("cancel-1", "queue.cancel", request_id="pending")

    first = controller.submit_and_wait(cancel, timeout_s=0.1)
    duplicate = controller.submit_and_wait(cancel, timeout_s=0.1)
    pending_response = controller.admission.wait_response("pending", timeout_s=0.1)

    assert first.ok is True
    assert first.result == {"cancelled": True, "request_id": "pending"}
    assert duplicate.ok is False
    assert duplicate.error is not None
    assert duplicate.error.code == "duplicate_request_id"
    assert pending_response.ok is False
    assert pending_response.error is not None
    assert pending_response.error.code == "cancelled"


def test_snapshot_set_rejects_unknown_argument_before_mutation() -> None:
    controller, _motion = _controller()
    response = controller.dispatch(
        _request(
            "s-2",
            "snapshot.set",
            snapshot={"schema": "test"},
            unexpected=True,
        )
    )

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "invalid_arguments"


def test_estop_rejects_new_motion_until_successful_reset() -> None:
    controller, _motion = _controller()
    estop = controller.submit_and_wait(_request("e-1", "runtime.estop"), timeout_s=0.1)
    blocked = controller.submit_and_wait(
        _request("m-blocked", "motion.hold", robot_id=0, duration_s=0.1),
        timeout_s=0.1,
    )
    reset = controller.dispatch(_request("reset-after-estop", "runtime.reset"))

    assert estop.ok is True
    assert blocked.ok is False
    assert blocked.error is not None
    assert blocked.error.code == "runtime_estopped"
    assert reset.ok is True
    controller.admission.submit(
        _request("m-allowed", "motion.hold", robot_id=0, duration_s=0.1)
    )


def test_estop_allows_state_get_but_rejects_state_set_without_clearing_latch() -> None:
    controller, _motion = _controller()
    estop = controller.submit_and_wait(
        _request("state-estop", "runtime.estop"), timeout_s=0.1
    )

    controller.admission.submit(_request("state-read-stopped", "state.get"))
    read = controller.process_next(timeout_s=0.1)
    blocked_at_admission = controller.submit_and_wait(
        _request("state-write-stopped", "state.set", state={"value": 2}),
        timeout_s=0.1,
    )
    blocked_direct = controller.dispatch(
        _request("state-write-direct-stopped", "state.set", state={"value": 3})
    )

    assert estop.ok is True
    assert read is not None and read.ok is True
    assert read.result == {"value": 1}
    for response in (blocked_at_admission, blocked_direct):
        assert response.ok is False
        assert response.error is not None
        assert response.error.code == "runtime_estopped"
    assert controller.admission.status().estopped is True


def test_estop_keeps_existing_snapshot_set_cold_restore_semantics() -> None:
    controller, _motion = _controller()
    controller.submit_and_wait(
        _request("snapshot-estop", "runtime.estop"), timeout_s=0.1
    )
    controller.admission.submit(
        _request(
            "snapshot-write-stopped",
            "snapshot.set",
            snapshot={"value": 4},
            strict=True,
        )
    )

    restored = controller.process_next(timeout_s=0.1)
    state = controller.dispatch(_request("snapshot-state-read", "state.get"))

    assert restored is not None and restored.ok is True
    assert state.ok is True
    assert state.result == {"value": 4}
    assert controller.admission.status().estopped is True


def test_estop_does_not_preempt_an_active_atomic_state_set_or_clear_latch() -> None:
    controller, _motion = _controller()
    entered_setter = Event()
    release_setter = Event()
    state_responses: list[MirrorResponse | None] = []

    def set_state(_value: Mapping[str, object], *, strict: bool = True) -> object:
        entered_setter.set()
        if not release_setter.wait(timeout=1.0):
            raise RuntimeError("test did not release state setter")
        return {"accepted": True, "strict": strict}

    controller.state.setter = set_state
    controller.admission.submit(
        _request("state-active", "state.set", state={"value": 2})
    )
    consumer = Thread(
        target=lambda: state_responses.append(controller.process_next(timeout_s=0.1))
    )
    consumer.start()
    assert entered_setter.wait(timeout=1.0)

    estop = controller.submit_and_wait(
        _request("state-active-estop", "runtime.estop"), timeout_s=0.1
    )
    release_setter.set()
    consumer.join(timeout=1.0)

    assert not consumer.is_alive()
    assert estop.ok is True
    assert len(state_responses) == 1
    state_response = state_responses[0]
    assert state_response is not None
    assert state_response.ok is True
    assert controller.admission.status().estopped is True
