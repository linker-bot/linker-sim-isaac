from __future__ import annotations

import json

import pytest

from linkerbot_sim.mirror.interface.protocol import (
    MIRROR_PROTOCOL,
    MIRROR_PROTOCOL_V2,
    MIRROR_PROTOCOL_V3,
    MIRROR_V1_OPERATIONS,
    MIRROR_V2_OPERATIONS,
    MIRROR_V3_OPERATIONS,
    MirrorResponse,
    decode_request,
    encode_response,
)


EXPECTED_OPERATIONS = {
    "motion.plan_timeline",
    "motion.joint_goal",
    "motion.joint_delta",
    "motion.joint_trajectory",
    "motion.plan_cspace_goal",
    "motion.plan_cspace_delta",
    "motion.ik_pose",
    "motion.ik_offset",
    "motion.plan_linear_pose_path",
    "motion.hold",
    "runtime.reset",
    "state.get",
    "state.set",
    "snapshot.get",
    "snapshot.set",
    "runtime.status",
    "queue.cancel",
    "queue.cancel_current",
    "runtime.estop",
    "runtime.quit",
}


def _request(**overrides: object) -> str:
    value = {
        "protocol": MIRROR_PROTOCOL,
        "request_id": "r-1",
        "operation": "runtime.status",
        "arguments": {},
        **overrides,
    }
    return json.dumps(value)


def test_mirror_v1_freezes_all_twenty_operations() -> None:
    assert set(MIRROR_V1_OPERATIONS) == EXPECTED_OPERATIONS
    assert len(MIRROR_V1_OPERATIONS) == 20


def test_mirror_v2_adds_only_control_and_effort_operations() -> None:
    assert MIRROR_V2_OPERATIONS == MIRROR_V1_OPERATIONS | {
        "control.get_mode",
        "control.set_mode",
        "motion.joint_effort",
    }
    assert len(MIRROR_V2_OPERATIONS) == 23
    for operation in MIRROR_V2_OPERATIONS - MIRROR_V1_OPERATIONS:
        request = decode_request(
            _request(protocol=MIRROR_PROTOCOL_V2, operation=operation)
        )
        assert request.protocol == MIRROR_PROTOCOL_V2
        assert request.operation == operation


def test_mirror_v3_adds_only_hybrid_operations() -> None:
    additions = {
        "control.get_hybrid_parameters",
        "control.set_hybrid_parameters",
        "control.tare_wrench",
        "motion.hybrid_force_position",
    }
    assert MIRROR_V3_OPERATIONS == MIRROR_V2_OPERATIONS | additions
    assert len(MIRROR_V3_OPERATIONS) == 27
    for operation in additions:
        request = decode_request(
            _request(protocol=MIRROR_PROTOCOL_V3, operation=operation)
        )
        assert request.protocol == MIRROR_PROTOCOL_V3
        assert request.operation == operation


@pytest.mark.parametrize("protocol", [MIRROR_PROTOCOL, MIRROR_PROTOCOL_V2])
@pytest.mark.parametrize(
    "operation",
    [
        "control.get_hybrid_parameters",
        "control.set_hybrid_parameters",
        "control.tare_wrench",
        "motion.hybrid_force_position",
    ],
)
def test_older_protocols_reject_v3_operations(protocol: str, operation: str) -> None:
    with pytest.raises(ValueError, match="does not support"):
        decode_request(_request(protocol=protocol, operation=operation))


@pytest.mark.parametrize(
    "operation",
    ["control.get_mode", "control.set_mode", "motion.joint_effort"],
)
def test_mirror_v1_rejects_v2_operations(operation: str) -> None:
    with pytest.raises(ValueError, match="does not support"):
        decode_request(_request(operation=operation))


def test_decode_request_returns_owned_strict_envelope() -> None:
    request = decode_request(
        _request(
            operation="motion.joint_delta",
            arguments={"robot_id": 0, "positions": [0.1, -0.2]},
        )
    )

    assert request.protocol == MIRROR_PROTOCOL
    assert request.request_id == "r-1"
    assert request.operation == "motion.joint_delta"
    assert request.arguments_dict() == {"robot_id": 0, "positions": [0.1, -0.2]}
    with pytest.raises(TypeError):
        request.arguments["late_mutation"] = True  # type: ignore[index]


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"status","id":"old"}',
        (
            '{"protocol":"linkerbot.mirror.v1","request_id":"r","operation":'
            '"runtime.status","arguments":{},"unknown":1}'
        ),
        (
            '{"protocol":"linkerbot.mirror.v1","request_id":"r","request_id":"r2",'
            '"operation":"runtime.status","arguments":{}}'
        ),
        (
            '{"protocol":"linkerbot.mirror.v1","request_id":"r","operation":'
            '"runtime.status","arguments":{"bad":NaN}}'
        ),
        (
            '{"protocol":"wrong","request_id":"r","operation":'
            '"runtime.status","arguments":{}}'
        ),
    ],
)
def test_decode_request_rejects_old_unknown_duplicate_and_nonfinite_json(
    payload: str,
) -> None:
    with pytest.raises(ValueError):
        decode_request(payload)


def test_response_has_exact_success_and_failure_shapes() -> None:
    success = MirrorResponse.success("r-1", {"ready": True})
    failure = MirrorResponse.failure(
        "r-2",
        code="invalid_arguments",
        message="bad input",
        details={"field": "x"},
    )

    assert json.loads(encode_response(success)) == {
        "protocol": MIRROR_PROTOCOL,
        "request_id": "r-1",
        "ok": True,
        "result": {"ready": True},
    }
    assert json.loads(encode_response(failure)) == {
        "protocol": MIRROR_PROTOCOL,
        "request_id": "r-2",
        "ok": False,
        "error": {
            "code": "invalid_arguments",
            "message": "bad input",
            "details": {"field": "x"},
        },
    }


def test_v2_response_preserves_request_protocol() -> None:
    response = MirrorResponse.success(
        "mode-1",
        {"active_mode": "position"},
        protocol=MIRROR_PROTOCOL_V2,
    )

    assert json.loads(encode_response(response))["protocol"] == MIRROR_PROTOCOL_V2
