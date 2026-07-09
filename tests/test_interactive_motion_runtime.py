from __future__ import annotations

import time
from threading import Thread

import numpy as np

from linkerbot_sim.app.motion.specs import (
    CSpaceGoalPlanMoveSpec,
    CumotionMoveSpec,
    DualHandMoveSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    RawJointSequenceMoveSpec,
    SpecifiedPathMoveSpec,
)
from linkerbot_sim.app.interactive.protocol import (
    parse_interactive_motion_message,
)
from linkerbot_sim.app.interactive.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.transports import handle_interactive_message
from linkerbot_sim.planning.requests import IKRequest, TaskSpacePath, TcpArcSegment
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


DEFAULT_TCP = {"left": "AR5V2_L_pinch_tcp", "right": "AR5V2_R_pinch_tcp"}


def test_parse_interactive_ik_pose_with_hand_overlay() -> None:
    command = parse_interactive_motion_message(
        {
            "type": "ik_pose",
            "side": "left",
            "position": [0.35, -0.2, 0.4],
            "duration_s": 1.0,
            "overlays": [
                {
                    "timing": "sync",
                    "left_hand": {
                        "joint_positions": {
                            "L6V1_L_hand_index_mcp_pitch": 0.7
                        }
                    },
                }
            ],
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    assert command.kind == "moves"
    move = command.moves[0]
    assert isinstance(move, CumotionMoveSpec)
    assert isinstance(move.request, IKRequest)
    assert move.tcp_frame_name == "AR5V2_L_pinch_tcp"
    assert move.overlays[0].left_hand is not None
    assert move.overlays[0].left_hand.duration_s == 1.0


def test_parse_interactive_orientation_defaults_to_rpy() -> None:
    rpy = [0.1, 0.2, -0.3]
    command = parse_interactive_motion_message(
        {
            "type": "ik_pose",
            "side": "left",
            "position": [0.35, -0.2, 0.4],
            "orientation": rpy,
            "duration_s": 1.0,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    move = command.moves[0]
    assert isinstance(move, CumotionMoveSpec)
    assert isinstance(move.request, IKRequest)
    np.testing.assert_allclose(
        move.request.target_orientation,
        rpy_xyz_to_quat_wxyz(rpy),
    )


def test_parse_interactive_explicit_quaternion_orientation() -> None:
    command = parse_interactive_motion_message(
        {
            "type": "ik_pose",
            "side": "left",
            "position": [0.35, -0.2, 0.4],
            "orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "duration_s": 1.0,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    move = command.moves[0]
    assert isinstance(move, CumotionMoveSpec)
    assert isinstance(move.request, IKRequest)
    np.testing.assert_allclose(
        move.request.target_orientation,
        [1.0, 0.0, 0.0, 0.0],
    )


def test_parse_interactive_rejects_ambiguous_orientation_fields() -> None:
    try:
        parse_interactive_motion_message(
            {
                "type": "ik_pose",
                "side": "left",
                "position": [0.35, -0.2, 0.4],
                "orientation": [0.0, 0.0, 0.0],
                "orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "duration_s": 1.0,
            },
            default_tcp_by_side=DEFAULT_TCP,
        )
    except ValueError as exc:
        assert "cannot both be set" in str(exc)
    else:
        raise AssertionError("expected ambiguous orientation fields to be rejected")


def test_parse_interactive_ik_offset_orientation_defaults_to_rpy() -> None:
    rpy = [0.0, 0.1, 0.2]
    command = parse_interactive_motion_message(
        {
            "type": "ik_offset",
            "side": "right",
            "offset": [0.0, 0.0, 0.02],
            "orientation_mode": "target",
            "orientation": rpy,
            "duration_s": 0.5,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    move = command.moves[0]
    assert isinstance(move, IkOffsetMoveSpec)
    np.testing.assert_allclose(
        move.target_orientation,
        rpy_xyz_to_quat_wxyz(rpy),
    )


def test_parse_interactive_task_space_orientation_defaults_to_rpy() -> None:
    rpy = [0.0, 0.0, 1.5707]
    command = parse_interactive_motion_message(
        {
            "type": "task_space_line",
            "side": "right",
            "target_offset": [0.0, 0.0, 0.05],
            "orientation_mode": "target",
            "target_orientation": rpy,
            "duration_s": 1.0,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    move = command.moves[0]
    assert isinstance(move, SpecifiedPathMoveSpec)
    segment = move.path.segments[0]
    np.testing.assert_allclose(
        segment.target_orientation,
        rpy_xyz_to_quat_wxyz(rpy),
    )


def test_parse_interactive_motion_modes() -> None:
    examples = [
        (
            {
                "type": "hand",
                "side": "left",
                "joint_positions": [0.1, 0.2],
                "duration_s": 0.5,
            },
            HandMoveSpec,
        ),
        (
            {
                "type": "dual_hand",
                "left": {"joint_positions": {"l_hand": 0.1}},
                "right": {"joint_positions": {"r_hand": 0.2}},
                "duration_s": 0.5,
            },
            DualHandMoveSpec,
        ),
        (
            {
                "type": "ik_offset",
                "side": "right",
                "offset": [0.0, 0.0, 0.02],
                "orientation_mode": "none",
                "duration_s": 0.5,
            },
            IkOffsetMoveSpec,
        ),
        (
            {
                "type": "cspace_goal",
                "side": "left",
                "joint_positions": [0.1, 0.2],
                "duration_s": 0.5,
            },
            CSpaceGoalPlanMoveSpec,
        ),
        (
            {
                "type": "task_space_arc",
                "side": "right",
                "target_offset": [0.0, 0.03, 0.0],
                "intermediate_offset": [0.0, 0.015, 0.01],
                "duration_s": 0.5,
            },
            SpecifiedPathMoveSpec,
        ),
    ]

    parsed = [
        parse_interactive_motion_message(message, default_tcp_by_side=DEFAULT_TCP)
        for message, _expected in examples
    ]

    for command, (_message, expected_type) in zip(parsed, examples):
        assert isinstance(command.moves[0], expected_type)
    arc_move = parsed[-1].moves[0]
    assert isinstance(arc_move, SpecifiedPathMoveSpec)
    assert isinstance(arc_move.path, TaskSpacePath)
    assert isinstance(arc_move.path.segments[0], TcpArcSegment)


def test_parse_batch_moves() -> None:
    command = parse_interactive_motion_message(
        {
            "id": "batch-1",
            "moves": [
                {
                    "type": "ik_offset",
                    "side": "left",
                    "offset": [0.01, 0.0, 0.0],
                    "duration_s": 0.4,
                },
                {
                    "type": "cspace_delta",
                    "side": "right",
                    "joint_deltas": [0.1],
                    "duration_s": 0.4,
                },
            ],
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    assert command.command_id == "batch-1"
    assert len(command.moves) == 2


def test_parse_interactive_reset_command() -> None:
    command = parse_interactive_motion_message(
        {
            "type": "reset",
            "id": "reset-a",
            "clear_queue": False,
            "hold_after_reset": False,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    assert command.kind == "reset"
    assert command.reset_id == "reset-a"
    assert command.reset_mode == "runtime"
    assert command.reset_clear_queue is False
    assert command.reset_hold_after_reset is False


def test_parse_interactive_snapshot_commands() -> None:
    get_command = parse_interactive_motion_message(
        {"type": "get_snapshot", "id": "snap-a"},
        default_tcp_by_side=DEFAULT_TCP,
    )
    set_command = parse_interactive_motion_message(
        {
            "type": "set_snapshot",
            "id": "snap-b",
            "snapshot": {"schema_version": "linkerbot.snapshot.v1", "robots": {}},
            "robot_map": {"left": "single"},
            "strict": False,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    assert get_command.kind == "get_snapshot"
    assert get_command.command_id == "snap-a"
    assert set_command.kind == "set_snapshot"
    assert set_command.command_id == "snap-b"
    assert set_command.snapshot_robot_map == {"left": "single"}
    assert set_command.snapshot_strict is False


def test_parse_raw_joint_sequence_matrix_and_mapping() -> None:
    command = parse_interactive_motion_message(
        {
            "id": "raw-1",
            "type": "raw_joint_sequence",
            "left": {
                "joint_positions": [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]
            },
            "right": {
                "joint_positions": {
                    "r0": [1.0, 1.1],
                    "r1": [2.0, 2.1],
                }
            },
            "step_interval": 2,
            "phase": "external_policy",
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    assert command.command_id == "raw-1"
    move = command.moves[0]
    assert isinstance(move, RawJointSequenceMoveSpec)
    assert move.step_interval == 2
    assert move.phase == "external_policy"
    assert move.left is not None
    assert move.left.joint_positions == ((0.1, 0.2), (0.3, 0.4))
    assert move.right is not None
    assert move.right.joint_positions["r0"] == (1.0, 1.1)


def test_interactive_queue_status_cancel_and_events() -> None:
    queue = InteractiveMotionQueue()
    events = []
    queue.add_listener(events.append)
    command = parse_interactive_motion_message(
        {
            "id": "cmd-a",
            "type": "hand",
            "side": "left",
            "joint_positions": [0.1],
            "duration_s": 0.2,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )

    queued = queue.submit(command)
    status = queue.status()
    cancelled = queue.cancel("cmd-a")

    assert queued.command_id == "cmd-a"
    assert status["commands"][0]["state"] == "pending"
    assert cancelled is True
    assert queue.status("cmd-a")["commands"][0]["state"] == "cancelled"
    assert events[0]["event"] == "accepted"
    assert events[-1]["event"] == "cancelled"


def test_interactive_queue_reset_cancels_pending_and_tracks_status() -> None:
    queue = InteractiveMotionQueue()
    events = []
    queue.add_listener(events.append)
    command = parse_interactive_motion_message(
        {
            "id": "cmd-a",
            "type": "hand",
            "side": "left",
            "joint_positions": [0.1],
            "duration_s": 0.2,
        },
        default_tcp_by_side=DEFAULT_TCP,
    )
    queue.submit(command)

    request = queue.request_reset(reset_id="reset-a")
    consumed = queue.consume_reset_request()
    status = queue.status()
    queue.mark_reset_done("reset-a", step=0)

    assert request.reset_id == "reset-a"
    assert consumed == request
    assert status["resetting"] is True
    assert status["last_reset"]["state"] == "requested"
    assert queue.status("cmd-a")["commands"][0]["state"] == "cancelled"
    assert events[-2]["event"] == "reset_requested"
    assert events[-1]["event"] == "reset_done"
    assert queue.status()["last_reset"]["state"] == "done"


def test_transport_snapshot_request_waits_for_simulation_response() -> None:
    queue = InteractiveMotionQueue()
    responses = []

    thread = Thread(
        target=lambda: responses.append(
            handle_interactive_message(
                message={"type": "get_snapshot", "id": "snap-a"},
                queue=queue,
                default_tcp_by_side=DEFAULT_TCP,
            )
        )
    )
    thread.start()
    request = None
    for _ in range(100):
        request = queue.consume_snapshot_request()
        if request is not None:
            break
        time.sleep(0.01)
    assert request is not None
    assert request.snapshot_id == "snap-a"
    assert request.kind == "get_snapshot"

    queue.mark_snapshot_done(
        request,
        {
            "event": "snapshot",
            "accepted": True,
            "snapshot": {"robots": {}},
        },
    )
    thread.join(timeout=1.0)

    assert responses == [
        {
            "event": "snapshot",
            "accepted": True,
            "snapshot": {"robots": {}},
            "id": "snap-a",
        }
    ]


def test_transport_status_and_cancel_current_responses() -> None:
    queue = InteractiveMotionQueue()

    submit = handle_interactive_message(
        message={
            "type": "hand",
            "side": "left",
            "joint_positions": [0.1],
            "duration_s": 0.2,
        },
        queue=queue,
        default_tcp_by_side=DEFAULT_TCP,
    )
    running = queue.next_pending(timeout_s=0.01)
    cancel = handle_interactive_message(
        message={"type": "cancel_current"},
        queue=queue,
        default_tcp_by_side=DEFAULT_TCP,
    )

    assert submit["event"] == "accepted"
    assert running is not None
    assert cancel["accepted"] is True
    assert queue.should_stop_current() is True


def test_transport_reset_response() -> None:
    queue = InteractiveMotionQueue()

    response = handle_interactive_message(
        message={"type": "reset", "id": "reset-a"},
        queue=queue,
        default_tcp_by_side=DEFAULT_TCP,
    )

    assert response["event"] == "reset"
    assert response["accepted"] is True
    assert response["id"] == "reset-a"
    assert queue.consume_reset_request().reset_id == "reset-a"
