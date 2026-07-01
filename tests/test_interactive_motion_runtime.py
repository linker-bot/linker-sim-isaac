from __future__ import annotations

from linkerbot_sim.app.motion.specs import (
    CSpaceGoalPlanMoveSpec,
    CumotionMoveSpec,
    DualHandMoveSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    SpecifiedPathMoveSpec,
)
from linkerbot_sim.app.interactive.protocol import (
    parse_interactive_motion_message,
)
from linkerbot_sim.app.interactive.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.transports import handle_interactive_message
from linkerbot_sim.planning.requests import IKRequest, TaskSpacePath, TcpArcSegment


DEFAULT_TCP = {"left": "left_demo_tcp", "right": "right_demo_tcp"}


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
    assert move.tcp_frame_name == "left_demo_tcp"
    assert move.overlays[0].left_hand is not None
    assert move.overlays[0].left_hand.duration_s == 1.0


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
