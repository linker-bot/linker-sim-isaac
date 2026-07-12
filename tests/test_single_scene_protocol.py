from __future__ import annotations

import pytest

from linkerbot_sim.app.interactive.single_scene.protocol import (
    parse_interactive_motion_message,
)
from linkerbot_sim.configs.runtime import (
    PlannerRequestDefaults,
    RuntimeCommandDefaults,
)


def test_single_scene_plan_requires_robot_id_and_defaults_pose_frame() -> None:
    with pytest.raises(ValueError, match="robot_id"):
        parse_interactive_motion_message(
            {
                "type": "ik_offset",
                "offset": [0.0, 0.0, 0.1],
                "offset_frame": "robot_base",
                "duration_s": 1.0,
            },
        )
    command = parse_interactive_motion_message(
        {
            "type": "ik_offset",
            "robot_id": 0,
            "offset": [0.0, 0.0, 0.1],
            "duration_s": 1.0,
        },
    )
    assert command.timeline is not None
    segment = command.timeline.tracks[0].units[0].group_tracks[0].segments[0]
    assert segment.offset_frame == "env"


def test_single_scene_request_defaults_resolve_before_queue_and_explicit_values_win() -> (
    None
):
    planner_defaults = PlannerRequestDefaults(
        duration_s=2.5,
        avoid_collisions=True,
        force_collision_refresh=True,
        coordination="static_others",
    )
    command_defaults = RuntimeCommandDefaults(
        joint_interpolation="linear",
        pose_frame="world",
        orientation_mode="current",
    )
    inherited = parse_interactive_motion_message(
        {
            "type": "plan_linear_pose_path",
            "robot_id": 0,
            "target_position": [0.1, 0.2, 0.3],
        },
        planner_defaults=planner_defaults,
        command_defaults=command_defaults,
    )
    explicit = parse_interactive_motion_message(
        {
            "type": "plan_linear_pose_path",
            "robot_id": 0,
            "target_position": [0.1, 0.2, 0.3],
            "duration_s": 0.4,
            "reference_frame": "env",
            "orientation_mode": "free",
            "avoid_collisions": False,
            "force_collision_refresh": False,
            "coordination": "independent",
        },
        planner_defaults=planner_defaults,
        command_defaults=command_defaults,
    )

    assert inherited.timeline is not None
    inherited_segment = (
        inherited.timeline.tracks[0].units[0].group_tracks[0].segments[0]
    )
    assert inherited.timeline.coordination == "static_others"
    assert inherited.timeline.force_collision_refresh is True
    assert inherited_segment.duration_s == pytest.approx(2.5)
    assert inherited_segment.reference_frame == "world"
    assert inherited_segment.orientation_mode == "current"
    assert inherited_segment.avoid_collisions is True
    assert inherited_segment.interpolation == "linear"

    assert explicit.timeline is not None
    explicit_segment = explicit.timeline.tracks[0].units[0].group_tracks[0].segments[0]
    assert explicit.timeline.coordination == "independent"
    assert explicit.timeline.force_collision_refresh is False
    assert explicit_segment.duration_s == pytest.approx(0.4)
    assert explicit_segment.reference_frame == "env"
    assert explicit_segment.orientation_mode == "free"
    assert explicit_segment.avoid_collisions is False


def test_single_scene_orientation_default_invalid_combination_is_rejected_after_resolve() -> (
    None
):
    with pytest.raises(ValueError, match="orientation_mode='target'"):
        parse_interactive_motion_message(
            {
                "type": "plan_linear_pose_path",
                "robot_id": 0,
                "target_position": [0.1, 0.2, 0.3],
            },
            command_defaults=RuntimeCommandDefaults(orientation_mode="target"),
        )


def test_single_scene_explicit_target_orientation_implies_target_mode() -> None:
    command = parse_interactive_motion_message(
        {
            "type": "plan_linear_pose_path",
            "robot_id": 0,
            "target_position": [0.1, 0.2, 0.3],
            "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        command_defaults=RuntimeCommandDefaults(orientation_mode="current"),
    )

    assert command.timeline is not None
    segment = command.timeline.tracks[0].units[0].group_tracks[0].segments[0]
    assert segment.orientation_mode == "target"
    assert segment.target_orientation_wxyz == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("orientation_mode", ("free", "current"))
def test_single_scene_rejects_explicit_non_target_mode_with_target_orientation(
    orientation_mode: str,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_interactive_motion_message(
            {
                "type": "plan_linear_pose_path",
                "robot_id": 0,
                "target_position": [0.1, 0.2, 0.3],
                "orientation_mode": orientation_mode,
                "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            command_defaults=RuntimeCommandDefaults(orientation_mode="target"),
        )


@pytest.mark.parametrize(
    "message",
    (
        {
            "type": "plan_timeline",
            "tracks": [
                {
                    "robot_id": 0,
                    "segments": [{"kind": "hold", "duration_s": 0.1}],
                }
            ],
            "coordinaton": "independent",
        },
        {
            "type": "plan_timeline",
            "tracks": [
                {
                    "robot_id": 0,
                    "segments": [{"kind": "hold", "duration_s": 0.1}],
                    "gruop": "arm",
                }
            ],
        },
        {
            "type": "plan_timeline",
            "tracks": [
                {
                    "robot_id": 0,
                    "units": [
                        {
                            "group_tracks": [
                                {
                                    "group": "arm",
                                    "segments": [{"kind": "hold", "duration_s": 0.1}],
                                }
                            ],
                            "group_traks": [],
                        }
                    ],
                }
            ],
        },
        {
            "type": "plan_timeline",
            "tracks": [
                {
                    "robot_id": 0,
                    "units": [
                        {
                            "group_tracks": [
                                {
                                    "group": "arm",
                                    "segments": [{"kind": "hold", "duration_s": 0.1}],
                                    "segmnts": [],
                                }
                            ]
                        }
                    ],
                }
            ],
        },
        {
            "type": "plan_linear_pose_path",
            "robot_id": 0,
            "target_position": [0.1, 0.2, 0.3],
            "duraton_s": 9.0,
        },
    ),
)
def test_single_scene_protocol_rejects_unknown_fields_at_every_timeline_level(
    message: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        parse_interactive_motion_message(message)


def test_timeline_rejects_duplicate_ids_and_coupled_backend() -> None:
    track = {
        "robot_id": 0,
        "segments": [{"kind": "hold", "duration_s": 0.1}],
    }
    with pytest.raises(ValueError, match="duplicate robot IDs"):
        parse_interactive_motion_message(
            {"type": "plan_timeline", "tracks": [track, track]},
        )
    with pytest.raises(RuntimeError, match="coupled"):
        parse_interactive_motion_message(
            {
                "type": "plan_timeline",
                "coordination": "coupled",
                "tracks": [track],
            },
        )


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_interactive_motion_message(
            {
                "type": "unknown_plan",
                "robot_id": 7,
                "offset": [0.0, 0.1, 0.0],
                "offset_frame": "world",
                "duration_s": 0.5,
            }
        )


def test_single_scene_planning_segment_accepts_sample_dt() -> None:
    command = parse_interactive_motion_message(
        {
            "type": "plan_cspace_goal",
            "robot_id": 0,
            "joint_positions": [1.0],
            "duration_s": 0.2,
            "sample_dt_s": 0.05,
        }
    )

    segment = command.timeline.tracks[0].units[0].group_tracks[0].segments[0]
    assert segment.sample_dt_s == 0.05


def test_single_scene_direct_segment_rejects_sample_dt() -> None:
    with pytest.raises(ValueError, match="only supported by planning kinds"):
        parse_interactive_motion_message(
            {
                "type": "joint_goal",
                "robot_id": 0,
                "joint_positions": [1.0],
                "duration_s": 0.2,
                "sample_dt_s": 0.05,
            }
        )


def test_single_scene_segments_reject_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported timeline segment kind"):
        parse_interactive_motion_message(
            {
                "type": "plan_timeline",
                "tracks": [
                    {
                        "robot_id": 0,
                        "segments": [
                            {
                                "kind": "unknown_kind",
                                "duration_s": 0.1,
                            }
                        ],
                    }
                ],
            }
        )


def test_snapshot_uses_label_map_and_rejects_unknown_fields() -> None:
    snapshot = {
        "schema": "linkerbot.snapshot",
        "robots": [
            {
                "robot_id": 0,
                "label": "source_robot",
                "joint_names": ["j0"],
                "joint_positions": [0.0],
                "joint_velocities": [0.0],
                "command_joint_names": [],
            }
        ],
        "objects": {},
    }

    command = parse_interactive_motion_message(
        {
            "type": "set_snapshot",
            "snapshot": snapshot,
            "label_map": {"source_robot": "target_robot"},
        }
    )
    assert command.snapshot_label_map == {"source_robot": "target_robot"}

    with pytest.raises(ValueError, match="unsupported fields"):
        parse_interactive_motion_message(
            {
                "type": "set_snapshot",
                "snapshot": snapshot,
                "unknown_selector": {"source_robot": "target_robot"},
            }
        )


@pytest.mark.parametrize(
    ("message", "error"),
    (
        (
            {
                "type": "plan_timeline",
                "tracks": (
                    {
                        "robot_id": 0,
                        "segments": [{"kind": "hold", "duration_s": 0.1}],
                    },
                ),
            },
            "tracks must be a list",
        ),
        (
            {
                "type": "plan_timeline",
                "tracks": [{"robot_id": 0, "units": ()}],
            },
            "units must be a list",
        ),
        (
            {
                "type": "plan_timeline",
                "tracks": [
                    {
                        "robot_id": 0,
                        "units": [{"group_tracks": ()}],
                    }
                ],
            },
            "group_tracks must be a list",
        ),
        (
            {
                "type": "plan_timeline",
                "tracks": [{"robot_id": 0, "segments": ()}],
            },
            "segments must be a list",
        ),
    ),
)
def test_single_scene_timeline_containers_require_json_arrays(
    message: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        parse_interactive_motion_message(message)


@pytest.mark.parametrize("invalid", ("true", 1, None))
def test_single_scene_control_booleans_require_json_booleans(invalid: object) -> None:
    with pytest.raises(ValueError, match="reset.clear_queue must be a boolean"):
        parse_interactive_motion_message({"type": "reset", "clear_queue": invalid})


@pytest.mark.parametrize("invalid", ("0.1", True, None))
def test_single_scene_duration_requires_json_number(invalid: object) -> None:
    with pytest.raises(ValueError, match="duration_s must be a number"):
        parse_interactive_motion_message(
            {
                "type": "hold",
                "robot_id": 0,
                "duration_s": invalid,
            }
        )


def test_single_scene_hold_uses_timeline_segment_duration() -> None:
    command = parse_interactive_motion_message(
        {"type": "hold", "robot_id": 0, "duration_s": 0.1}
    )
    segment = command.timeline.tracks[0].units[0].group_tracks[0].segments[0]
    assert segment.duration_s == pytest.approx(0.1)
