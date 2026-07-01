from __future__ import annotations

import numpy as np

from linkerbot_sim.app.motion.specs import (  # noqa: E402
    CartesianTcpFrameSpec,
    CSpaceDeltaPlanMoveSpec,
    CSpaceGoalPlanMoveSpec,
    CommandOverlaySpec,
    CumotionMoveSpec,
    DualArmTcpSpec,
    DualHandMoveSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    SpecifiedPathMoveSpec,
    default_move_phase,
    specified_path_planner_config,
    tcp_transform_from_spec,
    tcp_transforms_from_dual_spec,
)
from linkerbot_sim.app.motion.dual_arm import (  # noqa: E402
    dual_cspace_goal_to_command,
    dual_cspace_linear_trajectory,
    dual_cspace_vector_from_side_commands,
    dual_arm_cumotion_summary,
    load_dual_arm_semantic_config,
    side_joint_delta_goal,
)
from linkerbot_sim.backends.cumotion.motion_planner_config import (  # noqa: E402
    MotionPlannerBackendConfig,
)
from linkerbot_sim.planning.dual_arm_cspace_partition import DualArmJointPartitions  # noqa: E402
from linkerbot_sim.planning.requests import (  # noqa: E402
    CSpaceWaypointPath,
    IKRequest,
    SpecifiedPathRequest,
    TaskSpacePath,
    TcpLineSegment,
)


def test_cartesian_tcp_frame_spec_converts_to_tcp_transform() -> None:
    spec = CartesianTcpFrameSpec(
        frame_name="tool_tcp",
        xyz=(0.1, 0.2, 0.3),
        rpy=(0.0, 0.1, 0.2),
    )

    transform = tcp_transform_from_spec(spec)

    assert transform.frame_name == "tool_tcp"
    np.testing.assert_allclose(transform.xyz, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(transform.rpy, [0.0, 0.1, 0.2])


def test_dual_arm_tcp_spec_rejects_duplicate_frame_names() -> None:
    tcp = DualArmTcpSpec(
        left=CartesianTcpFrameSpec("shared_tcp"),
        right=CartesianTcpFrameSpec("shared_tcp"),
    )

    try:
        tcp_transforms_from_dual_spec(tcp)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("expected duplicate TCP names to be rejected")


def test_default_move_phase_is_optional_and_stable() -> None:
    move = IkOffsetMoveSpec(
        side="left",
        tcp_frame_name="left_tcp",
        tcp_offset=(0.0, 0.0, 0.01),
        duration_s=0.5,
    )

    assert default_move_phase(move, side="left") == "dual_left_ik"
    assert default_move_phase(move) == "cumotion_ik"
    assert default_move_phase(move, dual_cspace=True) == "dual_cspace_ik"


def test_specified_path_planner_config_selects_family_from_path_type() -> None:
    base = MotionPlannerBackendConfig.from_mapping(
        {"planning_pipeline": "graph_search"}
    )
    task_path = TaskSpacePath(
        segments=(
            TcpLineSegment(
                target_offset=np.asarray([0.0, 0.02, 0.0], dtype=float),
                orientation_mode="current",
            ),
        )
    )

    task_config = specified_path_planner_config(base, path=task_path)
    cspace_config = specified_path_planner_config(
        base,
        path=CSpaceWaypointPath(
            waypoints=(
                np.asarray([0.0, 0.0], dtype=float),
                np.asarray([0.1, 0.2], dtype=float),
            )
        ),
    )

    assert task_config.planning_pipeline == "specified_path"
    assert task_config.specified_path.family == "task_space_segments"
    assert task_config.graph_search == base.graph_search
    assert cspace_config.specified_path.family == "cspace_waypoints"


def test_move_specs_validate_side_only_when_dual_runtime_requires_it() -> None:
    single_move = CSpaceDeltaPlanMoveSpec(
        joint_deltas=(0.1,),
        duration_s=0.5,
    )
    single_move.validate(require_side=False)

    try:
        single_move.validate(require_side=True)
    except ValueError as exc:
        assert "side" in str(exc)
    else:
        raise AssertionError("expected dual-arm move without side to be rejected")


def test_hand_move_specs_and_overlays_validate() -> None:
    left_hand = HandMoveSpec(
        side="left",
        joint_positions={"L6V1_L_hand_index_mcp_pitch": 0.7},
        duration_s=0.5,
    )
    right_hand = HandMoveSpec(
        side="right",
        joint_positions=(0.2, 0.3),
        duration_s=0.5,
    )
    overlay = CommandOverlaySpec(
        timing="sync",
        left_hand=HandMoveSpec(
            side="left",
            joint_positions={"L6V1_L_hand_index_mcp_pitch": 0.4},
        ),
    )
    arm_move = CSpaceGoalPlanMoveSpec(
        side="left",
        tcp_frame_name="left_tcp",
        joint_positions=(0.1, 0.2),
        duration_s=1.0,
        overlays=(overlay,),
    )

    left_hand.validate()
    DualHandMoveSpec(left=left_hand, right=right_hand, duration_s=0.5).validate()
    arm_move.validate(require_side=True)


def test_ik_offset_accepts_target_orientation_mode() -> None:
    move = IkOffsetMoveSpec(
        side="right",
        tcp_frame_name="right_tcp",
        tcp_offset=(0.0, 0.0, 0.01),
        duration_s=0.5,
        orientation_mode="target",
        target_orientation=(1.0, 0.0, 0.0, 0.0),
    )

    move.validate(require_side=True)


def test_cumotion_move_spec_accepts_optional_phase() -> None:
    move = CumotionMoveSpec(
        side="left",
        tcp_frame_name="left_tcp",
        duration_s=0.2,
        request=IKRequest(
            target_position=np.asarray([0.1, 0.2, 0.3], dtype=float),
            tcp_frame_name="left_tcp",
        ),
        phase=None,
    )

    move.validate(require_side=True)
    assert default_move_phase(move, side="left") == "dual_left_ik"


def test_specified_path_move_uses_python_path_object() -> None:
    path = TaskSpacePath(
        segments=(
            TcpLineSegment(
                target_offset=np.asarray([0.0, 0.02, 0.0], dtype=float),
                orientation_mode="current",
            ),
        )
    )
    move = SpecifiedPathMoveSpec(
        side="right",
        tcp_frame_name="right_tcp",
        path=path,
        duration_s=1.0,
    )
    request = SpecifiedPathRequest(
        current_q=np.asarray([0.0, 0.0], dtype=float),
        path=move.path,
        tcp_frame_name=move.tcp_frame_name,
        duration_s=move.duration_s,
    )

    request.validate_structure()
    assert request.path is path


def test_dual_cspace_vector_from_side_commands_uses_joint_names() -> None:
    cspace = dual_cspace_vector_from_side_commands(
        joint_names=("r2", "l1", "r1", "l2"),
        left_command_joint_names=("lh", "l1", "l2"),
        right_command_joint_names=("r1", "rh", "r2"),
        left_command=np.asarray([0.5, 1.0, 2.0]),
        right_command=np.asarray([10.0, 0.6, 20.0]),
    )

    np.testing.assert_allclose(cspace, [20.0, 1.0, 10.0, 2.0])


def test_dual_cspace_vector_from_side_commands_rejects_missing_joint() -> None:
    try:
        dual_cspace_vector_from_side_commands(
            joint_names=("l1", "missing"),
            left_command_joint_names=("l1",),
            right_command_joint_names=("r1",),
            left_command=np.asarray([1.0]),
            right_command=np.asarray([10.0]),
        )
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing C-space joint to be rejected")


def test_dual_cspace_goal_to_command_updates_only_matching_arm_joints() -> None:
    class _Controller:
        command_joint_names = ("l1", "lh", "l2")

    class _SideRuntime:
        joint_controller = _Controller()

    target = dual_cspace_goal_to_command(
        side_runtime=_SideRuntime(),
        base_command=np.asarray([0.0, 0.5, 0.0]),
        joint_names=("l1", "r1", "l2", "r2"),
        goal_q=np.asarray([1.0, 10.0, 2.0, 20.0]),
    )

    np.testing.assert_allclose(target, [1.0, 0.5, 2.0])


def test_dual_cspace_linear_trajectory_reaches_goal() -> None:
    trajectory = dual_cspace_linear_trajectory(
        start_q=np.asarray([0.0, 10.0]),
        goal_q=np.asarray([1.0, 12.0]),
        joint_names=("l1", "r1"),
        duration_s=0.1,
        sample_dt=0.04,
        phase="ik",
    )

    assert trajectory.joint_names == ("l1", "r1")
    assert len(trajectory) == 3
    np.testing.assert_allclose(trajectory.positions[-1], [1.0, 12.0])
    assert set(trajectory.phases) == {"ik"}


def test_side_joint_delta_goal_updates_only_selected_partition() -> None:
    partitions = DualArmJointPartitions.from_joint_names(
        ("l1", "l2", "r1", "r2", "r3"),
        left_joint_names=("l1", "l2"),
        right_joint_names=("r1", "r2", "r3"),
    )

    goal = side_joint_delta_goal(
        base_q=np.asarray([1.0, 2.0, 10.0, 20.0, 30.0]),
        partitions=partitions,
        side="right",
        deltas=(0.5, -0.25),
    )

    np.testing.assert_allclose(goal, [1.0, 2.0, 10.5, 19.75, 30.0])


def test_load_dual_arm_semantic_config_reads_default_profile() -> None:
    dual_arm = load_dual_arm_semantic_config("ar5v2_l6v1_dual")

    assert dual_arm["left"]["tcp_frame"] == "left_pinch_tcp"
    assert dual_arm["right"]["tcp_frame"] == "right_pinch_tcp"


def test_dual_arm_cumotion_summary_can_report_script_tcp_names() -> None:
    tcp = DualArmTcpSpec(
        left=CartesianTcpFrameSpec("left_demo_tcp"),
        right=CartesianTcpFrameSpec("right_demo_tcp"),
    )

    summary = dual_arm_cumotion_summary(
        cumotion_profile="default",
        dual_arm_profile="ar5v2_l6v1_dual",
        tcp=tcp,
    )

    assert summary.left_tcp == "left_demo_tcp"
    assert summary.right_tcp == "right_demo_tcp"
