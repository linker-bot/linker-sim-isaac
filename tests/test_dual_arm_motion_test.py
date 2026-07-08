from __future__ import annotations

import numpy as np

import linkerbot_sim.app.motion.dual_arm as dual_arm_motion
from linkerbot_sim.app.motion.specs import (  # noqa: E402
    CSpaceDeltaPlanMoveSpec,
    CSpaceGoalPlanMoveSpec,
    CommandOverlaySpec,
    CumotionMoveSpec,
    DualHandMoveSpec,
    HandMoveSpec,
    IkOffsetMoveSpec,
    SpecifiedPathMoveSpec,
    default_move_phase,
    specified_path_planner_config,
)
from linkerbot_sim.app.motion.dual_arm import (  # noqa: E402
    DualArmCuMotionExecutionSession,
    dual_cspace_goal_to_command,
    dual_cspace_linear_trajectory,
    dual_cspace_vector_from_side_commands,
    side_joint_delta_goal,
)
from linkerbot_sim.app.motion.dual_arm_execution import (  # noqa: E402
    plan_dual_motion_trajectory,
)
from linkerbot_sim.app.motion.dual_arm_semantics import (  # noqa: E402
    dual_arm_semantics_from_robot_configs,
)
from linkerbot_sim.app.motion.runtime import MotionPlanningFailed  # noqa: E402
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
from linkerbot_sim.planning.results import (  # noqa: E402
    MotionResult,
    PlanningDiagnostics,
)
from linkerbot_sim.utils.config import load_yaml


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


def test_dual_planner_failure_raises_recoverable_exception() -> None:
    class _Planner:
        def plan(self, request):
            return MotionResult(
                path=None,
                trajectory=None,
                success=False,
                status="FAILED",
                diagnostics=PlanningDiagnostics(
                    message="path_conversion=failed frame=right_tcp"
                ),
            )

    class _Context:
        def make_motion_planner(self, *, tcp_frame_name, config):
            assert tcp_frame_name == "right_tcp"
            return _Planner()

    try:
        plan_dual_motion_trajectory(
            context=_Context(),
            request=SpecifiedPathRequest(
                current_q=np.asarray([0.0, 0.0], dtype=float),
                path=TaskSpacePath(
                    segments=(
                        TcpLineSegment(
                            target_offset=(0.0, 0.0, 0.1),
                            orientation_mode="none",
                        ),
                    ),
                ),
                tcp_frame_name="right_tcp",
                duration_s=1.0,
            ),
            tcp_frame_name="right_tcp",
            config=MotionPlannerBackendConfig.from_mapping(
                {"planning_pipeline": "specified_path"}
            ),
            joint_names=("j1", "j2"),
            duration_s=1.0,
            sample_dt=0.1,
            phase="right_tcp_line",
            move_index=3,
            side="right",
            execution_mode="selected_side",
        )
    except MotionPlanningFailed as exc:
        assert exc.phase == "right_tcp_line"
        assert exc.status == "FAILED"
        assert exc.move_index == 3
        assert exc.side == "right"
        assert exc.tcp_frame_name == "right_tcp"
        assert "path_conversion=failed" in str(exc)
    else:
        raise AssertionError("expected planner failure to use recoverable exception")


def test_execute_moves_result_returns_failure_without_raising(monkeypatch) -> None:
    def _fail_move(*args, **kwargs):
        raise MotionPlanningFailed(
            "planner failed",
            phase="right_tcp_arc",
            status="FAILED",
            solver_message="path_conversion=failed frame=right_tcp",
            move_index=kwargs["move_index"],
            side="right",
            tcp_frame_name="right_tcp",
            component="motion planner",
        )

    session = DualArmCuMotionExecutionSession.__new__(DualArmCuMotionExecutionSession)
    session.step = 5
    session.context = object()
    session.partitions = object()
    session.joint_names = ("l1", "r1")
    session.execution = object()
    session.sample_dt = 0.1
    session.motion_planner_config = MotionPlannerBackendConfig.from_mapping(None)
    session.default_tcp_by_side = {"left": "left_tcp", "right": "right_tcp"}
    session.refresh_current_state = lambda: (
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([0.0], dtype=float),
        np.asarray([0.0], dtype=float),
    )
    monkeypatch.setattr(dual_arm_motion, "_run_dual_move", _fail_move)

    result = session.execute_moves_result(
        (
            CSpaceDeltaPlanMoveSpec(
                side="right",
                tcp_frame_name="right_tcp",
                joint_deltas=(0.1,),
                duration_s=0.5,
                phase="right_tcp_arc",
            ),
        ),
        start_step=5,
    )

    assert not result.success
    assert result.step == 5
    assert result.failed_move_index == 1
    assert result.phase == "right_tcp_arc"
    assert result.side == "right"
    assert result.tcp_frame_name == "right_tcp"
    assert result.status == "FAILED"
    assert result.message == "path_conversion=failed frame=right_tcp"
    assert session.step == 5


def test_dual_arm_semantics_from_robot_configs_reads_xrdf_and_flange() -> None:
    semantics = dual_arm_semantics_from_robot_configs(
        {
            "left": load_yaml("configs/robots/ar5v2_l6v1_l.yaml"),
            "right": load_yaml("configs/robots/ar5v2_l6v1_r.yaml"),
        }
    )

    assert semantics.left_arm_joints == tuple(
        f"AR5V2_L_arm_joint_{index}" for index in range(1, 8)
    )
    assert semantics.right_arm_joints == tuple(
        f"AR5V2_R_arm_joint_{index}" for index in range(1, 8)
    )
    assert semantics.left_flange_frame == "AR5V2_L_arm_flan_link"
    assert semantics.right_flange_frame == "AR5V2_R_arm_flan_link"
    assert semantics.left_default_tcp_frame == "AR5V2_L_pinch_tcp"
    assert semantics.right_default_tcp_frame == "AR5V2_R_pinch_tcp"
