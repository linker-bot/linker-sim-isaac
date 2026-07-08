from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.app.motion.single_arm import (
    _run_single_move,
    cspace_goal_to_command,
    cspace_linear_trajectory,
    cspace_trajectory_from_motion_result,
    current_cspace_command,
)
from linkerbot_sim.app.motion.specs import CumotionMoveSpec
from linkerbot_sim.planning.requests import MotionRequest
from linkerbot_sim.planning.results import MotionResult


class _Controller:
    command_joint_names = ("j2", "hand", "j1")


class _Runtime:
    joint_controller = _Controller()


class _FakeContext:
    config = SimpleNamespace(default_tcp_frame="tool_tcp", flange_frame="flange")

    def has_frame(self, frame_name: str) -> bool:
        return frame_name in {"tool_tcp", "flange"}


def test_current_cspace_command_uses_joint_names() -> None:
    cspace = current_cspace_command(
        _Runtime(),
        joint_names=("j1", "j2"),
        current_command_values=np.asarray([2.0, 0.5, 1.0]),
    )

    np.testing.assert_allclose(cspace, [1.0, 2.0])


def test_cspace_goal_to_command_updates_only_matching_joints() -> None:
    command = cspace_goal_to_command(
        runtime=_Runtime(),
        base_command=np.asarray([0.0, 9.0, 0.0]),
        joint_names=("j1", "j2"),
        goal_q=np.asarray([1.0, 2.0]),
    )

    np.testing.assert_allclose(command, [2.0, 9.0, 1.0])


def test_cspace_linear_trajectory_reaches_goal() -> None:
    trajectory = cspace_linear_trajectory(
        start_q=np.asarray([0.0, 0.0]),
        goal_q=np.asarray([1.0, -1.0]),
        joint_names=("j1", "j2"),
        duration_s=0.1,
        sample_dt=0.04,
        phase="ik",
    )

    assert len(trajectory) == 3
    np.testing.assert_allclose(trajectory.positions[-1], [1.0, -1.0])
    assert trajectory.phases == ("ik",) * len(trajectory)


def test_cspace_trajectory_from_motion_result_uses_path_fallback() -> None:
    result = MotionResult(
        path=np.asarray(
            [
                [0.0, 0.0],
                [0.5, -0.25],
                [1.0, -0.5],
            ],
            dtype=float,
        ),
        trajectory=None,
        success=True,
        status="SUCCESS",
    )

    trajectory = cspace_trajectory_from_motion_result(
        result,
        joint_names=("j1", "j2"),
        duration_s=1.0,
        sample_dt=0.01,
        phase="plan",
    )

    np.testing.assert_allclose(trajectory.times, [0.5, 1.0])
    np.testing.assert_allclose(trajectory.positions[-1], [1.0, -0.5])
    assert trajectory.phases == ("plan",) * len(trajectory)


def test_single_arm_cumotion_move_rejects_dual_cspace_execution() -> None:
    move = CumotionMoveSpec(
        request=MotionRequest(
            current_q=np.asarray([0.0, 0.0], dtype=float),
            goal_q=np.asarray([0.1, 0.2], dtype=float),
        ),
        duration_s=0.5,
        execution="dual_cspace",
    )

    try:
        _run_single_move(
            move,
            move_index=1,
            context=_FakeContext(),
            current_q=np.asarray([0.0, 0.0], dtype=float),
            joint_names=("j1", "j2"),
            runtime=object(),
            command=np.asarray([0.0, 0.0], dtype=float),
            step=0,
            sample_dt=0.01,
            motion_planner_config=None,
        )
    except ValueError as exc:
        assert "execution='single'" in str(exc)
    else:
        raise AssertionError("expected single-arm runtime to reject dual_cspace")
