from __future__ import annotations

import numpy as np

from manipulation_project.planning.results import MotionResult
from manipulation_project.tasks.pinch_grasp import (
    build_planned_joint_motion_trajectory,
    build_specified_tcp_line_trajectory,
)


class _FakeTrajectoryState:
    def __init__(self, time_s: float, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float).reshape(-1)
        self.position = float(time_s) * target
        self.velocity = target.copy()
        self.acceleration = 2.0 * target
        self.jerk = 4.0 * target


class _FakeCumotionTrajectory:
    def __init__(self, target=(1.0, -1.0)) -> None:
        self.target = np.asarray(target, dtype=float)

    def domain(self):
        return (0.0, 1.0)

    def eval_all(self, time_s: float):
        return _FakeTrajectoryState(float(time_s), self.target)


class _FakeMotionPlanner:
    def __init__(self, joint_path, trajectory=None) -> None:
        self.path = np.asarray(joint_path, dtype=float)
        self.trajectory = (
            _FakeCumotionTrajectory(self.path[-1]) if trajectory is None else trajectory
        )
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        return MotionResult(
            path=self.path,
            trajectory=self.trajectory,
            success=True,
            status="SUCCESS",
        )

    def joint_names(self) -> list[str]:
        return ["arm0", "arm1"]


class _FakeContext:
    def __init__(self, planner) -> None:
        self.planner = planner
        self.calls = []

    def make_motion_planner(self, *, tcp_frame_name=None, config=None):
        self.calls.append((tcp_frame_name, config))
        return self.planner


def test_build_planned_joint_motion_trajectory_embeds_cumotion_trajectory() -> None:
    planner = _FakeMotionPlanner(
        [
            [0.0, 0.0],
            [0.25, -0.5],
            [1.0, -1.0],
        ]
    )

    trajectory, result = build_planned_joint_motion_trajectory(
        motion_planner=planner,
        dof_names=["arm0", "hand", "arm1"],
        arm_indices=np.asarray([0, 2], dtype=int),
        start_all=np.asarray([0.0, 2.0, 0.0], dtype=float),
        target_all=np.asarray([1.0, 4.0, -1.0], dtype=float),
        duration_s=2.0,
        phase="move_to_approach",
    )

    assert result.success
    assert len(planner.requests) == 1
    np.testing.assert_allclose(planner.requests[0].current_q, [0.0, 0.0])
    np.testing.assert_allclose(planner.requests[0].goal_q, [1.0, -1.0])
    np.testing.assert_allclose(result.path, planner.path)
    np.testing.assert_allclose(
        trajectory.positions[[0, -1]][:, [0, 2]], planner.path[[0, -1]]
    )
    np.testing.assert_allclose(trajectory.positions[[0, -1], 1], [2.0, 4.0])
    np.testing.assert_allclose(trajectory.times[[0, -1]], [0.0, 2.0])
    expected_hand = 2.0 + (trajectory.times / 2.0) * 2.0
    np.testing.assert_allclose(trajectory.positions[:, 1], expected_hand)
    assert trajectory.phases == ("move_to_approach",) * len(trajectory)


def test_build_planned_joint_motion_trajectory_requires_cumotion_trajectory() -> None:
    planner = _FakeMotionPlanner([[0.5, 0.5]], trajectory=None)
    planner.trajectory = None

    try:
        build_planned_joint_motion_trajectory(
            motion_planner=planner,
            dof_names=["arm0", "arm1"],
            arm_indices=np.asarray([0, 1], dtype=int),
            start_all=np.asarray([0.0, 0.0], dtype=float),
            target_all=np.asarray([1.0, 1.0], dtype=float),
            duration_s=1.0,
            phase="joint_motion",
        )
    except RuntimeError as exc:
        assert "returned no trajectory" in str(exc)
    else:
        raise AssertionError("expected missing cuMotion trajectory to fail")


def test_build_planned_joint_motion_trajectory_uses_cumotion_time_parameterization() -> (
    None
):
    planner = _FakeMotionPlanner(
        [[0.0, 0.0], [1.0, -1.0]],
        trajectory=_FakeCumotionTrajectory(),
    )

    trajectory, result = build_planned_joint_motion_trajectory(
        motion_planner=planner,
        dof_names=["arm0", "hand", "arm1"],
        arm_indices=np.asarray([0, 2], dtype=int),
        start_all=np.asarray([0.0, 2.0, 0.0], dtype=float),
        target_all=np.asarray([1.0, 4.0, -1.0], dtype=float),
        duration_s=2.0,
        phase="lift",
    )

    assert result.trajectory is planner.trajectory
    np.testing.assert_allclose(trajectory.times[[0, -1]], [0.0, 2.0])
    np.testing.assert_allclose(trajectory.positions[[0, -1]][:, [0, 2]], [[0.0, 0.0], [1.0, -1.0]])
    midpoint = len(trajectory) // 2
    np.testing.assert_allclose(trajectory.positions[midpoint, [0, 2]], [0.5, -0.5])
    np.testing.assert_allclose(trajectory.positions[midpoint, 1], 3.0)
    # The fake cuMotion trajectory is generated over 1 second and requested_duration_s is
    # 2 seconds, so derivatives should be scaled by 1/2, 1/4, and 1/8 respectively.
    np.testing.assert_allclose(trajectory.velocities[midpoint, [0, 2]], [0.5, -0.5])
    np.testing.assert_allclose(trajectory.accelerations[midpoint, [0, 2]], [0.5, -0.5])
    np.testing.assert_allclose(trajectory.jerks[midpoint, [0, 2]], [0.5, -0.5])
    assert trajectory.phases == ("lift",) * len(trajectory)


def test_build_specified_tcp_line_trajectory_uses_task_space_request() -> None:
    planner = _FakeMotionPlanner(
        [
            [0.0, 0.0],
            [0.4, -0.2],
            [1.0, -0.5],
        ]
    )
    context = _FakeContext(planner)

    trajectory, result = build_specified_tcp_line_trajectory(
        context=context,
        tcp_frame_name="pinch_tcp",
        dof_names=["arm0", "hand", "arm1"],
        arm_indices=np.asarray([0, 2], dtype=int),
        start_all=np.asarray([0.0, 9.0, 0.0], dtype=float),
        target_position=np.asarray([0.1, 0.2, 0.3], dtype=float),
        duration_s=1.5,
        phase="approach_box",
    )

    assert result.success
    assert len(context.calls) == 1
    tcp_frame_name, config = context.calls[0]
    assert tcp_frame_name == "pinch_tcp"
    assert config.planning_pipeline == "specified_path"
    assert config.specified_path.family == "task_space_segments"
    request = planner.requests[0]
    assert request.tcp_frame_name == "pinch_tcp"
    np.testing.assert_allclose(request.current_q, [0.0, 0.0])
    assert request.path.segments[0].orientation_mode == "current"
    np.testing.assert_allclose(
        request.path.segments[0].target_position, [0.1, 0.2, 0.3]
    )
    np.testing.assert_allclose(result.path, planner.path)
    np.testing.assert_allclose(
        trajectory.positions[[0, -1]][:, [0, 2]], planner.path[[0, -1]]
    )
    np.testing.assert_allclose(trajectory.positions[:, 1], 9.0)
    np.testing.assert_allclose(trajectory.times[[0, -1]], [0.0, 1.5])
    assert trajectory.phases == ("approach_box",) * len(trajectory)
