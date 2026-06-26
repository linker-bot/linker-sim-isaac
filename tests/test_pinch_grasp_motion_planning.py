from __future__ import annotations

import numpy as np

from manipulation_project.planning.results import MotionResult
from manipulation_project.tasks.pinch_grasp import build_planned_joint_motion_trajectory


class _FakeTrajectoryState:
    def __init__(self, time_s: float) -> None:
        self.position = np.asarray([time_s, -time_s], dtype=float)
        self.velocity = np.asarray([1.0, -1.0], dtype=float)
        self.acceleration = np.asarray([2.0, -2.0], dtype=float)
        self.jerk = np.asarray([4.0, -4.0], dtype=float)


class _FakeCumotionTrajectory:
    def domain(self):
        return (0.0, 1.0)

    def eval_all(self, time_s: float):
        return _FakeTrajectoryState(float(time_s))


class _FakeMotionPlanner:
    def __init__(self, joint_path, trajectory=None) -> None:
        self.joint_path = np.asarray(joint_path, dtype=float)
        self.trajectory = trajectory
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        return MotionResult(
            joint_path=self.joint_path,
            trajectory=self.trajectory,
            success=True,
            status="SUCCESS",
        )

    def joint_names(self) -> list[str]:
        return ["arm0", "arm1"]


def test_build_planned_joint_motion_trajectory_embeds_cumotion_path() -> None:
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
    np.testing.assert_allclose(trajectory.positions[:, [0, 2]], planner.joint_path)
    np.testing.assert_allclose(trajectory.positions[[0, -1], 1], [2.0, 4.0])
    np.testing.assert_allclose(trajectory.times[[0, -1]], [0.0, 2.0])
    expected_hand = 2.0 + (trajectory.times / 2.0) * 2.0
    np.testing.assert_allclose(trajectory.positions[:, 1], expected_hand)
    assert trajectory.phases == ("move_to_approach",) * len(trajectory)


def test_build_planned_joint_motion_trajectory_adds_missing_endpoints() -> None:
    planner = _FakeMotionPlanner([[0.5, 0.5]])

    trajectory, _result = build_planned_joint_motion_trajectory(
        motion_planner=planner,
        dof_names=["arm0", "arm1"],
        arm_indices=np.asarray([0, 1], dtype=int),
        start_all=np.asarray([0.0, 0.0], dtype=float),
        target_all=np.asarray([1.0, 1.0], dtype=float),
        duration_s=1.0,
        phase="joint_motion",
    )

    np.testing.assert_allclose(
        trajectory.positions,
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    )
    np.testing.assert_allclose(trajectory.times, [0.0, 0.5, 1.0])


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
