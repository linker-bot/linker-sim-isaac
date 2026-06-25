from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from manipulation_project.backends.cumotion import motion_planner as motion_module
from manipulation_project.backends.cumotion.motion_planner import CuMotionMotionPlanner
from manipulation_project.planning.collision_objects import CollisionObject
from manipulation_project.planning.requests import MotionRequest, PoseTarget


class _FakeResults:
    def __init__(
        self,
        *,
        path_found: bool = True,
        path=None,
        interpolated_path=None,
    ) -> None:
        self.path_found = path_found
        self.path = path or []
        self.interpolated_path = interpolated_path or []


class _FakePlanner:
    def __init__(self, results: _FakeResults) -> None:
        self.results = results
        self.cspace_calls = []
        self.translation_calls = []
        self.pose_calls = []

    def plan_to_cspace_target(self, current, goal, generate_interpolated_path):
        self.cspace_calls.append((current, goal, generate_interpolated_path))
        return self.results

    def plan_to_translation_target(
        self, current, translation, generate_interpolated_path
    ):
        self.translation_calls.append(
            (current, translation, generate_interpolated_path)
        )
        return self.results

    def plan_to_pose_target(self, current, pose_target, generate_interpolated_path):
        self.pose_calls.append((current, pose_target, generate_interpolated_path))
        return self.results


class _FakeTrajectory:
    def __init__(self, waypoints) -> None:
        self.waypoints = waypoints


class _FakeTrajectoryGenerator:
    def __init__(self, cumotion) -> None:
        self.cumotion = cumotion

    def generate_trajectory(self, waypoints):
        self.cumotion.generated_waypoints = waypoints
        return _FakeTrajectory(waypoints)


class _FakeRotation3:
    @staticmethod
    def from_matrix(matrix):
        return SimpleNamespace(matrix=np.asarray(matrix, dtype=float))


class _FakePose3:
    def __init__(self, rotation, translation) -> None:
        self.rotation = rotation
        self.translation = np.asarray(translation, dtype=float)

    @staticmethod
    def from_translation(translation):
        return _FakePose3(None, translation)


class _FakeCumotion:
    Rotation3 = _FakeRotation3
    Pose3 = _FakePose3

    def __init__(self, planner: _FakePlanner) -> None:
        self.planner = planner
        self.config_calls = []
        self.generated_waypoints = None

    def create_default_motion_planner_config(
        self, robot_description, frame_name, world_view
    ):
        self.config_calls.append((robot_description, frame_name, world_view))
        return SimpleNamespace(frame_name=frame_name, world_view=world_view)

    def create_motion_planner(self, config):
        self.planner.config = config
        return self.planner

    def create_cspace_trajectory_generator(self, kinematics):
        self.trajectory_generator_kinematics = kinematics
        return _FakeTrajectoryGenerator(self)


class _FakeContext:
    def __init__(self, cumotion) -> None:
        self.cumotion = cumotion
        self.config = SimpleNamespace(
            default_tcp_frame="tool",
            flange_frame="flange",
        )
        self.robot_description = "robot_description"
        self.kinematics = "kinematics"

    def joint_names(self) -> list[str]:
        return ["j0", "j1"]


def _collision_object() -> CollisionObject:
    return CollisionObject(
        name="table",
        shape="box",
        pose=np.eye(4),
        size=(1.0, 1.0, 0.1),
    )


def _patch_collision_world(monkeypatch):
    calls = []

    def fake_make_collision_world(context, collision_objects):
        calls.append((context, tuple(collision_objects)))
        return SimpleNamespace(world_view="world_view")

    monkeypatch.setattr(motion_module, "make_collision_world", fake_make_collision_world)
    return calls


def test_motion_planner_plans_to_joint_target_and_generates_trajectory(
    monkeypatch,
) -> None:
    collision_calls = _patch_collision_world(monkeypatch)
    results = _FakeResults(
        path=[[0.0, 0.0], [1.0, 1.0]],
        interpolated_path=[[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
    )
    fake_planner = _FakePlanner(results)
    fake_cumotion = _FakeCumotion(fake_planner)
    context = _FakeContext(fake_cumotion)

    planner = CuMotionMotionPlanner(context)
    obstacle = _collision_object()
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
            collision_objects=(obstacle,),
        )
    )

    assert result.success
    assert result.status == "SUCCESS"
    np.testing.assert_allclose(
        result.joint_path,
        [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
    )
    assert isinstance(result.trajectory, _FakeTrajectory)
    np.testing.assert_allclose(fake_cumotion.generated_waypoints[-1], [1.0, 1.0])
    assert fake_cumotion.config_calls == [
        ("robot_description", "tool", "world_view")
    ]
    assert collision_calls == [(context, (obstacle,))]
    current, goal, generate_interpolated_path = fake_planner.cspace_calls[0]
    np.testing.assert_allclose(current, [0.0, 0.0])
    np.testing.assert_allclose(goal, [1.0, 1.0])
    assert generate_interpolated_path is True
    assert result.diagnostics.metrics["num_waypoints"] == 3.0
    assert result.diagnostics.metrics["num_collision_objects"] == 1.0


def test_motion_planner_geometric_mode_ignores_environment_obstacles(
    monkeypatch,
) -> None:
    collision_calls = _patch_collision_world(monkeypatch)
    fake_planner = _FakePlanner(
        _FakeResults(path=[[0.0, 0.0], [1.0, 1.0]])
    )
    fake_cumotion = _FakeCumotion(fake_planner)
    context = _FakeContext(fake_cumotion)

    planner = CuMotionMotionPlanner(context, generate_trajectory=False)
    planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
            collision_objects=(_collision_object(),),
            mode="geometric",
        )
    )

    assert collision_calls == [(context, ())]


def test_motion_planner_plans_to_translation_target(monkeypatch) -> None:
    _patch_collision_world(monkeypatch)
    fake_planner = _FakePlanner(
        _FakeResults(path=[[0.0, 0.0], [0.1, 0.2]])
    )
    context = _FakeContext(_FakeCumotion(fake_planner))

    planner = CuMotionMotionPlanner(context)
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(position=np.asarray([0.1, 0.2, 0.3])),
            tcp_frame_name="pinch_tcp",
        )
    )

    assert result.success
    current, translation, generate_interpolated_path = (
        fake_planner.translation_calls[0]
    )
    np.testing.assert_allclose(current, [0.0, 0.0])
    np.testing.assert_allclose(translation, [0.1, 0.2, 0.3])
    assert generate_interpolated_path is True


def test_motion_planner_plans_to_pose_target(monkeypatch) -> None:
    _patch_collision_world(monkeypatch)
    fake_planner = _FakePlanner(
        _FakeResults(path=[[0.0, 0.0], [0.1, 0.2]])
    )
    context = _FakeContext(_FakeCumotion(fake_planner))

    planner = CuMotionMotionPlanner(context)
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(
                position=np.asarray([0.1, 0.2, 0.3]),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
        )
    )

    assert result.success
    _current, pose_target, _generate_interpolated_path = fake_planner.pose_calls[0]
    np.testing.assert_allclose(pose_target.translation, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose_target.rotation.matrix, np.eye(3))


def test_motion_planner_failure_returns_failed_motion_result(monkeypatch) -> None:
    _patch_collision_world(monkeypatch)
    fake_planner = _FakePlanner(_FakeResults(path_found=False))
    context = _FakeContext(_FakeCumotion(fake_planner))

    planner = CuMotionMotionPlanner(context)
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
        )
    )

    assert not result.success
    assert result.status == "FAILED"
    assert result.joint_path is None
    assert result.trajectory is None
