from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from manipulation_project.backends.cumotion.collision_world import (
    CuMotionCollisionWorld,
)
from manipulation_project.planning.collision_objects import CollisionObject


class _FakeRotation3:
    @staticmethod
    def from_matrix(matrix):
        return np.asarray(matrix, dtype=float)


class _FakePose3:
    def __init__(self, rotation, translation) -> None:
        self.rotation = rotation
        self.translation = np.asarray(translation, dtype=float)


class _FakeObstacle:
    def __init__(self, obstacle_type) -> None:
        self.obstacle_type = obstacle_type
        self.attributes = {}

    def set_attribute(self, attribute, value) -> None:
        self.attributes[attribute] = value


class _FakeWorldView:
    def __init__(self) -> None:
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1


class _FakeWorld:
    def __init__(self) -> None:
        self.obstacles = []
        self.world_view = _FakeWorldView()
        self.enabled = []
        self.disabled = []
        self.removed = []
        self.poses = []

    def add_obstacle(self, obstacle, pose):
        handle = SimpleNamespace(obstacle=obstacle, pose=pose)
        self.obstacles.append(handle)
        return handle

    def add_world_view(self):
        return self.world_view

    def set_pose(self, handle, pose) -> None:
        handle.pose = pose
        self.poses.append((handle, pose))

    def enable_obstacle(self, handle) -> None:
        self.enabled.append(handle)

    def disable_obstacle(self, handle) -> None:
        self.disabled.append(handle)

    def remove_obstacle(self, handle) -> None:
        self.removed.append(handle)
        self.obstacles.remove(handle)


class _FakeCumotion:
    Rotation3 = _FakeRotation3
    Pose3 = _FakePose3
    Obstacle = SimpleNamespace(
        Type=SimpleNamespace(CUBOID="cuboid", SPHERE="sphere", CAPSULE="capsule"),
        Attribute=SimpleNamespace(
            SIDE_LENGTHS="side_lengths",
            RADIUS="radius",
            HEIGHT="height",
        ),
    )

    def __init__(self) -> None:
        self.create_world_count = 0
        self.created_obstacles = []
        self.world = _FakeWorld()

    def create_world(self):
        self.create_world_count += 1
        return self.world

    def create_obstacle(self, obstacle_type):
        obstacle = _FakeObstacle(obstacle_type)
        self.created_obstacles.append(obstacle)
        return obstacle


def test_collision_world_uses_cumotion_factory_and_adds_enabled_obstacles() -> None:
    cumotion = _FakeCumotion()
    context = SimpleNamespace(cumotion=cumotion)
    enabled = CollisionObject(
        name="table",
        shape="cuboid",
        pose=np.eye(4),
        size=(0.4, 0.5, 0.1),
        padding=0.02,
    )
    disabled = CollisionObject(
        name="disabled",
        shape="sphere",
        pose=np.eye(4),
        size=(0.1,),
        enabled=False,
    )

    collision_world = CuMotionCollisionWorld(context, (enabled, disabled))

    assert cumotion.create_world_count == 1
    assert len(cumotion.world.obstacles) == 1
    assert "table" in collision_world.handles
    assert "disabled" not in collision_world.handles
    assert collision_world.world_view.update_count == 1
    obstacle = cumotion.created_obstacles[0]
    assert obstacle.obstacle_type == "cuboid"
    np.testing.assert_allclose(obstacle.attributes["side_lengths"], [0.44, 0.54, 0.14])


def test_collision_world_sync_updates_disables_and_removes_obstacles() -> None:
    cumotion = _FakeCumotion()
    context = SimpleNamespace(cumotion=cumotion)
    collision_world = CuMotionCollisionWorld(
        context,
        (
            CollisionObject("table", "cuboid", np.eye(4), (1.0, 1.0, 0.1)),
            CollisionObject("ball", "sphere", np.eye(4), (0.2,)),
        ),
    )
    table_handle = collision_world.handles["table"]
    ball_handle = collision_world.handles["ball"]
    moved_pose = np.eye(4)
    moved_pose[:3, 3] = [1.0, 2.0, 3.0]

    collision_world.sync(
        (
            CollisionObject("table", "cuboid", moved_pose, (1.0, 1.0, 0.1)),
            CollisionObject("new", "capsule", np.eye(4), (0.1, 0.4)),
        )
    )

    assert ball_handle in cumotion.world.removed
    assert table_handle in cumotion.world.enabled
    assert "ball" not in collision_world.handles
    assert "new" in collision_world.handles
    assert cumotion.world.world_view.update_count == 2
    np.testing.assert_allclose(table_handle.pose.translation, [1.0, 2.0, 3.0])

    collision_world.sync(
        (CollisionObject("table", "cuboid", moved_pose, (1.0, 1.0, 0.1), enabled=False),)
    )

    assert table_handle in cumotion.world.disabled


def test_collision_world_sync_recreates_changed_geometry() -> None:
    cumotion = _FakeCumotion()
    context = SimpleNamespace(cumotion=cumotion)
    collision_world = CuMotionCollisionWorld(
        context,
        (CollisionObject("table", "cuboid", np.eye(4), (1.0, 1.0, 0.1)),),
    )
    original_handle = collision_world.handles["table"]

    collision_world.sync(
        (CollisionObject("table", "cuboid", np.eye(4), (1.2, 1.0, 0.1)),)
    )

    assert original_handle in cumotion.world.removed
    assert collision_world.handles["table"] is not original_handle
    obstacle = collision_world.obstacles["table"]
    np.testing.assert_allclose(obstacle.attributes["side_lengths"], [1.2, 1.0, 0.1])
