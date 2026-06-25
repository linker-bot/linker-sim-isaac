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

    def add_obstacle(self, obstacle, pose):
        handle = SimpleNamespace(obstacle=obstacle, pose=pose)
        self.obstacles.append(handle)
        return handle

    def add_world_view(self):
        return self.world_view


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
        shape="box",
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
