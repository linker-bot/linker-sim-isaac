from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from manipulation_project.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)
from manipulation_project.planning.collision_objects import CollisionObject


class _FakeKinematics:
    def num_cspace_coords(self):
        return 2

    def cspace_coord_name(self, index):
        return ("j0", "j1")[index]

    def frame_names(self):
        return ["flange", "tool"]


class _FakeRobotDescription:
    def kinematics(self):
        return _FakeKinematics()


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
        self.world_view = _FakeWorldView()
        self.obstacles = []

    def add_obstacle(self, obstacle, pose):
        handle = SimpleNamespace(obstacle=obstacle, pose=pose)
        self.obstacles.append(handle)
        return handle

    def add_world_view(self):
        return self.world_view

    def set_pose(self, handle, pose) -> None:
        handle.pose = pose

    def enable_obstacle(self, _handle) -> None:
        pass

    def disable_obstacle(self, _handle) -> None:
        pass

    def remove_obstacle(self, handle) -> None:
        self.obstacles.remove(handle)


class _FakeCumotion(ModuleType):
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
        super().__init__("cumotion")
        self.created_worlds = []
        self.loaded_paths = []

    def load_robot_from_file(self, xrdf_path, urdf_path):
        self.loaded_paths.append((xrdf_path, urdf_path))
        return _FakeRobotDescription()

    def create_world(self):
        world = _FakeWorld()
        self.created_worlds.append(world)
        return world

    def create_obstacle(self, obstacle_type):
        return _FakeObstacle(obstacle_type)


def _context(monkeypatch) -> CuMotionContext:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    config = CuMotionConfig(
        xrdf_path=Path("robot.xrdf"),
        urdf_path=Path("robot.urdf"),
        flange_frame="flange",
        custom_tcp_frame="tool",
    )
    return CuMotionContext(config)


def test_context_syncs_and_reuses_collision_world(monkeypatch) -> None:
    context = _context(monkeypatch)
    table = CollisionObject("table", "cuboid", np.eye(4), (1.0, 1.0, 0.1))
    ball = CollisionObject("ball", "sphere", np.eye(4), (0.2,))

    first_world = context.sync_collision_world((table,))
    second_world = context.sync_collision_world((table, ball))

    assert first_world is second_world
    assert context.collision_world() is first_world
    assert sorted(first_world.handles) == ["ball", "table"]
    assert first_world.world_view.update_count == 2


def test_context_empty_collision_world_is_separate(monkeypatch) -> None:
    context = _context(monkeypatch)
    table = CollisionObject("table", "cuboid", np.eye(4), (1.0, 1.0, 0.1))

    environment_world = context.sync_collision_world((table,))
    empty_world = context.empty_collision_world()

    assert empty_world is context.empty_collision_world()
    assert empty_world is not environment_world
    assert empty_world.handles == {}
    assert sorted(context.collision_world().handles) == ["table"]


def test_context_clear_collision_world_removes_environment(monkeypatch) -> None:
    context = _context(monkeypatch)
    table = CollisionObject("table", "cuboid", np.eye(4), (1.0, 1.0, 0.1))

    context.sync_collision_world((table,))
    cleared = context.clear_collision_world()

    assert cleared.handles == {}
    assert context.collision_world() is cleared
