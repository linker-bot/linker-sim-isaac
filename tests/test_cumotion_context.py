from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from linkerbot_sim.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)
from linkerbot_sim.backends.cumotion.tcp_context import make_cumotion_context
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.tcp.tcp_frame import TcpFrame


class _FakeKinematics:
    def __init__(self, frame_names=("flange", "tool")) -> None:
        self._frame_names = list(frame_names)

    def num_cspace_coords(self):
        return 2

    def cspace_coord_name(self, index):
        return ("j0", "j1")[index]

    def frame_names(self):
        return list(self._frame_names)


class _FakeRobotDescription:
    def __init__(self, frame_names=("flange", "tool")) -> None:
        self._frame_names = list(frame_names)

    def kinematics(self):
        return _FakeKinematics(self._frame_names)


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
        self.frame_names_by_urdf = {}

    def load_robot_from_file(self, xrdf_path, urdf_path):
        self.loaded_paths.append((xrdf_path, urdf_path))
        frame_names = list(
            self.frame_names_by_urdf.get(str(urdf_path), ("flange", "tool"))
        )
        if Path(urdf_path).exists():
            root = ET.parse(urdf_path).getroot()
            for link in root.findall("link"):
                name = link.get("name")
                if name and name not in frame_names:
                    frame_names.append(name)
        return _FakeRobotDescription(frame_names)

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


def _write_urdf(path: Path, link_names=("flange",)) -> Path:
    links = "\n".join(f'  <link name="{name}" />' for name in link_names)
    path.write_text(
        f"<?xml version='1.0'?>\n<robot name='test'>\n{links}\n</robot>\n",
        encoding="utf-8",
    )
    return path


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


def test_make_cumotion_context_with_tcp_writes_temp_urdf(
    monkeypatch, tmp_path
) -> None:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    base_urdf = _write_urdf(tmp_path / "robot.urdf")
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
    )
    tcp = TcpFrame.from_xyz_rpy(
        "pinch_tcp",
        "flange",
        xyz=(0.0, 0.0, 0.12),
    )

    with make_cumotion_context(config, tcp=tcp) as context:
        loaded_urdf = Path(fake_cumotion.loaded_paths[-1][1])
        assert loaded_urdf != base_urdf
        assert loaded_urdf.exists()
        assert context.config.custom_tcp_frame == "pinch_tcp"
        assert context.has_frame("pinch_tcp")
        temp_parent = loaded_urdf.parent

    assert not temp_parent.exists()


def test_make_cumotion_context_with_tcp_uses_output_dir(
    monkeypatch, tmp_path
) -> None:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    base_urdf = _write_urdf(tmp_path / "robot.urdf")
    output_dir = tmp_path / "tcp_urdfs"
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
    )
    tcp = TcpFrame.from_xyz_rpy("pinch_tcp", "flange")

    with make_cumotion_context(config, tcp=tcp, output_dir=output_dir) as context:
        loaded_urdf = Path(fake_cumotion.loaded_paths[-1][1])
        assert loaded_urdf.parent == output_dir
        assert loaded_urdf.exists()
        assert context.config.custom_tcp_frame == "pinch_tcp"

    assert output_dir.exists()


def test_make_cumotion_context_with_flange_tcp_does_not_write_urdf(
    monkeypatch, tmp_path
) -> None:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    base_urdf = _write_urdf(tmp_path / "robot.urdf")
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
        custom_tcp_frame="tool",
    )
    tcp = TcpFrame.from_xyz_rpy("flange", "flange")

    with make_cumotion_context(config, tcp=tcp) as context:
        assert Path(fake_cumotion.loaded_paths[-1][1]) == base_urdf
        assert context.config.custom_tcp_frame is None
        assert context.has_frame("flange")


def test_make_cumotion_context_rejects_existing_non_flange_tcp(
    monkeypatch, tmp_path
) -> None:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    base_urdf = _write_urdf(tmp_path / "robot.urdf", link_names=("flange", "tool"))
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
    )
    tcp = TcpFrame.from_xyz_rpy("tool", "flange", xyz=(0.0, 0.0, 0.1))

    with pytest.raises(ValueError, match="already exists"):
        with make_cumotion_context(config, tcp=tcp):
            pass


def test_make_cumotion_context_rejects_missing_parent_frame(tmp_path) -> None:
    base_urdf = _write_urdf(tmp_path / "robot.urdf")
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
    )
    tcp = TcpFrame.from_xyz_rpy("pinch_tcp", "missing")

    with pytest.raises(ValueError, match="Parent frame"):
        with make_cumotion_context(config, tcp=tcp):
            pass
