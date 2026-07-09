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
    materialize_cumotion_config,
)
from linkerbot_sim.backends.cumotion.tcp_frame import TcpFrame
from linkerbot_sim.planning.collision_objects import CollisionObject


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
        default_tcp_frame="tool",
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


def test_config_rejects_removed_custom_tcp_frame_field(tmp_path) -> None:
    with pytest.raises(ValueError, match="custom_tcp_frame"):
        CuMotionConfig.from_mapping(
            {
                "xrdf_path": tmp_path / "robot.xrdf",
                "urdf_path": tmp_path / "robot.urdf",
                "flange_frame": "flange",
                "custom_tcp_frame": "tool",
            }
        )


def test_context_materializes_custom_tcps_from_config(monkeypatch, tmp_path) -> None:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    base_urdf = _write_urdf(tmp_path / "robot.urdf")
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
        default_tcp_frame="pinch_tcp",
        custom_tcp_frames=(
            TcpFrame.from_xyz_rpy(
                "pinch_tcp",
                "flange",
                xyz=(0.0, 0.0, 0.12),
            ),
        ),
    )

    context = CuMotionContext(config)

    loaded_urdf = Path(fake_cumotion.loaded_paths[-1][1])
    assert loaded_urdf != base_urdf
    assert loaded_urdf.exists()
    assert context.config.default_tcp_frame == "pinch_tcp"
    assert context.has_frame("pinch_tcp")
    root = ET.parse(loaded_urdf).getroot()
    joint = root.find("./joint[@name='pinch_tcp_joint']")
    assert joint is not None
    assert joint.find("parent").get("link") == "flange"


def test_materialize_cumotion_config_is_idempotent(tmp_path) -> None:
    base_urdf = _write_urdf(tmp_path / "robot.urdf")
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
        default_tcp_frame="tool_tcp",
        custom_tcp_frames=(
            TcpFrame.from_xyz_rpy("tool_tcp", "flange", xyz=(0.0, 0.0, 0.12)),
        ),
    )

    first = materialize_cumotion_config(config)
    second = materialize_cumotion_config(first)

    assert first.urdf_path == second.urdf_path
    assert first.custom_tcp_frames == ()
    assert second.custom_tcp_frames == ()


def test_context_materializes_multiple_custom_tcps(monkeypatch, tmp_path) -> None:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    base_urdf = _write_urdf(
        tmp_path / "robot.urdf",
        link_names=("left_flange", "right_flange"),
    )
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="left_flange",
        default_tcp_frame="left_pinch_tcp",
        custom_tcp_frames=(
            TcpFrame.from_xyz_rpy("left_pinch_tcp", "left_flange"),
            TcpFrame.from_xyz_rpy("right_pinch_tcp", "right_flange"),
        ),
    )

    context = CuMotionContext(config)

    loaded_urdf = Path(fake_cumotion.loaded_paths[-1][1])
    assert context.has_frame("left_pinch_tcp")
    assert context.has_frame("right_pinch_tcp")
    root = ET.parse(loaded_urdf).getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    assert {"left_pinch_tcp", "right_pinch_tcp"} <= link_names


def test_context_rejects_existing_custom_tcp_frame(monkeypatch, tmp_path) -> None:
    fake_cumotion = _FakeCumotion()
    monkeypatch.setitem(sys.modules, "cumotion", fake_cumotion)
    base_urdf = _write_urdf(tmp_path / "robot.urdf", link_names=("flange", "tool"))
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
        custom_tcp_frames=(TcpFrame.from_xyz_rpy("tool", "flange"),),
    )

    with pytest.raises(ValueError, match="already exists"):
        CuMotionContext(config)


def test_context_rejects_missing_parent_frame(tmp_path) -> None:
    base_urdf = _write_urdf(tmp_path / "robot.urdf")
    config = CuMotionConfig(
        xrdf_path=tmp_path / "robot.xrdf",
        urdf_path=base_urdf,
        flange_frame="flange",
        custom_tcp_frames=(TcpFrame.from_xyz_rpy("pinch_tcp", "missing"),),
    )

    with pytest.raises(ValueError, match="Parent frame"):
        materialize_cumotion_config(config)
