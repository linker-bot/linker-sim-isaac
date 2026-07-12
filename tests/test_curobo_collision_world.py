from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from linkerbot_sim.backends.curobo.collision_world import (
    CuroboCollisionWorld,
    make_curobo_scene_cfg,
)
from linkerbot_sim.backends.curobo.config import CuroboConfig
from linkerbot_sim.backends.curobo.context import CuroboContext
from linkerbot_sim.planning.collision_objects import CollisionObject


@dataclass
class _FakeObstacle:
    name: str
    pose: list[float]
    dims: list[float] | None = None
    radius: float | None = None
    base: list[float] | None = None
    tip: list[float] | None = None


class _FakeScene:
    Cuboid = _FakeObstacle
    Sphere = _FakeObstacle
    Capsule = _FakeObstacle

    class Scene:
        def __init__(self, *, cuboid=None, sphere=None, capsule=None, **_kwargs):
            self.cuboid = list(cuboid or [])
            self.sphere = list(sphere or [])
            self.capsule = list(capsule or [])
            self.objects = self.cuboid + self.sphere + self.capsule


class _FakeSolver:
    def __init__(self, *, supports_scene: bool = True) -> None:
        self.scene_collision_checker = object() if supports_scene else None
        self.world_updates = []

    def update_world(self, scene_cfg) -> None:
        self.world_updates.append(scene_cfg)


class _FakeContext:
    def __init__(self, *, supports_scene: bool = True) -> None:
        self.scene_module = _FakeScene
        self.ik_solver = _FakeSolver(supports_scene=supports_scene)
        self.motion_planner = _FakeSolver(supports_scene=supports_scene)
        self.batch_motion_planner = _FakeSolver(supports_scene=supports_scene)


class _LazyFakeContext:
    scene_module = _FakeScene

    def __init__(self) -> None:
        self.existing = (_FakeSolver(),)

    def existing_solvers(self):
        return self.existing

    @property
    def ik_solver(self):
        raise AssertionError("collision sync should not create lazy ik_solver")

    @property
    def motion_planner(self):
        raise AssertionError("collision sync should not create lazy motion_planner")

    @property
    def batch_motion_planner(self):
        raise AssertionError(
            "collision sync should not create lazy batch_motion_planner"
        )


def test_make_curobo_scene_cfg_converts_enabled_collision_objects() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    scene = make_curobo_scene_cfg(
        _FakeContext(),
        (
            CollisionObject(
                name="box",
                shape="cuboid",
                pose=pose,
                size=(0.1, 0.2, 0.3),
                padding=0.01,
            ),
            CollisionObject(
                name="ball",
                shape="sphere",
                pose=np.eye(4),
                size=(0.05,),
            ),
            CollisionObject(
                name="rope",
                shape="capsule",
                pose=np.eye(4),
                size=(0.02, 0.4),
            ),
            CollisionObject(
                name="disabled",
                shape="cuboid",
                pose=np.eye(4),
                size=(1.0, 1.0, 1.0),
                enabled=False,
            ),
        ),
    )

    assert len(scene.objects) == 3
    assert [item.name for item in scene.cuboid] == ["box", "ball", "rope"]
    np.testing.assert_allclose(
        scene.cuboid[0].pose, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(scene.cuboid[0].dims, [0.12, 0.22, 0.32])
    np.testing.assert_allclose(scene.cuboid[1].dims, [0.1, 0.1, 0.1])
    np.testing.assert_allclose(scene.cuboid[2].dims, [0.04, 0.04, 0.44])
    assert scene.sphere == []
    assert scene.capsule == []


def test_curobo_collision_world_sync_updates_supported_solvers() -> None:
    context = _FakeContext()

    world = CuroboCollisionWorld(
        context,
        (
            CollisionObject(
                name="box",
                shape="cuboid",
                pose=np.eye(4),
                size=(0.1, 0.2, 0.3),
            ),
        ),
    )

    assert world.num_enabled_obstacles == 1
    assert len(context.ik_solver.world_updates) == 1
    assert len(context.motion_planner.world_updates) == 1
    assert len(context.batch_motion_planner.world_updates) == 1

    world.sync(())

    assert world.num_enabled_obstacles == 0
    assert len(context.ik_solver.world_updates) == 2


def test_curobo_collision_world_skips_solvers_without_scene_checker() -> None:
    context = _FakeContext(supports_scene=False)

    CuroboCollisionWorld(
        context,
        (
            CollisionObject(
                name="box",
                shape="cuboid",
                pose=np.eye(4),
                size=(0.1, 0.2, 0.3),
            ),
        ),
    )

    assert context.ik_solver.world_updates == []
    assert context.motion_planner.world_updates == []
    assert context.batch_motion_planner.world_updates == []


def test_curobo_collision_world_updates_existing_lazy_solvers_only() -> None:
    context = _LazyFakeContext()

    CuroboCollisionWorld(
        context,
        (
            CollisionObject(
                name="box",
                shape="cuboid",
                pose=np.eye(4),
                size=(0.1, 0.2, 0.3),
            ),
        ),
    )

    assert len(context.existing[0].world_updates) == 1


def test_curobo_context_lazy_solver_receives_existing_collision_world() -> None:
    context = CuroboContext.__new__(CuroboContext)
    solver = _FakeSolver()
    context._ik_solver = None
    context._motion_planner = None
    context._batch_motion_planner = None
    context._collision_world = SimpleNamespace(scene_cfg="scene")
    context._make_ik_solver = lambda: solver

    assert context.existing_solvers() == ()
    assert context.ik_solver is solver

    assert context.existing_solvers() == (solver,)
    assert solver.world_updates == ["scene"]


def test_curobo_collision_world_rejects_unknown_shape() -> None:
    try:
        make_curobo_scene_cfg(
            _FakeContext(),
            (
                CollisionObject(
                    name="bad",
                    shape="mesh",
                    pose=np.eye(4),
                    size=(1.0,),
                ),
            ),
        )
    except ValueError as exc:
        assert "Unsupported cuRobo collision object shape" in str(exc)
    else:
        raise AssertionError("unsupported shape was accepted")


def test_fake_namespace_keeps_module_import_side_effect_free() -> None:
    fake = SimpleNamespace(name="curobo")
    assert fake.name == "curobo"


def test_curobo_context_collision_capability_checks_all_requirements() -> None:
    config = CuroboConfig.from_mapping(
        {
            "curobo": {
                "enabled": True,
                "planning_joint_group": "arm",
                "robot": {
                    "robot_config_path": "configs/robots/ar5v2_l.yaml",
                    "default_tcp_frame": "tool",
                },
                "kinematics": {"ik": {"collision_cache": {"cuboid": 4, "mesh": 1}}},
                "motion_planner": {"collision_cache": {"cuboid": 6, "mesh": 1}},
            }
        }
    )
    context = CuroboContext.__new__(CuroboContext)
    context.config = config
    context.kinematics = SimpleNamespace(total_spheres=lambda: 3)
    context._ik_solver = _FakeSolver()
    context._motion_planner = None
    context._batch_motion_planner = None
    context._collision_world = SimpleNamespace(
        materialized_counts={"cuboid": 4, "mesh": 1}
    )
    context.record_collision_sync(1, "test-view")

    capability = context.collision_capability()

    assert capability.available is True
    assert capability.robot_sphere_count == 3
    assert capability.configured_cache == {"cuboid": 4, "mesh": 1}
    assert capability.required_cache == {"cuboid": 4, "mesh": 1}
    assert capability.missing_requirements == ()


def test_collision_capability_uses_each_consumer_cache() -> None:
    config = CuroboConfig.from_mapping(
        {
            "curobo": {
                "enabled": True,
                "planning_joint_group": "arm",
                "robot": {
                    "robot_config_path": "configs/robots/ar5v2_l.yaml",
                    "default_tcp_frame": "tool",
                },
                "kinematics": {"ik": {"collision_cache": {"cuboid": 2}}},
                "motion_planner": {"collision_cache": {"cuboid": 6}},
            }
        }
    )
    context = CuroboContext.__new__(CuroboContext)
    context.config = config
    context.kinematics = SimpleNamespace(total_spheres=lambda: 3)
    context._ik_solver = _FakeSolver()
    context._motion_planner = _FakeSolver()
    context._batch_motion_planner = _FakeSolver()
    context._collision_world = SimpleNamespace(materialized_counts={"cuboid": 5})
    context.record_collision_sync(1, "batch-view")

    ik_capability = context.ensure_collision_checker("ik")
    planner_capability = context.ensure_collision_checker("planner")
    batch_capability = context.ensure_collision_checker("batch_planner")

    assert ik_capability.available is False
    assert ik_capability.configured_cache == {"cuboid": 2}
    assert planner_capability.available is True
    assert planner_capability.configured_cache == {"cuboid": 6}
    assert batch_capability.available is True
    assert batch_capability.configured_cache == {"cuboid": 6}
    assert batch_capability.required_cache == {"cuboid": 5}


def test_lazy_consumer_reports_undersized_cache_without_updating_world() -> None:
    config = CuroboConfig.from_mapping(
        {
            "curobo": {
                "enabled": True,
                "planning_joint_group": "arm",
                "robot": {
                    "robot_config_path": "configs/robots/ar5v2_l.yaml",
                    "default_tcp_frame": "tool",
                },
                "kinematics": {"ik": {"collision_cache": {"cuboid": 2}}},
                "motion_planner": {"collision_cache": {"cuboid": 4}},
            }
        }
    )
    context = CuroboContext.__new__(CuroboContext)
    context.config = config
    context.kinematics = SimpleNamespace(total_spheres=lambda: 3)
    context._ik_solver = None
    context._motion_planner = None
    context._batch_motion_planner = None
    context._collision_world = SimpleNamespace(
        materialized_counts={"cuboid": 5},
        scene_cfg="scene",
    )
    context.record_collision_sync(1, "batch-view")
    solver = _FakeSolver()
    context._make_motion_planner = lambda *, batch: solver

    capability = context.ensure_collision_checker("batch_planner")

    assert capability.available is False
    assert "scene_collision_cache_capacity" in capability.missing_requirements
    assert solver.world_updates == []


def test_cache_validation_only_checks_existing_consumers() -> None:
    config = CuroboConfig.from_mapping(
        {
            "curobo": {
                "enabled": True,
                "planning_joint_group": "arm",
                "robot": {
                    "robot_config_path": "configs/robots/ar5v2_l.yaml",
                    "default_tcp_frame": "tool",
                },
                "kinematics": {"ik": {"collision_cache": {"cuboid": 2}}},
                "motion_planner": {"collision_cache": {"cuboid": 6}},
            }
        }
    )
    context = CuroboContext.__new__(CuroboContext)
    context.config = config
    context.kinematics = SimpleNamespace(total_spheres=lambda: 3)
    context._ik_solver = None
    context._motion_planner = _FakeSolver()
    context._batch_motion_planner = None

    context.validate_collision_cache_capacity({"cuboid": 5})


def test_curobo_context_rejects_materialized_world_larger_than_cache() -> None:
    config = CuroboConfig.from_mapping(
        {
            "curobo": {
                "enabled": True,
                "planning_joint_group": "arm",
                "robot": {
                    "robot_config_path": "configs/robots/ar5v2_l.yaml",
                    "default_tcp_frame": "tool",
                },
                "kinematics": {"ik": {"collision_cache": {"cuboid": 4}}},
                "motion_planner": {"collision_cache": {"cuboid": 6}},
            }
        }
    )
    context = CuroboContext.__new__(CuroboContext)
    context.config = config
    context.kinematics = SimpleNamespace(total_spheres=2)
    context._ik_solver = _FakeSolver()
    context._motion_planner = None
    context._batch_motion_planner = None

    try:
        context.validate_collision_cache_capacity(
            {"cuboid": 5, "mesh": 0},
            consumer="ik",
        )
    except ValueError as exc:
        assert "IK collision cache capacity" in str(exc)
        assert "required=5" in str(exc)
        assert "configured=4" in str(exc)
    else:
        raise AssertionError("undersized cuRobo collision cache was accepted")
