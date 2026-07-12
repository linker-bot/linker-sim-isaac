from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.app.runtime.single_scene_reset import reset_single_scene_runtime
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.snapshots.transactions import (
    RuntimeMutationRejected,
    SnapshotRollbackError,
)


class _PhysicsContext:
    def __init__(self) -> None:
        self.gravity = None

    def set_gravity(self, value) -> None:
        self.gravity = float(value)


class _World:
    def __init__(self, *, fail_reset: bool = False) -> None:
        self.reset_count = 0
        self.physics_context = _PhysicsContext()
        self.fail_reset = fail_reset

    def reset(self) -> None:
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError("World.reset failed")

    def get_physics_context(self):
        return self.physics_context


class _Articulation:
    num_dof = 2

    def __init__(self) -> None:
        self.gravity_disabled = False
        self.velocities = None

    def disable_gravity(self) -> None:
        self.gravity_disabled = True

    def set_joint_velocities(self, values) -> None:
        self.velocities = np.asarray(values, dtype=float).copy()


class _Controller:
    def __init__(self, *, fail_configure: bool = False) -> None:
        self.configure_count = 0
        self.last_commanded_efforts = np.asarray([1.0, 2.0], dtype=float)
        self.fail_configure = fail_configure

    def configure_runtime(self) -> None:
        self.configure_count += 1
        if self.fail_configure:
            raise RuntimeError("controller configure failed")


class _Observer:
    def __init__(self, *, fail_reset: bool = False) -> None:
        self.reset_count = 0
        self.fail_reset = fail_reset

    def reset(self) -> None:
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError("observer reset failed")


class _App:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _CollisionRegistry:
    def __init__(self, *, fail_mark_dirty: bool = False) -> None:
        self.dirty = False
        self.fail_mark_dirty = fail_mark_dirty

    def mark_dirty(self) -> None:
        self.dirty = True
        if self.fail_mark_dirty:
            raise RuntimeError("collision invalidation failed")


def _make_runtime(
    *,
    failed_root_pose_calls: set[int] = frozenset(),
    failure_point: str | None = None,
) -> SimpleNamespace:
    world = _World(fail_reset=failure_point == "world")
    root_pose_calls: list[tuple[str, tuple[float, float, float]]] = []
    root_poses = {
        "/World/Robot0": RootPoseConfig(xyz=(10.0, 0.0, 0.0)),
        "/World/Robot1": RootPoseConfig(xyz=(-10.0, 0.0, 0.0)),
    }
    robots = {}
    for robot_id, x in enumerate((1.0, -1.0)):
        articulation = _Articulation()
        controller = _Controller(
            fail_configure=failure_point == "controller" and robot_id == 0
        )
        robots[robot_id] = SimpleNamespace(
            robot_id=robot_id,
            label=f"robot_{robot_id}",
            imported=SimpleNamespace(imported_root_path=f"/World/Robot{robot_id}"),
            scene_instance=SimpleNamespace(root_pose=RootPoseConfig(xyz=(x, 0.0, 0.0))),
            prepared=SimpleNamespace(
                articulation=articulation,
                joint_controller=controller,
                gravity_policy=SimpleNamespace(
                    disables_all_known_components=lambda: True
                ),
            ),
            execution=SimpleNamespace(
                state_observer=_Observer(
                    fail_reset=failure_point == "observer" and robot_id == 0
                ),
                camera_observer=None,
            ),
        )
    runtime_observer = _Observer()
    collision_registry = _CollisionRegistry(
        fail_mark_dirty=failure_point == "collision"
    )
    runtime = SimpleNamespace(
        session=SimpleNamespace(stage=object(), world=world, app=_App()),
        env_config={
            "env": {"gravity_z": -3.0},
            "robots": [],
            "objects": [],
        },
        object_handles=(),
        robots_by_id=robots,
        state_observer=runtime_observer,
        camera_observer=None,
        collision_registry=collision_registry,
    )

    def apply_root_pose(_stage, path: str, pose: RootPoseConfig) -> None:
        root_pose_calls.append((path, pose.xyz))
        if len(root_pose_calls) in failed_root_pose_calls:
            raise RuntimeError(f"root pose setter {len(root_pose_calls)} failed")
        root_poses[path] = pose

    return SimpleNamespace(
        runtime=runtime,
        world=world,
        robots=robots,
        runtime_observer=runtime_observer,
        collision_registry=collision_registry,
        root_poses=root_poses,
        root_pose_calls=root_pose_calls,
        root_pose_applier=apply_root_pose,
        root_pose_reader=lambda _stage, path: root_poses[path],
    )


def test_reset_single_scene_runtime_restores_all_registered_robots() -> None:
    fixture = _make_runtime()

    result = reset_single_scene_runtime(
        fixture.runtime,
        robot_root_pose_applier=fixture.root_pose_applier,
        root_pose_reader=fixture.root_pose_reader,
    )

    assert result.step == 0
    assert fixture.world.reset_count == 1
    assert fixture.world.physics_context.gravity == -3.0
    assert fixture.root_pose_calls == [
        ("/World/Robot0", (1.0, 0.0, 0.0)),
        ("/World/Robot1", (-1.0, 0.0, 0.0)),
    ]
    for robot in fixture.robots.values():
        np.testing.assert_allclose(robot.prepared.articulation.velocities, [0.0, 0.0])
        assert robot.prepared.articulation.gravity_disabled is True
        assert robot.prepared.joint_controller.configure_count == 1
        assert np.isnan(robot.prepared.joint_controller.last_commanded_efforts).all()
    assert fixture.runtime_observer.reset_count == 1
    assert fixture.collision_registry.dirty is True


def test_root_pose_write_failure_rolls_back_and_allows_retry() -> None:
    fixture = _make_runtime(failed_root_pose_calls={2})
    original_root_poses = fixture.root_poses.copy()

    with pytest.raises(RuntimeError, match="root pose setter 2 failed"):
        reset_single_scene_runtime(
            fixture.runtime,
            robot_root_pose_applier=fixture.root_pose_applier,
            root_pose_reader=fixture.root_pose_reader,
        )

    assert fixture.root_poses == original_root_poses
    assert fixture.root_pose_calls == [
        ("/World/Robot0", (1.0, 0.0, 0.0)),
        ("/World/Robot1", (-1.0, 0.0, 0.0)),
        ("/World/Robot1", (-10.0, 0.0, 0.0)),
        ("/World/Robot0", (10.0, 0.0, 0.0)),
    ]
    assert not hasattr(fixture.runtime, "fatal_error")
    assert fixture.runtime.session.app.close_count == 0
    assert fixture.world.reset_count == 0

    reset_single_scene_runtime(
        fixture.runtime,
        robot_root_pose_applier=fixture.root_pose_applier,
        root_pose_reader=fixture.root_pose_reader,
    )
    assert fixture.world.reset_count == 1


def test_root_pose_rollback_failure_permanently_fail_stops() -> None:
    fixture = _make_runtime(failed_root_pose_calls={2, 4})

    with pytest.raises(SnapshotRollbackError) as exc_info:
        reset_single_scene_runtime(
            fixture.runtime,
            robot_root_pose_applier=fixture.root_pose_applier,
            root_pose_reader=fixture.root_pose_reader,
        )

    assert isinstance(exc_info.value.cause, RuntimeError)
    assert "root pose setter 2 failed" in str(exc_info.value.cause)
    assert "root pose setter 4 failed" in fixture.runtime.fatal_error
    assert fixture.runtime.session.app.close_count == 1
    calls_before_retry = list(fixture.root_pose_calls)

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        reset_single_scene_runtime(
            fixture.runtime,
            robot_root_pose_applier=fixture.root_pose_applier,
            root_pose_reader=fixture.root_pose_reader,
        )
    assert fixture.root_pose_calls == calls_before_retry


@pytest.mark.parametrize(
    "failure_point", ["world", "controller", "observer", "collision"]
)
def test_failure_at_or_after_world_reset_permanently_fail_stops(
    failure_point: str,
) -> None:
    fixture = _make_runtime(failure_point=failure_point)
    original_root_poses = fixture.root_poses.copy()

    with pytest.raises(RuntimeError):
        reset_single_scene_runtime(
            fixture.runtime,
            robot_root_pose_applier=fixture.root_pose_applier,
            root_pose_reader=fixture.root_pose_reader,
        )

    assert fixture.root_poses == original_root_poses
    assert "irreversible_steps=['World.reset']" in fixture.runtime.fatal_error
    assert fixture.runtime.session.app.close_count == 1
    calls_before_retry = list(fixture.root_pose_calls)

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        reset_single_scene_runtime(
            fixture.runtime,
            robot_root_pose_applier=fixture.root_pose_applier,
            root_pose_reader=fixture.root_pose_reader,
        )
    assert fixture.root_pose_calls == calls_before_retry


def test_single_scene_reset_rejects_fatal_runtime_before_accessing_session() -> None:
    runtime = SimpleNamespace(fatal_error="snapshot rollback failed")

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        reset_single_scene_runtime(runtime)
