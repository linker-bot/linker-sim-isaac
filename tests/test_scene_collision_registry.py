from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.app.runtime.collision.object_provider import (
    collision_objects_from_runtime_objects,
)
from linkerbot_sim.app.runtime.robot_registry import (
    RobotPlanningRegistry,
    RobotRegistry,
)
from linkerbot_sim.backends.curobo.context import CuroboContext
from linkerbot_sim.app.runtime.collision.registry import SceneCollisionRegistry
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.robots.capabilities import PlanningCapability, RobotKind


def _obstacle(name: str, x: float = 0.0) -> CollisionObject:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = x
    return CollisionObject(name, "sphere", pose, (0.1,))


class _Context:
    def __init__(self) -> None:
        self.sync_calls: list[tuple[str, ...]] = []
        self.closed = False

    def sync_collision_world(self, objects):
        self.sync_calls.append(tuple(item.name for item in objects))
        return self.sync_calls[-1]

    def record_collision_sync(self, version, fingerprint) -> None:
        self.version = version
        self.fingerprint = fingerprint

    def close(self) -> None:
        self.closed = True


def _robot(robot_id: int):
    capability = PlanningCapability(
        kind=RobotKind.ARM,
        backend_enabled=True,
        planning_joint_group="arm",
        kinematics_binding_valid=True,
        arm_joint_mapping_valid=True,
    )
    return SimpleNamespace(
        robot_id=robot_id,
        label=f"r{robot_id}",
        planning_capability=capability,
        curobo_config=SimpleNamespace(robot=None),
        joint_groups=SimpleNamespace(arm=(f"j{robot_id}",)),
    )


def test_snapshot_is_shared_but_target_views_exclude_self() -> None:
    registry = SceneCollisionRegistry()
    registry.register_provider("object", lambda: (_obstacle("table"),), source="object")
    registry.register_provider(
        "robot0", lambda: (_obstacle("r0"),), owner_robot_id=0, source="robot"
    )
    registry.register_provider(
        "robot1", lambda: (_obstacle("r1"),), owner_robot_id=1, source="robot"
    )

    snapshot = registry.snapshot()

    assert [item.name for item in snapshot.collision_objects_for(0)] == ["table", "r1"]
    assert [item.name for item in snapshot.collision_objects_for(1)] == ["table", "r0"]
    assert registry.snapshot() is snapshot


def test_context_pool_isolated_by_consumer_and_syncs_if_dirty_or_forced() -> None:
    robots = RobotRegistry((_robot(0), _robot(1)))
    contexts: list[_Context] = []

    def factory(robot):
        context = _Context()
        contexts.append(context)
        return context

    planning = RobotPlanningRegistry(robots, context_factory=factory)
    collision = SceneCollisionRegistry()
    collision.register_provider("table", lambda: (_obstacle("table"),))
    snapshot = collision.snapshot()

    planning.sync_before_plan(0, snapshot, consumer_role="interactive")
    planning.sync_before_plan(0, snapshot, consumer_role="interactive")
    planning.sync_before_plan(0, snapshot, consumer_role="interactive", force=True)
    planning.context(0, consumer_role="planner", worker_slot=1)

    assert len(contexts) == 2
    assert len(contexts[0].sync_calls) == 2
    assert contexts[0] is not contexts[1]
    planning.close()
    assert all(context.closed for context in contexts)


def test_context_pool_closes_peers_and_retains_failed_context_for_retry() -> None:
    robots = RobotRegistry((_robot(0), _robot(1)))

    class RetriableContext(_Context):
        def __init__(self, *, fail_once: bool) -> None:
            super().__init__()
            self.fail_once = fail_once
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.fail_once and self.close_calls == 1:
                raise RuntimeError("context close failed")
            self.closed = True

    contexts = [RetriableContext(fail_once=True), RetriableContext(fail_once=False)]
    planning = RobotPlanningRegistry(
        robots,
        context_factory=lambda robot: contexts[int(robot.robot_id)],
    )
    planning.context(0)
    planning.context(1)

    with pytest.raises(RuntimeError, match="context close failed"):
        planning.close()

    assert contexts[0].closed is False
    assert contexts[1].closed is True
    assert planning.metrics()["context_count"] == 1

    planning.close()
    assert contexts[0].closed is True
    assert planning.metrics()["context_count"] == 0


def test_curobo_context_close_attempts_all_solvers_and_retries_failure() -> None:
    calls: list[str] = []

    class Solver:
        def __init__(self, name: str, *, fail_once: bool = False) -> None:
            self.name = name
            self.fail_once = fail_once
            self.attempts = 0

        def destroy(self) -> None:
            self.attempts += 1
            calls.append(self.name)
            if self.fail_once and self.attempts == 1:
                raise RuntimeError("solver destroy failed")

    context = CuroboContext.__new__(CuroboContext)
    failing = Solver("motion", fail_once=True)
    batch = Solver("batch")
    ik = Solver("ik")
    context._motion_planner = failing
    context._batch_motion_planner = batch
    context._ik_solver = ik

    with pytest.raises(RuntimeError, match="solver destroy failed"):
        context.close()

    assert calls == ["motion", "batch", "ik"]
    assert context._motion_planner is failing
    assert context._batch_motion_planner is None
    assert context._ik_solver is None

    context.close()
    assert calls == ["motion", "batch", "ik", "motion"]
    assert context._motion_planner is None


def test_object_collision_pose_does_not_fallback_when_stage_prim_is_missing() -> None:
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    handle = SimpleNamespace(
        name="block",
        model=SimpleNamespace(prim_path="/World/Missing"),
        config=SimpleNamespace(
            root_pose=SimpleNamespace(xyz=(1.0, 2.0, 3.0), rpy=(0.0, 0.0, 0.0)),
            planning_collision=SimpleNamespace(
                shape="sphere",
                size=(0.1,),
                xyz=(0.0, 0.0, 0.0),
                rpy=(0.0, 0.0, 0.0),
                enabled=True,
                padding=0.0,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="prim does not exist"):
        collision_objects_from_runtime_objects((handle,), stage=stage)

    fallback = collision_objects_from_runtime_objects((handle,), stage=None)
    np.testing.assert_allclose(fallback[0].pose[:3, 3], [1.0, 2.0, 3.0])
