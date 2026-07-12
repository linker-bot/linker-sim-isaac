from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.snapshots import get_snapshot, set_snapshot
from linkerbot_sim.snapshots.transactions import (
    RuntimeMutationRejected,
    SnapshotRollbackError,
)


class _Articulation:
    dof_names = ("fixed", "j0", "j1")

    def __init__(self) -> None:
        self.positions = np.asarray([9.0, 0.1, 0.2], dtype=float)
        self.velocities = np.asarray([8.0, 0.3, 0.4], dtype=float)

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def set_joint_positions(self, values):
        self.positions = np.asarray(values, dtype=float).copy()

    def set_joint_velocities(self, values):
        self.velocities = np.asarray(values, dtype=float).copy()


class _ResetObserver:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FailingArticulation(_Articulation):
    def __init__(self, *, offset: float, fail_position_calls: set[int]) -> None:
        super().__init__()
        self.positions += float(offset)
        self.velocities += float(offset)
        self.position_calls = 0
        self.fail_position_calls = set(fail_position_calls)

    def set_joint_positions(self, values):
        self.position_calls += 1
        if self.position_calls in self.fail_position_calls:
            raise RuntimeError(f"position setter {self.position_calls} failed")
        super().set_joint_positions(values)


def _transaction_single_scene_runtime(
    articulations: dict[str, _FailingArticulation],
) -> SimpleNamespace:
    robots = {}
    for robot_id, (label, articulation) in enumerate(articulations.items()):
        controller = SimpleNamespace(
            command_indices=np.asarray([1, 2], dtype=int),
            command_joint_names=("j0", "j1"),
            last_commanded_efforts=np.asarray(
                [robot_id + 0.1, robot_id + 0.2, robot_id + 0.3],
                dtype=float,
            ),
        )
        robots[robot_id] = SimpleNamespace(
            label=label,
            profile_name=None,
            imported=None,
            execution=SimpleNamespace(
                articulation=articulation,
                joint_controller=controller,
                state_observer=_ResetObserver(),
                camera_observer=None,
            ),
        )
    by_label = {robot.label: robot for robot in robots.values()}
    collision_registry = SimpleNamespace(mark_dirty=lambda: None)
    return SimpleNamespace(
        robots_by_id=robots,
        robot_id_by_label={label: index for index, label in enumerate(by_label)},
        robot_registry=object(),
        session=SimpleNamespace(stage=None),
        object_handles=(),
        object_state_views={},
        collision_registry=collision_registry,
        quit_event=threading.Event(),
        robot_by_label=by_label.__getitem__,
    )


def _changed_single_scene_snapshot(runtime: object) -> dict[str, object]:
    payload = get_snapshot(runtime).as_dict()
    for index, robot in enumerate(payload["robots"]):
        robot["joint_positions"] = [10.0 + index, 20.0 + index]
        robot["joint_velocities"] = [30.0 + index, 40.0 + index]
    return payload


def test_single_scene_snapshot_dispatch_restores_only_command_joints_and_caches() -> (
    None
):
    articulation = _Articulation()
    observer = _ResetObserver()
    controller = SimpleNamespace(
        command_indices=np.asarray([1, 2], dtype=int),
        command_joint_names=("j0", "j1"),
        last_commanded_efforts=np.zeros(3, dtype=float),
    )
    execution = SimpleNamespace(
        articulation=articulation,
        joint_controller=controller,
        state_observer=observer,
        camera_observer=None,
    )
    robot = SimpleNamespace(
        label="arm",
        profile_name=None,
        imported=None,
        execution=execution,
    )
    collision_registry = SimpleNamespace(mark_dirty_calls=0)

    def mark_dirty() -> None:
        collision_registry.mark_dirty_calls += 1

    collision_registry.mark_dirty = mark_dirty
    runtime = SimpleNamespace(
        robots_by_id={0: robot},
        robot_id_by_label={"arm": 0},
        robot_registry=object(),
        session=SimpleNamespace(stage=None),
        object_handles=(),
        config_fingerprint="scene-config",
        collision_registry=collision_registry,
        robot_by_label=lambda label: robot if label == "arm" else None,
    )

    snapshot = get_snapshot(runtime)

    assert snapshot.metadata.source_runtime == "single_scene"
    np.testing.assert_allclose(snapshot.robots["arm"].joint_positions, [0.1, 0.2])
    np.testing.assert_allclose(snapshot.robots["arm"].joint_velocities, [0.3, 0.4])
    articulation.positions[:] = [7.0, 1.0, 2.0]
    articulation.velocities[:] = [6.0, 3.0, 4.0]

    result = set_snapshot(runtime, snapshot)

    assert result.accepted is True
    assert result.robots == ("arm",)
    np.testing.assert_allclose(articulation.positions, [7.0, 0.1, 0.2])
    np.testing.assert_allclose(articulation.velocities, [6.0, 0.3, 0.4])
    assert np.isnan(controller.last_commanded_efforts).all()
    assert observer.reset_calls == 1
    assert collision_registry.mark_dirty_calls == 1


def test_single_scene_snapshot_rolls_back_all_robots_and_runtime_can_continue() -> None:
    articulations = {
        "left": _FailingArticulation(offset=0.0, fail_position_calls=set()),
        "right": _FailingArticulation(offset=1.0, fail_position_calls={1}),
    }
    runtime = _transaction_single_scene_runtime(articulations)
    payload = _changed_single_scene_snapshot(runtime)
    original_positions = {
        name: articulation.positions.copy()
        for name, articulation in articulations.items()
    }
    original_efforts = {
        robot.label: robot.execution.joint_controller.last_commanded_efforts.copy()
        for robot in runtime.robots_by_id.values()
    }

    with pytest.raises(RuntimeError, match="position setter 1 failed"):
        set_snapshot(runtime, payload)

    for name, articulation in articulations.items():
        np.testing.assert_allclose(articulation.positions, original_positions[name])
        np.testing.assert_allclose(
            runtime.robot_by_label(
                name
            ).execution.joint_controller.last_commanded_efforts,
            original_efforts[name],
        )
    assert not hasattr(runtime, "fatal_error")
    assert not runtime.quit_event.is_set()

    result = set_snapshot(runtime, payload)
    assert result.accepted
    np.testing.assert_allclose(articulations["left"].positions[1:], [10.0, 20.0])
    np.testing.assert_allclose(articulations["right"].positions[1:], [11.0, 21.0])


def test_single_scene_snapshot_rollback_failure_fail_stops_future_mutations() -> None:
    articulations = {
        "left": _FailingArticulation(offset=0.0, fail_position_calls={2}),
        "right": _FailingArticulation(offset=1.0, fail_position_calls={1}),
    }
    runtime = _transaction_single_scene_runtime(articulations)
    payload = _changed_single_scene_snapshot(runtime)

    with pytest.raises(SnapshotRollbackError) as exc_info:
        set_snapshot(runtime, payload)

    assert isinstance(exc_info.value.cause, RuntimeError)
    assert "position setter 1 failed" in str(exc_info.value.cause)
    assert runtime.quit_event.is_set()
    assert "rollback_errors" in runtime.fatal_error
    calls = {name: item.position_calls for name, item in articulations.items()}

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        set_snapshot(runtime, payload)
    assert {name: item.position_calls for name, item in articulations.items()} == calls
