from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.snapshots import (
    get_dual_robot_snapshot,
    get_single_robot_snapshot,
    set_dual_robot_snapshot,
    set_single_robot_snapshot,
    set_snapshot,
)


class FakeArticulation:
    def __init__(
        self,
        *,
        dof_names=("j0", "free", "j1"),
        positions=(0.0, 0.0, 0.0),
        velocities=(0.0, 0.0, 0.0),
    ) -> None:
        self.dof_names = tuple(dof_names)
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def set_joint_positions(self, values):
        self.positions = np.asarray(values, dtype=float).copy()

    def set_joint_velocities(self, values):
        self.velocities = np.asarray(values, dtype=float).copy()


class FakeController:
    def __init__(self, *, command_indices=(0, 2), command_joint_names=("j0", "j1")):
        self.command_indices = np.asarray(command_indices, dtype=int)
        self.command_joint_names = tuple(command_joint_names)
        self.last_commanded_efforts = np.ones(3, dtype=float)


class FakeObserver:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def make_single_runtime(*, positions=(0.1, 9.0, 0.3), velocities=(1.0, 2.0, 3.0)):
    observer = FakeObserver()
    execution = SimpleNamespace(
        articulation=FakeArticulation(positions=positions, velocities=velocities),
        joint_controller=FakeController(),
        state_observer=observer,
        camera_observer=None,
    )
    return SimpleNamespace(
        execution=execution,
        session=SimpleNamespace(stage=None),
        object_handles=(),
        imported_robot=None,
        observer=observer,
    )


def make_dual_runtime():
    left = SimpleNamespace(
        side="left",
        articulation=FakeArticulation(positions=(0.1, 9.0, 0.2)),
        joint_controller=FakeController(),
    )
    right = SimpleNamespace(
        side="right",
        articulation=FakeArticulation(positions=(0.4, 8.0, 0.5)),
        joint_controller=FakeController(),
    )

    def side(name: str):
        return left if name == "left" else right

    execution = SimpleNamespace(
        left=left,
        right=right,
        side=side,
        state_observer=FakeObserver(),
        camera_observer=None,
    )
    return SimpleNamespace(
        execution=execution,
        session=SimpleNamespace(stage=None),
        object_handles=(),
        imported={},
    )


def test_single_snapshot_roundtrip_updates_command_joints_only() -> None:
    source = make_single_runtime()
    target = make_single_runtime(positions=(9.0, 7.0, 8.0), velocities=(0.0, 0.0, 0.0))

    snapshot = get_single_robot_snapshot(source)
    result = set_single_robot_snapshot(target, snapshot)

    assert result.accepted
    assert result.robots == ("single",)
    np.testing.assert_allclose(
        target.execution.articulation.positions,
        [0.1, 7.0, 0.3],
    )
    np.testing.assert_allclose(
        target.execution.articulation.velocities,
        [1.0, 0.0, 3.0],
    )
    assert np.isnan(target.execution.joint_controller.last_commanded_efforts).all()
    assert target.observer.reset_calls == 1


def test_dual_snapshot_roundtrip_restores_both_sides() -> None:
    source = make_dual_runtime()
    target = make_dual_runtime()
    target.execution.left.articulation.positions[:] = [9.0, 9.0, 9.0]
    target.execution.right.articulation.positions[:] = [8.0, 8.0, 8.0]

    snapshot = get_dual_robot_snapshot(source)
    result = set_dual_robot_snapshot(target, snapshot)

    assert result.accepted
    assert result.robots == ("left", "right")
    np.testing.assert_allclose(
        target.execution.left.articulation.positions, [0.1, 9.0, 0.2]
    )
    np.testing.assert_allclose(
        target.execution.right.articulation.positions, [0.4, 8.0, 0.5]
    )


def test_dual_side_snapshot_can_restore_single_with_robot_map() -> None:
    source = make_dual_runtime()
    target = make_single_runtime(positions=(0.0, 6.0, 0.0))

    snapshot = get_dual_robot_snapshot(source)
    result = set_single_robot_snapshot(
        target,
        snapshot,
        robot_map={"right": "single"},
    )

    assert result.accepted
    assert result.partial
    assert result.robots == ("single",)
    np.testing.assert_allclose(target.execution.articulation.positions, [0.4, 6.0, 0.5])


def test_dispatch_set_snapshot_supports_single_runtime() -> None:
    source = make_single_runtime(positions=(0.2, 3.0, 0.4))
    target = make_single_runtime(positions=(0.0, 3.0, 0.0))

    result = set_snapshot(target, get_single_robot_snapshot(source))

    assert result.accepted
    np.testing.assert_allclose(target.execution.articulation.positions, [0.2, 3.0, 0.4])
