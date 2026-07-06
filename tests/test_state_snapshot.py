from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.telemetry.state_snapshot import (
    DualRobotStateSampler,
    StateStream,
)


class _World:
    def get_physics_dt(self) -> float:
        return 0.1


class _Articulation:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.dof_names = names
        self.num_dof = len(names)
        self.positions = np.zeros(self.num_dof, dtype=float)
        self.velocities = np.zeros(self.num_dof, dtype=float)

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def get_measured_joint_efforts(self, joint_indices=None):
        values = np.asarray([1.0, 2.0], dtype=float)[: self.num_dof]
        return values if joint_indices is None else values[joint_indices]

    def get_applied_joint_efforts(self, joint_indices=None):
        values = np.asarray([3.0, 4.0], dtype=float)[: self.num_dof]
        return values if joint_indices is None else values[joint_indices]


def _runtime() -> SimpleNamespace:
    left_articulation = _Articulation(("l0", "l1"))
    right_articulation = _Articulation(("r0", "r1"))
    return SimpleNamespace(
        simulation_world=_World(),
        left=SimpleNamespace(
            side="left",
            articulation=left_articulation,
            joint_controller=SimpleNamespace(
                last_commanded_efforts=np.asarray([np.nan, 0.2], dtype=float)
            ),
        ),
        right=SimpleNamespace(
            side="right",
            articulation=right_articulation,
            joint_controller=SimpleNamespace(
                last_commanded_efforts=np.asarray([0.3, 0.4], dtype=float)
            ),
        ),
    )


def test_dual_state_sampler_differentiates_joint_acceleration() -> None:
    runtime = _runtime()
    sampler = DualRobotStateSampler(
        stage=None,
        rate_hz=10.0,
        include_efforts=True,
        include_objects=False,
    )

    first = sampler.sample(runtime, step=0, phase="initial")
    runtime.left.articulation.velocities = np.asarray([0.2, -0.1], dtype=float)
    second = sampler.sample(runtime, step=1, phase="next")

    assert first.step == 0
    assert first.time_s == 0.1
    np.testing.assert_allclose(second.robots[0].accelerations_rad_s2, [2.0, -1.0])
    np.testing.assert_allclose(second.robots[0].measured_efforts, [1.0, 2.0])
    assert second.as_dict()["robots"]["left"]["commanded_efforts"][0] is None


def test_state_stream_keeps_latest_snapshot() -> None:
    stream = StateStream()
    snapshot = DualRobotStateSampler(
        stage=None, rate_hz=1.0, include_efforts=False
    ).sample(_runtime(), step=4)

    sequence = stream.publish(snapshot)
    latest = stream.latest()
    waited = stream.wait_next(0, timeout_s=0.01)

    assert sequence == 1
    assert latest == (1, snapshot)
    assert waited == (1, snapshot)
