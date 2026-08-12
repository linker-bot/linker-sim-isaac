from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.telemetry.state_snapshot import SceneRobotStateSampler, StateStream


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
    robots = {}
    for robot_id, label in enumerate(("robot_a", "robot_b")):
        articulation = _Articulation((f"j{robot_id}_0", f"j{robot_id}_1"))
        robots[robot_id] = SimpleNamespace(
            robot_id=robot_id,
            label=label,
            execution=SimpleNamespace(
                articulation=articulation,
                joint_controller=SimpleNamespace(
                    last_commanded_efforts=np.asarray(
                        [np.nan, 0.2 + robot_id], dtype=float
                    )
                ),
            ),
        )
    return SimpleNamespace(world=_World(), robots_by_id=robots)


def test_scene_state_sampler_differentiates_joint_acceleration() -> None:
    runtime = _runtime()
    sampler = SceneRobotStateSampler(
        stage=None,
        rate_hz=10.0,
        include_efforts=True,
        include_objects=False,
    )

    first = sampler.sample(runtime, step=0, phase="initial")
    runtime.robots_by_id[0].execution.articulation.velocities = np.asarray(
        [0.2, -0.1], dtype=float
    )
    second = sampler.sample(runtime, step=1, phase="next")

    assert first.step == 0
    assert first.time_s == 0.1
    np.testing.assert_allclose(second.robots[0].accelerations_rad_s2, [2.0, -1.0])
    np.testing.assert_allclose(second.robots[0].measured_efforts, [1.0, 2.0])
    payload = second.as_dict()
    assert payload["robots"][0]["robot_id"] == 0
    assert payload["robots"][0]["label"] == "robot_a"
    assert payload["robots"][0]["commanded_efforts"][0] is None


def test_scene_state_sampler_reset_clears_acceleration_history() -> None:
    runtime = _runtime()
    sampler = SceneRobotStateSampler(stage=None, rate_hz=10.0)

    sampler.sample(runtime, step=0, phase="before_reset")
    runtime.robots_by_id[0].execution.articulation.velocities = np.asarray(
        [0.2, -0.1], dtype=float
    )
    sampler.reset()
    after_reset = sampler.sample(runtime, step=0, phase="after_reset")

    assert np.isnan(after_reset.robots[0].accelerations_rad_s2).all()


def test_scene_state_sampler_freezes_cached_hybrid_diagnostics() -> None:
    source = {
        "active": True,
        "request_id": "hybrid-1",
        "force_axes": [False, False, True, False, False, False],
    }
    sampler = SceneRobotStateSampler(
        stage=None,
        rate_hz=10.0,
        include_hybrid_control=True,
        hybrid_diagnostics_provider=lambda: source,
    )

    snapshot = sampler.sample(_runtime(), step=0)
    source["force_axes"][2] = False  # type: ignore[index]

    assert snapshot.hybrid_control == {
        "active": True,
        "request_id": "hybrid-1",
        "force_axes": [False, False, True, False, False, False],
    }
    payload = snapshot.as_dict()
    payload["hybrid_control"]["force_axes"][2] = False  # type: ignore[index]
    assert snapshot.hybrid_control["force_axes"][2] is True  # type: ignore[index]


def test_scene_state_sampler_omits_disabled_and_collapses_inactive_hybrid_data() -> (
    None
):
    calls = 0

    def provider() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"active": False, "stale": "must-not-leak"}

    disabled = SceneRobotStateSampler(
        stage=None,
        include_hybrid_control=False,
        hybrid_diagnostics_provider=provider,
    ).sample(_runtime(), step=0)
    enabled = SceneRobotStateSampler(
        stage=None,
        include_hybrid_control=True,
        hybrid_diagnostics_provider=provider,
    ).sample(_runtime(), step=0)

    assert disabled.hybrid_control is None
    assert "hybrid_control" not in disabled.as_dict()
    assert enabled.hybrid_control == {"active": False}
    assert calls == 1


def test_state_stream_keeps_latest_snapshot() -> None:
    stream = StateStream()
    snapshot = SceneRobotStateSampler(stage=None, rate_hz=1.0).sample(
        _runtime(), step=4
    )

    sequence = stream.publish(snapshot)
    latest = stream.latest()
    waited = stream.wait_next(0, timeout_s=0.01)

    assert sequence == 1
    assert latest == (1, snapshot)
    assert waited == (1, snapshot)


def test_state_stream_applies_bounded_drop_policies() -> None:
    snapshots = tuple(
        SceneRobotStateSampler(stage=None, rate_hz=1.0).sample(_runtime(), step=step)
        for step in range(3)
    )
    oldest = StateStream(capacity=2, drop_policy="drop_oldest")
    newest = StateStream(capacity=2, drop_policy="drop_newest")
    latest = StateStream(capacity=3, drop_policy="latest")

    for snapshot in snapshots:
        oldest.publish(snapshot)
        newest.publish(snapshot)
        latest.publish(snapshot)

    assert oldest.wait_next(0, timeout_s=0.01) == (2, snapshots[1])
    assert newest.wait_next(0, timeout_s=0.01) == (1, snapshots[0])
    assert latest.wait_next(0, timeout_s=0.01) == (3, snapshots[2])
    assert oldest.status()["dropped_snapshots"] == 1
    assert newest.status()["dropped_snapshots"] == 1
    assert latest.status()["dropped_snapshots"] == 2


def test_state_stream_rejects_invalid_capacity_and_drop_policy() -> None:
    with pytest.raises(ValueError, match="capacity"):
        StateStream(capacity=0)
    with pytest.raises(ValueError, match="drop_policy"):
        StateStream(drop_policy="block")
