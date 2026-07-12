from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.planning.backend import PlannerBackend, normalize_planner_backend
from linkerbot_sim.planning.linear_backend import LinearPlannerBackend
from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    MotionRequest,
    TaskSpacePath,
    TcpLineSegment,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def test_linear_planner_implements_shared_backend_contract() -> None:
    backend = LinearPlannerBackend(("j0", "j1"))

    result = backend.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 1.0]),
            goal_q=np.asarray([1.0, 3.0]),
            duration_s=0.11,
            sample_dt_s=0.05,
        )
    )

    assert isinstance(backend, PlannerBackend)
    assert result.success is True
    assert isinstance(result.trajectory, JointTrajectory)
    np.testing.assert_allclose(result.trajectory.times, [0.0, 0.05, 0.1, 0.11])
    np.testing.assert_allclose(result.trajectory.positions[0], [0.0, 1.0])
    np.testing.assert_allclose(result.trajectory.positions[-1], [1.0, 3.0])


def test_linear_planner_reports_unsupported_capabilities() -> None:
    backend = LinearPlannerBackend(("j0",))

    collision_result = backend.plan(
        MotionRequest(
            current_q=np.asarray([0.0]),
            goal_q=np.asarray([1.0]),
            duration_s=0.1,
            sample_dt_s=0.05,
            avoid_collisions=True,
        )
    )
    path_result = backend.plan(
        LinearPosePathRequest(
            current_q=np.asarray([0.0]),
            path=TaskSpacePath(
                (TcpLineSegment(target_offset=np.asarray([0.0, 0.0, 0.1])),)
            ),
            duration_s=0.1,
            sample_dt_s=0.05,
        )
    )

    assert collision_result.status == "COLLISION_UNSUPPORTED"
    assert path_result.status == "UNSUPPORTED"
    assert normalize_planner_backend(" LINEAR ") == "linear"


def test_linear_planner_uses_injected_duration_and_physics_dt_defaults() -> None:
    backend = LinearPlannerBackend(
        ("j0",),
        default_duration_s=0.3,
        default_sample_dt_s=0.1,
    )

    result = backend.plan(
        MotionRequest(
            current_q=np.asarray([0.0]),
            goal_q=np.asarray([1.0]),
        )
    )

    assert result.success is True
    assert result.diagnostics.metrics["duration_s"] == pytest.approx(0.3)
    assert result.diagnostics.metrics["sample_dt_s"] == pytest.approx(0.1)
    assert result.trajectory is not None
    np.testing.assert_allclose(result.trajectory.times, [0.0, 0.1, 0.2, 0.3])


def test_linear_planner_rejects_missing_sample_period_without_physics_dt() -> None:
    backend = LinearPlannerBackend(("j0",))

    with pytest.raises(ValueError, match="inject the runtime physics dt"):
        backend.plan(
            MotionRequest(
                current_q=np.asarray([0.0]),
                goal_q=np.asarray([1.0]),
            )
        )
