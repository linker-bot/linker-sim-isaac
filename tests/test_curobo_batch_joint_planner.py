from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.backends.curobo.batch.joint_planner import (
    CuroboBatchJointPlanner,
)
from linkerbot_sim.backends.curobo.batch.types import CuroboBatchJointProblem


class _Context:
    def __init__(self, planner) -> None:
        self.batch_motion_planner = planner

    def joint_names(self):
        return ["j0", "j1"]

    def joint_state_from_positions(self, positions):
        return SimpleNamespace(position=np.asarray(positions, dtype=float))


class _LinearBatchPlanner:
    batch_size = 4

    def __init__(self) -> None:
        self.calls = []

    def plan_cspace(self, goal_state, current_state):
        current = np.asarray(current_state.position, dtype=float)
        goal = np.asarray(goal_state.position, dtype=float)
        self.calls.append((current.copy(), goal.copy()))
        alpha = np.asarray([0.0, 0.5, 1.0]).reshape(1, 1, 3, 1)
        positions = (
            current[:, None, None, :] + (goal - current)[:, None, None, :] * alpha
        )
        return SimpleNamespace(
            success=np.ones((self.batch_size, 1), dtype=bool),
            interpolated_trajectory=SimpleNamespace(position=positions),
            interpolated_trajectory_dt=np.asarray([5.0]),
        )


def _problem() -> CuroboBatchJointProblem:
    return CuroboBatchJointProblem(
        current_positions=np.asarray([[8.0, 0.0, 0.0], [9.0, 1.0, 1.0]], dtype=float),
        goal_positions=np.asarray([[7.0, 1.0, 2.0], [6.0, 2.0, 3.0]], dtype=float),
        command_joint_names=("hand", "j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )


def test_batch_joint_planner_pads_and_restores_command_space() -> None:
    backend = _LinearBatchPlanner()
    planner = CuroboBatchJointPlanner(_Context(backend))

    result = planner.plan(_problem())

    assert result.all_succeeded is True
    assert result.positions.shape == (2, 3, 3)
    current, goal = backend.calls[0]
    np.testing.assert_allclose(current[:2], [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(goal[:2], [[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(current[2:], [[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        result.positions[:, :, 0], [[8.0, 7.5, 7.0], [9.0, 7.5, 6.0]]
    )
    np.testing.assert_allclose(result.positions[:, -1, 1:], [[1.0, 2.0], [2.0, 3.0]])


def test_batch_joint_planner_reports_capacity_without_calling_backend() -> None:
    backend = _LinearBatchPlanner()
    backend.batch_size = 1
    planner = CuroboBatchJointPlanner(_Context(backend))

    result = planner.plan(_problem())

    assert result.all_succeeded is False
    assert result.status == ("BATCH_TOO_SMALL", "BATCH_TOO_SMALL")
    assert backend.calls == []


def test_batch_joint_planner_selects_first_successful_seed_per_row() -> None:
    class _SeedPlanner:
        batch_size = 2

        def plan_cspace(self, goal_state, current_state):
            del goal_state, current_state
            positions = np.asarray(
                [
                    [
                        [[0.0, 0.0], [10.0, 10.0]],
                        [[1.0, 1.0], [11.0, 11.0]],
                    ],
                    [
                        [[2.0, 2.0], [12.0, 12.0]],
                        [[3.0, 3.0], [13.0, 13.0]],
                    ],
                ],
                dtype=float,
            )
            return SimpleNamespace(
                success=np.asarray([[False, True], [True, False]]),
                interpolated_trajectory=SimpleNamespace(position=positions),
                interpolated_trajectory_dt=np.asarray([1.0]),
            )

    problem = CuroboBatchJointProblem(
        current_positions=np.zeros((2, 2)),
        goal_positions=np.ones((2, 2)),
        command_joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.1,
    )

    result = CuroboBatchJointPlanner(_Context(_SeedPlanner())).plan(problem)

    np.testing.assert_allclose(result.positions[0, -1], [11.0, 11.0])
    np.testing.assert_allclose(result.positions[1, -1], [12.0, 12.0])


def test_batch_joint_planner_uses_shared_non_divisible_tick_grid() -> None:
    class _UnevenSourcePlanner:
        batch_size = 1

        def plan_cspace(self, goal_state, current_state):
            current = np.asarray(current_state.position, dtype=float)
            goal = np.asarray(goal_state.position, dtype=float)
            progress = np.asarray([0.0, 0.1, 1.0]).reshape(1, 1, 3, 1)
            positions = (
                current[:, None, None, :]
                + (goal - current)[:, None, None, :] * progress
            )
            return SimpleNamespace(
                success=np.asarray([[True]]),
                interpolated_trajectory=SimpleNamespace(position=positions),
                interpolated_trajectory_dt=np.asarray([5.0]),
            )

    problem = CuroboBatchJointProblem(
        current_positions=np.zeros((1, 2)),
        goal_positions=np.ones((1, 2)),
        command_joint_names=("j0", "j1"),
        duration_s=0.11,
        sample_dt_s=0.05,
    )

    result = CuroboBatchJointPlanner(_Context(_UnevenSourcePlanner())).plan(problem)

    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1, 0.11])
    np.testing.assert_allclose(result.positions[0, 0], [0.0, 0.0])
    np.testing.assert_allclose(result.positions[0, -1], [1.0, 1.0])
