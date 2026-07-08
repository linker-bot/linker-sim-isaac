from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.tiled.command import (
    TiledCommandAction,
    TiledCommandAdapter,
    interpolate_joint_targets,
)
from linkerbot_sim.tiled.batched_ik import BatchedIKResult


class _FakeIKSolver:
    def __init__(self, success: np.ndarray | None = None) -> None:
        self.calls = []
        self.success = success

    def solve(
        self,
        *,
        target_positions,
        target_orientations_wxyz,
        seeds,
        tcp_frame_name,
    ) -> BatchedIKResult:
        self.calls.append(
            (target_positions.copy(), target_orientations_wxyz, seeds, tcp_frame_name)
        )
        q = np.asarray(target_positions[:, : seeds.shape[1]], dtype=float)
        success = (
            np.ones(seeds.shape[0], dtype=bool)
            if self.success is None
            else np.asarray(self.success, dtype=bool)
        )
        return BatchedIKResult(
            joint_positions=q,
            success=success,
            position_error=np.zeros(seeds.shape[0], dtype=float),
        )


def test_joint_position_target_broadcasts_single_row() -> None:
    adapter = TiledCommandAdapter(num_envs=3, command_dim=2)

    target = adapter.action_to_joint_target(
        TiledCommandAction("joint_position_target", np.asarray([[1.0, 2.0]])),
        current_positions=np.zeros((3, 2)),
    )

    np.testing.assert_allclose(target.joint_positions, [[1.0, 2.0]] * 3)


def test_joint_delta_pos_uses_current_positions() -> None:
    adapter = TiledCommandAdapter(num_envs=2, command_dim=2)

    target = adapter.action_to_joint_target(
        TiledCommandAction("joint_delta_pos", np.asarray([[0.1, -0.2]])),
        current_positions=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    )

    np.testing.assert_allclose(target.joint_positions, [[1.1, 1.8], [3.1, 3.8]])


def test_hold_uses_last_target() -> None:
    adapter = TiledCommandAdapter(num_envs=1, command_dim=2)
    adapter.action_to_joint_target(
        TiledCommandAction("joint_position_target", np.asarray([[1.0, 2.0]])),
        current_positions=np.zeros((1, 2)),
    )

    target = adapter.action_to_joint_target(
        TiledCommandAction("hold"),
        current_positions=np.asarray([[9.0, 9.0]]),
    )

    np.testing.assert_allclose(target.joint_positions, [[1.0, 2.0]])


def test_ee_pose_target_uses_env_local_origins_before_ik() -> None:
    solver = _FakeIKSolver()
    adapter = TiledCommandAdapter(
        num_envs=2,
        command_dim=2,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    target = adapter.action_to_joint_target(
        TiledCommandAction(
            "ee_pose_target",
            np.asarray([[0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]]),
        ),
        current_positions=np.zeros((2, 2)),
        env_origins=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(
        solver.calls[0][0],
        [[0.5, 0.0, 0.2], [2.5, 0.0, 0.2]],
    )
    np.testing.assert_allclose(target.joint_positions, [[0.5, 0.0], [2.5, 0.0]])


def test_ee_delta_pos_keeps_failed_ik_rows_at_current_positions() -> None:
    solver = _FakeIKSolver(success=np.asarray([True, False]))
    adapter = TiledCommandAdapter(
        num_envs=2,
        command_dim=2,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    target = adapter.action_to_joint_target(
        TiledCommandAction("ee_delta_pos", np.asarray([[0.1, 0.0, 0.0]])),
        current_positions=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        current_tcp_positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        current_tcp_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(target.joint_positions, [[0.1, 0.0], [3.0, 4.0]])
    np.testing.assert_array_equal(target.info["ik_success"], [True, False])


def test_interpolate_joint_targets_smoothstep_reaches_target() -> None:
    steps = interpolate_joint_targets(
        start=np.asarray([[0.0, 0.0]]),
        target=np.asarray([[1.0, 2.0]]),
        steps=3,
        mode="smoothstep",
    )

    assert steps.shape == (3, 1, 2)
    np.testing.assert_allclose(steps[-1], [[1.0, 2.0]])
    assert 0.0 < steps[0, 0, 0] < 1.0


def test_command_action_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported tiled command kind"):
        TiledCommandAction("plan_a_path")


def test_command_action_rejects_unknown_pose_reference_frame() -> None:
    with pytest.raises(ValueError, match="pose_reference_frame"):
        TiledCommandAction(
            "ee_pose_target",
            np.asarray([[0.0] * 7]),
            pose_reference_frame="bad",
        )
