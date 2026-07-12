from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.planning.batch_ik import BatchIKResult
from linkerbot_sim.tiled.control.adapter import TiledCommandAdapter
from linkerbot_sim.tiled.control.interpolation import interpolate_joint_targets
from linkerbot_sim.tiled.control.types import TiledCommandAction


class _FakeIKSolver:
    def __init__(
        self,
        success: np.ndarray | tuple[np.ndarray, ...] | None = None,
    ) -> None:
        self.calls = []
        self.success = success

    def solve(
        self,
        *,
        target_positions,
        target_orientations_wxyz,
        seeds,
        tcp_frame_name,
    ) -> BatchIKResult:
        self.calls.append(
            (target_positions.copy(), target_orientations_wxyz, seeds, tcp_frame_name)
        )
        q = np.asarray(target_positions[:, : seeds.shape[1]], dtype=float)
        success_value = self.success
        if isinstance(success_value, tuple):
            success_value = success_value[len(self.calls) - 1]
        success = np.ones(seeds.shape[0], dtype=bool)
        if success_value is not None:
            success = np.asarray(success_value, dtype=bool)
        return BatchIKResult(
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


def test_ee_linear_path_runs_batched_ik_per_tick_and_freezes_failed_envs() -> None:
    solver = _FakeIKSolver(
        success=(
            np.asarray([True, True]),
            np.asarray([True, False]),
            np.asarray([True, True]),
        )
    )
    adapter = TiledCommandAdapter(
        num_envs=2,
        command_dim=3,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    path = adapter.linear_path_to_joint_trajectory(
        TiledCommandAction(
            "ee_linear_path",
            np.asarray([[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]]),
            decimation=3,
            interpolation="linear",
        ),
        steps=3,
        current_positions=np.zeros((2, 3)),
        current_tcp_positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        current_tcp_orientations_wxyz=np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
    )

    assert len(solver.calls) == 3
    np.testing.assert_allclose(
        [call[0] for call in solver.calls],
        [
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.4, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [1.6, 0.0, 0.0]],
        ],
    )
    np.testing.assert_allclose(
        path.joint_positions,
        [
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
    )
    np.testing.assert_allclose(solver.calls[1][2], path.joint_positions[0])
    np.testing.assert_allclose(solver.calls[2][2], path.joint_positions[1])
    np.testing.assert_array_equal(path.info["ik_success"], [True, False])
    np.testing.assert_array_equal(path.info["ik_first_failure_step"], [-1, 2])
    np.testing.assert_array_equal(path.info["ik_completed_steps"], [3, 1])


def test_ee_linear_path_keeps_unselected_env_at_its_existing_target() -> None:
    solver = _FakeIKSolver()
    adapter = TiledCommandAdapter(
        num_envs=2,
        command_dim=3,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    path = adapter.linear_path_to_joint_trajectory(
        TiledCommandAction(
            "ee_linear_path",
            np.asarray([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            decimation=2,
            interpolation="linear",
        ),
        steps=2,
        current_positions=np.asarray([[0.0, 0.0, 0.0], [7.0, 8.0, 9.0]]),
        current_tcp_positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        current_tcp_orientations_wxyz=np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        active_env_ids=np.asarray([0]),
    )

    np.testing.assert_allclose(path.joint_positions[:, 1], [[7.0, 8.0, 9.0]] * 2)
    np.testing.assert_array_equal(path.info["ik_completed_steps"], [2, 0])


def test_ee_linear_path_named_offset_free_uses_position_only_ik() -> None:
    solver = _FakeIKSolver()
    adapter = TiledCommandAdapter(
        num_envs=2,
        command_dim=3,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    adapter.linear_path_to_joint_trajectory(
        TiledCommandAction(
            "ee_linear_path",
            target_offset=np.asarray([0.2, 0.0, 0.0]),
            orientation_mode="free",
            interpolation="linear",
        ),
        steps=2,
        current_positions=np.zeros((2, 3)),
        current_tcp_positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        current_tcp_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(
        [call[0] for call in solver.calls],
        [
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
    )
    assert all(call[1] is None for call in solver.calls)


def test_ee_linear_path_absolute_env_target_slerps_orientation() -> None:
    solver = _FakeIKSolver()
    adapter = TiledCommandAdapter(
        num_envs=2,
        command_dim=3,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    adapter.linear_path_to_joint_trajectory(
        TiledCommandAction(
            "ee_linear_path",
            target_position=np.asarray([0.4, 0.0, 0.0]),
            orientation_mode="target",
            target_orientation_wxyz=np.asarray([0.0, 0.0, 0.0, 1.0]),
            interpolation="linear",
            pose_reference_frame="env",
        ),
        steps=2,
        current_positions=np.zeros((2, 3)),
        current_tcp_positions=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        current_tcp_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        env_origins=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(
        [call[0] for call in solver.calls],
        [
            [[0.2, 0.0, 0.0], [2.2, 0.0, 0.0]],
            [[0.4, 0.0, 0.0], [2.4, 0.0, 0.0]],
        ],
    )
    halfway = np.sqrt(0.5)
    np.testing.assert_allclose(
        solver.calls[0][1],
        [[halfway, 0.0, 0.0, halfway]] * 2,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        solver.calls[1][1],
        [[0.0, 0.0, 0.0, 1.0]] * 2,
        atol=1.0e-8,
    )


def test_ee_linear_path_resamples_sparse_ik_waypoints_to_physics_steps() -> None:
    solver = _FakeIKSolver()
    adapter = TiledCommandAdapter(
        num_envs=1,
        command_dim=3,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    path = adapter.linear_path_to_joint_trajectory(
        TiledCommandAction(
            "ee_linear_path",
            target_offset=np.asarray([0.4, 0.0, 0.0]),
            orientation_mode="free",
            interpolation="linear",
        ),
        steps=2,
        execution_steps=4,
        current_positions=np.zeros((1, 3)),
        current_tcp_positions=np.zeros((1, 3)),
        current_tcp_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )

    assert len(solver.calls) == 2
    np.testing.assert_allclose(
        path.joint_positions[:, 0, 0],
        [0.1, 0.2, 0.3, 0.4],
        atol=1.0e-8,
    )


def test_ee_linear_path_sparse_resampling_preserves_smoothstep_progress() -> None:
    solver = _FakeIKSolver()
    adapter = TiledCommandAdapter(
        num_envs=1,
        command_dim=3,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    path = adapter.linear_path_to_joint_trajectory(
        TiledCommandAction(
            "ee_linear_path",
            target_offset=np.asarray([1.0, 0.0, 0.0]),
            orientation_mode="free",
            interpolation="smoothstep",
        ),
        steps=1,
        execution_steps=4,
        current_positions=np.zeros((1, 3)),
        current_tcp_positions=np.zeros((1, 3)),
        current_tcp_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )

    assert len(solver.calls) == 1
    np.testing.assert_allclose(
        path.joint_positions[:, 0, 0],
        [0.15625, 0.5, 0.84375, 1.0],
        atol=1.0e-8,
    )


def test_ee_linear_path_sparse_failure_freezes_before_dense_resampling() -> None:
    solver = _FakeIKSolver(
        success=(
            np.asarray([True, True]),
            np.asarray([True, False]),
        )
    )
    adapter = TiledCommandAdapter(
        num_envs=2,
        command_dim=3,
        tcp_frame_name="tcp",
        ik_solver=solver,
    )

    path = adapter.linear_path_to_joint_trajectory(
        TiledCommandAction(
            "ee_linear_path",
            target_offset=np.asarray([0.4, 0.0, 0.0]),
            orientation_mode="free",
            interpolation="linear",
        ),
        steps=2,
        execution_steps=4,
        current_positions=np.zeros((2, 3)),
        current_tcp_positions=np.zeros((2, 3)),
        current_tcp_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(path.joint_positions[:, 0, 0], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(path.joint_positions[:, 1, 0], [0.1, 0.2, 0.2, 0.2])
    np.testing.assert_array_equal(path.info["ik_first_failure_step"], [-1, 2])
    np.testing.assert_array_equal(path.info["ik_completed_steps"], [2, 1])


def test_ee_linear_path_rejects_ambiguous_target_or_missing_orientation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TiledCommandAction(
            "ee_linear_path",
            values=np.asarray([0.1, 0.0, 0.0]),
            target_offset=np.asarray([0.1, 0.0, 0.0]),
        )
    with pytest.raises(ValueError, match="target_orientation_quat_wxyz"):
        TiledCommandAction(
            "ee_linear_path",
            target_offset=np.asarray([0.1, 0.0, 0.0]),
            orientation_mode="target",
        )
    with pytest.raises(ValueError, match="sample_dt_s"):
        TiledCommandAction(
            "ee_linear_path",
            target_offset=np.asarray([0.1, 0.0, 0.0]),
            sample_dt_s=0.0,
        )


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
