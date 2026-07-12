from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.backends.curobo.batch.ik import CuroboBatchIKSolver


class _FakeIkSolver:
    def __init__(self, *, success=None) -> None:
        self.calls = []
        self.criteria_updates = []
        self.success = (
            np.asarray(success, dtype=bool)
            if success is not None
            else np.asarray([True, True], dtype=bool)
        )

    def update_tool_pose_criteria(self, criteria):
        self.criteria_updates.append(dict(criteria))

    def solve_pose(self, goal_tool_poses, *, current_state=None, seed_config=None):
        seeds = np.asarray(seed_config, dtype=float)
        positions = np.asarray(goal_tool_poses["positions"], dtype=float)
        seed_positions = seeds[:, 0, :] if seeds.ndim == 3 else seeds
        self.calls.append(
            {
                "goal": goal_tool_poses,
                "current_state": current_state,
                "seed_config": seeds.copy(),
            }
        )
        q = seed_positions.copy()
        q[:, : min(q.shape[1], 3)] += positions[:, : min(q.shape[1], 3)]
        return SimpleNamespace(
            solution=q,
            success=self.success.copy(),
            position_error=np.asarray([0.001, 0.002], dtype=float),
            rotation_error=np.asarray([0.01, 0.02], dtype=float),
        )


class _FakeToolPoseCriteria:
    @staticmethod
    def track_position():
        return "position"

    @staticmethod
    def track_position_and_orientation():
        return "pose"


class _FakeContext:
    default_tcp_frame = "tool"

    def __init__(
        self,
        *,
        success=None,
        joint_state_with_position: bool = False,
        frame_names=("tool",),
    ) -> None:
        self._frame_names = tuple(str(name) for name in frame_names)
        self.default_tcp_frame = self._frame_names[0]
        self.ik_solver = _FakeIkSolver(success=success)
        self.types = SimpleNamespace(ToolPoseCriteria=_FakeToolPoseCriteria)
        self.joint_state_with_position = bool(joint_state_with_position)
        self.fk_calls = []

    def joint_names(self):
        return ["j0", "j1", "j2"]

    def frame_names(self):
        return list(self._frame_names)

    def joint_state_from_positions(self, positions):
        if self.joint_state_with_position:
            return SimpleNamespace(position=np.asarray(positions, dtype=float))
        return {"joint_state": np.asarray(positions, dtype=float)}

    def compute_tcp_poses(self, joint_positions, *, tcp_frame_name=None):
        positions = np.asarray(joint_positions, dtype=float)
        self.fk_calls.append(
            {
                "joint_positions": positions.copy(),
                "tcp_frame_name": tcp_frame_name,
            }
        )
        return positions[:, :3], np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
            (positions.shape[0], 1),
        )


def test_curobo_batch_ik_solves_fake_batch() -> None:
    context = _FakeContext()
    solver = CuroboBatchIKSolver(context, tcp_frame_name="tool")

    result = solver.solve(
        target_positions=np.asarray([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]),
        target_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        seeds=np.zeros((2, 3)),
        tcp_frame_name="tool",
    )

    assert result.success.tolist() == [True, True]
    np.testing.assert_allclose(
        result.joint_positions, [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]
    )
    np.testing.assert_allclose(result.position_error, [0.001, 0.002])
    np.testing.assert_allclose(result.orientation_error, [0.01, 0.02])
    assert context.ik_solver.calls[0]["goal"]["tool_frames"] == ("tool",)
    assert context.ik_solver.criteria_updates[0] == {"tool": "pose"}


def test_curobo_batch_ik_keeps_failed_rows_at_seed() -> None:
    context = _FakeContext(success=[True, False])
    solver = CuroboBatchIKSolver(context, tcp_frame_name="tool")
    seeds = np.asarray([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]])

    result = solver.solve(
        target_positions=np.asarray([[1.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
        target_orientations_wxyz=None,
        seeds=seeds,
        tcp_frame_name="tool",
    )

    assert result.success.tolist() == [True, False]
    np.testing.assert_allclose(result.joint_positions[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.joint_positions[1], seeds[1])
    assert result.orientation_error is None
    assert context.ik_solver.criteria_updates[0] == {"tool": "position"}


def test_curobo_batch_ik_updates_only_active_tool_frame_criteria() -> None:
    context = _FakeContext(frame_names=("tool", "other_tool"))
    solver = CuroboBatchIKSolver(context, tcp_frame_name="tool")

    solver.solve(
        target_positions=np.asarray([[1.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
        target_orientations_wxyz=None,
        seeds=np.zeros((2, 3)),
        tcp_frame_name="tool",
    )

    assert context.ik_solver.criteria_updates[0] == {"tool": "position"}


def test_curobo_batch_ik_maps_command_space_to_cspace_and_back() -> None:
    context = _FakeContext()
    solver = CuroboBatchIKSolver(
        context,
        tcp_frame_name="tool",
        command_joint_names=("hand", "j0", "j1", "j2"),
    )
    seeds = np.asarray([[8.0, 0.0, 0.0, 0.0], [7.0, 1.0, 1.0, 1.0]])

    result = solver.solve(
        target_positions=np.asarray([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]),
        target_orientations_wxyz=None,
        seeds=seeds,
        tcp_frame_name="tool",
    )

    np.testing.assert_allclose(result.joint_positions[:, 0], [8.0, 7.0])
    np.testing.assert_allclose(
        result.joint_positions[:, 1:], [[1.0, 2.0, 3.0], [1.5, 1.5, 1.5]]
    )
    np.testing.assert_allclose(
        context.ik_solver.calls[0]["seed_config"],
        [[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]],
    )


def test_curobo_batch_ik_uses_joint_state_position_as_seed_config() -> None:
    context = _FakeContext(joint_state_with_position=True)
    solver = CuroboBatchIKSolver(context, tcp_frame_name="tool")

    solver.solve(
        target_positions=np.asarray([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]),
        target_orientations_wxyz=None,
        seeds=np.zeros((2, 3)),
        tcp_frame_name="tool",
    )

    np.testing.assert_allclose(
        context.ik_solver.calls[0]["seed_config"],
        [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
    )


def test_curobo_batch_ik_maps_command_space_for_tcp_fk() -> None:
    context = _FakeContext()
    solver = CuroboBatchIKSolver(
        context,
        tcp_frame_name="tool",
        command_joint_names=("hand", "j0", "j1", "j2"),
    )

    positions, orientations = solver.compute_tcp_poses(
        np.asarray([[8.0, 1.0, 2.0, 3.0], [7.0, 4.0, 5.0, 6.0]]),
        tcp_frame_name="tool",
    )

    np.testing.assert_allclose(positions, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(
        orientations,
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        context.fk_calls[0]["joint_positions"],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    assert context.fk_calls[0]["tcp_frame_name"] == "tool"


def test_curobo_batch_ik_rejects_unknown_frame() -> None:
    solver = CuroboBatchIKSolver(_FakeContext(), tcp_frame_name="tool")

    try:
        solver.solve(
            target_positions=np.zeros((1, 3)),
            target_orientations_wxyz=None,
            seeds=np.zeros((1, 3)),
            tcp_frame_name="missing",
        )
    except ValueError as exc:
        assert "Unknown cuRobo frame" in str(exc)
    else:
        raise AssertionError("unknown frame was accepted")
