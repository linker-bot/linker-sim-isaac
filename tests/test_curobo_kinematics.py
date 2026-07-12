from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.backends.curobo.tensor_adapter import as_curobo_seed_config
from linkerbot_sim.backends.curobo.context import CuroboContext
from linkerbot_sim.backends.curobo.forward_kinematics import CuroboForwardKinematics
from linkerbot_sim.backends.curobo.inverse_kinematics import CuroboInverseKinematics
from linkerbot_sim.backends.curobo.trajectory_adapter import (
    joint_trajectory_from_motion_result,
)
from linkerbot_sim.planning.requests import IKRequest
from linkerbot_sim.planning.results import MotionResult
from linkerbot_sim.trajectories.types import JointTrajectory


class _FakeIkSolver:
    def __init__(self) -> None:
        self.calls = []
        self.criteria_updates = []

    def update_tool_pose_criteria(self, criteria):
        self.criteria_updates.append(dict(criteria))

    def solve_pose(self, goal, *, current_state=None, seed_config=None):
        self.calls.append(
            {
                "goal": goal,
                "current_state": current_state,
                "seed_config": seed_config,
            }
        )
        seed = np.asarray(seed_config, dtype=float)
        return SimpleNamespace(
            success=np.asarray([True]),
            solution=seed + 0.5,
            position_error=np.asarray([0.001]),
            rotation_error=np.asarray([0.01]),
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
        joint_state_with_position: bool = False,
        frame_names=("tool",),
    ) -> None:
        self._frame_names = tuple(str(name) for name in frame_names)
        self.default_tcp_frame = self._frame_names[0]
        self.ik_solver = _FakeIkSolver()
        self.types = SimpleNamespace(ToolPoseCriteria=_FakeToolPoseCriteria)
        self.joint_state_with_position = bool(joint_state_with_position)
        self.config = SimpleNamespace(
            ik=SimpleNamespace(
                position_tolerance=0.002,
                orientation_tolerance=0.01,
            )
        )

    def joint_names(self):
        return ["j0", "j1"]

    def frame_names(self):
        return list(self._frame_names)

    def joint_state_from_positions(self, positions):
        if self.joint_state_with_position:
            return SimpleNamespace(position=np.asarray(positions, dtype=float))
        return {"position": np.asarray(positions, dtype=float)}

    def goal_tool_pose_from_arrays(self, *, positions, orientations_wxyz, tool_frames):
        return {
            "positions": np.asarray(positions, dtype=float),
            "orientations_wxyz": orientations_wxyz,
            "tool_frames": tuple(tool_frames),
        }

    def compute_tcp_poses(self, joint_positions, *, tcp_frame_name=None):
        frame_name = str(tcp_frame_name or self.default_tcp_frame)
        frame_index = self._frame_names.index(frame_name)
        return (
            np.asarray(
                [[0.1 + float(frame_index), 0.2 + float(frame_index), 0.3]],
                dtype=float,
            ),
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
        )

    def collision_queries_enabled(self):
        return True


def test_curobo_forward_kinematics_uses_context_tcp_pose() -> None:
    fk = CuroboForwardKinematics(_FakeContext())

    pose = fk.compute_pose(np.asarray([0.0, 0.0]), "tool")

    np.testing.assert_allclose(pose.position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose.orientation, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(pose.rotation_matrix, np.eye(3))


def test_curobo_inverse_kinematics_solves_fake_request() -> None:
    context = _FakeContext()
    ik = CuroboInverseKinematics(context, tcp_frame_name="tool")

    result = ik.solve(
        IKRequest(
            target_position=np.asarray([0.1, 0.2, 0.3]),
            target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            warm_start_ik_cspace_seed=np.asarray([1.0, 2.0]),
        )
    )

    assert result.success is True
    np.testing.assert_allclose(result.joint_positions, [1.5, 2.5])
    assert result.position_error == 0.001
    assert result.orientation_error == 0.01
    assert context.ik_solver.calls[0]["goal"]["tool_frames"] == ("tool",)
    assert context.ik_solver.criteria_updates[0] == {"tool": "pose"}


def test_curobo_inverse_kinematics_position_only_updates_tool_pose_criteria() -> None:
    context = _FakeContext()
    ik = CuroboInverseKinematics(context, tcp_frame_name="tool")

    result = ik.solve(
        IKRequest(
            target_position=np.asarray([0.1, 0.2, 0.3]),
            target_orientation=None,
            warm_start_ik_cspace_seed=np.asarray([1.0, 2.0]),
        )
    )

    assert result.success is True
    assert result.orientation_error is None
    assert context.ik_solver.criteria_updates[0] == {"tool": "position"}


def test_curobo_inverse_kinematics_does_not_silently_ignore_tolerances() -> None:
    context = _FakeContext()
    ik = CuroboInverseKinematics(context, tcp_frame_name="tool")
    request = IKRequest(
        target_position=np.asarray([0.1, 0.2, 0.3]),
        target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
        warm_start_ik_cspace_seed=np.asarray([1.0, 2.0]),
        position_tolerance=0.0001,
        orientation_tolerance=0.001,
    )

    with pytest.raises(ValueError, match="per-request tolerance"):
        ik.solve(request)

    assert context.ik_solver.calls == []


def test_curobo_inverse_kinematics_fills_inactive_tcp_goals() -> None:
    context = _FakeContext(frame_names=("left_tcp", "right_tcp"))
    ik = CuroboInverseKinematics(context, tcp_frame_name="left_tcp")

    ik.solve(
        IKRequest(
            target_position=np.asarray([9.0, 8.0, 7.0]),
            target_orientation=np.asarray([0.0, 1.0, 0.0, 0.0]),
            warm_start_ik_cspace_seed=np.asarray([1.0, 2.0]),
        )
    )

    goal = context.ik_solver.calls[0]["goal"]
    assert goal["tool_frames"] == ("left_tcp", "right_tcp")
    np.testing.assert_allclose(
        goal["positions"],
        [[[9.0, 8.0, 7.0], [1.1, 1.2, 0.3]]],
    )
    np.testing.assert_allclose(
        goal["orientations_wxyz"],
        [[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
    )
    assert context.ik_solver.criteria_updates[0] == {
        "left_tcp": "pose",
        "right_tcp": "pose",
    }


def test_curobo_inverse_kinematics_uses_joint_state_position_as_seed_config() -> None:
    context = _FakeContext(joint_state_with_position=True)
    ik = CuroboInverseKinematics(context, tcp_frame_name="tool")

    ik.solve(
        IKRequest(
            target_position=np.asarray([0.1, 0.2, 0.3]),
            warm_start_ik_cspace_seed=np.asarray([1.0, 2.0]),
        )
    )

    np.testing.assert_allclose(
        context.ik_solver.calls[0]["seed_config"],
        [[[1.0, 2.0]]],
    )


def test_curobo_context_makes_noncontiguous_joint_positions_contiguous() -> None:
    torch = pytest.importorskip("torch")
    context = CuroboContext.__new__(CuroboContext)
    context.torch = torch
    context.device_cfg = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    context.types = SimpleNamespace(
        JointState=SimpleNamespace(
            from_position=lambda position, joint_names: SimpleNamespace(
                position=position,
                joint_names=tuple(joint_names),
            )
        )
    )
    context.joint_names = lambda: [f"j{index}" for index in range(7)]

    command = np.arange(32 * 28, dtype=float).reshape(32, 28)
    cspace = command[:, (1, 3, 5, 7, 9, 11, 13)]
    assert not cspace.flags.c_contiguous

    state = context.joint_state_from_positions(cspace)

    assert state.position.is_contiguous()
    assert tuple(state.position.shape) == (32, 7)
    assert state.joint_names == ("j0", "j1", "j2", "j3", "j4", "j5", "j6")


def test_curobo_seed_config_makes_unsqueezed_torch_seed_contiguous() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    command = np.arange(32 * 28, dtype=float).reshape(32, 28)
    cspace = command[:, (1, 3, 5, 7, 9, 11, 13)]
    seed = torch.as_tensor(cspace, dtype=torch.float32)
    assert not seed.is_contiguous()

    seed_config = as_curobo_seed_config(seed)

    assert seed_config.is_contiguous()
    assert tuple(seed_config.shape) == (32, 1, 7)


def test_curobo_inverse_kinematics_rejects_collision_aware_without_model() -> None:
    context = _FakeContext()
    context.collision_queries_enabled = lambda: False
    ik = CuroboInverseKinematics(context, tcp_frame_name="tool")

    result = ik.solve(
        IKRequest(
            target_position=np.asarray([0.1, 0.2, 0.3]),
            warm_start_ik_cspace_seed=np.asarray([1.0, 2.0]),
            avoid_collisions=True,
        )
    )

    assert result.success is False
    assert result.status == "COLLISION_UNSUPPORTED"
    assert context.ik_solver.calls == []


def test_motion_result_adapter_accepts_project_trajectory() -> None:
    trajectory = JointTrajectory.from_samples(
        times=np.asarray([0.0, 0.1]),
        positions=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        joint_names=("j0", "j1"),
    )
    result = MotionResult(
        path=None,
        trajectory=trajectory,
        success=True,
        status="SUCCESS",
    )

    converted = joint_trajectory_from_motion_result(
        result,
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt=0.1,
        phase="curobo",
    )

    assert converted is trajectory
