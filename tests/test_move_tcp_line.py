from __future__ import annotations

import numpy as np

from manipulation_project.backends.cumotion.forward_kinematics import ForwardKinematicsPose
from manipulation_project.planning.results import IKResult
from manipulation_project.tasks.move_tcp_line import (
    MoveTcpLineConfig,
    build_tcp_line_command_trajectory,
)
from manipulation_project.utils.config import load_yaml


class _FakeForwardKinematics:
    def compute_pose(self, joint_positions, frame_name: str) -> ForwardKinematicsPose:
        assert frame_name == "tool"
        np.testing.assert_allclose(joint_positions, [0.2, 0.4])
        return ForwardKinematicsPose(
            position=np.asarray([1.0, 2.0, 3.0], dtype=float),
            orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
            rotation_matrix=np.eye(3),
        )


class _FakeInverseKinematics:
    def __init__(self) -> None:
        self.requests = []

    def solve(self, request):
        self.requests.append(request)
        z = float(np.asarray(request.target_position, dtype=float)[2])
        return IKResult(
            joint_positions=np.asarray([z, z + 1.0], dtype=float),
            success=True,
            position_error=0.001,
            status="SUCCESS",
        )


class _FakeContext:
    def __init__(self) -> None:
        self.solver = _FakeInverseKinematics()

    def joint_names(self) -> list[str]:
        return ["arm_joint_1", "arm_joint_2"]

    def make_forward_kinematics(self):
        return _FakeForwardKinematics()

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        assert tcp_frame_name == "tool"
        return self.solver


def test_default_tcp_line_config_parses() -> None:
    config = MoveTcpLineConfig.from_mapping(load_yaml("configs/trajectories/tcp_line.yaml"))

    config.validate()
    assert config.tcp_frame_name == "AR5V2_L_arm_flan_link"
    assert config.target_offset == (0.0, 0.0, 0.05)
    assert config.orientation_mode == "current"


def test_build_tcp_line_command_trajectory_warm_starts_each_waypoint() -> None:
    context = _FakeContext()
    config = MoveTcpLineConfig(
        tcp_frame_name="tool",
        target_offset=(0.0, 0.0, 0.2),
        orientation_mode="current",
        duration_s=1.0,
        sample_hz=2.0,
    )

    trajectory, diagnostics = build_tcp_line_command_trajectory(
        dof_names=["arm_joint_1", "hand_joint", "arm_joint_2"],
        command_indices=np.asarray([0, 1, 2], dtype=int),
        current_positions=np.asarray([0.2, 9.0, 0.4], dtype=float),
        config=config,
        context=context,
    )

    assert len(trajectory) == 3
    assert trajectory.joint_names == ("arm_joint_1", "hand_joint", "arm_joint_2")
    np.testing.assert_allclose(diagnostics.start_position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(diagnostics.target_position, [1.0, 2.0, 3.2])
    np.testing.assert_allclose(diagnostics.start_orientation, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(diagnostics.target_orientation, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(trajectory.positions[:, 1], 9.0)
    np.testing.assert_allclose(trajectory.positions[:, [0, 2]], [[0.2, 0.4], [3.1, 4.1], [3.2, 4.2]])
    assert len(context.solver.requests) == 2
    np.testing.assert_allclose(context.solver.requests[0].warm_start, [0.2, 0.4])
    np.testing.assert_allclose(context.solver.requests[1].warm_start, [3.1, 4.1])
    np.testing.assert_allclose(context.solver.requests[0].target_orientation, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(context.solver.requests[1].target_orientation, [1.0, 0.0, 0.0, 0.0])


def test_build_tcp_line_command_trajectory_slerps_target_orientation() -> None:
    context = _FakeContext()
    config = MoveTcpLineConfig(
        tcp_frame_name="tool",
        target_offset=(0.0, 0.0, 0.2),
        orientation_mode="target",
        target_orientation=(0.0, 0.0, 0.0, 1.0),
        duration_s=1.0,
        sample_hz=2.0,
    )

    _trajectory, diagnostics = build_tcp_line_command_trajectory(
        dof_names=["arm_joint_1", "arm_joint_2"],
        command_indices=np.asarray([0, 1], dtype=int),
        current_positions=np.asarray([0.2, 0.4], dtype=float),
        config=config,
        context=context,
    )

    np.testing.assert_allclose(diagnostics.start_orientation, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(diagnostics.target_orientation, [0.0, 0.0, 0.0, 1.0])
    assert len(context.solver.requests) == 2
    np.testing.assert_allclose(
        context.solver.requests[0].target_orientation,
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
    )
    np.testing.assert_allclose(context.solver.requests[1].target_orientation, [0.0, 0.0, 0.0, 1.0], atol=1.0e-12)
