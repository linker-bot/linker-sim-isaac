from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.controllers.joint_controller import JointController
from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlTargets,
    JointControlSettings,
)
from linkerbot_sim.robots.mimic import MimicFollowerControl


class _Action:
    def __init__(
        self,
        *,
        joint_positions=None,
        joint_velocities=None,
        joint_efforts=None,
        joint_indices=None,
    ) -> None:
        self.joint_positions = (
            None
            if joint_positions is None
            else np.asarray(joint_positions, dtype=float)
        )
        self.joint_velocities = (
            None
            if joint_velocities is None
            else np.asarray(joint_velocities, dtype=float)
        )
        self.joint_efforts = (
            None if joint_efforts is None else np.asarray(joint_efforts, dtype=float)
        )
        self.joint_indices = (
            None if joint_indices is None else np.asarray(joint_indices, dtype=int)
        )


class _FakeController:
    def __init__(self) -> None:
        self.gains: tuple[np.ndarray, np.ndarray] | None = None
        self.max_efforts: np.ndarray | None = None
        self.mode_switches: list[tuple[int, str]] = []
        self.effort_mode_calls: list[tuple[str, np.ndarray]] = []

    def set_gains(self, *, kps, kds) -> None:
        self.gains = (np.asarray(kps, dtype=float), np.asarray(kds, dtype=float))

    def set_max_efforts(self, values, joint_indices=None) -> None:
        self.max_efforts = np.asarray(values, dtype=float)

    def switch_dof_control_mode(self, *, dof_index: int, mode: str) -> None:
        self.mode_switches.append((int(dof_index), str(mode)))

    def set_effort_modes(self, mode: str, joint_indices=None) -> None:
        self.effort_mode_calls.append((str(mode), np.asarray(joint_indices, dtype=int)))


class _FakeRobot:
    def __init__(self) -> None:
        self.dof_names = ["arm_joint", "hand_joint", "follower_joint"]
        self.num_dof = len(self.dof_names)
        self.positions = np.asarray([0.2, -0.1, 0.0], dtype=float)
        self.velocities = np.asarray([0.05, -0.2, 0.0], dtype=float)
        self.controller = _FakeController()
        self.actions: list[_Action] = []

    def get_articulation_controller(self):
        return self.controller

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def apply_action(self, action) -> None:
        self.actions.append(action)


class _MissingDofModeController:
    def __init__(self) -> None:
        self.gains: tuple[np.ndarray, np.ndarray] | None = None
        self.max_efforts: np.ndarray | None = None
        self.effort_mode_calls: list[tuple[str, np.ndarray]] = []

    def set_gains(self, *, kps, kds) -> None:
        self.gains = (np.asarray(kps, dtype=float), np.asarray(kds, dtype=float))

    def set_max_efforts(self, values, joint_indices=None) -> None:
        self.max_efforts = np.asarray(values, dtype=float)

    def set_effort_modes(self, mode: str, joint_indices=None) -> None:
        self.effort_mode_calls.append((str(mode), np.asarray(joint_indices, dtype=int)))


class _MissingDofModeRobot(_FakeRobot):
    def __init__(self) -> None:
        super().__init__()
        self.controller = _MissingDofModeController()


def _controller(
    settings: ComponentControlSettings,
) -> tuple[JointController, _FakeRobot]:
    robot = _FakeRobot()
    controller = JointController(
        robot,
        joint_names=["arm_joint", "hand_joint", "follower_joint"],
        settings=JointControlSettings(default=settings, arm=settings, hand=settings),
        mjcf_path=None,
    )
    controller.follower_indices = np.asarray([2], dtype=int)
    controller.follower_joint_names = {"follower_joint"}
    controller.command_indices = np.asarray([0, 1], dtype=int)
    controller.driven_indices = np.asarray([0, 1, 2], dtype=int)
    controller.follower_mapper.controls = [
        MimicFollowerControl(
            dependent_joint="follower_joint",
            master_joint="hand_joint",
            dependent_index=2,
            master_index=1,
            polycoef=(0.1, 2.0),
        )
    ]
    return controller, robot


def test_explicit_position_control_computes_pd_effort_and_keeps_follower_position_drive() -> (
    None
):
    controller, robot = _controller(
        ComponentControlSettings(
            mode="position",
            method="explicit",
            stiffness=(10.0,),
            damping=(2.0,),
            max_force=100.0,
            follower_stiffness=(50.0,),
            follower_damping=(5.0,),
            follower_max_force=20.0,
        )
    )
    controller.configure_runtime()
    targets = controller.build_control_targets(
        command_positions=np.asarray([0.4, 0.2]),
        command_velocities=np.asarray([0.0, 0.1]),
        base_positions=robot.positions,
    )

    controller.apply_targets(_Action, targets)

    effort_action, follower_action = robot.actions
    np.testing.assert_array_equal(effort_action.joint_indices, [0, 1])
    np.testing.assert_allclose(effort_action.joint_efforts, [1.9, 3.6])
    np.testing.assert_array_equal(follower_action.joint_indices, [2])
    np.testing.assert_allclose(follower_action.joint_positions, [-0.1])
    assert (2, "position") in robot.controller.mode_switches
    assert (0, "effort") in robot.controller.mode_switches


def test_explicit_velocity_control_uses_damping_only() -> None:
    controller, robot = _controller(
        ComponentControlSettings(
            mode="velocity",
            method="explicit",
            damping=(4.0,),
            max_force=100.0,
        )
    )
    controller.configure_runtime()
    targets = controller.build_control_targets(
        command_velocities=np.asarray([0.25, -0.1]),
        base_positions=robot.positions,
    )

    controller.apply_targets(_Action, targets)

    effort_action = robot.actions[0]
    np.testing.assert_allclose(effort_action.joint_efforts, [0.8, 0.4])


def test_implicit_velocity_control_sends_velocity_action() -> None:
    controller, robot = _controller(
        ComponentControlSettings(
            mode="velocity",
            method="implicit",
            damping=(4.0,),
            max_force=100.0,
        )
    )
    controller.configure_runtime()
    targets = controller.build_control_targets(
        command_velocities=np.asarray([0.25, -0.1]),
        base_positions=robot.positions,
    )

    controller.apply_targets(_Action, targets)

    velocity_action = robot.actions[0]
    np.testing.assert_array_equal(velocity_action.joint_indices, [0, 1])
    np.testing.assert_allclose(velocity_action.joint_velocities, [0.25, -0.1])
    assert velocity_action.joint_efforts is None
    assert (0, "velocity") in robot.controller.mode_switches


def test_effort_control_sends_direct_clipped_effort() -> None:
    controller, robot = _controller(
        ComponentControlSettings(
            mode="effort",
            method="direct",
            max_force=100.0,
            effort_limit=1.5,
        )
    )
    controller.configure_runtime()
    targets = ControlTargets(
        positions=robot.positions.copy(),
        velocities=np.zeros(robot.num_dof, dtype=float),
        efforts=np.asarray([3.0, -1.0, 0.0], dtype=float),
    )

    controller.apply_targets(_Action, targets)

    effort_action = robot.actions[0]
    np.testing.assert_array_equal(effort_action.joint_indices, [0, 1])
    np.testing.assert_allclose(effort_action.joint_efforts, [1.5, -1.0])


def test_controller_requires_per_dof_control_mode_switching() -> None:
    settings = ComponentControlSettings(
        mode="velocity",
        method="explicit",
        damping=(4.0,),
        max_force=100.0,
    )
    robot = _MissingDofModeRobot()
    controller = JointController(
        robot,
        joint_names=["arm_joint", "hand_joint", "follower_joint"],
        settings=JointControlSettings(default=settings, arm=settings, hand=settings),
        mjcf_path=None,
    )
    controller.follower_indices = np.asarray([2], dtype=int)
    controller.follower_joint_names = {"follower_joint"}
    controller.command_indices = np.asarray([0, 1], dtype=int)
    controller.driven_indices = np.asarray([0, 1, 2], dtype=int)

    with pytest.raises(RuntimeError, match="switch_dof_control_mode"):
        controller.configure_runtime()


def test_command_joint_names_follow_command_indices_and_exclude_followers() -> None:
    controller, _robot = _controller(
        ComponentControlSettings(mode="position", method="implicit")
    )

    assert controller.command_joint_names == ("arm_joint", "hand_joint")
    assert "follower_joint" not in controller.command_joint_names
