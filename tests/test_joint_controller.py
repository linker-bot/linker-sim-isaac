from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linkerbot_sim.controllers.joint_controller import JointController
from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlTargets,
    JointControlSettings,
)
from linkerbot_sim.robots.classification import RobotComponentMapping
from linkerbot_sim.robots.mimic.runtime import MimicFollowerControl


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

    def get_gains(self) -> tuple[np.ndarray, np.ndarray]:
        if self.gains is None:
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
        return self.gains[0].copy(), self.gains[1].copy()

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

    def get_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

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


def test_apply_targets_revalidates_arrays_mutated_after_construction() -> None:
    controller, robot = _controller(
        ComponentControlSettings(mode="position", method="implicit")
    )
    targets = ControlTargets(
        positions=robot.positions.copy(),
        velocities=np.zeros(robot.num_dof, dtype=float),
        efforts=np.zeros(robot.num_dof, dtype=float),
    )
    targets.positions[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        controller.apply_targets(_Action, targets)

    assert robot.actions == []


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


def test_command_target_modes_follow_command_indices() -> None:
    controller, _robot = _controller(
        ComponentControlSettings(mode="position", method="implicit")
    )
    controller._active_specs = {
        0: ("velocity", "implicit"),
        1: ("effort", "direct"),
    }

    assert controller.command_target_modes == ("velocity", "effort")


def test_control_target_cache_is_deep_copied_and_explicitly_restorable() -> None:
    controller, robot = _controller(
        ComponentControlSettings(mode="position", method="implicit")
    )
    controller.configure_runtime()
    targets = controller.build_control_targets(
        command_positions=np.asarray([0.4, 0.2]),
        base_positions=robot.positions,
    )

    controller.apply_targets(_Action, targets)
    targets.positions[0] = 99.0
    snapshot = controller.snapshot_control_targets_cache()

    assert snapshot is not None
    np.testing.assert_allclose(snapshot.positions[:2], [0.4, 0.2])
    snapshot.positions[0] = 88.0
    np.testing.assert_allclose(
        controller.last_control_targets.positions[:2], [0.4, 0.2]
    )

    controller.restore_control_targets_cache(None)
    assert controller.last_control_targets is None
    controller.restore_control_targets_cache(snapshot)
    snapshot.positions[1] = 77.0
    np.testing.assert_allclose(
        controller.last_control_targets.positions[:2], [88.0, 0.2]
    )


def test_apply_failure_does_not_commit_effort_or_control_target_cache() -> None:
    controller, robot = _controller(
        ComponentControlSettings(
            mode="effort",
            method="direct",
            max_force=10.0,
            effort_limit=10.0,
        )
    )
    controller.configure_runtime()
    initial = ControlTargets(
        positions=robot.positions,
        velocities=np.zeros(robot.num_dof),
        efforts=np.asarray([1.0, 2.0, 0.0]),
    )
    controller.apply_targets(_Action, initial)
    cached_before = controller.snapshot_control_targets_cache()
    efforts_before = controller.last_commanded_efforts.copy()
    calls = 0

    def fail_on_follower(action) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("follower action failed")

    robot.apply_action = fail_on_follower
    changed = ControlTargets(
        positions=robot.positions,
        velocities=np.zeros(robot.num_dof),
        efforts=np.asarray([7.0, 8.0, 0.0]),
    )

    with pytest.raises(RuntimeError, match="follower action failed"):
        controller.apply_targets(_Action, changed)

    assert cached_before is not None
    np.testing.assert_allclose(
        controller.last_control_targets.efforts,
        cached_before.efforts,
    )
    np.testing.assert_allclose(
        controller.last_commanded_efforts,
        efforts_before,
        equal_nan=True,
    )


def test_configure_runtime_preserves_unmanaged_dof_gains() -> None:
    robot = _FakeRobot()
    initial_kps = np.asarray([3.0, 23.0, 47.0], dtype=float)
    initial_kds = np.asarray([0.3, 2.3, 4.7], dtype=float)
    robot.controller.gains = (initial_kps.copy(), initial_kds.copy())
    settings = ComponentControlSettings(
        mode="position",
        method="implicit",
        stiffness=(11.0,),
        damping=(1.1,),
    )
    controller = JointController(
        robot,
        joint_names=["arm_joint"],
        settings=JointControlSettings(default=settings, arm=settings, hand=settings),
    )

    controller.configure_runtime()

    assert robot.controller.gains is not None
    actual_kps, actual_kds = robot.controller.gains
    np.testing.assert_allclose(actual_kps, [11.0, 23.0, 47.0])
    np.testing.assert_allclose(actual_kds, [1.1, 2.3, 4.7])


def test_explicit_runtime_zeroes_managed_drive_and_preserves_unmanaged_gains() -> None:
    robot = _FakeRobot()
    robot.controller.gains = (
        np.asarray([3.0, 23.0, 47.0], dtype=float),
        np.asarray([0.3, 2.3, 4.7], dtype=float),
    )
    settings = ComponentControlSettings(
        mode="position",
        method="explicit",
        stiffness=(11.0,),
        damping=(1.1,),
    )
    controller = JointController(
        robot,
        joint_names=["arm_joint"],
        settings=JointControlSettings(default=settings, arm=settings, hand=settings),
    )

    controller.configure_runtime()

    assert robot.controller.gains is not None
    actual_kps, actual_kds = robot.controller.gains
    np.testing.assert_allclose(actual_kps, [0.0, 23.0, 47.0])
    np.testing.assert_allclose(actual_kds, [0.0, 2.3, 4.7])


def _nonstandard_mimic_urdf(path: Path) -> Path:
    path.write_text(
        """<robot name="test">
  <joint name="axis_b" type="revolute"/>
  <joint name="axis_shadow" type="revolute">
    <mimic joint="axis_b" multiplier="1" offset="0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_nonstandard_joint_groups_and_name_maps_configure_correct_components(
    tmp_path: Path,
) -> None:
    robot = _FakeRobot()
    robot.dof_names = ["axis_a", "axis_b", "axis_shadow"]
    mapping = RobotComponentMapping.from_profile(
        {"joint_groups": {"arm": ["axis_a"], "hand": ["axis_b"]}}
    )
    settings = JointControlSettings(
        default=ComponentControlSettings(stiffness=(99.0,), damping=(9.0,)),
        arm=ComponentControlSettings(
            stiffness={"axis_a": 11.0},
            damping=(1.0,),
            max_force={"axis_a": 101.0},
        ),
        hand=ComponentControlSettings(
            stiffness={"axis_b": 22.0},
            damping=2.0,
            max_force={"axis_b": 202.0},
            follower_stiffness={"axis_shadow": 33.0},
            follower_damping=(3.0,),
            follower_max_force={"axis_shadow": 303.0},
        ),
    )
    controller = JointController(
        robot,
        joint_names=["all"],
        settings=settings,
        mimic_path=_nonstandard_mimic_urdf(tmp_path / "robot.urdf"),
        component_mapping=mapping,
    )

    controller.configure_runtime()

    assert robot.controller.gains is not None
    np.testing.assert_allclose(robot.controller.gains[0], [11.0, 22.0, 33.0])
    np.testing.assert_allclose(robot.controller.gains[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(robot.controller.max_efforts, [101.0, 202.0, 303.0])


def test_controller_name_map_requires_exact_selected_joint_coverage() -> None:
    robot = _FakeRobot()
    robot.dof_names = ["axis_a", "axis_b", "unmanaged"]
    settings = ComponentControlSettings(
        stiffness={"axis_a": 1.0, "unknown": 2.0},
        damping=1.0,
    )
    controller = JointController(
        robot,
        joint_names=["axis_a", "axis_b"],
        settings=JointControlSettings(default=settings),
    )

    with pytest.raises(ValueError, match="unknown=.*missing="):
        controller.configure_runtime()


def test_native_mimic_excludes_follower_from_python_drive_and_actions(
    tmp_path: Path,
) -> None:
    robot = _FakeRobot()
    robot.dof_names = ["axis_a", "axis_b", "axis_shadow"]
    controller = JointController(
        robot,
        joint_names=["all"],
        settings=JointControlSettings(),
        mimic_path=_nonstandard_mimic_urdf(tmp_path / "native.urdf"),
        native_mimic=True,
    )

    controller.configure_runtime()
    targets = controller.build_control_targets(
        command_positions=np.asarray([0.1, 0.2]),
        base_positions=robot.positions,
    )
    controller.apply_targets(_Action, targets)

    np.testing.assert_array_equal(controller.follower_indices, [2])
    np.testing.assert_array_equal(controller.driven_indices, [0, 1])
    assert controller.follower_mapper.controls == []
    assert all(2 not in action.joint_indices for action in robot.actions)


def test_prepare_runtime_is_engine_and_cache_side_effect_free() -> None:
    controller, robot = _controller(
        ComponentControlSettings(mode="position", method="implicit")
    )
    controller.configure_runtime()
    cached = controller.build_control_targets(
        command_positions=np.asarray([0.4, 0.2]),
        base_positions=robot.positions,
    )
    controller.apply_targets(_Action, cached)
    actions_before = tuple(robot.actions)
    switches_before = tuple(robot.controller.mode_switches)
    gains_before = tuple(value.copy() for value in robot.controller.gains)
    cache_before = controller.snapshot_control_targets_cache()
    previous_settings = controller.settings
    velocity = ComponentControlSettings(
        mode="velocity",
        method="explicit",
        damping=4.0,
        max_force=3.0,
    )

    prepared = controller.prepare_runtime(
        JointControlSettings(default=velocity, arm=velocity, hand=velocity)
    )

    assert controller.settings is previous_settings
    assert tuple(robot.actions) == actions_before
    assert tuple(robot.controller.mode_switches) == switches_before
    assert robot.controller.gains is not None
    np.testing.assert_allclose(robot.controller.gains[0], gains_before[0])
    np.testing.assert_allclose(robot.controller.gains[1], gains_before[1])
    assert cache_before is not None
    np.testing.assert_allclose(
        controller.last_control_targets.positions,
        cache_before.positions,
    )
    assert prepared.active_specs == (
        (0, "velocity", "explicit"),
        (1, "velocity", "explicit"),
    )
    assert prepared.runtime_modes[:2] == ((0, "effort"), (1, "effort"))


def test_apply_prepared_runtime_commits_host_state_only_after_engine_success() -> None:
    controller, robot = _controller(
        ComponentControlSettings(mode="position", method="implicit")
    )
    controller.configure_runtime()
    previous_settings = controller.settings
    velocity = ComponentControlSettings(
        mode="velocity",
        method="explicit",
        damping=4.0,
        max_force=3.0,
    )
    candidate = JointControlSettings(
        default=velocity,
        arm=velocity,
        hand=velocity,
    )
    prepared = controller.prepare_runtime(candidate)
    original_switch = robot.controller.switch_dof_control_mode

    def fail_switch(*, dof_index: int, mode: str) -> None:
        if dof_index == 1:
            raise RuntimeError("mode write failed")
        original_switch(dof_index=dof_index, mode=mode)

    robot.controller.switch_dof_control_mode = fail_switch

    with pytest.raises(RuntimeError, match="mode write failed"):
        controller.apply_prepared_runtime(prepared, clear_target_cache=True)

    assert controller.settings is previous_settings
    assert controller.command_target_modes == ("position", "position")


def test_apply_prepared_runtime_rejects_another_controller_plan() -> None:
    first, _ = _controller(ComponentControlSettings())
    second, _ = _controller(ComponentControlSettings())

    with pytest.raises(ValueError, match="another JointController"):
        second.apply_prepared_runtime(
            first.prepare_runtime(),
            clear_target_cache=True,
        )
