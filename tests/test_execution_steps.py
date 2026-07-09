from __future__ import annotations

import numpy as np

from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    JointControlSettings,
)
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.steps import (
    CommandPositionTrajectoryStep,
    HoldCommandPositionTargetStep,
    SmoothCommandPositionTargetStep,
    SwitchControlModeStep,
    execute_command_position_trajectory,
)
from linkerbot_sim.trajectories.types import JointTrajectory


class _FakeWorld:
    def __init__(self) -> None:
        self.step_calls = []

    def get_physics_dt(self) -> float:
        return 0.1

    def step(self, *, render: bool) -> None:
        self.step_calls.append(bool(render))


class _FakeArticulation:
    def __init__(self) -> None:
        self.dof_names = ["j0", "j1", "j2"]
        self.num_dof = 3
        self.positions = np.asarray([0.1, 0.2, 0.3], dtype=float)
        self.velocities = np.zeros(3, dtype=float)

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def set_joint_velocities(self, values) -> None:
        self.velocities = np.asarray(values, dtype=float).copy()


class _FakeTargets:
    def __init__(self, positions, velocities, efforts) -> None:
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.efforts = np.asarray(efforts, dtype=float)


class _FakeController:
    def __init__(self, settings: JointControlSettings) -> None:
        self.settings = settings
        self.configure_runtime_calls = 0

    def configure_runtime(self) -> None:
        self.configure_runtime_calls += 1


class _FakeCommandController:
    def __init__(self) -> None:
        self.dof_names = ["j0", "j1", "j2"]
        self.command_indices = np.asarray([0, 2], dtype=int)
        self.driven_indices = np.asarray([0, 2], dtype=int)
        self.build_calls = []
        self.applied_targets = []

    def build_control_targets(
        self,
        command_positions=None,
        command_velocities=None,
        command_efforts=None,
        *,
        base_positions=None,
    ):
        base = np.asarray(base_positions, dtype=float).copy()
        self.build_calls.append((command_positions, command_velocities, base.copy()))
        base[self.command_indices] = np.asarray(command_positions, dtype=float)
        velocities = np.zeros(3, dtype=float)
        velocities[self.command_indices] = np.asarray(command_velocities, dtype=float)
        efforts = np.zeros(3, dtype=float)
        efforts[self.command_indices] = np.asarray(command_efforts, dtype=float)
        return _FakeTargets(base, velocities, efforts)

    def apply_targets(self, _articulation_action_type, targets) -> None:
        self.applied_targets.append(targets)


class _FakeLogger:
    def __init__(self) -> None:
        self.rows = []

    def should_write(self, _step: int) -> bool:
        return True

    def collect_step_values(self, _robot, _controller, targets, joint_indices):
        indices = np.asarray(joint_indices, dtype=int)
        return {
            "desired_position": targets.positions[indices],
            "actual_position": targets.positions[indices],
            "desired_velocity": targets.velocities[indices],
            "actual_velocity": targets.velocities[indices],
        }

    def write(self, **row) -> None:
        self.rows.append(row)


def test_switch_control_mode_step_reconfigures_controller_without_stepping() -> None:
    initial_settings = JointControlSettings(
        default=ComponentControlSettings(mode="position", method="implicit")
    )
    next_settings = JointControlSettings(
        default=ComponentControlSettings(mode="effort", method="direct")
    )
    controller = _FakeController(initial_settings)
    runtime = ExecutionRuntime(
        articulation=object(),
        simulation_world=object(),
        articulation_action_type=object(),
        joint_controller=controller,
        simulation_app=None,
        render_enabled=False,
    )

    step = SwitchControlModeStep(settings=next_settings, phase="switch_to_effort").run(
        runtime, 42
    )

    assert step == 42
    assert controller.settings is next_settings
    assert controller.configure_runtime_calls == 1


def test_execute_command_position_trajectory_expands_command_space_targets() -> None:
    articulation = _FakeArticulation()
    world = _FakeWorld()
    controller = _FakeCommandController()
    logger = _FakeLogger()
    trajectory = JointTrajectory.from_samples(
        times=np.asarray([0.0, 0.1]),
        positions=np.asarray([[1.0, 2.0], [1.5, 2.5]]),
        velocities=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        efforts=np.asarray([[0.0, 0.5], [0.0, 0.6]]),
        joint_names=("j0", "j2"),
        phases=("a", "b"),
    )

    step = execute_command_position_trajectory(
        articulation=articulation,
        simulation_world=world,
        articulation_action_type=object,
        joint_controller=controller,
        trajectory=trajectory,
        simulation_app=None,
        render_enabled=False,
        drive_logger=logger,
    )

    assert step == 2
    assert world.step_calls == [False, False]
    np.testing.assert_allclose(
        controller.applied_targets[-1].positions, [1.5, 0.2, 2.5]
    )
    assert logger.rows[-1]["phase"] == "b"


def test_command_position_steps_use_command_space_controller() -> None:
    articulation = _FakeArticulation()
    world = _FakeWorld()
    controller = _FakeCommandController()
    runtime = ExecutionRuntime(
        articulation=articulation,
        simulation_world=world,
        articulation_action_type=object,
        joint_controller=controller,
        simulation_app=None,
        render_enabled=False,
    )

    step = SmoothCommandPositionTargetStep(
        start_command=np.asarray([0.0, 1.0]),
        target_command=np.asarray([1.0, 2.0]),
        duration=0.2,
        phase="smooth_command",
    ).run(runtime, 0)
    step = HoldCommandPositionTargetStep(
        target_command=np.asarray([1.0, 2.0]),
        duration=0.1,
        phase="hold_command",
    ).run(runtime, step)
    step = CommandPositionTrajectoryStep(
        JointTrajectory.from_samples(
            times=np.asarray([0.1]),
            positions=np.asarray([[1.5, 2.5]]),
            velocities=np.asarray([[0.0, 0.0]]),
            joint_names=("j0", "j2"),
            phases=("trajectory_command",),
        )
    ).run(runtime, step)

    assert step == 4
    assert len(controller.build_calls) == 4
    np.testing.assert_allclose(
        controller.applied_targets[-1].positions, [1.5, 0.2, 2.5]
    )
