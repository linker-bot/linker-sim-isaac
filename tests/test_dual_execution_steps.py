from __future__ import annotations

import numpy as np

from linkerbot_sim.execution.dual_runtime import DualRobotRuntime, RobotSideRuntime
from linkerbot_sim.execution.dual_steps import (
    DualCommandExecutionInterrupted,
    execute_dual_command_position_trajectory,
)
from linkerbot_sim.trajectories.types import JointTrajectory


class _FakeWorld:
    def __init__(self) -> None:
        self.step_calls = 0

    def get_physics_dt(self) -> float:
        return 0.1

    def step(self, *, render: bool) -> None:
        self.step_calls += 1


class _FakeArticulation:
    def __init__(self) -> None:
        self.num_dof = 2
        self.positions = np.asarray([0.0, 0.0], dtype=float)
        self.velocities = np.zeros(2, dtype=float)

    def get_joint_positions(self):
        return self.positions.copy()

    def set_joint_velocities(self, values) -> None:
        self.velocities = np.asarray(values, dtype=float)


class _FakeController:
    command_indices = np.asarray([0, 1], dtype=int)
    driven_indices = np.asarray([0, 1], dtype=int)
    command_joint_names = ("j0", "j1")

    def __init__(self) -> None:
        self.applied = []

    def build_control_targets(
        self,
        command_positions,
        command_velocities,
        command_efforts,
        *,
        base_positions,
    ):
        class _Targets:
            pass

        targets = _Targets()
        targets.positions = np.asarray(command_positions, dtype=float)
        targets.velocities = np.asarray(command_velocities, dtype=float)
        targets.efforts = np.asarray(command_efforts, dtype=float)
        return targets

    def apply_targets(self, _action_type, targets) -> None:
        self.applied.append(targets)


def _side(side: str) -> RobotSideRuntime:
    return RobotSideRuntime(
        side=side,
        articulation=_FakeArticulation(),
        joint_controller=_FakeController(),
    )


def test_dual_command_trajectory_can_be_interrupted() -> None:
    runtime = DualRobotRuntime(
        left=_side("left"),
        right=_side("right"),
        simulation_world=_FakeWorld(),
        articulation_action_type=object,
        simulation_app=None,
        render_enabled=False,
    )
    trajectory = JointTrajectory.from_samples(
        times=np.asarray([0.1, 0.2, 0.3]),
        positions=np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        joint_names=("j0", "j1"),
        phases=("a", "b", "c"),
    )
    checks = {"count": 0}

    def should_stop() -> bool:
        checks["count"] += 1
        return checks["count"] > 1

    try:
        execute_dual_command_position_trajectory(
            runtime=runtime,
            left_trajectory=trajectory,
            right_trajectory=trajectory,
            step=0,
            should_stop=should_stop,
        )
    except DualCommandExecutionInterrupted:
        pass
    else:
        raise AssertionError("expected trajectory playback to be interrupted")

    assert runtime.simulation_world.step_calls == 1
