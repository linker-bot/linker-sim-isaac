from __future__ import annotations

import numpy as np

from linkerbot_sim.assets.robot_loader import DualRobotExecutionConfig
from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.execution.dual_runtime import DualRobotRuntime, RobotSideRuntime
from linkerbot_sim.execution.dual_steps import DualCommandPositionTargetStep
from linkerbot_sim.utils.config import load_yaml


class FakeWorld:
    def __init__(self, physics_dt: float = 0.01) -> None:
        self.physics_dt = physics_dt
        self.steps: list[bool] = []

    def get_physics_dt(self) -> float:
        return self.physics_dt

    def step(self, *, render: bool) -> None:
        self.steps.append(bool(render))


class FakeArticulation:
    def __init__(self, positions: np.ndarray) -> None:
        self.positions = np.asarray(positions, dtype=float)
        self.num_dof = self.positions.size
        self.zeroed_velocities: list[np.ndarray] = []

    def get_joint_positions(self) -> np.ndarray:
        return self.positions.copy()

    def set_joint_velocities(self, velocities: np.ndarray) -> None:
        self.zeroed_velocities.append(np.asarray(velocities, dtype=float).copy())


class FakeController:
    def __init__(self, command_indices: list[int]) -> None:
        self.command_indices = np.asarray(command_indices, dtype=int)
        self.driven_indices = self.command_indices.copy()
        self.applied: list[ControlTargets] = []

    def build_control_targets(
        self,
        *,
        command_positions,
        command_velocities,
        command_efforts,
        base_positions,
    ) -> ControlTargets:
        positions = np.asarray(base_positions, dtype=float).copy()
        velocities = np.zeros_like(positions)
        efforts = np.zeros_like(positions)
        positions[self.command_indices] = np.asarray(command_positions, dtype=float)
        velocities[self.command_indices] = np.asarray(command_velocities, dtype=float)
        efforts[self.command_indices] = np.asarray(command_efforts, dtype=float)
        return ControlTargets(positions, velocities, efforts)

    def apply_targets(self, _action_type, targets: ControlTargets) -> None:
        self.applied.append(targets)


def test_dual_robot_execution_config_parses_left_and_right_assets() -> None:
    yaml_config = load_yaml("configs/robots/ar5v2_l6v1_dual.yaml")
    config = DualRobotExecutionConfig.from_mapping(yaml_config)

    assert config.left.robot.asset_path.name == "AR5V2_L6V1_L.xml"
    assert config.right.robot.asset_path.name == "AR5V2_L6V1_R.xml"
    assert config.left.robot.prim_path == "/World/AR5V2_L6V1_L"
    assert config.right.robot.prim_path == "/World/AR5V2_L6V1_R"
    np.testing.assert_allclose(
        config.left.root_pose.xyz,
        yaml_config["robots"]["left"]["root_pose"]["xyz"],
    )
    np.testing.assert_allclose(
        config.left.root_pose.rpy,
        yaml_config["robots"]["left"]["root_pose"]["rpy"],
    )
    np.testing.assert_allclose(
        config.right.root_pose.xyz,
        yaml_config["robots"]["right"]["root_pose"]["xyz"],
    )
    np.testing.assert_allclose(
        config.right.root_pose.rpy,
        yaml_config["robots"]["right"]["root_pose"]["rpy"],
    )
    assert config.left.controlled_joints == ("all",)
    assert config.right.controlled_joints == ("all",)


def test_dual_command_target_step_applies_both_sides_before_single_world_step() -> None:
    world = FakeWorld()
    left_controller = FakeController([0, 1])
    right_controller = FakeController([0, 1])
    runtime = DualRobotRuntime(
        left=RobotSideRuntime(
            side="left",
            articulation=FakeArticulation(np.asarray([0.0, 0.0])),
            joint_controller=left_controller,
        ),
        right=RobotSideRuntime(
            side="right",
            articulation=FakeArticulation(np.asarray([10.0, 20.0])),
            joint_controller=right_controller,
        ),
        simulation_world=world,
        articulation_action_type=object,
        simulation_app=None,
        render_enabled=False,
    )

    step = DualCommandPositionTargetStep(
        left_start_command=np.asarray([0.0, 0.0]),
        left_target_command=np.asarray([1.0, 2.0]),
        right_start_command=np.asarray([10.0, 20.0]),
        right_target_command=np.asarray([30.0, 40.0]),
        duration=0.01,
        phase="dual_step",
    ).run(runtime, 0)

    assert step == 1
    assert len(world.steps) == 1
    assert len(left_controller.applied) == 1
    assert len(right_controller.applied) == 1
    np.testing.assert_allclose(left_controller.applied[0].positions, [1.0, 2.0])
    np.testing.assert_allclose(right_controller.applied[0].positions, [30.0, 40.0])


def test_dual_command_target_step_holds_missing_side() -> None:
    world = FakeWorld()
    left_controller = FakeController([0, 1])
    right_controller = FakeController([0, 1])
    runtime = DualRobotRuntime(
        left=RobotSideRuntime(
            side="left",
            articulation=FakeArticulation(np.asarray([0.0, 0.0])),
            joint_controller=left_controller,
        ),
        right=RobotSideRuntime(
            side="right",
            articulation=FakeArticulation(np.asarray([10.0, 20.0])),
            joint_controller=right_controller,
        ),
        simulation_world=world,
        articulation_action_type=object,
        simulation_app=None,
        render_enabled=False,
    )

    DualCommandPositionTargetStep(
        left_start_command=np.asarray([0.0, 0.0]),
        left_target_command=np.asarray([1.0, 2.0]),
        right_target_command=None,
        duration=0.01,
        phase="left_move_right_hold",
    ).run(runtime, 0)

    assert len(world.steps) == 1
    assert len(left_controller.applied) == 1
    assert len(right_controller.applied) == 1
    np.testing.assert_allclose(left_controller.applied[0].positions, [1.0, 2.0])
    np.testing.assert_allclose(right_controller.applied[0].positions, [10.0, 20.0])
