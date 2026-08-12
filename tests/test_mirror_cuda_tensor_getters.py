"""Mirror 冷边界对 CUDA-like runtime getter 的处理合同。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.mirror.motion.timeline.compiler import _current_command
from linkerbot_sim.mirror.motion.timeline.executor import _ExecutionState
from linkerbot_sim.mirror.collision.robot_provider import RobotObstacleProvider
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.controllers.joint_controller import JointController
from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlTargets,
    JointControlSettings,
)
from linkerbot_sim.execution.steps import execute_command_position_hold
from linkerbot_sim.configuration.outputs import LoggingOutputSettings
from linkerbot_sim.logging.joint_logger import JointTrackingLogger
from linkerbot_sim.objects.state_views import SceneObjectStateView
from linkerbot_sim.snapshots.runtime_objects import _robot_snapshot_from_execution
from linkerbot_sim.telemetry.state_snapshot import SceneRobotStateSampler
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


class _CudaLikeTensor:
    """Tensor fake that rejects every host conversion until ``cpu()`` is called."""

    def __init__(
        self,
        values: object,
        *,
        on_cpu: bool = False,
        calls: list[str] | None = None,
    ) -> None:
        self._values = np.asarray(values)
        self._on_cpu = bool(on_cpu)
        self.calls = [] if calls is None else calls

    def detach(self) -> _CudaLikeTensor:
        self.calls.append("detach")
        return self

    def cpu(self) -> _CudaLikeTensor:
        self.calls.append("cpu")
        return _CudaLikeTensor(
            self._values,
            on_cpu=True,
            calls=self.calls,
        )

    def numpy(self) -> np.ndarray:
        self.calls.append("numpy")
        if not self._on_cpu:
            raise RuntimeError("CUDA tensor must be moved to CPU before numpy()")
        return self._values.copy()

    def __array__(self, dtype=None, copy=None):
        del dtype, copy
        raise RuntimeError("direct np.asarray() on a CUDA tensor is forbidden")


class _Action:
    def __init__(
        self,
        *,
        joint_positions=None,
        joint_velocities=None,
        joint_efforts=None,
        joint_indices=None,
    ) -> None:
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_efforts = joint_efforts
        self.joint_indices = joint_indices


class _IsaacController:
    def __init__(self, dof_count: int) -> None:
        self.dof_count = int(dof_count)
        self.gains = None
        self.max_efforts = None
        self.mode_switches: list[tuple[int, str]] = []

    def get_gains(self):
        return (
            _CudaLikeTensor(np.zeros(self.dof_count)),
            _CudaLikeTensor(np.zeros(self.dof_count)),
        )

    def set_gains(self, *, kps, kds) -> None:
        self.gains = (np.asarray(kps), np.asarray(kds))

    def set_max_efforts(self, values, joint_indices=None) -> None:
        del joint_indices
        self.max_efforts = np.asarray(values)

    def set_effort_modes(self, mode, joint_indices=None) -> None:
        del mode, joint_indices

    def switch_dof_control_mode(self, *, dof_index: int, mode: str) -> None:
        self.mode_switches.append((int(dof_index), str(mode)))


class _CudaArticulation:
    dof_names = ("j0", "j1")
    num_dof = 2

    def __init__(self) -> None:
        self.positions = np.asarray([0.1, -0.2], dtype=float)
        self.velocities = np.asarray([0.3, -0.4], dtype=float)
        self.position_targets = np.asarray([0.5, -0.6], dtype=float)
        self.controller = _IsaacController(self.num_dof)
        self._articulation_view = self
        self.actions: list[_Action] = []

    def get_articulation_controller(self):
        return self.controller

    def get_max_efforts(self):
        return _CudaLikeTensor([10.0, 20.0])

    def get_joint_positions(self):
        return _CudaLikeTensor(self.positions)

    def get_joint_velocities(self):
        return _CudaLikeTensor(self.velocities)

    def get_joint_position_targets(self):
        return _CudaLikeTensor(self.position_targets)

    def get_measured_joint_efforts(self, joint_indices=None):
        values = np.asarray([1.0, 2.0], dtype=float)
        if joint_indices is not None:
            values = values[np.asarray(joint_indices, dtype=int)]
        return _CudaLikeTensor(values)

    def get_applied_joint_efforts(self, joint_indices=None):
        values = np.asarray([3.0, 4.0], dtype=float)
        if joint_indices is not None:
            values = values[np.asarray(joint_indices, dtype=int)]
        return _CudaLikeTensor(values)

    def apply_action(self, action: _Action) -> None:
        self.actions.append(action)


def _joint_controller(robot: _CudaArticulation) -> JointController:
    settings = ComponentControlSettings(mode="position", method="implicit")
    return JointController(
        robot,
        joint_names=list(robot.dof_names),
        settings=JointControlSettings(default=settings),
    )


def test_cuda_like_helper_and_joint_controller_read_back_on_cpu() -> None:
    tensor = _CudaLikeTensor([1.0, 2.0])
    np.testing.assert_allclose(tensor_like_to_numpy(tensor), [1.0, 2.0])
    assert tensor.calls == ["detach", "cpu", "numpy"]

    robot = _CudaArticulation()
    controller = _joint_controller(robot)
    controller.configure_runtime()
    targets = controller.build_control_targets(
        command_positions=np.asarray([0.7, -0.8]),
    )
    controller.apply_targets(_Action, targets)

    assert robot.controller.gains is not None
    assert robot.controller.gains[0].shape == (robot.num_dof,)
    assert robot.controller.max_efforts.shape == (robot.num_dof,)
    assert robot.actions


class _World:
    def __init__(self) -> None:
        self.steps = 0

    def get_physics_dt(self) -> float:
        return 0.1

    def step(self, *, render: bool) -> None:
        del render
        self.steps += 1


class _CommandController:
    command_indices = np.asarray([0, 1], dtype=int)
    driven_indices = command_indices
    command_joint_names = ("j0", "j1")

    def __init__(self) -> None:
        self.applied: list[ControlTargets] = []

    def build_control_targets(
        self,
        command_positions=None,
        command_velocities=None,
        command_efforts=None,
        *,
        base_positions=None,
    ) -> ControlTargets:
        base = np.asarray(base_positions, dtype=float).copy()
        base[self.command_indices] = np.asarray(command_positions, dtype=float)
        return ControlTargets(base, command_velocities, command_efforts)

    def apply_targets(self, action_type, targets: ControlTargets) -> None:
        del action_type
        self.applied.append(targets)


def test_execution_and_timeline_baselines_accept_cuda_getters() -> None:
    robot = _CudaArticulation()
    controller = _CommandController()
    world = _World()

    step = execute_command_position_hold(
        articulation=robot,
        simulation_world=world,
        articulation_action_type=_Action,
        joint_controller=controller,
        target_command=np.asarray([0.8, -0.9]),
        duration=0.1,
        phase="gpu",
        simulation_app=None,
        render_enabled=False,
        step=0,
    )
    execution = SimpleNamespace(
        articulation=robot,
        joint_controller=controller,
        articulation_action_type=_Action,
    )
    runtime_robot = SimpleNamespace(robot_id=4, execution=execution)

    assert step == 1
    assert world.steps == 1
    np.testing.assert_allclose(_current_command(runtime_robot), robot.positions)
    state = _ExecutionState.from_robot(runtime_robot)
    np.testing.assert_allclose(state.base_positions, robot.positions)


def test_snapshot_telemetry_and_logging_accept_cuda_getters() -> None:
    robot = _CudaArticulation()
    controller = _CommandController()
    execution = SimpleNamespace(articulation=robot, joint_controller=controller)

    snapshot = _robot_snapshot_from_execution(
        label="gpu_robot",
        robot_id=7,
        execution=execution,
        robot_profile="test",
        asset_fingerprint=None,
    )
    np.testing.assert_allclose(snapshot.joint_positions, robot.positions)
    np.testing.assert_allclose(snapshot.joint_velocities, robot.velocities)
    np.testing.assert_allclose(snapshot.command_targets, robot.position_targets)

    runtime = SimpleNamespace(
        world=_World(),
        robots_by_id={
            7: SimpleNamespace(
                robot_id=7,
                label="gpu_robot",
                execution=SimpleNamespace(
                    articulation=robot,
                    joint_controller=SimpleNamespace(
                        last_commanded_efforts=np.asarray([0.0, 0.0])
                    ),
                ),
            )
        },
    )
    sampled = SceneRobotStateSampler(
        stage=None,
        include_efforts=True,
    ).sample(runtime, step=0)
    np.testing.assert_allclose(sampled.robots[0].positions_rad, robot.positions)
    np.testing.assert_allclose(sampled.robots[0].measured_efforts, [1.0, 2.0])

    logger = JointTrackingLogger(
        None,
        list(robot.dof_names),
        settings=LoggingOutputSettings(
            enabled=False,
            existing_data_policy="error",
            joint_tracking_path=None,
            flush_interval_s=0.05,
            interval_steps=1,
            log_actual_position=True,
            log_actual_velocity=True,
            log_command_position=True,
            log_command_velocity=True,
            log_command_effort=False,
            log_action_effort=False,
            log_measured_effort=False,
            log_applied_effort=False,
        ),
        flush_interval_steps=1,
    )
    values = logger.collect_step_values(
        robot,
        SimpleNamespace(last_commanded_efforts=None),
        ControlTargets(robot.positions, robot.velocities, np.zeros(robot.num_dof)),
        np.asarray([0, 1], dtype=int),
    )
    np.testing.assert_allclose(values["actual_position"], robot.positions)
    np.testing.assert_allclose(values["actual_velocity"], robot.velocities)


class _RigidView:
    def get_world_poses(self, *, indices):
        count = np.asarray(indices).size
        return (
            _CudaLikeTensor(np.tile([1.0, 2.0, 3.0], (count, 1))),
            _CudaLikeTensor(np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))),
        )

    def get_velocities(self, *, indices):
        count = np.asarray(indices).size
        return _CudaLikeTensor(np.tile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (count, 1)))


class _Kinematics:
    def link_transforms(self, joint_values):
        assert joint_values == {"j0": 0.1, "j1": -0.2}
        return {"tool": np.eye(4, dtype=float)}


def test_collision_and_object_state_accept_cuda_getters() -> None:
    view = SceneObjectStateView(root_view=_RigidView())
    position, orientation = view.root_world_pose()
    linear, angular = view.root_velocities()
    np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(orientation, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(linear, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(angular, [0.4, 0.5, 0.6])

    provider = object.__new__(RobotObstacleProvider)
    provider.robot_id = 3
    provider.label = "gpu_robot"
    provider.articulation = _CudaArticulation()
    provider.root_pose = RootPoseConfig()
    provider.urdf_path = "fake.urdf"
    provider._kinematics = _Kinematics()
    provider._spheres = {"tool": ((np.zeros(3, dtype=float), 0.05),)}

    obstacles = provider.collision_objects()
    assert len(obstacles) == 1
    assert obstacles[0].name == "robot_3_tool_0"
