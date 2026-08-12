from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.configuration.catalog import load_mirror_config
from linkerbot_sim.controllers.hybrid_force_position import TaskSpaceObservation
from linkerbot_sim.controllers.projection import joint_control_settings
from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.mirror.hybrid_parameters import HybridParameterService
from linkerbot_sim.mirror.motion.hybrid_executor import (
    HybridCancelledError,
    HybridExecutionError,
    HybridParameterGenerationError,
    HybridRestoreFailedError,
    HybridTareStaleError,
    MirrorHybridExecutor,
    parse_hybrid_motion_request,
)
from linkerbot_sim.robots.tcp_binding import PhysicalTcpBinding


ARM_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
COMMAND_NAMES = (*ARM_NAMES, "hand")


class _Articulation:
    num_dof = 7

    def __init__(self) -> None:
        self.q = np.zeros(7, dtype=float)
        self.q[-1] = 0.4
        self.qd = np.zeros(7, dtype=float)

    def get_joint_positions(self):
        return self.q.copy()

    def get_joint_velocities(self):
        return self.qd.copy()


@dataclass
class _Prepared:
    settings: object
    active_effort_limits: np.ndarray


class _Controller:
    def __init__(self, articulation: _Articulation, profiles: object) -> None:
        self.robot = articulation
        self.command_indices = np.arange(7, dtype=int)
        self.command_joint_names = COMMAND_NAMES
        self.settings = joint_control_settings(profiles, mode="position")
        self._targets = ControlTargets(
            articulation.q.copy(),
            np.zeros(7),
            np.zeros(7),
        )
        self.writes: list[tuple[object, ControlTargets]] = []
        self.fail_restore = False

    @property
    def command_target_modes(self):
        return tuple(
            self.settings.arm.mode if name in ARM_NAMES else self.settings.hand.mode
            for name in self.command_joint_names
        )

    def prepare_runtime(self, settings=None):
        selected = self.settings if settings is None else settings
        return _Prepared(selected, np.full(7, 100.0, dtype=float))

    def apply_prepared_runtime(self, prepared, *, clear_target_cache: bool):
        del clear_target_cache
        if self.fail_restore and prepared.settings.arm.mode == "position":
            raise RuntimeError("restore injected failure")
        self.settings = prepared.settings

    def snapshot_control_targets_cache(self):
        return ControlTargets(
            self._targets.positions,
            self._targets.velocities,
            self._targets.efforts,
        )

    def build_control_targets(
        self,
        command_positions=None,
        command_velocities=None,
        command_efforts=None,
        *,
        base_positions=None,
    ):
        positions = (
            self.robot.q.copy()
            if base_positions is None
            else np.asarray(base_positions, dtype=float).copy()
        )
        velocities = np.zeros(7, dtype=float)
        efforts = np.zeros(7, dtype=float)
        if command_positions is not None:
            positions[self.command_indices] = command_positions
        if command_velocities is not None:
            velocities[self.command_indices] = command_velocities
        if command_efforts is not None:
            efforts[self.command_indices] = command_efforts
        return ControlTargets(positions, velocities, efforts)

    def targets_from_full_state(self, positions, velocities, efforts):
        return ControlTargets(positions, velocities, efforts)

    def apply_targets(self, _action_type, targets):
        copy = ControlTargets(targets.positions, targets.velocities, targets.efforts)
        self.writes.append((self.settings, copy))
        self._targets = copy


class _Port:
    def __init__(self) -> None:
        self.sequence = 0
        self.external = np.zeros(6, dtype=float)

    def observe(self):
        result = TaskSpaceObservation(
            position=np.zeros(3),
            orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            twist=np.zeros(6),
            jacobian=np.eye(6),
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            external_wrench_environment_on_tool=self.external,
            sequence=self.sequence,
        )
        self.sequence += 1
        return result


class _Physics:
    def __init__(self) -> None:
        self.steps = 0
        self.fail_at: int | None = None

    def get_physics_dt(self):
        return 1.0 / 240.0

    def step(self, *, render: bool):
        assert render is False
        if self.fail_at is not None and self.steps >= self.fail_at:
            raise RuntimeError("physics injected failure")
        self.steps += 1


class _Resources:
    def __init__(self, robots) -> None:
        self.robots_by_id = {robot.robot_id: robot for robot in robots}
        self.physics = _Physics()
        self.observed: list[tuple[int, str]] = []
        self._sample_step = 0
        self.hybrid_control_logger = None

    def robot(self, robot_id: int):
        return self.robots_by_id[robot_id]

    def claim_completed_step(self):
        result = self._sample_step
        self._sample_step += 1
        return result

    def observe_after_step(self, *, step, phase, write_idle_logs):
        assert write_idle_logs is False
        self.observed.append((step, phase))


class _RuntimeOwner:
    def __init__(self) -> None:
        self.fatal_error = None
        self.quit_requested = False

    def request_quit(self):
        self.quit_requested = True


def _fixture():
    config = load_mirror_config("physx_cpu_hybrid")
    settings = config.hybrid_control
    assert settings is not None
    profiles = config.controller_bundles["physx"]
    articulation = _Articulation()
    controller = _Controller(articulation, profiles)
    port = _Port()
    binding = PhysicalTcpBinding(
        tcp_frame_name="tcp",
        parent_frame_name="flange",
        parent_body_path="/World/Robot/flange",
        offset_xyz=(0.0, 0.0, 0.0),
        offset_rpy=(0.0, 0.0, 0.0),
    )
    robot = SimpleNamespace(
        robot_id=0,
        label="left",
        articulation=articulation,
        execution=SimpleNamespace(
            joint_controller=controller,
            articulation_action_type=object,
        ),
        joint_groups=SimpleNamespace(arm=ARM_NAMES),
        controller_profiles=profiles,
        physical_tcp_binding=binding,
        task_space_port=port,
    )
    other_controller = _Controller(_Articulation(), profiles)
    other = SimpleNamespace(
        robot_id=1,
        label="right",
        articulation=other_controller.robot,
        execution=SimpleNamespace(
            joint_controller=other_controller,
            articulation_action_type=object,
        ),
        joint_groups=SimpleNamespace(arm=ARM_NAMES),
        controller_profiles=profiles,
        physical_tcp_binding=binding,
        task_space_port=_Port(),
    )
    resources = _Resources((robot, other))
    parameters = HybridParameterService(settings)
    runtime = _RuntimeOwner()
    executor = MirrorHybridExecutor(
        resources,
        settings=settings,
        physics_engine="physx",
        physics_execution="cpu",
    )
    executor.bind_parameter_provider(parameters.snapshot)
    executor.bind_control_mode_provider(lambda: "position")
    executor.bind_runtime_owner(runtime)
    return executor, resources, robot, other, port, parameters, runtime


def _tare(executor: MirrorHybridExecutor, *, start_step: int = 0):
    return executor.tare_wrench(
        {
            "robot_id": 0,
            "robot_label": "left",
            "tcp_frame_name": "tcp",
            "reference_frame": "world",
        },
        start_step=start_step,
        should_cancel=lambda: False,
    )


def _motion(*, tare_generation: int = 1, parameter_generation: int = 0):
    return {
        "robot_id": 0,
        "robot_label": "left",
        "duration_s": 6.0 / 240.0,
        "tcp_frame_name": "tcp",
        "reference_frame": "world",
        "target_position": [0.0, 0.0, 0.0],
        "target_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "force_axes": [False, False, True, False, False, False],
        "target_wrench": [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
        "tare_generation": tare_generation,
        "hybrid_parameter_generation": parameter_generation,
        "phase": "normal_force_hold",
    }


def test_tare_then_hybrid_restores_position_and_leaves_other_robot_untouched() -> None:
    executor, resources, robot, other, port, _parameters, _runtime = _fixture()
    tare, step = _tare(executor)
    assert tare["tare_generation"] == 1
    assert tare["sample_count"] == 120
    port.external[2] = 2.0
    writes_before = len(robot.execution.joint_controller.writes)

    result, final_step = executor.execute(
        _motion(),
        start_step=step,
        should_cancel=lambda: False,
    )

    controller = robot.execution.joint_controller
    assert result["event"] == "hybrid_force_position_completed"
    assert result["executed_ticks"] == 6
    assert result["force_axes"] == [False, False, True, False, False, False]
    assert final_step == step + 6 + 36
    assert len(controller.writes) > writes_before
    assert any(settings.arm.mode == "effort" for settings, _ in controller.writes)
    assert controller.settings.arm.mode == "position"
    assert controller.settings.hand.method == "implicit"
    assert controller._targets.positions[-1] == pytest.approx(0.4)
    assert np.allclose(controller._targets.efforts, 0.0)
    assert other.execution.joint_controller.writes == []
    assert resources.physics.steps == final_step


def test_consecutive_motions_freeze_new_gains_and_independent_force_axes() -> None:
    executor, resources, _robot, _other, port, parameters, _runtime = _fixture()
    rows: list[dict[str, object]] = []
    resources.hybrid_control_logger = SimpleNamespace(
        write=lambda payload: rows.append(deepcopy(dict(payload)))
    )
    _tare_result, step = _tare(executor)
    port.external[2] = 2.0

    first, step = executor.execute(
        _motion(parameter_generation=0),
        start_step=step,
        should_cancel=lambda: False,
        request_id="segment-z",
    )
    changed = parameters.set_parameters(
        {"force_proportional": [0.3, 0.3, 0.4, 0.01, 0.01, 0.01]},
        expected_generation=0,
    )
    second_motion = _motion(parameter_generation=changed.generation)
    second_motion["force_axes"] = [True, False, False, False, False, False]
    second_motion["target_wrench"] = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    port.external[:] = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    second, _step = executor.execute(
        second_motion,
        start_step=step,
        should_cancel=lambda: False,
        request_id="segment-x",
    )

    assert first["hybrid_parameter_generation"] == 0
    assert first["force_axes"] == [False, False, True, False, False, False]
    assert second["hybrid_parameter_generation"] == 1
    assert second["force_axes"] == [True, False, False, False, False, False]
    assert {row["request_id"] for row in rows} == {"segment-z", "segment-x"}
    assert {row["hybrid_parameter_generation"] for row in rows} == {0, 1}
    assert executor.diagnostics() == {"active": False}


def test_stale_generations_fail_before_any_hybrid_engine_write() -> None:
    executor, _resources, robot, _other, port, parameters, _runtime = _fixture()
    _tare_result, step = _tare(executor)
    port.external[2] = 2.0
    before = len(robot.execution.joint_controller.writes)

    with pytest.raises(HybridTareStaleError):
        executor.execute(
            _motion(tare_generation=0),
            start_step=step,
            should_cancel=lambda: False,
        )
    assert len(robot.execution.joint_controller.writes) == before

    parameters.set_parameters({"posture_stiffness": 4.0})
    with pytest.raises(HybridParameterGenerationError):
        executor.execute(
            _motion(parameter_generation=0),
            start_step=step,
            should_cancel=lambda: False,
        )
    assert len(robot.execution.joint_controller.writes) == before


def test_cancel_and_physics_failure_both_restore_position_runtime() -> None:
    executor, resources, robot, _other, port, _parameters, _runtime = _fixture()
    _result, step = _tare(executor)
    port.external[2] = 2.0
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 6

    with pytest.raises(HybridCancelledError):
        executor.execute(_motion(), start_step=step, should_cancel=cancelled)
    assert robot.execution.joint_controller.settings.arm.mode == "position"
    assert np.allclose(robot.execution.joint_controller._targets.efforts, 0.0)

    executor, resources, robot, _other, port, _parameters, _runtime = _fixture()
    _result, step = _tare(executor)
    port.external[2] = 2.0
    resources.physics.fail_at = resources.physics.steps
    with pytest.raises(HybridExecutionError, match="physics injected"):
        executor.execute(_motion(), start_step=step, should_cancel=lambda: False)
    assert robot.execution.joint_controller.settings.arm.mode == "position"
    assert np.allclose(robot.execution.joint_controller._targets.efforts, 0.0)


def test_restore_failure_marks_runtime_fatal_and_requests_quit() -> None:
    executor, _resources, robot, _other, port, _parameters, runtime = _fixture()
    _result, step = _tare(executor)
    port.external[2] = 2.0
    robot.execution.joint_controller.fail_restore = True

    with pytest.raises(HybridRestoreFailedError):
        executor.execute(_motion(), start_step=step, should_cancel=lambda: False)

    assert runtime.fatal_error is not None
    assert "restore injected failure" in runtime.fatal_error
    assert runtime.quit_requested is True


@pytest.mark.parametrize(
    "force_axes",
    [
        [False] * 6,
        [True] * 6,
        [False, False, 1, False, False, False],
    ],
)
def test_hybrid_parser_rejects_invalid_axis_selection(force_axes) -> None:
    arguments = _motion()
    arguments["force_axes"] = force_axes

    with pytest.raises(ValueError, match="force_axes"):
        parse_hybrid_motion_request(arguments)
