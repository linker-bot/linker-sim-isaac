"""Owner-thread tare and hybrid force/position execution transactions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from linkerbot_sim.configuration.control import HybridForcePositionSettings
from linkerbot_sim.controllers.hybrid_force_position import (
    HybridControlError,
    HybridControlOutput,
    HybridControlParameters,
    HybridControlTarget,
    HybridForcePositionController,
    TaskSpaceObservation,
)
from linkerbot_sim.controllers.projection import (
    hybrid_force_position_settings,
    joint_control_settings,
)
from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.isaac.physics.physx_task_space import PhysxTaskSpaceError
from linkerbot_sim.mirror.hybrid_parameters import (
    HybridNotConfiguredError,
    HybridParameterSnapshot,
)
from linkerbot_sim.snapshots.transactions import (
    mark_runtime_fatal,
    require_runtime_mutable,
)
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


class HybridExecutionError(RuntimeError):
    """Stable wire-facing failure carrying the authoritative committed step."""

    code = "hybrid_control_failed"

    def __init__(
        self,
        message: str,
        *,
        step: int,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.step = int(step)
        self.details = dict(details or {})


class HybridBackendUnsupportedError(HybridExecutionError):
    code = "hybrid_backend_unsupported"


class HybridControlModeIncompatibleError(HybridExecutionError):
    code = "hybrid_control_mode_incompatible"


class HybridFrequencyUnsupportedError(HybridExecutionError):
    code = "hybrid_frequency_unsupported"


class HybridRobotUnsupportedError(HybridExecutionError):
    code = "hybrid_robot_unsupported"


class HybridTcpUnsupportedError(HybridExecutionError):
    code = "hybrid_tcp_unsupported"


class HybridTaskSpaceUnavailableError(HybridExecutionError):
    code = "hybrid_task_space_unavailable"


class HybridTareRequiredError(HybridExecutionError):
    code = "hybrid_tare_required"


class HybridTareStaleError(HybridExecutionError):
    code = "hybrid_tare_stale"


class HybridParameterGenerationError(HybridExecutionError):
    code = "hybrid_parameter_generation_conflict"


class HybridContactNotFoundError(HybridExecutionError):
    code = "hybrid_contact_not_found"


class HybridSensorInvalidError(HybridExecutionError):
    code = "hybrid_sensor_invalid"


class HybridCancelledError(HybridExecutionError):
    code = "cancelled"


class HybridRestoreFailedError(HybridExecutionError):
    code = "hybrid_restore_failed"


@dataclass(frozen=True, slots=True)
class TareRequest:
    robot_id: int
    robot_label: str | None
    tcp_frame_name: str
    reference_frame: str


@dataclass(frozen=True, slots=True)
class HybridMotionRequest:
    robot_id: int
    robot_label: str | None
    duration_s: float
    tcp_frame_name: str
    reference_frame: str
    target_position: tuple[float, float, float]
    target_orientation_wxyz: tuple[float, float, float, float]
    force_axes: tuple[bool, ...]
    target_wrench: tuple[float, ...]
    tare_generation: int
    hybrid_parameter_generation: int
    phase: str


@dataclass(frozen=True, slots=True)
class _TareRecord:
    robot_id: int
    tcp_frame_name: str
    reference_frame: str
    generation: int
    offset_environment_on_tool: tuple[float, ...]


@dataclass(slots=True)
class _PreparedRobot:
    robot: object
    articulation: object
    controller: object
    action_type: object
    port: object
    arm_names: tuple[str, ...]
    arm_command_slots: np.ndarray
    arm_dof_indices: np.ndarray
    original_runtime: object
    hybrid_runtime: object
    original_targets: ControlTargets | None
    baseline_targets: ControlTargets


@dataclass(slots=True)
class _OnlineStatistics:
    count: int
    force_error_sum: np.ndarray
    peak_abs_wrench: np.ndarray
    peak_abs_joint_effort: np.ndarray
    final_output: HybridControlOutput | None
    contacted: np.ndarray

    @classmethod
    def create(cls, joint_count: int) -> "_OnlineStatistics":
        return cls(
            count=0,
            force_error_sum=np.zeros(6, dtype=float),
            peak_abs_wrench=np.zeros(6, dtype=float),
            peak_abs_joint_effort=np.zeros(joint_count, dtype=float),
            final_output=None,
            contacted=np.zeros(6, dtype=bool),
        )

    def add(self, output: HybridControlOutput) -> None:
        self.count += 1
        self.force_error_sum += output.force_error
        self.peak_abs_wrench = np.maximum(
            self.peak_abs_wrench,
            np.abs(output.measured_wrench_tool_on_environment),
        )
        self.peak_abs_joint_effort = np.maximum(
            self.peak_abs_joint_effort,
            np.abs(output.joint_efforts),
        )
        self.contacted |= output.contact_axes
        self.final_output = output


class MirrorHybridExecutor:
    """Execute tare and explicit task-space control on the shared physics clock."""

    def __init__(
        self,
        resources: object,
        *,
        settings: HybridForcePositionSettings | None,
        physics_engine: str,
        physics_execution: str,
    ) -> None:
        self._resources = resources
        self._settings = settings
        self._physics_engine = str(physics_engine)
        self._physics_execution = str(physics_execution)
        self._render_frame: Callable[[], object] | None = None
        self._before_step: Callable[[float], None] | None = None
        self._control_mode_provider: Callable[[], str] = lambda: "position"
        self._parameter_provider: Callable[[], HybridParameterSnapshot] | None = None
        self._runtime_owner: object | None = None
        self._tare_by_robot: dict[int, _TareRecord] = {}
        self._next_tare_generation = 1
        self._latest_diagnostics: dict[str, object] = {"active": False}
        self._closed = False

    def bind_render_frame(self, callback: Callable[[], object]) -> None:
        self._render_frame = callback

    def bind_before_step(self, callback: Callable[[float], None]) -> None:
        self._before_step = callback

    def bind_control_mode_provider(self, callback: Callable[[], str]) -> None:
        self._control_mode_provider = callback

    def bind_parameter_provider(
        self, callback: Callable[[], HybridParameterSnapshot]
    ) -> None:
        self._parameter_provider = callback

    def bind_runtime_owner(self, runtime: object) -> None:
        if self._runtime_owner is not None:
            raise RuntimeError("hybrid executor runtime owner is already bound")
        self._runtime_owner = runtime

    def tare_wrench(
        self,
        arguments: Mapping[str, object],
        *,
        start_step: int,
        should_cancel: Callable[[], bool],
    ) -> tuple[dict[str, object], int]:
        step = int(start_step)
        self._set_inactive_diagnostics()
        request = parse_tare_request(arguments)
        robot, port = self._preflight_robot(
            request.robot_id,
            request.robot_label,
            request.tcp_frame_name,
            request.reference_frame,
            step=step,
        )
        self._require_position_mode(step=step)
        owner = self._runtime_owner or self._resources
        require_runtime_mutable(owner, operation="Mirror wrench tare")
        controller = robot.execution.joint_controller
        if any(mode != "position" for mode in controller.command_target_modes):
            raise HybridControlModeIncompatibleError(
                "tare requires every command joint to remain in position mode",
                step=step,
            )
        actual = _full_joint_vector(robot.articulation, "positions")
        zeros = np.zeros(len(controller.command_joint_names), dtype=float)
        hold = controller.build_control_targets(
            command_positions=actual[np.asarray(controller.command_indices, dtype=int)],
            command_velocities=zeros,
            command_efforts=zeros,
            base_positions=actual,
        )
        # Every check above is read-only. This hold is the first engine write.
        controller.apply_targets(robot.execution.articulation_action_type, hold)

        settings = self._require_settings(step=step)
        samples: list[np.ndarray] = []
        total_ticks = int(settings.tare.warmup_ticks + settings.tare.sample_count)
        for tick in range(total_ticks):
            if should_cancel():
                raise HybridCancelledError("wrench tare was cancelled", step=step)
            controller.apply_targets(robot.execution.articulation_action_type, hold)
            step = self._advance(step, phase="hybrid_tare")
            if should_cancel():
                raise HybridCancelledError("wrench tare was cancelled", step=step)
            observation = self._observe_port(port, step=step)
            speed = float(np.max(np.abs(observation.joint_velocities)))
            if speed > float(settings.tare.maximum_joint_speed):
                raise HybridSensorInvalidError(
                    "robot moved during wrench tare",
                    step=step,
                    details={
                        "limit": "tare.maximum_joint_speed",
                        "observed": speed,
                    },
                )
            if tick >= int(settings.tare.warmup_ticks):
                samples.append(
                    np.asarray(
                        observation.external_wrench_environment_on_tool,
                        dtype=float,
                    ).copy()
                )

        values = np.asarray(samples, dtype=float)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        maximum_std = np.asarray(settings.tare.maximum_std_wrench, dtype=float)
        if np.any(std > maximum_std):
            raise HybridSensorInvalidError(
                "wrench tare variance exceeds configured limits",
                step=step,
                details={
                    "limit": "tare.maximum_std_wrench",
                    "observed": std.tolist(),
                },
            )
        generation = self._next_tare_generation
        record = _TareRecord(
            robot_id=request.robot_id,
            tcp_frame_name=request.tcp_frame_name,
            reference_frame=request.reference_frame,
            generation=generation,
            offset_environment_on_tool=tuple(float(item) for item in mean),
        )
        self._tare_by_robot[request.robot_id] = record
        self._next_tare_generation += 1
        return (
            {
                "event": "wrench_tared",
                "robot_id": request.robot_id,
                "robot_label": str(robot.label),
                "tcp_frame_name": request.tcp_frame_name,
                "reference_frame": request.reference_frame,
                "tare_generation": generation,
                "sample_count": len(samples),
                "mean_wrench": mean.tolist(),
                "std_wrench": std.tolist(),
            },
            step,
        )

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        start_step: int,
        should_cancel: Callable[[], bool],
        request_id: str = "hybrid",
    ) -> tuple[dict[str, object], int]:
        step = int(start_step)
        self._set_inactive_diagnostics()
        request = parse_hybrid_motion_request(arguments)
        settings = self._require_settings(step=step)
        robot, port = self._preflight_robot(
            request.robot_id,
            request.robot_label,
            request.tcp_frame_name,
            request.reference_frame,
            step=step,
        )
        self._require_position_mode(step=step)
        parameter_snapshot = self._parameter_snapshot(step=step)
        if parameter_snapshot.generation != request.hybrid_parameter_generation:
            raise HybridParameterGenerationError(
                "hybrid parameter generation is stale",
                step=step,
                details={
                    "expected": request.hybrid_parameter_generation,
                    "actual": parameter_snapshot.generation,
                },
            )
        tare = self._tare_by_robot.get(request.robot_id)
        if tare is None:
            raise HybridTareRequiredError(
                "hybrid motion requires a successful wrench tare",
                step=step,
                details={"robot_id": request.robot_id},
            )
        if (
            tare.generation != request.tare_generation
            or tare.tcp_frame_name != request.tcp_frame_name
            or tare.reference_frame != request.reference_frame
        ):
            raise HybridTareStaleError(
                "hybrid motion references a stale wrench tare",
                step=step,
                details={
                    "expected": request.tare_generation,
                    "actual": tare.generation,
                },
            )

        dt = self._physics_dt(step=step)
        duration_ticks = max(1, int(math.ceil(request.duration_s / dt - 1.0e-12)))
        if request.duration_s > float(settings.max_duration_s):
            raise ValueError(
                f"duration_s exceeds configured maximum {settings.max_duration_s}"
            )
        prepared = self._prepare_robot(robot, port, step=step)
        initial_observation = self._observe_port(port, step=step)
        parameters = HybridControlParameters(**parameter_snapshot.values.as_dict())
        target = HybridControlTarget(
            position=np.asarray(request.target_position, dtype=float),
            orientation_wxyz=np.asarray(request.target_orientation_wxyz, dtype=float),
            force_axes=np.asarray(request.force_axes, dtype=bool),
            wrench_tool_on_environment=np.asarray(request.target_wrench, dtype=float),
        )
        effort_limits = np.asarray(
            prepared.hybrid_runtime.active_effort_limits, dtype=float
        )[prepared.arm_dof_indices]
        control = HybridForcePositionController(
            settings=settings,
            parameters=parameters,
            target=target,
            tare_external_wrench=np.asarray(
                tare.offset_environment_on_tool, dtype=float
            ),
            nominal_joint_positions=initial_observation.joint_positions,
            initial_position=initial_observation.position,
            initial_orientation_wxyz=initial_observation.orientation_wxyz,
            joint_effort_limits=effort_limits,
        )
        # A complete first calculation proves shape, gain, limit and Jacobian
        # validity while preflight is still engine-write free.
        try:
            first_output = control.step(initial_observation, dt=dt)
        except HybridControlError as exc:
            raise _control_failure(exc, step=step) from exc
        statistics = _OnlineStatistics.create(len(prepared.arm_names))
        original_error: BaseException | None = None
        override_started = False
        control_ticks = 0
        try:
            if should_cancel():
                raise HybridCancelledError(
                    "hybrid motion was cancelled before controller override",
                    step=step,
                )
            # First engine write: neutralize position targets at the current arm
            # pose, then switch only this robot's arm to direct effort.
            neutral = self._targets_for_effort(
                prepared,
                np.zeros(len(prepared.arm_names), dtype=float),
            )
            prepared.controller.apply_targets(prepared.action_type, neutral)
            override_started = True
            prepared.controller.apply_prepared_runtime(
                prepared.hybrid_runtime,
                clear_target_cache=False,
            )
            prepared.controller.apply_targets(prepared.action_type, neutral)

            output = first_output
            observation = initial_observation
            for tick in range(duration_ticks):
                if should_cancel():
                    raise HybridCancelledError("hybrid motion was cancelled", step=step)
                if tick > 0:
                    observation = self._observe_port(port, step=step)
                    try:
                        output = control.step(observation, dt=dt)
                    except HybridControlError as exc:
                        raise _control_failure(exc, step=step) from exc
                targets = self._targets_for_effort(
                    prepared,
                    output.joint_efforts,
                )
                if should_cancel():
                    raise HybridCancelledError("hybrid motion was cancelled", step=step)
                prepared.controller.apply_targets(prepared.action_type, targets)
                self._emit_diagnostics(
                    request_id=request_id,
                    request=request,
                    robot=robot,
                    observation=observation,
                    output=output,
                    tare_generation=tare.generation,
                    parameter_generation=parameter_snapshot.generation,
                    step=step,
                    tick=tick,
                    dt=dt,
                )
                step = self._advance(step, phase=request.phase)
                control_ticks += 1
                statistics.add(output)
                if should_cancel():
                    raise HybridCancelledError("hybrid motion was cancelled", step=step)
        except BaseException as exc:
            original_error = exc
        finally:
            if override_started:
                try:
                    step, cleanup_error = self._cleanup_override(
                        prepared,
                        last_efforts=control.last_joint_efforts,
                        step=step,
                        phase=request.phase,
                    )
                    if cleanup_error is not None and original_error is None:
                        original_error = cleanup_error
                except BaseException as restore_exc:
                    owner = self._runtime_owner or self._resources
                    mark_runtime_fatal(
                        owner,
                        operation="Mirror hybrid controller restore",
                        cause=restore_exc,
                    )
                    self._set_inactive_diagnostics()
                    raise HybridRestoreFailedError(
                        f"failed to restore position control: {restore_exc}",
                        step=step,
                        details={"robot_id": request.robot_id, "fatal": True},
                    ) from restore_exc
            self._set_inactive_diagnostics()

        if original_error is not None:
            if isinstance(original_error, HybridExecutionError):
                original_error.step = step
                raise original_error
            if isinstance(original_error, PhysxTaskSpaceError):
                raise HybridSensorInvalidError(
                    str(original_error), step=step
                ) from original_error
            raise HybridExecutionError(
                str(original_error),
                step=step,
                details={"exception_type": type(original_error).__name__},
            ) from original_error

        force_axes = np.asarray(request.force_axes, dtype=bool)
        if not np.all(statistics.contacted[force_axes]):
            raise HybridContactNotFoundError(
                "contact was not established on every force-controlled axis",
                step=step,
                details={
                    "contact_axes": statistics.contacted.tolist(),
                    "force_axes": force_axes.tolist(),
                },
            )
        final = statistics.final_output
        if final is None or statistics.count == 0:
            raise HybridExecutionError(
                "hybrid motion completed without a control sample", step=step
            )
        mean_error = statistics.force_error_sum / statistics.count
        mean_error[~force_axes] = 0.0
        return (
            {
                "event": "hybrid_force_position_completed",
                "robot_id": request.robot_id,
                "robot_label": str(robot.label),
                "phase": request.phase,
                "executed_ticks": control_ticks,
                "duration_s": control_ticks * dt,
                "tare_generation": tare.generation,
                "hybrid_parameter_generation": parameter_snapshot.generation,
                "force_axes": list(request.force_axes),
                "final_pose_error": final.pose_error.tolist(),
                "final_wrench": final.measured_wrench_tool_on_environment.tolist(),
                "mean_force_axis_error": mean_error.tolist(),
                "peak_abs_wrench": statistics.peak_abs_wrench.tolist(),
                "peak_abs_joint_effort": statistics.peak_abs_joint_effort.tolist(),
            },
            step,
        )

    def invalidate_tare(self) -> None:
        self._tare_by_robot.clear()

    def status(self) -> dict[str, object]:
        parameter_generation = None
        if self._parameter_provider is not None and self._settings is not None:
            parameter_generation = self._parameter_provider().generation
        return {
            "configured": self._settings is not None,
            "supported": (
                self._settings is not None
                and self._physics_engine == "physx"
                and self._physics_execution == "cpu"
            ),
            "active": bool(self._latest_diagnostics.get("active", False)),
            "parameter_generation": parameter_generation,
            "tare": [
                {
                    "robot_id": record.robot_id,
                    "tcp_frame_name": record.tcp_frame_name,
                    "reference_frame": record.reference_frame,
                    "tare_generation": record.generation,
                }
                for record in sorted(
                    self._tare_by_robot.values(), key=lambda item: item.robot_id
                )
            ],
        }

    def diagnostics(self) -> dict[str, object]:
        """Return the latest frozen owner-thread sample without physics access."""

        return deepcopy(self._latest_diagnostics)

    def close(self) -> bool:
        self.invalidate_tare()
        self._closed = True
        return True

    def _prepare_robot(
        self, robot: object, port: object, *, step: int
    ) -> _PreparedRobot:
        controller = robot.execution.joint_controller
        profiles = robot.controller_profiles
        if profiles is None:
            raise HybridRobotUnsupportedError(
                "robot has no controller profile binding", step=step
            )
        expected_position = joint_control_settings(profiles, mode="position")
        if controller.settings != expected_position or any(
            mode != "position" for mode in controller.command_target_modes
        ):
            raise HybridControlModeIncompatibleError(
                "robot controller state does not match logical position mode",
                step=step,
            )
        arm_names = tuple(str(name) for name in robot.joint_groups.arm)
        command_names = tuple(str(name) for name in controller.command_joint_names)
        command_by_name = {name: index for index, name in enumerate(command_names)}
        if any(name not in command_by_name for name in arm_names):
            raise HybridRobotUnsupportedError(
                "arm group is not fully represented in command joints", step=step
            )
        arm_slots = np.asarray([command_by_name[name] for name in arm_names], dtype=int)
        command_indices = np.asarray(controller.command_indices, dtype=int)
        arm_dof_indices = command_indices[arm_slots]
        actual = _full_joint_vector(robot.articulation, "positions")
        original_targets = controller.snapshot_control_targets_cache()
        if original_targets is None:
            zeros = np.zeros(len(command_names), dtype=float)
            baseline = controller.build_control_targets(
                command_positions=actual[command_indices],
                command_velocities=zeros,
                command_efforts=zeros,
                base_positions=actual,
            )
        else:
            baseline = original_targets
        original_runtime = controller.prepare_runtime()
        hybrid_runtime = controller.prepare_runtime(
            hybrid_force_position_settings(profiles)
        )
        return _PreparedRobot(
            robot=robot,
            articulation=robot.articulation,
            controller=controller,
            action_type=robot.execution.articulation_action_type,
            port=port,
            arm_names=arm_names,
            arm_command_slots=arm_slots,
            arm_dof_indices=arm_dof_indices,
            original_runtime=original_runtime,
            hybrid_runtime=hybrid_runtime,
            original_targets=original_targets,
            baseline_targets=baseline,
        )

    def _emit_diagnostics(
        self,
        *,
        request_id: str,
        request: HybridMotionRequest,
        robot: object,
        observation: TaskSpaceObservation,
        output: HybridControlOutput,
        tare_generation: int,
        parameter_generation: int,
        step: int,
        tick: int,
        dt: float,
    ) -> None:
        payload: dict[str, object] = {
            "active": True,
            "request_id": request_id,
            "robot_id": request.robot_id,
            "robot_label": str(robot.label),
            "step": int(step),
            "tick": int(tick),
            "time_s": (int(step) + 1) * float(dt),
            "phase": request.phase,
            "tare_generation": int(tare_generation),
            "hybrid_parameter_generation": int(parameter_generation),
            "force_axes": list(request.force_axes),
            "target_position": list(request.target_position),
            "target_orientation_wxyz": list(request.target_orientation_wxyz),
            "actual_position": observation.position.tolist(),
            "actual_orientation_wxyz": observation.orientation_wxyz.tolist(),
            "target_wrench_tool_on_environment": list(request.target_wrench),
            "raw_wrench_environment_on_tool": (
                observation.external_wrench_environment_on_tool.tolist()
            ),
            "filtered_wrench_tool_on_environment": (
                output.measured_wrench_tool_on_environment.tolist()
            ),
            "motion_wrench": output.motion_wrench.tolist(),
            "force_wrench": output.force_wrench.tolist(),
            "commanded_arm_effort": output.joint_efforts.tolist(),
            "contact_axes": output.contact_axes.tolist(),
            "wrench_saturated_axes": output.wrench_saturated_axes.tolist(),
            "effort_saturated_axes": output.effort_saturated_axes.tolist(),
            "minimum_singular_value": output.minimum_singular_value,
            "condition_number": output.condition_number,
        }
        self._latest_diagnostics = payload
        logger = getattr(self._resources, "hybrid_control_logger", None)
        write = getattr(logger, "write", None)
        if callable(write):
            write(payload)

    def _set_inactive_diagnostics(self) -> None:
        self._latest_diagnostics = {"active": False}

    def _targets_for_effort(
        self, prepared: _PreparedRobot, efforts: object
    ) -> ControlTargets:
        arm_efforts = np.asarray(efforts, dtype=float).reshape(-1)
        if arm_efforts.shape != (len(prepared.arm_names),) or not np.all(
            np.isfinite(arm_efforts)
        ):
            raise ValueError("arm efforts have an invalid shape or value")
        positions = prepared.baseline_targets.positions.copy()
        velocities = prepared.baseline_targets.velocities.copy()
        full_efforts = np.zeros_like(prepared.baseline_targets.efforts)
        actual = _full_joint_vector(prepared.articulation, "positions")
        positions[prepared.arm_dof_indices] = actual[prepared.arm_dof_indices]
        velocities[prepared.arm_dof_indices] = 0.0
        full_efforts[prepared.arm_dof_indices] = arm_efforts
        return prepared.controller.targets_from_full_state(
            positions,
            velocities,
            full_efforts,
        )

    def _cleanup_override(
        self,
        prepared: _PreparedRobot,
        *,
        last_efforts: np.ndarray,
        step: int,
        phase: str,
    ) -> tuple[int, BaseException | None]:
        settings = self._require_settings(step=step)
        current = np.asarray(last_efforts, dtype=float).reshape(-1)
        ramp_ticks = int(settings.limits.ramp_down_ticks)
        ramp_error: BaseException | None = None
        for remaining in range(ramp_ticks - 1, -1, -1):
            efforts = current * (remaining / ramp_ticks)
            targets = self._targets_for_effort(prepared, efforts)
            prepared.controller.apply_targets(prepared.action_type, targets)
            try:
                step = self._advance(step, phase=f"{phase}:ramp_down")
            except BaseException as exc:
                if isinstance(exc, HybridExecutionError):
                    step = exc.step
                ramp_error = exc
                break
        # Even when physics/post-step failed, make one explicit zero-effort
        # write before restoring drives.
        zero = self._targets_for_effort(
            prepared, np.zeros(len(prepared.arm_names), dtype=float)
        )
        prepared.controller.apply_targets(prepared.action_type, zero)
        final_positions = _full_joint_vector(prepared.articulation, "positions")
        prepared.controller.apply_prepared_runtime(
            prepared.original_runtime,
            clear_target_cache=False,
        )
        handover_positions = prepared.baseline_targets.positions.copy()
        handover_velocities = prepared.baseline_targets.velocities.copy()
        handover_positions[prepared.arm_dof_indices] = final_positions[
            prepared.arm_dof_indices
        ]
        handover_velocities[prepared.arm_dof_indices] = 0.0
        handover = prepared.controller.targets_from_full_state(
            handover_positions,
            handover_velocities,
            np.zeros_like(prepared.baseline_targets.efforts),
        )
        prepared.controller.apply_targets(prepared.action_type, handover)
        return step, ramp_error

    def _preflight_robot(
        self,
        robot_id: int,
        robot_label: str | None,
        tcp_frame_name: str,
        reference_frame: str,
        *,
        step: int,
    ) -> tuple[object, object]:
        if self._closed:
            raise HybridExecutionError("hybrid executor is closed", step=step)
        if self._physics_engine != "physx" or self._physics_execution != "cpu":
            raise HybridBackendUnsupportedError(
                "hybrid force/position control only supports PhysX CPU",
                step=step,
            )
        self._require_settings(step=step)
        try:
            resolver = getattr(self._resources, "robot", None)
            robot = (
                resolver(robot_id)
                if callable(resolver)
                else self._resources.robots_by_id[robot_id]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HybridRobotUnsupportedError(
                f"unknown hybrid robot_id {robot_id}", step=step
            ) from exc
        if robot_label is not None and str(robot.label) != robot_label:
            raise HybridRobotUnsupportedError(
                f"robot_id {robot_id} is {robot.label!r}, not {robot_label!r}",
                step=step,
            )
        binding = getattr(robot, "physical_tcp_binding", None)
        if binding is None or binding.tcp_frame_name != tcp_frame_name:
            raise HybridTcpUnsupportedError(
                f"robot {robot_id} does not bind physical TCP {tcp_frame_name!r}",
                step=step,
            )
        if reference_frame != "world":
            raise HybridTcpUnsupportedError(
                "first-phase hybrid control only supports reference_frame='world'",
                step=step,
            )
        port = getattr(robot, "task_space_port", None)
        if port is None or not callable(getattr(port, "observe", None)):
            raise HybridTaskSpaceUnavailableError(
                f"robot {robot_id} has no PhysX task-space port", step=step
            )
        self._physics_dt(step=step)
        return robot, port

    def _require_position_mode(self, *, step: int) -> None:
        if str(self._control_mode_provider()) != "position":
            raise HybridControlModeIncompatibleError(
                "hybrid control requires global active mode 'position'", step=step
            )

    def _parameter_snapshot(self, *, step: int) -> HybridParameterSnapshot:
        if self._parameter_provider is None:
            raise HybridNotConfiguredError("hybrid parameter provider is not bound")
        return self._parameter_provider()

    def _require_settings(self, *, step: int) -> HybridForcePositionSettings:
        if self._settings is None:
            raise HybridNotConfiguredError(
                "hybrid force/position control is not configured"
            )
        return self._settings

    def _physics_dt(self, *, step: int) -> float:
        physics = getattr(self._resources, "physics", None)
        if physics is None:
            physics = getattr(self._resources, "simulation_world", None)
        getter = None if physics is None else getattr(physics, "get_physics_dt", None)
        if not callable(getter):
            raise HybridFrequencyUnsupportedError(
                "hybrid runtime does not expose physics dt", step=step
            )
        dt = float(getter())
        if not math.isfinite(dt) or dt <= 0.0:
            raise HybridFrequencyUnsupportedError(
                "hybrid runtime returned invalid physics dt", step=step
            )
        settings = self._require_settings(step=step)
        frequency = 1.0 / dt
        if frequency + 1.0e-9 < float(settings.minimum_physics_frequency_hz):
            raise HybridFrequencyUnsupportedError(
                "hybrid physics frequency is below the configured minimum",
                step=step,
                details={
                    "actual_hz": frequency,
                    "minimum_hz": settings.minimum_physics_frequency_hz,
                },
            )
        return dt

    def _observe_port(self, port: object, *, step: int) -> TaskSpaceObservation:
        try:
            observation = port.observe()
        except PhysxTaskSpaceError as exc:
            raise HybridSensorInvalidError(str(exc), step=step) from exc
        if not isinstance(observation, TaskSpaceObservation):
            raise HybridSensorInvalidError(
                "task-space port returned an invalid observation type", step=step
            )
        return observation

    def _advance(self, step: int, *, phase: str) -> int:
        dt = self._physics_dt(step=step)
        physics = getattr(self._resources, "physics", None)
        if physics is None:
            physics = getattr(self._resources, "simulation_world", None)
        if self._before_step is not None:
            self._before_step(dt)
        physics.step(render=False)
        committed = int(step) + 1
        claim = getattr(self._resources, "claim_completed_step", None)
        sample_step = int(claim()) if callable(claim) else int(step)
        try:
            collision = getattr(self._resources, "collision_registry", None)
            mark_dirty = getattr(collision, "mark_dirty", None)
            if callable(mark_dirty):
                mark_dirty()
            if self._render_frame is not None:
                self._render_frame()
            observe = getattr(self._resources, "observe_after_step", None)
            if callable(observe):
                observe(
                    step=sample_step,
                    phase=phase,
                    write_idle_logs=False,
                )
        except BaseException as exc:
            raise HybridExecutionError(
                f"hybrid post-step processing failed: {exc}",
                step=committed,
                details={"exception_type": type(exc).__name__},
            ) from exc
        return committed


def parse_tare_request(arguments: Mapping[str, object]) -> TareRequest:
    required = {"robot_id", "tcp_frame_name", "reference_frame"}
    optional = {"robot_label"}
    _exact_fields(
        arguments, required=required, optional=optional, label="control.tare_wrench"
    )
    return TareRequest(
        robot_id=_non_negative_int(arguments["robot_id"], label="robot_id"),
        robot_label=_optional_string(arguments.get("robot_label"), label="robot_label"),
        tcp_frame_name=_string(arguments["tcp_frame_name"], label="tcp_frame_name"),
        reference_frame=_string(arguments["reference_frame"], label="reference_frame"),
    )


def parse_hybrid_motion_request(
    arguments: Mapping[str, object],
) -> HybridMotionRequest:
    required = {
        "robot_id",
        "duration_s",
        "tcp_frame_name",
        "reference_frame",
        "target_position",
        "target_orientation_wxyz",
        "force_axes",
        "target_wrench",
        "tare_generation",
        "hybrid_parameter_generation",
    }
    optional = {"robot_label", "phase"}
    _exact_fields(
        arguments,
        required=required,
        optional=optional,
        label="motion.hybrid_force_position",
    )
    force_axes = _bool_vector(arguments["force_axes"], label="force_axes")
    if not any(force_axes) or all(force_axes):
        raise ValueError("force_axes must select at least one but not all axes")
    target_wrench = _number_vector(arguments["target_wrench"], 6, label="target_wrench")
    if any(
        abs(value) > 1.0e-12
        for value, is_force in zip(target_wrench, force_axes, strict=True)
        if not is_force
    ):
        raise ValueError("target_wrench must be zero on motion-controlled axes")
    orientation = _number_vector(
        arguments["target_orientation_wxyz"],
        4,
        label="target_orientation_wxyz",
    )
    norm = math.sqrt(sum(item * item for item in orientation))
    if norm <= 1.0e-12:
        raise ValueError("target_orientation_wxyz cannot be zero")
    orientation = tuple(item / norm for item in orientation)
    duration = _number(arguments["duration_s"], label="duration_s")
    if duration <= 0.0:
        raise ValueError("duration_s must be positive")
    return HybridMotionRequest(
        robot_id=_non_negative_int(arguments["robot_id"], label="robot_id"),
        robot_label=_optional_string(arguments.get("robot_label"), label="robot_label"),
        duration_s=duration,
        tcp_frame_name=_string(arguments["tcp_frame_name"], label="tcp_frame_name"),
        reference_frame=_string(arguments["reference_frame"], label="reference_frame"),
        target_position=_number_vector(
            arguments["target_position"], 3, label="target_position"
        ),  # type: ignore[arg-type]
        target_orientation_wxyz=orientation,  # type: ignore[arg-type]
        force_axes=force_axes,
        target_wrench=target_wrench,
        tare_generation=_non_negative_int(
            arguments["tare_generation"], label="tare_generation"
        ),
        hybrid_parameter_generation=_non_negative_int(
            arguments["hybrid_parameter_generation"],
            label="hybrid_parameter_generation",
        ),
        phase=_string(arguments.get("phase", "hybrid_force_position"), label="phase"),
    )


def _exact_fields(
    arguments: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(arguments))
    unknown = sorted(set(arguments) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite JSON number")
    return result


def _number_vector(value: object, length: int, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    return tuple(
        _number(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )


def _bool_vector(value: object, *, label: str) -> tuple[bool, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{label} must contain exactly six booleans")
    if any(type(item) is not bool for item in value):
        raise ValueError(f"{label} must contain only booleans")
    return tuple(value)


def _non_negative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative JSON integer")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty string without edge whitespace")
    return value


def _optional_string(value: object, *, label: str) -> str | None:
    return None if value is None else _string(value, label=label)


def _full_joint_vector(articulation: object, kind: str) -> np.ndarray:
    callback = getattr(articulation, f"get_joint_{kind}", None)
    if not callable(callback):
        raise RuntimeError(f"articulation is missing get_joint_{kind}()")
    value = tensor_like_to_numpy(callback(), dtype=float).reshape(-1)
    expected = int(getattr(articulation, "num_dof", value.size))
    if value.shape != (expected,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"articulation returned invalid joint {kind}")
    return value.copy()


def _control_failure(exc: HybridControlError, *, step: int) -> HybridExecutionError:
    error = HybridExecutionError(str(exc), step=step, details=dict(exc.details))
    error.code = exc.code
    return error


__all__ = [
    "HybridBackendUnsupportedError",
    "HybridCancelledError",
    "HybridContactNotFoundError",
    "HybridControlModeIncompatibleError",
    "HybridExecutionError",
    "HybridFrequencyUnsupportedError",
    "HybridMotionRequest",
    "HybridParameterGenerationError",
    "HybridRestoreFailedError",
    "HybridRobotUnsupportedError",
    "HybridSensorInvalidError",
    "HybridTareRequiredError",
    "HybridTareStaleError",
    "HybridTaskSpaceUnavailableError",
    "HybridTcpUnsupportedError",
    "MirrorHybridExecutor",
    "TareRequest",
    "parse_hybrid_motion_request",
    "parse_tare_request",
]
