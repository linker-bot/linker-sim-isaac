"""Transactional all-robot control-mode switching for Kaleidoscope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.controllers.control_mode import (
    CONTROL_MODES,
    ControlModeChange,
    ControlModeGenerationConflict,
    ControlModeIncompatibleError,
    ControlModeRollbackError,
    ControlModeState,
    ControlModeSwitchError,
    require_control_mode,
    require_expected_generation,
)
from linkerbot_sim.controllers.projection import joint_control_settings
from linkerbot_sim.controllers.runtime_projection import project_command_runtime
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.kaleidoscope.control_commands import (
    ControlTrajectory,
    EffortControlTrajectory,
    PositionControlTrajectory,
    VelocityControlTrajectory,
)
from linkerbot_sim.kaleidoscope.tensors import assert_finite_async, require_cuda_tensor

if TYPE_CHECKING:
    from linkerbot_sim.kaleidoscope.resets import TBlockResetCommand


@dataclass(frozen=True, slots=True)
class KaleidoscopeControlBinding:
    """Construction-time profile and backend port binding for one robot."""

    label: str
    port: object
    controller_profiles: ControllerProfiles
    command_joint_names: tuple[str, ...]
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label.strip() or getattr(self.port, "label", None) != self.label:
            raise ValueError("Kaleidoscope control binding label must match its port")
        if not isinstance(self.controller_profiles, ControllerProfiles):
            raise TypeError("controller_profiles must be ControllerProfiles")
        width = int(getattr(self.port, "command_dim", 0))
        if (
            width < 1
            or len(self.command_joint_names) != width
            or len(self.components) != width
            or len(set(self.command_joint_names)) != width
        ):
            raise ValueError("Kaleidoscope control binding command metadata is invalid")
        for name in (
            "read_joint_positions",
            "prepare_control_runtime",
            "validate_prepared_control_runtime",
            "apply_prepared_control_runtime",
            "write_joint_position_targets",
            "write_joint_velocity_targets",
            "write_joint_effort_targets",
            "synchronize_control_writes",
        ):
            if not callable(getattr(self.port, name, None)):
                raise TypeError(f"Kaleidoscope control port must implement {name}()")


@dataclass(frozen=True, slots=True)
class _PreparedBinding:
    binding: KaleidoscopeControlBinding
    by_mode: dict[ControlMode, object]


@dataclass(frozen=True, slots=True)
class _SwitchPlan:
    prepared: _PreparedBinding
    current_q: torch.Tensor
    old_target: torch.Tensor


class KaleidoscopeControlModeCoordinator:
    """Own one logical mode and compensate device writes in reverse robot order."""

    def __init__(
        self,
        *,
        views: object,
        bindings: tuple[KaleidoscopeControlBinding, ...],
        supported_modes: tuple[ControlMode, ...],
    ) -> None:
        self.views = views
        self.num_envs = int(getattr(views, "num_envs"))
        if self.num_envs < 1:
            raise ValueError("Kaleidoscope control coordinator requires environments")
        raw_device = getattr(views, "device")
        # A binding-free coordinator is the position-only compatibility adapter used
        # by lightweight runtime tests and third-party views. Real control bindings
        # retain the strict CUDA contract below.
        self.device = torch.device(raw_device) if bindings else raw_device
        if bindings and self.device.type != "cuda":
            raise ValueError(
                "Kaleidoscope control coordinator requires CUDA environments"
            )
        normalized = tuple(
            require_control_mode(mode, label="supported control mode")
            for mode in supported_modes
        )
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("supported control modes must be non-empty and unique")
        if "position" not in normalized:
            raise ValueError("Kaleidoscope initial position mode must be supported")
        self._supported_modes = normalized
        self._initial_mode: ControlMode = "position"
        self._active_mode: ControlMode = "position"
        self._generation = 0
        self._runtime: object | None = None
        self._all_env_ids = (
            torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.int64,
            )
            if bindings
            else None
        )
        self._zero_by_width: dict[int, torch.Tensor] = {}

        labels = tuple(binding.label for binding in bindings)
        view_labels = tuple(
            str(getattr(port, "label")) for port in getattr(views, "robot_ports", ())
        )
        if bindings and labels != view_labels:
            raise ValueError("control bindings must follow the view robot order")
        if bindings:
            self._validate_canonical_buffers(
                command_dim=sum(
                    int(getattr(binding.port, "command_dim")) for binding in bindings
                )
            )
        prepared: list[_PreparedBinding] = []
        effort_limits: list[torch.Tensor] = []
        for binding in bindings:
            by_mode: dict[ControlMode, object] = {}
            for mode in CONTROL_MODES:
                projection = project_command_runtime(
                    joint_control_settings(binding.controller_profiles, mode=mode),
                    joint_names=binding.command_joint_names,
                    components=binding.components,
                )
                value = binding.port.prepare_control_runtime(projection)
                binding.port.validate_prepared_control_runtime(value)
                by_mode[mode] = value
                if mode == "effort":
                    effort_limits.append(value.common.effort_limits)
            width = len(binding.command_joint_names)
            self._zero_by_width[width] = torch.zeros(
                (self.num_envs, width),
                device=self.device,
                dtype=torch.float32,
            )
            prepared.append(_PreparedBinding(binding=binding, by_mode=by_mode))
        self._prepared = tuple(prepared)
        if effort_limits:
            self.command_effort_limits = torch.cat(effort_limits).contiguous()
        elif (
            isinstance(self.device, (str, torch.device))
            and torch.device(self.device).type == "cuda"
        ):
            # Legacy position-only assemblies still need to construct the canonical
            # joint action. Any real switch remains fail-closed at set_mode().
            self.command_effort_limits = torch.ones(
                int(getattr(views, "command_dim", 1)),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.command_effort_limits = None
        binder = getattr(views, "bind_control_mode_provider", None)
        if callable(binder):
            binder(lambda: self._active_mode)
        if self._prepared:
            self._initialize_position_mode()

    def bind_runtime(self, runtime: object) -> None:
        if self._runtime is not None:
            raise RuntimeError("Kaleidoscope control coordinator is already bound")
        self._runtime = runtime

    @property
    def active_mode(self) -> ControlMode:
        return self._active_mode

    def get_mode(self) -> ControlModeState:
        return ControlModeState(
            initial_mode=self._initial_mode,
            active_mode=self._active_mode,
            generation=self._generation,
            supported_modes=self._supported_modes,
        )

    def set_mode(
        self,
        mode: ControlMode,
        *,
        expected_generation: int | None = None,
    ) -> ControlModeChange:
        requested = require_control_mode(mode)
        if requested not in self._supported_modes:
            raise ControlModeIncompatibleError(
                f"configured Kaleidoscope action does not support {requested!r} mode",
                active_mode=self._active_mode,
                operation="set_control_mode",
            )
        if expected_generation is not None:
            expected = require_expected_generation(expected_generation)
            if expected != self._generation:
                raise ControlModeGenerationConflict(
                    expected=expected,
                    actual=self._generation,
                )
        previous = self._active_mode
        if requested == previous:
            return ControlModeChange(
                previous_mode=previous,
                active_mode=previous,
                generation=self._generation,
                changed=False,
            )
        if not self._prepared:
            raise ControlModeSwitchError(
                "Kaleidoscope assembly did not provide runtime control bindings"
            )

        try:
            plans = self._prepare_switch(previous)
        except BaseException as exc:
            raise ControlModeSwitchError(
                f"failed to prepare Kaleidoscope control mode "
                f"{previous}->{requested}: {exc}"
            ) from exc
        completed: list[_SwitchPlan] = []
        try:
            for plan in plans:
                completed.append(plan)
                self._write_neutral(plan, previous)
                port = plan.prepared.binding.port
                port.apply_prepared_control_runtime(plan.prepared.by_mode[requested])
                self._write_neutral(plan, requested)
            for plan in plans:
                plan.prepared.binding.port.synchronize_control_writes()
        except BaseException as forward_error:
            rollback_errors: list[BaseException] = []
            for plan in reversed(completed):
                try:
                    self._rollback(plan, previous)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                message = (
                    f"Kaleidoscope control-mode rollback failed after {forward_error}: "
                    f"{rollback_errors[0]}"
                )
                self._mark_runtime_fatal(message)
                raise ControlModeRollbackError(message) from rollback_errors[0]
            raise ControlModeSwitchError(
                f"failed to switch Kaleidoscope control mode "
                f"{previous}->{requested}: {forward_error}"
            ) from forward_error

        position_reference = torch.cat([plan.current_q for plan in plans], dim=1)
        target = (
            position_reference
            if requested == "position"
            else torch.zeros_like(position_reference)
        )
        self._commit_canonical(position_reference=position_reference, target=target)
        self._active_mode = requested
        self._generation += 1
        return ControlModeChange(
            previous_mode=previous,
            active_mode=requested,
            generation=self._generation,
            changed=True,
        )

    def dispatch(self, control: ControlTrajectory, tick: int) -> None:
        if isinstance(control, PositionControlTrajectory):
            if self._active_mode != "position":
                self._wrong_trajectory(control)
            writer = getattr(self.views, "write_position_targets", None)
            if callable(writer):
                writer(control.positions[tick], control.velocities[tick])
            else:
                self.views.write_joint_targets(control.positions[tick])
            return
        if isinstance(control, VelocityControlTrajectory):
            if self._active_mode != "velocity":
                self._wrong_trajectory(control)
            writer = getattr(self.views, "write_velocity_targets", None)
            if not callable(writer):
                raise RuntimeError("Kaleidoscope views do not expose velocity targets")
            writer(control.velocities[tick])
            return
        if isinstance(control, EffortControlTrajectory):
            if self._active_mode != "effort":
                self._wrong_trajectory(control)
            writer = getattr(self.views, "write_effort_targets", None)
            if not callable(writer):
                raise RuntimeError("Kaleidoscope views do not expose effort targets")
            writer(control.efforts[tick])
            return
        raise TypeError("unsupported Kaleidoscope control trajectory")

    def commit_position_reference(self, values: torch.Tensor) -> None:
        writer = getattr(self.views, "commit_position_reference", None)
        if callable(writer):
            writer(values)

    def write_reset(self, command: TBlockResetCommand) -> None:
        writer = getattr(self.views, "write_reset")
        if self._prepared or hasattr(self.views, "bind_control_mode_provider"):
            writer(command, control_mode=self._active_mode)
        else:
            writer(command)

    def _initialize_position_mode(self) -> None:
        current: list[torch.Tensor] = []
        for prepared in self._prepared:
            port = prepared.binding.port
            width = int(getattr(port, "command_dim"))
            current.append(
                self._validated_command_rows(
                    port.read_joint_positions(self._all_env_ids),
                    label=f"{prepared.binding.label} current joint positions",
                    width=width,
                ).clone()
            )
        torch.cuda.synchronize(self.device)
        for prepared, q in zip(self._prepared, current, strict=True):
            port = prepared.binding.port
            port.apply_prepared_control_runtime(prepared.by_mode["position"])
            port.write_joint_velocity_targets(
                self._all_env_ids,
                self._zero_by_width[q.shape[1]],
            )
            port.write_joint_position_targets(self._all_env_ids, q)
        for prepared in self._prepared:
            prepared.binding.port.synchronize_control_writes()
        position_reference = torch.cat(current, dim=1)
        self._commit_canonical(
            position_reference=position_reference,
            target=position_reference,
        )

    def _prepare_switch(self, previous: ControlMode) -> tuple[_SwitchPlan, ...]:
        control_targets = getattr(self.views, "control_targets")
        plans: list[_SwitchPlan] = []
        offset = 0
        for prepared in self._prepared:
            port = prepared.binding.port
            for value in prepared.by_mode.values():
                port.validate_prepared_control_runtime(value)
            width = int(getattr(port, "command_dim"))
            current_q = self._validated_command_rows(
                port.read_joint_positions(self._all_env_ids),
                label=f"{prepared.binding.label} current joint positions",
                width=width,
            )
            old_target = self._validated_command_rows(
                control_targets[:, offset : offset + width],
                label=f"{prepared.binding.label} active control target",
                width=width,
            )
            plans.append(
                _SwitchPlan(
                    prepared=prepared,
                    current_q=current_q.clone(),
                    old_target=old_target.clone(),
                )
            )
            offset += width
        if offset != int(getattr(self.views, "command_dim")):
            raise ValueError("control binding widths do not cover command targets")
        # Surface device-side finite assertions before the first engine mutation.
        torch.cuda.synchronize(self.device)
        return tuple(plans)

    def _validated_command_rows(
        self,
        value: object,
        *,
        label: str,
        width: int,
    ) -> torch.Tensor:
        rows = require_cuda_tensor(
            value,
            name=label,
            ndim=2,
            leading_dim=self.num_envs,
            dtype=torch.float32,
        )
        if rows.device != self.device or rows.shape[1:] != (width,):
            raise ValueError(
                f"{label} must have shape ({self.num_envs},{width}) on {self.device}"
            )
        assert_finite_async(rows, name=label)
        return rows

    def _validate_canonical_buffers(self, *, command_dim: int) -> None:
        expected_shape = (self.num_envs, command_dim)
        canonical: list[torch.Tensor] = []
        for name in ("position_references", "control_targets"):
            value = require_cuda_tensor(
                getattr(self.views, name, None),
                name=f"Kaleidoscope {name}",
                ndim=2,
                leading_dim=self.num_envs,
                dtype=torch.float32,
            )
            if value.device != self.device or value.shape != expected_shape:
                raise ValueError(
                    f"Kaleidoscope {name} must have shape {expected_shape} "
                    f"on {self.device}"
                )
            canonical.append(value)
        if (
            canonical[0].untyped_storage().data_ptr()
            == canonical[1].untyped_storage().data_ptr()
        ):
            raise ValueError(
                "Kaleidoscope position_references and control_targets must not alias"
            )
        if int(getattr(self.views, "command_dim")) != command_dim:
            raise ValueError(
                "control binding widths must match the view command dimension"
            )
        for name, value in zip(
            ("position_references", "control_targets"), canonical, strict=True
        ):
            assert_finite_async(value, name=f"Kaleidoscope {name}")
        # Construction is a cold boundary. Surface both device-side finite checks before
        # initialization applies the first controller profile or target to an engine.
        torch.cuda.synchronize(self.device)

    def _write_neutral(self, plan: _SwitchPlan, mode: ControlMode) -> None:
        target = (
            plan.current_q
            if mode == "position"
            else self._zero_by_width[plan.current_q.shape[1]]
        )
        self._write_port_target(plan.prepared.binding.port, mode, target)

    def _rollback(self, plan: _SwitchPlan, mode: ControlMode) -> None:
        port = plan.prepared.binding.port
        port.apply_prepared_control_runtime(plan.prepared.by_mode[mode])
        self._write_port_target(port, mode, plan.old_target)
        port.synchronize_control_writes()

    def _write_port_target(
        self,
        port: object,
        mode: ControlMode,
        values: torch.Tensor,
    ) -> None:
        if mode == "position":
            port.write_joint_velocity_targets(
                self._all_env_ids,
                self._zero_by_width[values.shape[1]],
            )
            port.write_joint_position_targets(self._all_env_ids, values)
        elif mode == "velocity":
            port.write_joint_velocity_targets(self._all_env_ids, values)
        else:
            port.write_joint_effort_targets(self._all_env_ids, values)

    def _commit_canonical(
        self,
        *,
        position_reference: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        references = getattr(self.views, "position_references", None)
        targets = getattr(self.views, "control_targets", None)
        if references is not None:
            references.copy_(position_reference)
        if targets is not None:
            targets.copy_(target)

    def _mark_runtime_fatal(self, message: str) -> None:
        runtime = self._runtime
        marker = getattr(runtime, "mark_control_mode_fatal", None)
        if callable(marker):
            marker(message)

    def _wrong_trajectory(self, control: object) -> None:
        raise ControlModeIncompatibleError(
            f"{type(control).__name__} does not match active mode {self._active_mode!r}",
            active_mode=self._active_mode,
            operation="action.dispatch",
        )


__all__ = [
    "KaleidoscopeControlBinding",
    "KaleidoscopeControlModeCoordinator",
]
