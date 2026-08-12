"""Transactional all-robot control-mode switching for one Mirror runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.controllers.control_mode import (
    CONTROL_MODES,
    ControlModeChange,
    ControlModeGenerationConflict,
    ControlModeRollbackError,
    ControlModeState,
    ControlModeSwitchError,
    require_control_mode,
    require_expected_generation,
)
from linkerbot_sim.controllers.projection import joint_control_settings
from linkerbot_sim.controllers.types import ControlMode, ControlTargets
from linkerbot_sim.snapshots.transactions import (
    SnapshotRollbackError,
    mutation_transaction,
    require_runtime_mutable,
)
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


@dataclass(frozen=True, slots=True)
class MirrorControlBinding:
    """Construction-time controller/profile binding for one robot."""

    label: str
    controller: object
    controller_profiles: ControllerProfiles
    articulation_action_type: object

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Mirror control binding label cannot be empty")
        if not isinstance(self.controller_profiles, ControllerProfiles):
            raise TypeError("controller_profiles must be ControllerProfiles")
        for method in (
            "prepare_runtime",
            "apply_prepared_runtime",
            "build_control_targets",
            "apply_targets",
        ):
            if not callable(getattr(self.controller, method, None)):
                raise TypeError(f"Mirror controller must implement {method}()")


@dataclass(frozen=True, slots=True)
class _PreparedSwitch:
    binding: MirrorControlBinding
    old_runtime: object
    new_runtime: object
    old_neutral: ControlTargets
    new_neutral: ControlTargets
    old_targets: ControlTargets | None
    old_efforts: np.ndarray


class MirrorControlModeService:
    """Own logical mode state and compensate engine writes in reverse order."""

    def __init__(
        self,
        *,
        initial_mode: ControlMode,
        bindings: tuple[MirrorControlBinding, ...],
    ) -> None:
        self._initial_mode = require_control_mode(initial_mode, label="initial_mode")
        self._active_mode = self._initial_mode
        self._generation = 0
        self._bindings = tuple(bindings)
        labels = tuple(binding.label for binding in self._bindings)
        if len(set(labels)) != len(labels):
            raise ValueError("Mirror control binding labels cannot repeat")
        self._runtime: object | None = None

    def bind_runtime(self, runtime: object) -> None:
        if self._runtime is not None:
            raise RuntimeError("Mirror control-mode service runtime is already bound")
        self._runtime = runtime

    def get_mode(self) -> ControlModeState:
        return ControlModeState(
            initial_mode=self._initial_mode,
            active_mode=self._active_mode,
            generation=self._generation,
            supported_modes=CONTROL_MODES,
        )

    def set_mode(
        self,
        mode: ControlMode,
        *,
        expected_generation: int | None = None,
    ) -> ControlModeChange:
        requested = require_control_mode(mode)
        if expected_generation is not None:
            expected = require_expected_generation(expected_generation)
            if expected != self._generation:
                raise ControlModeGenerationConflict(
                    expected=expected,
                    actual=self._generation,
                )
        runtime = self._require_runtime()
        require_runtime_mutable(runtime, operation="Mirror control-mode switch")
        previous = self._active_mode
        if requested == previous:
            return ControlModeChange(
                previous_mode=previous,
                active_mode=previous,
                generation=self._generation,
                changed=False,
            )

        prepared = tuple(
            self._prepare_binding(binding, requested) for binding in self._bindings
        )
        try:
            with mutation_transaction(
                runtime,
                operation=f"Mirror control-mode {previous}->{requested}",
            ) as transaction:
                for plan in prepared:
                    transaction.add_rollback(
                        f"robot {plan.binding.label} control mode",
                        lambda plan=plan: self._rollback_binding(plan),
                    )
                    controller = plan.binding.controller
                    controller.apply_targets(
                        plan.binding.articulation_action_type,
                        plan.old_neutral,
                    )
                    controller.apply_prepared_runtime(
                        plan.new_runtime,
                        clear_target_cache=False,
                    )
                    controller.apply_targets(
                        plan.binding.articulation_action_type,
                        plan.new_neutral,
                    )
        except SnapshotRollbackError as exc:
            raise ControlModeRollbackError(str(exc)) from exc
        except BaseException as exc:
            raise ControlModeSwitchError(
                f"failed to switch Mirror control mode {previous}->{requested}: {exc}"
            ) from exc

        self._active_mode = requested
        self._generation += 1
        return ControlModeChange(
            previous_mode=previous,
            active_mode=requested,
            generation=self._generation,
            changed=True,
        )

    def _prepare_binding(
        self,
        binding: MirrorControlBinding,
        requested: ControlMode,
    ) -> _PreparedSwitch:
        controller = binding.controller
        old_runtime = controller.prepare_runtime()
        new_runtime = controller.prepare_runtime(
            joint_control_settings(binding.controller_profiles, mode=requested)
        )
        robot = controller.robot
        current_q = tensor_like_to_numpy(
            robot.get_joint_positions(), dtype=float
        ).reshape(-1)
        if current_q.shape != (int(robot.num_dof),) or not np.all(
            np.isfinite(current_q)
        ):
            raise ValueError(
                f"robot {binding.label!r} returned invalid current joint positions"
            )
        zeros = np.zeros(len(controller.command_indices), dtype=float)
        neutral = controller.build_control_targets(
            command_positions=current_q[controller.command_indices],
            command_velocities=zeros,
            command_efforts=zeros,
            base_positions=current_q,
        )
        old_targets = controller.snapshot_control_targets_cache()
        old_efforts = np.asarray(controller.last_commanded_efforts, dtype=float).copy()
        return _PreparedSwitch(
            binding=binding,
            old_runtime=old_runtime,
            new_runtime=new_runtime,
            old_neutral=neutral,
            new_neutral=neutral,
            old_targets=old_targets,
            old_efforts=old_efforts,
        )

    @staticmethod
    def _rollback_binding(plan: _PreparedSwitch) -> None:
        controller = plan.binding.controller
        controller.apply_prepared_runtime(
            plan.old_runtime,
            clear_target_cache=False,
        )
        controller.apply_targets(
            plan.binding.articulation_action_type,
            plan.old_targets if plan.old_targets is not None else plan.old_neutral,
        )
        controller.restore_control_targets_cache(plan.old_targets)
        controller.last_commanded_efforts = plan.old_efforts.copy()

    def _require_runtime(self) -> object:
        if self._runtime is None:
            raise RuntimeError("Mirror control-mode service runtime is not bound")
        return self._runtime


__all__ = ["MirrorControlBinding", "MirrorControlModeService"]
