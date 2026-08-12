"""Pure projection of command-joint settings into backend-neutral arrays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.controllers.types import (
    ControlMethod,
    ControlMode,
    JointControlSettings,
    resolve_joint_parameter,
)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CommandRuntimeProjection:
    """Resolved command-DOF control law in stable command-joint order."""

    joint_names: tuple[str, ...]
    components: tuple[str, ...]
    modes: tuple[ControlMode, ...]
    methods: tuple[ControlMethod, ...]
    physical_modes: tuple[ControlMode, ...]
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limits: np.ndarray
    drive_stiffness: np.ndarray
    drive_damping: np.ndarray

    def __post_init__(self) -> None:
        width = len(self.joint_names)
        if width < 1 or len(set(self.joint_names)) != width:
            raise ValueError("command projection joint_names must be non-empty/unique")
        for name, values in (
            ("components", self.components),
            ("modes", self.modes),
            ("methods", self.methods),
            ("physical_modes", self.physical_modes),
        ):
            if len(values) != width:
                raise ValueError(f"command projection {name} has the wrong length")
        for name in (
            "stiffness",
            "damping",
            "effort_limits",
            "drive_stiffness",
            "drive_damping",
        ):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if values.shape != (width,) or not np.all(np.isfinite(values)):
                raise ValueError(f"command projection {name} must be finite ({width},)")
            object.__setattr__(self, name, _readonly(values))


def project_command_runtime(
    settings: JointControlSettings,
    *,
    joint_names: Sequence[str],
    components: Sequence[str],
) -> CommandRuntimeProjection:
    """Resolve component settings without reading or mutating an engine handle."""

    names = tuple(str(name) for name in joint_names)
    groups = tuple(str(component) for component in components)
    if not names or len(names) != len(groups):
        raise ValueError("joint_names/components must be non-empty and equal length")
    if len(set(names)) != len(names):
        raise ValueError("joint_names cannot contain duplicates")
    width = len(names)
    stiffness = np.zeros(width, dtype=float)
    damping = np.zeros(width, dtype=float)
    effort_limits = np.zeros(width, dtype=float)
    drive_stiffness = np.zeros(width, dtype=float)
    drive_damping = np.zeros(width, dtype=float)
    modes: list[ControlMode | None] = [None] * width
    methods: list[ControlMethod | None] = [None] * width

    for group in dict.fromkeys(groups):
        selected = np.asarray(
            [index for index, component in enumerate(groups) if component == group],
            dtype=int,
        )
        selected_names = tuple(names[index] for index in selected)
        component_settings = settings.component(selected_names[0], component=group)
        kp = resolve_joint_parameter(
            component_settings.stiffness,
            selected_names,
            label=f"{group} active stiffness",
        )
        kd = resolve_joint_parameter(
            component_settings.damping,
            selected_names,
            label=f"{group} active damping",
        )
        limits = np.abs(component_settings.active_effort_limits(selected_names))
        stiffness[selected] = kp
        damping[selected] = kd
        effort_limits[selected] = limits
        if (
            component_settings.mode == "position"
            and component_settings.method == "implicit"
        ):
            drive_stiffness[selected] = kp
            drive_damping[selected] = kd
        elif (
            component_settings.mode == "velocity"
            and component_settings.method == "implicit"
        ):
            drive_damping[selected] = kd
        for index in selected:
            modes[int(index)] = component_settings.mode
            methods[int(index)] = component_settings.method

    resolved_modes = tuple(mode for mode in modes if mode is not None)
    resolved_methods = tuple(method for method in methods if method is not None)
    if len(resolved_modes) != width or len(resolved_methods) != width:
        raise RuntimeError("command projection did not resolve every joint")
    physical_modes: tuple[ControlMode, ...] = tuple(
        mode if method == "implicit" else "effort"
        for mode, method in zip(resolved_modes, resolved_methods, strict=True)
    )
    return CommandRuntimeProjection(
        joint_names=names,
        components=groups,
        modes=resolved_modes,
        methods=resolved_methods,
        physical_modes=physical_modes,
        stiffness=stiffness,
        damping=damping,
        effort_limits=effort_limits,
        drive_stiffness=drive_stiffness,
        drive_damping=drive_damping,
    )


__all__ = ["CommandRuntimeProjection", "project_command_runtime"]
