"""Runtime control-mode state and product-neutral errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from linkerbot_sim.controllers.types import ControlMode


CONTROL_MODES: tuple[ControlMode, ...] = ("position", "velocity", "effort")


def require_control_mode(value: object, *, label: str = "mode") -> ControlMode:
    """Return one of the three exact public control-mode strings."""

    if not isinstance(value, str) or value not in CONTROL_MODES:
        raise ValueError(f"{label} must be one of {list(CONTROL_MODES)}, got {value!r}")
    return cast(ControlMode, value)


def require_expected_generation(value: object) -> int:
    """Validate an optimistic generation without accepting bool as int."""

    if type(value) is not int or value < 0:
        raise ValueError("expected_generation must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ControlModeState:
    initial_mode: ControlMode
    active_mode: ControlMode
    generation: int
    supported_modes: tuple[ControlMode, ...]
    scope: str = "all"

    def __post_init__(self) -> None:
        require_control_mode(self.initial_mode, label="initial_mode")
        require_control_mode(self.active_mode, label="active_mode")
        require_expected_generation(self.generation)
        if not self.supported_modes:
            raise ValueError("supported_modes cannot be empty")
        normalized = tuple(
            require_control_mode(mode, label="supported_modes")
            for mode in self.supported_modes
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("supported_modes cannot contain duplicates")
        if self.initial_mode not in normalized or self.active_mode not in normalized:
            raise ValueError("initial_mode and active_mode must be supported")
        if self.scope != "all":
            raise ValueError("control-mode scope must be 'all'")
        object.__setattr__(self, "supported_modes", normalized)

    def as_dict(self) -> dict[str, object]:
        return {
            "initial_mode": self.initial_mode,
            "active_mode": self.active_mode,
            "generation": self.generation,
            "supported_modes": list(self.supported_modes),
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class ControlModeChange:
    previous_mode: ControlMode
    active_mode: ControlMode
    generation: int
    changed: bool

    def __post_init__(self) -> None:
        require_control_mode(self.previous_mode, label="previous_mode")
        require_control_mode(self.active_mode, label="active_mode")
        require_expected_generation(self.generation)
        if type(self.changed) is not bool:
            raise TypeError("changed must be bool")

    def as_dict(self) -> dict[str, object]:
        return {
            "previous_mode": self.previous_mode,
            "active_mode": self.active_mode,
            "generation": self.generation,
            "changed": self.changed,
        }


class ControlModeError(RuntimeError):
    """Base error for a rejected or failed runtime mode operation."""


class ControlModeIncompatibleError(ControlModeError):
    """The requested operation/action cannot run in the active mode."""

    def __init__(
        self,
        message: str,
        *,
        active_mode: ControlMode | None = None,
        operation: str | None = None,
        location: dict[str, object] | None = None,
    ) -> None:
        self.active_mode = active_mode
        self.operation = operation
        self.location = None if location is None else dict(location)
        super().__init__(message)

    @property
    def details(self) -> dict[str, object]:
        result: dict[str, object] = {}
        if self.active_mode is not None:
            result["active_mode"] = self.active_mode
        if self.operation is not None:
            result["operation"] = self.operation
        if self.location is not None:
            result["location"] = dict(self.location)
        return result


class ControlModeGenerationConflict(ControlModeError):
    """Optimistic generation did not match the runtime state."""

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = require_expected_generation(expected)
        self.actual = require_expected_generation(actual)
        super().__init__(
            f"control mode generation conflict: expected={expected}, actual={actual}"
        )


class ControlModeSwitchError(ControlModeError):
    """A forward switch failed and compensation completed."""


class ControlModeRollbackError(ControlModeError):
    """A switch failed and at least one compensation step also failed."""


class ControlModeLockedError(ControlModeError):
    """The runtime is not at a legal between-motion mutation boundary."""


__all__ = [
    "CONTROL_MODES",
    "ControlModeChange",
    "ControlModeError",
    "ControlModeGenerationConflict",
    "ControlModeIncompatibleError",
    "ControlModeLockedError",
    "ControlModeRollbackError",
    "ControlModeState",
    "ControlModeSwitchError",
    "require_control_mode",
    "require_expected_generation",
]
