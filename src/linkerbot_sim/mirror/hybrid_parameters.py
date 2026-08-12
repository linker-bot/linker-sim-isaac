"""Owner-queued runtime tuning state for Mirror hybrid control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

from linkerbot_sim.configuration.control import (
    HybridForcePositionSettings,
    HybridTuningLimits,
)


HYBRID_PARAMETER_FIELDS = frozenset(
    {
        "motion_stiffness",
        "motion_damping",
        "force_proportional",
        "force_integral",
        "posture_stiffness",
        "posture_damping",
    }
)


class HybridParameterError(RuntimeError):
    """Base class for stable hybrid-parameter API failures."""

    code = "hybrid_parameter_error"


class HybridNotConfiguredError(HybridParameterError):
    code = "hybrid_not_configured"


class HybridParameterGenerationConflict(HybridParameterError):
    code = "hybrid_parameter_generation_conflict"

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = _generation(expected, label="expected_generation")
        self.actual = _generation(actual, label="actual_generation")
        super().__init__(
            "hybrid parameter generation conflict: "
            f"expected={self.expected}, actual={self.actual}"
        )


class HybridParameterOutOfRange(HybridParameterError):
    code = "hybrid_parameter_out_of_range"

    def __init__(
        self,
        *,
        field: str,
        value: float,
        maximum: float,
        index: int | None = None,
    ) -> None:
        self.field = field
        self.value = float(value)
        self.maximum = float(maximum)
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(
            f"{location}={self.value} exceeds configured maximum {self.maximum}"
        )

    @property
    def details(self) -> dict[str, object]:
        result: dict[str, object] = {
            "field": self.field,
            "value": self.value,
            "maximum": self.maximum,
        }
        if self.index is not None:
            result["index"] = self.index
        return result


@dataclass(frozen=True, slots=True)
class HybridParameterValues:
    motion_stiffness: tuple[float, ...]
    motion_damping: tuple[float, ...]
    force_proportional: tuple[float, ...]
    force_integral: tuple[float, ...]
    posture_stiffness: float
    posture_damping: float

    @classmethod
    def from_settings(
        cls, settings: HybridForcePositionSettings
    ) -> "HybridParameterValues":
        return cls(
            motion_stiffness=tuple(settings.motion.stiffness),
            motion_damping=tuple(settings.motion.damping),
            force_proportional=tuple(settings.force.proportional),
            force_integral=tuple(settings.force.integral),
            posture_stiffness=float(settings.posture.stiffness),
            posture_damping=float(settings.posture.damping),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "motion_stiffness": list(self.motion_stiffness),
            "motion_damping": list(self.motion_damping),
            "force_proportional": list(self.force_proportional),
            "force_integral": list(self.force_integral),
            "posture_stiffness": self.posture_stiffness,
            "posture_damping": self.posture_damping,
        }


@dataclass(frozen=True, slots=True)
class HybridParameterSnapshot:
    generation: int
    values: HybridParameterValues

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "parameters": self.values.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class HybridParameterChange:
    previous_generation: int
    generation: int
    changed: bool
    values: HybridParameterValues

    def as_dict(self) -> dict[str, object]:
        return {
            "event": "hybrid_parameters_updated",
            "previous_generation": self.previous_generation,
            "generation": self.generation,
            "changed": self.changed,
            "parameters": self.values.as_dict(),
        }


class HybridParameterService:
    """Own immutable gain snapshots at the Mirror owner-queue boundary.

    A hybrid motion obtains exactly one :meth:`snapshot` before its controller
    override. Since get/set and motion operations share the owner queue, no
    parameter mutation can interleave with an active motion.
    """

    def __init__(self, settings: HybridForcePositionSettings | None) -> None:
        self._limits = None if settings is None else settings.tuning
        self._values = (
            None if settings is None else HybridParameterValues.from_settings(settings)
        )
        self._generation = 0

    @property
    def configured(self) -> bool:
        return self._values is not None

    def snapshot(self) -> HybridParameterSnapshot:
        return HybridParameterSnapshot(
            generation=self._generation,
            values=self._require_values(),
        )

    def get_state(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            "event": "hybrid_parameters",
            **snapshot.as_dict(),
            "tuning_limits": _limits_dict(self._require_limits()),
        }

    def set_parameters(
        self,
        updates: Mapping[str, object],
        *,
        expected_generation: int | None = None,
    ) -> HybridParameterChange:
        current = self._require_values()
        limits = self._require_limits()
        unknown = sorted(set(updates) - HYBRID_PARAMETER_FIELDS)
        if unknown:
            raise ValueError(
                "control.set_hybrid_parameters contains unknown parameters: "
                + ", ".join(unknown)
            )
        if not updates:
            raise ValueError(
                "control.set_hybrid_parameters requires at least one parameter"
            )
        if expected_generation is not None:
            expected = _generation(expected_generation, label="expected_generation")
            if expected != self._generation:
                raise HybridParameterGenerationConflict(
                    expected=expected,
                    actual=self._generation,
                )

        normalized: dict[str, object] = {}
        vector_limits = {
            "motion_stiffness": limits.max_motion_stiffness,
            "motion_damping": limits.max_motion_damping,
            "force_proportional": limits.max_force_proportional,
            "force_integral": limits.max_force_integral,
        }
        scalar_limits = {
            "posture_stiffness": limits.max_posture_stiffness,
            "posture_damping": limits.max_posture_damping,
        }
        for field, value in updates.items():
            if field in vector_limits:
                vector = _non_negative_vector(value, label=field)
                maximum = vector_limits[field]
                for index, (item, bound) in enumerate(
                    zip(vector, maximum, strict=True)
                ):
                    if item > bound:
                        raise HybridParameterOutOfRange(
                            field=field,
                            index=index,
                            value=item,
                            maximum=bound,
                        )
                normalized[field] = vector
                continue
            scalar = _non_negative_number(value, label=field)
            maximum = scalar_limits[field]
            if scalar > maximum:
                raise HybridParameterOutOfRange(
                    field=field,
                    value=scalar,
                    maximum=maximum,
                )
            normalized[field] = scalar

        updated = replace(current, **normalized)
        previous_generation = self._generation
        changed = updated != current
        if changed:
            self._values = updated
            self._generation += 1
        return HybridParameterChange(
            previous_generation=previous_generation,
            generation=self._generation,
            changed=changed,
            values=self._require_values(),
        )

    def _require_values(self) -> HybridParameterValues:
        if self._values is None:
            raise HybridNotConfiguredError(
                "hybrid force/position control is not configured for this runtime"
            )
        return self._values

    def _require_limits(self) -> HybridTuningLimits:
        if self._limits is None:
            raise HybridNotConfiguredError(
                "hybrid force/position control is not configured for this runtime"
            )
        return self._limits


def _generation(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _non_negative_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _non_negative_vector(value: object, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array of six numbers")
    if len(value) != 6:
        raise ValueError(f"{label} must contain exactly six numbers")
    return tuple(
        _non_negative_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _limits_dict(limits: HybridTuningLimits) -> dict[str, object]:
    return {
        "motion_stiffness": list(limits.max_motion_stiffness),
        "motion_damping": list(limits.max_motion_damping),
        "force_proportional": list(limits.max_force_proportional),
        "force_integral": list(limits.max_force_integral),
        "posture_stiffness": limits.max_posture_stiffness,
        "posture_damping": limits.max_posture_damping,
    }


__all__ = [
    "HYBRID_PARAMETER_FIELDS",
    "HybridNotConfiguredError",
    "HybridParameterChange",
    "HybridParameterError",
    "HybridParameterGenerationConflict",
    "HybridParameterOutOfRange",
    "HybridParameterService",
    "HybridParameterSnapshot",
    "HybridParameterValues",
]
