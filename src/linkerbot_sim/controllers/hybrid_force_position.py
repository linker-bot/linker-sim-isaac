"""Pure task-space hybrid force/position control.

The module intentionally has no Isaac dependency.  It consumes one normalized
task-space observation per physics tick and returns arm joint efforts.  All
poses, twists, Jacobians and wrenches must already be expressed in the same
control frame; the PhysX adapter owns that normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _finite_vector(value: object, length: int, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (length,):
        raise ValueError(f"{label} must have shape ({length},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain finite values")
    return np.ascontiguousarray(result, dtype=float)


def _readonly(value: object, length: int, *, label: str) -> np.ndarray:
    result = _finite_vector(value, length, label=label).copy()
    result.setflags(write=False)
    return result


def _readonly_matrix(
    value: object,
    shape: tuple[int, int],
    *,
    label: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain finite values")
    result = np.ascontiguousarray(result, dtype=float).copy()
    result.setflags(write=False)
    return result


class HybridControlError(RuntimeError):
    """A stable hybrid-control failure that the product layer can map to JSON."""

    code = "hybrid_control_failed"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = MappingProxyType(dict(details or {}))


class HybridSafetyLimitError(HybridControlError):
    code = "hybrid_safety_limit"


class HybridSingularityError(HybridControlError):
    code = "hybrid_singularity"


@dataclass(frozen=True, slots=True)
class HybridControlParameters:
    """Runtime-tunable gains frozen for one motion execution."""

    motion_stiffness: tuple[float, ...]
    motion_damping: tuple[float, ...]
    force_proportional: tuple[float, ...]
    force_integral: tuple[float, ...]
    posture_stiffness: float
    posture_damping: float

    def __post_init__(self) -> None:
        for name in (
            "motion_stiffness",
            "motion_damping",
            "force_proportional",
            "force_integral",
        ):
            values = _finite_vector(getattr(self, name), 6, label=name)
            if np.any(values < 0.0):
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, tuple(float(item) for item in values))
        for name in ("posture_stiffness", "posture_damping"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

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
class HybridControlTarget:
    position: np.ndarray
    orientation_wxyz: np.ndarray
    force_axes: np.ndarray
    wrench_tool_on_environment: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _readonly(self.position, 3, label="target position")
        )
        quaternion = _finite_vector(
            self.orientation_wxyz, 4, label="target orientation"
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            raise ValueError("target orientation quaternion cannot be zero")
        quaternion = quaternion / norm
        quaternion.setflags(write=False)
        object.__setattr__(self, "orientation_wxyz", quaternion)
        force_axes = np.asarray(self.force_axes)
        if force_axes.shape != (6,) or force_axes.dtype.kind != "b":
            raise ValueError("force_axes must be a boolean vector with shape (6,)")
        if not np.any(force_axes) or np.all(force_axes):
            raise ValueError("force_axes must select at least one but not all axes")
        force_axes = np.ascontiguousarray(force_axes, dtype=bool).copy()
        force_axes.setflags(write=False)
        object.__setattr__(self, "force_axes", force_axes)
        wrench = _readonly(self.wrench_tool_on_environment, 6, label="target wrench")
        if np.any(np.abs(wrench[~force_axes]) > 1.0e-12):
            raise ValueError("target wrench must be zero on motion-controlled axes")
        object.__setattr__(self, "wrench_tool_on_environment", wrench)


@dataclass(frozen=True, slots=True)
class TaskSpaceObservation:
    """One CPU observation normalized to a single task-space frame."""

    position: np.ndarray
    orientation_wxyz: np.ndarray
    twist: np.ndarray
    jacobian: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    external_wrench_environment_on_tool: np.ndarray
    sequence: int

    def __post_init__(self) -> None:
        q = np.asarray(self.joint_positions, dtype=float).reshape(-1)
        n = int(q.size)
        if n < 6 or not np.all(np.isfinite(q)):
            raise ValueError("joint_positions must contain at least six finite values")
        object.__setattr__(
            self, "position", _readonly(self.position, 3, label="position")
        )
        orientation = _finite_vector(self.orientation_wxyz, 4, label="orientation_wxyz")
        norm = float(np.linalg.norm(orientation))
        if norm <= 1.0e-12:
            raise ValueError("orientation quaternion cannot be zero")
        orientation = orientation / norm
        orientation.setflags(write=False)
        object.__setattr__(self, "orientation_wxyz", orientation)
        object.__setattr__(self, "twist", _readonly(self.twist, 6, label="twist"))
        object.__setattr__(
            self,
            "jacobian",
            _readonly_matrix(self.jacobian, (6, n), label="jacobian"),
        )
        q = np.ascontiguousarray(q, dtype=float).copy()
        q.setflags(write=False)
        object.__setattr__(self, "joint_positions", q)
        object.__setattr__(
            self,
            "joint_velocities",
            _readonly(self.joint_velocities, n, label="joint_velocities"),
        )
        object.__setattr__(
            self,
            "external_wrench_environment_on_tool",
            _readonly(
                self.external_wrench_environment_on_tool,
                6,
                label="external wrench",
            ),
        )
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("observation sequence must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class HybridControlOutput:
    joint_efforts: np.ndarray
    commanded_wrench: np.ndarray
    motion_wrench: np.ndarray
    force_wrench: np.ndarray
    measured_wrench_tool_on_environment: np.ndarray
    pose_error: np.ndarray
    force_error: np.ndarray
    contact_axes: np.ndarray
    wrench_saturated_axes: np.ndarray
    effort_saturated_axes: np.ndarray
    minimum_singular_value: float
    condition_number: float

    def __post_init__(self) -> None:
        n = np.asarray(self.joint_efforts).size
        for name, length in (
            ("joint_efforts", n),
            ("commanded_wrench", 6),
            ("motion_wrench", 6),
            ("force_wrench", 6),
            ("measured_wrench_tool_on_environment", 6),
            ("pose_error", 6),
            ("force_error", 6),
        ):
            object.__setattr__(
                self, name, _readonly(getattr(self, name), length, label=name)
            )
        for name, length in (
            ("contact_axes", 6),
            ("wrench_saturated_axes", 6),
            ("effort_saturated_axes", n),
        ):
            value = np.asarray(getattr(self, name), dtype=bool).reshape(-1)
            if value.shape != (length,):
                raise ValueError(f"{name} must have shape ({length},)")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class HybridForcePositionController:
    """Stateful per-command hybrid controller."""

    def __init__(
        self,
        *,
        settings: object,
        parameters: HybridControlParameters,
        target: HybridControlTarget,
        tare_external_wrench: object,
        nominal_joint_positions: object,
        initial_position: object,
        initial_orientation_wxyz: object,
        joint_effort_limits: object,
    ) -> None:
        self.settings = settings
        self.parameters = parameters
        self.target = target
        n = np.asarray(nominal_joint_positions).size
        self._tare = _finite_vector(tare_external_wrench, 6, label="tare wrench")
        self._nominal_q = _finite_vector(
            nominal_joint_positions, n, label="nominal joint positions"
        )
        self._initial_position = _finite_vector(
            initial_position, 3, label="initial position"
        )
        initial_orientation = _finite_vector(
            initial_orientation_wxyz, 4, label="initial orientation"
        )
        initial_norm = float(np.linalg.norm(initial_orientation))
        if initial_norm <= 1.0e-12:
            raise ValueError("initial orientation quaternion cannot be zero")
        self._initial_orientation = initial_orientation / initial_norm
        configured_effort_limit = float(settings.limits.max_abs_joint_effort)
        physical_limits = _finite_vector(
            joint_effort_limits, n, label="joint effort limits"
        )
        if configured_effort_limit <= 0.0 or np.any(physical_limits <= 0.0):
            raise ValueError("joint effort limits must be positive")
        self._effort_limits = np.minimum(physical_limits, configured_effort_limit)
        allowed = np.asarray(settings.allowed_force_axes, dtype=bool)
        if np.any(target.force_axes & ~allowed):
            raise ValueError("force_axes selects an axis disabled by the profile")
        if np.any(
            np.abs(target.wrench_tool_on_environment)
            > np.asarray(settings.limits.max_abs_wrench, dtype=float)
        ):
            raise ValueError("target wrench exceeds profile limits")
        self._filtered_external: np.ndarray | None = None
        self._integral = np.zeros(6, dtype=float)
        self._contact = np.zeros(6, dtype=bool)
        self._contact_enter_counts = np.zeros(6, dtype=np.int64)
        self._contact_exit_counts = np.zeros(6, dtype=np.int64)
        self._last_efforts = np.zeros(n, dtype=float)
        self._last_sequence: int | None = None

    def step(
        self, observation: TaskSpaceObservation, *, dt: float
    ) -> HybridControlOutput:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        n = self._nominal_q.size
        if observation.joint_positions.shape != (n,):
            raise ValueError("observation joint dimension changed during execution")
        if (
            self._last_sequence is not None
            and observation.sequence <= self._last_sequence
        ):
            raise HybridControlError(
                "task-space observation is stale",
                details={"sequence": observation.sequence},
            )
        self._last_sequence = observation.sequence

        max_joint_speed = float(self.settings.limits.max_joint_speed)
        speed = np.abs(observation.joint_velocities)
        if np.any(speed > max_joint_speed):
            self._raise_limit("max_joint_speed", float(np.max(speed)))

        orientation_error = rotation_vector_error_wxyz(
            self.target.orientation_wxyz, observation.orientation_wxyz
        )
        pose_error = np.concatenate(
            [self.target.position - observation.position, orientation_error]
        )
        motion_axes = ~self.target.force_axes
        pose_limits = np.asarray(self.settings.limits.max_abs_pose_error, dtype=float)
        if np.any(np.abs(pose_error[motion_axes]) > pose_limits[motion_axes]):
            self._raise_limit(
                "max_abs_pose_error",
                float(np.max(np.abs(pose_error[motion_axes]))),
            )

        centered_external = observation.external_wrench_environment_on_tool - self._tare
        cutoff = float(self.settings.force.wrench_lpf_cutoff_hz)
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff * dt)
        if self._filtered_external is None:
            self._filtered_external = centered_external.copy()
        else:
            self._filtered_external += alpha * (
                centered_external - self._filtered_external
            )
        measured = -self._filtered_external
        wrench_limits = np.asarray(self.settings.limits.max_abs_wrench, dtype=float)
        if np.any(np.abs(measured) > wrench_limits):
            self._raise_limit("measured_wrench", float(np.max(np.abs(measured))))

        self._update_contact(measured)
        drift = np.concatenate(
            [
                observation.position - self._initial_position,
                rotation_vector_error_wxyz(
                    observation.orientation_wxyz, self._initial_orientation
                ),
            ]
        )
        translational_force_axes = self.target.force_axes[:3]
        if (
            np.any(translational_force_axes)
            and not np.all(self._contact[:3][translational_force_axes])
            and np.any(
                np.abs(drift[:3][translational_force_axes])
                > float(self.settings.contact.max_free_space_displacement_m)
            )
        ):
            self._raise_limit(
                "max_free_space_displacement_m",
                float(np.max(np.abs(drift[:3][translational_force_axes]))),
            )
        rotational_force_axes = self.target.force_axes[3:]
        if np.any(
            np.abs(drift[3:][rotational_force_axes])
            > pose_limits[3:][rotational_force_axes]
        ):
            self._raise_limit(
                "max_force_axis_rotation",
                float(np.max(np.abs(drift[3:][rotational_force_axes]))),
            )

        kp = np.asarray(self.parameters.motion_stiffness, dtype=float)
        kd = np.asarray(self.parameters.motion_damping, dtype=float)
        motion_wrench = motion_axes * (kp * pose_error - kd * observation.twist)
        force_error = self.target.wrench_tool_on_environment - measured
        contact_mask = self.target.force_axes & self._contact
        candidate_integral = np.clip(
            self._integral + dt * contact_mask * force_error,
            -np.asarray(self.settings.force.integral_abs_limit, dtype=float),
            np.asarray(self.settings.force.integral_abs_limit, dtype=float),
        )
        raw_wrench, force_wrench = self._combined_wrench(
            motion_wrench,
            force_error,
            candidate_integral,
            contact_mask,
        )
        clipped_wrench = np.clip(raw_wrench, -wrench_limits, wrench_limits)
        wrench_saturated = np.abs(raw_wrench - clipped_wrench) > 1.0e-12
        blocks_integral = (
            wrench_saturated & contact_mask & (np.sign(raw_wrench) * force_error > 0.0)
        )
        if np.any(blocks_integral):
            candidate_integral[blocks_integral] = self._integral[blocks_integral]
            raw_wrench, force_wrench = self._combined_wrench(
                motion_wrench,
                force_error,
                candidate_integral,
                contact_mask,
            )
            clipped_wrench = np.clip(raw_wrench, -wrench_limits, wrench_limits)
            wrench_saturated = np.abs(raw_wrench - clipped_wrench) > 1.0e-12
        self._integral = candidate_integral

        jacobian = np.asarray(observation.jacobian, dtype=float)
        scale = np.diag(
            [
                1.0 / float(self.settings.posture.characteristic_length_m),
                1.0 / float(self.settings.posture.characteristic_length_m),
                1.0 / float(self.settings.posture.characteristic_length_m),
                1.0,
                1.0,
                1.0,
            ]
        )
        scaled = scale @ jacobian
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        minimum_singular_value = float(np.min(singular_values))
        condition_number = float(
            np.inf
            if minimum_singular_value <= 0.0
            else np.max(singular_values) / minimum_singular_value
        )
        if minimum_singular_value < float(
            self.settings.posture.minimum_singular_value
        ) or condition_number > float(self.settings.posture.maximum_condition_number):
            raise HybridSingularityError(
                "task-space Jacobian is outside the configured singularity limits",
                details={
                    "minimum_singular_value": minimum_singular_value,
                    "condition_number": condition_number,
                },
            )

        posture_effort = np.zeros(n, dtype=float)
        if bool(self.settings.posture.enabled):
            nominal_effort = (
                float(self.parameters.posture_stiffness)
                * (self._nominal_q - observation.joint_positions)
                - float(self.parameters.posture_damping) * observation.joint_velocities
            )
            damping = float(self.settings.posture.singularity_damping)
            projector = np.eye(n) - scaled.T @ np.linalg.solve(
                scaled @ scaled.T + damping * damping * np.eye(6),
                scaled,
            )
            posture_effort = projector @ nominal_effort
        raw_effort = jacobian.T @ clipped_wrench + posture_effort
        saturated_effort = np.clip(
            raw_effort, -self._effort_limits, self._effort_limits
        )
        effort_saturated = np.abs(raw_effort - saturated_effort) > 1.0e-12
        max_delta = float(self.settings.limits.max_joint_effort_rate) * dt
        command_effort = np.clip(
            saturated_effort,
            self._last_efforts - max_delta,
            self._last_efforts + max_delta,
        )
        effort_saturated |= np.abs(command_effort - saturated_effort) > 1.0e-12
        self._last_efforts = command_effort.copy()

        return HybridControlOutput(
            joint_efforts=command_effort,
            commanded_wrench=clipped_wrench,
            motion_wrench=motion_wrench,
            force_wrench=force_wrench,
            measured_wrench_tool_on_environment=measured,
            pose_error=pose_error,
            force_error=force_error,
            contact_axes=self._contact,
            wrench_saturated_axes=wrench_saturated,
            effort_saturated_axes=effort_saturated,
            minimum_singular_value=minimum_singular_value,
            condition_number=condition_number,
        )

    @property
    def last_joint_efforts(self) -> np.ndarray:
        return self._last_efforts.copy()

    def _combined_wrench(
        self,
        motion_wrench: np.ndarray,
        force_error: np.ndarray,
        integral: np.ndarray,
        contact_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        kf = np.asarray(self.parameters.force_proportional, dtype=float)
        ki = np.asarray(self.parameters.force_integral, dtype=float)
        force_wrench = self.target.force_axes * (
            self.target.wrench_tool_on_environment
            + contact_mask * (kf * force_error + ki * integral)
        )
        return motion_wrench + force_wrench, force_wrench

    def _update_contact(self, measured: np.ndarray) -> None:
        enter = np.asarray(self.settings.contact.enter_abs_wrench, dtype=float)
        exit_values = np.asarray(self.settings.contact.exit_abs_wrench, dtype=float)
        for index in np.flatnonzero(self.target.force_axes):
            if self._contact[index]:
                if abs(measured[index]) <= exit_values[index]:
                    self._contact_exit_counts[index] += 1
                else:
                    self._contact_exit_counts[index] = 0
                if self._contact_exit_counts[index] >= int(
                    self.settings.contact.exit_ticks
                ):
                    self._contact[index] = False
                    self._contact_exit_counts[index] = 0
                    self._integral[index] = 0.0
            else:
                if abs(measured[index]) >= enter[index]:
                    self._contact_enter_counts[index] += 1
                else:
                    self._contact_enter_counts[index] = 0
                if self._contact_enter_counts[index] >= int(
                    self.settings.contact.enter_ticks
                ):
                    self._contact[index] = True
                    self._contact_enter_counts[index] = 0

    @staticmethod
    def _raise_limit(name: str, value: float) -> None:
        raise HybridSafetyLimitError(
            f"hybrid control safety limit exceeded: {name}",
            details={"limit": name, "observed": value},
        )


def quaternion_multiply_wxyz(left: object, right: object) -> np.ndarray:
    """Multiply normalized or non-normalized wxyz quaternions."""

    lw, lx, ly, lz = _finite_vector(left, 4, label="left quaternion")
    rw, rx, ry, rz = _finite_vector(right, 4, label="right quaternion")
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def rotation_vector_error_wxyz(target: object, current: object) -> np.ndarray:
    """Return the shortest-arc world-frame rotation from current to target."""

    target_q = _finite_vector(target, 4, label="target quaternion").copy()
    current_q = _finite_vector(current, 4, label="current quaternion").copy()
    target_q /= np.linalg.norm(target_q)
    current_q /= np.linalg.norm(current_q)
    error = quaternion_multiply_wxyz(
        target_q,
        np.asarray([current_q[0], -current_q[1], -current_q[2], -current_q[3]]),
    )
    if error[0] < 0.0:
        error = -error
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm < 1.0e-12:
        return 2.0 * error[1:]
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(error[0])))
    return angle * error[1:] / vector_norm


__all__ = [
    "HybridControlError",
    "HybridControlOutput",
    "HybridControlParameters",
    "HybridControlTarget",
    "HybridForcePositionController",
    "HybridSafetyLimitError",
    "HybridSingularityError",
    "TaskSpaceObservation",
    "quaternion_multiply_wxyz",
    "rotation_vector_error_wxyz",
]
