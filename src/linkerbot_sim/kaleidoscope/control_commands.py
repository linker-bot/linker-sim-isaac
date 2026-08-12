"""Typed, unit-bearing control trajectories for one Kaleidoscope decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class PositionControlTrajectory:
    """Position targets in rad with optional velocity feed-forward in rad/s."""

    positions: "torch.Tensor"
    velocities: "torch.Tensor"


@dataclass(frozen=True, slots=True)
class VelocityControlTrajectory:
    """Velocity targets in rad/s."""

    velocities: "torch.Tensor"


@dataclass(frozen=True, slots=True)
class EffortControlTrajectory:
    """Direct joint force/torque targets in backend articulation units."""

    efforts: "torch.Tensor"


ControlTrajectory: TypeAlias = (
    PositionControlTrajectory | VelocityControlTrajectory | EffortControlTrajectory
)


__all__ = [
    "ControlTrajectory",
    "EffortControlTrajectory",
    "PositionControlTrajectory",
    "VelocityControlTrajectory",
]
