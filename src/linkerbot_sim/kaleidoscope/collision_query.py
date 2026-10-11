"""Kaleidoscope articulated arm-and-hand collision-query contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


class CollisionQueryTarget(StrEnum):
    """Stable scene categories accepted by the first query API version."""

    TABLE = "table"
    DYNAMIC_OBJECT = "dynamic_object"
    OTHER_ROBOT = "other_robot"


class CollisionHitCategory(IntEnum):
    """Integer codes stored in ``first_hit_category`` result tensors."""

    NONE = 0
    TABLE = 1
    DYNAMIC_OBJECT = 2
    OTHER_ROBOT = 3


class CollisionQueryUnavailableError(RuntimeError):
    """Raised when the selected physics backend has no collision-query port."""

    def __init__(self, backend_kind: str) -> None:
        self.backend_kind = str(backend_kind)
        super().__init__(
            "articulated collision query is unavailable for physics backend "
            f"{self.backend_kind!r}"
        )


@dataclass(frozen=True, slots=True)
class ArticulatedCollisionQuerySpec:
    """Immutable command-space metadata for one queryable robot."""

    robot_label: str
    command_joint_names: tuple[str, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.robot_label, str) or not self.robot_label:
            raise ValueError("robot_label must be a non-empty string")
        names = self.command_joint_names
        if (
            not isinstance(names, tuple)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "command_joint_names must be a non-empty unique string tuple"
            )
        lower = _finite_float_tuple(
            self.joint_lower,
            name="joint_lower",
            width=len(names),
        )
        upper = _finite_float_tuple(
            self.joint_upper,
            name="joint_upper",
            width=len(names),
        )
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("joint_lower must be strictly below joint_upper")

    @property
    def command_dim(self) -> int:
        """Return the required final width ``J`` of a candidate tensor."""

        return len(self.command_joint_names)


@dataclass(frozen=True, slots=True)
class ArticulatedCollisionQueryResult:
    """Owned query result tensors.

    ``collision_mask`` has shape ``[selected_envs, samples]``. A single pose is
    represented by one sample. ``first_hit_sample`` and ``first_hit_category``
    each have shape ``[selected_envs]``; rows without a hit use ``-1`` and
    :attr:`CollisionHitCategory.NONE`, respectively.
    """

    collision_mask: torch.Tensor
    first_hit_sample: torch.Tensor
    first_hit_category: torch.Tensor


@runtime_checkable
class ArticulatedCollisionQueryPort(Protocol):
    """Backend port consumed by ``KaleidoscopeRuntime``.

    Implementations must be synchronous and read-only with respect to physics,
    control, task, snapshot, and RNG state.
    """

    def query(
        self,
        *,
        robot_label: str,
        joint_positions: torch.Tensor,
        env_ids: torch.Tensor,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> ArticulatedCollisionQueryResult: ...

    def close(self) -> None: ...


def _finite_float_tuple(
    value: tuple[float, ...],
    *,
    name: str,
    width: int,
) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != width:
        raise ValueError(f"{name} must be a tuple matching command_joint_names")
    if any(type(item) is not float for item in value):
        raise TypeError(f"{name} must contain only floats")
    if any(not math.isfinite(item) for item in value):
        raise ValueError(f"{name} must contain only finite values")
    return value


__all__ = [
    "ArticulatedCollisionQueryPort",
    "ArticulatedCollisionQueryResult",
    "ArticulatedCollisionQuerySpec",
    "CollisionHitCategory",
    "CollisionQueryTarget",
    "CollisionQueryUnavailableError",
]
