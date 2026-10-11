"""Pure compatibility gate and flat payload for runtime reset descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from linkerbot_sim.configuration.fingerprint import semantic_config_fingerprint
from linkerbot_sim.kaleidoscope.reset_description import (
    KaleidoscopeResetSceneDescription,
    ResetGeneratorDescription,
    validate_reset_scene_description,
)
from linkerbot_sim.kaleidoscope.reset_sampling import (
    derive_reset_generator_description,
)

if TYPE_CHECKING:
    from linkerbot_sim.configuration.modes.kaleidoscope import KaleidoscopeConfig


@dataclass(frozen=True, slots=True)
class ResetCommandPayload:
    """Validated one-row values ready for a later CPU-to-CUDA copy."""

    generator: ResetGeneratorDescription
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    joint_targets: tuple[float, ...]
    block_position: tuple[float, float, float]
    block_orientation_wxyz: tuple[float, float, float, float]
    block_velocity: tuple[float, float, float, float, float, float]
    goal_position: tuple[float, float, float]
    goal_yaw: float

    def __post_init__(self) -> None:
        if not isinstance(self.generator, ResetGeneratorDescription):
            raise TypeError("generator must be a ResetGeneratorDescription")
        positions = _finite_tuple(self.joint_positions, label="joint_positions")
        if not positions:
            raise ValueError("joint_positions cannot be empty")
        object.__setattr__(self, "joint_positions", positions)
        for name in ("joint_velocities", "joint_targets"):
            values = _finite_tuple(getattr(self, name), label=name)
            if len(values) != len(positions):
                raise ValueError(f"{name} must match joint_positions length")
            object.__setattr__(self, name, values)
        for name, length in (
            ("block_position", 3),
            ("block_orientation_wxyz", 4),
            ("block_velocity", 6),
            ("goal_position", 3),
        ):
            values = _finite_tuple(getattr(self, name), label=name)
            if len(values) != length:
                raise ValueError(f"{name} must contain exactly {length} values")
            object.__setattr__(self, name, values)
        goal_yaw = float(self.goal_yaw)
        if isinstance(self.goal_yaw, bool) or not math.isfinite(goal_yaw):
            raise ValueError("goal_yaw must be a finite number")
        object.__setattr__(self, "goal_yaw", goal_yaw)


def project_reset_description(
    config: "KaleidoscopeConfig",
    value: KaleidoscopeResetSceneDescription | Mapping[str, object],
) -> ResetCommandPayload:
    """Validate one portable description against a config before device writes."""

    description = validate_reset_scene_description(value)
    expected_fingerprint = semantic_config_fingerprint(config)
    if description.compatibility_fingerprint != expected_fingerprint:
        raise ValueError("reset description compatibility fingerprint does not match")
    if description.scene_id != config.scene.scene_id:
        raise ValueError("reset description scene_id does not match")
    if description.task_id != config.task.task_id:
        raise ValueError("reset description task_id does not match")

    expected_robots = config.scene.robots
    if len(description.robots) != len(expected_robots):
        raise ValueError("reset description robot count does not match")
    joint_count = 0
    for index, (actual, expected) in enumerate(
        zip(description.robots, expected_robots, strict=True)
    ):
        profile = expected.resolved_profile
        if profile is None:
            raise ValueError(f"config robot {expected.label!r} has no resolved profile")
        if actual.label != expected.label:
            raise ValueError(f"reset description robot[{index}] label does not match")
        if actual.robot_profile != expected.robot_profile:
            raise ValueError(
                f"reset description robot[{index}] robot_profile does not match"
            )
        expected_names = profile.joint_groups.command_joint_names
        if actual.joint_names != expected_names:
            raise ValueError(
                f"reset description robot[{index}] joint_names do not match"
            )
        joint_count += len(expected_names)

    expected_generator = derive_reset_generator_description(
        seed=description.generator.seed,
        sample_id=description.generator.sample_id,
        reset_index=description.generator.reset_index,
        joint_count=joint_count,
    )
    if description.generator != expected_generator:
        raise ValueError("reset description generator metadata is inconsistent")

    dynamic_instances = tuple(
        item for item in config.scene.objects if item.name == config.task.dynamic_object
    )
    if len(dynamic_instances) != 1:
        raise ValueError("config task.dynamic_object must resolve exactly once")
    expected_object = dynamic_instances[0]
    if description.dynamic_object.name != expected_object.name:
        raise ValueError("reset description dynamic object name does not match")
    if description.dynamic_object.object_profile != expected_object.object_profile:
        raise ValueError("reset description dynamic object profile does not match")

    return ResetCommandPayload(
        generator=description.generator,
        joint_positions=tuple(
            value for robot in description.robots for value in robot.joint_positions
        ),
        joint_velocities=tuple(
            value for robot in description.robots for value in robot.joint_velocities
        ),
        joint_targets=tuple(
            value for robot in description.robots for value in robot.joint_targets
        ),
        block_position=description.dynamic_object.position_xyz,
        block_orientation_wxyz=description.dynamic_object.orientation_wxyz,
        block_velocity=description.dynamic_object.com_velocity,
        goal_position=description.task.goal_position,
        goal_yaw=description.task.goal_yaw,
    )


def _finite_tuple(value: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a numeric sequence")
    result: list[float] = []
    for child in value:
        if isinstance(child, bool) or not isinstance(child, (int, float)):
            raise ValueError(f"{label} must contain only finite numbers")
        number = float(child)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain only finite numbers")
        result.append(number)
    return tuple(result)


__all__ = ["ResetCommandPayload", "project_reset_description"]
