"""Pure, stateless seed sampling for portable Kaleidoscope reset descriptions."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

from linkerbot_sim.configuration.fingerprint import semantic_config_fingerprint
from linkerbot_sim.kaleidoscope.reset_description import (
    KaleidoscopeResetSceneDescription,
    ResetGeneratorDescription,
    ResetObjectDescription,
    ResetRobotDescription,
    ResetTaskDescription,
)

if TYPE_CHECKING:
    from linkerbot_sim.configuration.modes.kaleidoscope import KaleidoscopeConfig


RESET_SAMPLING_ALGORITHM = "tblock-push-reset-v1"
_ENV_KEY_INCREMENT = 6364136223846793005
_LANE_INCREMENT = 747796405
_COUNTER_MULTIPLIER = 2891336453
_LCG_MULTIPLIER = 1103515245
_LCG_INCREMENT = 12345
_SIGNED_INT64_MIN = -(2**63)
_SIGNED_INT64_MAX = 2**63 - 1
_UINT64_MODULUS = 2**64


def sample_reset_scene_description(
    config: "KaleidoscopeConfig",
    *,
    seed: int,
    sample_id: int,
    reset_index: int = 0,
) -> KaleidoscopeResetSceneDescription:
    """Sample one T-block reset without importing Torch, Isaac, or CUDA.

    ``sample_id`` owns an independent logical RNG key. ``reset_index`` addresses a
    reset in that sample's sequence directly, so callers do not need mutable RNG
    state to reproduce a description.
    """

    if config.task.task_id != "tblock_push_v1":
        raise ValueError(
            "pure reset sampling currently supports only task.id='tblock_push_v1'"
        )

    joint_names_by_robot: list[tuple[str, ...]] = []
    nominal_joint_positions: list[float] = []
    for robot in config.scene.robots:
        profile = robot.resolved_profile
        positions = robot.reset_joint_positions
        if profile is None or positions is None:
            raise ValueError(
                f"robot {robot.label!r} must have a resolved profile and reset positions"
            )
        names = profile.joint_groups.command_joint_names
        joint_names_by_robot.append(names)
        nominal_joint_positions.extend(robot.ordered_reset_joint_positions())

    dynamic_instances = tuple(
        item for item in config.scene.objects if item.name == config.task.dynamic_object
    )
    if len(dynamic_instances) != 1:
        raise ValueError("task.dynamic_object must resolve to exactly one scene object")
    dynamic_object = dynamic_instances[0]

    generator = derive_reset_generator_description(
        seed=seed,
        sample_id=sample_id,
        reset_index=reset_index,
        joint_count=len(nominal_joint_positions),
    )
    draw_count = len(nominal_joint_positions) + 6
    counter_before = generator.rng_counter_after - draw_count

    first_draws, counter = _uniform_values(
        key=generator.rng_key,
        counter=counter_before,
        columns=len(nominal_joint_positions) + 5,
    )
    goal_yaw_draws, counter = _uniform_values(
        key=generator.rng_key,
        counter=counter,
        columns=1,
    )
    if counter != generator.rng_counter_after:  # pragma: no cover
        raise RuntimeError("reset sampler RNG layout is inconsistent")

    randomization = config.task.randomization
    joint_positions = tuple(
        _add_f32(
            nominal,
            _uniform_range(unit, randomization.robot_joint_delta_rad),
        )
        for nominal, unit in zip(
            nominal_joint_positions,
            first_draws[: len(nominal_joint_positions)],
            strict=True,
        )
    )
    offset = len(nominal_joint_positions)

    nominal_position = dynamic_object.root_pose.xyz
    block_position = (
        _add_f32(
            nominal_position[0],
            _uniform_range(first_draws[offset], randomization.block_x_delta_m),
        ),
        _add_f32(
            nominal_position[1],
            _uniform_range(first_draws[offset + 1], randomization.block_y_delta_m),
        ),
        _float32(nominal_position[2]),
    )
    reset_yaw = _uniform_range(
        first_draws[offset + 2], randomization.block_yaw_delta_rad
    )
    nominal_orientation = _quaternion_from_rpy_xyz(dynamic_object.root_pose.rpy)
    yaw_orientation = (
        _float32(math.cos(0.5 * reset_yaw)),
        0.0,
        0.0,
        _float32(math.sin(0.5 * reset_yaw)),
    )
    block_orientation = _normalize_quaternion(
        _quaternion_multiply(yaw_orientation, nominal_orientation)
    )
    nominal_heading = _planar_heading(
        nominal_orientation,
        config.task.heading_axis,
    )
    reset_heading = _wrap_to_pi(
        _planar_heading(block_orientation, config.task.heading_axis) - nominal_heading
    )

    goal_position = (
        _add_f32(
            block_position[0],
            _uniform_range(first_draws[offset + 3], randomization.goal_x_delta_m),
        ),
        _add_f32(
            block_position[1],
            _uniform_range(first_draws[offset + 4], randomization.goal_y_delta_m),
        ),
        block_position[2],
    )
    goal_yaw = _float32(
        _wrap_to_pi(
            reset_heading
            + _uniform_range(goal_yaw_draws[0], randomization.goal_yaw_delta_rad)
        )
    )

    robots: list[ResetRobotDescription] = []
    joint_offset = 0
    for robot, names in zip(
        config.scene.robots,
        joint_names_by_robot,
        strict=True,
    ):
        width = len(names)
        positions = joint_positions[joint_offset : joint_offset + width]
        joint_offset += width
        zeros = (0.0,) * width
        robots.append(
            ResetRobotDescription(
                label=robot.label,
                robot_profile=robot.robot_profile,
                joint_names=names,
                joint_positions=positions,
                joint_velocities=zeros,
                joint_targets=positions,
            )
        )

    return KaleidoscopeResetSceneDescription(
        scene_id=config.scene.scene_id,
        task_id=config.task.task_id,
        compatibility_fingerprint=semantic_config_fingerprint(config),
        generator=generator,
        robots=tuple(robots),
        dynamic_object=ResetObjectDescription(
            name=dynamic_object.name,
            object_profile=dynamic_object.object_profile,
            position_xyz=block_position,
            orientation_wxyz=block_orientation,
            com_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        task=ResetTaskDescription(
            goal_position=goal_position,
            goal_yaw=goal_yaw,
        ),
    )


def derive_reset_generator_description(
    *,
    seed: int,
    sample_id: int,
    reset_index: int,
    joint_count: int,
) -> ResetGeneratorDescription:
    """Derive the canonical v1 RNG identity and continuation metadata."""

    normalized_seed = _signed_int64(seed, "seed")
    normalized_sample_id = _nonnegative_int64(sample_id, "sample_id")
    normalized_reset_index = _nonnegative_int64(reset_index, "reset_index")
    if type(joint_count) is not int or joint_count < 1:
        raise ValueError("joint_count must be a positive int")
    draw_count = joint_count + 6
    counter_after = (normalized_reset_index + 1) * draw_count
    if counter_after > _SIGNED_INT64_MAX:
        raise ValueError("reset_index exceeds the signed int64 RNG counter range")
    return ResetGeneratorDescription(
        algorithm=RESET_SAMPLING_ALGORITHM,
        seed=normalized_seed,
        sample_id=normalized_sample_id,
        reset_index=normalized_reset_index,
        rng_key=_wrap_signed_int64(
            normalized_seed + normalized_sample_id * _ENV_KEY_INCREMENT
        ),
        rng_counter_after=counter_after,
    )


def _uniform_values(
    *,
    key: int,
    counter: int,
    columns: int,
) -> tuple[tuple[float, ...], int]:
    values: list[float] = []
    for lane in range(columns):
        state = key + (counter + lane * _LANE_INCREMENT) * _COUNTER_MULTIPLIER
        state = (state * _LCG_MULTIPLIER + _LCG_INCREMENT) & 0x7FFFFFFF
        values.append(_float32(_float32(state) * (1.0 / 2147483648.0)))
    return tuple(values), counter + columns


def _uniform_range(unit: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    span = _float32(high - low)
    return _float32(_float32(unit * span) + _float32(low))


def _quaternion_from_rpy_xyz(
    rpy: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (0.5 * value for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return _normalize_quaternion(
        (
            _float32(cr * cp * cy + sr * sp * sy),
            _float32(sr * cp * cy - cr * sp * sy),
            _float32(cr * sp * cy + sr * cp * sy),
            _float32(cr * cp * sy - sr * sp * cy),
        )
    )


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        _float32(lw * rw - lx * rx - ly * ry - lz * rz),
        _float32(lw * rx + lx * rw + ly * rz - lz * ry),
        _float32(lw * ry - lx * rz + ly * rw + lz * rx),
        _float32(lw * rz + lx * ry - ly * rx + lz * rw),
    )


def _normalize_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("reset quaternion must be finite and non-zero")
    return tuple(_float32(component / norm) for component in value)  # type: ignore[return-value]


def _planar_heading(
    quaternion: tuple[float, float, float, float],
    body_axis: tuple[float, float, float],
) -> float:
    w, x, y, z = quaternion
    vx, vy, vz = body_axis
    cross_x = y * vz - z * vy
    cross_y = z * vx - x * vz
    cross_z = x * vy - y * vx
    inner_x = cross_x + w * vx
    inner_y = cross_y + w * vy
    inner_z = cross_z + w * vz
    rotated_x = vx + 2.0 * (y * inner_z - z * inner_y)
    rotated_y = vy + 2.0 * (z * inner_x - x * inner_z)
    projection_norm = math.hypot(rotated_x, rotated_y)
    if not math.isfinite(projection_norm) or projection_norm <= 1.0e-6:
        raise ValueError("task heading axis has no finite planar projection")
    return math.atan2(rotated_y, rotated_x)


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _add_f32(left: float, right: float) -> float:
    return _float32(_float32(left) + _float32(right))


def _float32(value: int | float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _signed_int64(value: object, label: str) -> int:
    if type(value) is not int or not _SIGNED_INT64_MIN <= value <= _SIGNED_INT64_MAX:
        raise ValueError(f"{label} must be a signed int64")
    return value


def _nonnegative_int64(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _SIGNED_INT64_MAX:
        raise ValueError(f"{label} must be a non-negative signed int64")
    return value


def _wrap_signed_int64(value: int) -> int:
    unsigned = value % _UINT64_MODULUS
    if unsigned > _SIGNED_INT64_MAX:
        return unsigned - _UINT64_MODULUS
    return unsigned


__all__ = [
    "RESET_SAMPLING_ALGORITHM",
    "derive_reset_generator_description",
    "sample_reset_scene_description",
]
