"""Portable reset inputs and CPU-only seed sampling, without Torch, CUDA or Kit.

A template records the imported nominal pose once. A description starts a new
episode; it is deliberately not an arbitrary-time solver checkpoint.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Real
from typing import Any, cast

import numpy as np

from linkerbot_sim.configuration.fingerprint import (
    semantic_config_fingerprint,
    semantic_config_payload,
)
from linkerbot_sim.configuration.modes.kaleidoscope import KaleidoscopeConfig
from linkerbot_sim.configuration.tasks.kaleidoscope import RandomizationSettings
from linkerbot_sim.kaleidoscope.reset_math import (
    reset_rng_bits,
    reset_rng_keys,
    uniform_range,
)
from linkerbot_sim.utils.paths import repo_path


def _vector(values, length: int, name: str) -> tuple[float, ...]:
    if (
        not isinstance(values, (tuple, list))
        or len(values) != length
        or any(isinstance(v, bool) or not isinstance(v, Real) for v in values)
        or not all(
            math.isfinite(v) and abs(v) <= float(np.finfo(np.float32).max)
            for v in values
        )
    ):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(float(v) for v in values)


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} must be an int64 >= {minimum}")
    return value


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc


def _fields(data: Mapping, cls: type) -> dict:
    if not isinstance(data, Mapping) or set(data) != {f.name for f in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    return dict(data)


def _unit(values, length: int, name: str) -> tuple[float, ...]:
    result = _vector(values, length, name)
    if abs(sum(v * v for v in result) - 1.0) > 1.0e-5:
        raise ValueError(f"{name} must have unit norm")
    return result


@dataclass(frozen=True)
class KaleidoscopeResetTemplate:
    configuration_fingerprint: str
    joint_names: tuple[str, ...]
    nominal_joint_positions: tuple[float, ...]
    nominal_block_position: tuple[float, ...]
    nominal_block_orientation_wxyz: tuple[float, ...]
    heading_axis: tuple[float, ...]
    randomization: RandomizationSettings
    control_mode: str = "position"
    schema: str = "linkerbot.kaleidoscope-reset-template.v1"

    def __post_init__(self):
        if self.schema != "linkerbot.kaleidoscope-reset-template.v1":
            raise ValueError("unsupported reset template schema")
        _digest(self.configuration_fingerprint, "configuration fingerprint")
        if self.control_mode not in {"position", "velocity", "effort"}:
            raise ValueError("invalid reset control mode")
        if not isinstance(self.joint_names, (tuple, list)):
            raise ValueError("joint_names must be an array")
        names = tuple(self.joint_names)
        if not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("joint_names must contain nonempty names")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        object.__setattr__(self, "joint_names", names)
        for name, size in (
            ("nominal_joint_positions", len(names)),
            ("nominal_block_position", 3),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))
        for name, size in (("nominal_block_orientation_wxyz", 4), ("heading_axis", 3)):
            object.__setattr__(self, name, _unit(getattr(self, name), size, name))
        if not isinstance(self.randomization, RandomizationSettings):
            raise TypeError("randomization must be RandomizationSettings")
        # Reuse the configuration parser's finite/order validation.
        object.__setattr__(
            self,
            "randomization",
            RandomizationSettings.from_mapping(
                asdict(self.randomization), label="randomization"
            ),
        )
        _heading(
            np.asarray([self.nominal_block_orientation_wxyz], dtype=np.float32),
            np.asarray(self.heading_axis, dtype=np.float32),
        )

    @property
    def fingerprint(self) -> str:
        return semantic_config_fingerprint(self)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping) -> "KaleidoscopeResetTemplate":
        values = _fields(data, cls)
        values["randomization"] = RandomizationSettings.from_mapping(
            values["randomization"], label="randomization"
        )
        return cls(**values)


@dataclass(frozen=True)
class KaleidoscopeResetRow:
    source_env_id: int
    rng_key: int
    rng_counter: int
    joint_positions: tuple[float, ...]
    block_position: tuple[float, ...]
    block_orientation_wxyz: tuple[float, ...]
    goal_position: tuple[float, ...]
    goal_yaw: float

    def __post_init__(self):
        _integer(self.source_env_id, "source_env_id")
        _integer(self.rng_key, "rng_key", minimum=-(1 << 63))
        _integer(self.rng_counter, "rng_counter")
        if (
            not isinstance(self.joint_positions, (tuple, list))
            or not self.joint_positions
        ):
            raise ValueError("joint_positions must be nonempty")
        for name, size in (
            ("joint_positions", len(self.joint_positions)),
            ("block_position", 3),
            ("goal_position", 3),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))
        object.__setattr__(
            self,
            "block_orientation_wxyz",
            _unit(self.block_orientation_wxyz, 4, "block_orientation_wxyz"),
        )
        object.__setattr__(self, "goal_yaw", _vector([self.goal_yaw], 1, "goal_yaw")[0])


@dataclass(frozen=True)
class KaleidoscopeResetDescription:
    template_fingerprint: str
    rows: tuple[KaleidoscopeResetRow, ...]
    schema: str = "linkerbot.kaleidoscope-reset-description.v1"

    def __post_init__(self):
        if self.schema != "linkerbot.kaleidoscope-reset-description.v1":
            raise ValueError("unsupported reset description schema")
        _digest(self.template_fingerprint, "template fingerprint")
        rows = tuple(self.rows)
        if not rows or any(not isinstance(r, KaleidoscopeResetRow) for r in rows):
            raise ValueError("description requires reset rows")
        if len({len(r.joint_positions) for r in rows}) != 1:
            raise ValueError("reset row joint widths must match")
        object.__setattr__(self, "rows", rows)

    def as_dict(self) -> dict:
        return asdict(self)

    def validate_template(self, template: KaleidoscopeResetTemplate) -> None:
        """Preflight compatibility and heading before any physical/RNG write."""
        if self.template_fingerprint != template.fingerprint:
            raise ValueError(
                "reset description template/configuration/control mode mismatch"
            )
        if any(
            len(row.joint_positions) != len(template.joint_names) for row in self.rows
        ):
            raise ValueError("reset description joint width mismatch")
        _heading(
            np.asarray([r.block_orientation_wxyz for r in self.rows], dtype=np.float32),
            np.asarray(template.heading_axis, dtype=np.float32),
        )

    @classmethod
    def from_mapping(cls, data: Mapping) -> "KaleidoscopeResetDescription":
        values = _fields(data, cls)
        if not isinstance(values["rows"], (tuple, list)):
            raise ValueError("rows must be an array")
        values["rows"] = tuple(
            KaleidoscopeResetRow(**_fields(row, KaleidoscopeResetRow))
            for row in values["rows"]
        )
        return cls(**values)


def reset_configuration_fingerprint(config: KaleidoscopeConfig) -> str:
    """Cold identity: semantic config and root asset bytes, excluding GPU/batch size.

    Referenced meshes/textures must still come from the same asset revision. This is
    not a replacement for installing/pinning the complete asset package.
    """
    payload = cast(dict[str, Any], semantic_config_payload(config))
    payload.pop("compute")
    payload["environments"].pop("num_envs")
    paths = []
    for instance in (*config.scene.robots, *config.scene.objects):
        profile = instance.resolved_profile
        assert profile is not None  # Strict configuration has already bound profiles.
        paths.append(profile.asset_path)
    roots = {}
    for path in sorted(set(paths)):
        with repo_path(path).open("rb") as stream:
            roots[path] = hashlib.file_digest(stream, "sha256").hexdigest()
    return semantic_config_fingerprint({"configuration": payload, "asset_roots": roots})


def make_reset_template(
    config: KaleidoscopeConfig,
    *,
    nominal_joint_positions: Sequence[float],
    control_mode: str = "position",
) -> KaleidoscopeResetTemplate:
    """Build from explicit imported nominal joints; no inferred zero/home pose."""
    from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz

    if config.task.task_id != "tblock_push_v1":
        raise ValueError("portable reset only supports tblock_push_v1")
    block = next(
        o for o in config.scene.objects if o.name == config.task.dynamic_object
    )
    joint_names = []
    for robot in config.scene.robots:
        profile = robot.resolved_profile
        assert profile is not None
        joint_names.extend(
            f"{robot.label}/{name}" for name in profile.controlled_joints
        )
    return KaleidoscopeResetTemplate(
        configuration_fingerprint=reset_configuration_fingerprint(config),
        joint_names=tuple(joint_names),
        nominal_joint_positions=tuple(nominal_joint_positions),
        nominal_block_position=tuple(
            np.asarray(block.root_pose.xyz, dtype=np.float32).tolist()
        ),
        nominal_block_orientation_wxyz=tuple(
            rpy_xyz_to_quat_wxyz(block.root_pose.rpy).astype(np.float32).tolist()
        ),
        heading_axis=config.task.heading_axis,
        randomization=config.task.randomization,
        control_mode=control_mode,
    )


def _heading(quaternion: np.ndarray, axis: np.ndarray) -> np.ndarray:
    q = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    axis = axis / np.linalg.norm(axis)
    imaginary = q[:, 1:]
    rotated = axis + 2 * np.cross(
        imaginary, np.cross(imaginary, axis) + q[:, :1] * axis
    )
    if not np.all(np.linalg.norm(rotated[:, :2], axis=-1) > 1.0e-6):
        raise ValueError("block orientation has no finite planar heading")
    return np.arctan2(rotated[:, 1], rotated[:, 0])


def sample_reset_description(
    template: KaleidoscopeResetTemplate,
    *,
    seed: int,
    source_env_ids: Sequence[int] = (0,),
    reset_index: int = 0,
) -> KaleidoscopeResetDescription:
    """Pure float32 CPU sampling; index 0 is the first reset after reseed.

    Source IDs select logical RNG streams, not destination rows. The resulting RNG
    key/counter travels with each row when restoring into a different environment.
    """
    _integer(seed, "seed", minimum=-(1 << 63))
    _integer(reset_index, "reset_index")
    ids = tuple(source_env_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("source_env_ids must be nonempty and unique")
    for value in ids:
        _integer(value, "source_env_id")
    width = len(template.joint_names)
    counter = reset_index * (width + 6)
    _integer(counter + width + 6, "post-reset counter")
    keys = reset_rng_keys(np.asarray(ids, dtype=np.int64), seed)

    def uniform(start: int, columns: int) -> np.ndarray:
        bits = reset_rng_bits(
            keys[:, None], np.int64(start), np.arange(columns, dtype=np.int64)[None, :]
        )
        return bits.astype(np.float32) * np.float32(1.0 / 2147483648.0)

    random = uniform(counter, width + 5)
    ranges = template.randomization
    q = np.asarray(template.nominal_joint_positions, dtype=np.float32)[
        None, :
    ] + uniform_range(random[:, :width], ranges.robot_joint_delta_rad)
    p = np.tile(
        np.asarray(template.nominal_block_position, dtype=np.float32), (len(ids), 1)
    )
    p[:, 0] += uniform_range(random[:, width], ranges.block_x_delta_m)
    p[:, 1] += uniform_range(random[:, width + 1], ranges.block_y_delta_m)
    yaw = uniform_range(random[:, width + 2], ranges.block_yaw_delta_rad)
    w, x, y, z = np.asarray(template.nominal_block_orientation_wxyz, dtype=np.float32)
    c, s = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    quat = np.stack(
        (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w), axis=-1
    )
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    goal = p.copy()
    goal[:, 0] += uniform_range(random[:, width + 3], ranges.goal_x_delta_m)
    goal[:, 1] += uniform_range(random[:, width + 4], ranges.goal_y_delta_m)
    goal_delta = uniform_range(
        uniform(counter + width + 5, 1)[:, 0], ranges.goal_yaw_delta_rad
    )
    axis = np.asarray(template.heading_axis, dtype=np.float32)
    heading = _heading(quat, axis) - _heading(
        np.asarray([template.nominal_block_orientation_wxyz], dtype=np.float32), axis
    )
    heading = np.arctan2(np.sin(heading), np.cos(heading)) + goal_delta
    goal_yaw = np.arctan2(np.sin(heading), np.cos(heading))
    return KaleidoscopeResetDescription(
        template_fingerprint=template.fingerprint,
        rows=tuple(
            KaleidoscopeResetRow(
                source_env_id=source,
                rng_key=int(keys[row]),
                rng_counter=counter + width + 6,
                joint_positions=tuple(q[row].tolist()),
                block_position=tuple(p[row].tolist()),
                block_orientation_wxyz=tuple(quat[row].tolist()),
                goal_position=tuple(goal[row].tolist()),
                goal_yaw=float(goal_yaw[row]),
            )
            for row, source in enumerate(ids)
        ),
    )


def reset_json(value: KaleidoscopeResetTemplate | KaleidoscopeResetDescription) -> str:
    """Explicit portable JSON boundary; callers choose paths/transport themselves."""
    return (
        json.dumps(value.as_dict(), ensure_ascii=False, allow_nan=False, indent=2)
        + "\n"
    )
