"""Pure, versioned reset-scene descriptions for Kaleidoscope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
import re


RESET_SCENE_DESCRIPTION_SCHEMA = "linkerbot.kaleidoscope-reset.v1"
_SIGNED_INT64_MIN = -(2**63)
_SIGNED_INT64_MAX = 2**63 - 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ResetGeneratorDescription:
    """Deterministic identity and continuation state for one reset sample."""

    algorithm: str
    seed: int
    sample_id: int
    reset_index: int
    rng_key: int
    rng_counter_after: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "algorithm", _nonempty_string(self.algorithm, "generator.algorithm")
        )
        object.__setattr__(self, "seed", _signed_int64(self.seed, "generator.seed"))
        object.__setattr__(
            self,
            "sample_id",
            _nonnegative_int64(self.sample_id, "generator.sample_id"),
        )
        object.__setattr__(
            self,
            "reset_index",
            _nonnegative_int64(self.reset_index, "generator.reset_index"),
        )
        object.__setattr__(
            self, "rng_key", _signed_int64(self.rng_key, "generator.rng_key")
        )
        object.__setattr__(
            self,
            "rng_counter_after",
            _nonnegative_int64(self.rng_counter_after, "generator.rng_counter_after"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ResetGeneratorDescription:
        data = _mapping(value, "generator")
        _exact_keys(
            data,
            {
                "algorithm",
                "seed",
                "sample_id",
                "reset_index",
                "rng_key",
                "rng_counter_after",
            },
            "generator",
        )
        return cls(
            algorithm=data["algorithm"],  # type: ignore[arg-type]
            seed=data["seed"],  # type: ignore[arg-type]
            sample_id=data["sample_id"],  # type: ignore[arg-type]
            reset_index=data["reset_index"],  # type: ignore[arg-type]
            rng_key=data["rng_key"],  # type: ignore[arg-type]
            rng_counter_after=data["rng_counter_after"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "sample_id": self.sample_id,
            "reset_index": self.reset_index,
            "rng_key": self.rng_key,
            "rng_counter_after": self.rng_counter_after,
        }


@dataclass(frozen=True, slots=True)
class ResetRobotDescription:
    """Logical command-joint reset state for one robot."""

    label: str
    robot_profile: str
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    joint_targets: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _nonempty_string(self.label, "robot.label"))
        object.__setattr__(
            self,
            "robot_profile",
            _nonempty_string(self.robot_profile, "robot.robot_profile"),
        )
        names = _names(self.joint_names, "robot.joint_names")
        object.__setattr__(self, "joint_names", names)
        for field_name in (
            "joint_positions",
            "joint_velocities",
            "joint_targets",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_vector(
                    getattr(self, field_name),
                    len(names),
                    f"robot.{field_name}",
                ),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ResetRobotDescription:
        data = _mapping(value, "robot")
        _exact_keys(
            data,
            {
                "label",
                "robot_profile",
                "joint_names",
                "joint_positions",
                "joint_velocities",
                "joint_targets",
            },
            "robot",
        )
        return cls(
            label=data["label"],  # type: ignore[arg-type]
            robot_profile=data["robot_profile"],  # type: ignore[arg-type]
            joint_names=data["joint_names"],  # type: ignore[arg-type]
            joint_positions=data["joint_positions"],  # type: ignore[arg-type]
            joint_velocities=data["joint_velocities"],  # type: ignore[arg-type]
            joint_targets=data["joint_targets"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "robot_profile": self.robot_profile,
            "joint_names": list(self.joint_names),
            "joint_positions": list(self.joint_positions),
            "joint_velocities": list(self.joint_velocities),
            "joint_targets": list(self.joint_targets),
        }


@dataclass(frozen=True, slots=True)
class ResetObjectDescription:
    """Environment-local reset state for the task's dynamic rigid object."""

    name: str
    object_profile: str
    position_xyz: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    com_velocity: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "object.name"))
        object.__setattr__(
            self,
            "object_profile",
            _nonempty_string(self.object_profile, "object.object_profile"),
        )
        object.__setattr__(
            self,
            "position_xyz",
            _finite_vector(self.position_xyz, 3, "object.position_xyz"),
        )
        orientation = _finite_vector(
            self.orientation_wxyz, 4, "object.orientation_wxyz"
        )
        norm = math.sqrt(sum(component * component for component in orientation))
        if norm <= 1.0e-12:
            raise ValueError("object.orientation_wxyz must be non-zero")
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("object.orientation_wxyz must be normalized")
        object.__setattr__(self, "orientation_wxyz", orientation)
        object.__setattr__(
            self,
            "com_velocity",
            _finite_vector(self.com_velocity, 6, "object.com_velocity"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ResetObjectDescription:
        data = _mapping(value, "dynamic_object")
        _exact_keys(
            data,
            {
                "name",
                "object_profile",
                "position_xyz",
                "orientation_wxyz",
                "com_velocity",
            },
            "dynamic_object",
        )
        return cls(
            name=data["name"],  # type: ignore[arg-type]
            object_profile=data["object_profile"],  # type: ignore[arg-type]
            position_xyz=data["position_xyz"],  # type: ignore[arg-type]
            orientation_wxyz=data["orientation_wxyz"],  # type: ignore[arg-type]
            com_velocity=data["com_velocity"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "object_profile": self.object_profile,
            "position_xyz": list(self.position_xyz),
            "orientation_wxyz": list(self.orientation_wxyz),
            "com_velocity": list(self.com_velocity),
        }


@dataclass(frozen=True, slots=True)
class ResetTaskDescription:
    """Task-owned values that define the freshly reset objective."""

    goal_position: tuple[float, float, float]
    goal_yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "goal_position",
            _finite_vector(self.goal_position, 3, "task.goal_position"),
        )
        object.__setattr__(
            self, "goal_yaw", _finite_float(self.goal_yaw, "task.goal_yaw")
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ResetTaskDescription:
        data = _mapping(value, "task")
        _exact_keys(data, {"goal_position", "goal_yaw"}, "task")
        return cls(
            goal_position=data["goal_position"],  # type: ignore[arg-type]
            goal_yaw=data["goal_yaw"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "goal_position": list(self.goal_position),
            "goal_yaw": self.goal_yaw,
        }


@dataclass(frozen=True, slots=True)
class KaleidoscopeResetSceneDescription:
    """Portable fresh-episode state without physics-backend-private fields."""

    scene_id: str
    task_id: str
    compatibility_fingerprint: str
    generator: ResetGeneratorDescription
    robots: tuple[ResetRobotDescription, ...]
    dynamic_object: ResetObjectDescription
    task: ResetTaskDescription
    coordinate_frame: str = "env-local"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scene_id", _nonempty_string(self.scene_id, "scene_id")
        )
        object.__setattr__(self, "task_id", _nonempty_string(self.task_id, "task_id"))
        fingerprint = _nonempty_string(
            self.compatibility_fingerprint, "compatibility_fingerprint"
        )
        if _SHA256_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError("compatibility_fingerprint must be a lowercase SHA-256")
        object.__setattr__(self, "compatibility_fingerprint", fingerprint)
        if self.coordinate_frame != "env-local":
            raise ValueError("coordinate_frame must be 'env-local'")
        if not isinstance(self.generator, ResetGeneratorDescription):
            raise TypeError("generator must be a ResetGeneratorDescription")
        robots = tuple(self.robots)
        if not robots:
            raise ValueError("robots cannot be empty")
        if not all(isinstance(robot, ResetRobotDescription) for robot in robots):
            raise TypeError("robots must contain ResetRobotDescription values")
        labels = tuple(robot.label for robot in robots)
        if len(set(labels)) != len(labels):
            raise ValueError("robot labels must be unique")
        object.__setattr__(self, "robots", robots)
        if not isinstance(self.dynamic_object, ResetObjectDescription):
            raise TypeError("dynamic_object must be a ResetObjectDescription")
        if not isinstance(self.task, ResetTaskDescription):
            raise TypeError("task must be a ResetTaskDescription")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> KaleidoscopeResetSceneDescription:
        data = _mapping(value, "reset description")
        _exact_keys(
            data,
            {
                "schema",
                "scene_id",
                "task_id",
                "compatibility_fingerprint",
                "coordinate_frame",
                "generator",
                "robots",
                "dynamic_object",
                "task",
            },
            "reset description",
        )
        if data["schema"] != RESET_SCENE_DESCRIPTION_SCHEMA:
            raise ValueError("unsupported Kaleidoscope reset description schema")
        robot_values = _sequence(data["robots"], "robots")
        return cls(
            scene_id=data["scene_id"],  # type: ignore[arg-type]
            task_id=data["task_id"],  # type: ignore[arg-type]
            compatibility_fingerprint=data[  # type: ignore[arg-type]
                "compatibility_fingerprint"
            ],
            coordinate_frame=data["coordinate_frame"],  # type: ignore[arg-type]
            generator=ResetGeneratorDescription.from_mapping(
                _mapping(data["generator"], "generator")
            ),
            robots=tuple(
                ResetRobotDescription.from_mapping(_mapping(item, "robot"))
                for item in robot_values
            ),
            dynamic_object=ResetObjectDescription.from_mapping(
                _mapping(data["dynamic_object"], "dynamic_object")
            ),
            task=ResetTaskDescription.from_mapping(_mapping(data["task"], "task")),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": RESET_SCENE_DESCRIPTION_SCHEMA,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "coordinate_frame": self.coordinate_frame,
            "generator": self.generator.as_dict(),
            "robots": [robot.as_dict() for robot in self.robots],
            "dynamic_object": self.dynamic_object.as_dict(),
            "task": self.task.as_dict(),
        }


def validate_reset_scene_description(
    value: KaleidoscopeResetSceneDescription | Mapping[str, object],
) -> KaleidoscopeResetSceneDescription:
    """Validate a typed value or mapping and return an independently owned copy."""

    if isinstance(value, KaleidoscopeResetSceneDescription):
        return KaleidoscopeResetSceneDescription.from_mapping(value.as_dict())
    return KaleidoscopeResetSceneDescription.from_mapping(
        _mapping(value, "reset description")
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value  # type: ignore[return-value]


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _exact_keys(data: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{label} keys differ: missing={missing}, unknown={unknown}")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _signed_int64(value: object, label: str) -> int:
    if type(value) is not int or not (_SIGNED_INT64_MIN <= value <= _SIGNED_INT64_MAX):
        raise ValueError(f"{label} must be a signed 64-bit integer")
    return value


def _nonnegative_int64(value: object, label: str) -> int:
    result = _signed_int64(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _names(value: object, label: str) -> tuple[str, ...]:
    names = tuple(_nonempty_string(item, label) for item in _sequence(value, label))
    if not names:
        raise ValueError(f"{label} cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicates")
    return names


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _finite_vector(value: object, length: int, label: str) -> tuple[float, ...]:
    items = _sequence(value, label)
    if len(items) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return tuple(_finite_float(item, label) for item in items)


__all__ = [
    "KaleidoscopeResetSceneDescription",
    "RESET_SCENE_DESCRIPTION_SCHEMA",
    "ResetGeneratorDescription",
    "ResetObjectDescription",
    "ResetRobotDescription",
    "ResetTaskDescription",
    "validate_reset_scene_description",
]
