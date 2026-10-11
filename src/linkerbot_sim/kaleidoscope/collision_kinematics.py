"""Full arm-and-hand kinematics topology for collision queries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from linkerbot_sim.kaleidoscope.collision_catalog import RobotCollisionCatalog
from linkerbot_sim.utils.paths import repo_path


_MOVING_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic"})
_SUPPORTED_JOINT_TYPES = _MOVING_JOINT_TYPES | {"fixed"}


@dataclass(frozen=True, slots=True)
class ArticulatedMimicSpec:
    """One dependent URDF joint driven by another joint."""

    source_joint: str
    multiplier: float
    offset: float


@dataclass(frozen=True, slots=True)
class ArticulatedJointSpec:
    """Topology and motion facts required for one URDF joint."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    mimic: ArticulatedMimicSpec | None


@dataclass(frozen=True, slots=True)
class FullArticulatedKinematicsSpec:
    """Validated complete arm-and-hand URDF topology.

    ``joints`` is stored in parent-before-child order. ``command_joint_names``
    retains the articulation command order and excludes mimic followers.
    """

    urdf_path: Path
    root_link: str
    link_names: tuple[str, ...]
    joints: tuple[ArticulatedJointSpec, ...]
    command_joint_names: tuple[str, ...]
    required_link_names: tuple[str, ...]

    @property
    def mimic_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints if joint.mimic is not None)

    @property
    def moving_joint_names(self) -> tuple[str, ...]:
        return tuple(
            joint.name
            for joint in self.joints
            if joint.joint_type in _MOVING_JOINT_TYPES
        )


@dataclass(frozen=True, slots=True)
class ArticulatedLinkTransformBatch:
    """Batch of Gf row-major link-local to robot-root transforms.

    ``matrices`` has shape ``[environments, samples, links, 4, 4]``.  The
    contained NumPy allocation is read-only so callers cannot mutate a result
    after validation.
    """

    link_names: tuple[str, ...]
    matrices: np.ndarray

    def __post_init__(self) -> None:
        if not self.link_names or len(set(self.link_names)) != len(self.link_names):
            raise ValueError("link_names must be non-empty and unique")
        if not isinstance(self.matrices, np.ndarray):
            raise TypeError("matrices must be a NumPy array")
        if self.matrices.ndim != 5 or self.matrices.shape[2:] != (
            len(self.link_names),
            4,
            4,
        ):
            raise ValueError(
                "matrices must have shape [environments, samples, links, 4, 4]"
            )
        if self.matrices.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise TypeError("matrices must use float32 or float64")
        if not np.all(np.isfinite(self.matrices)):
            raise ValueError("matrices must contain only finite values")
        _validate_gf_affine_batch(self.matrices, label="candidate collider matrices")
        self.matrices.setflags(write=False)

    def matrices_for_link(self, link_name: str) -> np.ndarray:
        """Return a read-only ``[environments, samples, 4, 4]`` view."""

        try:
            index = self.link_names.index(link_name)
        except ValueError as exc:
            raise KeyError(f"unknown FK link: {link_name!r}") from exc
        return self.matrices[:, :, index]


@dataclass(frozen=True, slots=True)
class RobotRootWorldTransformBatch:
    """Exact env/robot/path identities paired with imported-root transforms."""

    env_ids: tuple[int, ...]
    robot_label: str
    imported_root_paths: tuple[str, ...]
    matrices: np.ndarray

    def __post_init__(self) -> None:
        if not self.env_ids or len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("env_ids must be non-empty and unique")
        if any(type(env_id) is not int or env_id < 0 for env_id in self.env_ids):
            raise ValueError("env_ids must contain non-negative ints")
        if not isinstance(self.robot_label, str) or not self.robot_label:
            raise ValueError("robot_label must be a non-empty string")
        if len(self.imported_root_paths) != len(self.env_ids) or any(
            not isinstance(path, str) or not path.startswith("/")
            for path in self.imported_root_paths
        ):
            raise ValueError(
                "imported_root_paths must contain one absolute path per env"
            )
        if len(set(self.imported_root_paths)) != len(self.imported_root_paths):
            raise ValueError("imported_root_paths must be globally unique")
        if not isinstance(self.matrices, np.ndarray):
            raise TypeError("matrices must be a NumPy array")
        if self.matrices.shape != (len(self.env_ids), 4, 4):
            raise ValueError("matrices must have shape [environments, 4, 4]")
        if self.matrices.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise TypeError("matrices must use float32 or float64")
        _validate_gf_affine_batch(self.matrices, label="robot root world matrices")
        self.matrices.setflags(write=False)


@dataclass(frozen=True, slots=True)
class CandidateColliderTransformBatch:
    """Candidate Gf world transforms for exact replicated robot colliders."""

    env_ids: tuple[int, ...]
    collision_suffixes: tuple[str, ...]
    collision_paths: tuple[tuple[str, ...], ...]
    matrices: np.ndarray

    def __post_init__(self) -> None:
        if not self.env_ids or len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("env_ids must be non-empty and unique")
        if not self.collision_suffixes or len(set(self.collision_suffixes)) != len(
            self.collision_suffixes
        ):
            raise ValueError("collision_suffixes must be non-empty and unique")
        expected_paths = (len(self.env_ids), len(self.collision_suffixes))
        if len(self.collision_paths) != expected_paths[0] or any(
            len(row) != expected_paths[1] for row in self.collision_paths
        ):
            raise ValueError(
                "collision_paths must have shape [environments, colliders]"
            )
        if len({path for row in self.collision_paths for path in row}) != (
            expected_paths[0] * expected_paths[1]
        ):
            raise ValueError("collision_paths must be globally unique")
        if not isinstance(self.matrices, np.ndarray):
            raise TypeError("matrices must be a NumPy array")
        if self.matrices.ndim != 5 or self.matrices.shape[:1] != (len(self.env_ids),):
            raise ValueError(
                "matrices must have shape [environments, samples, colliders, 4, 4]"
            )
        if self.matrices.shape[2:] != (len(self.collision_suffixes), 4, 4):
            raise ValueError(
                "matrices must have shape [environments, samples, colliders, 4, 4]"
            )
        if self.matrices.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise TypeError("matrices must use float32 or float64")
        if not np.all(np.isfinite(self.matrices)):
            raise ValueError("matrices must contain only finite values")
        self.matrices.setflags(write=False)


class FullArticulatedForwardKinematics:
    """Pure batched reference FK for the complete arm-and-hand URDF tree.

    This implementation deliberately has no Isaac, PhysX, cuRobo, Torch, or
    runtime dependency.  It establishes the numerical convention that a later
    CUDA adapter and the real-stage comparison smoke must reproduce.
    """

    def __init__(self, spec: FullArticulatedKinematicsSpec) -> None:
        if not isinstance(spec, FullArticulatedKinematicsSpec):
            raise TypeError("spec must be FullArticulatedKinematicsSpec")
        self.spec = spec
        self._joint_by_name = {joint.name: joint for joint in spec.joints}

    def compute(self, joint_positions: object) -> ArticulatedLinkTransformBatch:
        """Compute root-local transforms for ``[E, S, command joints]`` input."""

        positions = np.asarray(joint_positions)
        if positions.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise TypeError("joint_positions must use float32 or float64")
        if positions.ndim != 3:
            raise ValueError(
                "joint_positions must have shape [environments, samples, joints]"
            )
        if positions.shape[0] < 1 or positions.shape[1] < 1:
            raise ValueError(
                "joint_positions environment and sample axes cannot be empty"
            )
        if positions.shape[2] != len(self.spec.command_joint_names):
            raise ValueError(
                "joint_positions last dimension does not match command joint order"
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("joint_positions must contain only finite values")
        positions = np.ascontiguousarray(positions)
        command_values = {
            name: positions[:, :, index]
            for index, name in enumerate(self.spec.command_joint_names)
        }
        moving_values = dict(command_values)

        def resolve_joint_value(name: str, trail: frozenset[str]) -> np.ndarray:
            existing = moving_values.get(name)
            if existing is not None:
                return existing
            if name in trail:
                raise RuntimeError(f"mimic cycle escaped topology validation: {name!r}")
            joint = self._joint_by_name[name]
            if joint.mimic is None:
                raise RuntimeError(f"missing independent command joint value: {name!r}")
            source = resolve_joint_value(
                joint.mimic.source_joint,
                trail | {name},
            )
            value = source * joint.mimic.multiplier + joint.mimic.offset
            moving_values[name] = value
            return value

        shape = positions.shape[:2]
        identity = np.broadcast_to(
            np.eye(4, dtype=positions.dtype),
            (*shape, 4, 4),
        ).copy()
        column_transforms: dict[str, np.ndarray] = {self.spec.root_link: identity}
        for joint in self.spec.joints:
            parent = column_transforms[joint.parent_link]
            origin = _origin_matrix(joint, dtype=positions.dtype)
            if joint.joint_type == "fixed":
                local = origin
            else:
                value = resolve_joint_value(joint.name, frozenset())
                local = np.matmul(
                    origin,
                    _joint_motion_matrix(joint, value, dtype=positions.dtype),
                )
            column_transforms[joint.child_link] = np.matmul(parent, local)

        # URDF convention uses column vectors with translation in the last
        # column. Gf uses row vectors with translation in the last row, so the
        # complete matrix is transposed before it leaves this module.
        gf_transforms = np.stack(
            [
                np.swapaxes(column_transforms[link_name], -1, -2)
                for link_name in self.spec.link_names
            ],
            axis=2,
        )
        return ArticulatedLinkTransformBatch(
            link_names=self.spec.link_names,
            matrices=np.ascontiguousarray(gf_transforms),
        )


def compose_candidate_collider_world_transforms(
    *,
    robot_catalog: RobotCollisionCatalog,
    robot_label: str,
    kinematics_spec: FullArticulatedKinematicsSpec,
    link_transforms: ArticulatedLinkTransformBatch,
    robot_root_transforms: RobotRootWorldTransformBatch,
) -> CandidateColliderTransformBatch:
    """Compose collider-local, candidate-link, and imported-root transforms."""

    if not isinstance(robot_catalog, RobotCollisionCatalog):
        raise TypeError("robot_catalog must be RobotCollisionCatalog")
    if robot_label not in robot_catalog.robot_labels:
        raise KeyError(f"unknown robot label: {robot_label!r}")
    if not isinstance(kinematics_spec, FullArticulatedKinematicsSpec):
        raise TypeError("kinematics_spec must be FullArticulatedKinematicsSpec")
    if not isinstance(link_transforms, ArticulatedLinkTransformBatch):
        raise TypeError("link_transforms must be ArticulatedLinkTransformBatch")
    if link_transforms.link_names != kinematics_spec.link_names:
        raise ValueError("FK link order does not match kinematics spec")
    if not isinstance(robot_root_transforms, RobotRootWorldTransformBatch):
        raise TypeError("robot_root_transforms must be RobotRootWorldTransformBatch")
    if robot_root_transforms.robot_label != robot_label:
        raise ValueError("robot root transform label does not match requested robot")

    selected_envs = _env_ids(
        robot_root_transforms.env_ids,
        num_envs=robot_catalog.num_envs,
    )
    if link_transforms.matrices.shape[0] != len(selected_envs):
        raise ValueError("FK environment rows must match selected env_ids")
    roots = robot_root_transforms.matrices
    if roots.dtype != link_transforms.matrices.dtype:
        raise TypeError("robot root matrices and FK matrices must use the same dtype")

    per_env_shapes = tuple(
        robot_catalog.shapes_for(env_id=env_id, robot_label=robot_label)
        for env_id in selected_envs
    )
    canonical_suffixes = tuple(shape.collision_suffix for shape in per_env_shapes[0])
    link_indices: list[int] = []
    local_matrices: list[np.ndarray] = []
    links = set(kinematics_spec.link_names)
    for shape in per_env_shapes[0]:
        link_name = shape.rigid_body_path.rsplit("/", maxsplit=1)[-1]
        if link_name not in links:
            raise ValueError(
                "robot collider rigid-body name is absent from full URDF links: "
                f"{link_name!r}"
            )
        link_indices.append(kinematics_spec.link_names.index(link_name))
        local_matrices.append(
            np.asarray(
                shape.metadata.shape_to_rigid_body_matrix_row_major,
                dtype=roots.dtype,
            ).reshape(4, 4)
        )

    collision_paths: list[tuple[str, ...]] = []
    for row, (env_id, shapes) in enumerate(
        zip(selected_envs, per_env_shapes, strict=True)
    ):
        expected_root = shapes[0].imported_root_path
        actual_root = robot_root_transforms.imported_root_paths[row]
        if actual_root != expected_root:
            raise ValueError(
                f"robot imported root path mismatch for env {env_id}: "
                f"{actual_root!r} != {expected_root!r}"
            )
        suffixes = tuple(shape.collision_suffix for shape in shapes)
        if suffixes != canonical_suffixes:
            raise RuntimeError(
                "robot collider order changed after catalog validation for "
                f"env {env_id}"
            )
        for canonical, shape in zip(per_env_shapes[0], shapes, strict=True):
            expected_link = canonical.rigid_body_path.rsplit("/", maxsplit=1)[-1]
            actual_link = shape.rigid_body_path.rsplit("/", maxsplit=1)[-1]
            if actual_link != expected_link:
                raise RuntimeError(
                    f"robot collider rigid-body link changed in env {env_id}: "
                    f"{actual_link!r} != {expected_link!r}"
                )
        collision_paths.append(tuple(shape.collision_path for shape in shapes))

    local = np.stack(local_matrices, axis=0)
    candidate_links = link_transforms.matrices[:, :, link_indices]
    # Gf row-vector composition order:
    # shape-local -> rigid link -> imported robot root -> stage world.
    matrices = np.matmul(
        np.matmul(local[None, None, :, :, :], candidate_links),
        roots[:, None, None, :, :],
    )
    return CandidateColliderTransformBatch(
        env_ids=selected_envs,
        collision_suffixes=canonical_suffixes,
        collision_paths=tuple(collision_paths),
        matrices=np.ascontiguousarray(matrices),
    )


def full_articulated_urdf_path(robot_profile: object) -> Path:
    """Resolve the full URDF companion of the imported canonical robot asset."""

    asset_type = getattr(robot_profile, "asset_type", None)
    raw_asset_path = getattr(robot_profile, "asset_path", None)
    if not isinstance(asset_type, str) or asset_type not in {"mjcf", "urdf"}:
        raise ValueError("collision kinematics requires an MJCF or URDF robot asset")
    if not isinstance(raw_asset_path, str) or not raw_asset_path:
        raise ValueError("robot profile asset_path must be a non-empty string")
    imported_path = repo_path(raw_asset_path)
    urdf_path = (
        imported_path if asset_type == "urdf" else imported_path.with_suffix(".urdf")
    )
    if not imported_path.is_file():
        raise FileNotFoundError(f"imported robot asset does not exist: {imported_path}")
    if not urdf_path.is_file():
        raise FileNotFoundError(
            "full articulated URDF companion does not exist beside the imported "
            f"asset: {urdf_path}"
        )
    return urdf_path


def build_full_articulated_kinematics_spec(
    *,
    robot_profile: object,
    command_joint_names: Sequence[str],
    required_link_names: Sequence[str],
) -> FullArticulatedKinematicsSpec:
    """Parse and validate the complete arm-and-hand kinematics fact source."""

    urdf_path = full_articulated_urdf_path(robot_profile)
    commands = _unique_names(command_joint_names, label="command_joint_names")
    required_links = _unique_names(
        required_link_names,
        label="required_link_names",
        allow_empty=True,
    )
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(
            f"articulated kinematics source is not a URDF robot: {urdf_path}"
        )

    link_names = _urdf_link_names(root, urdf_path=urdf_path)
    joints = _urdf_joints(root, link_names=link_names, urdf_path=urdf_path)
    ordered_joints, root_link = _topological_joint_order(
        joints,
        link_names=link_names,
        urdf_path=urdf_path,
    )
    _validate_mimics(ordered_joints, urdf_path=urdf_path)

    missing_links = sorted(set(required_links) - set(link_names))
    if missing_links:
        raise ValueError(
            f"collision rigid-body links are absent from full URDF: {missing_links}"
        )
    independent = {
        joint.name
        for joint in ordered_joints
        if joint.joint_type in _MOVING_JOINT_TYPES and joint.mimic is None
    }
    if set(commands) != independent:
        missing = sorted(independent - set(commands))
        unexpected = sorted(set(commands) - independent)
        raise ValueError(
            "articulation command joints do not exactly match independent full-URDF "
            f"joints: missing={missing}, unexpected={unexpected}"
        )
    return FullArticulatedKinematicsSpec(
        urdf_path=urdf_path,
        root_link=root_link,
        link_names=link_names,
        joints=ordered_joints,
        command_joint_names=commands,
        required_link_names=required_links,
    )


def _urdf_link_names(root: ET.Element, *, urdf_path: Path) -> tuple[str, ...]:
    names = tuple(str(element.get("name", "")) for element in root.findall("link"))
    if not names or any(not name for name in names):
        raise ValueError(f"URDF links must have non-empty names: {urdf_path}")
    if len(set(names)) != len(names):
        raise ValueError(f"URDF contains duplicate link names: {urdf_path}")
    return names


def _urdf_joints(
    root: ET.Element,
    *,
    link_names: tuple[str, ...],
    urdf_path: Path,
) -> tuple[ArticulatedJointSpec, ...]:
    links = set(link_names)
    result: list[ArticulatedJointSpec] = []
    names: set[str] = set()
    children: set[str] = set()
    for element in root.findall("joint"):
        name = str(element.get("name", ""))
        joint_type = str(element.get("type", ""))
        if not name or name in names:
            raise ValueError(f"URDF joint names must be non-empty and unique: {name!r}")
        if joint_type not in _SUPPORTED_JOINT_TYPES:
            raise ValueError(f"unsupported URDF joint type {joint_type!r}: {name!r}")
        parent = element.find("parent")
        child = element.find("child")
        parent_link = "" if parent is None else str(parent.get("link", ""))
        child_link = "" if child is None else str(child.get("link", ""))
        if parent_link not in links or child_link not in links:
            raise ValueError(f"URDF joint {name!r} references an unknown link")
        if parent_link == child_link:
            raise ValueError(f"URDF joint {name!r} has the same parent and child")
        if child_link in children:
            raise ValueError(f"URDF link has multiple parent joints: {child_link!r}")

        origin = element.find("origin")
        origin_xyz = _vector3_attribute(origin, "xyz", default=(0.0, 0.0, 0.0))
        origin_rpy = _vector3_attribute(origin, "rpy", default=(0.0, 0.0, 0.0))
        axis = _vector3_attribute(
            element.find("axis"),
            "xyz",
            default=(1.0, 0.0, 0.0),
        )
        if joint_type in _MOVING_JOINT_TYPES:
            norm = math.sqrt(sum(value * value for value in axis))
            if norm == 0.0:
                raise ValueError(f"URDF moving joint {name!r} has a zero axis")
            axis = (axis[0] / norm, axis[1] / norm, axis[2] / norm)
        mimic_element = element.find("mimic")
        mimic = None
        if mimic_element is not None:
            if joint_type not in _MOVING_JOINT_TYPES:
                raise ValueError(f"fixed URDF joint {name!r} cannot define mimic")
            source = str(mimic_element.get("joint", ""))
            if not source:
                raise ValueError(f"URDF mimic joint {name!r} has no source joint")
            mimic = ArticulatedMimicSpec(
                source_joint=source,
                multiplier=_finite_float(
                    mimic_element.get("multiplier", "1"),
                    label=f"URDF mimic {name!r} multiplier",
                ),
                offset=_finite_float(
                    mimic_element.get("offset", "0"),
                    label=f"URDF mimic {name!r} offset",
                ),
            )
        names.add(name)
        children.add(child_link)
        result.append(
            ArticulatedJointSpec(
                name=name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis_xyz=axis,
                mimic=mimic,
            )
        )
    if not result:
        raise ValueError(f"URDF has no joints: {urdf_path}")
    return tuple(result)


def _topological_joint_order(
    joints: tuple[ArticulatedJointSpec, ...],
    *,
    link_names: tuple[str, ...],
    urdf_path: Path,
) -> tuple[tuple[ArticulatedJointSpec, ...], str]:
    child_links = {joint.child_link for joint in joints}
    roots = tuple(name for name in link_names if name not in child_links)
    if len(roots) != 1:
        raise ValueError(
            f"URDF must have exactly one root link, found {roots}: {urdf_path}"
        )
    resolved_links = {roots[0]}
    remaining = list(joints)
    ordered: list[ArticulatedJointSpec] = []
    while remaining:
        ready = [joint for joint in remaining if joint.parent_link in resolved_links]
        if not ready:
            unresolved = [joint.name for joint in remaining]
            raise ValueError(f"URDF joint tree cannot be resolved: {unresolved}")
        for joint in ready:
            ordered.append(joint)
            resolved_links.add(joint.child_link)
            remaining.remove(joint)
    if resolved_links != set(link_names):
        missing = sorted(set(link_names) - resolved_links)
        raise ValueError(f"URDF contains disconnected links: {missing}")
    return tuple(ordered), roots[0]


def _validate_mimics(
    joints: tuple[ArticulatedJointSpec, ...],
    *,
    urdf_path: Path,
) -> None:
    moving = {
        joint.name: joint for joint in joints if joint.joint_type in _MOVING_JOINT_TYPES
    }
    for joint in moving.values():
        if joint.mimic is None:
            continue
        source = joint.mimic.source_joint
        if source not in moving:
            raise ValueError(
                f"URDF mimic {joint.name!r} references unknown moving joint {source!r}"
            )
        visited = {joint.name}
        current = source
        while True:
            if current in visited:
                raise ValueError(
                    f"URDF mimic cycle detected at {current!r}: {urdf_path}"
                )
            visited.add(current)
            relation = moving[current].mimic
            if relation is None:
                break
            current = relation.source_joint
            if current not in moving:
                raise ValueError(
                    f"URDF mimic chain references unknown moving joint {current!r}"
                )


def _vector3_attribute(
    element: ET.Element | None,
    attribute: str,
    *,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if element is None or attribute not in element.attrib:
        return default
    parts = element.attrib[attribute].split()
    if len(parts) != 3:
        raise ValueError(f"URDF {attribute} must contain exactly three values")
    values = tuple(_finite_float(value, label=f"URDF {attribute}") for value in parts)
    return values[0], values[1], values[2]


def _finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _unique_names(
    values: Sequence[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of strings")
    result = tuple(values)
    if (not result and not allow_empty) or any(
        not isinstance(value, str) or not value for value in result
    ):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _env_ids(values: Sequence[int], *, num_envs: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("env_ids must be a sequence of ints")
    result = tuple(values)
    if not result or any(type(value) is not int for value in result):
        raise ValueError("env_ids must contain ints")
    if len(set(result)) != len(result):
        raise ValueError("env_ids must not contain duplicates")
    if any(value < 0 or value >= num_envs for value in result):
        raise ValueError("env_id is outside robot catalog range")
    return result


def _validate_gf_affine_batch(value: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain only finite values")
    has_zero_last_column = np.allclose(
        value[..., 0:3, 3],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    has_unit_bottom_right = np.allclose(
        value[..., 3, 3],
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    if not has_zero_last_column or not has_unit_bottom_right:
        raise ValueError(f"{label} must contain affine Gf matrices")
    determinant = np.linalg.det(value[..., :3, :3])
    if np.any(determinant == 0.0):
        raise ValueError(f"{label} linear transforms must be invertible")


def _origin_matrix(
    joint: ArticulatedJointSpec,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Return a URDF parent-to-joint origin using column-vector convention."""

    roll, pitch, yaw = joint.origin_rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result = np.eye(4, dtype=dtype)
    # URDF rpy is fixed-axis roll, pitch, yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    result[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    result[:3, 3] = joint.origin_xyz
    return result


def _joint_motion_matrix(
    joint: ArticulatedJointSpec,
    values: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Return batched joint motion matrices using column-vector convention."""

    shape = values.shape
    result = np.broadcast_to(np.eye(4, dtype=dtype), (*shape, 4, 4)).copy()
    x, y, z = joint.axis_xyz
    if joint.joint_type == "prismatic":
        result[..., 0, 3] = values * x
        result[..., 1, 3] = values * y
        result[..., 2, 3] = values * z
        return result
    if joint.joint_type not in {"revolute", "continuous"}:
        raise RuntimeError(
            f"unsupported moving joint escaped validation: {joint.name!r}"
        )

    cosine = np.cos(values)
    sine = np.sin(values)
    one_minus_cosine = 1.0 - cosine
    result[..., 0, 0] = cosine + x * x * one_minus_cosine
    result[..., 0, 1] = x * y * one_minus_cosine - z * sine
    result[..., 0, 2] = x * z * one_minus_cosine + y * sine
    result[..., 1, 0] = y * x * one_minus_cosine + z * sine
    result[..., 1, 1] = cosine + y * y * one_minus_cosine
    result[..., 1, 2] = y * z * one_minus_cosine - x * sine
    result[..., 2, 0] = z * x * one_minus_cosine - y * sine
    result[..., 2, 1] = z * y * one_minus_cosine + x * sine
    result[..., 2, 2] = cosine + z * z * one_minus_cosine
    return result


__all__ = [
    "ArticulatedJointSpec",
    "ArticulatedLinkTransformBatch",
    "ArticulatedMimicSpec",
    "CandidateColliderTransformBatch",
    "FullArticulatedForwardKinematics",
    "FullArticulatedKinematicsSpec",
    "RobotRootWorldTransformBatch",
    "build_full_articulated_kinematics_spec",
    "compose_candidate_collider_world_transforms",
    "full_articulated_urdf_path",
]
