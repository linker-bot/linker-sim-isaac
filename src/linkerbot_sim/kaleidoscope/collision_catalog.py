"""Kaleidoscope env/robot ownership catalog for PhysX collision shapes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from linkerbot_sim.isaac.physics.collision_shapes import (
    PhysxCollisionShapeMetadata,
    PhysxSceneCollisionIdentity,
    discover_articulated_collision_shapes,
    discover_scene_collision_identities,
)
from linkerbot_sim.kaleidoscope.collision_query import CollisionQueryTarget


CollisionShapeScanner = Callable[
    [object, str],
    Sequence[PhysxCollisionShapeMetadata],
]
SceneCollisionScanner = Callable[
    [object, str],
    Sequence[PhysxSceneCollisionIdentity],
]


_TARGET_ORDER = {
    CollisionQueryTarget.TABLE: 0,
    CollisionQueryTarget.DYNAMIC_OBJECT: 1,
    CollisionQueryTarget.OTHER_ROBOT: 2,
}


@dataclass(frozen=True, slots=True)
class RobotCollisionShape:
    """One exact collider bound to a replicated env and robot identity."""

    env_id: int
    robot_id: int
    robot_label: str
    env_root_path: str
    imported_root_path: str
    articulation_path: str
    collision_suffix: str
    rigid_body_suffix: str
    metadata: PhysxCollisionShapeMetadata

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise ValueError("env_id must be a non-negative int")
        if type(self.robot_id) is not int or self.robot_id < 0:
            raise ValueError("robot_id must be a non-negative int")
        if not isinstance(self.robot_label, str) or not self.robot_label:
            raise ValueError("robot_label must be a non-empty string")
        for label, path in (
            ("env_root_path", self.env_root_path),
            ("imported_root_path", self.imported_root_path),
            ("articulation_path", self.articulation_path),
        ):
            _absolute_prim_path(path, label=label)
        if not self.collision_suffix.startswith("/"):
            raise ValueError("collision_suffix must be a non-empty relative suffix")
        if self.rigid_body_suffix and not self.rigid_body_suffix.startswith("/"):
            raise ValueError("rigid_body_suffix must be empty or start with '/'")
        if not isinstance(self.metadata, PhysxCollisionShapeMetadata):
            raise TypeError("metadata must be PhysxCollisionShapeMetadata")
        suffix_root = _robot_shape_suffix_root(
            self.imported_root_path,
            self.articulation_path,
            self.metadata,
        )
        if f"{suffix_root}{self.collision_suffix}" != self.metadata.collision_path:
            raise ValueError("collision_suffix does not match metadata collision_path")
        if f"{suffix_root}{self.rigid_body_suffix}" != self.metadata.rigid_body_path:
            raise ValueError(
                "rigid_body_suffix does not match metadata rigid_body_path"
            )

    @property
    def collision_path(self) -> str:
        return self.metadata.collision_path

    @property
    def rigid_body_path(self) -> str:
        return self.metadata.rigid_body_path


@dataclass(frozen=True, slots=True)
class RobotCollisionCatalog:
    """Immutable exact-path index for every replicated articulated collider."""

    num_envs: int
    robot_labels: tuple[str, ...]
    shapes: tuple[RobotCollisionShape, ...]
    _by_env_robot: Mapping[tuple[int, str], tuple[RobotCollisionShape, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_collision_path: Mapping[str, RobotCollisionShape] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be a positive int")
        if (
            not isinstance(self.robot_labels, tuple)
            or not self.robot_labels
            or any(
                not isinstance(label, str) or not label for label in self.robot_labels
            )
            or len(set(self.robot_labels)) != len(self.robot_labels)
        ):
            raise ValueError("robot_labels must be a non-empty unique string tuple")
        if not isinstance(self.shapes, tuple) or any(
            not isinstance(shape, RobotCollisionShape) for shape in self.shapes
        ):
            raise TypeError("shapes must be a RobotCollisionShape tuple")
        label_order = {label: index for index, label in enumerate(self.robot_labels)}
        expected_order = tuple(
            sorted(
                self.shapes,
                key=lambda shape: (
                    shape.env_id,
                    label_order.get(shape.robot_label, len(label_order)),
                    shape.collision_suffix,
                ),
            )
        )
        if self.shapes != expected_order:
            raise ValueError("catalog shapes must use stable env/robot/path order")

        grouped: dict[tuple[int, str], list[RobotCollisionShape]] = {}
        by_path: dict[str, RobotCollisionShape] = {}
        for shape in self.shapes:
            if shape.env_id >= self.num_envs:
                raise ValueError("shape env_id is outside catalog num_envs")
            if shape.robot_label not in label_order:
                raise ValueError("shape robot_label is outside catalog robot_labels")
            grouped.setdefault((shape.env_id, shape.robot_label), []).append(shape)
            if shape.collision_path in by_path:
                raise ValueError(
                    f"duplicate catalog collision_path: {shape.collision_path}"
                )
            by_path[shape.collision_path] = shape

        expected_groups = {
            (env_id, robot_label)
            for env_id in range(self.num_envs)
            for robot_label in self.robot_labels
        }
        if set(grouped) != expected_groups:
            missing = sorted(expected_groups - set(grouped))
            raise ValueError(f"catalog is missing env/robot collider groups: {missing}")
        object.__setattr__(
            self,
            "_by_env_robot",
            MappingProxyType({key: tuple(values) for key, values in grouped.items()}),
        )
        object.__setattr__(
            self,
            "_by_collision_path",
            MappingProxyType(by_path),
        )

    def shapes_for(
        self,
        *,
        env_id: int,
        robot_label: str,
    ) -> tuple[RobotCollisionShape, ...]:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise ValueError("env_id is outside catalog range")
        if robot_label not in self.robot_labels:
            raise KeyError(f"unknown robot label: {robot_label!r}")
        return self._by_env_robot[(env_id, robot_label)]

    def shape_for_collision_path(self, collision_path: str) -> RobotCollisionShape:
        try:
            return self._by_collision_path[collision_path]
        except KeyError as exc:
            raise KeyError(f"unknown collision path: {collision_path!r}") from exc


@dataclass(frozen=True, slots=True)
class SceneCollisionTargetShape:
    """One exact collider that can be selected as a collision-query target."""

    env_id: int
    category: CollisionQueryTarget
    source_name: str
    owner_robot_label: str | None
    root_path: str
    collision_path: str
    rigid_body_path: str | None

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise ValueError("env_id must be a non-negative int")
        if not isinstance(self.category, CollisionQueryTarget):
            raise TypeError("category must be CollisionQueryTarget")
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("source_name must be a non-empty string")
        _absolute_prim_path(self.root_path, label="root_path")
        _absolute_prim_path(self.collision_path, label="collision_path")
        if not _is_descendant_or_self(self.collision_path, self.root_path):
            raise ValueError("collision_path is outside target root_path")
        if self.rigid_body_path is not None:
            _absolute_prim_path(self.rigid_body_path, label="rigid_body_path")
            if not _is_descendant_or_self(self.rigid_body_path, self.root_path):
                raise ValueError("rigid_body_path is outside target root_path")
        if self.category is CollisionQueryTarget.OTHER_ROBOT:
            if (
                not isinstance(self.owner_robot_label, str)
                or not self.owner_robot_label
            ):
                raise ValueError("other-robot target requires owner_robot_label")
        elif self.owner_robot_label is not None:
            raise ValueError("object target cannot define owner_robot_label")


@dataclass(frozen=True, slots=True)
class SceneCollisionTargetCatalog:
    """Immutable exact-path target index for replicated PhysX environments."""

    num_envs: int
    robot_labels: tuple[str, ...]
    shapes: tuple[SceneCollisionTargetShape, ...]
    _by_collision_path: Mapping[str, SceneCollisionTargetShape] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be a positive int")
        if (
            not isinstance(self.robot_labels, tuple)
            or len(self.robot_labels) < 2
            or any(
                not isinstance(label, str) or not label for label in self.robot_labels
            )
            or len(set(self.robot_labels)) != len(self.robot_labels)
        ):
            raise ValueError(
                "robot_labels must contain at least two unique non-empty strings"
            )
        if not isinstance(self.shapes, tuple) or any(
            not isinstance(shape, SceneCollisionTargetShape) for shape in self.shapes
        ):
            raise TypeError("shapes must be a SceneCollisionTargetShape tuple")
        expected_order = tuple(sorted(self.shapes, key=_scene_target_sort_key))
        if self.shapes != expected_order:
            raise ValueError(
                "target shapes must use stable env/category/source/path order"
            )

        by_path: dict[str, SceneCollisionTargetShape] = {}
        object_groups: set[tuple[int, CollisionQueryTarget]] = set()
        robot_groups: set[tuple[int, str]] = set()
        for shape in self.shapes:
            if shape.env_id >= self.num_envs:
                raise ValueError("target shape env_id is outside catalog num_envs")
            if shape.collision_path in by_path:
                raise ValueError(
                    "collision path belongs to multiple targets: "
                    f"{shape.collision_path}"
                )
            by_path[shape.collision_path] = shape
            if shape.category is CollisionQueryTarget.OTHER_ROBOT:
                if shape.owner_robot_label not in self.robot_labels:
                    raise ValueError("target shape has unknown owner_robot_label")
                robot_groups.add((shape.env_id, str(shape.owner_robot_label)))
            else:
                object_groups.add((shape.env_id, shape.category))

        expected_object_groups = {
            (env_id, category)
            for env_id in range(self.num_envs)
            for category in (
                CollisionQueryTarget.TABLE,
                CollisionQueryTarget.DYNAMIC_OBJECT,
            )
        }
        if object_groups != expected_object_groups:
            missing = sorted(
                expected_object_groups - object_groups,
                key=lambda item: (item[0], _TARGET_ORDER[item[1]]),
            )
            raise ValueError(f"catalog is missing scene target groups: {missing}")
        expected_robot_groups = {
            (env_id, label)
            for env_id in range(self.num_envs)
            for label in self.robot_labels
        }
        if robot_groups != expected_robot_groups:
            missing = sorted(expected_robot_groups - robot_groups)
            raise ValueError(f"catalog is missing robot target groups: {missing}")
        object.__setattr__(
            self,
            "_by_collision_path",
            MappingProxyType(by_path),
        )

    def targets_for(
        self,
        *,
        env_id: int,
        robot_label: str,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> tuple[SceneCollisionTargetShape, ...]:
        """Select exact same-env targets while excluding the current robot."""

        self._validate_selector(env_id=env_id, robot_label=robot_label, targets=targets)
        selected = set(targets)
        return tuple(
            shape
            for shape in self.shapes
            if shape.env_id == env_id
            and shape.category in selected
            and not (
                shape.category is CollisionQueryTarget.OTHER_ROBOT
                and shape.owner_robot_label == robot_label
            )
        )

    def collision_paths_for(
        self,
        *,
        env_id: int,
        robot_label: str,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> frozenset[str]:
        """Return the exact-path equality set used by overlap-hit filtering."""

        return frozenset(
            shape.collision_path
            for shape in self.targets_for(
                env_id=env_id,
                robot_label=robot_label,
                targets=targets,
            )
        )

    def category_for_collision_path(
        self,
        collision_path: str,
        *,
        env_id: int,
        robot_label: str,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> CollisionQueryTarget | None:
        """Classify one exact callback path, or reject it as out of scope."""

        self._validate_selector(env_id=env_id, robot_label=robot_label, targets=targets)
        shape = self._by_collision_path.get(collision_path)
        if (
            shape is None
            or shape.env_id != env_id
            or shape.category not in targets
            or (
                shape.category is CollisionQueryTarget.OTHER_ROBOT
                and shape.owner_robot_label == robot_label
            )
        ):
            return None
        return shape.category

    def _validate_selector(
        self,
        *,
        env_id: int,
        robot_label: str,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> None:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise ValueError("env_id is outside catalog range")
        if robot_label not in self.robot_labels:
            raise KeyError(f"unknown robot label: {robot_label!r}")
        if (
            not isinstance(targets, tuple)
            or not targets
            or any(not isinstance(target, CollisionQueryTarget) for target in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("targets must be a non-empty unique target tuple")


def build_robot_collision_catalog(
    *,
    stage: object,
    replicated_scene: object,
    scanner: CollisionShapeScanner = discover_articulated_collision_shapes,
) -> RobotCollisionCatalog:
    """Bind replicated PhysX robot topology to exact active collider metadata."""

    if getattr(replicated_scene, "collision_isolation_strategy", None) != "env_ids":
        raise ValueError("robot collision catalog requires PhysX env_ids isolation")
    env_roots = tuple(getattr(replicated_scene, "env_root_paths", ()))
    if not env_roots:
        raise ValueError("replicated scene must contain environment roots")
    for path in env_roots:
        _absolute_prim_path(path, label="env_root_path")
    robots = tuple(getattr(replicated_scene, "robots", ()))
    if not robots:
        raise ValueError("replicated scene must contain robots")
    raw_labels = tuple(getattr(robot, "label", None) for robot in robots)
    if any(not isinstance(label, str) or not label for label in raw_labels) or len(
        set(raw_labels)
    ) != len(raw_labels):
        raise ValueError("replicated robot labels must be non-empty and unique")
    labels = tuple(label for label in raw_labels if isinstance(label, str))
    raw_robot_ids = tuple(getattr(robot, "robot_id", None) for robot in robots)
    if any(type(robot_id) is not int or robot_id < 0 for robot_id in raw_robot_ids):
        raise ValueError("replicated robot ids must be non-negative ints")
    robot_ids = tuple(int(robot_id) for robot_id in raw_robot_ids)
    if len(set(robot_ids)) != len(robot_ids):
        raise ValueError("replicated robot ids must be unique")

    entries: list[RobotCollisionShape] = []
    for robot, robot_label, robot_id in zip(
        robots,
        labels,
        robot_ids,
        strict=True,
    ):
        articulation_paths = tuple(getattr(robot, "articulation_paths", ()))
        imported_root_paths = tuple(getattr(robot, "imported_root_paths", ()))
        if len(articulation_paths) != len(env_roots) or len(imported_root_paths) != len(
            env_roots
        ):
            raise ValueError(f"robot {robot_label!r} paths must exactly cover all envs")
        canonical: dict[str, RobotCollisionShape] | None = None
        for env_id, (env_root, imported_root, articulation_path) in enumerate(
            zip(
                env_roots,
                imported_root_paths,
                articulation_paths,
                strict=True,
            )
        ):
            _absolute_prim_path(imported_root, label="imported_root_path")
            _absolute_prim_path(articulation_path, label="articulation_path")
            if not _is_descendant_or_self(imported_root, env_root):
                raise ValueError("robot imported root is outside its env root")
            if not _is_descendant_or_self(articulation_path, imported_root):
                raise ValueError("articulation path is outside robot imported root")
            # Isaac Sim 6.1 may place ArticulationRootAPI on a fixed joint.
            # Colliders are then siblings of that joint, still within the
            # robot's imported subtree.
            metadata_values = tuple(scanner(stage, imported_root))
            if not metadata_values:
                raise RuntimeError(
                    f"robot {robot_label!r} env {env_id} has no collision shapes"
                )
            env_entries: dict[str, RobotCollisionShape] = {}
            for metadata in metadata_values:
                if not isinstance(metadata, PhysxCollisionShapeMetadata):
                    raise TypeError(
                        "collision scanner must return PhysxCollisionShapeMetadata"
                    )
                suffix_root = _robot_shape_suffix_root(
                    imported_root,
                    articulation_path,
                    metadata,
                )
                collision_suffix = _relative_suffix(
                    suffix_root,
                    metadata.collision_path,
                    label="collision_path",
                )
                rigid_body_suffix = _relative_suffix(
                    suffix_root,
                    metadata.rigid_body_path,
                    label="rigid_body_path",
                )
                if collision_suffix in env_entries:
                    raise RuntimeError(
                        f"duplicate robot collision suffix: {collision_suffix}"
                    )
                env_entries[collision_suffix] = RobotCollisionShape(
                    env_id=env_id,
                    robot_id=robot_id,
                    robot_label=robot_label,
                    env_root_path=env_root,
                    imported_root_path=imported_root,
                    articulation_path=articulation_path,
                    collision_suffix=collision_suffix,
                    rigid_body_suffix=rigid_body_suffix,
                    metadata=metadata,
                )
            if canonical is None:
                canonical = env_entries
            else:
                _validate_cloned_topology(
                    canonical,
                    env_entries,
                    robot_label=robot_label,
                    env_id=env_id,
                )
            entries.extend(env_entries.values())

    label_order = {label: index for index, label in enumerate(labels)}
    entries.sort(
        key=lambda shape: (
            shape.env_id,
            label_order[shape.robot_label],
            shape.collision_suffix,
        )
    )
    return RobotCollisionCatalog(
        num_envs=len(env_roots),
        robot_labels=labels,
        shapes=tuple(entries),
    )


def build_scene_collision_target_catalog(
    *,
    stage: object,
    replicated_scene: object,
    robot_catalog: RobotCollisionCatalog,
    dynamic_object_name: str,
    table_object_name: str = "workstation",
    scanner: SceneCollisionScanner = discover_scene_collision_identities,
) -> SceneCollisionTargetCatalog:
    """Build exact table, dynamic-object, and other-robot target identities."""

    if getattr(replicated_scene, "collision_isolation_strategy", None) != "env_ids":
        raise ValueError(
            "scene collision target catalog requires PhysX env_ids isolation"
        )
    if not isinstance(robot_catalog, RobotCollisionCatalog):
        raise TypeError("robot_catalog must be RobotCollisionCatalog")
    if not isinstance(dynamic_object_name, str) or not dynamic_object_name:
        raise ValueError("dynamic_object_name must be a non-empty string")
    if not isinstance(table_object_name, str) or not table_object_name:
        raise ValueError("table_object_name must be a non-empty string")
    if dynamic_object_name == table_object_name:
        raise ValueError("table and dynamic object names must differ")

    env_roots = tuple(getattr(replicated_scene, "env_root_paths", ()))
    if not env_roots:
        raise ValueError("replicated scene must contain environment roots")
    if robot_catalog.num_envs != len(env_roots):
        raise ValueError("robot catalog num_envs does not match replicated scene")
    scene_robot_labels = tuple(
        getattr(robot, "label", None)
        for robot in tuple(getattr(replicated_scene, "robots", ()))
    )
    if scene_robot_labels != robot_catalog.robot_labels:
        raise ValueError("robot catalog labels do not match replicated scene")
    for path in env_roots:
        _absolute_prim_path(path, label="env_root_path")

    handles = tuple(getattr(replicated_scene, "object_handles", ()))
    if not handles:
        raise ValueError("replicated scene must contain object handles")
    handles_by_name: dict[str, object] = {}
    static_names: list[str] = []
    dynamic_names: list[str] = []
    for handle in handles:
        name = getattr(handle, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("scene object handle name must be a non-empty string")
        if name in handles_by_name:
            raise ValueError(f"duplicate scene object handle name: {name!r}")
        if getattr(handle, "kind", None) != "rigid":
            raise ValueError("scene collision targets require rigid object handles")
        physics = getattr(getattr(handle, "config", None), "physics", None)
        is_static = getattr(physics, "static", None)
        if type(is_static) is not bool:
            raise ValueError("rigid object handle must expose boolean physics.static")
        handles_by_name[name] = handle
        (static_names if is_static else dynamic_names).append(name)

    if static_names != [table_object_name]:
        raise ValueError(
            "scene collision targets require exactly the configured static table: "
            f"found {static_names}"
        )
    if dynamic_names != [dynamic_object_name]:
        raise ValueError(
            "scene collision targets require exactly the task dynamic object: "
            f"found {dynamic_names}"
        )

    raw_object_paths = getattr(replicated_scene, "object_prim_paths", None)
    if not isinstance(raw_object_paths, Mapping):
        raise TypeError("replicated scene object_prim_paths must be a mapping")
    if set(raw_object_paths) != set(handles_by_name):
        raise ValueError("object_prim_paths keys must exactly match object handles")

    entries: list[SceneCollisionTargetShape] = []
    for object_name, category in (
        (table_object_name, CollisionQueryTarget.TABLE),
        (dynamic_object_name, CollisionQueryTarget.DYNAMIC_OBJECT),
    ):
        roots = tuple(raw_object_paths[object_name])
        if len(roots) != len(env_roots):
            raise ValueError(
                f"object {object_name!r} paths must exactly cover all envs"
            )
        canonical: tuple[tuple[str, str | None], ...] | None = None
        for env_id, (env_root, object_root) in enumerate(
            zip(env_roots, roots, strict=True)
        ):
            _absolute_prim_path(object_root, label="object_root_path")
            if not _is_descendant_or_self(object_root, env_root):
                raise ValueError("scene object root is outside its env root")
            identities = tuple(scanner(stage, object_root))
            if not identities:
                raise RuntimeError(
                    f"scene object {object_name!r} env {env_id} has no colliders"
                )
            topology: list[tuple[str, str | None]] = []
            for identity in identities:
                if not isinstance(identity, PhysxSceneCollisionIdentity):
                    raise TypeError(
                        "scene collision scanner must return "
                        "PhysxSceneCollisionIdentity"
                    )
                collision_suffix = _relative_suffix(
                    object_root,
                    identity.collision_path,
                    label="collision_path",
                )
                rigid_body_suffix = (
                    None
                    if identity.rigid_body_path is None
                    else _relative_suffix(
                        object_root,
                        identity.rigid_body_path,
                        label="rigid_body_path",
                    )
                )
                topology.append((collision_suffix, rigid_body_suffix))
                entries.append(
                    SceneCollisionTargetShape(
                        env_id=env_id,
                        category=category,
                        source_name=object_name,
                        owner_robot_label=None,
                        root_path=object_root,
                        collision_path=identity.collision_path,
                        rigid_body_path=identity.rigid_body_path,
                    )
                )
            stable_topology = tuple(sorted(topology))
            if canonical is None:
                canonical = stable_topology
            elif stable_topology != canonical:
                raise RuntimeError(
                    f"scene object {object_name!r} env {env_id} collision topology "
                    "differs from env 0"
                )

    for robot_shape in robot_catalog.shapes:
        entries.append(
            SceneCollisionTargetShape(
                env_id=robot_shape.env_id,
                category=CollisionQueryTarget.OTHER_ROBOT,
                source_name=robot_shape.robot_label,
                owner_robot_label=robot_shape.robot_label,
                root_path=robot_shape.imported_root_path,
                collision_path=robot_shape.collision_path,
                rigid_body_path=robot_shape.rigid_body_path,
            )
        )

    entries.sort(key=_scene_target_sort_key)
    return SceneCollisionTargetCatalog(
        num_envs=len(env_roots),
        robot_labels=robot_catalog.robot_labels,
        shapes=tuple(entries),
    )


def _scene_target_sort_key(
    shape: SceneCollisionTargetShape,
) -> tuple[int, int, str, str]:
    return (
        shape.env_id,
        _TARGET_ORDER[shape.category],
        shape.source_name,
        shape.collision_path,
    )


def _validate_cloned_topology(
    canonical: Mapping[str, RobotCollisionShape],
    candidate: Mapping[str, RobotCollisionShape],
    *,
    robot_label: str,
    env_id: int,
) -> None:
    if set(candidate) != set(canonical):
        raise RuntimeError(
            f"robot {robot_label!r} env {env_id} collision topology differs from env 0"
        )
    for suffix, expected in canonical.items():
        actual = candidate[suffix]
        same_fields = (
            actual.rigid_body_suffix == expected.rigid_body_suffix
            and actual.metadata.geometry_type == expected.metadata.geometry_type
            and actual.metadata.mesh_approximation
            == expected.metadata.mesh_approximation
        )
        same_matrix = all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-8)
            for left, right in zip(
                actual.metadata.shape_to_rigid_body_matrix_row_major,
                expected.metadata.shape_to_rigid_body_matrix_row_major,
                strict=True,
            )
        )
        if not same_fields or not same_matrix:
            raise RuntimeError(
                f"robot {robot_label!r} env {env_id} collider {suffix!r} "
                "metadata differs from env 0"
            )


def _absolute_prim_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
    ):
        raise ValueError(f"{label} must be a canonical absolute USD prim path")
    return value


def _is_descendant_or_self(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _robot_shape_suffix_root(
    imported_root: str,
    articulation_path: str,
    metadata: PhysxCollisionShapeMetadata,
) -> str:
    for path in (metadata.collision_path, metadata.rigid_body_path):
        if not _is_descendant_or_self(path, imported_root):
            raise RuntimeError(
                f"robot collision identity is outside imported root: {path}"
            )
    if all(
        _is_descendant_or_self(path, articulation_path)
        for path in (metadata.collision_path, metadata.rigid_body_path)
    ):
        return articulation_path
    return imported_root


def _relative_suffix(root: str, path: str, *, label: str) -> str:
    if not _is_descendant_or_self(path, root):
        raise RuntimeError(f"{label} is outside articulation path: {path}")
    return path[len(root) :]


__all__ = [
    "RobotCollisionCatalog",
    "RobotCollisionShape",
    "SceneCollisionTargetCatalog",
    "SceneCollisionTargetShape",
    "build_robot_collision_catalog",
    "build_scene_collision_target_catalog",
]
