"""Transactional construction of Kaleidoscope PhysX collision-query resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from linkerbot_sim.isaac.physics.query_shadow import PhysxQueryShadowOwner
from linkerbot_sim.isaac.physics.scene_query import PhysxSceneQueryAdapter
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    SceneCollisionTargetCatalog,
    build_robot_collision_catalog,
    build_scene_collision_target_catalog,
)
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    FullArticulatedKinematicsSpec,
    RobotRootWorldTransformBatch,
    build_full_articulated_kinematics_spec,
)
from linkerbot_sim.kaleidoscope.collision_overlap import PhysxCandidateOverlapOwner
from linkerbot_sim.kaleidoscope.collision_query import (
    ArticulatedCollisionQuerySpec,
)
from linkerbot_sim.kaleidoscope.collision_registry import (
    ArticulatedCollisionServiceRegistry,
)
from linkerbot_sim.kaleidoscope.collision_root_transforms import (
    capture_robot_root_world_transforms,
)
from linkerbot_sim.kaleidoscope.collision_service import (
    ReferenceArticulatedCollisionQueryService,
)
from linkerbot_sim.kaleidoscope.collision_torch import (
    TorchArticulatedCollisionQueryAdapter,
)


@dataclass(frozen=True, slots=True)
class RobotCommandJointLimits:
    """One robot's real command-space order and limits captured at startup."""

    robot_label: str
    command_joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

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
        lower = _limit_vector(self.lower, name="lower", width=len(names))
        upper = _limit_vector(self.upper, name="upper", width=len(names))
        if np.any(lower >= upper):
            raise ValueError("lower must be strictly below upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class PhysxCollisionQueryResources:
    """Built query adapter plus immutable facts used to construct it.

    ``query_adapter`` owns the registry, services, overlap owners, scene-query
    adapters, and shadow roots. Closing this bundle therefore has one entry point.
    """

    robot_catalog: RobotCollisionCatalog
    target_catalog: SceneCollisionTargetCatalog
    kinematics_specs: Mapping[str, FullArticulatedKinematicsSpec]
    robot_root_transforms: Mapping[str, RobotRootWorldTransformBatch]
    shadow_root_paths: Mapping[str, str]
    query_specs: Mapping[str, ArticulatedCollisionQuerySpec]
    query_adapter: TorchArticulatedCollisionQueryAdapter

    def close(self) -> None:
        self.query_adapter.close()


def build_physx_collision_query_resources(
    *,
    stage: object,
    replicated_scene: object,
    robot_profiles: Mapping[str, object],
    command_limits: Mapping[str, RobotCommandJointLimits],
    dynamic_object_name: str,
    device: object,
    table_object_name: str = "workstation",
    shadow_root_stem: str = "/World/__linkerbot_collision_query_robot",
    robot_catalog_factory: Callable[..., RobotCollisionCatalog] = (
        build_robot_collision_catalog
    ),
    target_catalog_factory: Callable[..., SceneCollisionTargetCatalog] = (
        build_scene_collision_target_catalog
    ),
    root_transform_factory: Callable[..., RobotRootWorldTransformBatch] = (
        capture_robot_root_world_transforms
    ),
    kinematics_factory: Callable[..., FullArticulatedKinematicsSpec] = (
        build_full_articulated_kinematics_spec
    ),
    shadow_owner_factory: Callable[..., PhysxQueryShadowOwner] = (
        PhysxQueryShadowOwner
    ),
    scene_query_factory: Callable[..., PhysxSceneQueryAdapter] = (
        PhysxSceneQueryAdapter
    ),
    overlap_factory: Callable[..., PhysxCandidateOverlapOwner] = (
        PhysxCandidateOverlapOwner
    ),
    service_factory: Callable[..., ReferenceArticulatedCollisionQueryService] = (
        ReferenceArticulatedCollisionQueryService
    ),
    registry_factory: Callable[..., ArticulatedCollisionServiceRegistry] = (
        ArticulatedCollisionServiceRegistry
    ),
    adapter_factory: Callable[..., TorchArticulatedCollisionQueryAdapter] = (
        TorchArticulatedCollisionQueryAdapter
    ),
) -> PhysxCollisionQueryResources:
    """Build the complete PhysX-only query ownership tree transactionally.

    Pure catalogs, FK specs, and root matrices are completed before native query
    resources are created. Once native construction begins, every successful
    object stays on a local ownership stack until a parent accepts it. Any
    failure closes that stack in reverse order without replacing the primary
    exception.
    """

    if getattr(replicated_scene, "collision_isolation_strategy", None) != "env_ids":
        raise ValueError("collision query resources require a PhysX env_ids scene")
    _absolute_prim_path(shadow_root_stem, label="shadow_root_stem")
    factories = (
        robot_catalog_factory,
        target_catalog_factory,
        root_transform_factory,
        kinematics_factory,
        shadow_owner_factory,
        scene_query_factory,
        overlap_factory,
        service_factory,
        registry_factory,
        adapter_factory,
    )
    if any(not callable(factory) for factory in factories):
        raise TypeError("collision query factories must be callable")

    robot_catalog = robot_catalog_factory(
        stage=stage,
        replicated_scene=replicated_scene,
    )
    target_catalog = target_catalog_factory(
        stage=stage,
        replicated_scene=replicated_scene,
        robot_catalog=robot_catalog,
        dynamic_object_name=dynamic_object_name,
        table_object_name=table_object_name,
    )
    labels = tuple(robot_catalog.robot_labels)
    if len(labels) < 2:
        raise ValueError("collision query resources require at least two robots")
    profiles = _exact_label_mapping(
        robot_profiles,
        labels=labels,
        name="robot_profiles",
    )
    limits_by_label = _exact_label_mapping(
        command_limits,
        labels=labels,
        name="command_limits",
    )
    robots_by_label = _replicated_robots_by_label(replicated_scene, labels=labels)

    all_env_ids = tuple(range(robot_catalog.num_envs))
    kinematics_specs: dict[str, FullArticulatedKinematicsSpec] = {}
    root_transforms: dict[str, RobotRootWorldTransformBatch] = {}
    query_specs: dict[str, ArticulatedCollisionQuerySpec] = {}
    shadow_paths = {
        label: f"{shadow_root_stem}_{index:04d}__" for index, label in enumerate(labels)
    }
    for label in labels:
        limits = limits_by_label[label]
        if not isinstance(limits, RobotCommandJointLimits):
            raise TypeError(
                f"command_limits[{label!r}] must be RobotCommandJointLimits"
            )
        if limits.robot_label != label:
            raise ValueError(
                f"command limit label does not match mapping key {label!r}"
            )
        scene_command_names = tuple(
            getattr(robots_by_label[label], "command_joint_names", ())
        )
        if limits.command_joint_names != scene_command_names:
            raise ValueError(
                f"command limits do not match replicated joint order for {label!r}"
            )
        query_specs[label] = ArticulatedCollisionQuerySpec(
            robot_label=label,
            command_joint_names=limits.command_joint_names,
            joint_lower=tuple(float(value) for value in limits.lower),
            joint_upper=tuple(float(value) for value in limits.upper),
        )
        required_links = tuple(
            dict.fromkeys(
                shape.rigid_body_path.rsplit("/", maxsplit=1)[-1]
                for shape in robot_catalog.shapes_for(env_id=0, robot_label=label)
            )
        )
        kinematics_specs[label] = kinematics_factory(
            robot_profile=profiles[label],
            command_joint_names=limits.command_joint_names,
            required_link_names=required_links,
        )
        root_transforms[label] = root_transform_factory(
            stage,
            robot_catalog=robot_catalog,
            robot_label=label,
            env_ids=all_env_ids,
            dtype=np.dtype(np.float32),
        )

    owned: list[object] = []
    try:
        services: list[object] = []
        for label in labels:
            shadow_owner = shadow_owner_factory(
                stage,
                root_path=shadow_paths[label],
            )
            owned.append(shadow_owner)
            scene_query = scene_query_factory()
            owned.append(scene_query)
            overlap = overlap_factory(
                robot_catalog=robot_catalog,
                robot_label=label,
                shadow_owner=shadow_owner,
                scene_query=scene_query,
            )
            _replace_owned_children(
                owned,
                children=(shadow_owner, scene_query),
                parent=overlap,
            )
            limits = limits_by_label[label]
            service = service_factory(
                robot_catalog=robot_catalog,
                robot_label=label,
                kinematics_spec=kinematics_specs[label],
                robot_root_transforms=root_transforms[label],
                joint_lower=limits.lower,
                joint_upper=limits.upper,
                overlap_port=overlap,
            )
            _replace_owned_children(owned, children=(overlap,), parent=service)
            services.append(service)

        registry = registry_factory(
            robot_labels=labels,
            services=tuple(services),
        )
        _replace_owned_children(owned, children=tuple(services), parent=registry)
        adapter = adapter_factory(
            registry,
            num_envs=robot_catalog.num_envs,
            device=device,
            target_catalog=target_catalog,
        )
        _replace_owned_children(owned, children=(registry,), parent=adapter)
        resources = PhysxCollisionQueryResources(
            robot_catalog=robot_catalog,
            target_catalog=target_catalog,
            kinematics_specs=MappingProxyType(dict(kinematics_specs)),
            robot_root_transforms=MappingProxyType(dict(root_transforms)),
            shadow_root_paths=MappingProxyType(dict(shadow_paths)),
            query_specs=MappingProxyType(dict(query_specs)),
            query_adapter=adapter,
        )
        owned.clear()
        return resources
    except BaseException as exc:
        _close_owned_resources(owned, original=exc)
        raise


def _limit_vector(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != (width,):
        raise ValueError(f"{name} must have shape [command joints]")
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must use float32")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(value, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _exact_label_mapping(
    value: Mapping[str, object],
    *,
    labels: tuple[str, ...],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != set(labels):
        raise ValueError(f"{name} keys must exactly match robot labels")
    return value


def _replicated_robots_by_label(
    replicated_scene: object,
    *,
    labels: tuple[str, ...],
) -> dict[str, object]:
    robots = tuple(getattr(replicated_scene, "robots", ()))
    by_label = {getattr(robot, "label", None): robot for robot in robots}
    if len(by_label) != len(robots) or set(by_label) != set(labels):
        raise ValueError(
            "replicated robots must exactly match collision catalog labels"
        )
    return {label: by_label[label] for label in labels}


def _replace_owned_children(
    owned: list[object],
    *,
    children: Sequence[object],
    parent: object,
) -> None:
    """Record an ownership transfer that only occurs after parent construction."""

    child_ids = {id(child) for child in children}
    owned[:] = [item for item in owned if id(item) not in child_ids]
    owned.append(parent)


def _close_owned_resources(
    owned: Sequence[object],
    *,
    original: BaseException,
) -> None:
    for resource in reversed(tuple(owned)):
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as cleanup_exc:
            original.add_note(
                "collision query construction cleanup failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
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


__all__ = [
    "PhysxCollisionQueryResources",
    "RobotCommandJointLimits",
    "build_physx_collision_query_resources",
]
