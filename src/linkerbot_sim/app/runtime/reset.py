"""Runtime reset helpers for already-created Isaac simulation sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from linkerbot_sim.app.runtime.objects import (
    RuntimeObjectConfig,
    runtime_objects_from_env_config,
)
from linkerbot_sim.app.runtime.settings import EnvRuntimeSettings
from linkerbot_sim.assets.robot_loader import (
    RootPoseConfig,
    apply_root_pose,
    robot_scene_instance_from_env_config,
)
from linkerbot_sim.objects.physics import apply_root_pose_to_prim


RobotRootPoseApplier = Callable[[object, str, RootPoseConfig], None]
ObjectRootPoseApplier = Callable[[object, str, RootPoseConfig], None]


@dataclass(frozen=True)
class RuntimeResetOptions:
    """Options for a lightweight runtime reset."""

    hold_after_reset: bool = True


@dataclass(frozen=True)
class RuntimeResetResult:
    """Result of a runtime reset."""

    step: int = 0
    message: str = ""


def reset_dual_robot_runtime(
    runtime,
    *,
    options: RuntimeResetOptions = RuntimeResetOptions(),
    robot_root_pose_applier: RobotRootPoseApplier = apply_root_pose,
    object_root_pose_applier: ObjectRootPoseApplier = apply_root_pose_to_prim,
) -> RuntimeResetResult:
    """Reset an existing dual-robot runtime without rebuilding SimulationApp."""

    stage = runtime.session.stage
    _reset_runtime_objects(
        stage=stage,
        handles=runtime.object_handles,
        configs=runtime_objects_from_env_config(runtime.env_config),
        object_root_pose_applier=object_root_pose_applier,
    )
    for side in ("left", "right"):
        _apply_robot_root_pose(
            stage=stage,
            imported=runtime.imported[side],
            root_pose=runtime.dual_config.side(side).root_pose,
            robot_root_pose_applier=robot_root_pose_applier,
        )
    _reset_world(runtime)
    for side in ("left", "right"):
        _reset_prepared_robot(runtime.prepared[side])
    _reset_execution_observers(runtime.execution)
    return RuntimeResetResult(
        step=0,
        message=(
            "runtime reset completed; hold_after_reset="
            f"{bool(options.hold_after_reset)}"
        ),
    )


def reset_single_robot_runtime(
    runtime,
    *,
    options: RuntimeResetOptions = RuntimeResetOptions(),
    robot_root_pose_applier: RobotRootPoseApplier = apply_root_pose,
    object_root_pose_applier: ObjectRootPoseApplier = apply_root_pose_to_prim,
) -> RuntimeResetResult:
    """Reset an existing single-robot runtime without rebuilding SimulationApp."""

    stage = runtime.session.stage
    _reset_runtime_objects(
        stage=stage,
        handles=runtime.object_handles,
        configs=runtime_objects_from_env_config(runtime.env_config),
        object_root_pose_applier=object_root_pose_applier,
    )
    robot_instance = robot_scene_instance_from_env_config(runtime.env_config, "single")
    _apply_robot_root_pose(
        stage=stage,
        imported=runtime.imported_robot,
        root_pose=robot_instance.root_pose,
        robot_root_pose_applier=robot_root_pose_applier,
    )
    _reset_world(runtime)
    _reset_prepared_robot(runtime.prepared_robot)
    _reset_execution_observers(runtime.execution)
    return RuntimeResetResult(
        step=0,
        message=(
            "runtime reset completed; hold_after_reset="
            f"{bool(options.hold_after_reset)}"
        ),
    )


def _reset_world(runtime) -> None:
    """Reset world and restore scene-level gravity."""

    runtime.session.world.reset()
    gravity_z = EnvRuntimeSettings.from_env_config(runtime.env_config).gravity_z
    runtime.session.world.get_physics_context().set_gravity(gravity_z)


def _reset_prepared_robot(prepared) -> None:
    """Restore reset-sensitive robot runtime settings."""

    articulation = prepared.articulation
    if prepared.gravity_policy.disables_all_known_components():
        articulation.disable_gravity()
    _call_optional(articulation, "set_joint_velocities", _zeros(_num_dof(articulation)))
    controller = prepared.joint_controller
    configure_runtime = getattr(controller, "configure_runtime", None)
    if configure_runtime is not None:
        configure_runtime()
    if hasattr(controller, "last_commanded_efforts"):
        controller.last_commanded_efforts = np.full(
            _num_dof(articulation), np.nan, dtype=float
        )


def _reset_execution_observers(execution) -> None:
    """Clear observer-derived caches after a world reset."""

    for name in ("state_observer", "camera_observer"):
        observer = getattr(execution, name, None)
        reset = getattr(observer, "reset", None)
        if reset is not None:
            reset()


def _reset_runtime_objects(
    *,
    stage: object,
    handles: Sequence[object],
    configs: Sequence[RuntimeObjectConfig],
    object_root_pose_applier: ObjectRootPoseApplier,
) -> None:
    """Restore runtime object root poses from env config."""

    configs_by_key: dict[str, RuntimeObjectConfig] = {}
    for config in configs:
        configs_by_key[config.name] = config
        if config.runtime_handle is not None:
            configs_by_key[config.runtime_handle] = config
    for handle in handles:
        config = _runtime_object_config_for_handle(handle, configs_by_key)
        prim_path = _runtime_object_prim_path(handle)
        if config is None or prim_path is None:
            continue
        object_root_pose_applier(stage, prim_path, config.root_pose)


def _runtime_object_config_for_handle(
    handle: object,
    configs_by_key: Mapping[str, RuntimeObjectConfig],
) -> RuntimeObjectConfig | None:
    """Find the env config that produced a runtime object handle."""

    for key in (getattr(handle, "runtime_handle", None), getattr(handle, "name", None)):
        if key is not None and str(key) in configs_by_key:
            return configs_by_key[str(key)]
    return None


def _runtime_object_prim_path(handle: object) -> str | None:
    """Read root prim path from a RuntimeObjectHandle-like object."""

    for source in (getattr(handle, "model", None), getattr(handle, "config", None)):
        if source is None:
            continue
        prim_path = getattr(source, "prim_path", None)
        if prim_path is not None:
            return str(prim_path)
        if isinstance(source, Mapping):
            root = source.get("root")
            if root is not None and hasattr(root, "GetPath"):
                return str(root.GetPath())
            prim_path = source.get("prim_path")
            if prim_path is not None:
                return str(prim_path)
    return None


def _apply_robot_root_pose(
    *,
    stage: object,
    imported: object,
    root_pose: RootPoseConfig,
    robot_root_pose_applier: RobotRootPoseApplier,
) -> None:
    """Apply robot root pose when the imported root path is known."""

    root_path = getattr(imported, "imported_root_path", None)
    if root_path is None:
        return
    robot_root_pose_applier(stage, str(root_path), root_pose)


def _call_optional(source: object, method_name: str, *args: Any) -> None:
    """Call an optional method if present."""

    method = getattr(source, method_name, None)
    if method is not None:
        method(*args)


def _num_dof(articulation: object) -> int:
    """Read articulation DOF count."""

    if hasattr(articulation, "num_dof"):
        return int(getattr(articulation, "num_dof"))
    return len(getattr(articulation, "dof_names", ()))


def _zeros(size: int) -> np.ndarray:
    """Return a float zero vector."""

    return np.zeros(int(size), dtype=float)
