"""env objects 到 runtime object handle 的通用导入分发。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from linkerbot_sim.assets.robot_loader import (
    RootPoseConfig,
)
from linkerbot_sim.objects.rigid.runtime import (
    RigidObjectConfig,
    add_rigid_objects,
)
from linkerbot_sim.objects.dynamic_chain.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
    apply_capsule_rope_runtime_physics,
)
from linkerbot_sim.objects.config import (
    ObjectProfileConfig,
    ObjectSceneInstanceConfig,
    expanded_object_mapping,
    object_scene_instances_from_env_config,
)
from linkerbot_sim.objects.physics import apply_root_pose_to_prim


@dataclass(frozen=True)
class RuntimeObjectConfig:
    """env ``objects[]`` 中一项对象声明。"""

    name: str
    kind: str
    source: str
    root_pose: RootPoseConfig
    object_profile: str
    profile: ObjectProfileConfig
    runtime_handle: str | None = None


@dataclass(frozen=True)
class RuntimeObjectHandle:
    """已导入 stage 的对象句柄。"""

    name: str
    kind: str
    source: str
    runtime_handle: str | None
    config: object
    model: object


def runtime_objects_from_env_config(
    env_config: Mapping[str, object],
) -> tuple[RuntimeObjectConfig, ...]:
    """解析 env YAML 顶层 ``objects`` 列表。"""

    return tuple(
        _runtime_object_from_scene_instance(item)
        for item in object_scene_instances_from_env_config(env_config)
    )


def add_runtime_objects(
    stage,
    objects: Sequence[RuntimeObjectConfig],
    *,
    status_prefix: str | None = None,
) -> tuple[RuntimeObjectHandle, ...]:
    """把 env 中声明的对象导入 stage。"""

    return tuple(
        add_runtime_object(stage, config=config, status_prefix=status_prefix)
        for config in objects
    )


def runtime_object_handles_by_name(
    handles: Sequence[RuntimeObjectHandle],
) -> dict[str, RuntimeObjectHandle]:
    """返回带 runtime_handle 的对象字典。"""

    result: dict[str, RuntimeObjectHandle] = {}
    for handle in handles:
        if handle.runtime_handle is None:
            continue
        if handle.runtime_handle in result:
            raise ValueError(
                f"Duplicate runtime object handle: {handle.runtime_handle}"
            )
        result[handle.runtime_handle] = handle
    return result


def add_runtime_object(
    stage,
    *,
    config: RuntimeObjectConfig,
    status_prefix: str | None = None,
) -> RuntimeObjectHandle:
    """导入单个 runtime object。"""

    if config.kind == "rigid":
        return _add_rigid_object(stage, config, status_prefix=status_prefix)
    if config.kind == "dynamic_chain" and config.source == "usd":
        return _add_capsule_rope_object(stage, config, status_prefix=status_prefix)
    raise ValueError(
        f"Unsupported runtime object kind/source: {config.kind!r}/{config.source!r}"
    )


def _runtime_object_from_scene_instance(instance) -> RuntimeObjectConfig:
    """把 env objects[] 实例和 object profile 合并成运行时对象配置。"""

    profile = ObjectProfileConfig.from_profile(instance.object_profile)
    return RuntimeObjectConfig(
        name=instance.name,
        kind=profile.kind,
        source=profile.source,
        root_pose=instance.root_pose,
        runtime_handle=instance.runtime_handle,
        object_profile=instance.object_profile,
        profile=profile,
    )


def _add_rigid_object(
    stage, config: RuntimeObjectConfig, *, status_prefix: str | None
) -> RuntimeObjectHandle:
    """导入 rigid object，并返回统一 RuntimeObjectHandle。"""

    rigid_config = _rigid_object_config_from_runtime_object(config)
    added = add_rigid_objects(stage, (rigid_config,))[0]
    _print_object_status(
        status_prefix,
        name=config.name,
        kind=config.kind,
        source=config.source,
        runtime_handle=config.runtime_handle,
        asset_path=added.asset_path,
        prim_path=added.prim_path,
        extra=f"static={added.static} imported_path={added.imported_path}",
    )
    return RuntimeObjectHandle(
        name=config.name,
        kind=config.kind,
        source=config.source,
        runtime_handle=config.runtime_handle,
        config=rigid_config,
        model=added,
    )


def _add_capsule_rope_object(
    stage, config: RuntimeObjectConfig, *, status_prefix: str | None
) -> RuntimeObjectHandle:
    """引用 capsule rope USD、摆放 root pose，并应用运行时物理覆盖。"""

    rope_config = _placed_capsule_rope_config(
        CapsuleRopeConfig.from_mapping(config.profile.raw or {}),
        config,
    )
    model = add_capsule_rope_reference(stage, rope_config)
    apply_root_pose_to_prim(stage, rope_config.prim_path, config.root_pose)
    physics_counts = apply_capsule_rope_runtime_physics(stage, rope_config)
    _print_object_status(
        status_prefix,
        name=config.name,
        kind=config.kind,
        source=config.source,
        runtime_handle=config.runtime_handle,
        asset_path=rope_config.asset_file(),
        prim_path=rope_config.prim_path,
        extra=(
            f"profile={config.object_profile} bodies={len(model['bodies'])} "
            f"segments={len(model['segments'])} joints={len(model['joints'])} "
            f"physics={physics_counts}"
        ),
    )
    return RuntimeObjectHandle(
        name=config.name,
        kind=config.kind,
        source=config.source,
        runtime_handle=config.runtime_handle,
        config=rope_config,
        model=model,
    )


def _placed_capsule_rope_config(
    object_config: CapsuleRopeConfig, runtime_config: RuntimeObjectConfig
) -> CapsuleRopeConfig:
    """把 object profile 中的资产路径和 stage prim 写入 rope runtime 配置。"""

    return replace(
        object_config,
        asset_path=runtime_config.profile.asset_path,
        prim_path=runtime_config.profile.prim_path,
        root_path=runtime_config.profile.root_path or object_config.root_path,
    )


def _rigid_object_config_from_runtime_object(
    config: RuntimeObjectConfig,
) -> RigidObjectConfig:
    """把通用 RuntimeObjectConfig 展开为 rigid runtime 模块需要的配置结构。"""

    data = expanded_object_mapping(
        ObjectSceneInstanceConfig(
            name=config.name,
            object_profile=config.object_profile,
            root_pose=config.root_pose,
            runtime_handle=config.runtime_handle,
        ),
        config.profile,
    )
    return RigidObjectConfig.from_mapping(data, index=0)


def _print_object_status(
    status_prefix: str | None,
    *,
    name: str,
    kind: str,
    source: str,
    runtime_handle: str | None,
    asset_path: Path,
    prim_path: str,
    extra: str,
) -> None:
    """按可选前缀输出对象导入状态，便于 smoke test 和日志排查。"""

    if status_prefix is None:
        return
    print(
        f"{status_prefix}_OBJECT "
        f"name={name} kind={kind} source={source} "
        f"handle={runtime_handle} asset={asset_path} prim_path={prim_path} {extra}",
        flush=True,
    )
