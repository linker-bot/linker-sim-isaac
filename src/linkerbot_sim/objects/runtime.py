"""env objects 到 runtime object handle 的通用导入分发。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.scenes import ObjectInstanceSettings
from linkerbot_sim.configuration.objects import (
    DynamicChainObjectProfileConfig,
    ObjectProfileConfig,
    ObjectStateSummaryConfig,
    RigidObjectProfileConfig,
)
from linkerbot_sim.objects.rigid.config import RigidObjectConfig
from linkerbot_sim.objects.rigid.importer import add_rigid_objects
from linkerbot_sim.objects.dynamic_chain.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
    apply_capsule_rope_runtime_physics,
)
from linkerbot_sim.isaac.physics.backend import normalize_physics_backend
from linkerbot_sim.isaac.physics.newton.render import prepare_newton_render_subtree
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class RuntimeObjectConfig:
    """env ``objects[]`` 中一项对象声明。"""

    name: str
    root_pose: RootPoseConfig
    object_profile: str
    profile: ObjectProfileConfig
    prim_path: str
    runtime_handle: str | None = None

    @property
    def kind(self) -> Literal["rigid", "dynamic_chain"]:
        """从判别联合 profile 派生对象类型，避免运行时保存第二份事实。"""

        return self.profile.kind

    @property
    def source(self) -> Literal["usd", "urdf"]:
        """从判别联合 profile 派生资产来源。"""

        return self.profile.source


@dataclass(frozen=True)
class RuntimeObjectHandle:
    """已导入 stage 的对象句柄。"""

    name: str
    kind: str
    source: str
    runtime_handle: str | None
    config: object
    model: object
    state_summary: ObjectStateSummaryConfig = ObjectStateSummaryConfig()


def runtime_object_prim_path(handle: object) -> str | None:
    """返回 runtime object 的 canonical root prim path。

    对象导入结果与配置都可能承载路径；统一在对象层解析可避免遥测、快照、reset 和碰撞
    各自维护一套反射逻辑。``prim_path`` 是 stage 上的最终路径，只有后端未暴露它时才读取
    ``imported_path``；dynamic-chain model 的 USD root 则由 mapping 中的 ``root`` 提供。
    """

    for source in (getattr(handle, "model", None), getattr(handle, "config", None)):
        if source is None:
            continue
        for attribute in ("prim_path", "imported_path"):
            value = getattr(source, attribute, None)
            if value is not None:
                return str(value)
        if isinstance(source, Mapping):
            value = source.get("prim_path")
            if value is not None:
                return str(value)
            root = source.get("root")
            path_getter = getattr(root, "GetPath", None)
            if callable(path_getter):
                return str(path_getter())
    return None


def runtime_objects_from_settings(
    instances: Sequence[ObjectInstanceSettings],
) -> tuple[RuntimeObjectConfig, ...]:
    """把场景 owner 已解析并绑定 profile 的实例投影为运行时对象。"""

    return tuple(_runtime_object_from_settings(item) for item in instances)


def add_runtime_objects(
    stage,
    objects: Sequence[RuntimeObjectConfig],
    *,
    physics_backend: object,
    prepare_newton_render_topology: bool,
    status_prefix: str | None = None,
) -> tuple[RuntimeObjectHandle, ...]:
    """把 env 中声明的对象导入 stage。"""

    backend = normalize_physics_backend(physics_backend)
    if prepare_newton_render_topology and backend != "newton":
        raise RuntimeError(
            "Newton render topology intent requires physics_backend='newton'"
        )
    return tuple(
        add_runtime_object(
            stage,
            config=config,
            physics_backend=backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
            status_prefix=status_prefix,
        )
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
    physics_backend: object,
    prepare_newton_render_topology: bool,
    status_prefix: str | None = None,
) -> RuntimeObjectHandle:
    """导入单个 runtime object。"""

    backend = normalize_physics_backend(physics_backend)
    if prepare_newton_render_topology and backend != "newton":
        raise RuntimeError(
            "Newton render topology intent requires physics_backend='newton'"
        )
    if isinstance(config.profile, RigidObjectProfileConfig):
        result = _add_rigid_object(
            stage,
            config,
            physics_backend=backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
            status_prefix=status_prefix,
        )
    elif isinstance(config.profile, DynamicChainObjectProfileConfig):
        result = _add_capsule_rope_object(
            stage,
            config,
            physics_backend=backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
            status_prefix=status_prefix,
        )
    else:
        raise ValueError(
            f"Unsupported runtime object kind/source: {config.kind!r}/{config.source!r}"
        )
    if prepare_newton_render_topology:
        prim_path = runtime_object_prim_path(result)
        if prim_path is None:
            raise RuntimeError(
                f"Newton object {config.name!r} did not expose an imported prim path"
            )
        # rigid 与 dynamic-chain 都在这里完成 reference/root pose/physics author。
        # 下一个资产导入或 renderer population 之前立即固定所有 body 的 op topology。
        prepare_newton_render_subtree(
            stage=stage,
            subtree_root=prim_path,
        )
    return result


def _runtime_object_from_settings(
    instance: ObjectInstanceSettings,
) -> RuntimeObjectConfig:
    """把 env objects[] 实例和 object profile 合并成运行时对象配置。"""

    profile = instance.resolved_profile
    if not isinstance(
        profile, (RigidObjectProfileConfig, DynamicChainObjectProfileConfig)
    ):
        raise TypeError(
            f"object instance {instance.name!r} requires a resolved ObjectProfileConfig"
        )

    return RuntimeObjectConfig(
        name=instance.name,
        root_pose=RootPoseConfig(
            xyz=instance.root_pose.xyz, rpy=instance.root_pose.rpy
        ),
        runtime_handle=None,
        object_profile=instance.object_profile,
        profile=profile,
        prim_path=instance.prim_path,
    )


def _add_rigid_object(
    stage,
    config: RuntimeObjectConfig,
    *,
    physics_backend: str,
    prepare_newton_render_topology: bool,
    status_prefix: str | None,
) -> RuntimeObjectHandle:
    """导入 rigid object，并返回统一 RuntimeObjectHandle。"""

    rigid_config = _rigid_object_config_from_runtime_object(config)
    added = add_rigid_objects(
        stage,
        (rigid_config,),
        physics_backend=physics_backend,
        prepare_newton_render_topology=prepare_newton_render_topology,
    )[0]
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
        state_summary=ObjectStateSummaryConfig(),
    )


def _add_capsule_rope_object(
    stage,
    config: RuntimeObjectConfig,
    *,
    physics_backend: str,
    prepare_newton_render_topology: bool,
    status_prefix: str | None,
) -> RuntimeObjectHandle:
    """引用 capsule rope USD、摆放 root pose，并应用运行时物理覆盖。"""

    profile = config.profile
    if not isinstance(profile, DynamicChainObjectProfileConfig):
        raise TypeError("dynamic-chain runtime object requires its typed profile")
    rope_config = CapsuleRopeConfig(
        asset_path=profile.asset_path,
        prim_path=config.prim_path,
        root_path=profile.root_path,
        physics=profile.physics,
    )
    model = add_capsule_rope_reference(
        stage,
        rope_config,
        physics_backend=physics_backend,
        root_pose=config.root_pose,
        prepare_newton_render_topology=prepare_newton_render_topology,
    )
    physics_counts = apply_capsule_rope_runtime_physics(
        stage,
        rope_config,
        physics_backend=physics_backend,
    )
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
        state_summary=profile.state_summary,
    )


def _rigid_object_config_from_runtime_object(
    config: RuntimeObjectConfig,
) -> RigidObjectConfig:
    """把通用 RuntimeObjectConfig 展开为 rigid runtime 模块需要的配置结构。"""

    profile = config.profile
    if not isinstance(profile, RigidObjectProfileConfig):
        raise TypeError("rigid runtime object requires its typed profile")
    return RigidObjectConfig(
        name=config.name,
        asset_type=profile.source,
        asset_path=repo_path(profile.asset_path),
        prim_path=config.prim_path,
        root_pose=config.root_pose,
        physics=profile.physics,
        planning_collision=profile.planning_collision,
        urdf_drive_type=profile.urdf_drive_type,
        import_config=profile.import_config,
    )


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
