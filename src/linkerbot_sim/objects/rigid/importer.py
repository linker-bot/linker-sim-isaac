"""Rigid object 的 USD/URDF 导入、通用物理属性与 PhysX leaf 投影。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.assets.robot_import import (
    configure_urdf_import,
    prepare_session_newton_render_reference_asset,
)
from linkerbot_sim.configuration.objects import (
    ObjectMaterialConfig,
    ObjectPhysxMaterialConfig,
    RigidObjectPhysicsConfig,
)
from linkerbot_sim.isaac.physics.backend import (
    active_physics_backend,
    normalize_physics_backend,
)
from linkerbot_sim.objects.physics import apply_root_pose_to_prim
from linkerbot_sim.objects.rigid.config import RigidObjectConfig


@dataclass(frozen=True)
class AddedRigidObject:
    """已经放入 stage 的 rigid object 摘要。"""

    name: str
    asset_type: str
    asset_path: Path
    prim_path: str
    imported_path: str
    static: bool


def add_rigid_objects(
    stage,
    objects: Sequence[RigidObjectConfig],
    *,
    physics_backend: object | None = None,
    prepare_newton_render_topology: bool = False,
) -> tuple[AddedRigidObject, ...]:
    """把一组已校验 rigid object 配置加入当前 USD stage。"""

    return tuple(
        _add_rigid_object(
            stage,
            config,
            physics_backend=physics_backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
        )
        for config in objects
    )


def _add_rigid_object(
    stage,
    config: RigidObjectConfig,
    *,
    physics_backend: object | None = None,
    prepare_newton_render_topology: bool = False,
) -> AddedRigidObject:
    """按 USD/URDF source 导入一个 object，再按安全顺序应用 physics override。"""

    asset_path = config.asset_path.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Rigid object asset not found: {asset_path}")
    if config.asset_type == "usd":
        imported_path = _add_usd_reference(
            stage,
            config,
            asset_path,
            physics_backend=physics_backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
        )
    elif config.asset_type == "urdf":
        imported_path = _import_urdf_rigid_object(
            stage,
            config,
            asset_path,
            physics_backend=physics_backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
        )
    else:
        raise ValueError(f"Unsupported rigid object asset type: {config.asset_type}")
    # URDF fix_base 会生成 root fixed joint；不能再把其刚体设为 kinematic/static，
    # 否则 PhysX 会拒绝创建 static-static root joint。
    effective_fix_base = _effective_urdf_fix_base(config)
    _apply_rigid_object_physics(
        stage,
        imported_path,
        config.physics,
        freeze_rigid_bodies=not (config.asset_type == "urdf" and effective_fix_base),
        physics_backend=physics_backend,
    )
    return AddedRigidObject(
        name=config.name,
        asset_type=config.asset_type,
        asset_path=asset_path,
        prim_path=config.prim_path,
        imported_path=imported_path,
        static=config.physics.static,
    )


def _add_usd_reference(
    stage,
    config: RigidObjectConfig,
    asset_path: Path,
    *,
    physics_backend: object | None,
    prepare_newton_render_topology: bool,
) -> str:
    """在目标 prim path 创建 USD reference，并写入配置 root pose。"""

    from pxr import Sdf, UsdGeom

    prim_path = Sdf.Path(config.prim_path)
    xform = UsdGeom.Xform.Define(stage, prim_path)
    reference_asset = asset_path
    if prepare_newton_render_topology:
        # 本地 canonical order 必须先于 reference 生效，不能让 Hydra 短暂观察到
        # 资产 root 的旧 order；nested body 则由离线 wrapper 提前固化。
        apply_root_pose_to_prim(
            stage,
            str(prim_path),
            config.root_pose,
            prepare_newton_render_topology=True,
        )
        reference_asset = prepare_session_newton_render_reference_asset(
            asset_path,
            root_pose=config.root_pose,
            physics_backend=physics_backend,
        )
    if not xform.GetPrim().GetReferences().AddReference(str(reference_asset)):
        raise RuntimeError(f"Failed to reference rigid object USD: {reference_asset}")
    if not prepare_newton_render_topology:
        apply_root_pose_to_prim(stage, str(prim_path), config.root_pose)
    return str(prim_path)


def _import_urdf_rigid_object(
    stage,
    config: RigidObjectConfig,
    asset_path: Path,
    *,
    physics_backend: object | None,
    prepare_newton_render_topology: bool,
) -> str:
    """通过 Isaac URDF importer 创建 object，写 root pose，并移动到 canonical path。"""

    imported_path = configure_urdf_import(
        asset_path,
        drive_type=config.urdf_drive_type,
        get_articulation_root=False,
        fix_base=_effective_urdf_fix_base(config),
        asset_import_config=config.import_config,
        # Static/dynamic environment URDFs have no actuators; Importer 3.0 may
        # therefore emit only the shared `physics` layer instead of backend layers.
        allow_common_physics_variant=True,
        physics_backend=physics_backend,
        prepare_newton_render_topology=prepare_newton_render_topology,
        root_pose=config.root_pose,
    )
    apply_root_pose_to_prim(
        stage,
        imported_path,
        config.root_pose,
        prepare_newton_render_topology=prepare_newton_render_topology,
    )
    if imported_path != config.prim_path:
        _rename_prim(stage, imported_path, config.prim_path)
        imported_path = config.prim_path
    return imported_path


def _effective_urdf_fix_base(config: RigidObjectConfig) -> bool:
    """优先采用显式 importer 设置；未配置时固定声明为 static 的对象基座。"""

    configured = config.import_config.fix_base
    return config.physics.static if configured is None else configured


def _rename_prim(stage, source_path: str, target_path: str) -> None:
    """用 Kit ``MovePrim`` 原子移动导入结果，并校验 source/target path。"""

    from pxr import Sdf
    import omni.kit.commands

    source = Sdf.Path(source_path)
    target = Sdf.Path(target_path)
    if source == target:
        return
    if not stage.GetPrimAtPath(source).IsValid():
        raise RuntimeError(f"Imported rigid object prim was not created: {source_path}")
    if stage.GetPrimAtPath(target).IsValid():
        raise RuntimeError(f"Rigid object target prim already exists: {target_path}")
    parent_path = target.GetParentPath()
    if (
        parent_path != Sdf.Path.absoluteRootPath
        and not stage.GetPrimAtPath(parent_path).IsValid()
    ):
        stage.DefinePrim(parent_path)
    result = omni.kit.commands.execute(
        "MovePrim",
        path_from=str(source),
        path_to=str(target),
    )
    status = result[0] if isinstance(result, tuple) else bool(result)
    if not status:
        raise RuntimeError(
            f"Failed to move rigid object prim {source_path} to {target_path}"
        )


def _apply_rigid_object_physics(
    stage,
    root_path: str,
    physics: RigidObjectPhysicsConfig,
    *,
    freeze_rigid_bodies: bool = True,
    physics_backend: object | None = None,
) -> None:
    """应用 static 与 material override，并避免 URDF fixed-base 的 static-static 冲突。"""

    if not physics.static and physics.material is None and physics.physx is None:
        return
    backend = _resolved_physics_backend(physics_backend)
    if physics.static and freeze_rigid_bodies:
        _make_rigid_object_static(
            stage,
            root_path,
            physics_backend=backend,
        )
    # PhysX leaf 只有在 PhysX session 中才投影；Newton 只消费标准 UsdPhysics 材质。
    physx_material = (
        physics.physx.material
        if backend == "physx" and physics.physx is not None
        else None
    )
    if physics.material is not None or physx_material is not None:
        _apply_rigid_object_material(
            stage,
            root_path,
            physics.material,
            physx_material_config=physx_material,
            physics_backend=backend,
        )


def _make_rigid_object_static(
    stage,
    root_path: str,
    *,
    physics_backend: object | None = None,
) -> None:
    """把 object 子树中的刚体设为 kinematic；PhysX 同时关闭重力。"""

    backend = _resolved_physics_backend(physics_backend)
    from pxr import Sdf, Usd, UsdPhysics

    if backend == "physx":
        from pxr import PhysxSchema

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot apply static physics; rigid object prim not found: {root_path}"
        )
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateKinematicEnabledAttr().Set(True)
        if backend == "physx":
            physx_rigid_body_api = (
                PhysxSchema.PhysxRigidBodyAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            )
            physx_rigid_body_api.CreateDisableGravityAttr().Set(True)


def _apply_rigid_object_material(
    stage,
    root_path: str,
    material_config: ObjectMaterialConfig | None,
    *,
    physx_material_config: ObjectPhysxMaterialConfig | None = None,
    physics_backend: object | None = None,
) -> None:
    """把通用材质与当前 PhysX leaf 合并写入同一个 material prim。"""

    backend = _resolved_physics_backend(physics_backend)
    if backend != "physx":
        physx_material_config = None
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    if backend == "physx" and physx_material_config is not None:
        from pxr import PhysxSchema

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot apply material physics; rigid object prim not found: {root_path}"
        )
    material = UsdShade.Material.Define(
        stage,
        Sdf.Path(root_path).AppendPath("PhysicsMaterial"),
    )
    material_prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    if material_config is not None and material_config.static_friction is not None:
        material_api.CreateStaticFrictionAttr().Set(
            float(material_config.static_friction)
        )
    if material_config is not None and material_config.dynamic_friction is not None:
        material_api.CreateDynamicFrictionAttr().Set(
            float(material_config.dynamic_friction)
        )
    if material_config is not None and material_config.restitution is not None:
        material_api.CreateRestitutionAttr().Set(float(material_config.restitution))
    if physx_material_config is not None:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        physx_material_api.CreateFrictionCombineModeAttr().Set(
            physx_material_config.friction_combine_mode
        )

    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )


def _resolved_physics_backend(value: object | None) -> str:
    """解析显式后端；省略时读取 Isaac 当前 active engine。"""

    return (
        active_physics_backend() if value is None else normalize_physics_backend(value)
    )


__all__ = ["AddedRigidObject", "add_rigid_objects"]
