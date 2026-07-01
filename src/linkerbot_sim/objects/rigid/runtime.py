"""Rigid object runtime import and physics helpers.

本模块只负责 rigid object 的运行时导入和物理覆盖。env 配置描述对象实例摆放；
object profile 描述资产路径、导入参数和运行时物理属性。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from linkerbot_sim.assets.robot_loader import (
    AssetImportConfig,
    RootPoseConfig,
    configure_urdf_import,
)
from linkerbot_sim.objects.config import (
    ObjectProfileConfig,
    expanded_object_mapping,
    object_scene_instances_from_env_config,
)
from linkerbot_sim.objects.physics import (
    ObjectMaterialConfig,
    apply_root_pose_to_prim,
    optional_mapping,
)
from linkerbot_sim.utils.paths import repo_path


RigidObjectMaterialConfig = ObjectMaterialConfig


@dataclass(frozen=True)
class RigidObjectPhysicsConfig:
    """Rigid object 的运行时物理语义。"""

    static: bool = False
    material: RigidObjectMaterialConfig | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RigidObjectPhysicsConfig":
        """解析 profile/env 合并后的 ``physics``；字段缺省表示不覆盖对应 USD 属性。"""

        if data is None:
            return cls()
        unsupported_keys = set(data) - {
            "static",
            "material",
        }
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(f"{label} contains unsupported keys: {unsupported}")
        static = data.get("static", False)
        if not isinstance(static, bool):
            raise ValueError(f"{label}.static must be a boolean")
        material_data = optional_mapping(data, "material", label)
        return cls(
            static=static,
            material=RigidObjectMaterialConfig.from_mapping(
                material_data, label=f"{label}.material"
            ),
        )


@dataclass(frozen=True)
class RigidObjectConfig:
    """单个 rigid object 的 stage 放置配置。"""

    name: str
    asset_type: str
    asset_path: Path
    prim_path: str
    root_pose: RootPoseConfig = RootPoseConfig()
    physics: RigidObjectPhysicsConfig = RigidObjectPhysicsConfig()
    urdf_drive_type: str = "none"
    import_config: AssetImportConfig = field(default_factory=AssetImportConfig)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object], *, index: int
    ) -> "RigidObjectConfig":
        """从 profile 展开后的低层导入配置解析 rigid object。"""

        if not isinstance(data, Mapping):
            raise ValueError(f"objects[{index}] must be a mapping")
        if "object_profile" in data:
            instance = object_scene_instances_from_env_config({"objects": [data]})[0]
            profile = ObjectProfileConfig.from_profile(instance.object_profile)
            if profile.kind != "rigid":
                raise ValueError(
                    f"objects[{index}].object_profile {instance.object_profile!r} "
                    f"is {profile.kind!r}; RigidObjectConfig only supports rigid objects"
                )
            data = expanded_object_mapping(instance, profile)

        name = str(data.get("name", f"object_{index}"))
        if not name:
            raise ValueError(f"objects[{index}].name cannot be empty")
        if "asset_type" in data:
            raise ValueError(
                f"objects[{index}].asset_type is removed; use source instead"
            )
        asset_type = str(data.get("source", "usd")).lower()
        if asset_type not in {"usd", "urdf"}:
            raise ValueError(
                f"objects[{index}].source must be 'usd' or 'urdf', got {asset_type!r}"
            )
        if "asset_path" not in data:
            raise ValueError(f"objects[{index}].asset_path is required")
        if "prim_path" not in data:
            raise ValueError(f"objects[{index}].prim_path is required")
        prim_path = str(data["prim_path"])
        if not prim_path.startswith("/"):
            raise ValueError(f"objects[{index}].prim_path must be an absolute USD path")
        return cls(
            name=name,
            asset_type=asset_type,
            asset_path=repo_path(str(data["asset_path"])),
            prim_path=prim_path,
            root_pose=RootPoseConfig.from_mapping(
                optional_mapping(data, "root_pose", f"objects[{index}]")
            ),
            physics=RigidObjectPhysicsConfig.from_mapping(
                optional_mapping(data, "physics", f"objects[{index}]"),
                label=f"objects[{index}].physics",
            ),
            urdf_drive_type=str(data.get("urdf_drive_type", "none")),
            import_config=AssetImportConfig.from_mapping(
                optional_mapping(data, "import", f"objects[{index}]"),
                label=f"objects[{index}].import",
            ),
        )


@dataclass(frozen=True)
class AddedRigidObject:
    """已放入 stage 的 rigid object 摘要。"""

    name: str
    asset_type: str
    asset_path: Path
    prim_path: str
    imported_path: str
    static: bool


def rigid_objects_from_env_config(
    config: Mapping[str, object],
) -> tuple[RigidObjectConfig, ...]:
    """解析 env YAML 顶层 ``objects`` 列表。"""

    rigid_objects: list[RigidObjectConfig] = []
    for index, item in enumerate(object_scene_instances_from_env_config(config)):
        profile = ObjectProfileConfig.from_profile(item.object_profile)
        if profile.kind != "rigid":
            continue
        rigid_objects.append(
            RigidObjectConfig.from_mapping(
                expanded_object_mapping(item, profile),
                index=index,
            )
        )
    return tuple(rigid_objects)


def add_rigid_objects(
    stage, objects: Sequence[RigidObjectConfig]
) -> tuple[AddedRigidObject, ...]:
    """把 rigid objects 加入当前 USD stage。"""

    return tuple(_add_rigid_object(stage, config) for config in objects)


def _add_rigid_object(stage, config: RigidObjectConfig) -> AddedRigidObject:
    """按 asset_type 导入单个 rigid object，并应用运行时物理覆盖。"""

    asset_path = config.asset_path.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Rigid object asset not found: {asset_path}")
    if config.asset_type == "usd":
        imported_path = _add_usd_reference(stage, config, asset_path)
    elif config.asset_type == "urdf":
        imported_path = _import_urdf_rigid_object(stage, config, asset_path)
    else:
        raise ValueError(f"Unsupported rigid object asset type: {config.asset_type}")
    # URDF importer 的 fix_base=True 会生成 root fixed joint；不要再把这些刚体设成
    # kinematic/static，否则 PhysX 会看到 static-static joint 并拒绝创建 root_joint。
    _apply_rigid_object_physics(
        stage,
        imported_path,
        config.physics,
        freeze_rigid_bodies=not (
            config.asset_type == "urdf" and config.physics.static
        ),
    )
    return AddedRigidObject(
        name=config.name,
        asset_type=config.asset_type,
        asset_path=asset_path,
        prim_path=config.prim_path,
        imported_path=imported_path,
        static=config.physics.static,
    )


def _add_usd_reference(stage, config: RigidObjectConfig, asset_path: Path) -> str:
    """把 USD 资产以 reference 形式挂到目标 prim_path，并应用 root_pose。"""

    from pxr import Sdf, UsdGeom

    prim_path = Sdf.Path(config.prim_path)
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.GetPrim().GetReferences().AddReference(str(asset_path))
    apply_root_pose_to_prim(stage, str(prim_path), config.root_pose)
    return str(prim_path)


def _import_urdf_rigid_object(stage, config: RigidObjectConfig, asset_path: Path) -> str:
    """通过 Isaac URDF importer 导入 rigid object，并移动到目标 prim_path。"""

    imported_path = configure_urdf_import(
        asset_path,
        create_physics_scene=False,
        drive_type=config.urdf_drive_type,
        get_articulation_root=False,
        make_default_prim=False,
        fix_base=config.physics.static,
        asset_import_config=config.import_config,
    )
    apply_root_pose_to_prim(stage, imported_path, config.root_pose)
    if imported_path != config.prim_path:
        _rename_prim(stage, imported_path, config.prim_path)
        imported_path = config.prim_path
    return imported_path


def _rename_prim(stage, source_path: str, target_path: str) -> None:
    """把 importer 生成的 prim 移到 profile 指定的目标路径。"""

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
        "MovePrim", path_from=str(source), path_to=str(target)
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
) -> None:
    """应用 rigid object 的 static/material 运行时覆盖。"""

    if physics.static and freeze_rigid_bodies:
        _make_rigid_object_static(stage, root_path)
    if physics.material is not None:
        _apply_rigid_object_material(stage, root_path, physics.material)


def _make_rigid_object_static(stage, root_path: str) -> None:
    """把 root_path 子树中的刚体设为 kinematic 并关闭重力。"""

    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

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
        physx_rigid_body_api = (
            PhysxSchema.PhysxRigidBodyAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        )
        physx_rigid_body_api.CreateDisableGravityAttr().Set(True)


def _apply_rigid_object_material(
    stage, root_path: str, material_config: RigidObjectMaterialConfig
) -> None:
    """创建物理材质并绑定到 rigid object 子树中的 collision prim。"""

    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot apply material physics; rigid object prim not found: {root_path}"
        )
    material = UsdShade.Material.Define(
        stage, Sdf.Path(root_path).AppendPath("PhysicsMaterial")
    )
    material_prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    if material_config.static_friction is not None:
        material_api.CreateStaticFrictionAttr().Set(
            float(material_config.static_friction)
        )
    if material_config.dynamic_friction is not None:
        material_api.CreateDynamicFrictionAttr().Set(
            float(material_config.dynamic_friction)
        )
    if material_config.restitution is not None:
        material_api.CreateRestitutionAttr().Set(float(material_config.restitution))
    if material_config.friction_combine_mode is not None:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        physx_material_api.CreateFrictionCombineModeAttr().Set(
            material_config.friction_combine_mode
        )

    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
