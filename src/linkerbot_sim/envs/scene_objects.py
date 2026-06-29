"""从 env 配置导入或引用物理世界中的环境物体。

本模块只负责“已有资产如何进入当前 USD stage”。对象自身如何生成、几何参数是什么，仍由
``configs/objects`` 和对应对象模块负责。这样 env 配置描述场景布置，object 配置描述对象资产。
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
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class SceneObjectMaterialConfig:
    """环境对象的可选物理材质覆盖；字段缺省表示不写入该 USD 属性。"""

    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None
    friction_combine_mode: str | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "SceneObjectMaterialConfig | None":
        if data is None:
            return None
        allowed_keys = {
            "static_friction",
            "dynamic_friction",
            "restitution",
            "friction_combine_mode",
        }
        unsupported_keys = set(data) - allowed_keys
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(f"{label} contains unsupported keys: {unsupported}")
        config = cls(
            static_friction=_optional_non_negative_float(
                data, "static_friction", label
            ),
            dynamic_friction=_optional_non_negative_float(
                data, "dynamic_friction", label
            ),
            restitution=_optional_non_negative_float(data, "restitution", label),
            friction_combine_mode=_optional_friction_combine_mode(data, label),
        )
        return config if config.has_overrides() else None

    def has_overrides(self) -> bool:
        return any(
            value is not None
            for value in (
                self.static_friction,
                self.dynamic_friction,
                self.restitution,
                self.friction_combine_mode,
            )
        )


@dataclass(frozen=True)
class SceneObjectPhysicsConfig:
    """环境层对场景物体的物理放置语义。"""

    static: bool = False
    material: SceneObjectMaterialConfig | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "SceneObjectPhysicsConfig":
        """解析 ``objects[].physics``；字段缺省表示不覆盖对应 USD 属性。"""

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
        material_data = _optional_mapping(data, "material", label)
        return cls(
            static=static,
            material=SceneObjectMaterialConfig.from_mapping(
                material_data, label=f"{label}.material"
            ),
        )


@dataclass(frozen=True)
class SceneObjectConfig:
    """单个环境物体的 stage 放置配置。"""

    name: str
    asset_type: str
    asset_path: Path
    prim_path: str
    root_pose: RootPoseConfig = RootPoseConfig()
    physics: SceneObjectPhysicsConfig = SceneObjectPhysicsConfig()
    urdf_drive_type: str = "none"
    import_config: AssetImportConfig = field(default_factory=AssetImportConfig)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object], *, index: int
    ) -> "SceneObjectConfig":
        """从 ``env.objects[]`` 项解析环境物体配置。"""

        if not isinstance(data, Mapping):
            raise ValueError(f"objects[{index}] must be a mapping")
        name = str(data.get("name", f"object_{index}"))
        if not name:
            raise ValueError(f"objects[{index}].name cannot be empty")
        asset_type = str(data.get("asset_type", "usd")).lower()
        if asset_type not in {"usd", "urdf"}:
            raise ValueError(
                f"objects[{index}].asset_type must be 'usd' or 'urdf', got {asset_type!r}"
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
                _optional_mapping(data, "root_pose", f"objects[{index}]")
            ),
            physics=SceneObjectPhysicsConfig.from_mapping(
                _optional_mapping(data, "physics", f"objects[{index}]"),
                label=f"objects[{index}].physics",
            ),
            urdf_drive_type=str(data.get("urdf_drive_type", "none")),
            import_config=AssetImportConfig.from_mapping(
                _optional_mapping(data, "import", f"objects[{index}]"),
                label=f"objects[{index}].import",
            ),
        )


@dataclass(frozen=True)
class AddedSceneObject:
    """已放入 stage 的环境物体摘要。"""

    name: str
    asset_type: str
    asset_path: Path
    prim_path: str
    imported_path: str
    static: bool


def scene_objects_from_env_config(
    config: Mapping[str, object],
) -> tuple[SceneObjectConfig, ...]:
    """解析 env YAML 顶层 ``objects`` 列表。"""

    objects = config.get("objects", ())
    if objects is None:
        return ()
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("objects must be a sequence")
    return tuple(
        SceneObjectConfig.from_mapping(item, index=index)
        for index, item in enumerate(objects)
    )


def add_scene_objects(
    stage, objects: Sequence[SceneObjectConfig]
) -> tuple[AddedSceneObject, ...]:
    """把环境物体加入当前 USD stage。"""

    return tuple(_add_scene_object(stage, config) for config in objects)


def _add_scene_object(stage, config: SceneObjectConfig) -> AddedSceneObject:
    asset_path = config.asset_path.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Scene object asset not found: {asset_path}")
    if config.asset_type == "usd":
        imported_path = _add_usd_reference(stage, config, asset_path)
    elif config.asset_type == "urdf":
        imported_path = _import_urdf_scene_object(stage, config, asset_path)
    else:
        raise ValueError(f"Unsupported scene object asset type: {config.asset_type}")
    # URDF importer 的 fix_base=True 会生成 root fixed joint；不要再把这些刚体设成
    # kinematic/static，否则 PhysX 会看到 static-static joint 并拒绝创建 root_joint。
    _apply_scene_object_physics(
        stage,
        imported_path,
        config.physics,
        freeze_rigid_bodies=not (
            config.asset_type == "urdf" and config.physics.static
        ),
    )
    return AddedSceneObject(
        name=config.name,
        asset_type=config.asset_type,
        asset_path=asset_path,
        prim_path=config.prim_path,
        imported_path=imported_path,
        static=config.physics.static,
    )


def _add_usd_reference(stage, config: SceneObjectConfig, asset_path: Path) -> str:
    from pxr import Sdf, UsdGeom

    prim_path = Sdf.Path(config.prim_path)
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.GetPrim().GetReferences().AddReference(str(asset_path))
    _apply_root_pose_to_prim(stage, str(prim_path), config.root_pose)
    return str(prim_path)


def _import_urdf_scene_object(stage, config: SceneObjectConfig, asset_path: Path) -> str:
    imported_path = configure_urdf_import(
        asset_path,
        create_physics_scene=False,
        drive_type=config.urdf_drive_type,
        get_articulation_root=False,
        make_default_prim=False,
        fix_base=config.physics.static,
        asset_import_config=config.import_config,
    )
    _apply_root_pose_to_prim(stage, imported_path, config.root_pose)
    if imported_path != config.prim_path:
        _rename_prim(stage, imported_path, config.prim_path)
        imported_path = config.prim_path
    return imported_path


def _rename_prim(stage, source_path: str, target_path: str) -> None:
    from pxr import Sdf
    import omni.kit.commands

    source = Sdf.Path(source_path)
    target = Sdf.Path(target_path)
    if source == target:
        return
    if not stage.GetPrimAtPath(source).IsValid():
        raise RuntimeError(f"Imported scene object prim was not created: {source_path}")
    if stage.GetPrimAtPath(target).IsValid():
        raise RuntimeError(f"Scene object target prim already exists: {target_path}")
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
            f"Failed to move scene object prim {source_path} to {target_path}"
        )


def _apply_scene_object_physics(
    stage,
    root_path: str,
    physics: SceneObjectPhysicsConfig,
    *,
    freeze_rigid_bodies: bool = True,
) -> None:
    if physics.static and freeze_rigid_bodies:
        _make_scene_object_static(stage, root_path)
    if physics.material is not None:
        _apply_scene_object_material(stage, root_path, physics.material)


def _make_scene_object_static(stage, root_path: str) -> None:
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot apply static physics; scene object prim not found: {root_path}"
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


def _apply_scene_object_material(
    stage, root_path: str, material_config: SceneObjectMaterialConfig
) -> None:
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot apply material physics; scene object prim not found: {root_path}"
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


def _apply_root_pose_to_prim(stage, prim_path: str, pose: RootPoseConfig) -> None:
    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid():
        raise RuntimeError(
            f"Cannot apply root_pose; scene object prim not found: {prim_path}"
        )
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose.xyz))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*_radians_to_degrees(pose.rpy)))


def _radians_to_degrees(values: Sequence[float]) -> tuple[float, float, float]:
    import math

    return tuple(float(value) * 180.0 / math.pi for value in values)


def _optional_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_non_negative_float(
    data: Mapping[str, object], key: str, parent_label: str
) -> float | None:
    if key not in data:
        return None
    value = float(data[key])
    if value < 0.0:
        raise ValueError(f"{parent_label}.{key} cannot be negative")
    return value


def _optional_friction_combine_mode(
    data: Mapping[str, object], parent_label: str
) -> str | None:
    if "friction_combine_mode" not in data:
        return None
    value = str(data["friction_combine_mode"]).lower()
    allowed = {"average", "min", "multiply", "max"}
    if value not in allowed:
        raise ValueError(
            f"{parent_label}.friction_combine_mode must be one of "
            f"{sorted(allowed)}, got {value!r}"
        )
    return value
