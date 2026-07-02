"""Capsule/cuboid 刚体链绳体运行时对象。

本模块只负责运行时读取对象 profile、引用已经生成好的 USD，并按对象 profile 写入
``src/linkerbot_sim`` 需要掌握的物理覆盖，例如接触材质和 solver iteration。绳体拓扑、几何、
质量、阻尼、关节限制和可视材质属于资产生成时的固有属性，由 ``tools`` 侧决定。

职责边界:
    * 引用已有 USD：把资产挂到当前 stage 的 ``prim_path`` 下，并收集 prim 句柄。
    * 应用运行时物理覆盖：接触摩擦、恢复系数和刚体 solver iteration。
    * 不生成 USD、不计算绳段/端块位置、不在运行时改变绳体拓扑。

函数中的 pxr/omni 依赖保持局部导入，便于在不启动 Isaac 的环境里解析配置和运行纯 Python 测试。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.objects.physics import (
    ObjectMaterialConfig,
    optional_mapping,
    optional_non_negative_int,
    optional_positive_int,
)
from linkerbot_sim.utils.paths import repo_path


CapsuleRopeMaterialConfig = ObjectMaterialConfig


@dataclass(frozen=True)
class CapsuleRopePhysicsConfig:
    """``src`` 运行时需要读取并写入当前 stage 的绳体物理属性。"""

    material: CapsuleRopeMaterialConfig | None = None
    solver_position_iterations: int | None = None
    solver_velocity_iterations: int | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "CapsuleRopePhysicsConfig":
        """解析 capsule rope 运行时物理覆盖。"""

        if data is None:
            return cls()
        allowed = {
            "material",
            "solver_position_iterations",
            "solver_velocity_iterations",
        }
        unsupported = set(data) - allowed
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"{label} contains unsupported keys: {names}")
        return cls(
            material=CapsuleRopeMaterialConfig.from_mapping(
                optional_mapping(data, "material", label),
                label=f"{label}.material",
            ),
            solver_position_iterations=optional_positive_int(
                data, "solver_position_iterations", label
            ),
            solver_velocity_iterations=optional_non_negative_int(
                data, "solver_velocity_iterations", label
            ),
        )

    def has_overrides(self) -> bool:
        """返回是否有任何需要写入 stage 的运行时物理覆盖。"""

        return (
            self.material is not None
            or self.solver_position_iterations is not None
            or self.solver_velocity_iterations is not None
        )


@dataclass(frozen=True)
class CapsuleRopeConfig:
    """运行时引用 capsule rope USD 所需的对象 profile。"""

    asset_path: str = (
        "assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda"
    )
    prim_path: str = "/World/CapsuleRope"
    root_path: str = "/CapsuleRope"
    physics: CapsuleRopePhysicsConfig = CapsuleRopePhysicsConfig()

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CapsuleRopeConfig":
        """从 ``configs/objects/*.yaml`` 映射构造绳体运行时对象配置。"""

        if "object" not in data:
            raise ValueError("Capsule rope config must contain top-level object section")
        if not isinstance(data["object"], Mapping):
            raise ValueError("object section must be a mapping")
        if "rope" in data:
            rope_data = data["rope"]
            if not isinstance(rope_data, Mapping):
                raise ValueError("rope section must be a mapping")
            _reject_generation_fields(rope_data, label="rope")
            if rope_data:
                raise ValueError(
                    "configs/objects capsule rope profiles no longer use rope; "
                    "put runtime physics under object.physics and generation fields "
                    "under tools/object_assets/flexible/rope"
                )

        object_cfg = dict(data["object"])
        _reject_generation_fields(object_cfg, label="object")
        return cls(
            asset_path=str(object_cfg.get("asset_path", cls.asset_path)),
            prim_path=str(object_cfg.get("prim_path", cls.prim_path)),
            root_path=str(object_cfg.get("root_path", cls.root_path)),
            physics=CapsuleRopePhysicsConfig.from_mapping(
                optional_mapping(object_cfg, "physics", "object"),
                label="object.physics",
            ),
        )

    def asset_file(self) -> Path:
        """返回引用绳体 USD 资产时使用的绝对路径。"""

        return repo_path(self.asset_path)

    def validate(self) -> None:
        """校验运行时引用配置。"""

        if not self.asset_path:
            raise ValueError("object.asset_path cannot be empty")
        if not self.prim_path.startswith("/"):
            raise ValueError("object.prim_path must be an absolute USD path")
        if not self.root_path.startswith("/"):
            raise ValueError("object.root_path must be an absolute USD path")


def collect_rope_model_prims(stage, root_path: str) -> dict[str, object]:
    """从已引用 stage 中收集绳体 prim。"""

    from pxr import Sdf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    bodies = []
    segments = []
    joints = []
    if root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(prim)
                if prim.GetName().startswith("segment_"):
                    segments.append(prim)
            if prim.IsA(UsdPhysics.Joint):
                joints.append(UsdPhysics.Joint(prim))
    return {"root": root, "segments": segments, "joints": joints, "bodies": bodies}


def add_capsule_rope_reference(stage, config: CapsuleRopeConfig) -> dict[str, object]:
    """把已生成的 capsule rope USD 资产引用到当前 stage。"""

    from pxr import Sdf, UsdGeom

    config.validate()
    asset_path = config.asset_file()
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Capsule rope asset does not exist: {asset_path}. "
            "Run tools/object_assets/flexible/rope/build_asset.py to generate it."
        )
    prim_path = Sdf.Path(config.prim_path)
    rope_xform = UsdGeom.Xform.Define(stage, prim_path)
    rope_xform.GetPrim().GetReferences().AddReference(str(asset_path))
    return collect_rope_model_prims(stage, str(prim_path))


def apply_capsule_rope_runtime_physics(
    stage, config: CapsuleRopeConfig
) -> dict[str, int]:
    """按 object profile 对已引用的 capsule rope 写入运行时物理覆盖。"""

    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(Sdf.Path(config.prim_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot apply capsule rope physics; prim not found: {config.prim_path}"
        )
    counts = {"collision_prims": 0, "rigid_bodies": 0}
    material = None
    if config.physics.material is not None:
        material = _define_runtime_material(
            stage,
            Sdf.Path(config.prim_path).AppendPath("RuntimePhysicsMaterial"),
            config.physics.material,
        )

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if material is not None:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
            counts["collision_prims"] += 1

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if (
            config.physics.solver_position_iterations is None
            and config.physics.solver_velocity_iterations is None
        ):
            counts["rigid_bodies"] += 1
            continue
        rigid_api = (
            PhysxSchema.PhysxRigidBodyAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        )
        if config.physics.solver_position_iterations is not None:
            rigid_api.CreateSolverPositionIterationCountAttr().Set(
                int(config.physics.solver_position_iterations)
            )
        if config.physics.solver_velocity_iterations is not None:
            rigid_api.CreateSolverVelocityIterationCountAttr().Set(
                int(config.physics.solver_velocity_iterations)
            )
        counts["rigid_bodies"] += 1
    return counts


def _define_runtime_material(stage, path, config: CapsuleRopeMaterialConfig):
    """在 stage 中创建绳体运行时物理材质 prim。"""

    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(prim)
    if config.static_friction is not None:
        material_api.CreateStaticFrictionAttr().Set(float(config.static_friction))
    if config.dynamic_friction is not None:
        material_api.CreateDynamicFrictionAttr().Set(float(config.dynamic_friction))
    if config.restitution is not None:
        material_api.CreateRestitutionAttr().Set(float(config.restitution))
    if config.friction_combine_mode is not None:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx_material_api.CreateFrictionCombineModeAttr().Set(
            config.friction_combine_mode
        )
    return material


def _reject_generation_fields(data: Mapping[str, object], *, label: str) -> None:
    """拒绝把资产生成期字段写进运行时 object profile。"""

    generation_fields = {
        "segments",
        "length",
        "radius",
        "center",
        "shape",
        "total_mass",
        "endpoint_box_mass",
        "endpoint_box_size",
        "endpoint_linear_damping",
        "endpoint_angular_damping",
        "segment_linear_damping",
        "segment_angular_damping",
        "bend_limit",
        "bend_limit_deg",
        "bend_stiffness",
        "bend_damping",
        "lock_twist",
        "twist_limit",
        "twist_limit_deg",
        "twist_stiffness",
        "twist_damping",
        "disable_adjacent_collisions",
        "endpoint_color",
        "rope_color",
        "env_static_friction",
        "env_dynamic_friction",
        "env_restitution",
    }
    intrinsic = set(data) & generation_fields
    if intrinsic:
        names = ", ".join(sorted(intrinsic))
        raise ValueError(
            f"{label} contains asset-generation field(s): {names}; "
            "move them to tools/object_assets/flexible/rope"
        )
    unsupported = set(data) - {
        "name",
        "kind",
        "source",
        "asset_path",
        "prim_path",
        "root_path",
        "physics",
    }
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"{label} contains unsupported keys: {names}")
