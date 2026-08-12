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

from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.assets.robot_import import (
    prepare_session_newton_render_reference_asset,
)
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.objects import (
    CapsuleRopePhysicsConfig,
    ObjectMaterialConfig,
    ObjectPhysxMaterialConfig,
)
from linkerbot_sim.isaac.physics.backend import (
    active_physics_backend,
    normalize_physics_backend,
)
from linkerbot_sim.objects.physics import apply_root_pose_to_prim
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class CapsuleRopeConfig:
    """运行时引用 capsule rope USD 所需的对象 profile。"""

    asset_path: str = (
        "assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda"
    )
    prim_path: str = "/World/CapsuleRope"
    root_path: str = "/CapsuleRope"
    physics: CapsuleRopePhysicsConfig = CapsuleRopePhysicsConfig()

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


def add_capsule_rope_reference(
    stage,
    config: CapsuleRopeConfig,
    *,
    physics_backend: object,
    root_pose: RootPoseConfig | None = None,
    prepare_newton_render_topology: bool = False,
) -> dict[str, object]:
    """把已生成的 capsule rope USD 资产引用到当前 stage。"""

    from pxr import Sdf, UsdGeom

    config.validate()
    asset_path = config.asset_file()
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Capsule rope asset does not exist: {asset_path}. "
            "Run tools/object_assets/flexible/rope/build_asset.py to generate it."
        )
    backend = _resolved_physics_backend(physics_backend)
    if prepare_newton_render_topology:
        if backend != "newton":
            raise RuntimeError(
                "Newton render topology intent requires physics_backend='newton'"
            )
        if not isinstance(root_pose, RootPoseConfig):
            raise TypeError(
                "Newton render capsule rope requires a resolved RootPoseConfig"
            )
        # 先在离线 wrapper 中冻结所有 segment/body 的最终 world-matrix op，避免
        # AddReference 后 Hydra 短暂观察源资产的 translate/scale xformOpOrder。
        reference_asset = prepare_session_newton_render_reference_asset(
            asset_path,
            source_path=config.root_path,
            root_pose=root_pose,
            physics_backend=backend,
        )
    else:
        reference_asset = asset_path
    prim_path = Sdf.Path(config.prim_path)
    rope_xform = UsdGeom.Xform.Define(stage, prim_path)
    if root_pose is not None and prepare_newton_render_topology:
        apply_root_pose_to_prim(
            stage,
            str(prim_path),
            root_pose,
            prepare_newton_render_topology=True,
        )
    if (
        not rope_xform.GetPrim()
        .GetReferences()
        .AddReference(str(reference_asset), config.root_path)
    ):
        raise RuntimeError(
            "Failed to reference capsule rope USD root: "
            f"{reference_asset}:{config.root_path}"
        )
    if root_pose is not None and not prepare_newton_render_topology:
        apply_root_pose_to_prim(stage, str(prim_path), root_pose)
    return collect_rope_model_prims(stage, str(prim_path))


def apply_capsule_rope_runtime_physics(
    stage,
    config: CapsuleRopeConfig,
    *,
    physics_backend: object | None = None,
) -> dict[str, int]:
    """按 object profile 对已引用的 capsule rope 写入运行时物理覆盖。"""

    backend = _resolved_physics_backend(physics_backend)

    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    # 后端 leaf 在这里一次性裁剪；Newton 后续逻辑不会读取任何 PhysX 配置。
    physx_config = config.physics.physx if backend == "physx" else None
    physx_material = physx_config.material if physx_config is not None else None
    physx_solver = physx_config.solver if physx_config is not None else None
    if physx_solver is not None:
        from pxr import PhysxSchema

    root = stage.GetPrimAtPath(Sdf.Path(config.prim_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot apply capsule rope physics; prim not found: {config.prim_path}"
        )
    counts = {
        "collision_prims": 0,
        "rigid_bodies": 0,
    }
    material = None
    if config.physics.material is not None or physx_material is not None:
        material = _define_runtime_material(
            stage,
            Sdf.Path(config.prim_path).AppendPath("RuntimePhysicsMaterial"),
            config.physics.material,
            physx_material_config=physx_material,
            physics_backend=backend,
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
        if physx_solver is None:
            counts["rigid_bodies"] += 1
            continue
        rigid_api = (
            PhysxSchema.PhysxRigidBodyAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        )
        if physx_solver.position_iterations is not None:
            rigid_api.CreateSolverPositionIterationCountAttr().Set(
                int(physx_solver.position_iterations)
            )
        if physx_solver.velocity_iterations is not None:
            rigid_api.CreateSolverVelocityIterationCountAttr().Set(
                int(physx_solver.velocity_iterations)
            )
        counts["rigid_bodies"] += 1
    return counts


def _define_runtime_material(
    stage,
    path,
    config: ObjectMaterialConfig | None,
    *,
    physx_material_config: ObjectPhysxMaterialConfig | None = None,
    physics_backend: object | None = None,
):
    """把通用字段与当前 PhysX leaf 写入同一个绳体物理材质。"""

    backend = _resolved_physics_backend(physics_backend)
    if backend != "physx":
        physx_material_config = None
    from pxr import UsdPhysics, UsdShade

    if physx_material_config is not None:
        from pxr import PhysxSchema

    material = UsdShade.Material.Define(stage, path)
    prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(prim)
    if config is not None and config.static_friction is not None:
        material_api.CreateStaticFrictionAttr().Set(float(config.static_friction))
    if config is not None and config.dynamic_friction is not None:
        material_api.CreateDynamicFrictionAttr().Set(float(config.dynamic_friction))
    if config is not None and config.restitution is not None:
        material_api.CreateRestitutionAttr().Set(float(config.restitution))
    if physx_material_config is not None:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx_material_api.CreateFrictionCombineModeAttr().Set(
            physx_material_config.friction_combine_mode
        )
    return material


def _resolved_physics_backend(value: object | None) -> str:
    """解析显式后端；省略时读取 Isaac 当前 active engine。"""

    return (
        active_physics_backend() if value is None else normalize_physics_backend(value)
    )
