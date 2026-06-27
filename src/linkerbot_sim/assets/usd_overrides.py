"""导入机器人后的 USD / PhysX 参数覆盖。

MJCF/URDF importer 写出的默认 drive、摩擦和阻尼未必适合抓取实验。本模块在导入后统一
修正碰撞材料、刚体阻尼、关节摩擦和 drive 初值。

职责边界:
    * 直接写当前 stage 上的 USD/PhysX schema 属性，影响 reset 后的物理初始状态。
    * 不下发 articulation action，也不改变控制器的每步目标。
    * 不重新生成资产文件；覆盖只作用于当前 stage 中已经导入/引用的 prim。

调用顺序约定:
    调用方应在资产导入完成后、创建或 reset articulation runtime 前应用这些覆盖；runtime
    controller 仍会在后续写入每步目标和最终 drive gain。这样 USD 层提供稳定默认值，运行时
    层再按动作需要细化控制参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linkerbot_sim.robots.classification import component_for_name
from linkerbot_sim.robots.mimic import (
    mjcf_equality_follower_joint_names,
    parse_mjcf_joint_frictionloss,
)


@dataclass(frozen=True)
class PhysxOverrideConfig:
    """机器人 USD/PhysX 覆盖参数。

    该配置聚合材料、刚体阻尼、solver iteration 和 joint drive 初始值。它面向 USD 属性
    覆盖，不等同于运行时控制器增益；后者由 ``controllers`` 子包在 articulation view 上设置。

    输入字段:
        contact_static_friction/contact_dynamic_friction/contact_restitution: 接触材质参数。
        joint_friction: 缺省关节摩擦；MJCF 中有 frictionloss 时会优先使用 MJCF。
        rigid_body_linear_damping/rigid_body_angular_damping: 刚体阻尼。
        drive_stiffness_seed/drive_damping_seed: 主动关节 drive 初值。
        follower_drive_*_seed: mimic follower 关节 drive 初值。
        max_force/follower_max_force: drive 最大力/力矩，<=0 时相当于不施加。
    输出:
        传给 ``apply_robot_usd_overrides`` 后写入当前 USD stage。
    """

    contact_static_friction: float = 0.8
    contact_dynamic_friction: float = 0.6
    contact_restitution: float = 0.0
    joint_friction: float = 0.5
    follower_joint_friction: float | None = None
    rigid_body_linear_damping: float = 0.0
    rigid_body_angular_damping: float = 0.1
    drive_stiffness_seed: float = 1000.0
    drive_damping_seed: float = 50.0
    follower_drive_stiffness_seed: float = 50000.0
    follower_drive_damping_seed: float = 50.0
    max_force: float = 100.0
    follower_max_force: float | None = None


def make_physics_material(
    stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
):
    """创建带 PhysX friction combine mode 的物理材质。

    参数:
        stage: 当前 USD stage。
        path: 新材质 prim 路径。
        static_friction: 静摩擦系数。
        dynamic_friction: 动摩擦系数。
        restitution: 恢复系数。
    返回:
        ``UsdShade.Material`` 对象。
    """

    from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade

    # 接触材质作为独立 prim 复用到多个 collision 上，避免每个碰撞几何重复创建 schema。
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(prim)
    material_api.CreateStaticFrictionAttr().Set(float(static_friction))
    material_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    material_api.CreateRestitutionAttr().Set(float(restitution))
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_material_api.CreateFrictionCombineModeAttr().Set("average")
    return material


def apply_robot_usd_overrides(
    root_path: str,
    config: PhysxOverrideConfig | dict[str, PhysxOverrideConfig],
    *,
    driven_joint_names: list[str] | tuple[str, ...] = ("all",),
    mjcf_path: Path | None = None,
    material_path: str = "/World/PhysicsMaterials/RobotContactMaterial",
) -> dict[str, int]:
    """对机器人 USD 子树写入接触、阻尼、摩擦和 drive 初值。

    参数:
        root_path: 机器人导入后的 USD 子树根路径。
        config: 覆盖参数配置；可传单个配置，或传 ``{"arm": ..., "hand": ...}``。
        driven_joint_names: 需要启用 drive 的关节名；``("all",)`` 表示所有关节。
        mjcf_path: 可选 MJCF 路径，用于读取 mimic follower 和 frictionloss。
        material_path: 创建/复用的物理材质 prim 路径。
    返回:
        统计字典，包含处理到的 collision、rigid body、joint 和 driven joint 数量。
    """

    from isaacsim.core.utils.prims import get_prim_at_path
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    configs = _normalize_physx_configs(config)
    # arm/hand/default 可以使用不同摩擦材质；即使当前只传单个配置，也规范化成字典，
    # 让下面遍历 prim 时只需按 component 名选择。
    materials = {
        name: make_physics_material(
            stage,
            _component_material_path(material_path, name),
            item.contact_static_friction,
            item.contact_dynamic_friction,
            item.contact_restitution,
        )
        for name, item in configs.items()
    }
    friction_by_name = (
        parse_mjcf_joint_frictionloss(mjcf_path) if mjcf_path is not None else {}
    )
    follower_joint_names = (
        mjcf_equality_follower_joint_names(mjcf_path)
        if mjcf_path is not None
        else set()
    )
    drive_all_joints = (
        len(driven_joint_names) == 1 and str(driven_joint_names[0]).lower() == "all"
    )
    driven_names = set(driven_joint_names)

    root = get_prim_at_path(root_path)
    counts = {"collision_prims": 0, "rigid_bodies": 0, "joints": 0, "driven_joints": 0}
    for prim in Usd.PrimRange(root):
        group = component_for_name(prim.GetName())
        item = configs.get(group, configs["default"])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            # 物理材质绑定到 collision purpose，避免影响纯视觉材质，同时使用 stronger 级别
            # 覆盖 importer 或资产内部已有的弱绑定。
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                materials.get(group, materials["default"]),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            counts["collision_prims"] += 1

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # importer 生成的刚体阻尼常偏向通用场景；抓取/绳体接触需要可控阻尼，
            # 因此按部件统一写入，减少局部高频振荡。
            rigid_api = (
                PhysxSchema.PhysxRigidBodyAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            )
            rigid_api.CreateLinearDampingAttr().Set(
                float(item.rigid_body_linear_damping)
            )
            rigid_api.CreateAngularDampingAttr().Set(
                float(item.rigid_body_angular_damping)
            )
            counts["rigid_bodies"] += 1

        if prim.GetTypeName() not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue

        joint_api = (
            PhysxSchema.PhysxJointAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxJointAPI)
            else PhysxSchema.PhysxJointAPI.Apply(prim)
        )
        is_follower = prim.GetName() in follower_joint_names
        # mimic follower 的摩擦和 drive 初值允许单独配置，因为它们不是独立命令关节，
        # 过低会跟随不稳，过高又可能和主关节约束竞争。
        joint_friction = item.joint_friction
        if is_follower and item.follower_joint_friction is not None:
            joint_friction = item.follower_joint_friction
        joint_api.CreateJointFrictionAttr().Set(
            float(friction_by_name.get(prim.GetName(), joint_friction))
        )

        is_driven = drive_all_joints or prim.GetName() in driven_names or is_follower
        stiffness = (
            item.follower_drive_stiffness_seed
            if is_follower
            else item.drive_stiffness_seed
        )
        damping = (
            item.follower_drive_damping_seed if is_follower else item.drive_damping_seed
        )
        max_force = (
            item.follower_max_force
            if is_follower and item.follower_max_force is not None
            else item.max_force
        )
        drive_name = (
            "angular" if prim.GetTypeName() == "PhysicsRevoluteJoint" else "linear"
        )
        # 先在 USD 层写入 drive seed，让 articulation 创建时具备合理默认值；运行时
        # JointController 仍会再根据配置写入最终 gain/max effort。
        drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_name)
        drive_api.CreateTypeAttr().Set("force")
        drive_api.CreateStiffnessAttr().Set(float(stiffness if is_driven else 0.0))
        drive_api.CreateDampingAttr().Set(float(damping if is_driven else 0.0))
        drive_api.CreateMaxForceAttr().Set(
            float(max_force if is_driven and max_force > 0 else 0.0)
        )
        counts["joints"] += 1
        if is_driven:
            counts["driven_joints"] += 1
    return counts


def _normalize_physx_configs(
    config: PhysxOverrideConfig | dict[str, PhysxOverrideConfig],
) -> dict[str, PhysxOverrideConfig]:
    """把单配置或分组配置规范成含 default 的字典。"""

    # 单配置适合简单机器人；分组配置适合 AR5+L6 这种机械臂和灵巧手物理尺度差异较大的资产。
    if isinstance(config, PhysxOverrideConfig):
        return {"default": config}
    if not config:
        raise ValueError("PhysX override config cannot be empty")
    first = next(iter(config.values()))
    normalized = dict(config)
    normalized.setdefault("default", first)
    return normalized


def _component_material_path(base_path: str, group: str) -> str:
    """为 arm/hand 生成独立 physics material prim 路径。"""

    if group == "default":
        return base_path
    return f"{base_path}_{group}"


def disable_robot_gravity(root_path: str) -> list[str]:
    """关闭指定 USD 子树下所有刚体的重力。

    参数:
        root_path: 机器人或其它 USD 子树根路径。
    返回:
        被写入 ``disableGravity`` 的刚体 prim 路径列表。
    """

    from isaacsim.core.utils.prims import get_prim_at_path
    from pxr import PhysxSchema, Usd, UsdPhysics

    # 关闭机器人重力常用于固定基座或外部控制的 articulation，避免 drive 还没稳定前
    # 由重力引入额外下坠误差。
    root = get_prim_at_path(root_path)
    disabled_paths: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body_api = (
            PhysxSchema.PhysxRigidBodyAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        )
        rigid_body_api.GetDisableGravityAttr().Set(True)
        disabled_paths.append(str(prim.GetPath()))
    return disabled_paths


def set_runtime_gravity(world, gravity_z: float) -> tuple[np.ndarray, float]:
    """设置世界重力，并返回 Isaac runtime 实际方向和大小。

    参数:
        world: Isaac ``World`` 实例。
        gravity_z: z 方向重力加速度，单位 m/s^2，通常为负数。
    返回:
        ``(direction, magnitude)``，direction 为 shape ``(3,)`` 数组，magnitude 为标量。
    """

    physics_context = world.get_physics_context()
    physics_context.set_gravity(float(gravity_z))
    direction, magnitude = physics_context.get_gravity()
    return np.asarray(direction, dtype=float), float(magnitude)
