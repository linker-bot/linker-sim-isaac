"""导入机器人后的 USD / PhysX 参数覆盖。

MJCF/URDF importer 写出的默认 drive、摩擦和阻尼未必适合抓取实验。本模块在导入后统一
修正碰撞材料、刚体阻尼、关节摩擦和 drive 初值。

职责边界:
    * 直接写当前 stage 上的 USD/PhysX schema 属性，影响 reset 后的物理初始状态。
    * 不下发 articulation action，也不改变控制器的每步目标。
    * 不重新生成资产文件；覆盖只作用于当前 stage 中已经导入/引用的 prim。

调用顺序约定:
    调用方应在资产导入完成后、创建或 reset articulation runtime 前应用这些覆盖；runtime
    controller 仍会在后续写入每步目标和最终 drive gain。joint friction 只在本层写入一次，
    避免运行时 ArticulationView 触发 Isaac 5.1 已弃用的 tensor friction API。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linkerbot_sim.robots.classification import (
    RobotComponentMapping,
    component_for_name,
)
from linkerbot_sim.robots.mimic.assets import parse_asset_mimic_relations
from linkerbot_sim.robots.mimic.mjcf import parse_mjcf_joint_frictionloss


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
    contact_material_override: bool = True
    friction_combine_mode: str | None = "average"
    joint_friction: object = 0.5
    follower_joint_friction: object | None = None
    rigid_body_linear_damping: float = 0.0
    rigid_body_angular_damping: float = 0.1
    drive_stiffness_seed: object = 1000.0
    drive_damping_seed: object = 50.0
    follower_drive_stiffness_seed: object = 50000.0
    follower_drive_damping_seed: object = 50.0
    max_force: object = 100.0
    follower_max_force: object | None = None


def make_physics_material(
    stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
    friction_combine_mode: str | None = "average",
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
    if friction_combine_mode is not None:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx_material_api.CreateFrictionCombineModeAttr().Set(friction_combine_mode)
    return material


def apply_robot_usd_overrides(
    root_path: str,
    config: PhysxOverrideConfig | dict[str, PhysxOverrideConfig],
    *,
    driven_joint_names: list[str] | tuple[str, ...] = ("all",),
    mjcf_path: Path | None = None,
    mimic_path: Path | None = None,
    material_path: str | None = None,
    component_mapping: RobotComponentMapping | None = None,
    native_mimic: bool = False,
) -> dict[str, int]:
    """对机器人 USD 子树写入接触、阻尼、摩擦和 drive 初值。

    参数:
        root_path: 机器人导入后的 USD 子树根路径。
        config: 覆盖参数配置；可传单个配置，或传 ``{"arm": ..., "hand": ...}``。
        driven_joint_names: 需要启用 drive 的关节名；``("all",)`` 表示所有关节。
        mjcf_path: 可选 MJCF 路径，仅用于读取 frictionloss。
        mimic_path: 可选 MJCF/URDF 路径，用于读取 follower 关系。
        material_path: 创建/复用的物理材质 prim 路径；省略时从 ``root_path`` 派生。
    返回:
        统计字典，包含处理到的 collision、rigid body、joint 和 driven joint 数量。
    """

    from isaacsim.core.utils.prims import get_prim_at_path
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = get_prim_at_path(root_path)
    is_valid = getattr(root, "IsValid", None)
    if root is None or (callable(is_valid) and not bool(is_valid())):
        raise ValueError(f"Robot root prim does not exist: {root_path}")
    configs = _normalize_physx_configs(config)
    components = component_mapping or RobotComponentMapping()
    if material_path is None:
        material_path = _robot_material_path(root_path)
    # arm/hand/default 可以使用不同摩擦材质；即使当前只传单个配置，也规范化成字典，
    # 让下面遍历 prim 时只需按 component 名选择。
    materials = {
        name: make_physics_material(
            stage,
            _component_material_path(material_path, name),
            item.contact_static_friction,
            item.contact_dynamic_friction,
            item.contact_restitution,
            item.friction_combine_mode,
        )
        for name, item in configs.items()
        if item.contact_material_override
    }
    friction_by_name = (
        parse_mjcf_joint_frictionloss(mjcf_path) if mjcf_path is not None else {}
    )
    mimic_relations = parse_asset_mimic_relations(mimic_path)
    follower_master_by_name = {
        relation.dependent_joint: relation.master_joint for relation in mimic_relations
    }
    follower_joint_names = set(follower_master_by_name)
    drive_all_joints = (
        len(driven_joint_names) == 1 and str(driven_joint_names[0]).lower() == "all"
    )
    driven_names = set(driven_joint_names)

    counts = {"collision_prims": 0, "rigid_bodies": 0, "joints": 0, "driven_joints": 0}
    prims = list(Usd.PrimRange(root))
    joint_values = _resolve_usd_joint_parameters(
        prims,
        configs,
        components=components,
        follower_master_by_name=follower_master_by_name,
    )
    for prim in prims:
        group = _component_for_prim(prim, components)
        item = configs.get(group, configs["default"])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            # 物理材质绑定到 collision purpose，避免影响纯视觉材质，同时使用 stronger 级别
            # 覆盖 importer 或资产内部已有的弱绑定。
            material_key = group if group in configs else "default"
            material = materials.get(material_key)
            if material is not None:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
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
        joint_name = prim.GetName()
        is_follower = joint_name in follower_joint_names
        # mimic follower 的摩擦和 drive 初值允许单独配置，因为它们不是独立命令关节，
        # 过低会跟随不稳，过高又可能和主关节约束竞争。
        values = joint_values[joint_name]
        joint_friction = values["joint_friction"]
        joint_api.CreateJointFrictionAttr().Set(
            float(friction_by_name.get(joint_name, joint_friction))
        )
        counts["joints"] += 1
        if is_follower and native_mimic:
            continue

        is_driven = drive_all_joints or joint_name in driven_names or is_follower
        stiffness = values["stiffness"]
        damping = values["damping"]
        max_force = values["max_force"]
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


def _resolve_usd_joint_parameters(
    prims: Sequence[object],
    configs: Mapping[str, PhysxOverrideConfig],
    *,
    components: RobotComponentMapping,
    follower_master_by_name: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """按最终 USD joint 遍历顺序一次性解析全部逐关节参数。

    active 与 mimic follower 分组解析，因为两者可以使用不同 seed。follower 的 component
    归属继承 master joint，确保组合机器人中从属手指不会因自身名称被分到错误部件。
    """

    grouped: dict[tuple[str, bool], list[str]] = {}
    for prim in prims:
        if prim.GetTypeName() not in {
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
        }:
            continue
        name = prim.GetName()
        is_follower = name in follower_master_by_name
        source_name = follower_master_by_name.get(name, name)
        group = components.joint_component(source_name)
        grouped.setdefault((group, is_follower), []).append(name)

    resolved: dict[str, dict[str, float]] = {}
    for (group, is_follower), names in grouped.items():
        item = configs.get(group, configs["default"])
        specs = {
            "joint_friction": (
                item.follower_joint_friction
                if is_follower and item.follower_joint_friction is not None
                else item.joint_friction
            ),
            "stiffness": (
                item.follower_drive_stiffness_seed
                if is_follower
                else item.drive_stiffness_seed
            ),
            "damping": (
                item.follower_drive_damping_seed
                if is_follower
                else item.drive_damping_seed
            ),
            "max_force": (
                item.follower_max_force
                if is_follower and item.follower_max_force is not None
                else item.max_force
            ),
        }
        values_by_field = {
            field_name: _resolve_joint_parameter(
                spec,
                names,
                label=f"{group} {'follower' if is_follower else 'active'} {field_name}",
            )
            for field_name, spec in specs.items()
        }
        for index, name in enumerate(names):
            resolved[name] = {
                field_name: float(values[index])
                for field_name, values in values_by_field.items()
            }
    return resolved


def _resolve_joint_parameter(
    value: object, names: Sequence[str], *, label: str
) -> np.ndarray:
    """把标量、序列或精确 name-map 展开到给定 joint 顺序。

    name-map 必须完整且不能含未知键；序列只能为单元素广播或与 joint 数严格等长。这样
    配置不会因 USD 遍历顺序变化而静默错配到其他关节。
    """

    if isinstance(value, Mapping):
        configured = set(value)
        expected = set(names)
        unknown = sorted(configured - expected)
        missing = sorted(expected - configured)
        if unknown or missing:
            raise ValueError(
                f"{label} name-map must exactly cover selected joints; "
                f"unknown={unknown}, missing={missing}"
            )
        return np.asarray([float(value[name]) for name in names], dtype=float)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = np.asarray(tuple(float(item) for item in value), dtype=float)
    else:
        values = np.asarray((float(value),), dtype=float)
    if values.size == 1:
        return np.full(len(names), float(values[0]), dtype=float)
    if values.size != len(names):
        raise ValueError(
            f"{label} expected a scalar or {len(names)} values, got {values.size}"
        )
    return values


def _component_for_prim(prim: object, components: RobotComponentMapping) -> str:
    """沿祖先链解析 collision/body component，优先继承最近的显式刚体。

    若没有显式映射，则保留遍历途中第一个可由名称分类的部件；循环检测兼顾测试替身和
    异常 prim 层级，最终无法判断时返回 ``default``。
    """

    current = prim
    fallback = "default"
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        is_valid = getattr(current, "IsValid", None)
        if callable(is_valid) and not bool(is_valid()):
            break
        get_name = getattr(current, "GetName", None)
        if not callable(get_name):
            break
        name = str(get_name())
        exact = components.explicit_rigid_body_component(name)
        if exact is not None:
            return exact
        classified = component_for_name(name)
        if fallback == "default" and classified != "default":
            fallback = classified
        get_parent = getattr(current, "GetParent", None)
        current = get_parent() if callable(get_parent) else None
    return fallback


def _component_material_path(base_path: str, group: str) -> str:
    """为 arm/hand 生成独立 physics material prim 路径。"""

    if group == "default":
        return base_path
    return f"{base_path}_{group}"


def _robot_material_path(root_path: str) -> str:
    """从机器人实例 root 派生不会跨实例共享的默认 material path。"""

    return f"{root_path.rstrip('/')}/PhysicsMaterials/RobotContactMaterial"


def apply_robot_gravity_policy(
    root_path: str,
    policy,
    *,
    component_mapping: RobotComponentMapping | None = None,
) -> dict[str, int]:
    """按 robot 重力策略写入指定 USD 子树下的刚体 ``disableGravity``。

    ``policy`` 只要求暴露 ``enabled_for_name(name)``，不直接依赖
    ``RobotGravityPolicy`` 类型。这样 USD 覆盖层保持为低层工具，后续测试或其它配置对象也
    可以复用同一写入逻辑。返回值统计开启/关闭重力的刚体数量，供脚本输出诊断。
    """

    from isaacsim.core.utils.prims import get_prim_at_path
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = get_prim_at_path(root_path)
    components = component_mapping or RobotComponentMapping()
    counts = {"enabled": 0, "disabled": 0}
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        gravity_enabled = bool(
            policy.enabled_for_component(
                components.rigid_body_component(prim.GetName())
            )
        )
        # PhysXSchema.PhysxRigidBodyAPI 可能还没由 importer 写入；这里按需 Apply，确保
        # disableGravity 属性有稳定落点。
        rigid_body_api = (
            PhysxSchema.PhysxRigidBodyAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        )
        rigid_body_api.GetDisableGravityAttr().Set(not gravity_enabled)
        if gravity_enabled:
            counts["enabled"] += 1
        else:
            counts["disabled"] += 1
    return counts


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
