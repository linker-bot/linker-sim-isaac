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
    避免运行时 ArticulationView 触发已弃用的 legacy tensor friction API。
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
class RobotUsdOverrideConfig:
    """机器人导入后的通用 USD drive seed 与可选 PhysX 覆盖。

    通用字段只包含 ``UsdPhysics.DriveAPI`` seed，PhysX 专属字段由
    ``robot.physics.physx`` 投影后才非空。Newton composition 只传通用 drive 字段，因而不会
    把 PhysX 材质、阻尼或关节摩擦误投影到 Newton stage。

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
    contact_material_override: bool = False
    friction_combine_mode: str | None = None
    joint_friction: object | None = None
    follower_joint_friction: object | None = None
    rigid_body_linear_damping: float | None = None
    rigid_body_angular_damping: float | None = None
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
    physics_backend: object | None = None,
):
    """创建标准 USD 物理材质，并在 PhysX 下可选写 combine mode。

    参数:
        stage: 当前 USD stage。
        path: 新材质 prim 路径。
        static_friction: 静摩擦系数。
        dynamic_friction: 动摩擦系数。
        restitution: 恢复系数。
    返回:
        ``UsdShade.Material`` 对象。
    """

    backend = _resolved_physics_backend(physics_backend)
    from pxr import Sdf, UsdPhysics, UsdShade

    # 接触材质作为独立 prim 复用到多个 collision 上，避免每个碰撞几何重复创建 schema。
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(prim)
    material_api.CreateStaticFrictionAttr().Set(float(static_friction))
    material_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    material_api.CreateRestitutionAttr().Set(float(restitution))
    if friction_combine_mode is not None and backend == "physx":
        from pxr import PhysxSchema

        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx_material_api.CreateFrictionCombineModeAttr().Set(friction_combine_mode)
    elif friction_combine_mode is not None:
        from linkerbot_sim.isaac.physics.backend import warn_unsupported_physics_fields

        warn_unsupported_physics_fields(
            backend=backend,
            feature="physics material overrides",
            fields=("friction_combine_mode",),
            reason="Newton has no PhysX material combine-mode schema",
            stacklevel=3,
        )
    return material


def apply_robot_usd_overrides(
    root_path: str,
    config: RobotUsdOverrideConfig | dict[str, RobotUsdOverrideConfig],
    *,
    driven_joint_names: list[str] | tuple[str, ...] = ("all",),
    mjcf_path: Path | None = None,
    mimic_path: Path | None = None,
    material_path: str | None = None,
    component_mapping: RobotComponentMapping | None = None,
    native_mimic: bool = False,
    physics_backend: object | None = None,
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

    backend = _resolved_physics_backend(physics_backend)

    from pxr import Usd, UsdPhysics, UsdShade
    import omni.usd

    if backend == "physx":
        from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage for robot USD overrides")
    root = stage.GetPrimAtPath(root_path)
    is_valid = getattr(root, "IsValid", None)
    if root is None or (callable(is_valid) and not bool(is_valid())):
        raise ValueError(f"Robot root prim does not exist: {root_path}")
    configs = _normalize_robot_usd_configs(config)
    components = component_mapping or RobotComponentMapping()
    skipped_fields = (
        list(_newton_robot_override_fields(configs)) if backend == "newton" else []
    )
    from linkerbot_sim.isaac.physics.backend import warn_unsupported_physics_fields

    skipped_field_count = warn_unsupported_physics_fields(
        backend=backend,
        feature="robot USD overrides",
        fields=skipped_fields,
        reason=(
            "these attributes are defined only by PhysX schemas; standard "
            "UsdPhysics material and drive attributes are still applied"
        ),
        stacklevel=3,
    )
    if material_path is None:
        material_path = _robot_material_path(root_path)

    # arm/hand/default 可以使用不同摩擦材质；即使当前只传单个配置，也规范化成字典，
    # 让下面遍历 prim 时只需按 component 名选择。
    def _make_material(name: str, item: RobotUsdOverrideConfig):
        args = (
            stage,
            _component_material_path(material_path, name),
            item.contact_static_friction,
            item.contact_dynamic_friction,
            item.contact_restitution,
            item.friction_combine_mode if backend == "physx" else None,
        )
        if backend == "physx":
            return make_physics_material(*args)
        return make_physics_material(*args, physics_backend=backend)

    materials = {
        name: _make_material(name, item)
        for name, item in configs.items()
        if item.contact_material_override
    }
    # Newton 的 SchemaResolverMjc 原生读取 importer author 的 ``mjc:frictionloss``；这里只在
    # PhysX 下把源 MJCF 数值投影为 PhysxJointAPI 属性，不能把它误报为 Newton 丢失字段。
    friction_by_name = (
        parse_mjcf_joint_frictionloss(mjcf_path)
        if backend == "physx" and mjcf_path is not None
        else {}
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

    counts = {
        "collision_prims": 0,
        "rigid_bodies": 0,
        "joints": 0,
        "driven_joints": 0,
        "skipped_physx_fields": skipped_field_count,
    }
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
            if backend == "physx" and (
                item.rigid_body_linear_damping is not None
                or item.rigid_body_angular_damping is not None
            ):
                rigid_api = (
                    PhysxSchema.PhysxRigidBodyAPI(prim)
                    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                    else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                )
                if item.rigid_body_linear_damping is not None:
                    rigid_api.CreateLinearDampingAttr().Set(
                        float(item.rigid_body_linear_damping)
                    )
                if item.rigid_body_angular_damping is not None:
                    rigid_api.CreateAngularDampingAttr().Set(
                        float(item.rigid_body_angular_damping)
                    )
            counts["rigid_bodies"] += 1

        if prim.GetTypeName() not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue

        joint_name = prim.GetName()
        is_follower = joint_name in follower_joint_names
        # mimic follower 的摩擦和 drive 初值允许单独配置，因为它们不是独立命令关节，
        # 过低会跟随不稳，过高又可能和主关节约束竞争。
        values = joint_values[joint_name]
        if backend == "physx":
            joint_api = (
                PhysxSchema.PhysxJointAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxJointAPI)
                else PhysxSchema.PhysxJointAPI.Apply(prim)
            )
            configured_friction = values.get("joint_friction")
            if joint_name in friction_by_name or configured_friction is not None:
                joint_api.CreateJointFrictionAttr().Set(
                    float(friction_by_name.get(joint_name, configured_friction))
                )
        counts["joints"] += 1
        if is_follower and native_mimic:
            # Importer 3.0 会保留 MJCF follower actuator/URDF drive，同时 author
            # NewtonMimicAPI。原生 mimic 必须是唯一执行者，因此把遗留 drive 清零。
            _disable_native_mimic_follower_drive(
                prim,
                UsdPhysics,
                physics_backend=backend,
            )
            continue

        is_driven = drive_all_joints or joint_name in driven_names or is_follower
        stiffness = values["stiffness"]
        damping = values["damping"]
        max_force = values["max_force"]
        drive_name = (
            "angular" if prim.GetTypeName() == "PhysicsRevoluteJoint" else "linear"
        )
        if backend == "newton" and (not is_driven or max_force <= 0):
            _remove_usd_drive(prim, UsdPhysics, drive_name)
            continue
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


def _disable_native_mimic_follower_drive(
    prim,
    usd_physics,
    *,
    physics_backend: object = "physx",
) -> None:
    """禁用 importer 遗留的 follower drive，避免与原生 mimic 竞争。"""

    from linkerbot_sim.isaac.physics.backend import normalize_physics_backend

    drive_name = "angular" if prim.GetTypeName() == "PhysicsRevoluteJoint" else "linear"
    if normalize_physics_backend(physics_backend) == "newton":
        # Newton/MuJoCo 会把 maxForce=0 转成零宽度 actfrcrange 并拒绝初始化。
        # 在强层删除 multiple-apply schema，使弱层 importer drive 不再参与组合；
        # follower 只由 NewtonMimicAPI/equality 约束驱动。
        _remove_usd_drive(prim, usd_physics, drive_name)
        return
    drive_api = usd_physics.DriveAPI.Apply(prim, drive_name)
    drive_api.CreateTypeAttr().Set("force")
    drive_api.CreateStiffnessAttr().Set(0.0)
    drive_api.CreateDampingAttr().Set(0.0)
    drive_api.CreateMaxForceAttr().Set(0.0)


def _remove_usd_drive(prim, usd_physics, drive_name: str) -> None:
    """在当前强层删除 importer 或弱层声明的 multiple-apply drive schema。"""

    prim.RemoveAPI(usd_physics.DriveAPI, drive_name)


def _normalize_robot_usd_configs(
    config: RobotUsdOverrideConfig | dict[str, RobotUsdOverrideConfig],
) -> dict[str, RobotUsdOverrideConfig]:
    """把单配置或分组配置规范成含 default 的字典。"""

    # 单配置适合简单机器人；分组配置适合 AR5+L6 这种机械臂和灵巧手物理尺度差异较大的资产。
    if isinstance(config, RobotUsdOverrideConfig):
        return {"default": config}
    if not config:
        raise ValueError("robot USD override config cannot be empty")
    first = next(iter(config.values()))
    normalized = dict(config)
    normalized.setdefault("default", first)
    return normalized


def _resolved_physics_backend(value: object | None) -> str:
    """解析显式后端；省略时读取 Isaac 当前 active engine。"""

    from linkerbot_sim.isaac.physics.backend import (
        active_physics_backend,
        normalize_physics_backend,
    )

    return (
        active_physics_backend() if value is None else normalize_physics_backend(value)
    )


def _newton_robot_override_fields(
    configs: Mapping[str, RobotUsdOverrideConfig],
) -> tuple[str, ...]:
    """列出 Newton 路径不会写入的 PhysX schema 配置字段。"""

    fields: list[str] = []
    for group, item in configs.items():
        prefix = f"robot.physics.physx.{group}"
        if item.contact_material_override and item.friction_combine_mode is not None:
            fields.append(f"{prefix}.material.friction_combine_mode")
        if item.rigid_body_linear_damping is not None:
            fields.append(f"{prefix}.rigid_body.linear_damping")
        if item.rigid_body_angular_damping is not None:
            fields.append(f"{prefix}.rigid_body.angular_damping")
        if item.joint_friction is not None:
            fields.append(f"{prefix}.joint.friction")
        if item.follower_joint_friction is not None:
            fields.append(f"{prefix}.joint.follower_friction")
    return tuple(fields)


def _resolve_usd_joint_parameters(
    prims: Sequence[object],
    configs: Mapping[str, RobotUsdOverrideConfig],
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
        friction = (
            item.follower_joint_friction
            if is_follower and item.follower_joint_friction is not None
            else item.joint_friction
        )
        specs = {
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
        if friction is not None:
            specs["joint_friction"] = friction
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
    physics_backend: object | None = None,
) -> dict[str, int]:
    """把 robot 重力策略投影到指定 USD 子树的后端刚体属性。

    ``policy`` 只要求暴露 ``enabled_for_name(name)``，不直接依赖
    ``RobotGravityPolicy`` 类型。这样 USD 覆盖层保持为低层工具，后续测试或其它配置对象也
    可以复用同一写入逻辑。PhysX 使用 ``disableGravity``；Newton 使用
    ``mjc:gravcomp``，该列由 ``SolverMuJoCo`` 在 model finalize 时读入，并由 MuJoCo-Warp
    在 GPU 上逐刚体计算补偿力。返回值统计开启/关闭及 Newton 补偿的刚体数量，供脚本诊断。
    """

    backend = _resolved_physics_backend(physics_backend)

    from pxr import Sdf, Usd, UsdPhysics
    import omni.usd

    if backend == "physx":
        from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage for robot gravity policy")
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Robot root prim does not exist: {root_path}")
    components = component_mapping or RobotComponentMapping()
    counts = {"enabled": 0, "disabled": 0, "newton_gravcomp": 0}
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        component = components.rigid_body_component(prim.GetName())
        gravity_enabled = bool(policy.enabled_for_component(component))
        # PhysXSchema.PhysxRigidBodyAPI 可能还没由 importer 写入；这里按需 Apply，确保
        # disableGravity 属性有稳定落点。
        if backend == "physx":
            rigid_body_api = (
                PhysxSchema.PhysxRigidBodyAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            )
            rigid_body_api.GetDisableGravityAttr().Set(not gravity_enabled)
        else:
            # SolverMuJoCo 在 builder 解析 USD 前注册了 body-frequency
            # ``mujoco:gravcomp``。1.0 会在 MuJoCo-Warp 的 GPU passive-force
            # kernel 中逐刚体抵消重力，因而能准确承载 robot profile 的逐组件
            # disableGravity 语义，同时不影响同一 world 内仍需重力的动态对象。
            prim.CreateAttribute(
                "mjc:gravcomp",
                Sdf.ValueTypeNames.Float,
            ).Set(0.0 if gravity_enabled else 1.0)
            if not gravity_enabled:
                counts["newton_gravcomp"] += 1
        if gravity_enabled:
            counts["enabled"] += 1
        else:
            counts["disabled"] += 1
    counts["skipped_physx_fields"] = 0
    return counts
