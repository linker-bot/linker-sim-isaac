"""Capsule/box 刚体链绳体对象。

本模块只负责“绳体对象”本身：从 YAML 参数生成 USD 资产，或在仿真 stage 中
引用已经生成好的 USD。场景环境只负责 world、重力、solver 等全局设置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from manipulation_project.utils.paths import repo_path


def _float_tuple(values, fallback: tuple[float, ...]) -> tuple[float, ...]:
    """把 YAML list/tuple 转成 float tuple。"""

    return tuple(float(value) for value in values) if values is not None else fallback


@dataclass(frozen=True)
class CapsuleRopeConfig:
    """抓取 demo 使用的 capsule/box 绳体对象参数。

    输入字段:
        asset_path: 生成后保存的 USD 资产路径。
        prim_path: 仿真运行时引用到 stage 的 prim 路径。
        root_path: 资产文件内部的默认 prim 路径。
        segments/length/radius/shape: 中间绳段数量、总长、半径和形状。
        total_mass: 中间绳段总质量，按段均分。
        center: 绳体中心坐标，单位 m。
        endpoint_box_mass/endpoint_box_size: 两端抓取 box 的质量和尺寸。
        bend_* / twist_*: D6 joint 弯曲和扭转限制/弹簧阻尼。
        disable_adjacent_collisions: 是否过滤相邻刚体碰撞。
        solver_*_iterations: 绳体刚体 PhysX solver 迭代次数。
        endpoint_color/rope_color: 可视材质颜色 RGB。
        env_*: 绳端接触物理材质参数。
    输出:
        传给 ``write_capsule_rope_asset`` 可生成 USD；传给
        ``add_capsule_rope_reference`` 可把 USD 引用进仿真 stage。
    """

    asset_path: str = "assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda"
    prim_path: str = "/World/CapsuleRope"
    root_path: str = "/CapsuleRope"
    segments: int = 18
    length: float = 0.75
    radius: float | None = None
    shape: str = "capsule"
    total_mass: float = 0.2
    center: tuple[float, float, float] = (0.4, -0.55, 0.05)
    endpoint_box_mass: float = 0.5
    endpoint_box_size: tuple[float, float, float] = (0.04, 0.03, 0.1)
    endpoint_linear_damping: float = 0.015
    endpoint_angular_damping: float = 0.05
    segment_linear_damping: float = 0.1
    segment_angular_damping: float = 0.12
    bend_limit_deg: float = 120.0
    bend_stiffness: float = 0.05
    bend_damping: float = 0.03
    lock_twist: bool = False
    twist_limit_deg: float | None = None
    twist_stiffness: float = 0.1
    twist_damping: float = 0.05
    disable_adjacent_collisions: bool = True
    solver_position_iterations: int = 32
    solver_velocity_iterations: int = 4
    endpoint_color: tuple[float, float, float] = (0.12, 0.34, 0.95)
    rope_color: tuple[float, float, float] = (0.78, 0.62, 0.22)
    env_static_friction: float = 0.7
    env_dynamic_friction: float = 0.5
    env_restitution: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict) -> "CapsuleRopeConfig":
        """从 YAML 映射构造绳体对象配置。

        参数:
            data: 完整对象配置，通常包含 ``object`` 和 ``rope`` 两个子 mapping。
                也兼容旧的直接 ``rope`` mapping。
        返回:
            ``CapsuleRopeConfig``；缺失或显式设为 ``null`` 的 ``radius`` 和
            ``twist_limit_deg`` 会根据 ``length/segments`` 自动估算。
        """

        object_cfg = dict(data.get("object") or {})
        rope = dict(data.get("rope") or data)
        default_length = float(rope.get("length", cls.length))
        default_segments = int(rope.get("segments", cls.segments))
        default_radius = default_length / default_segments * 0.15
        default_twist_limit = 560.0 / default_segments
        radius = rope.get("radius")
        twist_limit_deg = rope.get("twist_limit_deg")
        return cls(
            asset_path=str(object_cfg.get("asset_path", rope.get("asset_path", cls.asset_path))),
            prim_path=str(object_cfg.get("prim_path", rope.get("prim_path", cls.prim_path))),
            root_path=str(object_cfg.get("root_path", rope.get("root_path", cls.root_path))),
            segments=default_segments,
            length=default_length,
            radius=default_radius if radius is None else float(radius),
            shape=str(rope.get("shape", cls.shape)),
            total_mass=float(rope.get("total_mass", rope.get("mass", cls.total_mass))),
            center=_float_tuple(rope.get("center"), cls.center),
            endpoint_box_mass=float(rope.get("endpoint_box_mass", cls.endpoint_box_mass)),
            endpoint_box_size=_float_tuple(rope.get("endpoint_box_size"), cls.endpoint_box_size),
            endpoint_linear_damping=float(rope.get("endpoint_linear_damping", cls.endpoint_linear_damping)),
            endpoint_angular_damping=float(rope.get("endpoint_angular_damping", cls.endpoint_angular_damping)),
            segment_linear_damping=float(rope.get("segment_linear_damping", cls.segment_linear_damping)),
            segment_angular_damping=float(rope.get("segment_angular_damping", cls.segment_angular_damping)),
            bend_limit_deg=float(rope.get("bend_limit_deg", cls.bend_limit_deg)),
            bend_stiffness=float(rope.get("bend_stiffness", cls.bend_stiffness)),
            bend_damping=float(rope.get("bend_damping", cls.bend_damping)),
            lock_twist=bool(rope.get("lock_twist", cls.lock_twist)),
            twist_limit_deg=default_twist_limit if twist_limit_deg is None else float(twist_limit_deg),
            twist_stiffness=float(rope.get("twist_stiffness", cls.twist_stiffness)),
            twist_damping=float(rope.get("twist_damping", cls.twist_damping)),
            disable_adjacent_collisions=bool(rope.get("disable_adjacent_collisions", cls.disable_adjacent_collisions)),
            solver_position_iterations=int(
                rope.get("solver_position_iterations", rope.get("rope_solver_position_iterations", cls.solver_position_iterations))
            ),
            solver_velocity_iterations=int(
                rope.get("solver_velocity_iterations", rope.get("rope_solver_velocity_iterations", cls.solver_velocity_iterations))
            ),
            endpoint_color=_float_tuple(rope.get("endpoint_color"), cls.endpoint_color),
            rope_color=_float_tuple(rope.get("rope_color"), cls.rope_color),
            env_static_friction=float(rope.get("env_static_friction", cls.env_static_friction)),
            env_dynamic_friction=float(rope.get("env_dynamic_friction", cls.env_dynamic_friction)),
            env_restitution=float(rope.get("env_restitution", cls.env_restitution)),
        )

    def asset_file(self) -> Path:
        """返回 USD 资产绝对路径。"""

        return repo_path(self.asset_path)

    def validate(self) -> None:
        """校验绳体几何、质量和 solver 参数。"""

        if self.segments < 2:
            raise ValueError("rope segments must be at least 2")
        if self.length <= 0:
            raise ValueError("rope length must be positive")
        if self.radius is None or self.radius <= 0:
            raise ValueError("rope radius must be positive")
        if self.total_mass <= 0:
            raise ValueError("rope total_mass must be positive")
        if self.endpoint_box_mass <= 0:
            raise ValueError("endpoint_box_mass must be positive")
        if any(value <= 0 for value in self.endpoint_box_size):
            raise ValueError("endpoint_box_size values must be positive")
        if self.shape not in {"capsule", "box"}:
            raise ValueError(f"Unsupported rope shape: {self.shape}")
        segment_pitch = self.length / self.segments
        if self.radius >= 0.45 * segment_pitch:
            raise ValueError("rope radius is too large for the segment spacing")
        if self.solver_position_iterations <= 0:
            raise ValueError("solver_position_iterations must be positive")
        if self.solver_velocity_iterations < 0:
            raise ValueError("solver_velocity_iterations cannot be negative")


def make_physics_material(stage, path: str, static_friction: float, dynamic_friction: float, restitution: float):
    """创建 USD 物理材质。"""

    from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    prim = material.GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(prim)
    material_api.CreateStaticFrictionAttr().Set(float(static_friction))
    material_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    material_api.CreateRestitutionAttr().Set(float(restitution))
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx_material_api.CreateFrictionCombineModeAttr().Set("average")
    return material


def make_visual_material(stage, path: str, color):
    """创建 USD PreviewSurface 可视材质。"""

    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(path + "/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_visual_material(prim, material) -> None:
    """把可视材质绑定到 prim。"""

    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def bind_physics_material(prim, material) -> None:
    """把物理材质绑定到 prim 的 collision purpose。"""

    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def set_xform_translate(prim, xyz) -> None:
    """给 prim 添加 translate xform op。"""

    from pxr import Gf, UsdGeom

    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def apply_rigid_body(prim, mass: float, linear_damping: float, angular_damping: float) -> None:
    """给 prim 添加碰撞、刚体、质量和阻尼 API。"""

    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body_api.CreateLinearDampingAttr().Set(float(linear_damping))
    rigid_body_api.CreateAngularDampingAttr().Set(float(angular_damping))


def create_endpoint_box(stage, path: str, position, size_xyz, config: CapsuleRopeConfig, visual_material, physics_material):
    """创建一个绳端端块刚体。"""

    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    prim = box.GetPrim()
    set_xform_translate(prim, position)
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    apply_rigid_body(prim, config.endpoint_box_mass, config.endpoint_linear_damping, config.endpoint_angular_damping)
    bind_visual_material(prim, visual_material)
    bind_physics_material(prim, physics_material)
    return prim


def create_rope_segment(stage, path: str, position, pitch: float, mass: float, config: CapsuleRopeConfig, visual_material):
    """创建一个中间绳段刚体。"""

    from pxr import Gf, UsdGeom

    radius = float(config.radius)
    if config.shape == "box":
        segment = UsdGeom.Cube.Define(stage, path)
        segment.CreateSizeAttr(1.0)
        prim = segment.GetPrim()
        set_xform_translate(prim, position)
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(pitch, 2.0 * radius, 2.0 * radius))
    else:
        segment = UsdGeom.Capsule.Define(stage, path)
        segment.CreateAxisAttr("X")
        segment.CreateRadiusAttr(radius)
        segment.CreateHeightAttr(max(1.0e-4, pitch - 2.0 * radius))
        prim = segment.GetPrim()
        set_xform_translate(prim, position)
    apply_rigid_body(prim, mass, config.segment_linear_damping, config.segment_angular_damping)
    bind_visual_material(prim, visual_material)
    return prim


def lock_limit(joint_prim, axis: str) -> None:
    """锁定 D6 joint 的某个自由度。"""

    from pxr import UsdPhysics

    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(1.0)
    limit_api.CreateHighAttr(-1.0)


def bounded_limit(joint_prim, axis: str, low: float, high: float) -> None:
    """给 D6 joint 某个自由度设置上下限。"""

    from pxr import UsdPhysics

    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(float(low))
    limit_api.CreateHighAttr(float(high))


def add_angular_drive(joint_prim, axis: str, stiffness: float, damping: float) -> None:
    """给旋转自由度添加回中弹簧阻尼。"""

    from pxr import UsdPhysics

    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
    drive_api.CreateTypeAttr("force")
    drive_api.CreateTargetPositionAttr(0.0)
    drive_api.CreateTargetVelocityAttr(0.0)
    drive_api.CreateStiffnessAttr(float(stiffness))
    drive_api.CreateDampingAttr(float(damping))


def create_d6_rope_joint(stage, path: str, body0, body1, local_pos0, local_pos1, config: CapsuleRopeConfig):
    """在两个绳体刚体之间创建一个 D6 风格关节。"""

    from pxr import Gf, UsdPhysics

    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    prim = joint.GetPrim()
    for axis in ("transX", "transY", "transZ"):
        lock_limit(prim, axis)
    if config.lock_twist:
        lock_limit(prim, "rotX")
    else:
        if config.twist_limit_deg is not None and config.twist_limit_deg >= 0:
            bounded_limit(prim, "rotX", -config.twist_limit_deg, config.twist_limit_deg)
        add_angular_drive(prim, "rotX", config.twist_stiffness, config.twist_damping)
    for axis in ("rotY", "rotZ"):
        if config.bend_limit_deg >= 0:
            bounded_limit(prim, axis, -config.bend_limit_deg, config.bend_limit_deg)
        add_angular_drive(prim, axis, config.bend_stiffness, config.bend_damping)
    return joint


def filter_collision_pair(body_a, body_b) -> None:
    """过滤一对相邻绳段/端块之间的碰撞。"""

    from pxr import UsdPhysics

    filter_api = UsdPhysics.FilteredPairsAPI.Apply(body_a)
    filter_api.CreateFilteredPairsRel().AddTarget(body_b.GetPath())


def apply_rope_solver_iteration_overrides(rope_bodies: list, config: CapsuleRopeConfig) -> None:
    """给绳体刚体写入 PhysX solver 迭代次数。"""

    from pxr import PhysxSchema

    for body in rope_bodies:
        rigid_api = (
            PhysxSchema.PhysxRigidBodyAPI(body)
            if body.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        )
        rigid_api.CreateSolverPositionIterationCountAttr().Set(int(config.solver_position_iterations))
        rigid_api.CreateSolverVelocityIterationCountAttr().Set(int(config.solver_velocity_iterations))


def create_rope_model(stage, config: CapsuleRopeConfig) -> dict[str, object]:
    """在当前 USD stage 中创建完整绳体对象。"""

    from pxr import Sdf, UsdGeom, UsdPhysics

    config.validate()
    root_path = Sdf.Path(config.root_path)
    root = UsdGeom.Xform.Define(stage, root_path)
    bodies_scope = UsdGeom.Scope.Define(stage, root_path.AppendChild("Bodies"))
    joints_scope = UsdGeom.Scope.Define(stage, root_path.AppendChild("Joints"))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    box_visual = make_visual_material(stage, str(root_path.AppendPath("Looks/EndpointBoxMaterial")), config.endpoint_color)
    rope_visual = make_visual_material(stage, str(root_path.AppendPath("Looks/RopeMaterial")), config.rope_color)
    endpoint_physics = make_physics_material(
        stage,
        str(root_path.AppendPath("PhysicsMaterials/EndpointBoxMaterial")),
        config.env_static_friction,
        config.env_dynamic_friction,
        config.env_restitution,
    )

    cx, cy, cz = config.center
    segment_pitch = config.length / config.segments
    left_attach_x = cx - 0.5 * config.length
    right_attach_x = cx + 0.5 * config.length
    box_size = config.endpoint_box_size
    box_half_x = 0.5 * box_size[0]
    left_box = create_endpoint_box(
        stage,
        str(bodies_scope.GetPath().AppendChild("left_box")),
        (left_attach_x - box_half_x, cy, cz),
        box_size,
        config,
        box_visual,
        endpoint_physics,
    )
    right_box = create_endpoint_box(
        stage,
        str(bodies_scope.GetPath().AppendChild("right_box")),
        (right_attach_x + box_half_x, cy, cz),
        box_size,
        config,
        box_visual,
        endpoint_physics,
    )

    segment_mass = config.total_mass / config.segments
    segments = []
    for index in range(config.segments):
        x = left_attach_x + (index + 0.5) * segment_pitch
        segments.append(
            create_rope_segment(
                stage,
                str(bodies_scope.GetPath().AppendChild(f"segment_{index:02d}")),
                (x, cy, cz),
                segment_pitch,
                segment_mass,
                config,
                rope_visual,
            )
        )

    joints = [
        create_d6_rope_joint(
            stage,
            str(joints_scope.GetPath().AppendChild("left_box_to_segment_00")),
            left_box,
            segments[0],
            (box_half_x, 0.0, 0.0),
            (-0.5 * segment_pitch, 0.0, 0.0),
            config,
        )
    ]
    for index in range(config.segments - 1):
        joints.append(
            create_d6_rope_joint(
                stage,
                str(joints_scope.GetPath().AppendChild(f"segment_{index:02d}_to_segment_{index + 1:02d}")),
                segments[index],
                segments[index + 1],
                (0.5 * segment_pitch, 0.0, 0.0),
                (-0.5 * segment_pitch, 0.0, 0.0),
                config,
            )
        )
    joints.append(
        create_d6_rope_joint(
            stage,
            str(joints_scope.GetPath().AppendChild(f"segment_{config.segments - 1:02d}_to_right_box")),
            segments[-1],
            right_box,
            (0.5 * segment_pitch, 0.0, 0.0),
            (-box_half_x, 0.0, 0.0),
            config,
        )
    )

    if config.disable_adjacent_collisions:
        filter_collision_pair(left_box, segments[0])
        for body_a, body_b in zip(segments, segments[1:]):
            filter_collision_pair(body_a, body_b)
        filter_collision_pair(segments[-1], right_box)
    rope_bodies = [left_box, *segments, right_box]
    apply_rope_solver_iteration_overrides(rope_bodies, config)
    return {"root": root.GetPrim(), "segments": segments, "joints": joints, "bodies": rope_bodies}


def collect_rope_model_prims(stage, root_path: str) -> dict[str, object]:
    """从已引用/已生成的 stage 中收集绳体 prim。"""

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


def write_capsule_rope_asset(config: CapsuleRopeConfig, output_path: str | Path | None = None) -> Path:
    """按配置生成并保存 capsule rope USD 资产。

    参数:
        config: 绳体对象配置。
        output_path: 可选输出路径；为空时使用 ``config.asset_path``。
    返回:
        生成的 USD 资产绝对路径。
    """

    from pxr import Usd, UsdGeom

    output = repo_path(output_path or config.asset_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(0.0)
    model = create_rope_model(stage, config)
    stage.SetDefaultPrim(model["root"])
    stage.GetRootLayer().Save()
    return output


def add_capsule_rope_reference(stage, config: CapsuleRopeConfig) -> dict[str, object]:
    """把已生成的 capsule rope USD 资产引用到当前 stage。

    参数:
        stage: 当前仿真 USD stage。
        config: 绳体对象配置，使用 ``asset_path`` 和 ``prim_path``。
    返回:
        与 ``create_rope_model`` 相同形状的 prim 字典。
    """

    from pxr import Sdf, UsdGeom

    config.validate()
    asset_path = config.asset_file()
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Capsule rope asset does not exist: {asset_path}. "
            "Run scripts/build_capsule_rope_asset.py to generate it."
        )
    prim_path = Sdf.Path(config.prim_path)
    rope_xform = UsdGeom.Xform.Define(stage, prim_path)
    rope_xform.GetPrim().GetReferences().AddReference(str(asset_path))
    return collect_rope_model_prims(stage, str(prim_path))


def endpoint_center(config: CapsuleRopeConfig, endpoint: str) -> tuple[float, float, float]:
    """返回指定端块的世界坐标中心。"""

    cx, cy, cz = config.center
    rope_half = 0.5 * config.length
    box_half_x = 0.5 * config.endpoint_box_size[0]
    if endpoint == "left":
        return (cx - rope_half - box_half_x, cy, cz)
    if endpoint == "right":
        return (cx + rope_half + box_half_x, cy, cz)
    raise ValueError(f"Unsupported endpoint: {endpoint}")
