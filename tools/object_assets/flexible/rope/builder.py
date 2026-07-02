"""Offline capsule rope USD asset builder.

This module is intentionally outside ``src/linkerbot_sim`` runtime code. It writes
USD assets from tools-side YAML and is used by scripts/tools, not by simulation runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from linkerbot_sim.utils.paths import repo_path


def _float_tuple(values, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values) if values is not None else default


@dataclass(frozen=True)
class CapsuleRopeAssetConfig:
    """capsule rope USD 资产生成配置。

    这些字段描述资产固有结构：几何、质量、阻尼、关节限制、碰撞过滤和可视材质。
    运行时接触摩擦和 solver iteration 放在 ``configs/objects``，由 ``src`` 导入资产后覆盖。
    """

    asset_path: str = (
        "assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda"
    )
    root_path: str = "/CapsuleRope"
    segments: int = 12
    length: float = 0.75
    radius: float | None = None
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    total_mass: float = 0.2
    shape: str = "capsule"
    endpoint_box_mass: float = 0.5
    endpoint_box_size: tuple[float, float, float] = (0.04, 0.03, 0.1)
    endpoint_linear_damping: float = 0.015
    endpoint_angular_damping: float = 0.05
    segment_linear_damping: float = 0.1
    segment_angular_damping: float = 0.12
    bend_limit: float = 2.0943951023931953
    bend_stiffness: float = 0.1
    bend_damping: float = 0.1
    lock_twist: bool = True
    twist_limit: float | None = None
    twist_stiffness: float = 0.1
    twist_damping: float = 0.05
    disable_adjacent_collisions: bool = True
    endpoint_color: tuple[float, float, float] = (0.12, 0.34, 0.95)
    rope_color: tuple[float, float, float] = (0.78, 0.62, 0.22)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CapsuleRopeAssetConfig":
        if "object" not in data or "rope" not in data:
            raise ValueError(
                "Capsule rope asset config must contain top-level object and rope sections"
            )
        if not isinstance(data["object"], Mapping):
            raise ValueError("object section must be a mapping")
        if not isinstance(data["rope"], Mapping):
            raise ValueError("rope section must be a mapping")
        object_cfg = dict(data["object"])
        rope = dict(data["rope"])
        unsupported = set(rope) - {
            "segments",
            "length",
            "radius",
            "center",
            "total_mass",
            "shape",
            "endpoint_box_mass",
            "endpoint_box_size",
            "endpoint_linear_damping",
            "endpoint_angular_damping",
            "segment_linear_damping",
            "segment_angular_damping",
            "bend_limit",
            "bend_stiffness",
            "bend_damping",
            "lock_twist",
            "twist_limit",
            "twist_stiffness",
            "twist_damping",
            "disable_adjacent_collisions",
            "endpoint_color",
            "rope_color",
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"rope contains unsupported generation keys: {names}")
        length = float(rope.get("length", cls.length))
        segments = int(rope.get("segments", cls.segments))
        radius = rope.get("radius")
        twist_limit = rope.get("twist_limit")
        return cls(
            asset_path=str(object_cfg.get("asset_path", cls.asset_path)),
            root_path=str(object_cfg.get("root_path", cls.root_path)),
            segments=segments,
            length=length,
            radius=length / segments * 0.15 if radius is None else float(radius),
            center=_float_tuple(rope.get("center"), cls.center),
            total_mass=float(rope.get("total_mass", cls.total_mass)),
            shape=str(rope.get("shape", cls.shape)),
            endpoint_box_mass=float(
                rope.get("endpoint_box_mass", cls.endpoint_box_mass)
            ),
            endpoint_box_size=_float_tuple(
                rope.get("endpoint_box_size"), cls.endpoint_box_size
            ),
            endpoint_linear_damping=float(
                rope.get("endpoint_linear_damping", cls.endpoint_linear_damping)
            ),
            endpoint_angular_damping=float(
                rope.get("endpoint_angular_damping", cls.endpoint_angular_damping)
            ),
            segment_linear_damping=float(
                rope.get("segment_linear_damping", cls.segment_linear_damping)
            ),
            segment_angular_damping=float(
                rope.get("segment_angular_damping", cls.segment_angular_damping)
            ),
            bend_limit=float(rope.get("bend_limit", cls.bend_limit)),
            bend_stiffness=float(rope.get("bend_stiffness", cls.bend_stiffness)),
            bend_damping=float(rope.get("bend_damping", cls.bend_damping)),
            lock_twist=bool(rope.get("lock_twist", cls.lock_twist)),
            twist_limit=(
                math.radians(560.0 / segments)
                if twist_limit is None
                else float(twist_limit)
            ),
            twist_stiffness=float(rope.get("twist_stiffness", cls.twist_stiffness)),
            twist_damping=float(rope.get("twist_damping", cls.twist_damping)),
            disable_adjacent_collisions=bool(
                rope.get("disable_adjacent_collisions", cls.disable_adjacent_collisions)
            ),
            endpoint_color=_float_tuple(rope.get("endpoint_color"), cls.endpoint_color),
            rope_color=_float_tuple(rope.get("rope_color"), cls.rope_color),
        )

    def validate(self) -> None:
        if not self.asset_path:
            raise ValueError("object.asset_path cannot be empty")
        if not self.root_path.startswith("/"):
            raise ValueError("object.root_path must be an absolute USD path")
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
        if self.shape not in {"capsule", "cuboid"}:
            raise ValueError(f"Unsupported rope shape: {self.shape}")
        if self.radius >= 0.45 * (self.length / self.segments):
            raise ValueError("rope radius is too large for the segment spacing")


def make_visual_material(stage, path: str, color):
    """创建 USD PreviewSurface 可视材质。

    参数:
        stage: 当前 USD stage。
        path: 材质 prim 路径。
        color: RGB 颜色，取值按 USD PreviewSurface 约定写入 ``diffuseColor``。
    返回:
        ``UsdShade.Material``；后续可绑定到 endpoint 或 rope segment 的视觉材质槽。
    """

    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(path + "/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_visual_material(prim, material) -> None:
    """把可视材质绑定到指定几何 prim。

    只写默认 material binding，不影响 runtime 后续绑定的 physics purpose 材质。
    """

    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def set_xform_translate(prim, xyz) -> None:
    """给 prim 添加平移 xform op。

    参数:
        prim: 需要定位的 USD prim。
        xyz: 世界/父坐标系下的平移，单位 m。
    """

    from pxr import Gf, UsdGeom

    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def apply_rigid_body(
    prim, mass: float, linear_damping: float, angular_damping: float
) -> None:
    """给 prim 添加碰撞、刚体、质量和阻尼 API。

    该 helper 用于 endpoint cuboid 和中间绳段；它不会设置运行时接触材质、solver iteration 或
    joint。运行时接触和 solver 覆盖由 ``configs/objects`` 进入 ``src`` 后写入当前 stage。
    """

    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body_api.CreateLinearDampingAttr().Set(float(linear_damping))
    rigid_body_api.CreateAngularDampingAttr().Set(float(angular_damping))


def create_endpoint_box(
    stage,
    path: str,
    position,
    size_xyz,
    config: CapsuleRopeAssetConfig,
    visual_material,
):
    """创建一个可被夹捏抓取的绳端端块刚体。

    参数:
        stage/path: 目标 USD stage 和 cube prim 路径。
        position: 端块中心位置，单位 m。
        size_xyz: 端块三轴尺寸，单位 m。
        config: 绳体生成配置，提供质量和阻尼。
        visual_material: 已创建的可视材质。
    返回:
        endpoint cuboid 的 USD prim。
    """

    from pxr import Gf, UsdGeom

    cuboid = UsdGeom.Cube.Define(stage, path)
    cuboid.CreateSizeAttr(1.0)
    prim = cuboid.GetPrim()
    set_xform_translate(prim, position)
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    apply_rigid_body(
        prim,
        config.endpoint_box_mass,
        config.endpoint_linear_damping,
        config.endpoint_angular_damping,
    )
    bind_visual_material(prim, visual_material)
    return prim


def create_rope_segment(
    stage,
    path: str,
    position,
    pitch: float,
    mass: float,
    config: CapsuleRopeAssetConfig,
    visual_material,
):
    """创建一个中间绳段刚体。

    根据 ``config.shape`` 生成沿局部 X 轴排列的 cuboid 或 capsule。返回的 prim 已具备
    RigidBody/Collision/Mass/PhysX damping API 和可视材质，但相邻 D6 joint 会在
    ``create_rope_model`` 中统一创建。
    """

    from pxr import Gf, UsdGeom

    radius = float(config.radius)
    # cuboid 形状便于调试和稳定接触；capsule 形状更接近柔性绳段。两者都沿局部 X 方向排列，
    # 因此后续 D6 joint 的 local_pos 使用 ±0.5*pitch。
    if config.shape == "cuboid":
        segment = UsdGeom.Cube.Define(stage, path)
        segment.CreateSizeAttr(1.0)
        prim = segment.GetPrim()
        set_xform_translate(prim, position)
        UsdGeom.Xformable(prim).AddScaleOp().Set(
            Gf.Vec3f(pitch, 2.0 * radius, 2.0 * radius)
        )
    else:
        segment = UsdGeom.Capsule.Define(stage, path)
        segment.CreateAxisAttr("X")
        segment.CreateRadiusAttr(radius)
        segment.CreateHeightAttr(max(1.0e-4, pitch - 2.0 * radius))
        prim = segment.GetPrim()
        set_xform_translate(prim, position)
    apply_rigid_body(
        prim, mass, config.segment_linear_damping, config.segment_angular_damping
    )
    bind_visual_material(prim, visual_material)
    return prim


def lock_limit(joint_prim, axis: str) -> None:
    """锁定 D6 joint 的某个自由度。

    USD/PhysX 用 ``low > high`` 表达锁定状态；这里固定写入 ``1/-1``，避免调用方在
    多处重复依赖这个约定。
    """

    from pxr import UsdPhysics

    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(1.0)
    limit_api.CreateHighAttr(-1.0)


def bounded_limit(joint_prim, axis: str, low: float, high: float) -> None:
    """给 D6 joint 某个自由度设置上下限。

    参数:
        joint_prim: 已创建的 joint prim。
        axis: ``transX``/``rotY`` 等 PhysX LimitAPI 轴名。
        low/high: 下限和上限；旋转轴在 USD 属性层使用 degree。
    """

    from pxr import UsdPhysics

    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    limit_api.CreateLowAttr(float(low))
    limit_api.CreateHighAttr(float(high))


def add_angular_drive(joint_prim, axis: str, stiffness: float, damping: float) -> None:
    """给旋转自由度添加回中弹簧阻尼。

    drive target 位置和速度都设为 0，因此它表现为关节弯曲/扭转后的弱回正约束，
    用于让离散刚体链更接近有柔顺性的绳体。
    """

    from pxr import UsdPhysics

    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
    drive_api.CreateTypeAttr("force")
    drive_api.CreateTargetPositionAttr(0.0)
    drive_api.CreateTargetVelocityAttr(0.0)
    drive_api.CreateStiffnessAttr(float(stiffness))
    drive_api.CreateDampingAttr(float(damping))


def create_d6_rope_joint(
    stage, path: str, body0, body1, local_pos0, local_pos1, config: CapsuleRopeAssetConfig
):
    """在两个绳体刚体之间创建一个 D6 风格关节。

    参数:
        body0/body1: 需要连接的相邻刚体 prim。
        local_pos0/local_pos1: 两个刚体局部坐标中的连接点，单位 m。
        config: 提供弯曲/扭转 limit 和 drive 参数。
    返回:
        ``UsdPhysics.Joint``；平移自由度会被锁定，旋转自由度按配置限位/加 drive。
    """

    from pxr import Gf, UsdPhysics

    # 这里使用通用 Joint + LimitAPI 组合出 D6 行为：平移全部锁定，绕 X 轴表示扭转，
    # 绕 Y/Z 轴表示弯曲。这样 capsule/cuboid 链能近似连续绳体，同时仍是刚体系统。
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
        # 扭转锁定可提升稳定性，但会让绳体更像扁平链；允许打开 twist limit/drive 以获得
        # 更自然的旋转自由度。
        lock_limit(prim, "rotX")
    else:
        if config.twist_limit is not None and config.twist_limit >= 0:
            twist_limit_deg = float(np.degrees(config.twist_limit))
            bounded_limit(prim, "rotX", -twist_limit_deg, twist_limit_deg)
        add_angular_drive(prim, "rotX", config.twist_stiffness, config.twist_damping)
    for axis in ("rotY", "rotZ"):
        if config.bend_limit >= 0:
            bend_limit_deg = float(np.degrees(config.bend_limit))
            bounded_limit(prim, axis, -bend_limit_deg, bend_limit_deg)
        add_angular_drive(prim, axis, config.bend_stiffness, config.bend_damping)
    return joint


def filter_collision_pair(body_a, body_b) -> None:
    """过滤一对相邻绳段/端块之间的碰撞。

    只给 ``body_a`` 添加对 ``body_b`` 的 filtered pair 关系；本模块调用时按相邻链路逐对
    写入，避免 joint 连接点附近的重叠几何彼此顶开。
    """

    from pxr import UsdPhysics

    # 相邻段已经通过 joint 约束连接，如果还相互碰撞，容易在接触求解中产生抖动和能量注入。
    filter_api = UsdPhysics.FilteredPairsAPI.Apply(body_a)
    filter_api.CreateFilteredPairsRel().AddTarget(body_b.GetPath())


def create_rope_model(stage, config: CapsuleRopeAssetConfig) -> dict[str, object]:
    """在当前 USD stage 中创建完整绳体对象。

    该函数写入资产内部结构：root xform、Bodies/Joints scope、两端 endpoint cuboid、中间
    segments、相邻 D6 joint、可视材质和碰撞过滤。返回字典中的 prim/joint 句柄供测试或后续
    引用阶段收集。
    """

    from pxr import Sdf, UsdGeom, UsdPhysics

    config.validate()
    # 资产内部 root 使用 ``config.root_path``，运行时引用到 stage 的位置由 ``prim_path`` 决定。
    # 这样一个 USD 文件可以被多场景复用。
    root_path = Sdf.Path(config.root_path)
    root = UsdGeom.Xform.Define(stage, root_path)
    bodies_scope = UsdGeom.Scope.Define(stage, root_path.AppendChild("Bodies"))
    joints_scope = UsdGeom.Scope.Define(stage, root_path.AppendChild("Joints"))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    box_visual = make_visual_material(
        stage,
        str(root_path.AppendPath("Looks/EndpointBoxMaterial")),
        config.endpoint_color,
    )
    rope_visual = make_visual_material(
        stage, str(root_path.AppendPath("Looks/RopeMaterial")), config.rope_color
    )
    # 绳体中心和总长决定两端 attach 点；端块中心再向外偏移半个 cuboid 长度，使端块内侧面
    # 与第一/最后一个绳段的连接点对齐。
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
    )
    right_box = create_endpoint_box(
        stage,
        str(bodies_scope.GetPath().AppendChild("right_box")),
        (right_attach_x + box_half_x, cy, cz),
        box_size,
        config,
        box_visual,
    )

    # 中间绳段质量均分，总质量保持配置可读；端块质量单独设置，便于夹持端具有更稳定惯性。
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
                str(
                    joints_scope.GetPath().AppendChild(
                        f"segment_{index:02d}_to_segment_{index + 1:02d}"
                    )
                ),
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
            str(
                joints_scope.GetPath().AppendChild(
                    f"segment_{config.segments - 1:02d}_to_right_box"
                )
            ),
            segments[-1],
            right_box,
            (0.5 * segment_pitch, 0.0, 0.0),
            (-box_half_x, 0.0, 0.0),
            config,
        )
    )

    if config.disable_adjacent_collisions:
        # 仅过滤相邻刚体碰撞，非相邻段仍可互相接触，从而保留绳体自碰撞的大致效果。
        filter_collision_pair(left_box, segments[0])
        for body_a, body_b in zip(segments, segments[1:]):
            filter_collision_pair(body_a, body_b)
        filter_collision_pair(segments[-1], right_box)
    rope_bodies = [left_box, *segments, right_box]
    return {
        "root": root.GetPrim(),
        "segments": segments,
        "joints": joints,
        "bodies": rope_bodies,
    }


def write_capsule_rope_asset(
    config: CapsuleRopeAssetConfig, output_path: str | Path | None = None
) -> Path:
    """按配置生成并保存 capsule rope USD 资产。

    参数:
        config: 绳体对象配置。
        output_path: 可选输出路径；为空时使用 ``config.asset_path``。
    返回:
        生成的 USD 资产绝对路径。
    """

    from pxr import Usd, UsdGeom

    # 生成资产时先删除旧文件，避免 USD layer 复用旧内容；随后创建全新 stage 并设置单位。
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
