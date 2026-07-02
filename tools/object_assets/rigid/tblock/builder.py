"""Offline T block USD asset builder.

This module writes a simple T-shaped rigid object from tools-side YAML. It stays
outside ``src/linkerbot_sim`` so runtime object loading can keep referencing
already generated USD assets instead of generating geometry on startup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.utils.paths import repo_path


def _float_tuple(values, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values) if values is not None else default


def _vec3(values, default: tuple[float, float, float], *, label: str):
    parsed = _float_tuple(values, default)
    if len(parsed) != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    return parsed


@dataclass(frozen=True)
class TBlockAssetConfig:
    """T block USD 资产生成配置。

    ``*_size`` 按资产根坐标系 ``[x, y, z]`` 三个方向的长度解释，单位 m。
    当运行时 ``root_pose.rpy`` 为零时，资产根坐标系与世界坐标系对齐。
    """

    asset_path: str = "assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda"
    root_path: str = "/TBlock"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stem_size: tuple[float, float, float] = (0.04, 0.08, 0.16)
    stem_offset: tuple[float, float, float] = (-0.02, 0.0, 0.08)
    cap_size: tuple[float, float, float] = (0.04, 0.2, 0.06)
    cap_offset: tuple[float, float, float] = (-0.02, 0.0, 0.19)
    total_mass: float = 1.0
    linear_damping: float = 0.0
    angular_damping: float = 0.05
    color: tuple[float, float, float] = (0.22, 0.45, 0.78)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "TBlockAssetConfig":
        if "object" not in data or "tblock" not in data:
            raise ValueError(
                "T block asset config must contain top-level object and tblock sections"
            )
        if not isinstance(data["object"], Mapping):
            raise ValueError("object section must be a mapping")
        if not isinstance(data["tblock"], Mapping):
            raise ValueError("tblock section must be a mapping")
        object_cfg = dict(data["object"])
        tblock = dict(data["tblock"])
        allowed = {
            "center",
            "stem_size",
            "stem_offset",
            "cap_size",
            "cap_offset",
            "total_mass",
            "linear_damping",
            "angular_damping",
            "color",
        }
        unsupported = set(tblock) - allowed
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"tblock contains unsupported generation keys: {names}")
        return cls(
            asset_path=str(object_cfg.get("asset_path", cls.asset_path)),
            root_path=str(object_cfg.get("root_path", cls.root_path)),
            center=_vec3(tblock.get("center"), cls.center, label="tblock.center"),
            stem_size=_vec3(
                tblock.get("stem_size"), cls.stem_size, label="tblock.stem_size"
            ),
            stem_offset=_vec3(
                tblock.get("stem_offset"),
                cls.stem_offset,
                label="tblock.stem_offset",
            ),
            cap_size=_vec3(
                tblock.get("cap_size"), cls.cap_size, label="tblock.cap_size"
            ),
            cap_offset=_vec3(
                tblock.get("cap_offset"), cls.cap_offset, label="tblock.cap_offset"
            ),
            total_mass=float(tblock.get("total_mass", cls.total_mass)),
            linear_damping=float(tblock.get("linear_damping", cls.linear_damping)),
            angular_damping=float(tblock.get("angular_damping", cls.angular_damping)),
            color=_vec3(tblock.get("color"), cls.color, label="tblock.color"),
        )

    def validate(self) -> None:
        if not self.asset_path:
            raise ValueError("object.asset_path cannot be empty")
        if not self.root_path.startswith("/"):
            raise ValueError("object.root_path must be an absolute USD path")
        if any(value <= 0.0 for value in self.stem_size):
            raise ValueError("tblock.stem_size values must be positive")
        if any(value <= 0.0 for value in self.cap_size):
            raise ValueError("tblock.cap_size values must be positive")
        if self.total_mass <= 0.0:
            raise ValueError("tblock.total_mass must be positive")
        if self.linear_damping < 0.0:
            raise ValueError("tblock.linear_damping cannot be negative")
        if self.angular_damping < 0.0:
            raise ValueError("tblock.angular_damping cannot be negative")
        if any(value < 0.0 or value > 1.0 for value in self.color):
            raise ValueError("tblock.color values must be in the range [0, 1]")


def make_visual_material(stage, path: str, color):
    """创建 USD PreviewSurface 可视材质。"""

    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(path + "/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_visual_material(prim, material) -> None:
    """把可视材质绑定到指定几何 prim。"""

    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def set_xform_translate(prim, xyz) -> None:
    """给 prim 添加平移 xform op。"""

    from pxr import Gf, UsdGeom

    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def apply_rigid_body(
    prim, mass: float, linear_damping: float, angular_damping: float
) -> None:
    """给 T block 根 prim 添加单刚体、质量和阻尼 API。"""

    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body_api.CreateLinearDampingAttr().Set(float(linear_damping))
    rigid_body_api.CreateAngularDampingAttr().Set(float(angular_damping))


def create_cuboid_part(
    stage,
    path: str,
    position,
    size_xyz,
    visual_material,
):
    """创建 T block 的一个 compound collider/visual cuboid 子块。"""

    from pxr import Gf, UsdGeom
    from pxr import UsdPhysics

    cuboid = UsdGeom.Cube.Define(stage, path)
    cuboid.CreateSizeAttr(1.0)
    prim = cuboid.GetPrim()
    set_xform_translate(prim, position)
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    UsdPhysics.CollisionAPI.Apply(prim)
    bind_visual_material(prim, visual_material)
    return prim


def create_tblock_model(stage, config: TBlockAssetConfig) -> dict[str, object]:
    """在当前 USD stage 中创建完整 T block 对象。"""

    from pxr import Sdf, UsdGeom

    config.validate()
    root_path = Sdf.Path(config.root_path)
    root = UsdGeom.Xform.Define(stage, root_path)
    parts_scope = UsdGeom.Scope.Define(stage, root_path.AppendChild("Parts"))
    apply_rigid_body(
        root.GetPrim(),
        config.total_mass,
        config.linear_damping,
        config.angular_damping,
    )

    visual = make_visual_material(
        stage, str(root_path.AppendPath("Looks/TBlockMaterial")), config.color
    )
    stem_position = tuple(a + b for a, b in zip(config.center, config.stem_offset))
    cap_position = tuple(a + b for a, b in zip(config.center, config.cap_offset))
    stem = create_cuboid_part(
        stage,
        str(parts_scope.GetPath().AppendChild("stem")),
        stem_position,
        config.stem_size,
        visual,
    )
    cap = create_cuboid_part(
        stage,
        str(parts_scope.GetPath().AppendChild("cap")),
        cap_position,
        config.cap_size,
        visual,
    )
    return {"root": root.GetPrim(), "parts": [stem, cap]}


def write_tblock_asset(
    config: TBlockAssetConfig, output_path: str | Path | None = None
) -> Path:
    """按配置生成并保存 T block USD 资产。"""

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
    model = create_tblock_model(stage, config)
    stage.SetDefaultPrim(model["root"])
    stage.GetRootLayer().Save()
    return output
