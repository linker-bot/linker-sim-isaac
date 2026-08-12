from __future__ import annotations

import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET
import warnings

import pytest

from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    apply_solver_iteration_overrides,
)
from linkerbot_sim.assets.usd_overrides import (
    RobotUsdOverrideConfig,
    apply_robot_usd_overrides,
)
from linkerbot_sim.configuration.objects import (
    CapsuleRopePhysicsConfig,
    CapsuleRopePhysxConfig,
    CapsuleRopePhysxSolverConfig,
    ObjectMaterialConfig,
    ObjectPhysxMaterialConfig,
    RigidObjectPhysicsConfig,
    RigidObjectPhysxConfig,
)
from linkerbot_sim.objects.dynamic_chain.capsule_rope import (
    CapsuleRopeConfig,
    apply_capsule_rope_runtime_physics,
)
from linkerbot_sim.objects.rigid.importer import _apply_rigid_object_physics
from linkerbot_sim.isaac.physics.backend import PhysicsCompatibilityWarning


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
COMBINED_ASSET_ROOT = ASSET_ROOT / "combined_system"
SINGLE_ARM_ASSET_ROOT = ASSET_ROOT / "single_system" / "arm"


def _combined_equality_follower_damping(side: str) -> dict[str, float]:
    asset = COMBINED_ASSET_ROOT / f"AR5V2_L6V1_{side}" / f"AR5V2_L6V1_{side}.xml"
    root = ET.parse(asset).getroot()
    joints_by_name = {
        name: element
        for element in root.iter("joint")
        if (name := element.get("name")) is not None
    }
    equality_followers = tuple(
        joint1
        for element in root.findall("./equality/joint")
        if (joint1 := element.get("joint1")) is not None
    )
    assert len(equality_followers) == 5
    return {
        name.replace(f"L6V1_{side}_", "L6V1_SIDE_", 1): float(
            joints_by_name[name].get("damping", "nan")
        )
        for name in equality_followers
    }


def test_combined_hand_equality_followers_have_symmetric_joint_damping() -> None:
    left = _combined_equality_follower_damping("L")
    right = _combined_equality_follower_damping("R")

    assert set(left) == set(right)
    assert left == right
    assert set(left.values()) == {0.05}


@pytest.mark.parametrize(
    ("asset", "side", "nested_hand"),
    (
        (
            COMBINED_ASSET_ROOT / "AR5V2_L6V1_L" / "AR5V2_L6V1_L.xml",
            "L",
            "L6V1_L_hand_base_link",
        ),
        (
            COMBINED_ASSET_ROOT / "AR5V2_L6V1_R" / "AR5V2_L6V1_R.xml",
            "R",
            "L6V1_R_hand_base_link",
        ),
        (SINGLE_ARM_ASSET_ROOT / "AR5V2_L" / "AR5V2_L.xml", "L", None),
        (SINGLE_ARM_ASSET_ROOT / "AR5V2_R" / "AR5V2_R.xml", "R", None),
    ),
    ids=("combined-left", "combined-right", "single-left", "single-right"),
)
def test_flange_placeholder_inertia_passes_newton_validation(
    asset: Path,
    side: str,
    nested_hand: str | None,
) -> None:
    """所有公开 MJCF 的法兰占位惯量都应避开 Newton 的近零特征值修正。"""

    root = ET.parse(asset).getroot()
    flange_name = f"AR5V2_{side}_arm_flan_link"
    flanges = tuple(
        body for body in root.iter("body") if body.get("name") == flange_name
    )

    assert len(flanges) == 1, asset
    flange = flanges[0]
    # MJCF body 没有 joint 时固定在父 body；这里不能把用于 TCP/标签的法兰层折叠掉。
    assert flange.findall("./joint") == [], asset
    expected_children = [] if nested_hand is None else [nested_hand]
    assert [body.get("name") for body in flange.findall("./body")] == expected_children

    inertial = flange.find("./inertial")
    assert inertial is not None, asset
    mass = float(str(inertial.get("mass")))
    principal_moments = tuple(
        float(value) for value in str(inertial.get("diaginertia")).split()
    )
    assert len(principal_moments) == 3, asset
    assert mass == pytest.approx(1.0e-6), asset

    # Newton 1.2.1 在最小特征值低于 max(1e-6 * Imax, 1e-10) 时修正惯量。
    # 取 1e-9 留出十倍余量，并显式校验刚体主惯量的三角不等式。
    correction_floor = max(1.0e-6 * max(principal_moments), 1.0e-10)
    assert min(principal_moments) >= correction_floor, asset
    first, second, third = sorted(principal_moments)
    assert first + second >= third, asset

    # I=2/5*m*r^2：当前占位值对应 50 mm 均匀球，惯量与质量尺度相容。
    equivalent_radius = math.sqrt(2.5 * first / mass)
    assert equivalent_radius == pytest.approx(0.05), asset


class _Path(str):
    def AppendPath(self, child: str) -> "_Path":
        return _Path(f"{self.rstrip('/')}/{child}")


class _Attr:
    def __init__(self, authored: dict[str, object], name: str) -> None:
        self.authored = authored
        self.name = name

    def Set(self, value: object) -> None:
        self.authored[self.name] = value


class _Prim:
    def __init__(
        self,
        name: str,
        *,
        path: str,
        schemas: tuple[object, ...] = (),
        type_name: str = "Xform",
    ) -> None:
        self.name = name
        self.path = path
        self.schemas = set(schemas)
        self.type_name = type_name

    def GetName(self) -> str:
        return self.name

    def GetTypeName(self) -> str:
        return self.type_name

    def GetParent(self):
        return None

    def HasAPI(self, schema: object) -> bool:
        return schema in self.schemas

    def IsValid(self) -> bool:
        return True


def _install_neutral_usd(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prim_specs: tuple[tuple[str, str, str, tuple[str, ...]], ...],
):
    authored: dict[str, object] = {}
    bindings: list[tuple[str, object]] = []
    collision_schema = object()

    class RigidBodyAPI:
        def __init__(self, prim: _Prim) -> None:
            self.prim = prim

        def CreateKinematicEnabledAttr(self) -> _Attr:
            return _Attr(authored, f"{self.prim.path}.kinematic")

    class MaterialAPI:
        @staticmethod
        def Apply(_prim: object):
            return SimpleNamespace(
                CreateStaticFrictionAttr=lambda: _Attr(authored, "material.static"),
                CreateDynamicFrictionAttr=lambda: _Attr(authored, "material.dynamic"),
                CreateRestitutionAttr=lambda: _Attr(authored, "material.restitution"),
            )

    class DriveAPI:
        @staticmethod
        def Apply(prim: _Prim, _drive_name: str):
            prefix = f"{prim.path}.drive"
            return SimpleNamespace(
                CreateTypeAttr=lambda: _Attr(authored, f"{prefix}.type"),
                CreateStiffnessAttr=lambda: _Attr(authored, f"{prefix}.stiffness"),
                CreateDampingAttr=lambda: _Attr(authored, f"{prefix}.damping"),
                CreateMaxForceAttr=lambda: _Attr(authored, f"{prefix}.max_force"),
            )

    schema_by_name = {
        "collision": collision_schema,
        "rigid": RigidBodyAPI,
    }
    prims = tuple(
        _Prim(
            name,
            path=path,
            type_name=type_name,
            schemas=tuple(schema_by_name[item] for item in schema_names),
        )
        for name, path, type_name, schema_names in prim_specs
    )
    root = _Prim("root", path="/World/Object")

    class Stage:
        def GetPrimAtPath(self, _path: object) -> _Prim:
            return root

    stage = Stage()

    class Material:
        def __init__(self, path: object) -> None:
            self.path = str(path)
            self.prim = SimpleNamespace(path=self.path)

        @classmethod
        def Define(cls, _stage: object, path: object):
            authored["material.path"] = str(path)
            return cls(path)

        def GetPrim(self) -> object:
            return self.prim

    class Binding:
        def __init__(self, prim: _Prim) -> None:
            self.prim = prim

        def Bind(self, material: object, **_kwargs: object) -> None:
            bindings.append((self.prim.path, material))

    pxr = ModuleType("pxr")
    pxr.Sdf = SimpleNamespace(Path=_Path)  # type: ignore[attr-defined]
    pxr.Usd = SimpleNamespace(PrimRange=lambda _root: prims)  # type: ignore[attr-defined]
    pxr.UsdPhysics = SimpleNamespace(  # type: ignore[attr-defined]
        CollisionAPI=collision_schema,
        DriveAPI=DriveAPI,
        MaterialAPI=MaterialAPI,
        RigidBodyAPI=RigidBodyAPI,
    )
    pxr.UsdShade = SimpleNamespace(  # type: ignore[attr-defined]
        Material=Material,
        MaterialBindingAPI=SimpleNamespace(Apply=Binding),
        Tokens=SimpleNamespace(strongerThanDescendants="stronger"),
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    return stage, root, authored, bindings, pxr


def _install_robot_runtime_modules(
    monkeypatch: pytest.MonkeyPatch, *, stage: object, root: object
) -> None:
    prims = ModuleType("isaacsim.core.utils.prims")
    prims.get_prim_at_path = lambda _path: root  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.utils", ModuleType("isaacsim.core.utils")
    )
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.prims", prims)

    usd = ModuleType("omni.usd")
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)  # type: ignore[attr-defined]
    omni = ModuleType("omni")
    omni.usd = usd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)


def test_newton_solver_pgs_keeps_experience_solver_without_physx_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_schema = object()
    scene = SimpleNamespace(IsA=lambda schema: schema is scene_schema)
    pxr = ModuleType("pxr")
    pxr.Usd = SimpleNamespace(PrimRange=lambda _root: ())  # type: ignore[attr-defined]
    pxr.UsdPhysics = SimpleNamespace(  # type: ignore[attr-defined]
        Scene=scene_schema, RigidBodyAPI=object()
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    stage = SimpleNamespace(Traverse=lambda: (scene,))
    with pytest.warns(PhysicsCompatibilityWarning, match="solver.type") as captured:
        counts = apply_solver_iteration_overrides(
            stage,
            "/World/Robot",
            SolverIterationConfig(solver_type="PGS"),
            physics_backend="newton",
        )

    assert len(captured) == 1
    assert captured[0].message.backend == "newton"
    assert captured[0].message.feature == "solver overrides"
    assert captured[0].message.skipped_fields == ("solver.type",)
    assert counts["physics_scenes"] == 1
    assert counts["skipped_physx_fields"] == 1
    assert not hasattr(pxr, "PhysxSchema")


def test_newton_solver_pgs_does_not_require_registered_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_schema = object()
    pxr = ModuleType("pxr")
    pxr.Usd = SimpleNamespace(PrimRange=lambda _root: ())  # type: ignore[attr-defined]
    pxr.UsdPhysics = SimpleNamespace(  # type: ignore[attr-defined]
        Scene=scene_schema, RigidBodyAPI=object()
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    stage = SimpleNamespace(Traverse=lambda: ())
    with pytest.warns(PhysicsCompatibilityWarning, match="experience-defined solver"):
        counts = apply_solver_iteration_overrides(
            stage,
            "/World/Robot",
            SolverIterationConfig(solver_type="PGS"),
            physics_backend="newton",
        )
    assert counts["physics_scenes"] == 0
    assert counts["skipped_physx_fields"] == 1


def test_newton_robot_projection_keeps_drive_and_native_mjcf_friction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, root, authored, bindings, pxr = _install_neutral_usd(
        monkeypatch,
        prim_specs=(
            ("shape", "/World/Robot/shape", "Mesh", ("collision",)),
            ("link", "/World/Robot/link", "Xform", ("rigid",)),
            (
                "joint_a",
                "/World/Robot/joint_a",
                "PhysicsRevoluteJoint",
                (),
            ),
        ),
    )
    _install_robot_runtime_modules(monkeypatch, stage=stage, root=root)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        counts = apply_robot_usd_overrides(
            "/World/Robot",
            RobotUsdOverrideConfig(),
            mjcf_path=Path("native_friction.xml"),
            physics_backend="newton",
        )

    assert authored["/World/Robot/joint_a.drive.type"] == "force"
    assert authored["/World/Robot/joint_a.drive.stiffness"] == 1000.0
    assert bindings == []
    assert counts["driven_joints"] == 1
    assert counts["skipped_physx_fields"] == 0
    assert not [
        item
        for item in captured
        if isinstance(item.message, PhysicsCompatibilityWarning)
    ]
    assert not hasattr(pxr, "PhysxSchema")


def test_newton_rigid_object_keeps_standard_static_and_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, _root, authored, bindings, pxr = _install_neutral_usd(
        monkeypatch,
        prim_specs=(
            (
                "body",
                "/World/Object/body",
                "Xform",
                ("collision", "rigid"),
            ),
        ),
    )
    physics = RigidObjectPhysicsConfig(
        static=True,
        material=ObjectMaterialConfig(
            static_friction=0.7,
            dynamic_friction=0.5,
            restitution=0.1,
        ),
        physx=RigidObjectPhysxConfig(
            material=ObjectPhysxMaterialConfig(friction_combine_mode="average")
        ),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _apply_rigid_object_physics(
            stage,
            "/World/Object",
            physics,
            physics_backend="newton",
        )

    assert authored["/World/Object/body.kinematic"] is True
    assert authored["material.static"] == 0.7
    assert authored["material.dynamic"] == 0.5
    assert authored["material.restitution"] == 0.1
    assert bindings[0][0] == "/World/Object/body"
    assert not [
        item
        for item in captured
        if isinstance(item.message, PhysicsCompatibilityWarning)
    ]
    assert not hasattr(pxr, "PhysxSchema")


def test_newton_rope_keeps_material_without_projecting_physx_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, _root, authored, bindings, pxr = _install_neutral_usd(
        monkeypatch,
        prim_specs=(
            (
                "segment_0",
                "/World/CapsuleRope/segment_0",
                "Xform",
                ("collision", "rigid"),
            ),
        ),
    )
    config = CapsuleRopeConfig(
        prim_path="/World/CapsuleRope",
        physics=CapsuleRopePhysicsConfig(
            material=ObjectMaterialConfig(
                static_friction=0.7,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            physx=CapsuleRopePhysxConfig(
                material=ObjectPhysxMaterialConfig(friction_combine_mode="average"),
                solver=CapsuleRopePhysxSolverConfig(
                    position_iterations=48,
                    velocity_iterations=4,
                ),
            ),
        ),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        counts = apply_capsule_rope_runtime_physics(
            stage, config, physics_backend="newton"
        )

    assert authored["material.static"] == 0.7
    assert bindings[0][0] == "/World/CapsuleRope/segment_0"
    assert counts == {
        "collision_prims": 1,
        "rigid_bodies": 1,
    }
    assert not [
        item
        for item in captured
        if isinstance(item.message, PhysicsCompatibilityWarning)
    ]
    assert not hasattr(pxr, "PhysxSchema")
