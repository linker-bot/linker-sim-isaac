from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import warnings

import pytest

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
    def __init__(self, path: str, schemas: tuple[object, ...]) -> None:
        self.path = path
        self.schemas = set(schemas)

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def HasAPI(self, schema: object) -> bool:
        return schema in self.schemas

    def IsValid(self) -> bool:
        return True


def _install_physics_stage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_physx: bool,
) -> tuple[object, dict[str, object], list[str], ModuleType]:
    """安装只覆盖对象材质与 solver 投影所需的最小 USD schema。"""

    authored: dict[str, object] = {}
    bindings: list[str] = []
    collision_schema = object()

    class RigidBodyAPI:
        def __init__(self, prim: _Prim) -> None:
            self.prim = prim

        def CreateKinematicEnabledAttr(self) -> _Attr:
            return _Attr(authored, f"{self.prim.path}.kinematic")

    body = _Prim(
        "/World/Object/body",
        (collision_schema, RigidBodyAPI),
    )
    root = _Prim("/World/Object", ())

    class Stage:
        def GetPrimAtPath(self, _path: object) -> _Prim:
            return root

    class Material:
        def __init__(self, path: object) -> None:
            self.path = str(path)
            self.prim = _Prim(self.path, ())

        @classmethod
        def Define(cls, _stage: object, path: object) -> "Material":
            authored["material.path"] = str(path)
            return cls(path)

        def GetPrim(self) -> _Prim:
            return self.prim

    class MaterialBindingAPI:
        @classmethod
        def Apply(cls, prim: _Prim) -> "MaterialBindingAPI":
            return cls(prim)

        def __init__(self, prim: _Prim) -> None:
            self.prim = prim

        def Bind(self, _material: object, **_kwargs: object) -> None:
            bindings.append(self.prim.path)

    material_api = SimpleNamespace(
        Apply=lambda _prim: SimpleNamespace(
            CreateStaticFrictionAttr=lambda: _Attr(authored, "material.static"),
            CreateDynamicFrictionAttr=lambda: _Attr(authored, "material.dynamic"),
            CreateRestitutionAttr=lambda: _Attr(authored, "material.restitution"),
        )
    )
    pxr = ModuleType("pxr")
    pxr.Sdf = SimpleNamespace(Path=_Path)  # type: ignore[attr-defined]
    pxr.Usd = SimpleNamespace(PrimRange=lambda _root: (body,))  # type: ignore[attr-defined]
    pxr.UsdPhysics = SimpleNamespace(  # type: ignore[attr-defined]
        CollisionAPI=collision_schema,
        MaterialAPI=material_api,
        RigidBodyAPI=RigidBodyAPI,
    )
    pxr.UsdShade = SimpleNamespace(  # type: ignore[attr-defined]
        Material=Material,
        MaterialBindingAPI=MaterialBindingAPI,
        Tokens=SimpleNamespace(strongerThanDescendants="stronger"),
    )

    if include_physx:

        class PhysxMaterialAPI:
            @staticmethod
            def Apply(_prim: object) -> object:
                return SimpleNamespace(
                    CreateFrictionCombineModeAttr=lambda: _Attr(
                        authored, "material.combine"
                    )
                )

        class PhysxRigidBodyAPI:
            def __init__(self, prim: _Prim) -> None:
                self.prim = prim

            @classmethod
            def Apply(cls, prim: _Prim) -> "PhysxRigidBodyAPI":
                return cls(prim)

            def CreateSolverPositionIterationCountAttr(self) -> _Attr:
                return _Attr(authored, f"{self.prim.path}.solver.position")

            def CreateSolverVelocityIterationCountAttr(self) -> _Attr:
                return _Attr(authored, f"{self.prim.path}.solver.velocity")

            def CreateDisableGravityAttr(self) -> _Attr:
                return _Attr(authored, f"{self.prim.path}.disable_gravity")

        pxr.PhysxSchema = SimpleNamespace(  # type: ignore[attr-defined]
            PhysxMaterialAPI=PhysxMaterialAPI,
            PhysxRigidBodyAPI=PhysxRigidBodyAPI,
        )

    monkeypatch.setitem(sys.modules, "pxr", pxr)
    return Stage(), authored, bindings, pxr


def _material() -> ObjectMaterialConfig:
    return ObjectMaterialConfig(
        static_friction=0.7,
        dynamic_friction=0.5,
        restitution=0.1,
    )


def _physx_material() -> ObjectPhysxMaterialConfig:
    return ObjectPhysxMaterialConfig(friction_combine_mode="average")


def _rope_physics() -> CapsuleRopePhysicsConfig:
    return CapsuleRopePhysicsConfig(
        material=_material(),
        physx=CapsuleRopePhysxConfig(
            material=_physx_material(),
            solver=CapsuleRopePhysxSolverConfig(
                position_iterations=48,
                velocity_iterations=4,
            ),
        ),
    )


def test_object_physics_paths_parse_to_typed_backend_leaves() -> None:
    rigid = RigidObjectPhysicsConfig.from_mapping(
        {
            "material": {"static_friction": 0.7},
            "physx": {"material": {"friction_combine_mode": "average"}},
        },
        label="object.physics",
    )
    rope = CapsuleRopePhysicsConfig.from_mapping(
        {
            "material": {"dynamic_friction": 0.5},
            "physx": {
                "material": {"friction_combine_mode": "max"},
                "solver": {
                    "position_iterations": 48,
                    "velocity_iterations": 4,
                },
            },
        },
        label="object.physics",
    )

    assert rigid.material is not None
    assert rigid.material.static_friction == 0.7
    assert rigid.physx is not None and rigid.physx.material is not None
    assert rigid.physx.material.friction_combine_mode == "average"
    assert rope.material is not None
    assert rope.material.dynamic_friction == 0.5
    assert rope.physx is not None and rope.physx.material is not None
    assert rope.physx.material.friction_combine_mode == "max"
    assert rope.physx.solver is not None
    assert rope.physx.solver.position_iterations == 48
    assert rope.physx.solver.velocity_iterations == 4


def test_legacy_object_physics_paths_are_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"object\.physics\.material\.friction_combine_mode"
    ):
        ObjectMaterialConfig.from_mapping(
            {"friction_combine_mode": "average"},
            label="object.physics.material",
        )
    with pytest.raises(
        ValueError, match=r"object\.physics\.solver_position_iterations"
    ):
        CapsuleRopePhysicsConfig.from_mapping(
            {"solver_position_iterations": 48},
            label="object.physics",
        )


def test_newton_rigid_object_projects_only_common_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, authored, bindings, pxr = _install_physics_stage(
        monkeypatch, include_physx=False
    )
    physics = RigidObjectPhysicsConfig(
        material=_material(),
        physx=RigidObjectPhysxConfig(material=_physx_material()),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _apply_rigid_object_physics(
            stage,
            "/World/Object",
            physics,
            physics_backend="newton",
        )

    assert captured == []
    assert authored["material.static"] == 0.7
    assert authored["material.dynamic"] == 0.5
    assert authored["material.restitution"] == 0.1
    assert "material.combine" not in authored
    assert bindings == ["/World/Object/body"]
    assert not hasattr(pxr, "PhysxSchema")


def test_physx_rigid_object_combines_common_and_physx_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, authored, bindings, _pxr = _install_physics_stage(
        monkeypatch, include_physx=True
    )
    physics = RigidObjectPhysicsConfig(
        material=_material(),
        physx=RigidObjectPhysxConfig(material=_physx_material()),
    )

    _apply_rigid_object_physics(
        stage,
        "/World/Object",
        physics,
        physics_backend="physx",
    )

    assert authored["material.static"] == 0.7
    assert authored["material.combine"] == "average"
    assert bindings == ["/World/Object/body"]


def test_newton_rope_ignores_physx_leaf_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, authored, bindings, pxr = _install_physics_stage(
        monkeypatch, include_physx=False
    )
    config = CapsuleRopeConfig(
        prim_path="/World/Object",
        physics=_rope_physics(),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        counts = apply_capsule_rope_runtime_physics(
            stage,
            config,
            physics_backend="newton",
        )

    assert captured == []
    assert counts == {"collision_prims": 1, "rigid_bodies": 1}
    assert authored["material.static"] == 0.7
    assert "material.combine" not in authored
    assert all("solver" not in key for key in authored)
    assert bindings == ["/World/Object/body"]
    assert not hasattr(pxr, "PhysxSchema")


def test_physx_rope_projects_material_and_solver_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, authored, bindings, _pxr = _install_physics_stage(
        monkeypatch, include_physx=True
    )
    config = CapsuleRopeConfig(
        prim_path="/World/Object",
        physics=_rope_physics(),
    )

    counts = apply_capsule_rope_runtime_physics(
        stage,
        config,
        physics_backend="physx",
    )

    assert counts == {"collision_prims": 1, "rigid_bodies": 1}
    assert authored["material.static"] == 0.7
    assert authored["material.combine"] == "average"
    assert authored["/World/Object/body.solver.position"] == 48
    assert authored["/World/Object/body.solver.velocity"] == 4
    assert bindings == ["/World/Object/body"]
