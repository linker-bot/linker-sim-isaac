from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from linkerbot_sim.assets import usd_overrides
from linkerbot_sim.assets.usd_overrides import PhysxOverrideConfig
from linkerbot_sim.robots.classification import RobotComponentMapping


def test_robot_usd_overrides_isolate_material_prims_per_robot(
    monkeypatch,
) -> None:
    material_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        usd_overrides,
        "make_physics_material",
        lambda _stage, path, *values: (
            material_calls.append((path, tuple(values))) or object()
        ),
    )
    _install_empty_usd_stage(monkeypatch)

    for robot_root, static_friction in (
        ("/World/Robots/left", 0.2),
        ("/World/Robots/right", 0.9),
    ):
        usd_overrides.apply_robot_usd_overrides(
            robot_root,
            PhysxOverrideConfig(contact_static_friction=static_friction),
        )

    assert material_calls == [
        (
            "/World/Robots/left/PhysicsMaterials/RobotContactMaterial",
            (0.2, 0.6, 0.0, "average"),
        ),
        (
            "/World/Robots/right/PhysicsMaterials/RobotContactMaterial",
            (0.9, 0.6, 0.0, "average"),
        ),
    ]


def test_robot_usd_overrides_rejects_missing_root_before_creating_material(
    monkeypatch,
) -> None:
    material_paths: list[str] = []
    monkeypatch.setattr(
        usd_overrides,
        "make_physics_material",
        lambda _stage, path, *_values: material_paths.append(path) or object(),
    )
    _install_empty_usd_stage(monkeypatch, root_valid=False)

    try:
        usd_overrides.apply_robot_usd_overrides(
            "/World/Robots/missing",
            PhysxOverrideConfig(),
        )
    except ValueError as exc:
        assert "root prim does not exist" in str(exc)
    else:  # pragma: no cover - assertion helper keeps fake Isaac imports simple
        raise AssertionError("missing robot root must be rejected")

    assert material_paths == []


def test_robot_usd_overrides_bind_distinct_material_values_per_robot(
    monkeypatch,
) -> None:
    bindings: list[tuple[str, object]] = []
    _install_collision_usd_stage(monkeypatch, bindings=bindings)

    def make_material(_stage, path, static, dynamic, restitution, combine_mode):
        return SimpleNamespace(
            path=path,
            static=float(static),
            dynamic=float(dynamic),
            restitution=float(restitution),
            combine_mode=combine_mode,
        )

    monkeypatch.setattr(usd_overrides, "make_physics_material", make_material)

    usd_overrides.apply_robot_usd_overrides(
        "/World/Robots/left",
        PhysxOverrideConfig(
            contact_static_friction=0.2,
            contact_dynamic_friction=0.1,
        ),
    )
    left_material = bindings[0][1]
    usd_overrides.apply_robot_usd_overrides(
        "/World/Robots/right",
        PhysxOverrideConfig(
            contact_static_friction=0.9,
            contact_dynamic_friction=0.7,
        ),
    )

    assert [path for path, _material in bindings] == [
        "/World/Robots/left/collision",
        "/World/Robots/right/collision",
    ]
    assert left_material.path.startswith("/World/Robots/left/")
    assert left_material.static == 0.2
    assert left_material.dynamic == 0.1
    right_material = bindings[1][1]
    assert right_material.path.startswith("/World/Robots/right/")
    assert right_material.static == 0.9
    assert right_material.dynamic == 0.7
    assert left_material is not right_material


def test_robot_usd_overrides_preserves_asset_material_binding(monkeypatch) -> None:
    bindings: list[tuple[str, object]] = []
    material_paths: list[str] = []
    _install_collision_usd_stage(monkeypatch, bindings=bindings)
    monkeypatch.setattr(
        usd_overrides,
        "make_physics_material",
        lambda _stage, path, *_values: material_paths.append(path) or object(),
    )

    usd_overrides.apply_robot_usd_overrides(
        "/World/Robots/preserved",
        PhysxOverrideConfig(contact_material_override=False),
    )

    assert material_paths == []
    assert bindings == []


def test_nonstandard_collision_and_joint_usd_overrides_use_exact_groups() -> None:
    class _Prim:
        def __init__(self, name: str, type_name: str, parent=None) -> None:
            self.name = name
            self.type_name = type_name
            self.parent = parent

        def GetName(self) -> str:
            return self.name

        def GetTypeName(self) -> str:
            return self.type_name

        def GetParent(self):
            return self.parent

    mapping = RobotComponentMapping.from_profile(
        {
            "joint_groups": {"arm": ["axis_a"], "hand": ["axis_b"]},
            "rigid_body_groups": {"arm": ["body_a"], "hand": ["body_b"]},
        }
    )
    body = _Prim("body_b", "Xform")
    collision = _Prim("shape_x", "Mesh", body)
    active = _Prim("axis_b", "PhysicsRevoluteJoint")
    follower = _Prim("axis_shadow", "PhysicsRevoluteJoint")
    configs = {
        "default": PhysxOverrideConfig(drive_stiffness_seed=1.0),
        "arm": PhysxOverrideConfig(drive_stiffness_seed=2.0),
        "hand": PhysxOverrideConfig(
            drive_stiffness_seed={"axis_b": 22.0},
            follower_drive_stiffness_seed={"axis_shadow": 33.0},
            drive_damping_seed={"axis_b": 2.0},
            follower_drive_damping_seed={"axis_shadow": 3.0},
            joint_friction={"axis_b": 0.2},
            follower_joint_friction={"axis_shadow": 0.3},
            max_force={"axis_b": 202.0},
            follower_max_force={"axis_shadow": 303.0},
        ),
    }

    assert usd_overrides._component_for_prim(collision, mapping) == "hand"
    values = usd_overrides._resolve_usd_joint_parameters(
        [active, follower],
        configs,
        components=mapping,
        follower_master_by_name={"axis_shadow": "axis_b"},
    )
    assert values["axis_b"] == {
        "joint_friction": 0.2,
        "stiffness": 22.0,
        "damping": 2.0,
        "max_force": 202.0,
    }
    assert values["axis_shadow"] == {
        "joint_friction": 0.3,
        "stiffness": 33.0,
        "damping": 3.0,
        "max_force": 303.0,
    }


def _install_empty_usd_stage(monkeypatch, *, root_valid: bool = True) -> None:
    prims_module = ModuleType("isaacsim.core.utils.prims")
    prims_module.get_prim_at_path = lambda _path: SimpleNamespace(
        IsValid=lambda: root_valid
    )
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.utils", ModuleType("isaacsim.core.utils")
    )
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.prims", prims_module)

    pxr_module = ModuleType("pxr")
    pxr_module.PhysxSchema = SimpleNamespace()
    pxr_module.Usd = SimpleNamespace(PrimRange=lambda _root: ())
    pxr_module.UsdPhysics = SimpleNamespace()
    pxr_module.UsdShade = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    usd_module = ModuleType("omni.usd")
    usd_module.get_context = lambda: SimpleNamespace(get_stage=lambda: object())
    omni_module = ModuleType("omni")
    omni_module.usd = usd_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)


def _install_collision_usd_stage(monkeypatch, *, bindings) -> None:
    collision_api = object()

    class _Root:
        def __init__(self, path: str) -> None:
            self.path = path

        def IsValid(self) -> bool:
            return True

    class _Collision:
        def __init__(self, path: str) -> None:
            self.path = path

        def GetName(self) -> str:
            return "collision"

        def GetTypeName(self) -> str:
            return "Mesh"

        def HasAPI(self, schema) -> bool:
            return schema is collision_api

    def get_prim_at_path(path: str):
        return _Root(path)

    prims_module = ModuleType("isaacsim.core.utils.prims")
    prims_module.get_prim_at_path = get_prim_at_path
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.utils", ModuleType("isaacsim.core.utils")
    )
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.prims", prims_module)

    def prim_range(root):
        return (_Collision(f"{root.path}/collision"),)

    class _Binding:
        def __init__(self, prim) -> None:
            self.prim = prim

        def Bind(self, material, **_kwargs) -> None:
            bindings.append((self.prim.path, material))

    pxr_module = ModuleType("pxr")
    pxr_module.PhysxSchema = SimpleNamespace()
    pxr_module.Usd = SimpleNamespace(PrimRange=prim_range)
    pxr_module.UsdPhysics = SimpleNamespace(
        CollisionAPI=collision_api,
        RigidBodyAPI=object(),
    )
    pxr_module.UsdShade = SimpleNamespace(
        MaterialBindingAPI=SimpleNamespace(Apply=_Binding),
        Tokens=SimpleNamespace(strongerThanDescendants="stronger"),
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    usd_module = ModuleType("omni.usd")
    usd_module.get_context = lambda: SimpleNamespace(get_stage=lambda: object())
    omni_module = ModuleType("omni")
    omni_module.usd = usd_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)
