from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from linkerbot_sim.assets import usd_overrides
from linkerbot_sim.assets.usd_overrides import RobotUsdOverrideConfig
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
            RobotUsdOverrideConfig(
                contact_static_friction=static_friction,
                contact_material_override=True,
                friction_combine_mode="average",
            ),
            physics_backend="physx",
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
            RobotUsdOverrideConfig(),
            physics_backend="physx",
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
        RobotUsdOverrideConfig(
            contact_static_friction=0.2,
            contact_dynamic_friction=0.1,
            contact_material_override=True,
            friction_combine_mode="average",
        ),
        physics_backend="physx",
    )
    left_material = bindings[0][1]
    usd_overrides.apply_robot_usd_overrides(
        "/World/Robots/right",
        RobotUsdOverrideConfig(
            contact_static_friction=0.9,
            contact_dynamic_friction=0.7,
            contact_material_override=True,
            friction_combine_mode="average",
        ),
        physics_backend="physx",
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
        RobotUsdOverrideConfig(contact_material_override=False),
        physics_backend="physx",
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
        "default": RobotUsdOverrideConfig(drive_stiffness_seed=1.0),
        "arm": RobotUsdOverrideConfig(drive_stiffness_seed=2.0),
        "hand": RobotUsdOverrideConfig(
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


def test_newton_gravity_policy_projects_per_body_mujoco_gravcomp(
    monkeypatch,
) -> None:
    authored: dict[str, object] = {}
    rigid_body_api = object()

    class _Attr:
        def __init__(self, path: str, name: str) -> None:
            self.path = path
            self.name = name

        def Set(self, value: object) -> None:
            authored[f"{self.path}.{self.name}"] = value

    class _Body:
        def __init__(self, name: str, path: str) -> None:
            self.name = name
            self.path = path

        def GetName(self) -> str:
            return self.name

        def HasAPI(self, schema: object) -> bool:
            return schema is rigid_body_api

        def CreateAttribute(self, name: str, value_type: object) -> _Attr:
            assert value_type is float_type
            return _Attr(self.path, name)

    bodies = (
        _Body("arm_body", "/World/Robot/arm_body"),
        _Body("hand_body", "/World/Robot/hand_body"),
    )
    root = SimpleNamespace(IsValid=lambda: True)
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: root)
    float_type = object()

    pxr_module = ModuleType("pxr")
    pxr_module.Sdf = SimpleNamespace(ValueTypeNames=SimpleNamespace(Float=float_type))
    pxr_module.Usd = SimpleNamespace(PrimRange=lambda _root: bodies)
    pxr_module.UsdPhysics = SimpleNamespace(RigidBodyAPI=rigid_body_api)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    usd_module = ModuleType("omni.usd")
    usd_module.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni_module = ModuleType("omni")
    omni_module.usd = usd_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)

    mapping = RobotComponentMapping.from_profile(
        {"rigid_body_groups": {"arm": ["arm_body"], "hand": ["hand_body"]}}
    )
    policy = SimpleNamespace(enabled_for_component=lambda component: component == "arm")

    counts = usd_overrides.apply_robot_gravity_policy(
        "/World/Robot",
        policy,
        component_mapping=mapping,
        physics_backend="newton",
    )

    assert authored == {
        "/World/Robot/arm_body.mjc:gravcomp": 0.0,
        "/World/Robot/hand_body.mjc:gravcomp": 1.0,
    }
    assert counts == {
        "enabled": 1,
        "disabled": 1,
        "newton_gravcomp": 1,
        "skipped_physx_fields": 0,
    }


def test_native_mimic_follower_drive_is_zeroed() -> None:
    authored: dict[str, object] = {}

    class _Attr:
        def __init__(self, name: str) -> None:
            self.name = name

        def Set(self, value: object) -> None:
            authored[self.name] = value

    class _Drive:
        def CreateTypeAttr(self) -> _Attr:
            return _Attr("type")

        def CreateStiffnessAttr(self) -> _Attr:
            return _Attr("stiffness")

        def CreateDampingAttr(self) -> _Attr:
            return _Attr("damping")

        def CreateMaxForceAttr(self) -> _Attr:
            return _Attr("max_force")

    class _DriveAPI:
        @staticmethod
        def Apply(_prim, name: str) -> _Drive:
            authored["drive_name"] = name
            return _Drive()

    prim = SimpleNamespace(GetTypeName=lambda: "PhysicsRevoluteJoint")

    usd_overrides._disable_native_mimic_follower_drive(
        prim,
        SimpleNamespace(DriveAPI=_DriveAPI),
    )

    assert authored == {
        "drive_name": "angular",
        "type": "force",
        "stiffness": 0.0,
        "damping": 0.0,
        "max_force": 0.0,
    }


def test_newton_native_mimic_follower_drive_api_is_removed() -> None:
    removed: list[tuple[object, str]] = []
    drive_api = object()
    prim = SimpleNamespace(
        GetTypeName=lambda: "PhysicsRevoluteJoint",
        RemoveAPI=lambda schema, name: removed.append((schema, name)),
    )

    usd_overrides._disable_native_mimic_follower_drive(
        prim,
        SimpleNamespace(DriveAPI=drive_api),
        physics_backend="newton",
    )

    assert removed == [(drive_api, "angular")]


def test_remove_usd_drive_uses_multiple_apply_instance_name() -> None:
    removed: list[tuple[object, str]] = []
    drive_api = object()
    prim = SimpleNamespace(
        RemoveAPI=lambda schema, name: removed.append((schema, name))
    )

    usd_overrides._remove_usd_drive(prim, SimpleNamespace(DriveAPI=drive_api), "linear")

    assert removed == [(drive_api, "linear")]


def _install_empty_usd_stage(monkeypatch, *, root_valid: bool = True) -> None:
    pxr_module = ModuleType("pxr")
    pxr_module.PhysxSchema = SimpleNamespace()
    pxr_module.Usd = SimpleNamespace(PrimRange=lambda _root: ())
    pxr_module.UsdPhysics = SimpleNamespace()
    pxr_module.UsdShade = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    stage = SimpleNamespace(
        GetPrimAtPath=lambda _path: SimpleNamespace(IsValid=lambda: root_valid)
    )
    usd_module = ModuleType("omni.usd")
    usd_module.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
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

    stage = SimpleNamespace(GetPrimAtPath=lambda path: _Root(path))
    usd_module = ModuleType("omni.usd")
    usd_module.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni_module = ModuleType("omni")
    omni_module.usd = usd_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)
