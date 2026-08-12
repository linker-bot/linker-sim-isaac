from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from linkerbot_sim.assets import robot_import
from linkerbot_sim.assets.robot_config import RobotAssetConfig
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.robots import AssetImportConfig
from linkerbot_sim.assets.robot_import import (
    _apply_mesh_collision_approximation,
    _deactivate_imported_mjcf_actuators,
    _discover_imported_root_path,
    _reference_imported_prim_from_usd,
    configure_mjcf_import,
    configure_urdf_import,
    release_imported_asset_files,
)


@pytest.fixture(autouse=True)
def _cleanup_imported_asset_files():
    yield
    release_imported_asset_files()


class FakeImportConfig:
    def __init__(self, **kwargs: object) -> None:
        vars(self).update(kwargs)


def test_configure_mjcf_import_sets_self_collision(monkeypatch) -> None:
    importer = _install_fake_importer(monkeypatch, "mjcf")

    imported_path = configure_mjcf_import(
        Path("robot.xml"),
        "/World/Robot",
        asset_import_config=AssetImportConfig(self_collision=True),
    )

    assert imported_path == "/World/Robot"
    assert importer.method_calls == ["import_mjcf"]
    assert importer.references == [
        (
            importer.generated_path,
            "/ImportedRobot",
            "/World/Robot",
        )
    ]
    assert importer.config.allow_self_collision is True
    assert importer.config.collision_type == "Convex Decomposition"
    assert importer.config.fix_base is True
    assert importer.config.import_scene is False
    assert importer.config.run_asset_transformer is True
    assert importer.config.run_multi_physics_conversion is True


def test_configure_mjcf_import_defaults_self_collision_to_false(monkeypatch) -> None:
    importer = _install_fake_importer(monkeypatch, "mjcf")

    configure_mjcf_import(Path("robot.xml"), "/World/Robot")

    assert importer.config.allow_self_collision is False


def test_configure_mjcf_import_forwards_explicit_backend(monkeypatch) -> None:
    _install_fake_importer(monkeypatch, "mjcf")
    forwarded: list[tuple[object, bool]] = []
    prepared: list[tuple[object, object]] = []
    deactivated: list[str] = []
    monkeypatch.setattr(
        robot_import,
        "active_physics_backend",
        lambda: pytest.fail("explicit backend must not read the runtime registry"),
    )
    monkeypatch.setattr(
        robot_import,
        "_reference_imported_prim_from_usd",
        lambda _path, **kwargs: forwarded.append(
            (
                kwargs["physics_backend"],
                kwargs["prepare_newton_render_topology"],
            )
        ),
    )
    monkeypatch.setattr(
        robot_import,
        "_prepare_newton_render_reference_asset",
        lambda path, **kwargs: (
            prepared.append((kwargs["physics_backend"], kwargs["root_pose"])) or path
        ),
    )
    monkeypatch.setattr(
        robot_import,
        "_deactivate_imported_mjcf_actuators",
        lambda root: deactivated.append(root),
    )

    configure_mjcf_import(
        Path("robot.xml"),
        "/World/Robot",
        physics_backend="newton",
        prepare_newton_render_topology=True,
    )

    assert forwarded == [("newton", True)]
    assert prepared == [("newton", RootPoseConfig())]
    assert deactivated == ["/World/Robot"]


def test_configure_mjcf_import_applies_named_importer_settings(monkeypatch) -> None:
    importer = _install_fake_importer(monkeypatch, "mjcf")

    configure_mjcf_import(
        Path("robot.xml"),
        "/World/Robot",
        asset_import_config=AssetImportConfig(
            fix_base=False,
            collision_approximation="convex_hull",
        ),
    )

    assert importer.config.fix_base is False
    assert importer.config.collision_type == "Convex Hull"


def test_configure_mjcf_import_rejects_removed_direct_config(monkeypatch) -> None:
    _install_fake_importer(monkeypatch, "mjcf")

    with pytest.raises(ValueError, match="does not support merge_fixed_joints"):
        configure_mjcf_import(
            Path("robot.xml"),
            "/World/Robot",
            asset_import_config=AssetImportConfig(merge_fixed_joints=True),
        )


def test_configure_mjcf_import_rejects_nonlinear_native_mimic(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_importer(monkeypatch, "mjcf")
    mjcf_path = tmp_path / "nonlinear.xml"
    mjcf_path.write_text(
        """<mujoco>
  <worldbody><body><joint name="master"/><joint name="follower"/></body></worldbody>
  <equality><joint name="curve" joint1="follower" joint2="master"
                   polycoef="0 1 0.5 0 0"/></equality>
</mujoco>
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="NewtonMimicAPI only supports"):
        configure_mjcf_import(mjcf_path, "/World/Robot")


def test_configure_urdf_import_sets_self_collision(monkeypatch) -> None:
    importer = _install_fake_importer(monkeypatch, "urdf")

    imported_path = configure_urdf_import(
        Path("robot.urdf"),
        asset_import_config=AssetImportConfig(
            collision_approximation="convex_hull",
            self_collision=True,
        ),
    )

    assert imported_path == "/ImportedRobot"
    assert importer.method_calls == ["import_urdf"]
    assert importer.config.allow_self_collision is True
    assert importer.config.collision_type == "Convex Hull"


def test_configure_urdf_import_defaults_self_collision_to_false(monkeypatch) -> None:
    importer = _install_fake_importer(monkeypatch, "urdf")

    imported_path = configure_urdf_import(
        Path("robot.urdf"),
        get_articulation_root=False,
    )

    assert importer.config.allow_self_collision is False
    assert imported_path == "/ImportedRobot"
    assert importer.config.merge_fixed_joints is False
    assert importer.config.fix_base is True


def test_configure_urdf_import_forwards_explicit_backend(monkeypatch) -> None:
    _install_fake_importer(monkeypatch, "urdf")
    forwarded: list[object] = []
    monkeypatch.setattr(
        robot_import,
        "active_physics_backend",
        lambda: pytest.fail("explicit backend must not read the runtime registry"),
    )
    monkeypatch.setattr(
        robot_import,
        "_reference_imported_prim_from_usd",
        lambda _path, **kwargs: forwarded.append(kwargs["physics_backend"]),
    )

    configure_urdf_import(
        Path("robot.urdf"),
        physics_backend="physx",
    )

    assert forwarded == ["physx"]


def test_configure_urdf_import_applies_named_importer_settings(monkeypatch) -> None:
    importer = _install_fake_importer(monkeypatch, "urdf")

    configure_urdf_import(
        Path("robot.urdf"),
        asset_import_config=AssetImportConfig(
            fix_base=False,
            merge_fixed_joints=False,
            collision_from_visuals=True,
        ),
    )

    assert importer.config.fix_base is False
    assert importer.config.merge_fixed_joints is False
    assert importer.config.collision_from_visuals is True
    assert importer.config.joint_drive_type == "force"
    assert importer.config.joint_target_type == "position"
    assert importer.config.override_joint_stiffness == 1.0e5
    assert importer.config.override_joint_damping == 1.0e4


def test_configure_urdf_import_disables_drive_with_public_target_type(
    monkeypatch,
) -> None:
    importer = _install_fake_importer(monkeypatch, "urdf")

    configure_urdf_import(Path("robot.urdf"), drive_type="none")

    assert importer.config.joint_target_type == "none"
    assert importer.config.override_joint_stiffness == 0.0
    assert importer.config.override_joint_damping == 0.0


def test_import_robot_asset_honors_resolved_urdf_prim_path(
    monkeypatch, tmp_path: Path
) -> None:
    asset_path = tmp_path / "robot.urdf"
    asset_path.write_text("<robot name='test'/>", encoding="utf-8")
    resolved_prim_path = "/World/Robots/urdf_instance"
    received: dict[str, object] = {}

    def fake_configure_urdf_import(
        urdf_path: Path, *args: object, **kwargs: object
    ) -> str:
        received["urdf_path"] = urdf_path
        received["prim_path"] = kwargs.get("prim_path", args[0] if args else None)
        received["physics_backend"] = kwargs.get("physics_backend")
        received["prepare_newton_render_topology"] = kwargs.get(
            "prepare_newton_render_topology"
        )
        return str(received["prim_path"] or "/World/ImportedRobot")

    monkeypatch.setattr(
        robot_import, "configure_urdf_import", fake_configure_urdf_import
    )
    config = RobotAssetConfig(
        asset_type="urdf",
        asset_path=asset_path,
        prim_path=resolved_prim_path,
    )

    articulation_path, imported_asset_path, imported_root_path = (
        robot_import.import_robot_asset(
            config,
            physics_backend="newton",
            prepare_newton_render_topology=True,
            root_pose=RootPoseConfig(),
        )
    )

    assert received == {
        "urdf_path": asset_path.resolve(),
        "prim_path": resolved_prim_path,
        "physics_backend": "newton",
        "prepare_newton_render_topology": True,
    }
    assert articulation_path == resolved_prim_path
    assert imported_root_path == resolved_prim_path
    assert imported_asset_path == asset_path.resolve()


def test_configure_urdf_import_maps_base_before_resolving_articulation(
    monkeypatch,
) -> None:
    importer = _install_fake_importer(monkeypatch, "urdf")
    monkeypatch.setattr(
        robot_import,
        "find_articulation_root",
        lambda root: f"{root}/base_link",
    )

    articulation_path = configure_urdf_import(
        Path("robot.urdf"),
        prim_path="/World/Robots/robot_a",
    )

    assert articulation_path == "/World/Robots/robot_a/base_link"
    assert importer.references == [
        (
            importer.generated_path,
            "/ImportedRobot",
            "/World/Robots/robot_a",
        )
    ]


def test_find_articulation_root_accepts_importer_scope_with_rigid_children() -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/MjcfRobot", "Xform")
    articulation = stage.DefinePrim("/World/MjcfRobot/Geometry", "Scope")
    rigid_body = stage.DefinePrim(
        "/World/MjcfRobot/Geometry/base_link",
        "Xform",
    )
    UsdPhysics.ArticulationRootAPI.Apply(articulation)
    UsdPhysics.RigidBodyAPI.Apply(rigid_body)

    assert (
        robot_import.find_articulation_root("/World/MjcfRobot", stage=stage)
        == "/World/MjcfRobot/Geometry"
    )


def test_file_backed_import_reference_remaps_internal_targets(tmp_path: Path) -> None:
    from pxr import Usd

    source_path = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = source_stage.DefinePrim("/Source", "Xform")
    source_root.GetVariantSets().AddVariantSet("Physics").AddVariant("physx")
    source_stage.DefinePrim("/Source/Child", "Xform")
    source_root.CreateRelationship("child").AddTarget("/Source/Child")
    source_stage.GetRootLayer().Save()
    destination_stage = Usd.Stage.CreateInMemory()
    destination_stage.DefinePrim("/World")

    _reference_imported_prim_from_usd(
        source_path,
        source_path="/Source",
        target_path="/World/Target",
        destination_stage=destination_stage,
        physics_backend="physx",
    )

    target = destination_stage.GetPrimAtPath("/World/Target")
    assert target.IsValid()
    assert [str(path) for path in target.GetRelationship("child").GetTargets()] == [
        "/World/Target/Child"
    ]


def test_newton_render_reference_starts_with_canonical_root_topology(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source_path = tmp_path / "newton-render-source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/Source")
    source_root.AddTranslateOp().Set((1.0, 2.0, 3.0))
    source_root.AddRotateXYZOp().Set((10.0, 20.0, 30.0))
    variants = source_root.GetPrim().GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("mujoco")
    source_stage.GetRootLayer().Save()
    destination_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(destination_stage, "/World")

    _reference_imported_prim_from_usd(
        source_path,
        source_path="/Source",
        target_path="/World/Target",
        destination_stage=destination_stage,
        physics_backend="newton",
        prepare_newton_render_topology=True,
    )

    target = destination_stage.GetPrimAtPath("/World/Target")
    xform = UsdGeom.Xformable(target)
    assert tuple(str(op.GetOpName()) for op in xform.GetOrderedXformOps()) == (
        "xformOp:transform:newtonRenderWorld",
    )
    assert xform.GetResetXformStack()
    assert target.GetVariantSet("Physics").GetVariantSelection() == "mujoco"


def test_newton_render_wrapper_bakes_nested_body_topology_and_root_pose(
    tmp_path: Path,
) -> None:
    from linkerbot_sim.isaac.physics.newton.render import (
        prepare_newton_render_subtree,
    )
    from pxr import Tf, Usd, UsdGeom, UsdPhysics

    source_path = tmp_path / "nested-source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/Source")
    source_root.AddTranslateOp().Set((9.0, 0.0, 0.0))
    variants = source_root.GetPrim().GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("mujoco")
    body = UsdGeom.Xform.Define(source_stage, "/Source/Body")
    body.AddTranslateOp().Set((2.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    source_stage.SetDefaultPrim(source_root.GetPrim())
    source_stage.GetRootLayer().Save()
    pose = RootPoseConfig(xyz=(1.0, 0.0, 0.0))

    wrapper_path = robot_import._prepare_newton_render_reference_asset(
        source_path,
        source_path="/Source",
        root_pose=pose,
        physics_backend="newton",
    )

    wrapper_stage = Usd.Stage.Open(str(wrapper_path))
    wrapper_body = UsdGeom.Xformable(wrapper_stage.GetPrimAtPath("/Source/Body"))
    assert tuple(str(op.GetOpName()) for op in wrapper_body.GetOrderedXformOps()) == (
        "xformOp:transform:newtonRenderWorld",
    )
    assert wrapper_body.GetResetXformStack()
    assert tuple(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(wrapper_body.GetPrim())
        .ExtractTranslation()
    ) == pytest.approx((3.0, 0.0, 0.0))

    live_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(live_stage, "/World")
    robot_import._reference_imported_prim_from_usd(
        wrapper_path,
        source_path="/Source",
        target_path="/World/Robot",
        destination_stage=live_stage,
        physics_backend="newton",
        prepare_newton_render_topology=True,
    )
    changed_paths: list[str] = []

    def _record_changed_paths(notice: object, _sender: object) -> None:
        changed_paths.extend(str(path) for path in notice.GetChangedInfoOnlyPaths())

    registration = Tf.Notice.Register(
        Usd.Notice.ObjectsChanged,
        _record_changed_paths,
        live_stage,
    )
    try:
        robot_import.apply_root_pose(
            live_stage,
            "/World/Robot",
            pose,
            prepare_newton_render_topology=True,
        )
        prepare_newton_render_subtree(
            stage=live_stage,
            subtree_root="/World/Robot",
        )
    finally:
        registration.Revoke()

    assert not any(path.endswith(".xformOpOrder") for path in changed_paths)
    live_body = UsdGeom.Xformable(live_stage.GetPrimAtPath("/World/Robot/Body"))
    assert tuple(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(live_body.GetPrim())
        .ExtractTranslation()
    ) == pytest.approx((3.0, 0.0, 0.0))


def test_discover_imported_root_prefers_default_prim(tmp_path: Path) -> None:
    from pxr import Usd

    source_path = tmp_path / "default.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    selected = source_stage.DefinePrim("/Selected", "Xform")
    source_stage.DefinePrim("/Other", "Xform")
    source_stage.SetDefaultPrim(selected)
    source_stage.GetRootLayer().Save()

    assert _discover_imported_root_path(source_path) == "/Selected"


def test_discover_imported_root_rejects_ambiguous_stage(tmp_path: Path) -> None:
    from pxr import Usd

    source_path = tmp_path / "ambiguous.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_stage.DefinePrim("/First", "Xform")
    source_stage.DefinePrim("/Second", "Xform")
    source_stage.GetRootLayer().Save()

    with pytest.raises(RuntimeError, match="must have one root prim"):
        _discover_imported_root_path(source_path)


@pytest.mark.parametrize(
    ("physics_backend", "expected"),
    (("physx", "physx"), ("newton", "mujoco")),
)
def test_file_backed_reference_selects_backend_variant(
    tmp_path: Path,
    physics_backend: str,
    expected: str,
) -> None:
    from pxr import Usd

    source_path = tmp_path / "variants.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = source_stage.DefinePrim("/Source", "Xform")
    variants = source_root.GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("mujoco")
    variants.AddVariant("physx")
    source_stage.GetRootLayer().Save()
    destination_stage = Usd.Stage.CreateInMemory()
    destination_stage.DefinePrim("/World")

    _reference_imported_prim_from_usd(
        source_path,
        source_path="/Source",
        target_path="/World/Target",
        destination_stage=destination_stage,
        physics_backend=physics_backend,
    )

    target = destination_stage.GetPrimAtPath("/World/Target")
    assert target.GetVariantSet("Physics").GetVariantSelection() == expected


def test_file_backed_reference_rejects_missing_backend_variant(tmp_path: Path) -> None:
    from pxr import Usd

    source_path = tmp_path / "variants.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = source_stage.DefinePrim("/Source", "Xform")
    source_root.GetVariantSets().AddVariantSet("Physics").AddVariant("physx")
    source_stage.GetRootLayer().Save()
    destination_stage = Usd.Stage.CreateInMemory()
    destination_stage.DefinePrim("/World")

    with pytest.raises(RuntimeError, match="required='mujoco'"):
        _reference_imported_prim_from_usd(
            source_path,
            source_path="/Source",
            target_path="/World/Target",
            destination_stage=destination_stage,
            physics_backend="newton",
        )


def test_file_backed_reference_allows_common_physics_for_rigid_object(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    source_path = tmp_path / "common-physics.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = source_stage.DefinePrim("/Source", "Xform")
    variants = source_root.GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("none")
    variants.AddVariant("physics")
    source_stage.GetRootLayer().Save()
    destination_stage = Usd.Stage.CreateInMemory()
    destination_stage.DefinePrim("/World")

    _reference_imported_prim_from_usd(
        source_path,
        source_path="/Source",
        target_path="/World/Target",
        destination_stage=destination_stage,
        physics_backend="newton",
        allow_common_physics_variant=True,
    )

    target = destination_stage.GetPrimAtPath("/World/Target")
    assert target.GetVariantSet("Physics").GetVariantSelection() == "physics"


def test_apply_mesh_collision_approximation_only_updates_collision_meshes() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot")
    collision_mesh = UsdGeom.Mesh.Define(stage, "/World/Robot/collision")
    visual_mesh = UsdGeom.Mesh.Define(stage, "/World/Robot/visual")
    UsdPhysics.CollisionAPI.Apply(collision_mesh.GetPrim())

    count = _apply_mesh_collision_approximation(
        "/World/Robot",
        approximation="convex_decomposition",
        stage=stage,
    )

    assert count == 1
    assert (
        UsdPhysics.MeshCollisionAPI(collision_mesh.GetPrim())
        .GetApproximationAttr()
        .Get()
        == "convexDecomposition"
    )
    assert not visual_mesh.GetPrim().HasAPI(UsdPhysics.MeshCollisionAPI)


def test_deactivate_imported_mjcf_actuators_keeps_joint_target_and_other_prims() -> (
    None
):
    pytest.importorskip("pxr")
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot", "Xform")
    joint = stage.DefinePrim(
        "/World/Robot/Geometry/joint",
        "PhysicsRevoluteJoint",
    )
    actuator = stage.DefinePrim(
        "/World/Robot/Physics/joint_motor",
        "MjcActuator",
    )
    actuator.CreateRelationship("mjc:target").SetTargets([joint.GetPath()])
    visual = stage.DefinePrim("/World/Robot/Geometry/visual", "Mesh")

    count = _deactivate_imported_mjcf_actuators("/World/Robot", stage=stage)

    assert count == 1
    assert not stage.GetPrimAtPath(actuator.GetPath()).IsActive()
    assert stage.GetPrimAtPath(joint.GetPath()).IsActive()
    assert stage.GetPrimAtPath(visual.GetPath()).IsActive()


def test_deactivate_imported_mjcf_actuators_rejects_non_joint_target() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot", "Xform")
    site = stage.DefinePrim("/World/Robot/Geometry/site", "Xform")
    actuator = stage.DefinePrim(
        "/World/Robot/Physics/site_motor",
        "MjcActuator",
    )
    actuator.CreateRelationship("mjc:target").SetTargets([site.GetPath()])

    with pytest.raises(RuntimeError, match="only supports joint motors"):
        _deactivate_imported_mjcf_actuators("/World/Robot", stage=stage)

    assert stage.GetPrimAtPath(actuator.GetPath()).IsActive()


class FakeImporterRuntime:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.config: FakeImportConfig | None = None
        self.generated_path = Path()
        self.method_calls: list[str] = []
        self.references: list[tuple[object, str, str]] = []
        self.collision_approximations: list[tuple[str, str]] = []


def _install_fake_importer(monkeypatch, mode: str) -> FakeImporterRuntime:
    runtime = FakeImporterRuntime(mode)

    class PublicConfig(FakeImportConfig):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            runtime.config = self

    class PublicImporter:
        def __init__(self, config: FakeImportConfig) -> None:
            self.config = config

        def _import(self, method: str) -> str:
            runtime.method_calls.append(method)
            source_path = Path(str(self.config.usd_path))
            runtime.generated_path = source_path / "robot" / "robot.usda"
            return str(runtime.generated_path)

        def import_mjcf(self) -> str:
            return self._import("import_mjcf")

        def import_urdf(self) -> str:
            return self._import("import_urdf")

    isaacsim_module = types.ModuleType("isaacsim")
    asset_module = types.ModuleType("isaacsim.asset")
    importer_module = types.ModuleType("isaacsim.asset.importer")
    public_module = types.ModuleType(f"isaacsim.asset.importer.{mode}")
    if mode == "mjcf":
        public_module.MJCFImporter = PublicImporter
        public_module.MJCFImporterConfig = PublicConfig
    else:
        public_module.URDFImporter = PublicImporter
        public_module.URDFImporterConfig = PublicConfig
    setattr(importer_module, mode, public_module)
    asset_module.importer = importer_module
    isaacsim_module.asset = asset_module
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.asset", asset_module)
    monkeypatch.setitem(sys.modules, "isaacsim.asset.importer", importer_module)
    monkeypatch.setitem(sys.modules, f"isaacsim.asset.importer.{mode}", public_module)
    monkeypatch.setattr(
        robot_import,
        "_reference_imported_prim_from_usd",
        lambda source_usd_path, *, source_path, target_path, **_kwargs: (
            runtime.references.append((source_usd_path, source_path, target_path))
        ),
    )
    monkeypatch.setattr(
        robot_import,
        "_apply_mesh_collision_approximation",
        lambda root_path, *, approximation: runtime.collision_approximations.append(
            (root_path, approximation)
        ),
    )
    monkeypatch.setattr(
        robot_import,
        "_discover_imported_root_path",
        lambda _path: "/ImportedRobot",
    )
    monkeypatch.setattr(
        robot_import,
        "find_articulation_root",
        lambda root, **_kwargs: root,
    )
    return runtime
