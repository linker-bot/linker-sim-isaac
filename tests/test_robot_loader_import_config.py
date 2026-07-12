from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from linkerbot_sim.assets import robot_import
from linkerbot_sim.assets.robot_config import AssetImportConfig, RobotAssetConfig
from linkerbot_sim.assets.robot_import import (
    _reference_imported_prim_from_usd,
    configure_mjcf_import,
    configure_urdf_import,
    release_imported_asset_files,
)


@pytest.fixture(autouse=True)
def _cleanup_imported_asset_files():
    yield
    release_imported_asset_files()


class FakeMJCFImportConfig:
    def __init__(self) -> None:
        self.fix_base = None
        self.import_inertia_tensor = None
        self.import_sites = None
        self.merge_fixed_joints = None
        self.convex_decomp = None
        self.self_collision = None

    def set_fix_base(self, value: bool) -> None:
        self.fix_base = value

    def set_import_inertia_tensor(self, value: bool) -> None:
        self.import_inertia_tensor = value

    def set_import_sites(self, value: bool) -> None:
        self.import_sites = value

    def set_merge_fixed_joints(self, value: bool) -> None:
        self.merge_fixed_joints = value

    def set_convex_decomp(self, value: bool) -> None:
        self.convex_decomp = value

    def set_self_collision(self, value: bool) -> None:
        self.self_collision = value


class FakeURDFImportConfig:
    pass


def test_configure_mjcf_import_sets_self_collision(monkeypatch) -> None:
    commands, created_config = _install_fake_omni_commands(monkeypatch, "mjcf")

    imported_path = configure_mjcf_import(
        Path("robot.xml"),
        "/World/Robot",
        asset_import_config=AssetImportConfig(self_collision=True),
    )

    assert imported_path == "/World/Robot"
    assert commands.calls[0][0] == "MJCFCreateImportConfig"
    assert commands.calls[1][0] == "MJCFCreateAsset"
    assert commands.calls[1][1]["import_config"] is created_config
    assert commands.calls[1][1]["prim_path"] == "/Robot"
    assert str(commands.calls[1][1]["dest_path"]).endswith("/robot.usd")
    assert commands.references == [
        (
            Path(str(commands.calls[1][1]["dest_path"])),
            "/Robot",
            "/World/Robot",
        )
    ]
    assert created_config.self_collision is True
    assert created_config.convex_decomp is True
    assert created_config.import_sites is True
    assert created_config.merge_fixed_joints is False


def test_configure_mjcf_import_defaults_self_collision_to_false(monkeypatch) -> None:
    _, created_config = _install_fake_omni_commands(monkeypatch, "mjcf")

    configure_mjcf_import(Path("robot.xml"), "/World/Robot")

    assert created_config.self_collision is False


def test_configure_mjcf_import_applies_named_importer_settings(monkeypatch) -> None:
    _, created_config = _install_fake_omni_commands(monkeypatch, "mjcf")

    configure_mjcf_import(
        Path("robot.xml"),
        "/World/Robot",
        asset_import_config=AssetImportConfig(
            fix_base=False,
            import_inertia_tensor=False,
            import_sites=False,
            merge_fixed_joints=True,
        ),
    )

    assert created_config.fix_base is False
    assert created_config.import_inertia_tensor is False
    assert created_config.import_sites is False
    assert created_config.merge_fixed_joints is True


def test_configure_urdf_import_sets_self_collision(monkeypatch) -> None:
    commands, created_config = _install_fake_omni_commands(monkeypatch, "urdf")
    _install_fake_urdf_joint_target_type(monkeypatch)

    imported_path = configure_urdf_import(
        Path("robot.urdf"),
        asset_import_config=AssetImportConfig(
            collision_approximation="convex_hull",
            self_collision=True,
        ),
    )

    assert imported_path == "/World/ImportedRobot"
    assert commands.calls[0][0] == "URDFCreateImportConfig"
    assert commands.calls[1][0] == "URDFParseAndImportFile"
    assert commands.calls[1][1]["import_config"] is created_config
    assert created_config.self_collision is True
    assert created_config.convex_decomp is False


def test_configure_urdf_import_defaults_self_collision_to_false(monkeypatch) -> None:
    commands, created_config = _install_fake_omni_commands(monkeypatch, "urdf")
    _install_fake_urdf_joint_target_type(monkeypatch)

    imported_path = configure_urdf_import(
        Path("robot.urdf"),
        get_articulation_root=False,
        make_default_prim=False,
    )

    assert created_config.self_collision is False
    assert created_config.make_default_prim is False
    assert imported_path == "/World/ImportedRobot"
    assert commands.calls[1][1]["get_articulation_root"] is False
    assert created_config.parse_mimic is True
    assert created_config.merge_fixed_joints is True


def test_configure_urdf_import_applies_named_importer_settings(monkeypatch) -> None:
    _, created_config = _install_fake_omni_commands(monkeypatch, "urdf")
    _install_fake_urdf_joint_target_type(monkeypatch)

    configure_urdf_import(
        Path("robot.urdf"),
        asset_import_config=AssetImportConfig(
            fix_base=False,
            merge_fixed_joints=False,
            collision_from_visuals=True,
            import_inertia_tensor=False,
        ),
    )

    assert created_config.fix_base is False
    assert created_config.merge_fixed_joints is False
    assert created_config.collision_from_visuals is True
    assert created_config.import_inertia_tensor is False


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
        robot_import.import_robot_asset(config)
    )

    assert received == {
        "urdf_path": asset_path.resolve(),
        "prim_path": resolved_prim_path,
    }
    assert articulation_path == resolved_prim_path
    assert imported_root_path == resolved_prim_path
    assert imported_asset_path == asset_path.resolve()


def test_configure_urdf_import_maps_base_before_resolving_articulation(
    monkeypatch,
) -> None:
    commands, _ = _install_fake_omni_commands(monkeypatch, "urdf")
    _install_fake_urdf_joint_target_type(monkeypatch)
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
    assert commands.references == [
        (
            Path(str(commands.calls[1][1]["dest_path"])),
            "/World/ImportedRobot",
            "/World/Robots/robot_a",
        )
    ]
    assert commands.import_config.make_default_prim is False
    assert commands.calls[1][1]["get_articulation_root"] is False


def test_file_backed_import_reference_remaps_internal_targets(tmp_path: Path) -> None:
    from pxr import Usd

    source_path = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = source_stage.DefinePrim("/Source", "Xform")
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
    )

    target = destination_stage.GetPrimAtPath("/World/Target")
    assert target.IsValid()
    assert [str(path) for path in target.GetRelationship("child").GetTargets()] == [
        "/World/Target/Child"
    ]


class FakeCommands:
    def __init__(self, mode: str, import_config) -> None:
        self.mode = mode
        self.import_config = import_config
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.references: list[tuple[object, str, str]] = []

    def execute(self, command_name: str, **kwargs):
        self.calls.append((command_name, kwargs))
        if command_name in {"MJCFCreateImportConfig", "URDFCreateImportConfig"}:
            return True, self.import_config
        if command_name == "MJCFCreateAsset":
            return True, kwargs["prim_path"]
        if command_name == "URDFParseAndImportFile":
            return True, "/World/ImportedRobot"
        raise AssertionError(f"Unexpected command for {self.mode}: {command_name}")


def _install_fake_omni_commands(monkeypatch, mode: str):
    created_config = (
        FakeMJCFImportConfig() if mode == "mjcf" else FakeURDFImportConfig()
    )
    commands = FakeCommands(mode, created_config)
    omni_module = types.ModuleType("omni")
    kit_module = types.ModuleType("omni.kit")
    commands_module = types.ModuleType("omni.kit.commands")
    commands_module.execute = commands.execute
    kit_module.commands = commands_module
    omni_module.kit = kit_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
    monkeypatch.setitem(sys.modules, "omni.kit.commands", commands_module)
    monkeypatch.setattr(
        robot_import,
        "_reference_imported_prim_from_usd",
        lambda source_usd_path, *, source_path, target_path: commands.references.append(
            (source_usd_path, source_path, target_path)
        ),
    )
    monkeypatch.setattr(
        robot_import,
        "find_articulation_root",
        lambda root, **_kwargs: root,
    )
    return commands, created_config


def _install_fake_urdf_joint_target_type(monkeypatch) -> None:
    isaacsim_module = types.ModuleType("isaacsim")
    asset_module = types.ModuleType("isaacsim.asset")
    importer_module = types.ModuleType("isaacsim.asset.importer")
    urdf_package_module = types.ModuleType("isaacsim.asset.importer.urdf")
    urdf_module = types.ModuleType("isaacsim.asset.importer.urdf._urdf")

    class FakeUrdfJointTargetType:
        JOINT_DRIVE_NONE = "none"
        JOINT_DRIVE_POSITION = "position"

    urdf_module.UrdfJointTargetType = FakeUrdfJointTargetType
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.asset", asset_module)
    monkeypatch.setitem(sys.modules, "isaacsim.asset.importer", importer_module)
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.asset.importer.urdf",
        urdf_package_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.asset.importer.urdf._urdf",
        urdf_module,
    )
