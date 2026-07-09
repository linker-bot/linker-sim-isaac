from __future__ import annotations

import sys
import types
from pathlib import Path

from linkerbot_sim.assets.robot_loader import (
    AssetImportConfig,
    configure_mjcf_import,
    configure_urdf_import,
)


class FakeMJCFImportConfig:
    def __init__(self) -> None:
        self.fix_base = None
        self.import_inertia_tensor = None
        self.convex_decomp = None
        self.self_collision = None

    def set_fix_base(self, value: bool) -> None:
        self.fix_base = value

    def set_import_inertia_tensor(self, value: bool) -> None:
        self.import_inertia_tensor = value

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
    assert created_config.self_collision is True
    assert created_config.convex_decomp is True


def test_configure_mjcf_import_defaults_self_collision_to_false(monkeypatch) -> None:
    _, created_config = _install_fake_omni_commands(monkeypatch, "mjcf")

    configure_mjcf_import(Path("robot.xml"), "/World/Robot")

    assert created_config.self_collision is False


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
    _, created_config = _install_fake_omni_commands(monkeypatch, "urdf")
    _install_fake_urdf_joint_target_type(monkeypatch)

    configure_urdf_import(Path("robot.urdf"))

    assert created_config.self_collision is False


class FakeCommands:
    def __init__(self, mode: str, import_config) -> None:
        self.mode = mode
        self.import_config = import_config
        self.calls: list[tuple[str, dict[str, object]]] = []

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
