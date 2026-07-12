from __future__ import annotations

import importlib
import inspect
import os
import sys
import tomllib
from types import ModuleType

import pytest

from linkerbot_sim.app import launch
from linkerbot_sim.configs.runtime import (
    SimulationAppSettings,
    SimulationGpuSettings,
    SimulationRenderSettings,
)


class FakeSimulationApp:
    configs: list[dict[str, object]] = []
    experiences: list[str] = []

    def __init__(self, config: dict[str, object], *, experience: str) -> None:
        self.configs.append(config)
        self.experiences.append(experience)


def _install_fake_isaacsim(monkeypatch) -> None:
    FakeSimulationApp.experiences.clear()
    module = ModuleType("isaacsim")
    module.SimulationApp = FakeSimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", module)


def test_importing_launch_does_not_import_isaacsim(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "isaacsim", raising=False)

    importlib.reload(launch)

    assert "isaacsim" not in sys.modules


def test_launch_uses_only_typed_named_settings(monkeypatch) -> None:
    _install_fake_isaacsim(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    FakeSimulationApp.configs.clear()
    settings = SimulationAppSettings(
        gui=False,
        gpu=SimulationGpuSettings(
            multi_gpu=True,
            max_gpu_count=3,
            active_gpu=1,
            physics_gpu=2,
        ),
        render=SimulationRenderSettings(
            gui_size=(1200, 700),
            headless_size=(320, 240),
            window_size=(1500, 950),
            renderer="PathTracing",
            anti_aliasing_gui=4,
            anti_aliasing_headless=2,
            samples_per_pixel_per_frame=8,
            denoiser=True,
            material_sync_loads=True,
            hydra_material_sync_loads=False,
        ),
    )

    result = launch.launch_simulation_app(settings)

    assert isinstance(result, FakeSimulationApp)
    assert FakeSimulationApp.experiences == [str(launch._experience_path())]
    assert FakeSimulationApp.configs == [
        {
            "headless": True,
            "hide_ui": None,
            "disable_viewport_updates": True,
            "fast_shutdown": True,
            "multi_gpu": True,
            "max_gpu_count": 3,
            "active_gpu": 1,
            "physics_gpu": 2,
            "width": 320,
            "height": 240,
            "window_width": 1500,
            "window_height": 950,
            "renderer": "PathTracing",
            "anti_aliasing": 2,
            "samples_per_pixel_per_frame": 8,
            "denoiser": True,
            "extra_args": [
                "--/app/window/hideUi=1",
                "--/rtx/materialDb/syncLoads=true",
                "--/rtx/hydra/materialSyncLoads=false",
            ],
        }
    ]
    assert tuple(inspect.signature(launch.launch_simulation_app).parameters) == (
        "settings",
    )


def test_bundled_experience_excludes_legacy_isaac_extensions() -> None:
    experience_path = launch._experience_path()

    with experience_path.open("rb") as stream:
        experience = tomllib.load(stream)

    dependencies = set(experience["dependencies"])
    assert not any(name.startswith("omni.isaac.") for name in dependencies)
    assert "omni.replicator.isaac" not in dependencies
    assert experience["settings"]["rtx"]["post"]["dlss"]["execMode"] == 3


def test_gui_mode_selects_gui_values_and_derived_flags(monkeypatch) -> None:
    _install_fake_isaacsim(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "1")
    FakeSimulationApp.configs.clear()

    launch.launch_simulation_app(SimulationAppSettings(gui=True))

    config = FakeSimulationApp.configs[0]
    assert (config["width"], config["height"]) == (1280, 720)
    assert config["anti_aliasing"] == 3
    assert config["headless"] is False
    assert config["hide_ui"] is False
    assert config["disable_viewport_updates"] is False
    assert config["fast_shutdown"] is False
    assert "--/app/window/hideUi=1" not in config["extra_args"]


@pytest.mark.parametrize(
    ("gui", "value"),
    [(False, False), (True, True)],
)
def test_nullable_render_flags_explicitly_override_mode_defaults(
    monkeypatch, gui: bool, value: bool
) -> None:
    _install_fake_isaacsim(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "Y")
    FakeSimulationApp.configs.clear()
    render = SimulationRenderSettings(
        hide_ui=value,
        disable_viewport_updates=value,
        fast_shutdown=value,
    )

    launch.launch_simulation_app(SimulationAppSettings(gui=gui, render=render))

    config = FakeSimulationApp.configs[0]
    assert config["hide_ui"] is value
    assert config["disable_viewport_updates"] is value
    assert config["fast_shutdown"] is value
    if not gui:
        assert "--/app/window/hideUi=1" not in config["extra_args"]


@pytest.mark.parametrize("value", [None, "", "N", "true", " Y "])
def test_launch_rejects_missing_empty_or_unaccepted_eula_without_mutating_env(
    monkeypatch, value: str | None
) -> None:
    _install_fake_isaacsim(monkeypatch)
    FakeSimulationApp.configs.clear()
    if value is None:
        monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    else:
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", value)
    before = value

    with pytest.raises(RuntimeError, match="OMNI_KIT_ACCEPT_EULA=Y"):
        launch.launch_simulation_app(SimulationAppSettings())

    assert FakeSimulationApp.configs == []
    assert os.environ.get("OMNI_KIT_ACCEPT_EULA") == before


@pytest.mark.parametrize("value", ["y", "Y", "yes", "YES", "1"])
def test_launch_accepts_the_values_supported_by_isaac_sim(
    monkeypatch, value: str
) -> None:
    _install_fake_isaacsim(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", value)
    FakeSimulationApp.configs.clear()

    launch.launch_simulation_app(SimulationAppSettings())

    assert len(FakeSimulationApp.configs) == 1
