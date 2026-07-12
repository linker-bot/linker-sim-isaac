from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from linkerbot_sim.app.runtime import simulation_session
from linkerbot_sim.envs.settings import EnvRuntimeSettings
from linkerbot_sim.configs.runtime import (
    SimulationAppSettings,
    SimulationRenderSettings,
)


class FakeSingleArticulation:
    pass


class FakeArticulationAction:
    pass


def _install_runtime_modules(monkeypatch, *, stage: object) -> None:
    isaacsim_module = ModuleType("isaacsim")
    core_module = ModuleType("isaacsim.core")
    prims_module = ModuleType("isaacsim.core.prims")
    utils_module = ModuleType("isaacsim.core.utils")
    types_module = ModuleType("isaacsim.core.utils.types")
    prims_module.SingleArticulation = FakeSingleArticulation
    types_module.ArticulationAction = FakeArticulationAction

    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.types", types_module)

    usd_module = ModuleType("omni.usd")
    usd_module.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni_module = ModuleType("omni")
    omni_module.usd = usd_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)


def _settings(
    *, viewport_enabled: bool = True, camera_output_enabled: bool = False
) -> EnvRuntimeSettings:
    config: dict[str, object] = {
        "env": {
            "physics_frequency": 300.0,
            "render_frequency": 60.0,
            "gravity_z": -3.5,
            "add_ground": False,
            "ground_height": 0.12,
        },
        "visuals": {
            "viewport": {"enabled": viewport_enabled},
            "lights": {
                "key": {"enabled": True, "intensity": 875.0},
                "fill": {"enabled": False},
            },
        },
    }
    if camera_output_enabled:
        config["sensors"] = {
            "cameras": {
                "world_rgb": {
                    "enabled": True,
                    "prim_path": "/World/WorldRGB",
                    "output": {"save_dir": "logs/test-camera"},
                }
            }
        }
    return EnvRuntimeSettings.from_env_config(config)


def _create_fake_session(
    monkeypatch,
    *,
    gui: bool,
    settings: EnvRuntimeSettings,
    headless_dt_policy: str = "camera_aware",
):
    stage = object()
    app = object()
    world = object()
    app_settings = SimulationAppSettings(
        gui=gui,
        render=SimulationRenderSettings(
            headless_dt_policy=headless_dt_policy,
        ),
    )
    launches: list[SimulationAppSettings] = []
    world_kwargs: list[dict[str, object]] = []
    configured_visuals: list[tuple[object, bool]] = []
    _install_runtime_modules(monkeypatch, stage=stage)

    monkeypatch.setattr(
        simulation_session,
        "launch_simulation_app",
        lambda settings: launches.append(settings) or app,
    )
    monkeypatch.setattr(
        simulation_session,
        "build_world",
        lambda **kwargs: world_kwargs.append(kwargs) or world,
    )

    def fake_configure_visuals(
        visuals: object, *, configure_viewport: bool = True
    ) -> None:
        configured_visuals.append((visuals, configure_viewport))

    monkeypatch.setattr(simulation_session, "configure_visuals", fake_configure_visuals)

    session = simulation_session.create_simulation_session(
        simulation_app=app_settings,
        settings=settings,
    )
    return SimpleNamespace(
        session=session,
        stage=stage,
        app=app,
        world=world,
        launches=launches,
        world_kwargs=world_kwargs,
        configured_visuals=configured_visuals,
        app_settings=app_settings,
    )


def test_gui_session_uses_configured_world_and_visual_settings(monkeypatch) -> None:
    settings = _settings()

    result = _create_fake_session(monkeypatch, gui=True, settings=settings)

    assert result.launches == [result.app_settings]
    assert result.world_kwargs == [
        {
            "physics_dt": 1.0 / 300.0,
            "rendering_dt": 1.0 / 60.0,
            "gravity_z": -3.5,
            "add_ground": False,
            "ground_height": 0.12,
        }
    ]
    assert result.configured_visuals == [(settings.visuals, True)]
    assert result.session.app is result.app
    assert result.session.world is result.world
    assert result.session.stage is result.stage
    assert result.session.articulation_action_type is FakeArticulationAction
    assert result.session.single_articulation_type is FakeSingleArticulation


def test_headless_session_preserves_configured_render_frequency(monkeypatch) -> None:
    settings = _settings(camera_output_enabled=True)

    result = _create_fake_session(monkeypatch, gui=False, settings=settings)

    assert result.world_kwargs[0]["rendering_dt"] == 1.0 / 60.0


def test_headless_physics_dt_policy_overrides_camera_render_frequency(
    monkeypatch,
) -> None:
    settings = _settings(camera_output_enabled=True)

    result = _create_fake_session(
        monkeypatch,
        gui=False,
        settings=settings,
        headless_dt_policy="physics",
    )

    assert result.world_kwargs[0]["rendering_dt"] == 1.0 / 300.0


def test_headless_session_configures_lights_when_viewport_is_disabled(
    monkeypatch,
) -> None:
    settings = _settings(viewport_enabled=False, camera_output_enabled=True)

    result = _create_fake_session(monkeypatch, gui=False, settings=settings)

    assert result.configured_visuals == [(settings.visuals, False)]


def test_headless_session_without_camera_uses_physics_cadence(monkeypatch) -> None:
    settings = _settings()

    result = _create_fake_session(monkeypatch, gui=False, settings=settings)

    assert result.world_kwargs[0]["rendering_dt"] == 1.0 / 300.0
    assert result.configured_visuals == []
