from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from linkerbot_sim.envs.scene_builder import build_world, configure_visuals
from linkerbot_sim.envs.visual_settings import (
    DistantLightSettings,
    DomeLightSettings,
    SceneVisualSettings,
)


def test_build_world_passes_ground_height_to_default_ground(monkeypatch) -> None:
    """World 构建不启动 Isaac，也应能验证默认地面高度的参数传递。"""

    class FakePhysicsContext:
        def __init__(self) -> None:
            self.gravity: float | None = None

        def set_gravity(self, value: float) -> None:
            self.gravity = value

    class FakeScene:
        def __init__(self) -> None:
            self.default_ground_plane_kwargs: dict[str, float] | None = None

        def add_default_ground_plane(self, **kwargs: float) -> None:
            self.default_ground_plane_kwargs = kwargs

    class FakeWorld:
        def __init__(
            self,
            *,
            stage_units_in_meters: float,
            physics_dt: float | None,
            rendering_dt: float | None,
        ) -> None:
            self.stage_units_in_meters = stage_units_in_meters
            self.physics_dt = physics_dt
            self.rendering_dt = rendering_dt
            self.scene = FakeScene()
            self.physics_context = FakePhysicsContext()

        def get_physics_context(self) -> FakePhysicsContext:
            return self.physics_context

    world_module = ModuleType("isaacsim.core.api.world")
    setattr(world_module, "World", FakeWorld)

    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.api", ModuleType("isaacsim.core.api")
    )
    monkeypatch.setitem(sys.modules, "isaacsim.core.api.world", world_module)

    world = build_world(
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        add_ground=True,
        ground_height=0.12,
    )

    assert world.stage_units_in_meters == 1.0
    assert world.physics_dt == 1.0 / 240.0
    assert world.rendering_dt == 1.0 / 60.0
    assert world.physics_context.gravity == -9.81
    assert world.scene.default_ground_plane_kwargs == {"z_position": 0.12}


def test_configure_visuals_can_skip_viewport_import(monkeypatch) -> None:
    stage = object()
    usd_module = ModuleType("omni.usd")
    usd_module.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni_module = ModuleType("omni")
    omni_module.usd = usd_module
    pxr_module = ModuleType("pxr")
    pxr_module.Gf = object()
    pxr_module.Sdf = object()
    pxr_module.UsdGeom = object()
    pxr_module.UsdLux = object()

    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)
    monkeypatch.delitem(sys.modules, "isaacsim.core.utils.viewports", raising=False)

    configure_visuals(
        SceneVisualSettings(
            key_light=DistantLightSettings(enabled=False),
            fill_light=DomeLightSettings(enabled=False),
        ),
        configure_viewport=False,
    )

    assert "isaacsim.core.utils.viewports" not in sys.modules
