from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from linkerbot_sim.configuration.scenes import (
    DistantLightSettings,
    DomeLightSettings,
    SceneVisualSettings,
)
from linkerbot_sim.isaac.spec import (
    IsaacComputeSpec,
    IsaacPhysxCpuSpec,
    IsaacSessionSpec,
)
from linkerbot_sim.isaac.world import (
    build_physx_world,
    configure_visuals,
    set_physics_gravity,
)


def _cpu_spec() -> IsaacSessionSpec:
    return IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=0),
        physics=IsaacPhysxCpuSpec(),
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        add_ground=True,
        ground_height=0.12,
    )


def test_build_physx_world_uses_spec_timing_gravity_and_ground(monkeypatch) -> None:
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
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.scene = FakeScene()
            self.physics_context = FakePhysicsContext()

        def get_physics_context(self) -> FakePhysicsContext:
            return self.physics_context

    world_module = ModuleType("isaacsim.core.api.world")
    world_module.World = FakeWorld
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.api"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "isaacsim.core.api.world", world_module)

    world = build_physx_world(spec=_cpu_spec())

    assert world.kwargs == {
        "stage_units_in_meters": 1.0,
        "physics_dt": 1.0 / 240.0,
        "rendering_dt": 1.0 / 60.0,
    }
    assert world.physics_context.gravity == -9.81
    assert world.scene.default_ground_plane_kwargs == {"z_position": 0.12}


def test_build_physx_world_rejects_non_spec_before_importing_isaac() -> None:
    with pytest.raises(TypeError, match="spec must be IsaacSessionSpec"):
        build_physx_world(spec=object())  # type: ignore[arg-type]


def test_set_physics_gravity_dispatches_by_concrete_runtime_owner() -> None:
    calls: list[float] = []
    context = SimpleNamespace(set_gravity=lambda value: calls.append(float(value)))
    physx = SimpleNamespace(
        backend="physx",
        execution="cpu",
        world=SimpleNamespace(get_physics_context=lambda: context),
    )
    newton = SimpleNamespace(
        backend="newton",
        execution="cuda",
        set_gravity=lambda value: calls.append(float(value)),
    )

    set_physics_gravity(physx, -9.81)
    set_physics_gravity(newton, -3.0)

    assert calls == [-9.81, -3.0]


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
