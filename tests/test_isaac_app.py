from __future__ import annotations

from dataclasses import replace
import importlib
import inspect
from pathlib import Path
import sys
import tomllib
from types import ModuleType, SimpleNamespace

import pytest

from linkerbot_sim.isaac import app as app_module
from linkerbot_sim.isaac.spec import (
    IsaacAppSpec,
    IsaacComputeSpec,
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacPhysxCpuSpec,
    IsaacPhysxCudaSpec,
    IsaacRenderSpec,
    IsaacSessionSpec,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeSimulationApp:
    configs: list[dict[str, object]] = []
    experiences: list[str] = []
    closed: list["FakeSimulationApp"] = []

    def __init__(self, config: dict[str, object], *, experience: str) -> None:
        type(self).configs.append(config)
        type(self).experiences.append(experience)

    def close(self) -> None:
        type(self).closed.append(self)


class FakeSimulationManager:
    active_engine = "physx"
    device = "cpu"
    requested_devices: list[str] = []

    @classmethod
    def get_active_physics_engine(cls) -> str:
        return cls.active_engine

    @classmethod
    def set_device(cls, device: str) -> None:
        cls.requested_devices.append(device)
        cls.device = device

    @classmethod
    def get_physics_sim_device(cls) -> str:
        return cls.device


def _fields() -> dict[str, object]:
    return {
        "physics_dt": 1.0 / 240.0,
        "rendering_dt": 1.0 / 60.0,
        "gravity_z": -9.81,
    }


def _mirror_physx(*, device: int = 0, **kwargs: object) -> IsaacSessionSpec:
    return IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=device),
        physics=IsaacPhysxCpuSpec(),
        **_fields(),
        **kwargs,
    )


def _kaleidoscope(device: int = 0, *, rendering: bool = False) -> IsaacSessionSpec:
    return IsaacSessionSpec(
        experience_family="kaleidoscope",
        compute=IsaacComputeSpec(cuda_device=device),
        physics=IsaacPhysxCudaSpec(),
        app=IsaacAppSpec(gui=rendering),
        render=IsaacRenderSpec(
            enabled=rendering,
            visible_world_indices=((0,) if rendering else None),
        ),
        **_fields(),
    )


def _kaleidoscope_newton(
    device: int = 0, *, rendering: bool = False
) -> IsaacSessionSpec:
    return IsaacSessionSpec(
        experience_family="kaleidoscope",
        compute=IsaacComputeSpec(cuda_device=device),
        physics=IsaacNewtonCudaSpec(world_count=8),
        app=IsaacAppSpec(gui=rendering),
        render=IsaacRenderSpec(
            enabled=rendering,
            visible_world_indices=((3,) if rendering else None),
        ),
        **_fields(),
    )


def _mirror_newton(
    *,
    device: int = 1,
    rendering: bool = False,
    execution: str = "cuda",
) -> IsaacSessionSpec:
    if execution not in {"cpu", "cuda"}:
        raise ValueError("execution must be cpu or cuda")
    return IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=device),
        physics=(IsaacNewtonCpuSpec() if execution == "cpu" else IsaacNewtonCudaSpec()),
        render=IsaacRenderSpec(enabled=rendering),
        **_fields(),
    )


def _install_fake_isaacsim(monkeypatch) -> None:
    FakeSimulationApp.configs.clear()
    FakeSimulationApp.experiences.clear()
    FakeSimulationApp.closed.clear()
    FakeSimulationManager.active_engine = "physx"
    FakeSimulationManager.device = "cpu"
    FakeSimulationManager.requested_devices.clear()
    simulation_app = ModuleType("isaacsim.simulation_app")
    simulation_app.SimulationApp = FakeSimulationApp
    simulation_manager = ModuleType("isaacsim.core.simulation_manager")
    simulation_manager.SimulationManager = FakeSimulationManager
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(sys.modules, "isaacsim.simulation_app", simulation_app)
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.core.simulation_manager",
        simulation_manager,
    )


def _disable_provenance(monkeypatch) -> None:
    provenance = importlib.import_module("linkerbot_sim.isaac.provenance")
    monkeypatch.setattr(
        provenance,
        "collect_runtime_provenance",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        provenance,
        "validate_target_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(provenance, "format_runtime_provenance", lambda _value: "{}")


def test_importing_app_does_not_import_isaacsim(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "isaacsim", raising=False)

    importlib.reload(app_module)

    assert "isaacsim" not in sys.modules


@pytest.mark.parametrize(
    ("spec", "filename"),
    (
        (_mirror_physx(), "linkerbot_sim.mirror.physx.python.kit"),
        (
            _mirror_newton(execution="cpu"),
            "linkerbot_sim.mirror.newton.python.kit",
        ),
        (
            _mirror_newton(),
            "linkerbot_sim.mirror.newton.python.kit",
        ),
        (
            _mirror_newton(rendering=True),
            "linkerbot_sim.mirror.newton_render.python.kit",
        ),
        (
            _kaleidoscope(),
            "linkerbot_sim.kaleidoscope.physx_cuda.python.kit",
        ),
        (
            _kaleidoscope_newton(),
            "linkerbot_sim.kaleidoscope.newton.python.kit",
        ),
        (
            _kaleidoscope(rendering=True),
            "linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit",
        ),
        (
            _kaleidoscope_newton(rendering=True),
            "linkerbot_sim.kaleidoscope.newton_viewport.python.kit",
        ),
    ),
)
def test_experience_selector_maps_strict_specs_to_seven_formal_kits(
    spec: IsaacSessionSpec,
    filename: str,
) -> None:
    path = app_module._experience_path(spec)

    assert path.name == filename
    assert path.is_file()


def test_apps_directory_contains_exactly_seven_formal_kits() -> None:
    assert {path.name for path in (ROOT / "apps").glob("*.kit")} == {
        "linkerbot_sim.mirror.physx.python.kit",
        "linkerbot_sim.mirror.newton.python.kit",
        "linkerbot_sim.mirror.newton_render.python.kit",
        "linkerbot_sim.kaleidoscope.physx_cuda.python.kit",
        "linkerbot_sim.kaleidoscope.newton.python.kit",
        "linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit",
        "linkerbot_sim.kaleidoscope.newton_viewport.python.kit",
    }


def test_newton_runtime_kits_are_self_contained_and_exclude_physics_owners() -> None:
    required_base = {
        "isaacsim.simulation_app",
        "isaacsim.asset.importer.mjcf",
        "isaacsim.asset.importer.urdf",
        "omni.kit.loop-isaac",
        "omni.kit.usd.layers",
        "omni.warp.core",
    }
    forbidden = {
        "isaacsim.core.api",
        "isaacsim.core.cloner",
        "isaacsim.core.simulation_manager",
        "isaacsim.physics.newton",
        "omni.physics.physx",
        "omni.physics.stageupdate",
        "omni.physx.fabric",
    }
    for spec in (
        _mirror_newton(execution="cpu"),
        _mirror_newton(),
        _mirror_newton(rendering=True),
        _kaleidoscope_newton(),
        _kaleidoscope_newton(rendering=True),
    ):
        with app_module._experience_path(spec).open("rb") as stream:
            kit = tomllib.load(stream)
        dependencies = set(kit["dependencies"])
        assert required_base <= dependencies
        assert not any(name.startswith("linkerbot_sim.") for name in dependencies)
        assert forbidden.isdisjoint(dependencies)
        assert forbidden <= set(kit["settings"]["app"]["extensions"]["excluded"])


def test_kaleidoscope_kit_has_no_renderer_or_newton_dependency() -> None:
    with app_module._experience_path(_kaleidoscope()).open("rb") as stream:
        kit = tomllib.load(stream)

    dependencies = set(kit["dependencies"])
    assert "isaacsim.core.api" in dependencies
    assert "omni.physics.physx" in dependencies
    assert "omni.physx.fabric" in dependencies
    assert not any("newton" in name for name in dependencies)
    assert not any(
        name.startswith(("omni.hydra", "omni.kit.viewport", "omni.syntheticdata"))
        for name in dependencies
    )


@pytest.mark.parametrize(
    "spec",
    (_kaleidoscope(rendering=True), _kaleidoscope_newton(rendering=True)),
)
def test_kaleidoscope_viewport_kits_are_camera_free_debug_closures(
    spec: IsaacSessionSpec,
) -> None:
    with app_module._experience_path(spec).open("rb") as stream:
        kit = tomllib.load(stream)

    dependencies = set(kit["dependencies"])
    excluded = set(kit["settings"]["app"]["extensions"]["excluded"])
    assert {
        "omni.hydra.rtx",
        "omni.kit.viewport.utility",
        "omni.kit.viewport.window",
    } <= dependencies
    assert "isaacsim.sensors.camera" in excluded
    assert "omni.syntheticdata" in excluded
    assert "omni.replicator.core" in excluded
    assert not any(
        name.startswith(
            ("isaacsim.sensors.camera", "omni.syntheticdata", "omni.replicator")
        )
        for name in dependencies
    )


def test_kit_config_derives_one_device_and_render_settings_from_spec() -> None:
    spec = IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=3),
        physics=IsaacNewtonCudaSpec(),
        app=IsaacAppSpec(
            hide_ui=True,
            disable_viewport_updates=False,
            fast_shutdown=False,
            material_sync_loads=True,
        ),
        render=IsaacRenderSpec(
            enabled=True,
            width=960,
            height=540,
            window_width=1200,
            window_height=700,
            renderer="PathTracing",
            anti_aliasing=2,
            samples_per_pixel_per_frame=4,
            denoiser=True,
        ),
        **_fields(),
    )

    config = app_module._kit_config(spec)

    assert config["active_gpu"] == config["physics_gpu"] == 3
    assert config["max_gpu_count"] == 4
    assert config["multi_gpu"] is False
    assert (config["width"], config["height"]) == (960, 540)
    assert (config["window_width"], config["window_height"]) == (1200, 700)
    assert config["renderer"] == "PathTracing"
    assert config["anti_aliasing"] == 2
    assert config["samples_per_pixel_per_frame"] == 4
    assert config["denoiser"] is True
    disable_default_viewport = (
        "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true"
    )
    assert disable_default_viewport in config["extra_args"]

    gui_config = app_module._kit_config(replace(spec, app=replace(spec.app, gui=True)))
    assert disable_default_viewport not in gui_config["extra_args"]


def test_mirror_physx_cpu_keeps_root_cuda_device_for_kit_and_rtx() -> None:
    spec = _mirror_physx(device=3)

    config = app_module._kit_config(spec)

    assert spec.compute_device == "cuda:3"
    assert spec.physics_device == "cpu"
    assert config["active_gpu"] == 3
    assert config["physics_gpu"] == 3
    assert config["max_gpu_count"] == 4


def test_backend_validation_sets_resolved_cpu_and_cuda_physics_devices(
    monkeypatch,
) -> None:
    _install_fake_isaacsim(monkeypatch)

    app_module._configure_and_validate_physics_backend(_mirror_physx())
    app_module._configure_and_validate_physics_backend(_kaleidoscope(device=2))

    assert FakeSimulationManager.requested_devices == ["cpu", "cuda:2"]


def test_backend_validation_rejects_active_engine_mismatch(monkeypatch) -> None:
    _install_fake_isaacsim(monkeypatch)
    FakeSimulationManager.active_engine = "newton"

    with pytest.raises(RuntimeError, match="backend mismatch"):
        app_module._configure_and_validate_physics_backend(_mirror_physx())


@pytest.mark.parametrize("execution", ("cpu", "cuda"))
def test_newton_backend_validation_audits_then_registers(
    monkeypatch,
    execution: str,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        app_module,
        "validate_newton_exclusivity",
        lambda **kwargs: calls.append(("audit", kwargs)),
    )
    monkeypatch.setattr(
        app_module,
        "set_runtime_physics_backend",
        lambda backend, **kwargs: calls.append((backend, kwargs)),
    )

    app_module._configure_and_validate_physics_backend(
        _mirror_newton(execution=execution)
    )

    assert calls == [
        ("audit", {"phase": "startup"}),
        ("newton", {"execution": execution}),
    ]


def test_launch_requires_explicit_eula_before_importing_isaac(monkeypatch) -> None:
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.delitem(sys.modules, "isaacsim", raising=False)

    with pytest.raises(RuntimeError, match="EULA"):
        app_module.launch_simulation_app(_mirror_physx())

    assert "isaacsim" not in sys.modules


def test_newton_cpu_launch_derives_backend_override_and_provenance_execution(
    monkeypatch,
) -> None:
    _install_fake_isaacsim(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("LINKERBOT_RUNTIME_PROVENANCE", "0")
    backend_calls: list[tuple[str, dict[str, object]]] = []
    provenance_calls: list[dict[str, object]] = []
    validation_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        app_module,
        "validate_newton_exclusivity",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        app_module,
        "set_runtime_physics_backend",
        lambda backend, **kwargs: backend_calls.append((backend, dict(kwargs))),
    )
    provenance = importlib.import_module("linkerbot_sim.isaac.provenance")
    marker = SimpleNamespace()
    monkeypatch.setattr(
        provenance,
        "collect_runtime_provenance",
        lambda **kwargs: provenance_calls.append(dict(kwargs)) or marker,
    )
    monkeypatch.setattr(
        provenance,
        "validate_target_runtime",
        lambda _value, **kwargs: validation_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(provenance, "format_runtime_provenance", lambda _value: "{}")

    app_module.launch_simulation_app(_mirror_newton(execution="cpu", device=3))

    assert backend_calls == [("newton", {"execution": "cpu"})]
    assert provenance_calls == [
        {
            "cuda_device": 3,
            "include_curobo": False,
            "physics_execution": "cpu",
        }
    ]
    assert validation_calls == [
        {
            "expected_physics_backend": "newton",
            "physics_execution": "cpu",
            "experience_family": "mirror",
            "rendering_required": False,
        }
    ]


@pytest.mark.parametrize(
    ("spec", "physics_only", "viewport_updates_disabled"),
    (
        (_mirror_physx(), False, True),
        (_kaleidoscope(), True, True),
        (_kaleidoscope_newton(), True, True),
        (_mirror_newton(execution="cpu"), True, True),
        (_mirror_newton(), True, True),
        (_mirror_newton(rendering=True), False, False),
        (_kaleidoscope(rendering=True), False, False),
        (_kaleidoscope_newton(rendering=True), False, False),
    ),
)
def test_launch_uses_expected_app_class_and_forwards_exact_kit(
    monkeypatch,
    spec: IsaacSessionSpec,
    physics_only: bool,
    viewport_updates_disabled: bool,
) -> None:
    _install_fake_isaacsim(monkeypatch)
    _disable_provenance(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("LINKERBOT_RUNTIME_PROVENANCE", "0")
    monkeypatch.setattr(
        app_module,
        "_configure_and_validate_physics_backend",
        lambda _spec: None,
    )

    result = app_module.launch_simulation_app(spec)

    assert (type(result).__name__ == "PhysicsOnlySimulationApp") is physics_only
    assert (
        FakeSimulationApp.configs[-1]["disable_viewport_updates"]
        is viewport_updates_disabled
    )
    assert FakeSimulationApp.experiences == [str(app_module._experience_path(spec))]
    assert tuple(inspect.signature(app_module.launch_simulation_app).parameters) == (
        "spec",
    )


def test_launch_validation_failure_closes_app_and_preserves_primary_error(
    monkeypatch,
) -> None:
    _install_fake_isaacsim(monkeypatch)
    _disable_provenance(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "Y")
    monkeypatch.setattr(
        app_module,
        "_configure_and_validate_physics_backend",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("backend rejected")),
    )

    with pytest.raises(RuntimeError, match="backend rejected"):
        app_module.launch_simulation_app(_mirror_physx())

    assert len(FakeSimulationApp.closed) == 1


def test_newton_launch_does_not_release_backend_when_registration_failed(
    monkeypatch,
) -> None:
    _install_fake_isaacsim(monkeypatch)
    _disable_provenance(monkeypatch)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "Y")
    clears: list[object] = []
    monkeypatch.setattr(
        app_module,
        "_configure_and_validate_physics_backend",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("registration rejected")),
    )
    monkeypatch.setattr(
        app_module,
        "clear_runtime_physics_backend",
        lambda **kwargs: clears.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="registration rejected"):
        app_module.launch_simulation_app(_mirror_newton(execution="cpu"))

    assert clears == []
