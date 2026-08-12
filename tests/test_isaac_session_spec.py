from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac import session as session_module
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


def _base_session_fields(*, cuda_device: int = 0) -> dict[str, object]:
    return {
        "compute": IsaacComputeSpec(cuda_device=cuda_device),
        "physics_dt": 1.0 / 600.0,
        "rendering_dt": 1.0 / 100.0,
        "gravity_z": -9.81,
        "add_ground": True,
        "ground_height": 0.125,
    }


def _run_fake_composition(
    monkeypatch: pytest.MonkeyPatch,
    spec: IsaacSessionSpec,
) -> SimpleNamespace:
    """在不 import Isaac/Omni 的情况下捕获新工厂的全部装配事实。"""

    app = object()
    stage = object()
    runtime = SimpleNamespace(kind=spec.physics.kind)
    app_calls: list[tuple[object, dict[str, object]]] = []
    runtime_calls: list[dict[str, object]] = []
    registrations: list[tuple[object, object]] = []

    monkeypatch.setattr(session_module, "_active_or_new_stage", lambda: stage)
    monkeypatch.setattr(
        session_module,
        "_runtime_core_types",
        lambda **_kwargs: ("action", "articulation"),
    )
    monkeypatch.setattr(
        session_module,
        "register_simulation_app_physics_runtime",
        lambda owner, physics: registrations.append((owner, physics)),
    )

    def launch(settings: object, **kwargs: object) -> object:
        app_calls.append((settings, dict(kwargs)))
        return app

    def create_runtime(**kwargs: object) -> object:
        runtime_calls.append(dict(kwargs))
        return runtime

    world_builder = object()
    fabric_configurer = object()
    result = session_module.create_isaac_session_from_spec(
        spec=spec,
        app_launcher=launch,
        physics_runtime_factory=create_runtime,  # type: ignore[arg-type]
        world_builder=world_builder,  # type: ignore[arg-type]
        fabric_output_configurer=fabric_configurer,  # type: ignore[arg-type]
    )
    return SimpleNamespace(
        app=app,
        stage=stage,
        runtime=runtime,
        session=result,
        app_calls=app_calls,
        runtime_calls=runtime_calls,
        registrations=registrations,
        world_builder=world_builder,
        fabric_configurer=fabric_configurer,
    )


def test_spec_import_is_pure_and_does_not_load_product_or_isaac_modules() -> None:
    code = """
import sys
import linkerbot_sim.isaac.spec
forbidden = (
    'linkerbot_sim.configuration',
    'linkerbot_sim.configs',
    'linkerbot_sim.mirror',
    'linkerbot_sim.kaleidoscope',
    'isaacsim',
    'omni',
    'torch',
)
loaded = sorted(
    name for name in sys.modules
    if any(name == item or name.startswith(item + '.') for item in forbidden)
)
assert loaded == [], loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_spec_rejects_unknown_family_and_cross_mode_backend_leaks() -> None:
    fields = _base_session_fields()
    with pytest.raises(ValueError, match="mirror or kaleidoscope"):
        IsaacSessionSpec(
            experience_family="obsolete",  # type: ignore[arg-type]
            physics=IsaacPhysxCpuSpec(),
            **fields,
        )
    with pytest.raises(ValueError, match="Mirror.*physx_cpu.*newton_cpu.*newton_cuda"):
        IsaacSessionSpec(
            experience_family="mirror",
            physics=IsaacPhysxCudaSpec(),
            **fields,
        )
    with pytest.raises(ValueError, match="Kaleidoscope.*physx_cuda"):
        IsaacSessionSpec(
            experience_family="kaleidoscope",
            physics=IsaacPhysxCpuSpec(),
            **fields,
        )


def test_compute_and_physics_resolve_devices_as_strict_single_facts() -> None:
    with pytest.raises(ValueError, match="compute.cuda_device"):
        IsaacComputeSpec(cuda_device=-1)
    with pytest.raises(ValueError, match="kind must be physx_cpu"):
        IsaacPhysxCpuSpec(kind="physx_cuda")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be physx_cuda"):
        IsaacPhysxCudaSpec(
            kind="physx_cpu",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="kind must be newton_cuda"):
        IsaacNewtonCudaSpec(
            kind="physx_cuda",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="kind must be newton_cpu"):
        IsaacNewtonCpuSpec(
            kind="newton_cuda",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="world_count=1"):
        IsaacNewtonCpuSpec(world_count=2)
    with pytest.raises(ValueError, match="contact_pipeline.*auto or mujoco"):
        IsaacNewtonCpuSpec(
            contact_pipeline="newton",  # type: ignore[arg-type]
        )

    cpu = IsaacPhysxCpuSpec()
    cuda = IsaacPhysxCudaSpec()
    newton = IsaacNewtonCudaSpec()
    assert not hasattr(cpu, "cuda_device")
    assert not hasattr(cuda, "cuda_device")
    assert not hasattr(newton, "cuda_device")
    spec = IsaacSessionSpec(
        experience_family="kaleidoscope",
        physics=cuda,
        **_base_session_fields(cuda_device=3),
    )
    assert spec.physics_kind == "physx_cuda"
    assert spec.compute_device == "cuda:3"
    assert spec.physics_device == "cuda:3"
    assert not hasattr(spec, "device")

    newton_spec = IsaacSessionSpec(
        experience_family="kaleidoscope",
        physics=IsaacNewtonCudaSpec(world_count=32),
        **_base_session_fields(cuda_device=4),
    )
    assert newton_spec.physics_kind == "newton_cuda"
    assert newton_spec.compute_device == "cuda:4"
    assert newton_spec.physics_device == "cuda:4"

    mirror_cpu = IsaacSessionSpec(
        experience_family="mirror",
        physics=cpu,
        **_base_session_fields(cuda_device=7),
    )
    assert mirror_cpu.compute_device == "cuda:7"
    assert mirror_cpu.physics_device == "cpu"
    assert mirror_cpu.physics_execution == "cpu"

    mirror_newton_cpu = IsaacSessionSpec(
        experience_family="mirror",
        physics=IsaacNewtonCpuSpec(),
        **_base_session_fields(cuda_device=6),
    )
    assert mirror_newton_cpu.physics_kind == "newton_cpu"
    assert mirror_newton_cpu.compute_device == "cuda:6"
    assert mirror_newton_cpu.physics_device == "cpu"
    assert mirror_newton_cpu.physics_execution == "cpu"
    assert not hasattr(mirror_newton_cpu.physics, "use_cuda_graph")


def test_spec_enforces_mode_render_query_and_world_count_boundaries() -> None:
    fields = _base_session_fields()
    with pytest.raises(ValueError, match="visible_world_indices"):
        IsaacSessionSpec(
            experience_family="kaleidoscope",
            physics=IsaacPhysxCudaSpec(),
            render=IsaacRenderSpec(enabled=True),
            **fields,
        )
    viewport = IsaacSessionSpec(
        experience_family="kaleidoscope",
        physics=IsaacPhysxCudaSpec(),
        app=IsaacAppSpec(gui=True),
        render=IsaacRenderSpec(enabled=True, visible_world_indices=(0,)),
        **fields,
    )
    assert viewport.render.visible_world_indices == (0,)
    with pytest.raises(ValueError, match="exceeds Newton world_count"):
        IsaacSessionSpec(
            experience_family="kaleidoscope",
            physics=IsaacNewtonCudaSpec(world_count=2),
            render=IsaacRenderSpec(enabled=True, visible_world_indices=(2,)),
            **fields,
        )
    with pytest.raises(ValueError, match="scene-query"):
        IsaacSessionSpec(
            experience_family="kaleidoscope",
            physics=IsaacPhysxCudaSpec(
                enable_scene_query_support=True,
            ),
            **fields,
        )
    with pytest.raises(ValueError, match="world_count=1"):
        IsaacSessionSpec(
            experience_family="mirror",
            physics=IsaacNewtonCudaSpec(world_count=2),
            **fields,
        )
    with pytest.raises(ValueError, match="gui=true requires render.enabled=true"):
        IsaacSessionSpec(
            experience_family="mirror",
            physics=IsaacPhysxCpuSpec(),
            app=IsaacAppSpec(gui=True),
            **fields,
        )


def test_kaleidoscope_spec_is_forwarded_without_legacy_settings_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = IsaacSessionSpec(
        experience_family="kaleidoscope",
        physics=IsaacPhysxCudaSpec(),
        **_base_session_fields(cuda_device=2),
    )

    result = _run_fake_composition(monkeypatch, spec)

    assert len(result.app_calls) == 1
    app_spec, launch_kwargs = result.app_calls[0]
    assert launch_kwargs == {}
    assert app_spec is spec

    call = result.runtime_calls[0]
    assert call["spec"] is spec
    assert call["app"] is result.app
    assert call["stage"] is result.stage
    assert call["world_builder"] is result.world_builder
    assert call["fabric_output_configurer"] is result.fabric_configurer
    assert spec.compute_device == "cuda:2"
    assert spec.physics_device == "cuda:2"
    assert not hasattr(spec.physics, "gpu_buffers")
    assert result.registrations == [(result.app, result.runtime)]
    assert result.session.physics_runtime is result.runtime
    assert result.session.stage is result.stage
    assert not hasattr(result.session, "world")
    assert not hasattr(result.session, "physics_manager")


def test_mirror_newton_spec_reaches_runtime_factory_without_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = IsaacSessionSpec(
        experience_family="mirror",
        physics=IsaacNewtonCudaSpec(
            nconmax_per_world=321,
            njmax_per_world=654,
            use_cuda_graph=False,
            substeps=3,
            iterations=42,
            line_search_iterations=17,
            constraint_solver="cg",
            contact_pipeline="newton",
        ),
        app=IsaacAppSpec(
            gui=False,
            hide_ui=True,
            disable_viewport_updates=False,
            fast_shutdown=False,
            material_sync_loads=True,
            hydra_material_sync_loads=True,
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
        **_base_session_fields(cuda_device=1),
    )

    result = _run_fake_composition(monkeypatch, spec)

    app_spec, launch_kwargs = result.app_calls[0]
    assert launch_kwargs == {}
    assert app_spec is spec

    solver = spec.physics
    assert spec.compute_device == "cuda:1"
    assert spec.physics_device == "cuda:1"
    assert solver.nconmax_per_world == 321
    assert solver.njmax_per_world == 654
    assert solver.use_cuda_graph is False
    assert solver.substeps == 3
    assert solver.iterations == 42
    assert solver.line_search_iterations == 17
    assert solver.constraint_solver == "cg"
    assert solver.contact_pipeline == "newton"
    call = result.runtime_calls[0]
    assert call["spec"] is spec


def test_mirror_newton_cpu_spec_reaches_runtime_factory_without_cuda_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = IsaacSessionSpec(
        experience_family="mirror",
        physics=IsaacNewtonCpuSpec(
            nconmax_per_world=123,
            njmax_per_world=456,
            substeps=2,
            iterations=31,
            line_search_iterations=7,
            constraint_solver="cg",
            contact_pipeline="mujoco",
        ),
        **_base_session_fields(cuda_device=5),
    )

    result = _run_fake_composition(monkeypatch, spec)

    assert result.app_calls == [(spec, {})]
    assert result.runtime_calls[0]["spec"] is spec
    assert spec.compute_device == "cuda:5"
    assert spec.physics_device == "cpu"
    assert spec.physics_execution == "cpu"
    assert not hasattr(spec.physics, "use_cuda_graph")


def test_spec_factory_rolls_back_launched_app_when_runtime_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = IsaacSessionSpec(
        experience_family="mirror",
        physics=IsaacPhysxCpuSpec(),
        **_base_session_fields(),
    )
    app = object()
    closes: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(session_module, "_active_or_new_stage", lambda: object())
    monkeypatch.setattr(
        session_module,
        "_runtime_core_types",
        lambda **_kwargs: (object(), object()),
    )

    def fail_runtime(**_kwargs: object) -> object:
        raise RuntimeError("fake owner creation failed")

    with pytest.raises(RuntimeError, match="fake owner creation failed"):
        session_module.create_isaac_session_from_spec(
            spec=spec,
            app_launcher=lambda *_args, **_kwargs: app,
            physics_runtime_factory=fail_runtime,  # type: ignore[arg-type]
            app_closer=lambda owner, **kwargs: closes.append((owner, dict(kwargs))),
        )

    assert closes == [(app, {"exit_code": 1})]


def test_newton_factory_failure_releases_only_its_backend_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linkerbot_sim.isaac.physics import backend as backend_module

    monkeypatch.setattr(backend_module, "_RUNTIME_OVERRIDE", None)
    monkeypatch.setattr(backend_module, "_RUNTIME_EXECUTION", None)
    monkeypatch.setattr(backend_module, "_RUNTIME_REGISTRATION_COUNT", 0)
    backend_module.set_runtime_physics_backend("newton", execution="cpu")
    # 模拟已有同 execution session 后，又成功启动了当前 App。
    backend_module.set_runtime_physics_backend("newton", execution="cpu")
    spec = IsaacSessionSpec(
        experience_family="mirror",
        physics=IsaacNewtonCpuSpec(),
        **_base_session_fields(),
    )
    monkeypatch.setattr(session_module, "_active_or_new_stage", lambda: object())
    monkeypatch.setattr(
        session_module,
        "_runtime_core_types",
        lambda **_kwargs: (object(), object()),
    )

    with pytest.raises(RuntimeError, match="fake Newton owner creation failed"):
        session_module.create_isaac_session_from_spec(
            spec=spec,
            app_launcher=lambda _spec: object(),
            physics_runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("fake Newton owner creation failed")
            ),
            app_closer=lambda _app, **_kwargs: None,
        )

    assert backend_module.active_physics_backend() == "newton"
    assert backend_module.active_physics_execution() == "cpu"
    assert backend_module._RUNTIME_REGISTRATION_COUNT == 1
    backend_module.clear_runtime_physics_backend(backend="newton")
    assert backend_module._RUNTIME_OVERRIDE is None


def test_session_close_delegates_exact_runtime_and_is_idempotent(monkeypatch) -> None:
    app = object()
    runtime = SimpleNamespace()
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        session_module,
        "close_simulation_app",
        lambda owner, **kwargs: calls.append((owner, dict(kwargs))),
    )
    session = session_module.IsaacSession(
        app=app,
        stage=object(),
        physics_runtime=runtime,  # type: ignore[arg-type]
        articulation_action_type=object(),
        single_articulation_type=object(),
    )

    session.close(exit_code=7)
    session.close(exit_code=9)

    assert calls == [(app, {"exit_code": 7, "physics_runtime": runtime})]
    assert session.is_closed is True


def test_spec_factory_preserves_primary_error_when_cleanup_also_fails(
    monkeypatch,
) -> None:
    spec = IsaacSessionSpec(
        experience_family="mirror",
        physics=IsaacPhysxCpuSpec(),
        **_base_session_fields(),
    )
    monkeypatch.setattr(session_module, "_active_or_new_stage", lambda: object())
    monkeypatch.setattr(
        session_module,
        "_runtime_core_types",
        lambda **_kwargs: (object(), object()),
    )

    with pytest.raises(RuntimeError, match="owner failed") as exc_info:
        session_module.create_isaac_session_from_spec(
            spec=spec,
            app_launcher=lambda _spec: object(),
            physics_runtime_factory=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("owner failed")
            ),
            app_closer=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("cleanup failed")
            ),
        )

    assert any("cleanup failed" in note for note in (exc_info.value.__notes__ or ()))
