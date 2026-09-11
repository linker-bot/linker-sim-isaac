from __future__ import annotations

from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac.physics import manager as manager_module
from linkerbot_sim.isaac.physics.factory import create_physics_runtime
from linkerbot_sim.isaac.physics.newton.manager import NewtonRuntime
from linkerbot_sim.isaac.physics.physx import PhysxRuntime
from linkerbot_sim.isaac.physics.manager import (
    active_physics_manager,
    install_physics_manager,
    release_physics_manager,
)
from linkerbot_sim.isaac.physics.runtime import PhysicsCapabilities, PhysicsRuntime
from linkerbot_sim.isaac.spec import (
    IsaacComputeSpec,
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacPhysxCpuSpec,
    IsaacRenderSpec,
    IsaacSessionSpec,
)


class _RegistryRuntime:
    backend = "newton"
    kind = "newton_cuda"
    execution = "cuda"
    capabilities = PhysicsCapabilities()
    scene = object()

    def __init__(self, *, fail_close_count: int = 0) -> None:
        self.fail_close_count = fail_close_count
        self.close_calls = 0

    def reset(self) -> None:
        return None

    def forward(self) -> None:
        return None

    def step(self, *, render: bool = False) -> None:
        del render

    def render(self) -> None:
        return None

    def pre_render(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls <= self.fail_close_count:
            raise RuntimeError("close failed")

    def get_physics_dt(self) -> float:
        return 0.01

    def get_rendering_dt(self) -> float:
        return 0.02


def test_canonical_manager_exports_no_legacy_owner_or_world_facade() -> None:
    assert not hasattr(manager_module, "PhysicsManager")
    assert not hasattr(manager_module, "PhysicsWorldFacade")
    assert not hasattr(manager_module, "LegacyWorldPhysicsManager")


def test_factory_accepts_only_complete_session_spec() -> None:
    with pytest.raises(TypeError, match="spec must be IsaacSessionSpec"):
        create_physics_runtime(
            app=object(),
            stage=object(),
            spec=object(),  # type: ignore[arg-type]
            world_builder=lambda **_kwargs: pytest.fail(
                "invalid specification must fail before World creation"
            ),
            fabric_output_configurer=lambda **_kwargs: pytest.fail(
                "invalid specification must fail before Fabric configuration"
            ),
        )


def test_physx_factory_forwards_the_same_frozen_spec() -> None:
    spec = IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=0),
        physics=IsaacPhysxCpuSpec(),
        physics_dt=1.0 / 600.0,
        rendering_dt=1.0 / 100.0,
        gravity_z=-9.81,
    )
    calls: list[dict[str, object]] = []
    world = SimpleNamespace()

    runtime = create_physics_runtime(
        app=object(),
        stage=object(),
        spec=spec,
        world_builder=lambda **kwargs: calls.append(dict(kwargs)) or world,
        fabric_output_configurer=lambda **_kwargs: pytest.fail(
            "PhysX CPU must not configure Fabric outputs"
        ),
    )

    assert runtime.world is world
    assert runtime.kind == "physx_cpu"
    assert calls == [{"spec": spec, "fabric_outputs": None}]


def test_newton_factory_forwards_selected_render_world_to_runtime(monkeypatch) -> None:
    from linkerbot_sim.isaac.physics.newton import manager as runtime_module

    calls: list[dict[str, object]] = []

    class _NewtonRuntime(_RegistryRuntime):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            calls.append(dict(kwargs))

    monkeypatch.setattr(runtime_module, "NewtonRuntime", _NewtonRuntime)
    spec = IsaacSessionSpec(
        experience_family="kaleidoscope",
        compute=IsaacComputeSpec(cuda_device=2),
        physics=IsaacNewtonCudaSpec(world_count=4),
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        render=IsaacRenderSpec(enabled=True, visible_world_indices=(3,)),
    )
    app = SimpleNamespace(update=lambda: None)

    runtime = create_physics_runtime(
        app=app,
        stage=object(),
        spec=spec,
        world_builder=lambda **_kwargs: pytest.fail("Newton runtime owns no World"),
        fabric_output_configurer=lambda **_kwargs: pytest.fail(
            "Newton runtime does not configure PhysX Fabric"
        ),
    )
    try:
        assert calls[0]["rendering_enabled"] is True
        assert calls[0]["device"] == "cuda:2"
        assert calls[0]["render_callback"] == app.update
        assert calls[0]["render_world_indices"] == (3,)
    finally:
        release_physics_manager(runtime, close=False)


def test_newton_cpu_factory_forwards_cpu_device_without_fabric(monkeypatch) -> None:
    from linkerbot_sim.isaac.physics.newton import manager as runtime_module

    calls: list[dict[str, object]] = []

    class _NewtonCpuRuntime(_RegistryRuntime):
        kind = "newton_cpu"
        execution = "cpu"

        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            calls.append(dict(kwargs))

    monkeypatch.setattr(runtime_module, "NewtonRuntime", _NewtonCpuRuntime)
    spec = IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=4),
        physics=IsaacNewtonCpuSpec(),
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
    )

    runtime = create_physics_runtime(
        app=SimpleNamespace(update=lambda: None),
        stage=object(),
        spec=spec,
        world_builder=lambda **_kwargs: pytest.fail("Newton runtime owns no World"),
        fabric_output_configurer=lambda **_kwargs: pytest.fail(
            "Newton runtime does not configure PhysX Fabric"
        ),
    )
    try:
        assert calls[0]["physics_spec"] is spec.physics
        assert calls[0]["device"] == "cpu"
        assert calls[0]["render_world_indices"] is None
    finally:
        release_physics_manager(runtime, close=False)


def test_registry_retains_owner_when_close_fails_then_allows_exact_retry() -> None:
    runtime = _RegistryRuntime(fail_close_count=1)
    install_physics_manager(runtime)
    try:
        with pytest.raises(RuntimeError, match="close failed"):
            release_physics_manager(runtime, close=True)

        assert active_physics_manager() is runtime
        release_physics_manager(runtime, close=True)
        assert active_physics_manager(required=False) is None
        assert runtime.close_calls == 2
    finally:
        if active_physics_manager(required=False) is runtime:
            release_physics_manager(runtime, close=False)


def test_registry_refuses_to_release_a_different_owner() -> None:
    active = _RegistryRuntime()
    wrong = _RegistryRuntime()
    install_physics_manager(active)
    try:
        with pytest.raises(RuntimeError, match="different active physics manager"):
            release_physics_manager(wrong, close=True)
        assert active_physics_manager() is active
        assert active.close_calls == 0
        assert wrong.close_calls == 0
    finally:
        release_physics_manager(active, close=False)


def test_physx_runtime_owns_and_delegates_to_world() -> None:
    calls: list[object] = []
    world = SimpleNamespace(
        scene=object(),
        reset=lambda: calls.append("reset"),
        forward=lambda: calls.append("forward"),
        step=lambda **kwargs: calls.append(("step", kwargs)),
        render=lambda: calls.append("render"),
        get_physics_dt=lambda: 0.01,
        get_rendering_dt=lambda: 0.02,
    )
    runtime = PhysxRuntime(world, kind="physx_cuda")

    assert isinstance(runtime, PhysicsRuntime)
    assert runtime.world is world
    assert runtime.kind == "physx_cuda"
    runtime.reset()
    runtime.forward()
    runtime.step(render=True)
    runtime.render()
    assert runtime.get_physics_dt() == pytest.approx(0.01)
    assert runtime.get_rendering_dt() == pytest.approx(0.02)
    runtime.close()
    runtime.close()

    assert calls == [
        "reset",
        "forward",
        ("step", {"render": True}),
        "render",
    ]
    with pytest.raises(RuntimeError, match="closed"):
        runtime.step()


def test_physx_cuda_forward_updates_articulation_kinematics_without_step() -> None:
    calls: list[str] = []
    world = SimpleNamespace(
        scene=object(),
        physics_sim_view=SimpleNamespace(
            update_articulations_kinematic=lambda: calls.append("kinematics")
        ),
    )

    PhysxRuntime(world, kind="physx_cuda").forward()

    assert calls == ["kinematics"]


def test_newton_runtime_has_no_world_and_exposes_mirror_single_world_gate() -> None:
    # 构造函数需要真实 Isaac USD schema；窄 fixture 显式提供实例级 execution 合同。
    runtime = NewtonRuntime.__new__(NewtonRuntime)
    runtime.kind = "newton_cuda"
    runtime.execution = "cuda"
    runtime.capabilities = PhysicsCapabilities(
        supports_multiple_worlds=True,
        cuda_graph=True,
    )
    runtime.scene = object()
    runtime._num_worlds = 1

    assert isinstance(runtime, PhysicsRuntime)
    assert runtime.kind == "newton_cuda"
    assert not hasattr(runtime, "world")
    runtime.assert_single_world(consumer="Mirror")

    runtime._num_worlds = 2
    with pytest.raises(RuntimeError, match="Mirror.*exactly one world.*actual=2"):
        runtime.assert_single_world(consumer="Mirror")
