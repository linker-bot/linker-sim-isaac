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
        ("step", {"render": False}),
        "render",
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


def test_physx_cpu_publishes_paused_poses_and_exposes_real_clock(monkeypatch) -> None:
    import sys

    calls = []
    iface = SimpleNamespace(update_transformations=lambda *args: calls.append(args))
    physx = SimpleNamespace(get_physx_interface=lambda: iface)
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(physx=physx))
    monkeypatch.setitem(sys.modules, "omni.physx", physx)
    world = SimpleNamespace(
        current_time=0.5,
        current_time_step_index=60,
        render=lambda: calls.append("render"),
    )
    runtime = PhysxRuntime(world, kind="physx_cpu")
    runtime.render()
    assert calls == [(False, True, False), "render"]
    assert runtime.simulation_time == 0.5
    assert runtime.current_time_step_index == 60


@pytest.mark.parametrize("control_hz,physical_hz", [(60, 480), (120, 600), (20, 240)])
def test_physx_exact_native_steps_and_render_does_not_advance_clock(
    control_hz, physical_hz
):
    world = SimpleNamespace(current_time=0.0, current_time_step_index=0)
    renders = []

    def step(*, render):
        assert not render
        world.current_time += 1 / physical_hz
        world.current_time_step_index += 1

    world.step = step
    world.render = lambda: renders.append(world.current_time)
    world.get_physics_dt = lambda: 1 / physical_hz
    runtime = PhysxRuntime(world, kind="physx_cuda", substeps=physical_hz // control_hz)
    for _ in range(control_hz):
        runtime.step(render=True)
    assert world.current_time_step_index == physical_hz
    assert runtime.current_time_step_index == control_hz
    assert runtime.simulation_time == pytest.approx(1)
    assert runtime.get_physics_dt() == pytest.approx(1 / control_hz)
    assert len(renders) == control_hz
    runtime.render()
    assert world.current_time_step_index == physical_hz


@pytest.mark.parametrize("substeps", [0, -1, True, 1.5])
def test_physx_rejects_invalid_substeps(substeps):
    with pytest.raises(ValueError, match="substeps"):
        PhysxRuntime(object(), kind="physx_cpu", substeps=substeps)


@pytest.mark.parametrize(
    "command_hz,feedback_hz,physics_hz",
    [(60, 240, 480), (120, 240, 720), (480, 480, 960)],
)
def test_physx_feedback_and_command_steps_share_one_clock(
    command_hz, feedback_hz, physics_hz
):
    world = SimpleNamespace(current_time=0.0, current_time_step_index=0)

    def step(*, render):
        assert render is False
        world.current_time += 1 / physics_hz
        world.current_time_step_index += 1

    world.step = step
    world.get_physics_dt = lambda: 1 / physics_hz
    runtime = PhysxRuntime(world, kind="physx_cpu", substeps=physics_hz // command_hz)
    runtime.step()
    for _ in range(feedback_hz):
        runtime.step_for_duration(1 / feedback_hz)
    runtime.step()
    assert world.current_time_step_index == physics_hz + 2 * physics_hz // command_hz
    assert runtime.simulation_time == pytest.approx(1 + 2 / command_hz)
    assert runtime.get_physics_dt() == pytest.approx(1 / command_hz)
    before = world.current_time_step_index
    for duration in (False, True, 0, -1, float("nan"), 1 / (physics_hz + 1)):
        with pytest.raises(ValueError):
            runtime.step_for_duration(duration)
        assert world.current_time_step_index == before
