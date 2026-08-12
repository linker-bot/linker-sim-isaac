from __future__ import annotations

from contextlib import nullcontext
import gc
import sys
from types import ModuleType
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

import linkerbot_sim.isaac.physics.newton.manager as newton_manager
from linkerbot_sim.isaac.physics.newton.constraints import (
    ExpectedMasterFollowerConstraint,
    MasterFollowerExecutorMetadata,
    NativeMasterFollowerAudit,
    NativeMasterFollowerBinding,
    NewtonColdStateProjector,
    NewtonDeviceWorldMasks,
)
from linkerbot_sim.isaac.physics.newton.manager import NewtonRuntime
from linkerbot_sim.isaac.physics.newton.integration_state import (
    create_solver_integration_state_store,
)
from linkerbot_sim.isaac.physics.newton.replication import (
    NewtonReplicationResult,
)
from linkerbot_sim.isaac.spec import IsaacNewtonCpuSpec, IsaacNewtonCudaSpec


class _HostArray:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)

    def numpy(self) -> np.ndarray:
        return self._values.copy()


def _planar_mesh_model() -> SimpleNamespace:
    planar = SimpleNamespace(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    nonplanar = SimpleNamespace(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    return SimpleNamespace(
        shape_type=_HostArray([8, 10, 8]),
        shape_flags=_HostArray([2, 2, 0]),
        shape_collision_group=_HostArray([1, 1, 1]),
        shape_scale=_HostArray(np.ones((3, 3))),
        shape_source=(planar, nonplanar, planar),
        shape_label=("/floor", "/solid", "/visual"),
        custom_frequency_counts={},
        mujoco=None,
    )


@pytest.mark.parametrize(
    ("configured", "kinds", "expected"),
    (
        ("auto", (), "newton"),
        ("auto", ("rigid",), "newton"),
        ("auto", ("rigid", "dynamic_chain"), "cg"),
        ("cg", ("dynamic_chain",), "cg"),
        ("cg", ("rigid",), "cg"),
        ("newton", ("rigid",), "newton"),
    ),
)
def test_constraint_solver_resolution(
    configured: str,
    kinds: tuple[str, ...],
    expected: str,
) -> None:
    handles = tuple(SimpleNamespace(kind=kind) for kind in kinds)

    assert (
        newton_manager._resolve_constraint_solver(
            configured,
            object_handles=handles,
        )
        == expected
    )


def test_explicit_newton_constraint_solver_rejects_dynamic_chain() -> None:
    with pytest.raises(
        RuntimeError, match=r"constraint_solver='newton'.*dynamic_chain"
    ):
        newton_manager._resolve_constraint_solver(
            "newton",
            object_handles=(SimpleNamespace(kind="dynamic_chain"),),
        )


@pytest.mark.parametrize(
    ("configured", "labels", "expected"),
    (
        ("auto", (), "mujoco"),
        ("auto", ("/floor",), "newton"),
        ("newton", (), "newton"),
        ("newton", ("/floor",), "newton"),
        ("mujoco", (), "mujoco"),
    ),
)
def test_contact_pipeline_resolution(
    configured: str,
    labels: tuple[str, ...],
    expected: str,
) -> None:
    assert (
        newton_manager._resolve_contact_pipeline(
            configured,
            trigger_labels=labels,
            execution="cuda",
        )
        == expected
    )


def test_explicit_mujoco_contact_pipeline_rejects_planar_mesh() -> None:
    with pytest.raises(RuntimeError, match=r"contact_pipeline='mujoco'.*/floor"):
        newton_manager._resolve_contact_pipeline(
            "mujoco",
            trigger_labels=("/floor",),
            execution="cuda",
        )


@pytest.mark.parametrize("configured", ("auto", "mujoco"))
def test_cpu_contact_pipeline_always_uses_mujoco(configured: str) -> None:
    assert (
        newton_manager._resolve_contact_pipeline(
            configured,
            trigger_labels=(),
            execution="cpu",
        )
        == "mujoco"
    )


@pytest.mark.parametrize("configured", ("auto", "mujoco"))
def test_cpu_contact_pipeline_rejects_colliding_planar_mesh(
    configured: str,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"CPU cannot simulate.*planar mesh.*/floor",
    ):
        newton_manager._resolve_contact_pipeline(
            configured,
            trigger_labels=("/floor",),
            execution="cpu",
        )


def test_planar_mesh_detection_ignores_solid_and_noncolliding_meshes() -> None:
    assert newton_manager._colliding_planar_mesh_labels(_planar_mesh_model()) == (
        "/floor",
    )


def test_initialize_rechecks_extension_and_stage_owner_before_newton_import(
    monkeypatch,
) -> None:
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.stage = object()
    manager._initialized = False
    manager._require_open = lambda: None
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        newton_manager,
        "validate_newton_exclusivity",
        lambda **kwargs: calls.append(dict(kwargs)),
    )

    with pytest.raises(ValueError, match="at least one environment"):
        manager.initialize_worlds(
            env_root_paths=(),
            env_origins=np.empty((0, 3)),
            robots={},
            object_handles=(),
        )

    assert calls == [{"stage": manager.stage, "phase": "pre_finalize"}]


def test_manager_rejects_invalid_render_world_selection_before_stage_mutation() -> None:
    common = {
        "stage": object(),
        "physics_spec": IsaacNewtonCudaSpec(world_count=2),
        "device": "cuda:3",
        "physics_dt": 1.0 / 240.0,
        "rendering_dt": 1.0 / 60.0,
        "gravity_z": -9.81,
        "add_ground": False,
        "ground_height": 0.0,
    }

    invalid_device = dict(common)
    invalid_device["device"] = "cpu"
    with pytest.raises(ValueError, match=r"canonical cuda:N device"):
        NewtonRuntime(**invalid_device)
    with pytest.raises(ValueError, match="requires rendering"):
        NewtonRuntime(**common, render_world_indices=(0,))
    with pytest.raises(ValueError, match="must be unique"):
        NewtonRuntime(
            **common,
            rendering_enabled=True,
            render_callback=lambda: None,
            render_world_indices=(1, 1),
        )


def test_cpu_manager_constructor_owns_cpu_execution_without_stream_or_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[tuple[object, bool, float, bool]] = []

    def configure_stage(
        stage: object,
        *,
        add_ground: bool,
        ground_height: float,
        prepare_newton_render_topology: bool,
    ) -> None:
        configured.append(
            (
                stage,
                add_ground,
                ground_height,
                prepare_newton_render_topology,
            )
        )

    monkeypatch.setattr(
        newton_manager,
        "_configure_newton_stage",
        configure_stage,
    )
    stage = object()

    manager = NewtonRuntime(
        stage=stage,
        physics_spec=IsaacNewtonCpuSpec(),
        device="cpu",
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        add_ground=False,
        ground_height=0.0,
    )

    assert manager.execution == "cpu"
    assert manager.kind == "newton_cpu"
    assert manager.stream is None
    assert manager.cuda_graph_state == "disabled"
    assert manager.capabilities.supports_multiple_worlds is False
    assert manager.capabilities.cuda_graph is False
    assert configured == [(stage, False, 0.0, False)]
    with pytest.raises(ValueError, match="requires device='cpu'"):
        NewtonRuntime(
            stage=stage,
            physics_spec=IsaacNewtonCpuSpec(),
            device="cuda:0",
            physics_dt=1.0 / 240.0,
            rendering_dt=1.0 / 60.0,
            gravity_z=-9.81,
            add_ground=False,
            ground_height=0.0,
        )


def test_newton_render_world_root_is_canonical_when_stage_is_configured() -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()

    newton_manager._configure_newton_stage(
        stage,
        add_ground=False,
        ground_height=0.0,
        prepare_newton_render_topology=True,
    )

    xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World"))
    assert tuple(str(op.GetOpName()) for op in xform.GetOrderedXformOps()) == (
        "xformOp:transform:newtonRenderWorld",
    )
    assert xform.GetResetXformStack()


def test_cpu_manager_initializes_one_world_with_mujoco_cpu_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("warp")
    events: list[object] = []
    holder: dict[str, object] = {}

    class _State:
        joint_q = object()
        joint_qd = object()
        body_q = object()

        def assign(self, source: object) -> None:
            events.append(("assign_state", source))

    class _Model:
        world_count = 1
        articulation_world = object()

        def __init__(self, device: object) -> None:
            self.device = device

        def set_gravity(self, gravity: object) -> None:
            events.append(("gravity", gravity))

        def state(self) -> _State:
            return _State()

        def control(self) -> SimpleNamespace:
            return SimpleNamespace()

    class _Builder:
        def finalize(self, *, device: object) -> _Model:
            events.append(("finalize", str(device)))
            model = _Model(device)
            holder["model"] = model
            return model

    replication = SimpleNamespace(
        builder=_Builder(),
        prototype_builder=object(),
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0",),
        world_transforms=(object(),),
    )
    fake_newton = ModuleType("newton")
    fake_newton.eval_fk = lambda *_args: events.append("eval_fk")

    def _solver_factory(model: object, **kwargs: object) -> SimpleNamespace:
        events.append(("solver", kwargs))
        solver = SimpleNamespace(
            model=model,
            use_mujoco_cpu=kwargs["use_mujoco_cpu"],
            update_data_interval=kwargs["update_data_interval"],
        )
        holder["solver"] = solver
        return solver

    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=_solver_factory)
    monkeypatch.setitem(sys.modules, "newton", fake_newton)
    monkeypatch.setattr(
        newton_manager, "_configure_newton_stage", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        newton_manager, "validate_newton_exclusivity", lambda **_k: None
    )
    monkeypatch.setattr(
        newton_manager,
        "build_replicated_newton_builder",
        lambda *_a, **_k: replication,
    )
    monkeypatch.setattr(
        newton_manager,
        "parse_mjcf_joint_equalities",
        lambda _path: tuple(range(5)),
    )
    monkeypatch.setattr(
        newton_manager,
        "_audit_prototype_constraints",
        lambda _prototype, *, expected_relation_count: events.append(
            ("prototype_equalities", expected_relation_count)
        ),
    )
    monkeypatch.setattr(newton_manager, "_colliding_planar_mesh_labels", lambda _m: ())
    expectations = tuple(object() for _ in range(5))
    monkeypatch.setattr(
        newton_manager,
        "_asset_expectations",
        lambda **_kwargs: expectations,
    )
    monkeypatch.setattr(
        newton_manager, "_executor_metadata", lambda **_kwargs: object()
    )
    audit = SimpleNamespace(relation_count=5, relations_per_world=5)

    def _audit_constraints(
        _model: object,
        actual_expectations: object,
        **kwargs: object,
    ) -> object:
        assert actual_expectations == expectations
        assert kwargs["expected_world_count"] == 1
        assert kwargs["expected_relations_per_world"] == 5
        return audit

    monkeypatch.setattr(
        newton_manager,
        "audit_native_master_follower_constraints",
        _audit_constraints,
    )
    monkeypatch.setattr(
        newton_manager,
        "NewtonColdStateProjector",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        newton_manager,
        "NewtonDeviceWorldMasks",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        newton_manager,
        "_audit_solver_equality_mapping",
        lambda solver, actual_audit: events.append(
            ("solver_equalities", solver, actual_audit)
        ),
    )

    manager = NewtonRuntime(
        stage=object(),
        physics_spec=IsaacNewtonCpuSpec(),
        device="cpu",
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        add_ground=False,
        ground_height=0.0,
    )
    manager._require_mujoco_variants = lambda _robots: None
    manager._project_all_worlds = lambda: events.append("project_all")
    manager._initialize_solver_integration_state = lambda solver, *, device: (
        events.append(("integration", solver, str(device), manager.stream))
    )
    robot = SimpleNamespace(asset_path="robot.xml")

    manager.initialize_worlds(
        env_root_paths=("/World/envs/env_0",),
        env_origins=np.zeros((1, 3), dtype=np.float32),
        robots={"arm": robot},
        object_handles=(),
    )

    assert manager.execution == "cpu"
    assert manager.stream is None
    assert manager.world_count == 1
    assert manager.model is holder["model"]
    assert manager.solver is holder["solver"]
    assert manager.native_master_follower_audit is audit
    assert manager._contact_pipeline_kind == "mujoco"
    assert manager._collision_pipeline is None
    assert manager._initialized is True
    assert ("prototype_equalities", 5) in events
    solver_kwargs = next(
        event[1]
        for event in events
        if isinstance(event, tuple) and event[0] == "solver"
    )
    assert solver_kwargs["use_mujoco_cpu"] is True
    assert solver_kwargs["use_mujoco_contacts"] is True
    assert solver_kwargs["separate_worlds"] is False
    integration = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "integration"
    )
    assert integration[3] is None


def test_initialize_rejects_world_count_different_from_frozen_spec(monkeypatch) -> None:
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.stage = object()
    manager.physics_spec = IsaacNewtonCudaSpec(world_count=2)
    manager._initialized = False
    manager._require_open = lambda: None
    monkeypatch.setattr(
        newton_manager,
        "validate_newton_exclusivity",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match=r"actual=1, expected=2"):
        manager.initialize_worlds(
            env_root_paths=("/World/envs/env_0",),
            env_origins=np.zeros((1, 3)),
            robots={},
            object_handles=(),
        )


def test_initialize_rejects_render_selection_outside_final_world_count(
    monkeypatch,
) -> None:
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.stage = object()
    manager.physics_spec = IsaacNewtonCudaSpec(world_count=2)
    manager._render_world_indices = (2,)
    manager._initialized = False
    manager._require_open = lambda: None
    monkeypatch.setattr(
        newton_manager,
        "validate_newton_exclusivity",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match="render world index"):
        manager.initialize_worlds(
            env_root_paths=("/World/envs/env_0", "/World/envs/env_1"),
            env_origins=np.zeros((2, 3)),
            robots={},
            object_handles=(),
        )


@pytest.mark.parametrize(
    ("world_count", "separate_worlds", "contact_pipeline", "native_contacts"),
    ((1, False, "newton", False), (2, True, "mujoco", True)),
)
def test_solver_constructor_kwargs_include_effective_constraint_solver(
    world_count: int,
    separate_worlds: bool,
    contact_pipeline: str,
    native_contacts: bool,
) -> None:
    settings = IsaacNewtonCudaSpec(
        nconmax_per_world=200,
        njmax_per_world=1200,
        iterations=100,
        line_search_iterations=50,
    )

    assert newton_manager._solver_constructor_kwargs(
        settings,
        world_count=world_count,
        constraint_solver="cg",
        contact_pipeline=contact_pipeline,
    ) == {
        "separate_worlds": separate_worlds,
        "njmax": 1200,
        "nconmax": 200,
        "iterations": 100,
        "ls_iterations": 50,
        "solver": "cg",
        "use_mujoco_cpu": False,
        "use_mujoco_contacts": native_contacts,
        "update_data_interval": 1,
    }


def test_cpu_solver_constructor_kwargs_select_mujoco_cpu() -> None:
    settings = IsaacNewtonCpuSpec(
        nconmax_per_world=200,
        njmax_per_world=1200,
        iterations=100,
        line_search_iterations=50,
    )

    assert newton_manager._solver_constructor_kwargs(
        settings,
        world_count=1,
        constraint_solver="cg",
        contact_pipeline="mujoco",
    ) == {
        "separate_worlds": False,
        "njmax": 1200,
        "nconmax": 200,
        "iterations": 100,
        "ls_iterations": 50,
        "solver": "cg",
        "use_mujoco_cpu": True,
        "use_mujoco_contacts": True,
        "update_data_interval": 1,
    }


def test_manager_diagnostics_report_effective_constraint_solver() -> None:
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.physics_spec = IsaacNewtonCudaSpec()
    manager.execution = "cuda"
    manager.device = "cuda:0"
    manager._num_worlds = 1
    manager._constraint_solver = "cg"
    manager._contact_pipeline_kind = "newton"
    manager._contact_pipeline_trigger_labels = ("/floor",)
    manager.native_master_follower_audit = None
    manager._graph_state = "disabled"
    manager._graph_error = None
    manager._rendering_enabled = False
    manager._render_sync = None
    manager._solver_integration_store = create_solver_integration_state_store("cuda")

    diagnostics = manager.diagnostics()
    assert diagnostics["constraint_solver"] == "cg"
    assert diagnostics["contact_pipeline"] == "newton"
    assert diagnostics["contact_pipeline_trigger_labels"] == ["/floor"]


def _scoped_stream_recorder(monkeypatch: pytest.MonkeyPatch):
    wp = pytest.importorskip("warp")
    active_streams: list[object] = []
    scopes: list[tuple[object, bool, bool]] = []

    class _ScopedStream:
        def __init__(
            self,
            stream: object,
            sync_enter: bool = True,
            sync_exit: bool = False,
        ) -> None:
            self.stream = stream
            self.sync_enter = sync_enter
            self.sync_exit = sync_exit

        def __enter__(self) -> object:
            scopes.append((self.stream, self.sync_enter, self.sync_exit))
            active_streams.append(self.stream)
            return self.stream

        def __exit__(self, *args: object) -> None:
            del args
            assert active_streams.pop() is self.stream

    monkeypatch.setattr(wp, "ScopedStream", _ScopedStream)
    return wp, active_streams, scopes


def test_simulate_collides_once_then_reuses_contacts_for_all_substeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wp, _active_streams, scopes = _scoped_stream_recorder(monkeypatch)
    events: list[object] = []
    contacts = object()

    class _CollisionPipeline:
        def collide(self, state: object, output: object) -> None:
            events.append(("collide", state, output))

    class _Solver:
        def step(
            self,
            state_in: object,
            state_out: object,
            control: object,
            step_contacts: object,
            dt: float,
        ) -> None:
            events.append(("step", state_in, state_out, control, step_contacts, dt))

    class _State:
        def clear_forces(self) -> None:
            events.append("clear")

    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.stream = object()
    manager.solver = _Solver()
    manager.state = _State()
    manager.control = object()
    manager._collision_pipeline = _CollisionPipeline()
    manager._contacts = contacts
    manager.physics_dt = 0.03
    manager.physics_spec = SimpleNamespace(substeps=3)

    manager._simulate()

    assert scopes == [(manager.stream, False, False)]
    assert events[0] == ("collide", manager.state, contacts)
    steps = [
        event for event in events if isinstance(event, tuple) and event[0] == "step"
    ]
    assert len(steps) == 3
    assert all(
        event[4] is contacts and event[5] == pytest.approx(0.01) for event in steps
    )
    assert events.count("clear") == 3


def test_step_captures_solver_persistent_state_after_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wp, _active_streams, scopes = _scoped_stream_recorder(monkeypatch)
    events: list[str] = []
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager._require_initialized = lambda: None
    manager._flush_cold_state_updates = lambda: events.append("flush")
    manager._simulate = lambda: events.append("simulate")
    manager._capture_solver_integration_state = lambda: events.append("capture")
    manager._graph_state = "disabled"
    manager.stream = object()
    manager.physics_dt = 0.01
    manager._sim_time = 0.0
    manager._step_callbacks = []

    manager.step(render=False)

    assert events == ["flush", "simulate", "capture"]
    # owner stream scope 由 ``_simulate`` 持有；这里替换了该方法，因此 ``step`` 自身
    # 不应再创建第二层嵌套 scope。
    assert scopes == []
    assert manager._sim_time == pytest.approx(0.01)


def test_cpu_step_runs_eager_without_owner_stream_and_captures_persistent_state() -> (
    None
):
    events: list[object] = []

    class _State:
        def clear_forces(self) -> None:
            events.append("clear")

    class _Solver:
        def step(self, *args: object) -> None:
            events.append(("solver_step", args))

    class _Store:
        width = 1

        def capture(self) -> None:
            events.append("capture")

    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.execution = "cpu"
    manager.stream = None
    manager.solver = _Solver()
    manager.state = _State()
    manager.control = object()
    manager._collision_pipeline = None
    manager._contacts = None
    manager.physics_dt = 0.02
    manager.physics_spec = IsaacNewtonCpuSpec(substeps=2)
    manager._solver_integration_store = _Store()
    manager._graph_state = "disabled"
    manager._sim_time = 0.0
    manager._step_callbacks = [lambda dt: events.append(("callback", dt))]
    manager._require_initialized = lambda: None
    manager._flush_cold_state_updates = lambda: events.append("flush")

    manager.step(render=False)

    assert events[0] == "flush"
    solver_steps = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "solver_step"
    ]
    assert len(solver_steps) == 2
    assert all(event[1][3] is None for event in solver_steps)
    assert all(event[1][4] == pytest.approx(0.01) for event in solver_steps)
    assert events.count("clear") == 2
    assert events[-2:] == ["capture", ("callback", 0.02)]
    assert manager.simulation_time == pytest.approx(0.02)


def test_reset_assigns_state_and_control_on_owner_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wp, active_streams, scopes = _scoped_stream_recorder(monkeypatch)
    owner_stream = object()
    operations: list[tuple[str, object | None, object | None]] = []

    class _State:
        def assign(self, source: object) -> None:
            operations.append(
                (
                    "state",
                    source,
                    active_streams[-1] if active_streams else None,
                )
            )

    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._initialized = True
    manager.stream = owner_stream
    manager.state = _State()
    manager._initial_state = object()
    manager.control = object()
    manager._initial_control = object()
    manager._dirty_worlds = set()
    manager._num_worlds = 2
    manager._sim_time = 3.0
    manager._solver_integration_store = create_solver_integration_state_store("cuda")
    manager._flush_cold_state_updates = lambda: operations.append(
        ("flush", None, active_streams[-1] if active_streams else None)
    )

    def _copy_control(
        destination: object,
        source: object,
        *,
        stream: object | None = None,
    ) -> None:
        del destination, source
        operations.append(
            (
                "control",
                stream,
                active_streams[-1] if active_streams else None,
            )
        )

    monkeypatch.setattr(newton_manager, "_copy_control", _copy_control)

    manager.reset()

    assert scopes == [(owner_stream, False, False)]
    assert operations[:2] == [
        ("state", manager._initial_state, owner_stream),
        ("control", owner_stream, owner_stream),
    ]
    assert operations[2] == ("flush", None, None)
    assert manager._dirty_worlds == {0, 1}
    assert manager._sim_time == 0.0


def test_commit_initial_state_rebases_reset_state_and_control_on_owner_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wp, active_streams, scopes = _scoped_stream_recorder(monkeypatch)
    owner_stream = object()
    events: list[tuple[str, object | None]] = []

    class _State:
        def __init__(self, value: float) -> None:
            self.value = value

        def assign(self, source: object) -> None:
            self.value = float(source.value)
            events.append(("state", active_streams[-1] if active_streams else None))

    class _Control:
        def __init__(self, value: float) -> None:
            self.value = value

    def _copy_control(
        destination: object,
        source: object,
        *,
        stream: object | None = None,
    ) -> None:
        destination.value = float(source.value)
        events.append(("control", stream))

    monkeypatch.setattr(newton_manager, "_copy_control", _copy_control)
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._initialized = True
    manager.stream = owner_stream
    manager.state = _State(7.0)
    manager.control = _Control(8.0)
    manager._initial_state = _State(1.0)
    manager._initial_control = _Control(2.0)
    manager._sim_time = 0.0
    manager._dirty_worlds = {1}
    manager._num_worlds = 2
    manager._solver_integration_store = SimpleNamespace(
        width=0,
        commit=lambda: events.append(("integration", owner_stream)),
    )

    def _flush() -> None:
        events.append(("flush", active_streams[-1] if active_streams else None))
        manager._dirty_worlds.clear()

    manager._flush_cold_state_updates = _flush

    manager.commit_initial_state()

    assert events == [
        ("flush", None),
        ("state", owner_stream),
        ("control", owner_stream),
        ("integration", owner_stream),
    ]
    assert scopes == [(owner_stream, False, False)]
    assert manager._initial_state.value == 7.0
    assert manager._initial_control.value == 8.0

    manager.state.value = -3.0
    manager.control.value = -4.0
    events.clear()
    manager.reset()

    assert manager.state.value == 7.0
    assert manager.control.value == 8.0
    assert events == [
        ("state", owner_stream),
        ("control", owner_stream),
        ("flush", None),
    ]


def test_cpu_commit_and_reset_rebase_state_without_cuda_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class _State:
        def __init__(self, value: float) -> None:
            self.value = value

        def assign(self, source: object) -> None:
            self.value = float(source.value)
            events.append(("state", self.value))

    class _Control:
        def __init__(self, value: float) -> None:
            self.value = value

    class _Store:
        width = 1

        def commit(self) -> None:
            events.append("integration_commit")

        def reset(self, _mask: object | None = None) -> None:
            events.append("integration_reset")

    def _copy_control(
        destination: object,
        source: object,
        *,
        stream: object | None = None,
    ) -> None:
        assert stream is None
        destination.value = float(source.value)
        events.append(("control", destination.value))

    monkeypatch.setattr(newton_manager, "_copy_control", _copy_control)
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._initialized = True
    manager.execution = "cpu"
    manager.stream = None
    manager.state = _State(7.0)
    manager.control = _Control(8.0)
    manager._initial_state = _State(1.0)
    manager._initial_control = _Control(2.0)
    manager._solver_integration_store = _Store()
    manager.solver = SimpleNamespace(_step=9)
    manager._sim_time = 0.0
    manager._num_worlds = 1
    manager._dirty_worlds = set()
    manager._projection_worlds = set()

    def _flush() -> None:
        events.append("flush")
        manager._dirty_worlds.clear()
        manager._projection_worlds.clear()

    manager._flush_cold_state_updates = _flush
    manager.commit_initial_state()

    assert manager._initial_state.value == 7.0
    assert manager._initial_control.value == 8.0
    assert events == [
        "flush",
        ("state", 7.0),
        ("control", 8.0),
        "integration_commit",
    ]

    manager.state.value = -3.0
    manager.control.value = -4.0
    events.clear()
    manager.reset()

    assert manager.state.value == 7.0
    assert manager.control.value == 8.0
    assert manager.solver._step == 0
    assert manager._sim_time == 0.0
    assert events == [
        ("state", 7.0),
        ("control", 8.0),
        "integration_reset",
        "flush",
    ]


def test_commit_initial_state_rejects_nonzero_simulation_time() -> None:
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._initialized = True
    manager._sim_time = 0.25
    manager._flush_cold_state_updates = lambda: pytest.fail(
        "nonzero-time commit must fail before flushing state"
    )

    with pytest.raises(RuntimeError, match="before simulation advances"):
        manager.commit_initial_state()


def test_solver_integration_state_round_trip_restores_selected_time_and_warmstart() -> (
    None
):
    """使用真实 MJWarp Data 验证 manager 的 GPU integration state ABI。"""

    torch = pytest.importorskip("torch")
    mujoco = pytest.importorskip("mujoco")
    mujoco_warp = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MuJoCo-Warp integration state")

    mj_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint type='hinge'/>"
        "<geom type='sphere' size='0.1' mass='1'/></body></worldbody></mujoco>"
    )
    with wp.ScopedDevice("cuda:0"):
        mjw_model = mujoco_warp.put_model(mj_model)
        mjw_data = mujoco_warp.put_data(
            mj_model,
            mujoco.MjData(mj_model),
            nworld=2,
        )
    solver = SimpleNamespace(
        mj_model=mj_model,
        mjw_model=mjw_model,
        mjw_data=mjw_data,
        update_data_interval=1,
    )
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._initialized = True
    manager._num_worlds = 2
    manager.execution = "cuda"
    manager.device = "cuda:0"
    manager.stream = wp.Stream(wp.get_device("cuda:0"))
    manager.solver = solver
    manager._solver_integration_store = create_solver_integration_state_store("cuda")
    with wp.ScopedStream(manager.stream, sync_enter=False, sync_exit=False):
        manager._initialize_solver_integration_state(solver, device="cuda:0")
    assert manager.solver_integration_state_signature == 41
    assert manager.solver_integration_state_width == 1 + mj_model.na + mj_model.nv
    assert manager.solver_integration_activation_width == mj_model.na

    time = torch.from_dlpack(wp.to_dlpack(mjw_data.time))
    warmstart = torch.from_dlpack(wp.to_dlpack(mjw_data.qacc_warmstart))
    time.copy_(torch.tensor([1.25, 2.5], device="cuda"))
    warmstart.copy_(torch.tensor([[3.0], [4.0]], device="cuda"))
    torch.cuda.synchronize()
    with wp.ScopedStream(manager.stream, sync_enter=False, sync_exit=False):
        manager._capture_solver_integration_state()
    wp.synchronize_stream(manager.stream)
    saved = torch.from_dlpack(
        wp.to_dlpack(manager.borrow_solver_integration_state())
    ).clone()

    time.zero_()
    warmstart.zero_()
    active_torch = torch.tensor([False, True], device="cuda")
    active = wp.from_torch(active_torch, dtype=wp.bool)
    torch.cuda.synchronize()
    manager.set_solver_integration_state(
        manager.borrow_solver_integration_state(),
        active_world_mask=active,
    )
    wp.synchronize_stream(manager.stream)

    assert time[0].item() == pytest.approx(0.0)
    assert time[1].item() == pytest.approx(2.5)
    assert warmstart[0, 0].item() == pytest.approx(0.0)
    assert warmstart[1, 0].item() == pytest.approx(4.0)
    current = torch.from_dlpack(wp.to_dlpack(manager.borrow_solver_integration_state()))
    torch.testing.assert_close(current[1], saved[1])
    # baseline 在初始化时捕获为零，selected reset 也只影响第二个 world。
    manager.reset_solver_integration_state(active)
    wp.synchronize_stream(manager.stream)
    assert time[1].item() == pytest.approx(0.0)
    assert warmstart[1, 0].item() == pytest.approx(0.0)


def test_cpu_solver_integration_host_payload_round_trip_is_owned() -> None:
    class _Store:
        execution = "cpu"
        width = 3
        activation_width = 1
        signature = 41

        def __init__(self) -> None:
            self.values = np.asarray([[1.25, 2.5, 3.75]], dtype=np.float64)
            self.baseline = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
            self.capture_count = 0

        def capture(self) -> None:
            self.capture_count += 1

        def borrow(self) -> np.ndarray:
            return self.values

        def restore(
            self,
            values: object,
            *,
            active_world_mask: object | None = None,
        ) -> None:
            assert active_world_mask is None
            np.copyto(self.values, np.asarray(values, dtype=np.float64))

        def reset(self, active_world_mask: object | None = None) -> None:
            assert active_world_mask is None
            np.copyto(self.values, self.baseline)

    store = _Store()
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._initialized = True
    manager.execution = "cpu"
    manager.device = "cpu"
    manager.stream = None
    manager._num_worlds = 1
    manager._sim_time = 2.0
    manager._solver_integration_store = store

    payload = manager.capture_solver_integration_state_host()

    assert store.capture_count == 1
    assert payload == {
        "schema": "linkerbot.newton-solver-integration-state.v1",
        "source_execution": "cpu",
        "world_count": 1,
        "state_signature": 41,
        "state_width": 3,
        "simulation_time_s": 2.0,
        "values": [[1.25, 2.5, 3.75]],
    }
    # payload 必须拥有独立 host storage，调用方修改它不能污染 manager canonical buffer。
    payload["values"][0][0] = 9.0
    assert store.values[0, 0] == pytest.approx(1.25)

    payload["source_execution"] = "cuda"
    payload["simulation_time_s"] = 4.5
    manager.set_solver_integration_state_host(payload)

    np.testing.assert_array_equal(store.values, [[9.0, 2.5, 3.75]])
    assert manager.simulation_time == pytest.approx(4.5)

    manager.reset_solver_integration_state_host()

    np.testing.assert_array_equal(store.values, [[0.0, 0.0, 0.0]])
    assert manager.simulation_time == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"values": [[1.0, 2.0]]}, "values shape mismatch"),
        ({"state_signature": 42}, "state_signature mismatch"),
        ({"simulation_time_s": -0.5}, "simulation_time_s must be finite"),
        ({"simulation_time_s": float("inf")}, "simulation_time_s must be finite"),
        ({"simulation_time_s": True}, "simulation_time_s must be a JSON number"),
        ({"simulation_time_s": "2.0"}, "simulation_time_s must be a JSON number"),
        ({"values": [["1.0", "2.0", "3.0"]]}, "non-numeric values"),
    ),
)
def test_cpu_solver_integration_host_payload_rejects_shape_signature_and_time(
    replacement: dict[str, object],
    message: str,
) -> None:
    store = SimpleNamespace(
        width=3,
        activation_width=1,
        signature=41,
    )
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._initialized = True
    manager.execution = "cpu"
    manager._num_worlds = 1
    manager._solver_integration_store = store
    payload: dict[str, object] = {
        "schema": "linkerbot.newton-solver-integration-state.v1",
        "source_execution": "cpu",
        "world_count": 1,
        "state_signature": 41,
        "state_width": 3,
        "simulation_time_s": 2.0,
        "values": [[1.0, 2.0, 3.0]],
    }
    payload.update(replacement)

    with pytest.raises(ValueError, match=message):
        manager.validate_solver_integration_state_host(payload)


def test_model_write_notifies_solver_on_owner_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("newton.solvers")
    _wp, active_streams, scopes = _scoped_stream_recorder(monkeypatch)
    owner_stream = object()
    notifications: list[tuple[object, object | None]] = []

    class _Solver:
        def notify_model_changed(self, flag: object) -> None:
            notifications.append((flag, active_streams[-1] if active_streams else None))

    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.stream = owner_stream
    manager.solver = _Solver()
    manager.physics_spec = SimpleNamespace(use_cuda_graph=True)
    manager._graph = object()
    manager._graph_state = "captured"

    manager.on_newton_view_write(
        view=object(),
        category="model",
        field="joint_target_ke",
        world_indices=(0,),
    )

    assert scopes == [(owner_stream, False, False)]
    assert len(notifications) == 1
    assert notifications[0][1] is owner_stream
    assert manager._graph is None
    assert manager._graph_state == "pending"


def test_cpu_gain_write_notifies_solver_without_enabling_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notification_flag = object()
    solvers = ModuleType("newton.solvers")
    solvers.SolverNotifyFlags = SimpleNamespace(JOINT_DOF_PROPERTIES=notification_flag)
    newton = ModuleType("newton")
    newton.solvers = solvers
    monkeypatch.setitem(sys.modules, "newton", newton)
    monkeypatch.setitem(sys.modules, "newton.solvers", solvers)
    notifications: list[object] = []
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.execution = "cpu"
    manager.stream = None
    manager.solver = SimpleNamespace(
        notify_model_changed=lambda flag: notifications.append(flag)
    )
    manager.physics_spec = IsaacNewtonCpuSpec()
    manager._graph = None
    manager._graph_state = "disabled"

    manager.on_newton_view_write(
        view=object(),
        category="model",
        field="joint_target_ke",
        world_indices=(0,),
    )

    assert notifications == [notification_flag]
    assert manager._graph is None
    assert manager.cuda_graph_state == "disabled"


def test_cpu_gravity_write_notifies_solver_without_enabling_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notification_flag = object()
    solvers = ModuleType("newton.solvers")
    solvers.SolverNotifyFlags = SimpleNamespace(MODEL_PROPERTIES=notification_flag)
    newton = ModuleType("newton")
    newton.solvers = solvers
    monkeypatch.setitem(sys.modules, "newton", newton)
    monkeypatch.setitem(sys.modules, "newton.solvers", solvers)
    events: list[object] = []
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.execution = "cpu"
    manager.stream = None
    manager.model = SimpleNamespace(
        set_gravity=lambda gravity: events.append(("gravity", gravity))
    )
    manager.solver = SimpleNamespace(
        notify_model_changed=lambda flag: events.append(("notify", flag))
    )
    manager.physics_spec = IsaacNewtonCpuSpec()
    manager._graph = None
    manager._graph_state = "disabled"

    manager.set_gravity(-3.5)

    assert manager.gravity == (0.0, 0.0, -3.5)
    assert events == [
        ("gravity", (0.0, 0.0, -3.5)),
        ("notify", notification_flag),
    ]
    assert manager._graph is None
    assert manager.cuda_graph_state == "disabled"


def test_full_generalized_state_write_is_fk_dirty_without_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wp, _active_streams, _scopes = _scoped_stream_recorder(monkeypatch)
    newton = pytest.importorskip("newton")
    events: list[object] = []

    class _Masks:
        def world_mask(
            self,
            worlds: tuple[int, ...],
            *,
            masked_rows: tuple[tuple[object, object], ...] = (),
        ) -> object:
            result = ("world_mask", worlds, masked_rows)
            events.append(result)
            return result

        def articulation_mask(self, world_mask: object) -> object:
            result = ("articulation_mask", world_mask)
            events.append(result)
            return result

    class _Projector:
        def project(self, **_kwargs: object) -> None:
            pytest.fail("full q/qd restore must not cold-project equality followers")

    state = SimpleNamespace(joint_q=object(), joint_qd=object())
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.stream = object()
    manager.device = "cuda:0"
    manager.model = object()
    manager.state = state
    manager._projector = _Projector()
    manager._world_masks = _Masks()
    manager._dirty_worlds = set()
    manager._projection_worlds = set()
    manager._device_dirty_rows = []
    manager._device_projection_rows = []
    manager._num_worlds = 3
    monkeypatch.setattr(
        newton,
        "eval_fk",
        lambda model, q, qd, output, mask: events.append(
            ("fk", model, q, qd, output, mask)
        ),
    )

    manager.on_newton_view_write(
        view=object(),
        category="state",
        field="joint_q_full",
        world_indices=(1,),
    )
    manager.on_newton_view_write(
        view=object(),
        category="state",
        field="joint_qd_full",
        world_indices=(1,),
    )
    assert manager._dirty_worlds == {1}
    assert manager._projection_worlds == set()

    manager._flush_cold_state_updates()

    assert events[0] == ("world_mask", (1,), ())
    assert events[1][0] == "articulation_mask"
    assert events[2][0] == "fk"
    assert manager._dirty_worlds == set()
    assert manager._projection_worlds == set()


def test_command_and_body_writes_use_distinct_projection_sets() -> None:
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager._dirty_worlds = set()
    manager._projection_worlds = set()
    manager._device_dirty_rows = []
    manager._device_projection_rows = []

    manager.on_newton_view_write(
        view=object(),
        category="state",
        field="body_q",
        world_indices=(0,),
    )
    manager.on_newton_view_write(
        view=object(),
        category="state",
        field="joint_q",
        world_indices=(1,),
    )

    assert manager._dirty_worlds == {0, 1}
    assert manager._projection_worlds == {1}


def test_device_partial_reset_projection_preserves_unselected_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wp = pytest.importorskip("warp")
    newton = pytest.importorskip("newton")
    bindings = tuple(
        NativeMasterFollowerBinding(
            equality_index=world,
            world=world,
            follower_joint_index=world * 2,
            master_joint_index=world * 2 + 1,
            follower_joint_label=f"/world_{world}/follower",
            master_joint_label=f"/world_{world}/master",
            follower_q_index=world * 2,
            master_q_index=world * 2 + 1,
            follower_qd_index=world * 2,
            master_qd_index=world * 2 + 1,
            polycoef=(0.0, 2.0, 0.0, 0.0, 0.0),
            constraint_label=f"/world_{world}/equality",
        )
        for world in range(2)
    )
    audit = NativeMasterFollowerAudit(
        representation="model",
        world_count=2,
        relations_per_world=1,
        bindings=bindings,
        executor_metadata=MasterFollowerExecutorMetadata(),
    )
    state = SimpleNamespace(
        joint_q=wp.array([-10.0, 3.0, 123.25, 7.0], dtype=wp.float32, device="cpu"),
        joint_qd=wp.array([-20.0, 4.0, -456.5, 8.0], dtype=wp.float32, device="cpu"),
    )
    model = SimpleNamespace(
        articulation_world=wp.array([0, 1], dtype=wp.int32, device="cpu")
    )
    row_world = wp.array([0, 1], dtype=wp.int32, device="cpu")
    partial = wp.array([True, False], dtype=wp.bool, device="cpu")
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.stream = None
    manager.device = "cpu"
    manager.model = model
    manager.state = state
    manager._projector = NewtonColdStateProjector(audit, device="cpu")
    manager._world_masks = NewtonDeviceWorldMasks(
        world_count=2,
        articulation_world=model.articulation_world,
        device="cpu",
    )
    manager._dirty_worlds = set()
    manager._projection_worlds = set()
    manager._device_dirty_rows = [(row_world, partial, partial)]
    manager._device_projection_rows = [(row_world, partial, partial)]
    manager._num_worlds = 2
    monkeypatch.setattr(newton, "eval_fk", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wp, "ScopedStream", lambda *_args, **_kwargs: nullcontext())

    q_before = state.joint_q.numpy().copy()
    qd_before = state.joint_qd.numpy().copy()
    manager._flush_cold_state_updates()
    actual_q = state.joint_q.numpy()
    actual_qd = state.joint_qd.numpy()

    assert actual_q[0] == pytest.approx(2.0 * q_before[1])
    assert actual_qd[0] == pytest.approx(2.0 * qd_before[1])
    assert actual_q[2] == q_before[2]
    assert actual_qd[2] == qd_before[2]


def test_device_row_mask_notification_stays_on_cuda() -> None:
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for zero-copy Torch/Warp mask aliasing")
    wp.init()
    view = object()
    row_world = wp.array([0, 1], dtype=wp.int32, device="cuda:0")
    device_mask = torch.tensor([False, True], device="cuda:0", dtype=torch.bool)
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.execution = "cuda"
    manager.device = "cuda:0"
    manager._view_world_rows = {view: row_world}
    manager._dirty_worlds = set()
    manager._projection_worlds = set()
    manager._device_dirty_rows = []
    manager._device_projection_rows = []

    manager.on_newton_view_write(
        view=view,
        category="state",
        field="joint_q",
        world_indices=(0, 1),
        device_row_mask=device_mask,
    )

    assert manager._dirty_worlds == set()
    assert manager._projection_worlds == set()
    assert len(manager._device_dirty_rows) == 1
    assert len(manager._device_projection_rows) == 1
    _worlds, alias, owner = manager._device_dirty_rows[0]
    assert alias.ptr == device_mask.data_ptr()
    assert owner is device_mask


def test_control_copy_forwards_owner_stream_to_every_warp_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wp = pytest.importorskip("warp")
    owner_stream = object()
    copies: list[tuple[object, object, object | None]] = []

    def _copy(
        destination: object,
        source: object,
        *,
        stream: object | None = None,
    ) -> None:
        copies.append((destination, source, stream))

    monkeypatch.setattr(wp, "copy", _copy)
    destination = SimpleNamespace(
        joint_f=object(),
        joint_target_pos=object(),
        joint_target_vel=object(),
        joint_act=None,
    )
    source = SimpleNamespace(
        joint_f=object(),
        joint_target_pos=object(),
        joint_target_vel=object(),
        joint_act=object(),
    )

    newton_manager._copy_control(
        destination,
        source,
        stream=owner_stream,
    )

    assert len(copies) == 3
    assert all(stream is owner_stream for _destination, _source, stream in copies)


def test_close_synchronizes_releases_registered_views_and_is_idempotent() -> None:
    events: list[object] = []
    manager = NewtonRuntime.__new__(NewtonRuntime)
    owner_stream = object()
    model = object()
    state = object()
    control = object()
    solver = object()

    class _View:
        def __init__(self) -> None:
            self.manager = manager
            self.gpu_mapping = object()

        def _release_from_manager(self) -> None:
            # Manager-owned arrays and stream must remain live until every view
            # has dropped its mappings and output/staging buffers.
            assert manager.model is model
            assert manager.state is state
            assert manager.control is control
            assert manager.solver is solver
            assert manager.stream is owner_stream
            self.manager = None
            self.gpu_mapping = None
            events.append(self)

    first = _View()
    second = _View()
    manager.closed = False
    manager.stream = owner_stream
    manager.model = model
    manager.state = state
    manager.control = control
    manager.solver = solver
    manager._contacts = object()
    manager._collision_pipeline = object()
    manager._graph = object()
    manager._projector = object()
    manager._initial_control = object()
    manager._initial_state = object()
    manager.replication = object()
    manager.native_master_follower_audit = object()
    manager.constraint_audit = object()
    manager._dirty_worlds = {0}
    manager._step_callbacks = [object()]
    manager._initialized = True
    manager.scene = newton_manager._NewtonSceneRegistry()
    manager.scene.add(object())
    manager.stage = object()
    manager._registered_views = weakref.WeakSet((first, second))
    manager.execution = "cuda"
    manager._synchronize_owner_stream = lambda: events.append("synchronize")

    manager.close()
    manager.close()

    assert events[0] == "synchronize"
    assert set(events[1:]) == {first, second}
    assert first.manager is None and first.gpu_mapping is None
    assert second.manager is None and second.gpu_mapping is None
    assert manager.closed is True
    assert len(manager._registered_views) == 0
    assert manager._graph is None
    assert manager._projector is None
    assert manager._initial_control is None
    assert manager._initial_state is None
    assert manager.control is None
    assert manager.state is None
    assert manager.solver is None
    assert manager._contacts is None
    assert manager._collision_pipeline is None
    assert manager.model is None
    assert manager.stream is None
    assert manager.replication is None
    assert manager.stage is None
    assert manager.scene._items == []


def test_manager_registers_newton_views_weakly() -> None:
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager._registered_views = weakref.WeakSet()

    class _View:
        pass

    view = _View()
    reference = weakref.ref(view)
    manager.register_newton_view(view)
    assert tuple(manager._registered_views) == (view,)

    del view
    gc.collect()

    assert reference() is None
    assert len(manager._registered_views) == 0


def test_manager_leaves_external_camera_resources_to_mode_owner() -> None:
    events: list[str] = []

    class _Resource:
        def close(self) -> None:
            events.append("camera")

    class _RenderSync:
        def close(self) -> None:
            events.append("render_sync")

    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.closed = False
    manager.stream = object()
    manager.model = object()
    manager.state = object()
    manager.control = object()
    manager.solver = object()
    manager._graph = object()
    manager._projector = object()
    manager._initial_control = object()
    manager._initial_state = object()
    manager.replication = object()
    manager.native_master_follower_audit = object()
    manager.constraint_audit = object()
    manager._dirty_worlds = set()
    manager._step_callbacks = []
    manager._initialized = True
    manager.scene = newton_manager._NewtonSceneRegistry()
    manager.stage = object()
    manager._registered_views = weakref.WeakSet()
    manager._render_resources = [_Resource()]
    manager._render_sync = _RenderSync()
    manager._render_callback = object()
    manager.execution = "cuda"
    manager._synchronize_owner_stream = lambda: events.append("synchronize")

    manager.close()

    assert events == ["synchronize", "render_sync"]
    assert manager.stage is None
    assert not hasattr(NewtonRuntime, "register_render_resource")


def test_cpu_close_releases_views_and_render_sync_without_owner_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        newton_manager, "_configure_newton_stage", lambda *_a, **_k: None
    )
    events: list[str] = []
    manager = NewtonRuntime(
        stage=object(),
        physics_spec=IsaacNewtonCpuSpec(),
        device="cpu",
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        add_ground=False,
        ground_height=0.0,
    )

    class _View:
        def _release_from_manager(self) -> None:
            assert manager.stream is None
            events.append("view")

    class _RenderSync:
        def close(self) -> None:
            assert manager.stream is None
            events.append("render_sync")

    view = _View()
    manager._registered_views.add(view)
    manager._render_sync = _RenderSync()

    manager.close()
    manager.close()

    assert events == ["view", "render_sync"]
    assert manager.closed is True
    assert manager.stream is None
    assert manager.model is None
    assert manager.solver is None
    assert manager._solver_integration_store.execution == "cpu"


def test_render_publishes_once_without_owning_camera_products() -> None:
    events: list[object] = []

    class _RenderTarget:
        render_update_count = 2

        def __init__(self, name: str) -> None:
            self.name = name

        def set_render_active(self, active: bool) -> None:
            events.append((self.name, active))

    first = _RenderTarget("first")
    second = _RenderTarget("second")
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager._render_resources = [first, object(), second]
    manager._render_callback = lambda: events.append("app.update")
    manager._sim_time = 3.25
    manager.pre_render = lambda: events.append("pre_render")

    manager.render()

    assert manager.simulation_time == 3.25
    assert events == ["pre_render", "app.update"]


def test_cpu_pre_render_flushes_and_publishes_without_cuda_synchronization() -> None:
    events: list[object] = []

    class _RenderSync:
        def sync(self, body_q: object) -> None:
            events.append(("render_sync", body_q))

    body_q = object()
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.execution = "cpu"
    manager.stream = None
    manager._rendering_enabled = True
    manager._render_sync = _RenderSync()
    manager.state = SimpleNamespace(body_q=body_q)
    manager._require_initialized = lambda: None
    manager._flush_cold_state_updates = lambda: events.append("flush")

    manager.pre_render()

    assert events == ["flush", ("render_sync", body_q)]


def test_render_without_camera_target_ticks_app_once() -> None:
    events: list[str] = []
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager._render_resources = [object()]
    manager._render_callback = lambda: events.append("app.update")
    manager.pre_render = lambda: events.append("pre_render")

    manager.render()

    assert events == ["pre_render", "app.update"]


def test_render_update_does_not_publish_or_advance_physics() -> None:
    events: list[str] = []
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager._render_callback = lambda: events.append("app.update")
    manager._sim_time = 1.5
    manager.pre_render = lambda: events.append("pre_render")

    manager.render_update()

    assert manager.simulation_time == 1.5
    assert events == ["app.update"]


def test_render_does_not_consume_external_camera_update_count() -> None:
    events: list[object] = []

    class _RenderTarget:
        render_update_count = 4

        def set_render_active(self, active: bool) -> None:
            events.append(("active", active))

    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager._render_resources = [_RenderTarget()]
    manager._render_callback = lambda: events.append("app.update")
    manager._sim_time = 7.5
    manager.pre_render = lambda: events.append("pre_render")

    manager.render()

    assert manager.simulation_time == 7.5
    assert events == ["pre_render", "app.update"]


@pytest.mark.parametrize(
    ("schemas", "expected_drive_paths"),
    [
        ((), ()),
        (
            ("PhysicsDriveAPI:angular",),
            (
                "/World/envs/env_0/Robot/follower",
                "/World/envs/env_1/Robot/follower",
            ),
        ),
    ],
)
def test_multi_world_executor_metadata_reads_only_prototype_usd(
    schemas: tuple[str, ...],
    expected_drive_paths: tuple[str, ...],
) -> None:
    queried_paths: list[str] = []

    class _Prim:
        def __init__(self, *, valid: bool) -> None:
            self._valid = valid

        def IsValid(self) -> bool:
            return self._valid

        def GetAppliedSchemas(self) -> tuple[str, ...]:
            if not self._valid:
                raise AssertionError("null destination prim must not be inspected")
            return schemas

    class _Stage:
        def GetPrimAtPath(self, path: str) -> _Prim:
            queried_paths.append(str(path))
            return _Prim(valid=str(path) == "/World/envs/env_0/Robot/follower")

    replication = NewtonReplicationResult(
        builder=object(),
        prototype_builder=object(),
        global_stage_info={},
        prototype_stage_info={},
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        world_transforms=(object(), object()),
        source_world_transform=object(),
        environment_root="/World/envs",
    )
    expectations = tuple(
        ExpectedMasterFollowerConstraint(
            world=world,
            follower_joint_label=f"/World/envs/env_{world}/Robot/follower",
            master_joint_label=f"/World/envs/env_{world}/Robot/master",
            polycoef=(0.0, 1.0),
        )
        for world in range(2)
    )
    model = SimpleNamespace(
        joint_label=(
            "/World/envs/env_0/Robot/follower",
            "/World/envs/env_0/Robot/master",
            "/World/envs/env_1/Robot/follower",
            "/World/envs/env_1/Robot/master",
        ),
        joint_qd_start=(0, 1, 2, 3),
        joint_target_ke=(0.0, 0.0, 0.0, 0.0),
        joint_target_kd=(0.0, 0.0, 0.0, 0.0),
        mujoco=SimpleNamespace(actuator_target_label=()),
    )

    metadata = newton_manager._executor_metadata(
        stage=_Stage(),
        model=model,
        expectations=expectations,
        replication=replication,
    )

    assert queried_paths == ["/World/envs/env_0/Robot/follower"]
    assert metadata.follower_drive_prim_paths == expected_drive_paths
    assert metadata.follower_actuator_labels == ()
