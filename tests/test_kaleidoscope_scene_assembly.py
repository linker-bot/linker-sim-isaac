from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from linkerbot_sim.configuration import (
    load_kaleidoscope_config,
    load_kaleidoscope_viewport_config,
)
from linkerbot_sim.kaleidoscope import scene_assembly
from linkerbot_sim.kaleidoscope.ik import (
    BatchIKTensorResult,
    EnvLocalDeviceBatchIKSolver,
)
from linkerbot_sim.kaleidoscope.physx_ports import (
    IsaacArticulationTensorPort,
    IsaacRigidObjectTensorPort,
)
from linkerbot_sim.kaleidoscope.runtime import KaleidoscopeRuntime


class _RawArticulation:
    def __init__(self, *, rows: int, offset: float) -> None:
        self.dof_names = ["j0", "j1"]
        self.q = torch.full((rows, 2), offset, device="cuda")
        self.qd = torch.zeros_like(self.q)
        self.target = self.q.clone()
        self.velocity_target = torch.zeros_like(self.q)
        # Isaac Sim 6 的真实 CUDA articulation 仍会把 DOF limits 作为 CPU 冷元数据返回。
        self.limits = torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]] * rows)
        self.invalidated = False

    def get_applied_actions(self, *, clone):
        assert clone is False
        return SimpleNamespace(
            joint_positions=self.target,
            joint_velocities=self.velocity_target,
        )

    def get_dof_limits(self):
        return self.limits

    def get_joint_positions(self, *, indices, joint_indices, clone):
        assert clone is False
        return self.q.index_select(0, indices).index_select(1, joint_indices)

    def get_joint_velocities(self, *, indices, joint_indices, clone):
        assert clone is False
        return self.qd.index_select(0, indices).index_select(1, joint_indices)

    def set_joint_positions(self, values, *, indices, joint_indices):
        selected = self.q.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.q.index_copy_(0, indices, selected)
        self.target.index_copy_(0, indices, self.q.index_select(0, indices))

    def set_joint_velocities(self, values, *, indices, joint_indices):
        selected = self.qd.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.qd.index_copy_(0, indices, selected)
        self.velocity_target.index_copy_(0, indices, self.qd.index_select(0, indices))

    def set_joint_position_targets(self, values, *, indices, joint_indices):
        selected = self.target.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.target.index_copy_(0, indices, selected)

    def set_joint_velocity_targets(self, values, *, indices, joint_indices):
        selected = self.velocity_target.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.velocity_target.index_copy_(0, indices, selected)

    def invalidate(self):
        self.invalidated = True


class _RawRigid:
    def __init__(self, positions: torch.Tensor) -> None:
        self.position = positions.clone()
        self.orientation = torch.zeros((positions.shape[0], 4), device="cuda")
        self.orientation[:, 0] = 1.0
        self.velocity = torch.zeros((positions.shape[0], 6), device="cuda")
        self.invalidated = False

    def get_world_poses(self, *, indices, clone):
        assert clone is False
        return self.position.index_select(0, indices), self.orientation.index_select(
            0, indices
        )

    def get_velocities(self, *, indices, clone):
        assert clone is False
        return self.velocity.index_select(0, indices)

    def set_world_poses(self, *, positions, orientations, indices):
        self.position.index_copy_(0, indices, positions)
        self.orientation.index_copy_(0, indices, orientations)

    def set_velocities(self, values, *, indices):
        self.velocity.index_copy_(0, indices, values)

    def invalidate(self):
        self.invalidated = True


class _Physics:
    kind = "physx_cuda"

    def __init__(self) -> None:
        self.world = object()
        self.resets = 0
        self.forwards = 0

    def reset(self) -> None:
        self.resets += 1

    def forward(self) -> None:
        self.forwards += 1


class _Session:
    def __init__(self) -> None:
        self.stage = object()
        self.physics_runtime = _Physics()
        self.close_codes: list[int] = []

    def close(self, *, exit_code: int = 0) -> None:
        self.close_codes.append(exit_code)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_default_assembly_uses_spec_replicated_scene_and_no_joint_only_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_kaleidoscope_config()
    session = _Session()
    origins = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    left = _RawArticulation(rows=2, offset=0.1)
    right = _RawArticulation(rows=2, offset=-0.1)
    robots = tuple(
        SimpleNamespace(
            label=label,
            articulation_view=view,
            command_joint_indices=np.asarray([0, 1], dtype=np.int64),
            command_joint_names=("j0", "j1"),
            tcp_offset_xyz=(0.0, 0.0, 0.0),
            tcp_offset_rpy=(0.0, 0.0, 0.0),
            asset_path=None,
            asset_type="usd",
            profile={},
            tcp_frame_name="tcp",
        )
        for label, view in (("ar5v2_l6v1_0", left), ("ar5v2_l6v1_1", right))
    )
    replicated = SimpleNamespace(robots=robots, env_origins=origins)
    tcp_views = {
        robot.label: _RawRigid(
            torch.as_tensor(origins, device="cuda")
            + torch.tensor([0.0, 0.0, -0.3], device="cuda")
        )
        for robot in robots
    }
    block = _RawRigid(
        torch.as_tensor(origins, device="cuda")
        + torch.tensor([0.15, 0.0, -0.4], device="cuda")
    )
    captured: dict[str, object] = {}

    def session_factory(*, spec):
        captured["spec"] = spec
        return session

    def builder(**kwargs):
        captured["builder"] = kwargs
        return replicated

    monkeypatch.setattr(
        scene_assembly,
        "finalize_replicated_robot_views",
        lambda value: value,
    )
    monkeypatch.setattr(
        scene_assembly,
        "create_tcp_rigid_views",
        lambda _scene: tcp_views,
    )
    monkeypatch.setattr(
        scene_assembly,
        "create_dynamic_object_rigid_view",
        lambda _scene, *, object_name: block,
    )

    assembly = scene_assembly.build_kaleidoscope_scene_assembly(
        config=config,
        num_envs=2,
        session_factory=session_factory,
        replicated_scene_builder=builder,
        ik_solver_factory=lambda **_kwargs: pytest.fail(
            "joint_delta must not construct IK"
        ),
    )

    assert captured["spec"].experience_family == "kaleidoscope"
    assert captured["builder"]["controller_bundle"] == config.default_controller_bundle
    assert captured["builder"]["controller_bundles"] is config.controller_bundles
    assert captured["builder"]["environment_settings"] is config.environments
    assert "replication_settings" not in captured["builder"]
    assert captured["builder"]["dynamic_object_name"] == "Tblock"
    assert session.physics_runtime.resets == 1
    assert session.physics_runtime.forwards == 1
    torch.testing.assert_close(
        assembly.nominal_joint_positions,
        torch.tensor([0.1, 0.1, -0.1, -0.1], device="cuda"),
    )
    torch.testing.assert_close(
        assembly.joint_lower,
        torch.tensor([-1.0, -2.0, -1.0, -2.0], device="cuda"),
    )
    assert assembly.ik_solvers == {}
    for port in (*assembly.robot_ports, assembly.object_port):
        port.close()
    session.close()


def test_session_spec_is_renderer_free_and_uses_engine_gpu_buffer_defaults() -> None:
    config = load_kaleidoscope_config()
    spec = scene_assembly.session_spec_from_config(config)
    assert spec.experience_family == "kaleidoscope"
    assert spec.physics.kind == "physx_cuda"
    assert spec.compute.cuda_device == config.cuda_device
    assert spec.compute_device == config.torch_device
    assert spec.physics_device == config.torch_device
    assert spec.physics.enable_scene_query_support is False
    assert not hasattr(config.physics, "gpu_buffers")
    assert not hasattr(spec.physics, "gpu_buffers")
    assert spec.physics_dt == pytest.approx(1.0 / 240.0)
    assert spec.render.enabled is False
    assert spec.app.gui is False


def test_newton_session_spec_uses_final_override_as_world_count() -> None:
    config = load_kaleidoscope_config("newton_cuda")
    spec = scene_assembly.session_spec_from_config(config, num_envs=17)

    assert spec.experience_family == "kaleidoscope"
    assert spec.physics.kind == "newton_cuda"
    assert spec.compute.cuda_device == config.cuda_device
    assert spec.physics_device == config.torch_device
    assert spec.physics.world_count == 17
    assert spec.physics.nconmax_per_world == config.physics.nconmax_per_world
    assert spec.physics.njmax_per_world == config.physics.njmax_per_world
    assert spec.render.enabled is False
    assert spec.app.gui is False


@pytest.mark.parametrize("profile", ["physx_cuda", "newton_cuda"])
@pytest.mark.parametrize("invalid_num_envs", [True, 1.5, 0, -1])
def test_session_spec_rejects_noncanonical_num_envs(
    profile: str,
    invalid_num_envs: object,
) -> None:
    with pytest.raises(ValueError, match="positive int"):
        scene_assembly.session_spec_from_config(
            load_kaleidoscope_config(profile),
            num_envs=invalid_num_envs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("profile", ("physx_cuda", "newton_cuda"))
def test_viewport_session_spec_selects_one_world_without_changing_physics_batch(
    profile: str,
) -> None:
    from dataclasses import replace

    config = load_kaleidoscope_config(profile)
    viewport = replace(
        load_kaleidoscope_viewport_config(),
        selected_env=3,
        render_every_n_steps=2,
    )
    spec = scene_assembly.session_spec_from_config(
        config,
        num_envs=4,
        viewport=viewport,
    )

    assert spec.app.gui is True
    assert spec.app.disable_viewport_updates is False
    assert spec.app.fast_shutdown is True
    assert spec.render.enabled is True
    assert spec.render.visible_world_indices == (3,)
    assert spec.render.width == viewport.width
    assert spec.physics_dt == pytest.approx(1.0 / config.scene.physics_frequency_hz)
    assert spec.rendering_dt == pytest.approx(
        spec.physics_dt
        * config.task.action.physics_ticks_per_action
        * viewport.render_every_n_steps
    )
    if profile == "newton_cuda":
        assert spec.physics.world_count == 4


def test_viewport_session_spec_rejects_selection_outside_final_batch() -> None:
    from dataclasses import replace

    viewport = replace(load_kaleidoscope_viewport_config(), selected_env=2)
    with pytest.raises(ValueError, match="selected_env"):
        scene_assembly.session_spec_from_config(
            load_kaleidoscope_config(),
            num_envs=2,
            viewport=viewport,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_newton_assembly_uses_project_runtime_without_isaac_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linkerbot_sim.isaac.replicated_scene import newton_builder
    from linkerbot_sim.kaleidoscope import newton_ports

    config = load_kaleidoscope_config("newton_cuda")
    session = _Session()
    session.physics_runtime.kind = "newton_cuda"
    del session.physics_runtime.world
    origins = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    left = _RawArticulation(rows=2, offset=0.2)
    right = _RawArticulation(rows=2, offset=-0.2)
    robots = tuple(
        SimpleNamespace(
            label=label,
            articulation_view=view,
            command_joint_indices=np.asarray([0, 1], dtype=np.int64),
            command_joint_names=("j0", "j1"),
            tcp_offset_xyz=(0.0, 0.0, 0.0),
            tcp_offset_rpy=(0.0, 0.0, 0.0),
            asset_path=None,
            asset_type="usd",
            profile={},
            tcp_frame_name="tcp",
        )
        for label, view in (("ar5v2_l6v1_0", left), ("ar5v2_l6v1_1", right))
    )
    replicated = SimpleNamespace(robots=robots, env_origins=origins)
    tcp_views = {
        robot.label: _RawRigid(
            torch.as_tensor(origins, device="cuda")
            + torch.tensor([0.0, 0.0, -0.3], device="cuda")
        )
        for robot in robots
    }
    block = _RawRigid(
        torch.as_tensor(origins, device="cuda")
        + torch.tensor([0.15, 0.0, -0.4], device="cuda")
    )
    captured: dict[str, object] = {}

    def session_factory(*, spec):
        captured["spec"] = spec
        return session

    def builder(**kwargs):
        captured["builder"] = kwargs
        return replicated

    def articulation_port(**kwargs):
        return IsaacArticulationTensorPort(
            label=kwargs["label"],
            view=kwargs["view"],
            tcp_view=kwargs["tcp_view"],
            command_joint_indices=torch.tensor(
                [0, 1], device="cuda", dtype=torch.int64
            ),
            device=kwargs["device"],
            orientation_order="wxyz",
        )

    def object_tensor_port(**kwargs):
        return IsaacRigidObjectTensorPort(
            label=kwargs["label"],
            view=kwargs["view"],
            device=kwargs["device"],
            orientation_order="wxyz",
        )

    class _SolverStatePort:
        field_name = "solver.persistent"

        def __init__(self, *, runtime, device) -> None:
            del runtime
            self.device = torch.device(device)
            self.num_envs = 2
            self.tensor = torch.zeros((2, 5), device=self.device)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        newton_builder,
        "create_newton_tcp_rigid_views",
        lambda _scene, *, runtime: tcp_views,
    )
    monkeypatch.setattr(
        newton_builder,
        "create_newton_dynamic_object_rigid_view",
        lambda _scene, *, runtime, object_name: block,
    )
    monkeypatch.setattr(
        newton_builder,
        "newton_command_joint_limits",
        lambda _robot, *, runtime, device: (
            torch.tensor([-1.0, -2.0], device=device),
            torch.tensor([1.0, 2.0], device=device),
        ),
    )
    monkeypatch.setattr(
        newton_ports,
        "NewtonArticulationTensorPort",
        articulation_port,
    )
    monkeypatch.setattr(
        newton_ports,
        "NewtonRigidObjectTensorPort",
        object_tensor_port,
    )
    monkeypatch.setattr(
        newton_ports,
        "NewtonSolverIntegrationTensorPort",
        _SolverStatePort,
    )

    assembly = scene_assembly.build_kaleidoscope_scene_assembly(
        config=config,
        num_envs=2,
        session_factory=session_factory,
        newton_scene_builder=builder,
    )

    assert captured["spec"].physics.kind == "newton_cuda"
    assert captured["spec"].physics.world_count == 2
    assert captured["builder"]["prepare_newton_render_topology"] is False
    assert captured["builder"]["runtime"] is session.physics_runtime
    assert captured["builder"]["controller_bundle"] == config.default_controller_bundle
    assert captured["builder"]["controller_bundles"] is config.controller_bundles
    assert captured["builder"]["environment_settings"] is config.environments
    assert "replication_settings" not in captured["builder"]
    assert captured["builder"]["dynamic_object_name"] == "Tblock"
    assert session.physics_runtime.resets == 1
    assert session.physics_runtime.forwards == 1
    torch.testing.assert_close(
        assembly.nominal_joint_positions,
        torch.tensor([0.2, 0.2, -0.2, -0.2], device="cuda"),
    )
    assert assembly.physics_state_port.field_name == "solver.persistent"
    for port in (*assembly.robot_ports, assembly.object_port):
        port.close()
    assembly.physics_state_port.close()
    session.close()


def test_partial_assembly_failure_closes_session_with_failure_status() -> None:
    config = load_kaleidoscope_config()
    session = _Session()

    with pytest.raises(RuntimeError, match="stage import failed"):
        scene_assembly.build_kaleidoscope_scene_assembly(
            config=config,
            num_envs=2,
            session_factory=lambda *, spec: session,
            replicated_scene_builder=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("stage import failed")
            ),
        )
    assert session.close_codes == [1]


class _CloseResource:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once

    def close(self) -> None:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise RuntimeError("retry me")


def test_runtime_close_retries_failed_child_before_closing_session() -> None:
    runtime = object.__new__(KaleidoscopeRuntime)
    action = _CloseResource(fail_once=True)
    views = _CloseResource()
    task = _CloseResource()
    session = _CloseResource()
    runtime.action_term = action
    runtime.views = views
    runtime.task = task
    runtime.session = session
    runtime._closed = False
    runtime._closing_started = False
    runtime._close_completed = set()

    with pytest.raises(RuntimeError, match="retry me"):
        runtime.close()
    assert views.calls == task.calls == 1
    assert session.calls == 0
    assert runtime._closed is False

    runtime.close()
    assert action.calls == 2
    assert views.calls == task.calls == session.calls == 1
    assert runtime._closed is True
    runtime.close()
    assert session.calls == 1


def test_clone_state_dispatches_exactly_once_before_forward() -> None:
    runtime = object.__new__(KaleidoscopeRuntime)
    calls: list[str] = []
    runtime._closed = False
    runtime._closing_started = False
    runtime._failed = False
    runtime._outstanding_token = None
    runtime.state_api = SimpleNamespace(
        poisoned=False,
        clone_state=lambda *args, **kwargs: calls.append("clone"),
    )
    runtime.session = SimpleNamespace(
        physics_runtime=SimpleNamespace(forward=lambda: calls.append("forward"))
    )
    runtime.views = SimpleNamespace(refresh=lambda: calls.append("refresh"))
    runtime.clone_state("source", "target")
    assert calls == ["clone", "forward", "refresh"]


@pytest.mark.parametrize(
    ("runtime_method", "state_method", "arguments"),
    (
        ("set_state", "set_state", ({"robot.q": object()},)),
        ("restore_snapshot", "restore_snapshot", (object(),)),
        ("clone_state", "clone_state", (object(), object())),
    ),
)
def test_state_forward_failure_makes_runtime_fail_stop(
    runtime_method: str,
    state_method: str,
    arguments: tuple[object, ...],
) -> None:
    runtime = object.__new__(KaleidoscopeRuntime)
    runtime._closed = False
    runtime._closing_started = False
    runtime._failed = False
    runtime._outstanding_token = None
    writes: list[str] = []
    state_api = SimpleNamespace(poisoned=False, get_state=lambda: {})
    setattr(
        state_api,
        state_method,
        lambda *_args, **_kwargs: writes.append(state_method),
    )
    runtime.state_api = state_api
    runtime.session = SimpleNamespace(
        physics_runtime=SimpleNamespace(
            forward=lambda: (_ for _ in ()).throw(RuntimeError("forward failed"))
        )
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        getattr(runtime, runtime_method)(*arguments)
    assert writes == [state_method]
    assert runtime._failed is True
    with pytest.raises(RuntimeError, match="fail-stop"):
        runtime.get_state()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_env_local_ik_wrapper_converts_goal_to_robot_base_on_gpu() -> None:
    class Solver:
        device = torch.device("cuda:0")
        command_dim = 2

        def __init__(self) -> None:
            self.positions = None
            self.orientations = None
            self.closed = False

        def solve(
            self,
            *,
            target_positions,
            target_orientations_wxyz,
            seeds,
            active_mask=None,
        ):
            self.positions = target_positions.clone()
            self.orientations = target_orientations_wxyz.clone()
            return BatchIKTensorResult(
                joint_positions=seeds.clone(),
                success=torch.ones(seeds.shape[0], device="cuda", dtype=torch.bool),
                position_error=torch.zeros(seeds.shape[0], device="cuda"),
                orientation_error=torch.zeros(seeds.shape[0], device="cuda"),
            )

        def close(self) -> None:
            self.closed = True

    raw = Solver()
    half = 2.0**-0.5
    solver = EnvLocalDeviceBatchIKSolver(
        raw,
        robot_root_position_local=torch.tensor([1.0, 0.0, 0.0], device="cuda"),
        robot_root_orientation_wxyz=torch.tensor([half, 0.0, 0.0, half], device="cuda"),
    )
    solver.solve(
        target_positions=torch.tensor([[1.0, 1.0, 0.0]], device="cuda"),
        target_orientations_wxyz=torch.tensor([[half, 0.0, 0.0, half]], device="cuda"),
        seeds=torch.zeros((1, 2), device="cuda"),
    )
    torch.testing.assert_close(
        raw.positions,
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        raw.orientations,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        atol=1.0e-6,
        rtol=0.0,
    )
    solver.close()
    assert raw.closed is True
