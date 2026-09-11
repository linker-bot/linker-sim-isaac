"""Kaleidoscope 真实物理 smoke 入口的轻量合同测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from linkerbot_sim.controllers.control_mode import ControlModeChange, ControlModeState
from linkerbot_sim.kaleidoscope.snapshot import KaleidoscopeEpisodeSnapshot
from scripts import smoke_kaleidoscope_physics as smoke


class _FakeCudaEnv:
    def __init__(self, *, physics_kind: str) -> None:
        self.device = torch.device("cuda:0")
        self.num_envs = 2
        self.action_dim = 3
        self.observation_dim = 4
        self.closed = False
        backend = "newton" if physics_kind == "newton_cuda" else "physx"
        physics_runtime = SimpleNamespace(
            kind=physics_kind,
            backend=backend,
            execution="cuda",
            physics_dt=1.0 / 240.0,
            solver_integration_activation_width=0,
        )
        self.runtime = SimpleNamespace(
            physics_runtime=physics_runtime,
            session=SimpleNamespace(physics_runtime=physics_runtime),
            task=SimpleNamespace(),
        )
        self.state = {
            "robot.q": torch.zeros((2, 2), device=self.device),
            "robot.qd": torch.zeros((2, 2), device=self.device),
            "robot.target": torch.zeros((2, 2), device=self.device),
            "robot.position_reference": torch.zeros((2, 2), device=self.device),
            "object.pose_local_wxyz": torch.zeros((2, 7), device=self.device),
            "rng.key": torch.arange(2, device=self.device, dtype=torch.int64),
            "rng.counter": torch.zeros(2, device=self.device, dtype=torch.int64),
        }
        self.runtime.action_term = SimpleNamespace(
            physics_ticks_per_action=2,
            lower=torch.full((2,), -2.0, device=self.device),
            upper=torch.full((2,), 2.0, device=self.device),
        )
        self.runtime.views = SimpleNamespace(
            joint_positions=self.state["robot.q"],
            joint_velocities=self.state["robot.qd"],
        )
        self.restore_calls = 0
        self.set_state_calls = 0
        self.reset_idx_calls = 0
        self.control_mode = "position"
        self.control_generation = 0

    def reset(self, *, seed: int):
        assert seed == 123
        for name in (
            "robot.q",
            "robot.qd",
            "robot.target",
            "robot.position_reference",
        ):
            self.state[name].zero_()
        return (
            torch.zeros((2, 4), device=self.device),
            {"env_ids": torch.arange(2, device=self.device)},
        )

    def step(self, actions: torch.Tensor):
        assert actions.shape == (2, 3)
        if self.control_mode == "position":
            self.state["robot.target"].add_(actions[:, :2] * 0.05)
            self.state["robot.position_reference"].copy_(self.state["robot.target"])
            self.state["robot.q"].copy_(self.state["robot.target"])
        elif self.control_mode == "velocity":
            self.state["robot.target"].copy_(actions[:, :2])
            self.state["robot.position_reference"].add_(actions[:, :2] * 0.01)
            self.state["robot.q"].copy_(self.state["robot.position_reference"])
        else:
            self.state["robot.target"].copy_(actions[:, :2])
            self.state["robot.position_reference"].copy_(self.state["robot.q"])
        return (
            torch.zeros((2, 4), device=self.device),
            torch.zeros(2, device=self.device),
            torch.zeros(2, device=self.device, dtype=torch.bool),
            torch.zeros(2, device=self.device, dtype=torch.bool),
            {"metric": torch.zeros(2, device=self.device)},
        )

    def snapshot(self) -> KaleidoscopeEpisodeSnapshot:
        return KaleidoscopeEpisodeSnapshot(
            env_ids=torch.arange(2, device=self.device),
            fields={name: value.clone() for name, value in self.state.items()},
            control_mode=self.control_mode,
            control_generation=self.control_generation,
        )

    def restore_snapshot(self, snapshot: KaleidoscopeEpisodeSnapshot) -> None:
        assert snapshot.count == 2
        if snapshot.control_mode != self.control_mode:
            raise ValueError("snapshot control mode does not match runtime mode")
        self.restore_calls += 1
        for name, value in snapshot.fields.items():
            self.state[name].index_copy_(0, snapshot.env_ids, value)

    def set_state(
        self,
        state: dict[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.set_state_calls += 1
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for name, value in state.items():
            self.state[name].index_copy_(0, env_ids, value)
        if self.control_mode == "position":
            target = state.get("robot.target")
            reference = state.get("robot.position_reference")
            if target is not None and reference is None:
                self.state["robot.position_reference"].index_copy_(0, env_ids, target)
            elif reference is not None and target is None:
                self.state["robot.target"].index_copy_(0, env_ids, reference)

    def reset_idx(self, env_ids: torch.Tensor):
        self.reset_idx_calls += 1
        self.state["robot.q"].index_add_(
            0,
            env_ids,
            torch.full(
                (env_ids.numel(), 2),
                0.01,
                device=self.device,
            ),
        )
        self.state["robot.target"].index_copy_(
            0,
            env_ids,
            self.state["robot.q"].index_select(0, env_ids),
        )
        self.state["robot.position_reference"].index_copy_(
            0,
            env_ids,
            self.state["robot.q"].index_select(0, env_ids),
        )
        self.state["rng.counter"].index_add_(
            0,
            env_ids,
            torch.ones(env_ids.numel(), device=self.device, dtype=torch.int64),
        )
        return (
            torch.zeros((env_ids.numel(), 4), device=self.device),
            {"env_ids": env_ids.clone()},
        )

    def clone_state(self, source: torch.Tensor, target: torch.Tensor) -> None:
        for value in self.state.values():
            value.index_copy_(0, target, value.index_select(0, source))

    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        fields: tuple[str, ...] | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        names = tuple(self.state) if fields is None else fields
        return {
            name: value.index_select(0, env_ids).clone()
            for name, value in self.state.items()
            if name in names
        }

    def get_control_mode(self) -> ControlModeState:
        return ControlModeState(
            initial_mode="position",
            active_mode=self.control_mode,
            generation=self.control_generation,
            supported_modes=("position", "velocity", "effort"),
        )

    def set_control_mode(
        self,
        mode: str,
        *,
        expected_generation: int | None = None,
    ) -> ControlModeChange:
        if (
            expected_generation is not None
            and expected_generation != self.control_generation
        ):
            raise RuntimeError("control generation conflict")
        previous = self.control_mode
        if mode == previous:
            return ControlModeChange(
                previous_mode=previous,
                active_mode=previous,
                generation=self.control_generation,
                changed=False,
            )
        self.control_mode = mode
        self.control_generation += 1
        if mode == "position":
            self.state["robot.target"].copy_(self.state["robot.q"])
            self.state["robot.position_reference"].copy_(self.state["robot.q"])
        else:
            self.state["robot.target"].zero_()
            self.state["robot.position_reference"].copy_(self.state["robot.q"])
        return ControlModeChange(
            previous_mode=previous,
            active_mode=mode,
            generation=self.control_generation,
            changed=True,
        )

    def close(self, *, exit_code: int = 0) -> None:
        del exit_code
        self.closed = True


def test_parse_args_rejects_nonpositive_workload() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--num-envs", "0"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--steps", "0"])


def test_parse_args_rejects_unknown_profile() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--profile", "legacy_tiled"])


def test_parse_args_rejects_unknown_action_mode() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--action-mode", "batch_trajectory_planning"])


@pytest.mark.parametrize("profile", tuple(smoke.PROFILE_CONTRACTS))
@pytest.mark.parametrize(
    "action_mode", ("ee_delta_position", "ee_linear_path_position")
)
def test_action_smoke_variant_uses_strict_noncollision_kinematics(
    profile: str,
    action_mode: str,
) -> None:
    config = smoke._load_smoke_config(profile, action_mode=action_mode)

    assert config.task.action.mode == action_mode
    assert config.profiles.curobo == "kaleidoscope_batch_ik"
    assert config.curobo is not None
    assert config.curobo.motion_planner is None
    assert config.curobo.kinematics.max_batch_size >= config.environments.num_envs
    assert config.curobo.kinematics.collision_check is False
    assert config.curobo.kinematics.collision_cache is None
    assert all(item.resolved_profile is not None for item in config.scene.robots)
    assert all(item.resolved_profile is not None for item in config.scene.objects)


def test_finite_tensor_guard_rejects_nan_and_infinity() -> None:
    value = torch.tensor([0.0, float("nan"), float("inf")])

    with pytest.raises(RuntimeError, match="2 non-finite"):
        smoke._require_finite_tensor(value, label="zero_action.state")


def test_zero_action_physical_bounds_reject_finite_joint_explosion() -> None:
    state = {
        "robot.q": torch.tensor([[1.0e8, 0.0]]),
        "robot.qd": torch.zeros((1, 2)),
        "robot.target": torch.zeros((1, 2)),
    }
    env = SimpleNamespace(
        runtime=SimpleNamespace(
            physics_runtime=SimpleNamespace(physics_dt=1.0 / 240.0),
            action_term=SimpleNamespace(
                controller=SimpleNamespace(
                    lower=torch.full((2,), -2.0),
                    upper=torch.full((2,), 2.0),
                )
            ),
            views=SimpleNamespace(
                joint_positions=torch.zeros((1, 2)),
                joint_velocities=torch.zeros((1, 2)),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="asset angle domain"):
        smoke._zero_action_physical_metrics(env, state, label="zero_action")


def test_solver_successor_tolerance_only_applies_to_warmstart() -> None:
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    warmstart_difference = expected.clone()
    warmstart_difference[0, 3] += smoke.SOLVER_WARMSTART_SUCCESSOR_ATOL * 0.5

    smoke._assert_solver_persistent_successor(
        expected,
        warmstart_difference,
        warmstart_offset=2,
        label="solver.persistent",
    )

    excessive_warmstart_difference = expected.clone()
    excessive_warmstart_difference[0, 3] += smoke.SOLVER_WARMSTART_SUCCESSOR_ATOL * 2.0
    with pytest.raises(AssertionError, match="qacc_warmstart"):
        smoke._assert_solver_persistent_successor(
            expected,
            excessive_warmstart_difference,
            warmstart_offset=2,
            label="solver.persistent",
        )

    activation_difference = expected.clone()
    activation_difference[0, 1] += 1.0e-6
    with pytest.raises(AssertionError, match="time_act"):
        smoke._assert_solver_persistent_successor(
            expected,
            activation_difference,
            warmstart_offset=2,
            label="solver.persistent",
        )


def test_newton_contact_probe_requires_active_contacts_in_every_world() -> None:
    synchronized: list[bool] = []
    counts = SimpleNamespace(numpy=lambda: np.array([2], dtype=np.int32))
    world_ids = SimpleNamespace(numpy=lambda: np.array([0, 1, -1, -1], dtype=np.int32))
    runtime = SimpleNamespace(
        _synchronize_owner_stream=lambda: synchronized.append(True),
        diagnostics=lambda: {
            "contact_pipeline": "mujoco",
            "nconmax_per_world": 200,
        },
        solver=SimpleNamespace(
            mjw_data=SimpleNamespace(
                nacon=counts,
                naconmax=4,
                contact=SimpleNamespace(worldid=world_ids),
            )
        ),
    )
    env = SimpleNamespace(
        num_envs=2,
        runtime=SimpleNamespace(physics_runtime=runtime),
    )
    contract = smoke.PROFILE_CONTRACTS["newton_cuda"]

    report = smoke._newton_contact_probe(env, contract)

    assert synchronized == [True]
    assert report == {
        "performed": True,
        "pipeline": "mujoco",
        "active_worlds": 2,
        "total_contacts": 2,
        "min_contacts": 1,
        "max_contacts": 1,
        "contact_capacity": 4,
    }
    world_ids.numpy = lambda: np.array([0, 0, -1, -1], dtype=np.int32)
    with pytest.raises(RuntimeError, match="no physical contacts in 1 world"):
        smoke._newton_contact_probe(env, contract)

    host = lambda values: SimpleNamespace(  # noqa: E731
        numpy=lambda: np.asarray(values, dtype=np.int32)
    )
    runtime.diagnostics = lambda: {
        "contact_pipeline": "newton",
        "nconmax_per_world": 200,
    }
    runtime._contacts = SimpleNamespace(
        rigid_contact_count=host([2]),
        rigid_contact_max=8,
        rigid_contact_shape0=host([0, 0, -1, -1, -1, -1, -1, -1]),
        rigid_contact_shape1=host([1, 3, -1, -1, -1, -1, -1, -1]),
    )
    runtime.model = SimpleNamespace(shape_world=host([-1, 0, 0, 1, 1]))

    report = smoke._newton_contact_probe(env, contract)

    assert report == {
        "performed": True,
        "pipeline": "newton",
        "active_worlds": 2,
        "total_contacts": 2,
        "min_contacts": 1,
        "max_contacts": 1,
        "raw_contact_capacity": 8,
    }


@pytest.mark.parametrize(
    ("profile", "engine", "runtime_kind", "kit_filename"),
    (
        (
            "physx_cuda",
            "physx",
            "physx_cuda",
            "linkerbot_sim.kaleidoscope.physx_cuda.python.kit",
        ),
        (
            "newton_cuda",
            "newton",
            "newton_cuda",
            "linkerbot_sim.kaleidoscope.newton.python.kit",
        ),
    ),
)
def test_formal_profiles_select_their_backend_specific_kit(
    profile: str,
    engine: str,
    runtime_kind: str,
    kit_filename: str,
) -> None:
    config, contract = smoke._resolve_profile_contract(profile, num_envs=2)

    assert (config.physics.engine, config.physics.execution) == (engine, "cuda")
    assert contract.runtime_kind == runtime_kind
    assert contract.kit_filename == kit_filename


@pytest.mark.parametrize("profile", tuple(smoke.PROFILE_CONTRACTS))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_smoke_uses_public_env_and_checks_clone(
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract = smoke.PROFILE_CONTRACTS[profile]
    config = SimpleNamespace(
        physics=SimpleNamespace(engine=contract.engine, execution=contract.execution)
    )
    env = _FakeCudaEnv(physics_kind=contract.runtime_kind)
    captured = SimpleNamespace(config=None, num_envs=None)

    def make_env(*, config: object, num_envs: int):
        captured.config = config
        captured.num_envs = num_envs
        return env

    monkeypatch.setattr(smoke, "load_kaleidoscope_config", lambda name: config)
    monkeypatch.setattr(
        smoke,
        "_selected_kit_name",
        lambda _config, *, num_envs: contract.kit_filename,
    )
    monkeypatch.setattr(smoke, "_enabled_extension_names", lambda: ())
    monkeypatch.setattr(
        smoke,
        "_newton_contact_probe",
        lambda _env, _contract, **_kwargs: {"performed": True},
    )
    repeated_seed_reset = {
        "verified": contract.runtime_kind == "physx_cuda",
        "reason": "not_physx_cuda"
        if contract.runtime_kind != "physx_cuda"
        else "test_double",
    }
    monkeypatch.setattr(
        smoke,
        "_exercise_physx_repeated_seed_reset",
        lambda _env, _contract, **_kwargs: repeated_seed_reset,
    )
    monkeypatch.setattr(smoke, "make_torch_env", make_env)

    result = smoke.run_smoke(profile=profile, num_envs=2, steps=2)

    assert captured.config is config
    assert captured.num_envs == 2
    assert result["physics_engine"] == contract.engine
    assert result["physics_execution"] == contract.execution
    assert result["runtime_kind"] == contract.runtime_kind
    assert result["kit"] == contract.kit_filename
    assert result["snapshot_round_trip_verified"] is True
    assert result["repeated_seed_reset"] == repeated_seed_reset
    assert result["control_mode_switching"] == {
        "verified": True,
        "sequence": ["position", "velocity", "effort", "position"],
        "generation": 3,
        "identity_owners": [
            "action_term",
            "physics_runtime",
            "runtime",
            "session",
            "task",
        ],
        "same_mode_snapshot_restores": 4,
        "cross_mode_restore_rejected": True,
        "reset_preserved_mode": True,
    }
    assert result["zero_action_finite_verified"] is True
    assert result["zero_action_physical_bounds_verified"] is True
    assert result["physical_contacts"] == {"performed": True}
    assert result["zero_action_metrics"]["max_abs_robot_q_rad"] == 0.0
    assert result["set_state_verified"] is True
    assert result["clone_verified"] is True
    assert result["partial_reset_isolation_verified"] is True
    assert result["action_row_isolation_verified"] is True
    assert result["clone_successor_verified"] is True
    assert result["state_fields"] == sorted(env.state)
    assert env.set_state_calls == 5
    assert env.restore_calls == 5
    assert env.reset_idx_calls == 1
    if profile == "newton_cuda":
        assert result["extension_audit"] == {"checked": True, "forbidden": []}
    assert env.closed is True
    assert smoke.SUCCESS_MARKER in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_runtime_backend_mismatch_fails_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = smoke.PROFILE_CONTRACTS["newton_cuda"]
    config = SimpleNamespace(
        physics=SimpleNamespace(engine=contract.engine, execution=contract.execution)
    )
    env = _FakeCudaEnv(physics_kind="physx_cuda")
    monkeypatch.setattr(smoke, "load_kaleidoscope_config", lambda _name: config)
    monkeypatch.setattr(
        smoke,
        "_selected_kit_name",
        lambda _config, *, num_envs: contract.kit_filename,
    )
    monkeypatch.setattr(smoke, "make_torch_env", lambda **_kwargs: env)

    with pytest.raises(RuntimeError, match="runtime differs"):
        smoke.run_smoke(profile="newton_cuda", num_envs=2, steps=1)

    assert env.closed is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_newton_smoke_rejects_isaac_newton_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = smoke.PROFILE_CONTRACTS["newton_cuda"]
    config = SimpleNamespace(
        physics=SimpleNamespace(engine=contract.engine, execution=contract.execution)
    )
    env = _FakeCudaEnv(physics_kind=contract.runtime_kind)
    monkeypatch.setattr(smoke, "load_kaleidoscope_config", lambda _name: config)
    monkeypatch.setattr(
        smoke,
        "_selected_kit_name",
        lambda _config, *, num_envs: contract.kit_filename,
    )
    monkeypatch.setattr(
        smoke,
        "_enabled_extension_names",
        lambda: ("isaacsim.physics.newton",),
    )
    monkeypatch.setattr(smoke, "make_torch_env", lambda **_kwargs: env)

    with pytest.raises(RuntimeError, match="physics-owner extensions"):
        smoke.run_smoke(profile="newton_cuda", num_envs=2, steps=1)

    assert env.closed is True
