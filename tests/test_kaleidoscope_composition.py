from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import linkerbot_sim.kaleidoscope.isaac_adapter as isaac_adapter
from linkerbot_sim.configuration import (
    load_kaleidoscope_config,
    semantic_config_fingerprint,
)
from linkerbot_sim.kaleidoscope.isaac_adapter import (
    KaleidoscopeSceneAssembly,
    create_kaleidoscope_runtime,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for composition"
)


class _Physics:
    kind = "physx_cuda"
    capabilities = SimpleNamespace(rendering=True)

    def __init__(self) -> None:
        self.steps = 0
        self.renders = 0

    def step(self, *, render: bool) -> None:
        assert render is False
        self.steps += 1

    def forward(self) -> None:
        return None

    def render(self) -> None:
        self.renders += 1


class _Session:
    def __init__(self) -> None:
        self.physics_runtime = _Physics()
        self.closed = False

    def close(self, *, exit_code: int = 0) -> None:
        del exit_code
        self.closed = True


class _Robot:
    def __init__(self, label: str, origins: torch.Tensor, y: float) -> None:
        self.label = label
        self.command_dim = 2
        self.state_dim = 2
        self.device = origins.device
        self.command_state_indices = torch.arange(
            2, device=self.device, dtype=torch.int64
        )
        self.q = torch.zeros((2, 2), device=self.device)
        self.qd = torch.zeros_like(self.q)
        self.target = torch.zeros_like(self.q)
        self.tcp = origins + torch.tensor([0.0, y, -0.38], device=self.device)
        self.tcp_q = torch.zeros((2, 4), device=self.device)
        self.tcp_q[:, 0] = 1.0

    def read_joint_positions(self, ids):
        return self.q.index_select(0, ids)

    def read_joint_velocities(self, ids):
        return self.qd.index_select(0, ids)

    def read_all_joint_positions(self, ids):
        return self.q.index_select(0, ids)

    def read_all_joint_velocities(self, ids):
        return self.qd.index_select(0, ids)

    def read_tcp_pose_wxyz(self, ids):
        return self.tcp.index_select(0, ids), self.tcp_q.index_select(0, ids)

    def write_joint_positions(self, ids, values):
        self.q.index_copy_(0, ids, values)

    def write_joint_velocities(self, ids, values):
        self.qd.index_copy_(0, ids, values)

    def write_joint_targets(self, ids, values):
        self.target.index_copy_(0, ids, values)

    def write_all_joint_positions(self, ids, values):
        self.q.index_copy_(0, ids, values)

    def write_all_joint_velocities(self, ids, values):
        self.qd.index_copy_(0, ids, values)

    def close(self):
        return None


class _Object:
    label = "Tblock"

    def __init__(self, origins: torch.Tensor) -> None:
        self.device = origins.device
        self.position = origins + torch.tensor([0.15, 0.0, -0.4], device=self.device)
        self.orientation = torch.tensor(
            [[0.70714084, 0.0, 0.70707272, 0.0]] * 2, device=self.device
        )
        self.velocity = torch.zeros((2, 6), device=self.device)

    def read_pose_wxyz(self, ids):
        return self.position.index_select(0, ids), self.orientation.index_select(0, ids)

    def read_com_velocity(self, ids):
        return self.velocity.index_select(0, ids)

    def write_pose_wxyz(self, ids, positions_world, orientations_wxyz):
        self.position.index_copy_(0, ids, positions_world)
        self.orientation.index_copy_(0, ids, orientations_wxyz)

    def write_velocity(self, ids, values):
        self.velocity.index_copy_(0, ids, values)

    def close(self):
        return None


def test_composition_consumes_strict_config_and_constructs_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_kaleidoscope_config()
    session = _Session()
    state_api_fingerprints: list[str] = []
    state_api_type = isaac_adapter.KaleidoscopeStateAPI

    def capture_state_api(*args, **kwargs):
        state_api_fingerprints.append(kwargs["compatibility_fingerprint"])
        return state_api_type(*args, **kwargs)

    monkeypatch.setattr(isaac_adapter, "KaleidoscopeStateAPI", capture_state_api)

    viewport = SimpleNamespace(render_every_n_steps=2)

    def assemble(*, config, num_envs, viewport):
        del config
        assert num_envs == 2
        assert viewport is not None
        origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device="cuda")
        robots = (
            _Robot("ar5v2_l6v1_0", origins, 0.03),
            _Robot("ar5v2_l6v1_1", origins, -0.03),
        )
        return KaleidoscopeSceneAssembly(
            session=session,
            robot_ports=robots,
            object_port=_Object(origins),
            env_origins=origins,
            nominal_joint_positions=torch.zeros(4, device="cuda"),
            nominal_block_position_local=torch.tensor([0.15, 0.0, -0.4], device="cuda"),
            nominal_block_orientation_wxyz=torch.tensor(
                [0.70714084, 0.0, 0.70707272, 0.0], device="cuda"
            ),
            joint_lower=torch.full((4,), -1.0, device="cuda"),
            joint_upper=torch.full((4,), 1.0, device="cuda"),
            ik_solvers={},
            fixed_orientations_wxyz={},
        )

    runtime = create_kaleidoscope_runtime(
        config=config,
        num_envs=2,
        viewport=viewport,
        assembly_factory=assemble,
    )
    assert state_api_fingerprints == [semantic_config_fingerprint(config)]
    assert runtime.viewport_enabled is True
    assert runtime.render_every_n_steps == 2
    observation, _ = runtime.reset()
    assert observation.shape == (2, runtime.observation_dim)
    result = runtime.step(torch.zeros((2, 4), device="cuda"))
    assert result.rewards.device.type == "cuda"
    assert session.physics_runtime.steps == config.task.action.physics_ticks_per_action
    steps_before_render = session.physics_runtime.steps
    runtime.render()
    assert session.physics_runtime.renders == 1
    assert session.physics_runtime.steps == steps_before_render
    runtime.close()
    assert session.closed


def test_state_api_compatibility_fingerprint_ignores_viewport_launch_settings() -> None:
    config = load_kaleidoscope_config()
    sessions: list[_Session] = []

    def assemble(*, config, num_envs, viewport):
        del config, viewport
        assert num_envs == 2
        session = _Session()
        sessions.append(session)
        origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device="cuda")
        robots = (
            _Robot("ar5v2_l6v1_0", origins, 0.03),
            _Robot("ar5v2_l6v1_1", origins, -0.03),
        )
        return KaleidoscopeSceneAssembly(
            session=session,
            robot_ports=robots,
            object_port=_Object(origins),
            env_origins=origins,
            nominal_joint_positions=torch.zeros(4, device="cuda"),
            nominal_block_position_local=torch.tensor([0.15, 0.0, -0.4], device="cuda"),
            nominal_block_orientation_wxyz=torch.tensor(
                [0.70714084, 0.0, 0.70707272, 0.0], device="cuda"
            ),
            joint_lower=torch.full((4,), -1.0, device="cuda"),
            joint_upper=torch.full((4,), 1.0, device="cuda"),
            ik_solvers={},
            fixed_orientations_wxyz={},
        )

    viewports = (
        None,
        SimpleNamespace(selected_env=0, render_every_n_steps=1),
        SimpleNamespace(selected_env=1, render_every_n_steps=3),
    )
    runtimes = []
    try:
        for viewport in viewports:
            runtimes.append(
                create_kaleidoscope_runtime(
                    config=config,
                    num_envs=2,
                    viewport=viewport,
                    assembly_factory=assemble,
                )
            )

        fingerprints = {
            runtime.state_api.compatibility_fingerprint for runtime in runtimes
        }
        assert len(fingerprints) == 1
        assert [runtime.viewport_enabled for runtime in runtimes] == [False, True, True]
        assert [runtime.render_every_n_steps for runtime in runtimes] == [0, 1, 3]
    finally:
        for runtime in reversed(runtimes):
            runtime.close()

    assert all(session.closed for session in sessions)


@pytest.mark.parametrize("invalid_num_envs", [True, 1.5, 0, -1])
def test_composition_rejects_noncanonical_num_envs(invalid_num_envs: object) -> None:
    config = load_kaleidoscope_config()

    with pytest.raises(ValueError, match="positive int"):
        create_kaleidoscope_runtime(
            config=config,
            num_envs=invalid_num_envs,  # type: ignore[arg-type]
            assembly_factory=lambda **_kwargs: pytest.fail(
                "invalid batch must fail before scene assembly"
            ),
        )
