from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

from linkerbot_sim.kaleidoscope.adapters.gymnasium import (  # noqa: E402
    GymnasiumKaleidoscopeAdapter,
)
from linkerbot_sim.kaleidoscope.checkpoint import (  # noqa: E402
    load_kaleidoscope_checkpoint,
    save_kaleidoscope_checkpoint,
)
from linkerbot_sim.kaleidoscope.registration import (  # noqa: E402
    GYMNASIUM_ENV_ID,
    register_gymnasium_envs,
)
from linkerbot_sim.kaleidoscope.snapshot import (  # noqa: E402
    KaleidoscopeEpisodeSnapshot,
)


class _NativeEnv:
    def __init__(self) -> None:
        self.num_envs = 3
        self.action_dim = 2
        self.action_low = -0.25
        self.action_high = 0.25
        self.observation_dim = 4
        self.device = torch.device("cuda:0")
        self.closed = False
        self.viewport_enabled = False
        self.render_calls = 0
        self.done = torch.zeros(3, device=self.device, dtype=torch.bool)
        self.reset_ids: list[torch.Tensor] = []
        self.runtime = SimpleNamespace(
            task=SimpleNamespace(
                buffers=SimpleNamespace(
                    last_finite_observation=torch.zeros((3, 4), device=self.device)
                )
            )
        )

    def reset(self, *, seed=None):
        observation = torch.full((3, 4), float(seed or 0), device=self.device)
        self.runtime.task.buffers.last_finite_observation.copy_(observation)
        return observation, {"episode_id": torch.zeros(3, device=self.device)}

    def reset_idx(self, ids):
        self.reset_ids.append(ids.clone())
        observation = torch.ones((ids.numel(), 4), device=self.device)
        self.runtime.task.buffers.last_finite_observation.index_copy_(
            0, ids, observation
        )
        return observation, {"env_ids": ids}

    def step(self, actions):
        observation = torch.nn.functional.pad(actions, (0, 2))
        return (
            observation,
            actions.sum(dim=1),
            self.done.clone(),
            torch.zeros(3, device=self.device, dtype=torch.bool),
            {"metric": actions[:, 0]},
        )

    def close(self) -> None:
        self.closed = True

    def render(self) -> None:
        self.render_calls += 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gymnasium_adapter_is_the_only_numpy_boundary() -> None:
    native = _NativeEnv()
    adapter = GymnasiumKaleidoscopeAdapter(native)
    observation, info = adapter.reset(seed=7)
    assert observation.shape == (3, 4)
    assert observation.dtype == np.float32
    assert np.all(observation == 7.0)
    assert info["episode_id"].shape == (3,)
    result = adapter.step(np.ones((3, 2), dtype=np.float32))
    assert all(isinstance(value, np.ndarray) for value in result[:4])
    assert isinstance(result[4]["metric"], np.ndarray)
    assert adapter.metadata["autoreset_mode"] is gym.vector.AutoresetMode.DISABLED
    assert np.all(adapter.single_action_space.low == -0.25)
    assert np.all(adapter.single_action_space.high == 0.25)
    adapter.close()
    assert native.closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gymnasium_same_step_handles_empty_and_partial_done_masks() -> None:
    native = _NativeEnv()
    adapter = GymnasiumKaleidoscopeAdapter(native, autoreset_mode="same_step")

    no_done = adapter.step(np.ones((3, 2), dtype=np.float32))
    assert native.reset_ids == []
    assert not no_done[4]["_final_obs"].any()
    assert not no_done[4]["_final_info"].any()

    native.done[1] = True
    partial = adapter.step(np.full((3, 2), 2.0, dtype=np.float32))
    assert len(native.reset_ids) == 1
    torch.testing.assert_close(
        native.reset_ids[0], torch.tensor([1], device="cuda", dtype=torch.int64)
    )
    np.testing.assert_allclose(partial[0][1], np.ones(4, dtype=np.float32))
    np.testing.assert_allclose(partial[4]["final_obs"][1, :2], [2.0, 2.0])
    assert partial[4]["_final_obs"].tolist() == [False, True, False]
    np.testing.assert_allclose(partial[4]["final_info"]["metric"][1], 2.0)
    assert partial[4]["final_info"]["_metric"].tolist() == [False, True, False]
    assert partial[4]["_metric"].tolist() == [True, False, True]
    assert partial[4]["_final_info"].tolist() == [False, True, False]

    observation, info = adapter.reset(
        options={"reset_mask": np.zeros(3, dtype=np.bool_)}
    )
    assert observation.shape == (3, 4)
    assert info["reset_mask"].tolist() == [False, False, False]
    assert len(native.reset_ids) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gymnasium_human_render_requires_and_delegates_to_viewport_env() -> None:
    headless = _NativeEnv()
    with pytest.raises(ValueError, match="requires a Kaleidoscope viewport"):
        GymnasiumKaleidoscopeAdapter(headless, render_mode="human")

    viewport = _NativeEnv()
    viewport.viewport_enabled = True
    adapter = GymnasiumKaleidoscopeAdapter(viewport, render_mode="human")
    adapter.render()

    assert adapter.render_mode == "human"
    assert adapter.metadata["render_modes"] == ["human"]
    assert viewport.render_calls == 1


def test_registration_is_explicit_and_idempotent() -> None:
    register_gymnasium_envs()
    register_gymnasium_envs()
    spec = gym.spec(GYMNASIUM_ENV_ID)
    assert spec.vector_entry_point == (
        "linkerbot_sim.kaleidoscope.bootstrap:make_gymnasium_env"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_persistent_checkpoint_is_an_explicit_cold_round_trip(tmp_path) -> None:
    snapshot = KaleidoscopeEpisodeSnapshot(
        env_ids=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
        fields={
            "robot.q": torch.arange(6, device="cuda", dtype=torch.float32).reshape(
                2, 3
            ),
            "rng.counter": torch.tensor([4, 9], device="cuda", dtype=torch.int64),
        },
        control_mode="effort",
        control_generation=7,
    )
    path = save_kaleidoscope_checkpoint(snapshot, tmp_path / "episode.npz")
    restored = load_kaleidoscope_checkpoint(path, device="cuda:0")
    assert restored.device.type == "cuda"
    assert restored.compatibility_fingerprint == snapshot.compatibility_fingerprint
    assert restored.control_mode == "effort"
    assert restored.control_generation == 7
    assert restored.schema_version == 2
    assert restored.fields.keys() == snapshot.fields.keys()
    for name in snapshot.fields:
        torch.testing.assert_close(restored.fields[name], snapshot.fields[name])
