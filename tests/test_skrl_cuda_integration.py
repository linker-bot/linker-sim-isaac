from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from linkerbot_sim.configuration.training.skrl import SkrlTrainingSettings
from linkerbot_sim.kaleidoscope import KaleidoscopeTrainingPort
from linkerbot_sim.training.skrl.env import SkrlTorchAdapter
from linkerbot_sim.training.skrl.final_observation_ppo import (
    FinalObservationPPO,
    validate_skrl_ppo_source,
)
from linkerbot_sim.training.skrl.factory import make_skrl_trainer
from linkerbot_sim.training.skrl.memory import CudaRolloutMemory


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for skrl integration"
)


class _TokenNativeEnv:
    def __init__(self) -> None:
        self.num_envs = 3
        self.action_dim = 2
        self.action_low = -0.5
        self.action_high = 0.5
        self.observation_dim = 4
        self.device = torch.device("cuda:0")
        self._generation = 0
        self._outstanding = False
        self.closed = False
        self.last_observation = torch.zeros((3, 4), device=self.device)
        self.episode_return = torch.tensor([1.0, 12.0, 18.0], device=self.device)
        self.episode_length = torch.tensor(
            [1, 6, 9], device=self.device, dtype=torch.int64
        )

    def reset(self):
        self.last_observation.zero_()
        return self.last_observation, {}

    def begin_same_step(self) -> object:
        assert not self._outstanding
        self._outstanding = True
        return self._generation

    def step_same_step(self, token: object, actions: torch.Tensor):
        assert self._outstanding and token == self._generation
        observation = torch.full((3, 4), 5.0, device=self.device)
        return (
            observation,
            actions.sum(dim=1),
            torch.tensor([False, True, False], device=self.device),
            torch.tensor([False, False, True], device=self.device),
            {
                "dense_metric": torch.arange(3, device=self.device),
                "episode_return": self.episode_return,
                "episode_length": self.episode_length,
            },
        )

    def complete_same_step(self, token: object):
        assert self._outstanding and token == self._generation
        self.last_observation.fill_(5.0)
        self.last_observation[1:].zero_()
        self.episode_return[1:].zero_()
        self.episode_length[1:].zero_()
        self._outstanding = False
        self._generation += 1
        return self.last_observation

    def close(self) -> None:
        self.closed = True


def test_skrl_adapter_preserves_terminal_observation_before_same_step_reset() -> None:
    native = _TokenNativeEnv()
    assert isinstance(native, KaleidoscopeTrainingPort)
    adapter = SkrlTorchAdapter(native)
    adapter.reset()
    observation, reward, terminated, truncated, info = adapter.step(
        torch.ones((3, 2), device="cuda")
    )
    assert observation.shape == (3, 4)
    torch.testing.assert_close(observation[1:], torch.zeros((2, 4), device="cuda"))
    torch.testing.assert_close(
        info["final_obs"], torch.full((3, 4), 5.0, device="cuda")
    )
    assert "final_state" not in info
    assert "_final_state" not in info
    assert info["_final_obs"].tolist() == [False, True, True]
    torch.testing.assert_close(
        info["episode_return"], torch.tensor([1.0, 12.0, 18.0], device="cuda")
    )
    torch.testing.assert_close(
        info["episode_length"],
        torch.tensor([1, 6, 9], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(
        native.episode_return, torch.tensor([1.0, 0.0, 0.0], device="cuda")
    )
    torch.testing.assert_close(
        native.episode_length,
        torch.tensor([1, 0, 0], device="cuda", dtype=torch.int64),
    )
    assert info["episode_return"].data_ptr() != native.episode_return.data_ptr()
    assert info["episode_length"].data_ptr() != native.episode_length.data_ptr()
    assert reward.shape == terminated.shape == truncated.shape == (3, 1)
    assert native._generation == 1 and not native._outstanding


def test_cuda_rollout_memory_never_uses_cpu_or_numpy_selectors() -> None:
    memory = CudaRolloutMemory(memory_size=4, num_envs=2, device="cuda:0")
    memory.create_tensor(name="observations", size=3, dtype=torch.float32)
    memory.create_tensor(name="values", size=1, dtype=torch.float32)
    for index in range(4):
        memory.add_samples(
            observations=torch.full((2, 3), float(index), device="cuda"),
            values=torch.full((2, 1), float(index), device="cuda"),
        )
    batches = memory.sample(
        names=["observations", "values"],
        batch_size=len(memory),
        mini_batches=2,
    )
    assert len(batches) == 2
    assert all(tensor.device.type == "cuda" for batch in batches for tensor in batch)
    assert memory.sampling_indexes.device.type == "cuda"
    with pytest.raises(RuntimeError, match="sample_by_index"):
        memory.sample_by_index(names=["values"], indexes=[0])


def test_skrl_source_guard_and_environment_spaces() -> None:
    validate_skrl_ppo_source()
    adapter = SkrlTorchAdapter(_TokenNativeEnv())
    assert isinstance(adapter.observation_space, gym.spaces.Box)
    assert isinstance(adapter.action_space, gym.spaces.Box)
    assert np.all(adapter.action_space.low == -0.5)
    assert np.all(adapter.action_space.high == 0.5)
    assert adapter.num_envs == 3


class _Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, state_space=None) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=observation_space if state_space is None else state_space,
            action_space=action_space,
            device="cuda:0",
        )
        GaussianMixin.__init__(self, role="policy")
        self.net = torch.nn.Linear(4, 2)
        self.log_std = torch.nn.Parameter(torch.zeros(2))
        self.to(self.device)

    def compute(self, inputs, role=""):
        del role
        return self.net(inputs["observations"]), {"log_std": self.log_std}


class _Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, state_space=None) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=observation_space if state_space is None else state_space,
            action_space=action_space,
            device="cuda:0",
        )
        DeterministicMixin.__init__(self, role="value")
        self.net = torch.nn.Linear(4, 1, bias=False)
        torch.nn.init.ones_(self.net.weight)
        self.to(self.device)

    def compute(self, inputs, role=""):
        del role
        return self.net(inputs["states"]), {}


def _model_spaces(native: _TokenNativeEnv):
    observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        shape=(native.observation_dim,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        native.action_low,
        native.action_high,
        shape=(native.action_dim,),
        dtype=np.float32,
    )
    return observation_space, action_space


def _training_settings() -> SkrlTrainingSettings:
    return SkrlTrainingSettings(
        framework="skrl",
        algorithm="final_observation_ppo",
        device_source="environment",
        rollout_length=32,
        mini_batches=8,
        learning_epochs=5,
        learning_rate=3e-4,
        discount_factor=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
    )


def test_make_skrl_trainer_freezes_cuda_training_composition() -> None:
    native = _TokenNativeEnv()
    observation_space, action_space = _model_spaces(native)
    trainer = make_skrl_trainer(
        native,
        models={
            "policy": _Policy(observation_space, action_space),
            "value": _Value(observation_space, action_space),
        },
        settings=_training_settings(),
        timesteps=3,
        close_environment_at_exit=False,
        disable_progressbar=True,
    )

    assert isinstance(trainer.agents, FinalObservationPPO)
    assert isinstance(trainer.agents.memory, CudaRolloutMemory)
    assert trainer.agents.memory.memory_size == 32
    assert trainer.cfg.timesteps == 3
    assert trainer.cfg.environment_info == "__disabled__"
    assert trainer.cfg.close_environment_at_exit is False
    trainer.env.close()
    assert native.closed is True


@pytest.mark.parametrize(
    ("model_name", "field_name", "mismatch", "error_type", "message"),
    [
        ("policy", "action_space", "action_bounds", ValueError, "bounds must match"),
        ("value", "action_space", "action_bounds", ValueError, "bounds must match"),
        ("policy", "observation_space", "shape", ValueError, "shape must match"),
        ("value", "observation_space", "dtype", TypeError, "dtype must match"),
        ("policy", "state_space", "type", TypeError, "must be a Gymnasium Box"),
        ("value", "state_space", "shape", ValueError, "shape must match"),
        ("policy", "state_space", "dtype", TypeError, "dtype must match"),
        ("value", "state_space", "state_bounds", ValueError, "bounds must match"),
    ],
)
def test_make_skrl_trainer_rejects_model_space_mismatch(
    model_name: str,
    field_name: str,
    mismatch: str,
    error_type: type[Exception],
    message: str,
) -> None:
    native = _TokenNativeEnv()
    observation_space, action_space = _model_spaces(native)
    wrong_space = {
        "type": gym.spaces.Discrete(native.observation_dim),
        "action_bounds": gym.spaces.Box(
            -1.0, 1.0, shape=(native.action_dim,), dtype=np.float32
        ),
        "state_bounds": gym.spaces.Box(
            -1.0, 1.0, shape=(native.observation_dim,), dtype=np.float32
        ),
        "shape": gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(native.observation_dim + 1,),
            dtype=np.float32,
        ),
        "dtype": gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(native.observation_dim,),
            dtype=np.float64,
        ),
    }[mismatch]
    model_spaces = {
        "policy": {
            "observation_space": observation_space,
            "state_space": observation_space,
            "action_space": action_space,
        },
        "value": {
            "observation_space": observation_space,
            "state_space": observation_space,
            "action_space": action_space,
        },
    }
    model_spaces[model_name][field_name] = wrong_space
    models = {
        "policy": _Policy(**model_spaces["policy"]),
        "value": _Value(**model_spaces["value"]),
    }

    with pytest.raises(error_type, match=message):
        make_skrl_trainer(
            native,
            models=models,
            settings=_training_settings(),
            timesteps=3,
        )


@pytest.mark.parametrize("model_name", ("policy", "value"))
def test_make_skrl_trainer_rejects_model_device_mismatch(model_name: str) -> None:
    native = _TokenNativeEnv()
    observation_space, action_space = _model_spaces(native)
    models = {
        "policy": _Policy(observation_space, action_space),
        "value": _Value(observation_space, action_space),
    }
    models[model_name].device = torch.device("cpu")

    with pytest.raises(ValueError, match=rf"{model_name}\.device must match"):
        make_skrl_trainer(
            native,
            models=models,
            settings=_training_settings(),
            timesteps=3,
        )


def test_final_observation_ppo_bootstraps_only_time_limit_rows_and_updates() -> None:
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    memory = CudaRolloutMemory(memory_size=2, num_envs=3, device="cuda:0")
    cfg = PPO_CFG(
        rollouts=2,
        learning_epochs=1,
        mini_batches=1,
        time_limit_bootstrap=True,
        kl_threshold=0,
        mixed_precision=False,
        experiment=ExperimentCfg(write_interval=0, checkpoint_interval=0),
    )
    agent = FinalObservationPPO(
        models={
            "policy": _Policy(observation_space, action_space),
            "value": _Value(observation_space, action_space),
        },
        memory=memory,
        observation_space=observation_space,
        state_space=observation_space,
        action_space=action_space,
        device="cuda:0",
        cfg=cfg,
    )
    agent.init()
    agent.enable_training_mode(True)
    observation = torch.zeros((3, 4), device="cuda")
    actions, _ = agent.act(observation, observation, timestep=0, timesteps=2)
    final = torch.tensor([[10.0] * 4, [20.0] * 4, [30.0] * 4], device="cuda")
    agent.record_transition(
        observations=observation,
        states=observation,
        actions=actions,
        rewards=torch.ones((3, 1), device="cuda"),
        next_observations=observation,
        next_states=observation,
        terminated=torch.tensor([[False], [True], [False]], device="cuda"),
        truncated=torch.tensor([[True], [True], [False]], device="cuda"),
        infos={
            "final_obs": final,
            "_final_obs": torch.ones(3, device="cuda", dtype=torch.bool),
        },
        timestep=0,
        timesteps=2,
    )
    stored = memory.get_tensor_by_name("rewards")[0]
    torch.testing.assert_close(
        stored[:, 0], torch.tensor([40.6, 1.0, 1.0], device="cuda")
    )

    actions, _ = agent.act(observation, observation, timestep=1, timesteps=2)
    agent.record_transition(
        observations=observation,
        states=observation,
        actions=actions,
        rewards=torch.ones((3, 1), device="cuda"),
        next_observations=observation,
        next_states=observation,
        terminated=torch.zeros((3, 1), device="cuda", dtype=torch.bool),
        truncated=torch.zeros((3, 1), device="cuda", dtype=torch.bool),
        infos={
            "final_obs": observation,
            "_final_obs": torch.zeros(3, device="cuda", dtype=torch.bool),
        },
        timestep=1,
        timesteps=2,
    )
    agent.update(timestep=1, timesteps=2)
    assert all(np.isfinite(value) for value in agent.export_training_metrics().values())
