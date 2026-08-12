"""skrl 2.1 PPO 的 final-observation 与 CUDA-residency 修正版。"""

from __future__ import annotations

import hashlib
import inspect
import itertools
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from skrl import __version__ as skrl_version
from skrl import config as skrl_config
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo import ppo as skrl_ppo_module

from linkerbot_sim.training.skrl.memory import CudaRolloutMemory

if TYPE_CHECKING:
    import gymnasium
    from skrl.agents.torch.ppo import PPO_CFG
    from skrl.models.torch import Model


_SKRL_VERSION = "2.1.0"
_SOURCE_FINGERPRINTS = {
    "record_transition": "1622cf31e1086ad9df1873d8e0f983945b6dbb90e8118b985b6e376d6bf32c84",
    "update": "d1dbaa7913e04a1adb3811e312862cdc3c358102505534330ee79cfd097a8166",
    "compute_gae": "8956a9e281fda79f0ee89bee5608b96e2b48500fdacabbf895b13a8d17498f02",
}


def validate_skrl_ppo_source() -> None:
    """锁定被项目复制/替换的上游方法，版本漂移时拒绝静默运行。"""

    if skrl_version != _SKRL_VERSION:
        raise RuntimeError(
            f"FinalObservationPPO requires skrl {_SKRL_VERSION}, got {skrl_version}"
        )
    targets = {
        "record_transition": PPO.record_transition,
        "update": PPO.update,
        "compute_gae": skrl_ppo_module.compute_gae,
    }
    mismatches: list[str] = []
    for name, target in targets.items():
        digest = hashlib.sha256(inspect.getsource(target).encode()).hexdigest()
        if digest != _SOURCE_FINGERPRINTS[name]:
            mismatches.append(
                f"{name}: expected {_SOURCE_FINGERPRINTS[name]}, got {digest}"
            )
    if mismatches:
        raise RuntimeError(
            "skrl PPO source changed; re-audit the CUDA/final-observation bridge:\n"
            + "\n".join(mismatches)
        )


class FinalObservationPPO(PPO):
    """使用 terminal observation bootstrap，并移除 stock PPO 的 host sync。

    该类有意 pin 住 skrl 2.1.0。``record_transition`` 和 ``update`` 是对上游两个方法的
    source-guarded replacement：前者使用 SAME_STEP adapter 保存的 final observation；后者使用
    ``CudaRolloutMemory``，指标累计保留为 0-d CUDA tensor。
    """

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: CudaRolloutMemory,
        observation_space: "gymnasium.Space | None" = None,
        state_space: "gymnasium.Space | None" = None,
        action_space: "gymnasium.Space | None" = None,
        device: str | torch.device | None = None,
        cfg: "PPO_CFG | dict[str, Any]" = {},
    ) -> None:
        validate_skrl_ppo_source()
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        if not isinstance(memory, CudaRolloutMemory):
            raise TypeError("FinalObservationPPO requires CudaRolloutMemory")
        if memory.device != self.device or self.device.type != "cuda":
            raise ValueError("PPO, memory and environment must share one CUDA device")
        if not self.cfg.time_limit_bootstrap:
            raise ValueError("FinalObservationPPO requires time_limit_bootstrap=true")
        if self.cfg.kl_threshold != 0:
            raise ValueError("FinalObservationPPO requires kl_threshold=0")
        if any(item is not None for item in self.cfg.learning_rate_scheduler):
            raise ValueError("FinalObservationPPO disables learning-rate schedulers")
        if self.cfg.mixed_precision:
            raise ValueError(
                "FinalObservationPPO currently requires mixed_precision=false"
            )
        if self.cfg.rewards_shaper is not None:
            raise ValueError("FinalObservationPPO requires task-owned reward shaping")
        if self.cfg.experiment.write_interval != 0:
            raise ValueError("FinalObservationPPO requires experiment.write_interval=0")
        if self.cfg.experiment.checkpoint_interval != 0:
            raise ValueError("FinalObservationPPO requires checkpoint_interval=0")
        self._metric_accumulators = {
            name: torch.zeros((), device=self.device)
            for name in ("policy_loss", "value_loss", "entropy_loss", "stddev")
        }

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """从 final_obs bootstrap truncated 行，再写入原始 done flags。"""

        del timestep, timesteps
        if not self.training:
            return
        if not isinstance(infos, dict):
            raise TypeError("FinalObservationPPO requires dense CUDA info mapping")
        final_observation = _cuda_tensor(
            infos.get("final_obs"), "infos.final_obs", device=self.device, ndim=2
        )
        final_mask = _cuda_tensor(
            infos.get("_final_obs"),
            "infos._final_obs",
            device=self.device,
            ndim=1,
            dtype=torch.bool,
        )
        terminated = _done_column(terminated, "terminated", self.device)
        truncated = _done_column(truncated, "truncated", self.device)
        rewards = _column(rewards, "rewards", self.device)
        if final_observation.shape != next_observations.shape:
            raise ValueError("final_obs and next_observations shapes must match")
        if final_mask.shape[0] != next_observations.shape[0]:
            raise ValueError("_final_obs must have shape (N,)")
        final_mask_column = final_mask.reshape(-1, 1)
        bootstrap_mask = truncated & ~terminated & final_mask_column
        dense_next_observation = torch.where(
            final_mask_column, final_observation, next_observations
        )

        final_state = infos.get("final_state")
        if final_state is None:
            if next_states.shape != next_observations.shape:
                raise ValueError("central critic requires infos.final_state")
            final_state = final_observation
        final_state = _cuda_tensor(
            final_state, "infos.final_state", device=self.device, ndim=2
        )
        if final_state.shape != next_states.shape:
            raise ValueError("final_state and next_states shapes must match")
        dense_next_state = torch.where(final_mask_column, final_state, next_states)
        with torch.no_grad():
            inputs = {
                "observations": self._observation_preprocessor(dense_next_observation),
                "states": self._state_preprocessor(dense_next_state),
            }
            final_values, _ = self.value.act(inputs, role="value")
            final_values = self._value_preprocessor(final_values, inverse=True)
            stored_rewards = rewards + (
                self.cfg.discount_factor
                * final_values
                * bootstrap_mask.to(dtype=rewards.dtype)
            )

        self._current_next_observations = next_observations
        self._current_next_states = next_states
        self.memory.add_samples(
            observations=observations,
            states=states,
            actions=actions,
            rewards=stored_rewards,
            terminated=terminated,
            truncated=truncated,
            log_prob=self._current_log_prob,
            values=self._current_values,
        )

    def update(self, *, timestep: int, timesteps: int) -> None:
        """PPO update 的全 CUDA replacement；签名匹配 2.1 动态分派目标。"""

        del timestep, timesteps
        with torch.no_grad():
            inputs = {
                "observations": self._observation_preprocessor(
                    self._current_next_observations
                ),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = _compute_gae_device(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )
        self.memory.set_tensor_by_name(
            "values", self._value_preprocessor(values, train=True)
        )
        self.memory.set_tensor_by_name(
            "returns", self._value_preprocessor(returns, train=True)
        )
        self.memory.set_tensor_by_name("advantages", advantages)
        for accumulator in self._metric_accumulators.values():
            accumulator.zero_()

        for epoch in range(self.cfg.learning_epochs):
            batches = self.memory.sample(
                names=self._tensors_names,
                batch_size=len(self.memory),
                mini_batches=self.cfg.mini_batches,
            )
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in batches:
                inputs = {
                    "observations": self._observation_preprocessor(
                        sampled_observations, train=not epoch
                    ),
                    "states": self._state_preprocessor(sampled_states, train=not epoch),
                }
                _, outputs = self.policy.act(
                    {**inputs, "taken_actions": sampled_actions}, role="policy"
                )
                next_log_prob = outputs["log_prob"]
                if self.cfg.entropy_loss_scale:
                    entropy_loss = (
                        -self.cfg.entropy_loss_scale
                        * self.policy.get_entropy(role="policy").mean()
                    )
                else:
                    entropy_loss = torch.zeros((), device=self.device)
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio,
                    1.0 - self.cfg.ratio_clip,
                    1.0 + self.cfg.ratio_clip,
                )
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
                predicted_values, _ = self.value.act(inputs, role="value")
                if self.cfg.value_clip > 0:
                    predicted_values = sampled_values + torch.clip(
                        predicted_values - sampled_values,
                        min=-self.cfg.value_clip,
                        max=self.cfg.value_clip,
                    )
                value_loss = self.cfg.value_loss_scale * F.mse_loss(
                    sampled_returns, predicted_values
                )

                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                if skrl_config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()
                if self.cfg.grad_norm_clip > 0:
                    parameters = (
                        self.policy.parameters()
                        if self.policy is self.value
                        else itertools.chain(
                            self.policy.parameters(), self.value.parameters()
                        )
                    )
                    nn.utils.clip_grad_norm_(parameters, self.cfg.grad_norm_clip)
                self.optimizer.step()
                with torch.no_grad():
                    self._metric_accumulators["policy_loss"].add_(policy_loss.detach())
                    self._metric_accumulators["value_loss"].add_(value_loss.detach())
                    self._metric_accumulators["entropy_loss"].add_(
                        entropy_loss.detach()
                    )

        with torch.no_grad():
            self._metric_accumulators["stddev"].copy_(
                self.policy.distribution(role="policy").stddev.mean().detach()
            )

    def track_data(self, tag: str, value: float) -> None:
        """禁用 stock Python list tracking；显式指标只从 cold export 获取。"""

        del tag, value

    def export_training_metrics(self) -> dict[str, float]:
        """低频冷边界：同步导出四个聚合标量并原地归零。"""

        metrics = {
            name: float(value.detach().cpu().item())
            for name, value in self._metric_accumulators.items()
        }
        for value in self._metric_accumulators.values():
            value.zero_()
        return metrics


def _compute_gae_device(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float,
    lambda_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """reward 已完成 time-limit bootstrap；两类 done 都切断跨 episode GAE。"""

    advantage = torch.zeros_like(last_values)
    advantages = torch.zeros_like(rewards)
    not_done = ~(terminated | truncated)
    for index in reversed(range(rewards.shape[0])):
        next_values = values[index + 1] if index < rewards.shape[0] - 1 else last_values
        advantage = (
            rewards[index]
            - values[index]
            + discount_factor
            * not_done[index]
            * (next_values + lambda_coefficient * advantage)
        )
        advantages[index] = advantage
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    return returns, advantages


def _cuda_tensor(
    value: object,
    name: str,
    *,
    device: torch.device,
    ndim: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device != device or value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim} on {device}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use dtype={dtype}")
    return value


def _column(value: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    value = _cuda_tensor(value, name, device=device, ndim=value.ndim)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape (N,) or (N,1)")
    return value


def _done_column(value: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    result = _column(value, name, device)
    if result.dtype != torch.bool:
        raise TypeError(f"{name} must use bool dtype")
    return result


__all__ = ["FinalObservationPPO", "validate_skrl_ppo_source"]
