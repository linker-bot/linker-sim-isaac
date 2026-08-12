"""Gymnasium 1.3 NumPy VectorEnv 边界。

所有 D2H/H2D 转换都集中在本模块，方便 import-closure 和 profiler gate 证明 native/skrl 路径
不会经过 NumPy。默认 autoreset 为 DISABLED；可显式选择 SAME_STEP，绝不实现 NEXT_STEP。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from linkerbot_sim.kaleidoscope.env import TorchKaleidoscopeEnv


def _gymnasium_types():
    try:
        import gymnasium as gym
        from gymnasium.vector import AutoresetMode
        from gymnasium.vector.utils import batch_space
    except ImportError as exc:
        raise RuntimeError(
            "Gymnasium adapter requires the project 'training' extra"
        ) from exc
    return gym, AutoresetMode, batch_space


try:
    _gym, _AutoresetMode, _batch_space = _gymnasium_types()
    _VectorEnvBase = _gym.vector.VectorEnv
except RuntimeError:
    # 模块仍可被文档工具导入；实际构造时会由 __init__ 给出明确 dependency error。
    _gym = None
    _AutoresetMode = None
    _batch_space = None

    class _VectorEnvBase:  # type: ignore[no-redef]
        pass


class GymnasiumKaleidoscopeAdapter(_VectorEnvBase):
    """把一个 native Torch 环境暴露为严格 Gymnasium VectorEnv。"""

    def __init__(
        self,
        env: TorchKaleidoscopeEnv,
        *,
        autoreset_mode: str = "disabled",
        render_mode: str | None = None,
    ) -> None:
        gym, AutoresetMode, batch_space = _gymnasium_types()
        super().__init__()
        if autoreset_mode not in {"disabled", "same_step"}:
            raise ValueError("autoreset_mode must be disabled or same_step")
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")
        if render_mode == "human" and not env.viewport_enabled:
            raise ValueError(
                "render_mode='human' requires a Kaleidoscope viewport environment"
            )
        self.env = env
        self.num_envs = env.num_envs
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.observation_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=env.action_low,
            high=env.action_high,
            shape=(env.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.render_mode = render_mode
        self.metadata = {
            "render_modes": ["human"],
            "autoreset_mode": (
                AutoresetMode.SAME_STEP
                if autoreset_mode == "same_step"
                else AutoresetMode.DISABLED
            ),
        }
        self._same_step = autoreset_mode == "same_step"
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options is None:
            observations, info = self.env.reset(seed=seed)
            return _to_numpy(observations), _info_to_numpy(info)
        unknown = set(options) - {"reset_mask"}
        if unknown or "reset_mask" not in options:
            raise ValueError(f"unsupported Gymnasium reset options: {sorted(unknown)}")
        mask = np.asarray(options["reset_mask"])
        if mask.dtype != np.bool_ or mask.shape != (self.num_envs,):
            raise ValueError("reset_mask must be bool ndarray with shape (num_envs,)")
        import torch

        mask_device = torch.as_tensor(mask, device=self.env.device, dtype=torch.bool)
        ids = torch.nonzero(mask_device, as_tuple=False).flatten()
        if ids.numel() > 0:
            self.env.reset_idx(ids)
        observations = self.env.runtime.task.buffers.last_finite_observation.clone()
        return _to_numpy(observations), {"reset_mask": mask.copy()}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        import torch

        values = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.env.action_dim)
        if values.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("actions must contain finite values")
        device_actions = torch.as_tensor(
            values, device=self.env.device, dtype=torch.float32
        )
        observations, rewards, terminated, truncated, info = self.env.step(
            device_actions
        )
        if self._same_step:
            done = terminated | truncated
            final_observation = observations.clone()
            terminal_info: dict[str, object] = {}
            active_info = dict(info)
            for name, value in tuple(active_info.items()):
                terminal_info[name] = (
                    value.clone() if isinstance(value, torch.Tensor) else value
                )
                terminal_info[f"_{name}"] = done
                # done 行返回的是 reset observation；原 step info 只对未 reset 行有效。
                active_info[f"_{name}"] = ~done
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            observations = observations.clone()
            if done_ids.numel() > 0:
                reset_observation, _ = self.env.reset_idx(done_ids)
                observations.index_copy_(0, done_ids, reset_observation)
            info = active_info
            info["final_obs"] = final_observation
            info["_final_obs"] = done
            info["final_info"] = terminal_info
            info["_final_info"] = done
        return (
            _to_numpy(observations),
            _to_numpy(rewards),
            _to_numpy(terminated),
            _to_numpy(truncated),
            _info_to_numpy(info),
        )

    def close_extras(self, **_kwargs: Any) -> None:
        self.env.close()

    def render(self) -> None:
        if self.render_mode != "human":
            raise RuntimeError("Gymnasium render requires render_mode='human'")
        self.env.render()


def _to_numpy(value: object) -> np.ndarray:
    """唯一允许的整批 CUDA→CPU→NumPy 转换函数。"""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    return value.detach().cpu().numpy()


def _info_to_numpy(info: Mapping[str, object]) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for name, value in info.items():
        if isinstance(value, torch.Tensor):
            result[name] = _to_numpy(value)
        elif isinstance(value, Mapping):
            result[name] = _info_to_numpy(value)
        else:
            result[name] = value
    return result


__all__ = ["GymnasiumKaleidoscopeAdapter"]
