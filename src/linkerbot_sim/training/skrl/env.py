"""skrl trainer 必经的 CUDA SAME_STEP 环境 adapter。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import torch
from skrl.envs.wrappers.torch import Wrapper

from linkerbot_sim.kaleidoscope import KaleidoscopeTrainingPort


class SkrlTorchAdapter(Wrapper):
    """保存 terminal transition 后，在同一拍 reset done rows。

    skrl 2.1 的 vector trainer 不会替 done 行调用 reset。adapter 因此通过 runtime generation token
    完成一次不可遗漏的握手，并把 terminal observation/mask/info 保存在 owned CUDA buffer 中。
    该 adapter 只借用环境，不拥有 IsaacSession；``close`` 只能沿 public training port 请求产品
    runtime 关闭，不能直接释放 physics view 或 App。
    """

    def __init__(self, env: KaleidoscopeTrainingPort) -> None:
        if not isinstance(env, KaleidoscopeTrainingPort):
            raise TypeError("env must implement KaleidoscopeTrainingPort")
        if env.num_envs < 2:
            raise ValueError(
                "SkrlTorchAdapter requires num_envs >= 2; use native Torch env for N=1"
            )
        super().__init__(env)
        self._native = env
        self._observation_space = gym.spaces.Box(
            -float("inf"),
            float("inf"),
            shape=(env.observation_dim,),
            dtype="float32",
        )
        self._action_space = gym.spaces.Box(
            env.action_low,
            env.action_high,
            shape=(env.action_dim,),
            dtype="float32",
        )
        shape = (env.num_envs, env.observation_dim)
        self._post_reset_observation = torch.zeros(
            shape, device=env.device, dtype=torch.float32
        )
        self._final_observation = torch.zeros_like(self._post_reset_observation)
        self._final_mask = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        self._reward = torch.zeros(
            (env.num_envs, 1), device=env.device, dtype=torch.float32
        )
        self._terminated = torch.zeros(
            (env.num_envs, 1), device=env.device, dtype=torch.bool
        )
        self._truncated = torch.zeros_like(self._terminated)
        self._step_info_buffers: dict[str, torch.Tensor] = {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def state_space(self) -> gym.Space:
        return self._observation_space

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        observation, info = self._native.reset()
        self._post_reset_observation.copy_(observation)
        return self._post_reset_observation, dict(info)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # 训练层只校验自己消费的 public tensor 形状；不导入 Kaleidoscope 内部 helper，
        # 也绝不通过 as_tensor/to 隐式搬运 CPU 数据。
        if not isinstance(actions, torch.Tensor):
            raise TypeError("skrl actions must be a torch.Tensor")
        action = actions
        if action.device.type != "cuda":
            raise ValueError("skrl actions must live on CUDA")
        if action.ndim != 2 or action.shape[0] != self.num_envs:
            raise ValueError("skrl actions have the wrong batch shape")
        if action.dtype != torch.float32:
            raise TypeError("skrl actions must use torch.float32")
        if action.device != self.device or action.shape[1] != self._native.action_dim:
            raise ValueError("skrl actions have the wrong CUDA device/shape")
        token = self._native.begin_same_step()
        observations, rewards, terminated, truncated, step_info = (
            self._native.step_same_step(token, action)
        )

        # 先复制 terminal transition，再允许 reset 改写 task/runtime buffer。
        self._final_observation.copy_(observations)
        self._reward.copy_(rewards[:, None])
        self._terminated.copy_(terminated[:, None])
        self._truncated.copy_(truncated[:, None])
        self._final_mask.copy_(terminated | truncated)
        info = self._copy_step_info(step_info)
        post_reset_observation = self._native.complete_same_step(token)
        self._post_reset_observation.copy_(post_reset_observation)

        # info 保持 dense CUDA tensor，不逐 env 组装 Python dict/list。
        info.update(
            {
                "final_obs": self._final_observation,
                "_final_obs": self._final_mask,
            }
        )
        return (
            self._post_reset_observation,
            self._reward,
            self._terminated,
            self._truncated,
            info,
        )

    def _copy_step_info(self, step_info: object) -> dict[str, torch.Tensor]:
        """在 SAME_STEP reset 前保存 task 借出的逐环境 CUDA tensor。"""

        if not isinstance(step_info, Mapping):
            raise TypeError("skrl step info must be a tensor mapping")
        owned: dict[str, torch.Tensor] = {}
        for key, value in step_info.items():
            if not isinstance(key, str):
                raise TypeError("skrl step info keys must be strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"skrl step info {key!r} must be a torch.Tensor")
            if value.device != self.device:
                raise ValueError(f"skrl step info {key!r} must live on {self.device}")
            if value.ndim < 1 or value.shape[0] != self.num_envs:
                raise ValueError(
                    f"skrl step info {key!r} must have leading shape ({self.num_envs},)"
                )
            buffer = self._step_info_buffers.get(key)
            if (
                buffer is None
                or buffer.shape != value.shape
                or buffer.dtype != value.dtype
                or buffer.device != value.device
            ):
                buffer = torch.empty_like(value)
                self._step_info_buffers[key] = buffer
            buffer.copy_(value)
            owned[key] = buffer
        return owned

    def state(self) -> torch.Tensor:
        return self._post_reset_observation

    def render(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Kaleidoscope is headless")

    def close(self) -> None:
        self._native.close()


__all__ = ["SkrlTorchAdapter"]
