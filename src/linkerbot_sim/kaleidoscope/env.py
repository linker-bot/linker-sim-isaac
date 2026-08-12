"""训练框架无关的 TorchKaleidoscopeEnv。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from linkerbot_sim.kaleidoscope.runtime import KaleidoscopeRuntime, SameStepToken

if TYPE_CHECKING:
    import torch


def _runtime_token(value: object) -> SameStepToken:
    """在产品内部恢复具体 token 类型，public training port 仍只看到 object。"""

    if not isinstance(value, SameStepToken):
        raise TypeError("invalid Kaleidoscope SAME_STEP token")
    return value


class TorchKaleidoscopeEnv:
    """直接返回 CUDA tensor 的高性能入口。

    它刻意不继承 Gymnasium ``VectorEnv``，因为 Gymnasium 的标准边界是 NumPy。需要生态兼容时
    使用外层 ``GymnasiumKaleidoscopeAdapter``；skrl 则使用 training 层 SAME_STEP adapter。
    """

    def __init__(self, runtime: KaleidoscopeRuntime) -> None:
        self.runtime = runtime
        self.num_envs = runtime.num_envs
        self.device = runtime.device
        self.action_dim = runtime.action_dim
        self.action_low = runtime.action_low
        self.action_high = runtime.action_high
        self.observation_dim = runtime.observation_dim
        self.viewport_enabled = runtime.viewport_enabled
        self.render_every_n_steps = runtime.render_every_n_steps
        self._closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple["torch.Tensor", Mapping[str, "torch.Tensor"]]:
        """重置全部环境；seed 只允许在构造前/显式 reseed 冷边界设置。"""

        if seed is not None:
            self.runtime.reseed(seed)
        if options:
            raise ValueError(
                "TorchKaleidoscopeEnv does not accept dynamic reset options"
            )
        return self.runtime.reset()

    def reset_idx(
        self, env_ids: "torch.Tensor"
    ) -> tuple["torch.Tensor", Mapping[str, "torch.Tensor"]]:
        return self.runtime.reset_idx(env_ids)

    def step(
        self, actions: "torch.Tensor"
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        Mapping[str, "torch.Tensor"],
    ]:
        result = self.runtime.step(actions)
        return (
            result.observations,
            result.rewards,
            result.terminated,
            result.truncated,
            result.info,
        )

    def get_state(self, *args: object, **kwargs: object):
        return self.runtime.get_state(*args, **kwargs)

    def set_state(self, *args: object, **kwargs: object) -> None:
        self.runtime.set_state(*args, **kwargs)

    def snapshot(self, *args: object, **kwargs: object):
        return self.runtime.snapshot(*args, **kwargs)

    def restore_snapshot(self, *args: object, **kwargs: object) -> None:
        self.runtime.restore_snapshot(*args, **kwargs)

    def clone_state(self, *args: object, **kwargs: object) -> None:
        self.runtime.clone_state(*args, **kwargs)

    def get_control_mode(self):
        return self.runtime.get_control_mode()

    def set_control_mode(self, *args: object, **kwargs: object):
        return self.runtime.set_control_mode(*args, **kwargs)

    def render(self) -> None:
        """显式刷新 human viewport；headless 环境给出明确错误。"""

        self.runtime.render()

    def is_running(self) -> bool:
        """返回 viewport 窗口是否仍打开。"""

        return self.runtime.is_running()

    def begin_same_step(self) -> object:
        """为训练 adapter 开启一拍事务；token 对产品边界保持不透明。"""

        return self.runtime.issue_same_step_token()

    def step_same_step(
        self, token: object, actions: "torch.Tensor"
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        Mapping[str, "torch.Tensor"],
    ]:
        """执行 token 所属 decision，返回 reset 前的 terminal transition。"""

        # Runtime 是 token 真实性的唯一 owner；env 只做 public tuple 形状适配，不把
        # TaskStepResult 或 SameStepToken 类型泄漏给 training package。
        result = self.runtime.tokenized_step(_runtime_token(token), actions)
        return (
            result.observations,
            result.rewards,
            result.terminated,
            result.truncated,
            result.info,
        )

    def complete_same_step(self, token: object) -> "torch.Tensor":
        """完成 SAME_STEP reset，并暴露整批 post-reset observation。"""

        self.runtime.complete_same_step_reset(_runtime_token(token))
        return self.runtime.task.buffers.last_finite_observation

    def close(self, *, exit_code: int = 0) -> None:
        if self._closed:
            return
        self.runtime.close(exit_code=exit_code)
        self._closed = True

    def __enter__(self) -> "TorchKaleidoscopeEnv":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["TorchKaleidoscopeEnv"]
