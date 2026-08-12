"""Kaleidoscope 到 skrl 2.1 的唯一 trainer composition root。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg

from linkerbot_sim.configuration.training.skrl import SkrlTrainingSettings
from linkerbot_sim.kaleidoscope import KaleidoscopeTrainingPort
from linkerbot_sim.training.skrl.env import SkrlTorchAdapter
from linkerbot_sim.training.skrl.final_observation_ppo import FinalObservationPPO
from linkerbot_sim.training.skrl.memory import CudaRolloutMemory

if TYPE_CHECKING:
    from skrl.models.torch import Model


def make_skrl_trainer(
    env: KaleidoscopeTrainingPort,
    *,
    models: Mapping[str, Model],
    settings: SkrlTrainingSettings,
    timesteps: int,
    close_environment_at_exit: bool = True,
    disable_progressbar: bool = False,
) -> SequentialTrainer:
    """构造只走 CUDA SAME_STEP 路径的 skrl trainer。

    模型和已解析训练设置由调用方提供，因为网络结构与 profile 选择属于实验组合。
    工厂负责固定容易出错的其余组合：环境 adapter、rollout memory、
    final-observation PPO、禁用热路径日志/自动 checkpoint，以及唯一由
    ``env.device`` 派生的设备。
    """

    if not isinstance(env, KaleidoscopeTrainingPort):
        raise TypeError("env must implement KaleidoscopeTrainingPort")
    if not isinstance(settings, SkrlTrainingSettings):
        raise TypeError("settings must be SkrlTrainingSettings")
    if type(timesteps) is not int or timesteps <= 0:
        raise ValueError("timesteps must be a positive integer")
    if type(close_environment_at_exit) is not bool:
        raise TypeError("close_environment_at_exit must be boolean")
    if type(disable_progressbar) is not bool:
        raise TypeError("disable_progressbar must be boolean")
    if set(models) != {"policy", "value"}:
        raise ValueError("models must contain exactly 'policy' and 'value'")

    adapter = SkrlTorchAdapter(env)
    _validate_model_spaces(models, adapter=adapter)
    memory = CudaRolloutMemory(
        memory_size=settings.rollout_length,
        num_envs=env.num_envs,
        device=env.device,
    )
    agent = FinalObservationPPO(
        models=dict(models),
        memory=memory,
        observation_space=adapter.observation_space,
        state_space=adapter.state_space,
        action_space=adapter.action_space,
        device=env.device,
        cfg=PPO_CFG(
            rollouts=settings.rollout_length,
            learning_epochs=settings.learning_epochs,
            mini_batches=settings.mini_batches,
            learning_rate=settings.learning_rate,
            discount_factor=settings.discount_factor,
            gae_lambda=settings.gae_lambda,
            ratio_clip=settings.clip_ratio,
            time_limit_bootstrap=True,
            kl_threshold=0.0,
            rewards_shaper=None,
            mixed_precision=False,
            experiment=ExperimentCfg(write_interval=0, checkpoint_interval=0),
        ),
    )
    return SequentialTrainer(
        env=adapter,
        agents=agent,
        cfg=SequentialTrainerCfg(
            timesteps=timesteps,
            headless=True,
            disable_progressbar=disable_progressbar,
            close_environment_at_exit=close_environment_at_exit,
            # skrl 默认读取 infos["episode"] 并触发逐拍 host 聚合。这里使用一个
            # adapter 明确禁止产生的保留 key，使训练热路径保持 CUDA 常驻。
            environment_info="__disabled__",
        ),
    )


def _validate_model_spaces(
    models: Mapping[str, Model], *, adapter: SkrlTorchAdapter
) -> None:
    """在分配 rollout/agent 前冻结 model 的设备与 Box tensor 合同。"""

    expected = {
        "observation_space": adapter.observation_space,
        "state_space": adapter.state_space,
        "action_space": adapter.action_space,
    }
    for model_name in ("policy", "value"):
        model = models[model_name]
        if model.device != adapter.device:
            raise ValueError(
                f"{model_name}.device must match the environment: "
                f"{model.device} != {adapter.device}"
            )
        for field_name, expected_space in expected.items():
            actual_space = getattr(model, field_name, None)
            label = f"{model_name}.{field_name}"
            if not isinstance(actual_space, type(expected_space)):
                raise TypeError(f"{label} must be a Gymnasium Box")
            if actual_space.shape != expected_space.shape:
                raise ValueError(
                    f"{label} shape must match the environment: "
                    f"{actual_space.shape} != {expected_space.shape}"
                )
            if actual_space.dtype != expected_space.dtype:
                raise TypeError(
                    f"{label} dtype must match the environment: "
                    f"{actual_space.dtype} != {expected_space.dtype}"
                )
            if not bool((actual_space.low == expected_space.low).all()) or not bool(
                (actual_space.high == expected_space.high).all()
            ):
                raise ValueError(f"{label} bounds must match the environment")


__all__ = ["make_skrl_trainer"]
