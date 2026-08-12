"""``configs/training/skrl`` 对应的严格训练配置。

本模块只解释已经由 configuration catalog 读取的 raw mapping。训练工厂接收
``SkrlTrainingSettings``，不再拥有路径、默认 profile 或 YAML 解析职责。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..common import (
    ConfigurationError,
    as_float,
    as_int,
    as_string,
    require_keys,
    strict_mapping,
)


@dataclass(frozen=True, slots=True)
class SkrlTrainingSettings:
    """已严格校验的 skrl 训练叶配置；设备始终继承 environment。"""

    framework: Literal["skrl"]
    algorithm: Literal["final_observation_ppo"]
    device_source: Literal["environment"]
    rollout_length: int
    mini_batches: int
    learning_epochs: int
    learning_rate: float
    discount_factor: float
    gae_lambda: float
    clip_ratio: float

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        label: str = "training",
    ) -> "SkrlTrainingSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "framework",
            "algorithm",
            "device_source",
            "rollout_length",
            "mini_batches",
            "learning_epochs",
            "learning_rate",
            "discount_factor",
            "gae_lambda",
            "clip_ratio",
        }
        require_keys(mapping, required=required, label=label)
        rollout_length = as_int(
            mapping["rollout_length"],
            label=f"{label}.rollout_length",
            minimum=1,
        )
        mini_batches = as_int(
            mapping["mini_batches"],
            label=f"{label}.mini_batches",
            minimum=1,
        )
        if mini_batches > rollout_length:
            raise ConfigurationError(
                f"{label}.mini_batches 不能大于 {label}.rollout_length"
            )
        return cls(
            framework=as_string(
                mapping["framework"],
                label=f"{label}.framework",
                choices={"skrl"},
            ),  # type: ignore[arg-type]
            algorithm=as_string(
                mapping["algorithm"],
                label=f"{label}.algorithm",
                choices={"final_observation_ppo"},
            ),  # type: ignore[arg-type]
            device_source=as_string(
                mapping["device_source"],
                label=f"{label}.device_source",
                choices={"environment"},
            ),  # type: ignore[arg-type]
            rollout_length=rollout_length,
            mini_batches=mini_batches,
            learning_epochs=as_int(
                mapping["learning_epochs"],
                label=f"{label}.learning_epochs",
                minimum=1,
            ),
            learning_rate=as_float(
                mapping["learning_rate"],
                label=f"{label}.learning_rate",
                strictly_positive=True,
            ),
            discount_factor=as_float(
                mapping["discount_factor"],
                label=f"{label}.discount_factor",
                minimum=0.0,
                maximum=1.0,
            ),
            gae_lambda=as_float(
                mapping["gae_lambda"],
                label=f"{label}.gae_lambda",
                minimum=0.0,
                maximum=1.0,
            ),
            clip_ratio=as_float(
                mapping["clip_ratio"],
                label=f"{label}.clip_ratio",
                strictly_positive=True,
            ),
        )


__all__ = ["SkrlTrainingSettings"]
