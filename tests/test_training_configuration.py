from __future__ import annotations

from pathlib import Path

import pytest

from linkerbot_sim.configuration.common import ConfigurationError
from linkerbot_sim.configuration.training.skrl import (
    SkrlTrainingSettings,
)
from linkerbot_sim.utils.config import load_yaml


PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "training"
    / "skrl"
    / "tblock_push_v1_ppo.yaml"
)


def test_default_skrl_profile_is_strictly_typed() -> None:
    settings = SkrlTrainingSettings.from_mapping(load_yaml(PROFILE)["training"])

    assert isinstance(settings, SkrlTrainingSettings)
    assert settings.framework == "skrl"
    assert settings.algorithm == "final_observation_ppo"
    assert settings.device_source == "environment"
    assert settings.rollout_length == 32
    assert settings.mini_batches == 8


def test_skrl_settings_reject_more_batches_than_rollout() -> None:
    with pytest.raises(ConfigurationError, match="mini_batches 不能大于"):
        SkrlTrainingSettings.from_mapping(
            {
                "framework": "skrl",
                "algorithm": "final_observation_ppo",
                "device_source": "environment",
                "rollout_length": 2,
                "mini_batches": 3,
                "learning_epochs": 1,
                "learning_rate": 3e-4,
                "discount_factor": 0.99,
                "gae_lambda": 0.95,
                "clip_ratio": 0.2,
            }
        )


def test_skrl_schema_rejects_unknown_yaml_fields(tmp_path: Path) -> None:
    profile = tmp_path / "invalid.yaml"
    profile.write_text(
        """training:
  framework: skrl
  algorithm: final_observation_ppo
  device_source: environment
  rollout_length: 32
  mini_batches: 4
  learning_epochs: 4
  learning_rate: 0.0003
  discount_factor: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  typo: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="包含未知字段: typo"):
        SkrlTrainingSettings.from_mapping(load_yaml(profile)["training"])
