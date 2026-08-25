from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from linkerbot_sim.configuration.catalog import load_yaml_mapping
from linkerbot_sim.configuration.common import ConfigurationError
from linkerbot_sim.configuration.outputs import (
    CameraOutputSettings,
    TelemetryOutputSettings,
)


OUTPUT_PROFILE = Path("configs/outputs/mirror_default.yaml")


def _output_section(name: str) -> dict[str, object]:
    document = load_yaml_mapping(OUTPUT_PROFILE)
    outputs = document["outputs"]
    assert isinstance(outputs, Mapping)
    section = outputs[name]
    assert isinstance(section, Mapping)
    return dict(section)


def test_disabled_camera_output_may_retain_consumer_settings() -> None:
    raw = _output_section("camera")
    raw.update(
        enabled=False,
        foxglove_mcap_path="logs/cameras/mirror.mcap",
    )

    settings = CameraOutputSettings.from_mapping(raw, label="outputs.camera")

    assert settings.enabled is False
    assert settings.save_root == "logs/cameras"
    assert settings.foxglove_live_port == 8849
    assert settings.foxglove_mcap_path == "logs/cameras/mirror.mcap"


def test_enabled_camera_output_requires_a_consumer() -> None:
    raw = _output_section("camera")
    raw.update(
        enabled=True,
        save_root=None,
        foxglove_live_port=None,
        foxglove_mcap_path=None,
    )

    with pytest.raises(ConfigurationError, match="enabled=true"):
        CameraOutputSettings.from_mapping(raw, label="outputs.camera")


def test_disabled_telemetry_may_retain_endpoint_settings() -> None:
    raw = _output_section("telemetry")
    raw["enabled"] = False

    settings = TelemetryOutputSettings.from_mapping(raw, label="outputs.telemetry")

    assert settings.enabled is False
    assert settings.foxglove_live_port == 8848
    assert settings.mcap_path == "logs/telemetry/mirror.mcap"


def test_enabled_telemetry_requires_an_output_endpoint() -> None:
    raw = _output_section("telemetry")
    raw.update(enabled=True, foxglove_live_port=None, mcap_path=None)

    with pytest.raises(ConfigurationError, match="enabled=true"):
        TelemetryOutputSettings.from_mapping(raw, label="outputs.telemetry")


def test_disabled_effort_output_may_retain_selected_source() -> None:
    raw = _output_section("telemetry")
    raw.update(include_efforts=False, joint_effort_field="measured")

    settings = TelemetryOutputSettings.from_mapping(raw, label="outputs.telemetry")

    assert settings.include_efforts is False
    assert settings.joint_effort_field == "measured"


def test_enabled_effort_output_requires_selected_source() -> None:
    raw = _output_section("telemetry")
    raw.update(include_efforts=True, joint_effort_field="none")

    with pytest.raises(ConfigurationError, match="include_efforts=true"):
        TelemetryOutputSettings.from_mapping(raw, label="outputs.telemetry")


def test_hybrid_control_can_be_the_only_telemetry_modality() -> None:
    raw = _output_section("telemetry")
    raw.update(
        include_joint_states=False,
        include_state_json=False,
        include_scene_markers=False,
        include_hybrid_control=True,
    )

    settings = TelemetryOutputSettings.from_mapping(raw, label="outputs.telemetry")

    assert settings.include_hybrid_control is True
    assert settings.topics.hybrid_control == "/linkerbot/mirror/hybrid_control"


def test_hybrid_control_topic_must_not_collide() -> None:
    raw = _output_section("telemetry")
    topics = dict(raw["topics"])  # type: ignore[arg-type]
    topics["hybrid_control"] = topics["state"]
    raw["topics"] = topics

    with pytest.raises(ConfigurationError, match="must be distinct"):
        TelemetryOutputSettings.from_mapping(raw, label="outputs.telemetry")
