from __future__ import annotations

from dataclasses import asdict

import pytest

from linkerbot_sim.configuration.common import ConfigurationError
from linkerbot_sim.configuration.outputs import LoggingOutputSettings


def _logging_settings(**updates: object) -> LoggingOutputSettings:
    values = {
        "enabled": True,
        "existing_data_policy": "error",
        "joint_tracking_path": "logs/joint_tracking/test.csv",
        "flush_interval_s": 0.05,
        "interval_steps": 1,
        "log_actual_position": True,
        "log_actual_velocity": True,
        "log_command_position": True,
        "log_command_velocity": True,
        "log_command_effort": True,
        "log_action_effort": False,
        "log_measured_effort": False,
        "log_applied_effort": False,
    }
    values.update(updates)
    return LoggingOutputSettings(**values)


@pytest.mark.parametrize(
    ("step", "expected"),
    ((0, True), (1, False), (4, False), (5, True), (10, True)),
)
def test_joint_logging_sampling_interval(step: int, expected: bool) -> None:
    settings = _logging_settings(interval_steps=5)

    assert settings.should_write_step(step) is expected


def test_disabled_joint_logging_never_writes() -> None:
    settings = _logging_settings(
        enabled=False,
        joint_tracking_path=None,
        interval_steps=1,
    )

    assert not settings.should_write_step(0)
    assert not settings.should_write_step(1)


def test_disabled_joint_logging_may_retain_configured_path() -> None:
    raw = asdict(_logging_settings(enabled=False))

    settings = LoggingOutputSettings.from_mapping(raw, label="outputs.logging")

    assert settings.joint_tracking_path == "logs/joint_tracking/test.csv"
    assert not settings.should_write_step(0)


def test_enabled_joint_logging_requires_path() -> None:
    raw = asdict(_logging_settings(joint_tracking_path=None))

    with pytest.raises(ConfigurationError, match="joint_tracking_path.*enabled=true"):
        LoggingOutputSettings.from_mapping(raw, label="outputs.logging")


@pytest.mark.parametrize(
    ("physics_dt", "expected"),
    ((0.01, 5), (0.02, 2), (0.1, 1), (0.0, 1), (-0.01, 1)),
)
def test_flush_interval_is_projected_to_physics_steps(
    physics_dt: float, expected: int
) -> None:
    settings = _logging_settings(flush_interval_s=0.05)

    assert settings.flush_interval_steps(physics_dt) == expected


def test_existing_data_policy_belongs_to_logging_settings() -> None:
    settings = _logging_settings(existing_data_policy="resume")

    assert settings.existing_data_policy == "resume"


def test_hybrid_logging_requires_master_switch_and_path() -> None:
    disabled = asdict(
        _logging_settings(
            enabled=False,
            log_hybrid_control=True,
            hybrid_control_path="logs/hybrid.csv",
        )
    )
    missing_path = asdict(
        _logging_settings(log_hybrid_control=True, hybrid_control_path=None)
    )

    with pytest.raises(
        ConfigurationError, match="requires enabled=true|要求 enabled=true"
    ):
        LoggingOutputSettings.from_mapping(disabled, label="outputs.logging")
    with pytest.raises(ConfigurationError, match="hybrid_control_path"):
        LoggingOutputSettings.from_mapping(missing_path, label="outputs.logging")
