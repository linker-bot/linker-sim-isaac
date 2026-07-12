from __future__ import annotations

from pathlib import Path

import pytest

from linkerbot_sim.logging.config import (
    JointLoggingConfig,
    joint_logging_config_from_mapping,
    load_joint_logging_profile,
)
from linkerbot_sim.utils.config import load_yaml


def _profile(logging: dict[str, object]) -> dict[str, object]:
    return {"logging": logging}


def test_all_bundled_logging_yaml_loads_strictly() -> None:
    paths = sorted(Path("configs/logging").glob("*.yaml"))

    assert paths
    for path in paths:
        data = load_yaml(path)
        config = joint_logging_config_from_mapping(data, source_path=path)
        assert isinstance(config, JointLoggingConfig)


def test_default_logging_profile_matches_runtime_consumer_contract() -> None:
    config = load_joint_logging_profile("default_logger")

    assert config == JointLoggingConfig(
        enabled=False,
        joint_tracking_path=Path("logs/joint_tracking/pinch_grasp.csv"),
        flush_interval_s=0.2,
        interval_steps=5,
        log_actual_position=True,
        log_actual_velocity=True,
        log_command_position=True,
        log_command_velocity=True,
        log_command_effort=True,
        log_action_effort=False,
        log_measured_effort=False,
        log_applied_effort=False,
    )
    assert not config.should_write_step(0)
    assert config.flush_interval_steps(0.01) == 20


@pytest.mark.parametrize("name", ("../default", "a/b", r"a\b", "", "."))
def test_logging_profile_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="logging profile name"):
        load_joint_logging_profile(name, logging_root=tmp_path)


def test_logging_profile_reports_missing_named_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        load_joint_logging_profile("missing", logging_root=tmp_path)


@pytest.mark.parametrize(
    ("data", "path"),
    (
        ({"loging": {}}, "logging profile.loging"),
        (_profile({"interval_step": 2}), "logging.interval_step"),
        (_profile({"actual_position": False}), "logging.actual_position"),
    ),
)
def test_logging_unknown_key_reports_complete_path(
    data: dict[str, object], path: str
) -> None:
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        joint_logging_config_from_mapping(data)


@pytest.mark.parametrize("value", ("false", 0, 1, None, [], {}))
def test_logging_boolean_fields_are_strict(value: object) -> None:
    with pytest.raises(ValueError, match=r"logging\.enabled"):
        joint_logging_config_from_mapping(_profile({"enabled": value}))


@pytest.mark.parametrize("value", (True, "0.2", 0, -0.1, float("nan"), float("inf")))
def test_logging_flush_interval_requires_positive_finite_number(value: object) -> None:
    with pytest.raises(ValueError, match=r"logging\.flush_interval_s"):
        joint_logging_config_from_mapping(_profile({"flush_interval_s": value}))


@pytest.mark.parametrize("value", (True, "5", 0, -1, 1.5, None))
def test_logging_interval_steps_requires_strict_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match=r"logging\.interval_steps"):
        joint_logging_config_from_mapping(_profile({"interval_steps": value}))


@pytest.mark.parametrize("value", ("", "   ", 3, False, []))
def test_logging_output_path_is_strict(value: object) -> None:
    with pytest.raises(ValueError, match=r"logging\.joint_tracking_path"):
        joint_logging_config_from_mapping(_profile({"joint_tracking_path": value}))


def test_logging_nullable_output_path_is_preserved() -> None:
    config = joint_logging_config_from_mapping(_profile({"joint_tracking_path": None}))

    assert config.joint_tracking_path is None
