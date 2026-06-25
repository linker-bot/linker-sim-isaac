from __future__ import annotations

import csv

import numpy as np

from manipulation_project.logging.config import (
    JointLoggingConfig,
    joint_logging_config_from_mapping,
)
from manipulation_project.logging.effort_logger import (
    EffortLogger,
    commanded_efforts_from_controller,
    read_joint_efforts,
)
from manipulation_project.logging.joint_logger import JointTrackingLogger


class _RobotWithEfforts:
    num_dof = 3
    dof_names = ["j0", "j1", "j2"]

    def __init__(self) -> None:
        self.position_reads = 0
        self.velocity_reads = 0
        self.measured_reads = 0
        self.applied_reads = 0

    def get_joint_positions(self):
        self.position_reads += 1
        return np.asarray([0.1, 0.2, 0.3], dtype=float)

    def get_joint_velocities(self):
        self.velocity_reads += 1
        return np.asarray([1.1, 1.2, 1.3], dtype=float)

    def get_measured_joint_efforts(self, joint_indices=None):
        self.measured_reads += 1
        values = np.asarray([1.0, 2.0, 3.0], dtype=float)
        return (
            values
            if joint_indices is None
            else values[np.asarray(joint_indices, dtype=int)]
        )

    def get_applied_joint_efforts(self, joint_indices=None):
        self.applied_reads += 1
        values = np.asarray([4.0, 5.0, 6.0], dtype=float)
        return (
            values
            if joint_indices is None
            else values[np.asarray(joint_indices, dtype=int)]
        )


class _RobotWithoutEfforts:
    num_dof = 2
    dof_names = ["j0", "j1"]


class _Controller:
    last_commanded_efforts = np.asarray([0.1, 0.2, 0.3], dtype=float)


class _Targets:
    positions = np.asarray([0.4, 0.5, 0.6], dtype=float)
    velocities = np.asarray([1.4, 1.5, 1.6], dtype=float)


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_read_joint_efforts_uses_robot_methods_and_joint_indices() -> None:
    sample = read_joint_efforts(_RobotWithEfforts(), np.asarray([2, 0], dtype=int))

    np.testing.assert_allclose(sample.measured, [3.0, 1.0])
    np.testing.assert_allclose(sample.applied, [6.0, 4.0])


def test_read_joint_efforts_returns_nan_when_api_missing() -> None:
    sample = read_joint_efforts(_RobotWithoutEfforts(), np.asarray([0, 1], dtype=int))

    assert np.isnan(sample.measured).all()
    assert np.isnan(sample.applied).all()


def test_commanded_efforts_from_controller_slices_last_command() -> None:
    values = commanded_efforts_from_controller(
        _Controller(), np.asarray([1, 2], dtype=int)
    )

    np.testing.assert_allclose(values, [0.2, 0.3])


def test_effort_logger_writes_commanded_measured_and_applied_efforts(tmp_path) -> None:
    path = tmp_path / "efforts.csv"
    logger = EffortLogger(path, ["j0", "j1"])
    logger.write(
        step=7,
        time_s=0.25,
        phase="hold",
        drive_update=True,
        commanded_effort=np.asarray([0.1, 0.2]),
        measured_effort=np.asarray([1.0, 2.0]),
        applied_effort=np.asarray([3.0, 4.0]),
    )
    logger.close()

    row = _read_rows(path)[0]
    assert row["step"] == "7"
    assert row["phase"] == "hold"
    assert row["tau_cmd_j0"] == "0.1"
    assert row["tau_measured_j1"] == "2"
    assert row["tau_applied_j1"] == "4"


def test_joint_tracking_logger_includes_optional_effort_columns(tmp_path) -> None:
    path = tmp_path / "joint_tracking.csv"
    logger = JointTrackingLogger(
        path,
        ["j0"],
        config=JointLoggingConfig(
            log_action_effort=True,
            log_measured_effort=True,
            log_applied_effort=True,
        ),
    )
    logger.write(
        step=1,
        time_s=0.1,
        phase="move",
        drive_update=True,
        desired_position=np.asarray([0.5]),
        actual_position=np.asarray([0.25]),
        desired_velocity=np.asarray([0.0]),
        actual_velocity=np.asarray([0.1]),
        commanded_effort=np.asarray([0.7]),
        action_effort=np.asarray([0.75]),
        measured_effort=np.asarray([0.8]),
        applied_effort=np.asarray([0.9]),
    )
    logger.close()

    row = _read_rows(path)[0]
    assert row["tau_cmd_j0"] == "0.7"
    assert row["tau_action_j0"] == "0.75"
    assert row["tau_measured_j0"] == "0.8"
    assert row["tau_applied_j0"] == "0.9"


def test_joint_tracking_logger_omits_disabled_effort_columns(tmp_path) -> None:
    path = tmp_path / "joint_tracking.csv"
    logger = JointTrackingLogger(
        path, ["j0"], config=JointLoggingConfig(log_command_effort=False)
    )
    logger.write(
        step=1,
        time_s=0.1,
        phase="move",
        drive_update=True,
        desired_position=np.asarray([0.5]),
        actual_position=np.asarray([0.25]),
        desired_velocity=np.asarray([0.0]),
        actual_velocity=np.asarray([0.1]),
        commanded_effort=np.asarray([0.7]),
    )
    logger.close()

    row = _read_rows(path)[0]
    assert "tau_cmd_j0" not in row
    assert "tau_measured_j0" not in row
    assert "tau_applied_j0" not in row


def test_joint_tracking_logger_collects_efforts_only_when_enabled(tmp_path) -> None:
    robot = _RobotWithEfforts()
    logger = JointTrackingLogger(
        tmp_path / "joint_tracking.csv",
        ["j0", "j1"],
        config=JointLoggingConfig(log_measured_effort=False, log_applied_effort=True),
    )

    values = logger.collect_efforts(robot, _Controller(), np.asarray([2, 0], dtype=int))

    assert robot.measured_reads == 0
    assert robot.applied_reads == 1
    np.testing.assert_allclose(values["commanded_effort"], [0.3, 0.1])
    assert values["measured_effort"] is None
    np.testing.assert_allclose(values["applied_effort"], [6.0, 4.0])


def test_joint_tracking_logger_collects_step_values_only_for_enabled_columns(
    tmp_path,
) -> None:
    robot = _RobotWithEfforts()
    logger = JointTrackingLogger(
        tmp_path / "joint_tracking.csv",
        ["j0", "j1"],
        config=JointLoggingConfig(
            log_actual_position=False,
            log_actual_velocity=True,
            log_command_position=False,
            log_command_velocity=True,
            log_command_effort=False,
            log_measured_effort=False,
            log_applied_effort=False,
        ),
    )

    values = logger.collect_step_values(
        robot, _Controller(), _Targets(), np.asarray([2, 0], dtype=int)
    )

    assert robot.position_reads == 0
    assert robot.velocity_reads == 1
    assert robot.measured_reads == 0
    assert robot.applied_reads == 0
    assert values["desired_position"] is None
    assert values["actual_position"] is None
    np.testing.assert_allclose(values["desired_velocity"], [1.6, 1.4])
    np.testing.assert_allclose(values["actual_velocity"], [1.3, 1.1])
    assert values["commanded_effort"] is None


def test_joint_logging_config_parses_yaml_mapping() -> None:
    config = joint_logging_config_from_mapping(
        {
            "logging": {
                "enabled": False,
                "interval_steps": 5,
                "measured_effort": True,
                "applied_effort": True,
                "action_effort": True,
                "command_effort": False,
            }
        }
    )

    assert not config.enabled
    assert config.interval_steps == 5
    assert config.log_measured_effort
    assert config.log_applied_effort
    assert config.log_action_effort
    assert not config.log_command_effort
