from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from linkerbot_sim.configuration.outputs import LoggingOutputSettings
from linkerbot_sim.logging.csv_writer import (
    CsvWriter,
    apply_csv_output_plans,
    plan_csv_output,
)
from linkerbot_sim.logging.effort_logger import (
    commanded_efforts_from_controller,
    read_joint_efforts,
)
from linkerbot_sim.logging.joint_logger import JointTrackingLogger


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


def test_csv_writer_existing_data_policies(tmp_path: Path) -> None:
    path = tmp_path / "tracking.csv"
    path.write_text("step\n1\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        CsvWriter(path, ["step"])
    assert path.read_text(encoding="utf-8") == "step\n1\n"

    resumed = CsvWriter(path, ["step"], existing_data_policy="resume")
    resumed.write({"step": 2})
    resumed.close()
    assert _read_rows(path) == [{"step": "1"}, {"step": "2"}]

    truncated = CsvWriter(path, ["step"], existing_data_policy="truncate")
    truncated.write({"step": 3})
    truncated.close()
    assert _read_rows(path) == [{"step": "3"}]


def test_csv_writer_resume_rejects_header_mismatch_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tracking.csv"
    original = "wrong,columns\n1,2\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="header does not match"):
        CsvWriter(path, ["step"], existing_data_policy="resume")

    assert path.read_text(encoding="utf-8") == original


def test_csv_writer_resume_rejects_unterminated_final_record_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tracking.csv"
    original = "step\n1"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="unterminated final record"):
        CsvWriter(path, ["step"], existing_data_policy="resume")

    assert path.read_text(encoding="utf-8") == original


def test_csv_writer_resume_rejects_unclosed_quote_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tracking.csv"
    original = 'step,note\n1,"unfinished\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        CsvWriter(path, ["step", "note"], existing_data_policy="resume")

    assert path.read_text(encoding="utf-8") == original


def test_csv_group_validates_every_resume_header_before_any_mutation(
    tmp_path: Path,
) -> None:
    truncate_path = tmp_path / "robot_0.csv"
    truncate_path.write_text("old\nvalue\n", encoding="utf-8")
    invalid_resume = tmp_path / "robot_1.csv"
    invalid_resume.write_text("wrong\nvalue\n", encoding="utf-8")

    truncate_plan = plan_csv_output(
        truncate_path,
        ["step"],
        existing_data_policy="truncate",
    )
    with pytest.raises(ValueError, match="header does not match"):
        plan_csv_output(
            invalid_resume,
            ["step"],
            existing_data_policy="resume",
        )

    assert truncate_path.read_text(encoding="utf-8") == "old\nvalue\n"
    assert invalid_resume.read_text(encoding="utf-8") == "wrong\nvalue\n"
    assert truncate_plan.path_plan.existed_at_preflight is True


def test_prepared_csv_group_applies_before_opening_writers(tmp_path: Path) -> None:
    paths = (tmp_path / "robot_0.csv", tmp_path / "robot_1.csv")
    plans = tuple(
        plan_csv_output(path, ["step"], existing_data_policy="error") for path in paths
    )

    apply_csv_output_plans(plans)
    writers = tuple(
        CsvWriter(
            path,
            ["step"],
            existing_data_policy="error",
            output_plan=plan,
            paths_applied=True,
        )
        for path, plan in zip(paths, plans, strict=True)
    )
    try:
        for step, writer in enumerate(writers):
            writer.write({"step": step})
    finally:
        for writer in writers:
            writer.close()

    assert [_read_rows(path) for path in paths] == [
        [{"step": "0"}],
        [{"step": "1"}],
    ]


def test_csv_writer_timestamped_dir_uses_unique_run_namespace(tmp_path: Path) -> None:
    requested = tmp_path / "tracking.csv"
    writer = CsvWriter(
        requested,
        ["step"],
        existing_data_policy="timestamped_dir",
        timestamped_run_name="20260711T120000.000000Z",
    )
    try:
        writer.write({"step": 1})
        assert writer.path == (tmp_path / "20260711T120000.000000Z" / "tracking.csv")
    finally:
        writer.close()


def test_joint_tracking_logger_propagates_csv_existing_data_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "joint_tracking.csv"
    path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        JointTrackingLogger(
            path,
            ["j0"],
            settings=_logging_settings(existing_data_policy="error"),
            flush_interval_steps=1,
        )


def test_joint_tracking_logger_includes_optional_effort_columns(tmp_path) -> None:
    path = tmp_path / "joint_tracking.csv"
    settings = _logging_settings(
        log_action_effort=True,
        log_measured_effort=True,
        log_applied_effort=True,
    )
    logger = JointTrackingLogger(
        path,
        ["j0"],
        settings=settings,
        flush_interval_steps=1,
    )
    assert logger.settings is settings
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
        path,
        ["j0"],
        settings=_logging_settings(log_command_effort=False),
        flush_interval_steps=1,
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
        settings=_logging_settings(
            log_measured_effort=False,
            log_applied_effort=True,
        ),
        flush_interval_steps=1,
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
        settings=_logging_settings(
            log_actual_position=False,
            log_actual_velocity=True,
            log_command_position=False,
            log_command_velocity=True,
            log_command_effort=False,
            log_measured_effort=False,
            log_applied_effort=False,
        ),
        flush_interval_steps=1,
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


def test_logging_output_settings_control_effort_columns() -> None:
    settings = _logging_settings(
        enabled=False,
        joint_tracking_path=None,
        interval_steps=5,
        log_measured_effort=True,
        log_applied_effort=True,
        log_action_effort=True,
        log_command_effort=False,
    )

    assert not settings.enabled
    assert settings.interval_steps == 5
    assert settings.log_measured_effort
    assert settings.log_applied_effort
    assert settings.log_action_effort
    assert not settings.log_command_effort
