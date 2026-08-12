from __future__ import annotations

import csv
from dataclasses import replace
import json

from linkerbot_sim.configuration.catalog import load_mirror_config
from linkerbot_sim.logging.hybrid_control_logger import HybridControlLogger


def _diagnostics(*, step: int) -> dict[str, object]:
    return {
        "active": True,
        "request_id": "hybrid-1",
        "robot_id": 0,
        "robot_label": "left",
        "step": step,
        "tick": step,
        "time_s": (step + 1) / 240.0,
        "phase": "press",
        "tare_generation": 1,
        "hybrid_parameter_generation": 2,
        "force_axes": [False, False, True, False, False, False],
        "target_position": [0.1, 0.2, 0.3],
        "target_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "actual_position": [0.1, 0.2, 0.29],
        "actual_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "target_wrench_tool_on_environment": [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
        "raw_wrench_environment_on_tool": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        "filtered_wrench_tool_on_environment": [0.0, 0.0, -1.9, 0.0, 0.0, 0.0],
        "motion_wrench": [0.0] * 6,
        "force_wrench": [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
        "commanded_arm_effort": [0.0] * 6,
        "contact_axes": [False, False, True, False, False, False],
        "wrench_saturated_axes": [False] * 6,
        "effort_saturated_axes": [False] * 6,
        "minimum_singular_value": 0.4,
        "condition_number": 3.0,
    }


def test_hybrid_logger_decimates_and_writes_json_vectors(tmp_path) -> None:
    base = load_mirror_config("physx_cpu_hybrid").outputs.logging
    path = tmp_path / "hybrid.csv"
    settings = replace(
        base,
        enabled=True,
        existing_data_policy="error",
        joint_tracking_path=str(tmp_path / "joint.csv"),
        hybrid_control_path=str(path),
        log_hybrid_control=True,
        interval_steps=2,
    )
    logger = HybridControlLogger(path, settings=settings, physics_dt=1.0 / 240.0)

    logger.write({"active": False})
    logger.write(_diagnostics(step=1))
    logger.write(_diagnostics(step=2))
    logger.close()

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["request_id"] == "hybrid-1"
    assert rows[0]["hybrid_parameter_generation"] == "2"
    assert json.loads(rows[0]["force_axes"]) == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
