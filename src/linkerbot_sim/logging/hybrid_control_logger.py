"""Bounded-row CSV logging for owner-thread hybrid-control diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from linkerbot_sim.configuration.outputs import LoggingOutputSettings
from linkerbot_sim.logging.csv_writer import CsvOutputPlan, CsvWriter


_VECTOR_FIELDS = (
    "force_axes",
    "target_position",
    "target_orientation_wxyz",
    "actual_position",
    "actual_orientation_wxyz",
    "target_wrench_tool_on_environment",
    "raw_wrench_environment_on_tool",
    "filtered_wrench_tool_on_environment",
    "motion_wrench",
    "force_wrench",
    "commanded_arm_effort",
    "contact_axes",
    "wrench_saturated_axes",
    "effort_saturated_axes",
)


def hybrid_control_fieldnames() -> list[str]:
    return [
        "request_id",
        "robot_id",
        "robot_label",
        "step",
        "tick",
        "time_s",
        "phase",
        "tare_generation",
        "hybrid_parameter_generation",
        "minimum_singular_value",
        "condition_number",
        *_VECTOR_FIELDS,
    ]


class HybridControlLogger:
    """Write one compact row per configured hybrid-control sample."""

    def __init__(
        self,
        path: str | Path,
        *,
        settings: LoggingOutputSettings,
        physics_dt: float,
        timestamped_run_name: str | None = None,
        output_plan: CsvOutputPlan | None = None,
        paths_applied: bool = False,
    ) -> None:
        if not settings.enabled or not settings.log_hybrid_control:
            raise ValueError("HybridControlLogger requires enabled hybrid logging")
        self.settings = settings
        self.physics_dt = float(physics_dt)
        self.writer = CsvWriter(
            path,
            hybrid_control_fieldnames(),
            flush_interval_rows=settings.flush_interval_steps(self.physics_dt),
            existing_data_policy=settings.existing_data_policy,
            timestamped_run_name=timestamped_run_name,
            output_plan=output_plan,
            paths_applied=paths_applied,
        )

    def should_write(self, step: int) -> bool:
        return self.settings.should_write_step(step)

    def write(self, diagnostics: Mapping[str, object]) -> None:
        if diagnostics.get("active") is not True:
            return
        step = int(diagnostics["step"])
        if not self.should_write(step):
            return
        row: dict[str, object] = {
            name: diagnostics[name]
            for name in (
                "request_id",
                "robot_id",
                "robot_label",
                "step",
                "tick",
                "time_s",
                "phase",
                "tare_generation",
                "hybrid_parameter_generation",
                "minimum_singular_value",
                "condition_number",
            )
        }
        for name in _VECTOR_FIELDS:
            row[name] = json.dumps(
                diagnostics[name],
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        self.writer.write(row)

    def close(self) -> None:
        self.writer.close()


__all__ = ["HybridControlLogger", "hybrid_control_fieldnames"]
