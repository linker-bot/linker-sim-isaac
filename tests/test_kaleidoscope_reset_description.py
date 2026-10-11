from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from linkerbot_sim.kaleidoscope.reset_description import (
    KaleidoscopeResetSceneDescription,
    RESET_SCENE_DESCRIPTION_SCHEMA,
    ResetGeneratorDescription,
    ResetObjectDescription,
    ResetRobotDescription,
    ResetTaskDescription,
    validate_reset_scene_description,
)


def _description() -> KaleidoscopeResetSceneDescription:
    return KaleidoscopeResetSceneDescription(
        scene_id="tblock_push",
        task_id="tblock_push_v1",
        compatibility_fingerprint="a" * 64,
        generator=ResetGeneratorDescription(
            algorithm="tblock-push-reset-v1",
            seed=7,
            sample_id=12,
            reset_index=0,
            rng_key=19,
            rng_counter_after=32,
        ),
        robots=(
            ResetRobotDescription(
                label="left",
                robot_profile="ar5_08_l6_l",
                joint_names=("j0", "j1"),
                joint_positions=(0.1, 0.2),
                joint_velocities=(0.0, 0.0),
                joint_targets=(0.1, 0.2),
            ),
            ResetRobotDescription(
                label="right",
                robot_profile="ar5_08_l6_r",
                joint_names=("j0", "j1"),
                joint_positions=(-0.1, -0.2),
                joint_velocities=(0.0, 0.0),
                joint_targets=(-0.1, -0.2),
            ),
        ),
        dynamic_object=ResetObjectDescription(
            name="Tblock",
            object_profile="TblockV1_default",
            position_xyz=(0.15, 0.01, -0.4),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            com_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        task=ResetTaskDescription(
            goal_position=(0.25, 0.02, -0.4),
            goal_yaw=0.1,
        ),
    )


def test_reset_description_round_trips_json_mapping() -> None:
    source = _description()
    payload = source.as_dict()
    restored = KaleidoscopeResetSceneDescription.from_mapping(
        json.loads(json.dumps(payload))
    )

    assert payload["schema"] == RESET_SCENE_DESCRIPTION_SCHEMA
    assert restored == source
    assert restored.as_dict() == payload


def test_validation_returns_an_independently_owned_description() -> None:
    payload = _description().as_dict()
    validated = validate_reset_scene_description(payload)

    payload["robots"][0]["joint_positions"][0] = 9.0  # type: ignore[index]
    payload["task"]["goal_position"][0] = 9.0  # type: ignore[index]

    assert validated.robots[0].joint_positions[0] == pytest.approx(0.1)
    assert validated.task.goal_position[0] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema="obsolete"), "unsupported"),
        (lambda value: value.update(extra=True), "keys differ"),
        (
            lambda value: value.update(compatibility_fingerprint="not-a-hash"),
            "lowercase SHA-256",
        ),
        (lambda value: value.update(coordinate_frame="world"), "env-local"),
        (
            lambda value: value["robots"].append(value["robots"][0]),
            "labels must be unique",
        ),
        (
            lambda value: value["robots"][0].update(joint_names=["j0", "j0"]),
            "must not contain duplicates",
        ),
        (
            lambda value: value["robots"][0].update(joint_positions=[0.1]),
            "exactly 2 values",
        ),
        (
            lambda value: value["dynamic_object"].update(
                orientation_wxyz=[0.0, 0.0, 0.0, 0.0]
            ),
            "must be non-zero",
        ),
        (
            lambda value: value["task"].update(goal_yaw=float("nan")),
            "finite number",
        ),
        (
            lambda value: value["generator"].update(sample_id=-1),
            "must be non-negative",
        ),
    ],
)
def test_reset_description_rejects_invalid_payloads(mutate, message: str) -> None:
    payload = _description().as_dict()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        KaleidoscopeResetSceneDescription.from_mapping(payload)


def test_reset_description_module_import_stays_pure() -> None:
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
import linkerbot_sim.kaleidoscope.reset_description
forbidden = ('torch', 'gymnasium', 'newton', 'omni', 'isaacsim')
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
assert loaded == [], loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
