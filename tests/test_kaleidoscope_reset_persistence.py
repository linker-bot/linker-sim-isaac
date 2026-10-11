from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from linkerbot_sim.kaleidoscope.reset_description import (
    KaleidoscopeResetSceneDescription,
    ResetGeneratorDescription,
    ResetObjectDescription,
    ResetRobotDescription,
    ResetTaskDescription,
)
from linkerbot_sim.kaleidoscope.reset_persistence import (
    load_reset_scene_description,
    save_reset_scene_description,
)


ROOT = Path(__file__).resolve().parents[1]


def _description() -> KaleidoscopeResetSceneDescription:
    return KaleidoscopeResetSceneDescription(
        scene_id="tblock_push",
        task_id="tblock_push_v1",
        compatibility_fingerprint="a" * 64,
        generator=ResetGeneratorDescription(
            algorithm="tblock-push-reset-v1",
            seed=7,
            sample_id=3,
            reset_index=0,
            rng_key=10,
            rng_counter_after=32,
        ),
        robots=(
            ResetRobotDescription(
                label="left",
                robot_profile="ar5_08_l6_l",
                joint_names=("j0",),
                joint_positions=(0.1,),
                joint_velocities=(0.0,),
                joint_targets=(0.1,),
            ),
        ),
        dynamic_object=ResetObjectDescription(
            name="Tblock",
            object_profile="TblockV1_default",
            position_xyz=(0.15, 0.0, -0.4),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            com_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        task=ResetTaskDescription(
            goal_position=(0.25, 0.0, -0.4),
            goal_yaw=0.0,
        ),
    )


def test_reset_description_save_load_and_replace_policy(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "reset.json"
    saved = save_reset_scene_description(_description(), destination)
    loaded = load_reset_scene_description(saved)

    assert saved == destination
    assert loaded == _description()
    with pytest.raises(FileExistsError, match="already exists"):
        save_reset_scene_description(_description(), destination)
    save_reset_scene_description(_description(), destination, replace=True)


def test_reset_description_json_is_canonical_and_deterministic(tmp_path: Path) -> None:
    first = save_reset_scene_description(_description(), tmp_path / "first.json")
    second = save_reset_scene_description(
        _description().as_dict(), tmp_path / "second.json"
    )

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert json.loads(first.read_text(encoding="utf-8"))["scene_id"] == "tblock_push"


def test_reset_description_load_rejects_invalid_json_and_non_object(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid .* JSON"):
        load_reset_scene_description(malformed)

    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        load_reset_scene_description(non_object)


def test_save_validates_before_creating_destination_directory(tmp_path: Path) -> None:
    payload = _description().as_dict()
    payload["schema"] = "obsolete"
    destination = tmp_path / "must-not-exist" / "reset.json"

    with pytest.raises(ValueError, match="unsupported"):
        save_reset_scene_description(payload, destination)

    assert not destination.parent.exists()


def test_atomic_save_removes_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "reset.json"

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(
        "linkerbot_sim.kaleidoscope.reset_persistence.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="replace failed"):
        save_reset_scene_description(_description(), destination)

    assert not destination.exists()
    assert tuple(tmp_path.glob(".reset.json.*.tmp")) == ()


def test_public_reset_description_round_trip_across_clean_processes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "portable-reset.json"
    producer = r"""
import json
import sys
from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope import (
    sample_reset_scene_description,
    save_reset_scene_description,
)
description = sample_reset_scene_description(
    load_kaleidoscope_config(),
    seed=7,
    sample_id=12,
    reset_index=4,
)
save_reset_scene_description(description, sys.argv[1])
print(json.dumps(description.as_dict(), allow_nan=False, sort_keys=True))
forbidden = ('torch', 'gymnasium', 'skrl', 'newton', 'mujoco_warp', 'omni', 'isaacsim')
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
assert loaded == [], loaded
"""
    consumer = r"""
import json
import sys
from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope import load_reset_scene_description
from linkerbot_sim.kaleidoscope.reset_projection import project_reset_description
config = load_kaleidoscope_config()
description = load_reset_scene_description(sys.argv[1])
payload = project_reset_description(config, description)
print(json.dumps({
    'description': description.as_dict(),
    'projection': {
        'rng_key': payload.generator.rng_key,
        'rng_counter_after': payload.generator.rng_counter_after,
        'joint_positions': payload.joint_positions,
        'goal_position': payload.goal_position,
        'goal_yaw': payload.goal_yaw,
    },
}, allow_nan=False, sort_keys=True))
forbidden = ('torch', 'gymnasium', 'skrl', 'newton', 'mujoco_warp', 'omni', 'isaacsim')
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
assert loaded == [], loaded
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    produced = subprocess.run(
        [sys.executable, "-c", producer, str(destination)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    consumed = subprocess.run(
        [sys.executable, "-c", consumer, str(destination)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    produced_description = json.loads(produced.stdout)
    consumed_payload = json.loads(consumed.stdout)
    assert json.loads(destination.read_text(encoding="utf-8")) == produced_description
    assert consumed_payload["description"] == produced_description
    assert consumed_payload["projection"] == {
        "rng_key": 2582658391323309603,
        "rng_counter_after": 160,
        "joint_positions": [
            value
            for robot in produced_description["robots"]
            for value in robot["joint_positions"]
        ],
        "goal_position": produced_description["task"]["goal_position"],
        "goal_yaw": produced_description["task"]["goal_yaw"],
    }
