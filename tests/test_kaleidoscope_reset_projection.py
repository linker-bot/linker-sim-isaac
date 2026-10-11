from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope.reset_projection import project_reset_description
from linkerbot_sim.kaleidoscope.reset_sampling import sample_reset_scene_description


ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, object]:
    return sample_reset_scene_description(
        load_kaleidoscope_config(),
        seed=7,
        sample_id=12,
        reset_index=4,
    ).as_dict()


def test_projection_flattens_canonical_robot_order_and_owns_values() -> None:
    config = load_kaleidoscope_config()
    source = _payload()

    result = project_reset_description(config, source)

    assert len(result.joint_positions) == 26
    assert result.joint_positions == tuple(
        value
        for robot in source["robots"]  # type: ignore[union-attr]
        for value in robot["joint_positions"]  # type: ignore[index]
    )
    assert result.joint_velocities == (0.0,) * 26
    assert result.joint_targets == result.joint_positions
    assert result.generator.rng_counter_after == 160
    source["robots"][0]["joint_positions"][0] = 9.0  # type: ignore[index]
    assert result.joint_positions[0] != 9.0


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"joint_positions": ()}, "joint_positions cannot be empty"),
        ({"joint_velocities": (0.0,)}, "joint_velocities"),
        ({"block_position": (0.0, 0.0)}, "block_position"),
        ({"goal_yaw": float("nan")}, "goal_yaw"),
    ),
)
def test_flat_payload_direct_construction_preserves_validated_invariants(
    changes: dict[str, object],
    expected: str,
) -> None:
    baseline = project_reset_description(load_kaleidoscope_config(), _payload())

    with pytest.raises(ValueError, match=expected):
        replace(baseline, **changes)


def _change_fingerprint(payload: dict[str, object]) -> None:
    payload["compatibility_fingerprint"] = "b" * 64


def _change_scene(payload: dict[str, object]) -> None:
    payload["scene_id"] = "other_scene"


def _change_task(payload: dict[str, object]) -> None:
    payload["task_id"] = "other_task"


def _remove_robot(payload: dict[str, object]) -> None:
    payload["robots"].pop()  # type: ignore[union-attr]


def _swap_robots(payload: dict[str, object]) -> None:
    payload["robots"].reverse()  # type: ignore[union-attr]


def _change_robot_profile(payload: dict[str, object]) -> None:
    payload["robots"][0]["robot_profile"] = "other_profile"  # type: ignore[index]


def _change_joint_names(payload: dict[str, object]) -> None:
    payload["robots"][0]["joint_names"].reverse()  # type: ignore[index]


def _change_object_name(payload: dict[str, object]) -> None:
    payload["dynamic_object"]["name"] = "other_object"  # type: ignore[index]


def _change_object_profile(payload: dict[str, object]) -> None:
    payload["dynamic_object"]["object_profile"] = "other_profile"  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (_change_fingerprint, "fingerprint"),
        (_change_scene, "scene_id"),
        (_change_task, "task_id"),
        (_remove_robot, "robot count"),
        (_swap_robots, r"robot\[0\] label"),
        (_change_robot_profile, r"robot\[0\] robot_profile"),
        (_change_joint_names, r"robot\[0\] joint_names"),
        (_change_object_name, "object name"),
        (_change_object_profile, "object profile"),
    ),
)
def test_projection_rejects_description_identity_mismatches(
    mutation: Callable[[dict[str, object]], None],
    expected: str,
) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match=expected):
        project_reset_description(load_kaleidoscope_config(), payload)


def test_v1_description_rejects_cross_backend_restore() -> None:
    physx = load_kaleidoscope_config("physx_cuda")
    newton = load_kaleidoscope_config("newton_cuda")
    description = sample_reset_scene_description(
        physx,
        seed=7,
        sample_id=12,
        reset_index=4,
    )

    with pytest.raises(ValueError, match="compatibility fingerprint"):
        project_reset_description(newton, description)


@pytest.mark.parametrize("field", ("algorithm", "rng_key", "rng_counter_after"))
def test_projection_rejects_inconsistent_generator_metadata(field: str) -> None:
    payload = _payload()
    generator = payload["generator"]
    if field == "algorithm":
        generator[field] = "other-algorithm"  # type: ignore[index]
    else:
        generator[field] += 1  # type: ignore[index,operator]

    with pytest.raises(ValueError, match="generator metadata"):
        project_reset_description(load_kaleidoscope_config(), payload)


def test_projection_import_and_execution_do_not_load_heavy_runtime_modules() -> None:
    code = """
import sys
from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope.reset_projection import project_reset_description
from linkerbot_sim.kaleidoscope.reset_sampling import sample_reset_scene_description
config = load_kaleidoscope_config()
description = sample_reset_scene_description(config, seed=1, sample_id=2)
assert len(project_reset_description(config, description).joint_positions) == 26
forbidden = ('torch', 'gymnasium', 'skrl', 'newton', 'mujoco_warp', 'omni', 'isaacsim')
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
assert loaded == [], loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    subprocess.run([sys.executable, "-c", code], env=env, check=True)
