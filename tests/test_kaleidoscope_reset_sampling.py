from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from linkerbot_sim.configuration import (
    load_kaleidoscope_config,
    semantic_config_fingerprint,
)
from linkerbot_sim.kaleidoscope.reset_description import (
    KaleidoscopeResetSceneDescription,
)
from linkerbot_sim.kaleidoscope.reset_sampling import (
    RESET_SAMPLING_ALGORITHM,
    sample_reset_scene_description,
)


ROOT = Path(__file__).resolve().parents[1]


def test_same_seed_identity_and_index_produce_the_same_owned_description() -> None:
    config = load_kaleidoscope_config()

    first = sample_reset_scene_description(
        config,
        seed=7,
        sample_id=12,
        reset_index=4,
    )
    second = sample_reset_scene_description(
        config,
        seed=7,
        sample_id=12,
        reset_index=4,
    )

    assert first == second
    assert first is not second
    assert first.generator.algorithm == RESET_SAMPLING_ALGORITHM
    assert first.generator.rng_key == 2582658391323309603
    assert first.generator.rng_counter_after == 160
    assert first.compatibility_fingerprint == semantic_config_fingerprint(config)
    assert first.robots[0].joint_positions[:2] == pytest.approx(
        (0.01623452641069889, 0.004826268181204796),
        abs=1e-9,
    )
    assert first.dynamic_object.position_xyz == pytest.approx(
        (0.16307981312274933, 0.00547429732978344, -0.4000000059604645),
        abs=1e-8,
    )
    assert first.task.goal_position == pytest.approx(
        (0.24360637366771698, -0.0465523824095726, -0.4000000059604645),
        abs=1e-8,
    )
    assert first.task.goal_yaw == pytest.approx(-0.035124048590660095, abs=1e-8)
    assert (
        KaleidoscopeResetSceneDescription.from_mapping(
            json.loads(json.dumps(first.as_dict()))
        )
        == first
    )


def test_reset_index_directly_addresses_the_logical_rng_sequence() -> None:
    config = load_kaleidoscope_config()

    first = sample_reset_scene_description(
        config,
        seed=3,
        sample_id=5,
        reset_index=0,
    )
    next_reset = sample_reset_scene_description(
        config,
        seed=3,
        sample_id=5,
        reset_index=1,
    )

    assert first.generator.rng_key == next_reset.generator.rng_key
    assert first.generator.rng_counter_after == 32
    assert next_reset.generator.rng_counter_after == 64
    assert first.robots[0].joint_positions != next_reset.robots[0].joint_positions
    assert first.dynamic_object.position_xyz != next_reset.dynamic_object.position_xyz


def test_sample_preserves_config_identity_order_ranges_and_zero_velocities() -> None:
    config = load_kaleidoscope_config()
    result = sample_reset_scene_description(
        config,
        seed=-9,
        sample_id=2,
    )
    randomization = config.task.randomization

    assert result.scene_id == config.scene.scene_id
    assert result.task_id == config.task.task_id
    assert tuple(robot.label for robot in result.robots) == tuple(
        robot.label for robot in config.scene.robots
    )
    for result_robot, config_robot in zip(
        result.robots,
        config.scene.robots,
        strict=True,
    ):
        assert config_robot.resolved_profile is not None
        assert result_robot.robot_profile == config_robot.robot_profile
        assert (
            result_robot.joint_names
            == config_robot.resolved_profile.joint_groups.command_joint_names
        )
        nominal = config_robot.ordered_reset_joint_positions()
        for actual, center in zip(
            result_robot.joint_positions,
            nominal,
            strict=True,
        ):
            assert center + randomization.robot_joint_delta_rad[0] <= actual
            assert actual <= center + randomization.robot_joint_delta_rad[1]
        assert result_robot.joint_velocities == (0.0,) * len(nominal)
        assert result_robot.joint_targets == result_robot.joint_positions

    dynamic = next(
        item for item in config.scene.objects if item.name == config.task.dynamic_object
    )
    x, y, z = result.dynamic_object.position_xyz
    assert dynamic.root_pose.xyz[0] + randomization.block_x_delta_m[0] <= x
    assert x <= dynamic.root_pose.xyz[0] + randomization.block_x_delta_m[1]
    assert dynamic.root_pose.xyz[1] + randomization.block_y_delta_m[0] <= y
    assert y <= dynamic.root_pose.xyz[1] + randomization.block_y_delta_m[1]
    assert z == pytest.approx(dynamic.root_pose.xyz[2])
    assert result.dynamic_object.com_velocity == (0.0,) * 6

    goal_dx = result.task.goal_position[0] - x
    goal_dy = result.task.goal_position[1] - y
    assert randomization.goal_x_delta_m[0] - 1e-7 <= goal_dx
    assert goal_dx <= randomization.goal_x_delta_m[1] + 1e-7
    assert randomization.goal_y_delta_m[0] - 1e-7 <= goal_dy
    assert goal_dy <= randomization.goal_y_delta_m[1] + 1e-7
    assert result.task.goal_position[2] == z


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"seed": True, "sample_id": 0}, "seed"),
        ({"seed": 2**63, "sample_id": 0}, "seed"),
        ({"seed": 0, "sample_id": -1}, "sample_id"),
        ({"seed": 0, "sample_id": True}, "sample_id"),
        ({"seed": 0, "sample_id": 0, "reset_index": -1}, "reset_index"),
        (
            {"seed": 0, "sample_id": 0, "reset_index": 2**63 - 1},
            "RNG counter",
        ),
    ],
)
def test_sampler_rejects_invalid_seed_identity_or_counter(
    arguments: dict[str, int],
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        sample_reset_scene_description(load_kaleidoscope_config(), **arguments)


def test_sampling_import_and_execution_do_not_load_heavy_runtime_modules() -> None:
    code = """
import json
import sys
from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope.reset_sampling import sample_reset_scene_description
value = sample_reset_scene_description(
    load_kaleidoscope_config(), seed=11, sample_id=4, reset_index=2
)
print(json.dumps(value.as_dict(), allow_nan=False, sort_keys=True))
forbidden = ('torch', 'gymnasium', 'skrl', 'newton', 'mujoco_warp', 'omni', 'isaacsim')
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
assert loaded == [], loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    first = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert first == second
