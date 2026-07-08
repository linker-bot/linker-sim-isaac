from __future__ import annotations

import pytest

from linkerbot_sim.tiled.config import TiledEnvConfig


def test_tiled_config_defaults_to_disabled_single_env() -> None:
    config = TiledEnvConfig.from_env_config({})

    assert config.enabled is False
    assert config.num_envs == 1
    assert config.base_env_path == "/World/envs"
    assert config.env_prefix == "env"
    assert config.clone.replicate_physics is True
    assert config.runtime.inspect_env_ids == (0,)


def test_tiled_config_parses_nested_settings() -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 8,
                "base_env_path": "/World/Tiled",
                "env_prefix": "scene",
                "spacing": 3.5,
                "num_per_row": 4,
                "clone": {
                    "replicate_physics": False,
                    "filter_collisions": False,
                    "collision_root_path": "/World/CollisionGroups",
                },
                "runtime": {
                    "use_batched_articulation_view": True,
                    "inspect_env_ids": [0, 3],
                },
            }
        }
    )

    assert config.enabled is True
    assert config.num_envs == 8
    assert config.base_env_path == "/World/Tiled"
    assert config.env_prefix == "scene"
    assert config.spacing == 3.5
    assert config.effective_num_per_row == 4
    assert config.clone.replicate_physics is False
    assert config.clone.filter_collisions is False
    assert config.runtime.inspect_env_ids == (0, 3)


def test_tiled_config_parses_per_env_object_pose_overrides() -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "per_env_config_dir": "envs",
                "per_env": [
                    {
                        "env_id": 1,
                        "objects": {
                            "Tblock": {
                                "root_pose": {
                                    "xyz": [0.12, 0.04, -0.4],
                                    "rpy": [0.0, 1.5707, 0.2],
                                }
                            }
                        },
                        "metadata": {"replay_id": "case_001"},
                    }
                ],
            }
        }
    )

    assert config.per_env_config_dir == "envs"
    assert [item.env_id for item in config.per_env] == [1]
    assert config.per_env[0].object_root_poses["Tblock"].xyz == (0.12, 0.04, -0.4)
    assert config.per_env[0].metadata == {"replay_id": "case_001"}


def test_tiled_num_envs_comes_from_yaml() -> None:
    config = TiledEnvConfig.from_env_config(
        {"tiled": {"enabled": True, "num_envs": 4}},
    )

    assert config.enabled is True
    assert config.num_envs == 4


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"tiled": {"num_envs": 0}}, "num_envs"),
        ({"tiled": {"spacing": 0.0}}, "spacing"),
        ({"tiled": {"base_env_path": "World/envs"}}, "base_env_path"),
        ({"tiled": {"env_prefix": "bad/name"}}, "env_prefix"),
        ({"tiled": {"per_env_config_dir": "../bad"}}, "per_env_config_dir"),
        ({"tiled": {"runtime": {"inspect_env_ids": [2]}}}, "inspect_env_ids"),
        ({"tiled": {"runtime": {"render_env_ids": [0]}}}, "unsupported"),
        (
            {"tiled": {"clone": {"use_grid_cloner": False}}},
            "use_grid_cloner",
        ),
        (
            {"tiled": {"runtime": {"use_batched_articulation_view": False}}},
            "use_batched_articulation_view",
        ),
        (
            {"tiled": {"num_envs": 1, "per_env": [{"env_id": 2, "objects": {}}]}},
            "env_id",
        ),
        ({"tiled": {"unknown": True}}, "unsupported"),
    ],
)
def test_tiled_config_rejects_invalid_values(
    data: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TiledEnvConfig.from_env_config(data)
