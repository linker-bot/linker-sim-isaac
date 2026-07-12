from __future__ import annotations

import pytest

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.tiled.config import TiledEnvConfig


def test_tiled_config_defaults_to_disabled_single_env() -> None:
    config = TiledEnvConfig.from_env_config({})

    assert config.enabled is False
    assert config.num_envs == 1
    assert config.base_env_path == "/World/envs"
    assert config.env_prefix == "env"
    assert config.layout.origin_xyz == (0.0, 0.0, 0.0)
    assert config.clone.replicate_physics is True
    assert config.clone.physics_scene_path is None
    assert config.clone.global_collision_paths == "auto"
    assert config.clone.extra_global_collision_paths == ()
    assert config.diagnostics.inspect_env_ids == (0,)


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
                "layout": {"origin_xyz": [10.0, -2.0, 0.5]},
                "clone": {
                    "replicate_physics": False,
                    "filter_collisions": True,
                    "collision_root_path": "/World/CollisionGroups",
                    "physics_scene_path": "/World/physicsScene",
                    "global_collision_paths": ["/World/customGround"],
                    "extra_global_collision_paths": ["/World/fixture"],
                },
                "diagnostics": {
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
    assert config.layout.origin_xyz == (10.0, -2.0, 0.5)
    assert config.clone.replicate_physics is False
    assert config.clone.filter_collisions is True
    assert config.clone.physics_scene_path == "/World/physicsScene"
    assert config.clone.global_collision_paths == ("/World/customGround",)
    assert config.clone.extra_global_collision_paths == ("/World/fixture",)
    assert config.diagnostics.inspect_env_ids == (0, 3)


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
    assert config.metadata_for_env(1) == {"replay_id": "case_001"}
    assert config.metadata_for_env(0) == {}


def test_tiled_config_parses_complete_per_env_robot_root_pose() -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "per_env": [
                    {
                        "env_id": 1,
                        "robots": {
                            "left_arm": {
                                "root_pose": {
                                    "xyz": [0.25, 0.1, 0.05],
                                    "rpy": [-1.2, 0.1, 0.2],
                                }
                            }
                        },
                    }
                ],
            }
        }
    )
    base_pose = RootPoseConfig(
        xyz=(0.0, 0.09, 0.0),
        rpy=(-1.5707, 0.0, 0.0),
    )

    assert config.robot_root_pose_for_env(0, "left_arm", base_pose) is base_pose
    assert config.robot_root_pose_for_env(1, "left_arm", base_pose) == RootPoseConfig(
        xyz=(0.25, 0.1, 0.05),
        rpy=(-1.2, 0.1, 0.2),
    )


def test_tiled_per_env_metadata_rejects_non_json_values() -> None:
    with pytest.raises(ValueError, match=r"tiled\.per_env\[0\]\.metadata\.bad"):
        TiledEnvConfig.from_env_config(
            {
                "tiled": {
                    "enabled": True,
                    "num_envs": 1,
                    "per_env": [
                        {
                            "env_id": 0,
                            "metadata": {"bad": object()},
                        }
                    ],
                }
            }
        )


def test_tiled_config_parses_per_env_camera_pose_overrides() -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "per_env": [
                    {
                        "env_id": 1,
                        "cameras": {
                            "world_rgbd": {
                                "pose": {
                                    "xyz": [0.2, 0.1, 0.3],
                                    "rpy": [0.0, 1.2, 0.1],
                                }
                            }
                        },
                    }
                ],
            }
        }
    )

    assert config.per_env[0].camera_poses["world_rgbd"].xyz == (0.2, 0.1, 0.3)
    assert config.per_env[0].camera_poses["world_rgbd"].rpy == (0.0, 1.2, 0.1)


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
        ({"tiled": {"clone": {"collision_root_path": "/"}}}, "cannot be '/'"),
        ({"tiled": {"env_prefix": "bad/name"}}, "env_prefix"),
        ({"tiled": {"per_env_config_dir": "../bad"}}, "per_env_config_dir"),
        ({"tiled": {"layout": {"origin_rpy": [0, 0, 0]}}}, "unsupported"),
        ({"tiled": {"layout": {"origin_xyz": [0, float("nan"), 0]}}}, "finite"),
        (
            {"tiled": {"clone": {"physics_scene_path": "World/physicsScene"}}},
            "absolute USD path",
        ),
        (
            {"tiled": {"clone": {"global_collision_paths": "default"}}},
            "'auto' or a sequence",
        ),
        (
            {"tiled": {"clone": {"extra_global_collision_paths": "auto"}}},
            "sequence",
        ),
        (
            {
                "tiled": {
                    "clone": {
                        "filter_collisions": False,
                        "physics_scene_path": "/World/physicsScene",
                    }
                }
            },
            "require filter_collisions=true.*collision_groups",
        ),
        (
            {
                "tiled": {
                    "clone": {
                        "filter_collisions": False,
                        "global_collision_paths": [],
                    }
                }
            },
            "require filter_collisions=true.*collision_groups",
        ),
        (
            {
                "tiled": {
                    "clone": {
                        "collision_filter_strategy": "filtered_pairs",
                        "extra_global_collision_paths": ["/World/ground"],
                    }
                }
            },
            "require filter_collisions=true.*collision_groups",
        ),
        (
            {
                "tiled": {
                    "clone": {
                        "collision_filter_strategy": "filtered_pairs",
                        "collision_root_path": "/World/customCollisions",
                    }
                }
            },
            "require filter_collisions=true.*collision_groups",
        ),
        (
            {"tiled": {"clone": {"global_collision_paths": ["/"]}}},
            "cannot be '/'",
        ),
        ({"tiled": {"diagnostics": {"inspect_env_ids": [2]}}}, "inspect_env_ids"),
        ({"tiled": {"diagnostics": {"render_env_ids": [0]}}}, "unsupported"),
        (
            {"tiled": {"clone": {"use_grid_cloner": True}}},
            "unsupported",
        ),
        (
            {"tiled": {"diagnostics": {"use_batched_articulation_view": True}}},
            "unsupported",
        ),
        (
            {"tiled": {"runtime": {"planner": "curobo"}}},
            "unsupported",
        ),
        (
            {"tiled": {"runtime": {"planner": {"profile": "default"}}}},
            "unsupported",
        ),
        (
            {
                "tiled": {
                    "runtime": {
                        "planner": {
                            "backend": "curobo_tiled_joint",
                        }
                    }
                }
            },
            "unsupported",
        ),
        (
            {
                "tiled": {
                    "runtime": {
                        "planner": {
                            "joint_batch_mode": "batch-only",
                        }
                    }
                }
            },
            "unsupported",
        ),
        (
            {"tiled": {"num_envs": 1, "per_env": [{"env_id": 2, "objects": {}}]}},
            "env_id",
        ),
        (
            {
                "tiled": {
                    "per_env": [
                        {
                            "env_id": 0,
                            "robots": {
                                1: {
                                    "root_pose": {
                                        "xyz": [0, 0, 0],
                                        "rpy": [0, 0, 0],
                                    }
                                }
                            },
                        }
                    ]
                }
            },
            "keys must be non-empty strings",
        ),
        (
            {
                "tiled": {
                    "per_env": [
                        {
                            "env_id": 0,
                            "robots": {"left_arm": {"root_pose": {"xyz": [0, 0, 0]}}},
                        }
                    ]
                }
            },
            "complete xyz and rpy",
        ),
        (
            {
                "tiled": {
                    "per_env": [
                        {
                            "env_id": 0,
                            "cameras": {"world_rgbd": {"root_pose": {}}},
                        }
                    ]
                }
            },
            "unsupported",
        ),
        (
            {
                "tiled": {
                    "per_env": [
                        {
                            "env_id": 0,
                            "objects": {
                                "overrides": {
                                    "Tblock": {
                                        "root_pose": {
                                            "xyz": [0.0, 0.0, 0.0],
                                            "rpy": [0.0, 0.0, 0.0],
                                        }
                                    }
                                }
                            },
                        }
                    ]
                }
            },
            "unsupported",
        ),
        (
            {
                "tiled": {
                    "per_env": [
                        {
                            "env_id": 0,
                            "cameras": {
                                "overrides": {
                                    "world_rgbd": {
                                        "pose": {
                                            "xyz": [0.0, 0.0, 0.0],
                                            "rpy": [0.0, 0.0, 0.0],
                                        }
                                    }
                                }
                            },
                        }
                    ]
                }
            },
            "unsupported",
        ),
        ({"tiled": {"unknown": True}}, "unsupported"),
    ],
)
def test_tiled_config_rejects_invalid_values(
    data: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TiledEnvConfig.from_env_config(data)
