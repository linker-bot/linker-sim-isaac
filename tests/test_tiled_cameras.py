from __future__ import annotations

import pytest

from linkerbot_sim.sensors.camera_config import SceneSensorSettings
from linkerbot_sim.tiled.cameras import tiled_sensor_camera_settings
from linkerbot_sim.tiled.config import TiledEnvConfig


def test_tiled_sensor_camera_settings_expand_per_env_cameras() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": True,
                        "parent_prim_path": "/World",
                        "prim_path": "/World/WorldRGBD",
                        "pose": {
                            "xyz": [0.08, 0.0, 0.08],
                            "rpy": [0.0, 1.1, 0.0],
                        },
                        "resolution": [320, 240],
                        "frequency": 15.0,
                        "modalities": ["rgb", "depth"],
                        "output": {
                            "save_dir": "logs/cameras/world_rgbd",
                            "foxglove_topic_prefix": "/cameras/world_rgbd",
                            "foxglove_live_port": 8770,
                            "foxglove_mcap_path": "logs/cameras/tiled.mcap",
                        },
                    }
                }
            }
        }
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "base_env_path": "/World/envs",
                "env_prefix": "env",
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

    expanded = tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)

    assert [camera.name for camera in expanded.cameras] == [
        "env_000_world_rgbd",
        "env_001_world_rgbd",
    ]
    first, second = expanded.cameras
    assert first.parent_prim_path == "/World/envs/env_0"
    assert first.prim_path == "/World/envs/env_0/WorldRGBD"
    assert first.pose_xyz == (0.08, 0.0, 0.08)
    assert first.pose_rpy == (0.0, 1.1, 0.0)
    assert first.output.save_dir == "logs/cameras/world_rgbd/env_000"
    assert first.output.foxglove_topic_prefix == "/cameras/world_rgbd/env_000"
    assert first.output.foxglove_live_port == 8770
    assert first.output.foxglove_mcap_path == "logs/cameras/tiled.mcap"

    assert second.parent_prim_path == "/World/envs/env_1"
    assert second.prim_path == "/World/envs/env_1/WorldRGBD"
    assert second.pose_xyz == (0.2, 0.1, 0.3)
    assert second.pose_rpy == (0.0, 1.2, 0.1)
    assert second.output.save_dir == "logs/cameras/world_rgbd/env_001"
    assert second.output.foxglove_topic_prefix == "/cameras/world_rgbd/env_001"


def test_tiled_sensor_camera_settings_maps_nested_parent_paths() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "wrist": {
                        "parent_prim_path": "/World/Robot/Wrist",
                        "prim_path": "/World/Robot/Wrist/WristCamera",
                    }
                }
            }
        }
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {"tiled": {"enabled": True, "num_envs": 1}}
    )

    expanded = tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)

    assert expanded.cameras[0].parent_prim_path == "/World/envs/env_0/Robot/Wrist"
    assert expanded.cameras[0].prim_path == "/World/envs/env_0/Robot/Wrist/WristCamera"


def test_tiled_sensor_camera_settings_rejects_unknown_camera_override() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {"sensors": {"cameras": {"world_rgbd": {"prim_path": "/World/WorldRGBD"}}}}
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 1,
                "per_env": [
                    {
                        "env_id": 0,
                        "cameras": {
                            "missing": {
                                "pose": {
                                    "xyz": [0.0, 0.0, 0.0],
                                    "rpy": [0.0, 0.0, 0.0],
                                }
                            }
                        },
                    }
                ],
            }
        }
    )

    with pytest.raises(ValueError, match="unknown camera"):
        tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)
