from __future__ import annotations

from linkerbot_sim.sensors.camera_config import SceneSensorSettings


def test_scene_sensor_settings_parse_camera_values() -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "wrist_rgbd": {
                        "enabled": True,
                        "prim_path": "/World/Robot/Wrist/WristRGBD",
                        "parent_prim_path": "/World/Robot/Wrist",
                        "pose": {
                            "xyz": [0.0, 0.0, 0.08],
                            "rpy": [0.1, 0.2, 0.3],
                        },
                        "resolution": [640, 480],
                        "frequency": 30.0,
                        "modalities": ["rgb", "depth"],
                        "clipping_range": [0.01, 5.0],
                        "output": {
                            "save_dir": "logs/cameras/wrist_rgbd",
                            "foxglove_topic_prefix": "/cameras/wrist_rgbd",
                            "foxglove_live_host": "0.0.0.0",
                            "foxglove_live_port": 8770,
                            "foxglove_mcap_path": "logs/cameras/wrist_rgbd.mcap",
                        },
                    }
                }
            }
        }
    )

    assert len(settings.cameras) == 1
    assert len(settings.enabled_cameras) == 1
    camera = settings.cameras[0]
    assert camera.name == "wrist_rgbd"
    assert camera.enabled is True
    assert camera.prim_path == "/World/Robot/Wrist/WristRGBD"
    assert camera.parent_prim_path == "/World/Robot/Wrist"
    assert camera.pose_xyz == (0.0, 0.0, 0.08)
    assert camera.pose_rpy == (0.1, 0.2, 0.3)
    assert camera.resolution == (640, 480)
    assert camera.frequency == 30.0
    assert camera.modalities == ("rgb", "depth")
    assert camera.clipping_range == (0.01, 5.0)
    assert camera.output.save_dir == "logs/cameras/wrist_rgbd"
    assert camera.output.foxglove_topic_prefix == "/cameras/wrist_rgbd"
    assert camera.output.foxglove_live_host == "0.0.0.0"
    assert camera.output.foxglove_live_port == 8770
    assert camera.output.foxglove_mcap_path == "logs/cameras/wrist_rgbd.mcap"


def test_scene_sensor_settings_use_empty_defaults() -> None:
    settings = SceneSensorSettings.from_env_config({})

    assert settings.cameras == ()
    assert settings.enabled_cameras == ()


def test_scene_sensor_settings_reject_invalid_camera_values() -> None:
    invalid_configs = [
        (
            {"sensors": {"cameras": {"bad": {"prim_path": "World/Camera"}}}},
            "sensors.cameras.bad.prim_path",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "resolution": [640, 0],
                        }
                    }
                }
            },
            "sensors.cameras.bad.resolution",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "frequency": 0.0,
                        }
                    }
                }
            },
            "sensors.cameras.bad.frequency",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "modalities": ["rgb", "thermal"],
                        }
                    }
                }
            },
            "unsupported modality",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "clipping_range": [1.0, 0.5],
                        }
                    }
                }
            },
            "clipping_range",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Sensors/WristRGBD",
                            "parent_prim_path": "/World/Robot/Wrist",
                        }
                    }
                }
            },
            "must be under parent_prim_path",
        ),
    ]

    for config, expected_message in invalid_configs:
        try:
            SceneSensorSettings.from_env_config(config)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"SceneSensorSettings accepted {config!r}")
