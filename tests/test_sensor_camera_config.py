from __future__ import annotations

import pytest

from linkerbot_sim.sensors import SceneSensorSettings


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
                        "env_ids": [0, 2],
                        "modalities": ["rgb", "depth"],
                        "clipping_range": [0.01, 5.0],
                        "intrinsics": {
                            "fx": 615.0,
                            "fy": 616.0,
                            "cx": 320.0,
                            "cy": 240.0,
                        },
                        "output": {
                            "save_dir": "logs/cameras/wrist_rgbd",
                            "foxglove_topic_prefix": "/cameras/wrist_rgbd",
                            "foxglove_live_host": "127.0.0.2",
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
    assert settings.has_output_consumers is True
    camera = settings.cameras[0]
    assert camera.name == "wrist_rgbd"
    assert camera.enabled is True
    assert camera.prim_path == "/World/Robot/Wrist/WristRGBD"
    assert camera.parent_prim_path == "/World/Robot/Wrist"
    assert camera.pose_xyz == (0.0, 0.0, 0.08)
    assert camera.pose_rpy == (0.1, 0.2, 0.3)
    assert camera.resolution == (640, 480)
    assert camera.frequency == 30.0
    assert camera.env_ids == (0, 2)
    assert camera.modalities == ("rgb", "depth")
    assert camera.clipping_range == (0.01, 5.0)
    assert camera.intrinsics is not None
    assert camera.intrinsics.fx == 615.0
    assert camera.intrinsics.fy == 616.0
    assert camera.intrinsics.cx == 320.0
    assert camera.intrinsics.cy == 240.0
    assert camera.intrinsics.matrix() == (
        (615.0, 0.0, 320.0),
        (0.0, 616.0, 240.0),
        (0.0, 0.0, 1.0),
    )
    assert camera.output.save_dir == "logs/cameras/wrist_rgbd"
    assert camera.output.foxglove_topic_prefix == "/cameras/wrist_rgbd"
    assert camera.output.foxglove_live_host == "127.0.0.2"
    assert camera.output.foxglove_live_port == 8770
    assert camera.output.foxglove_mcap_path == "logs/cameras/wrist_rgbd.mcap"


def test_scene_sensor_settings_use_empty_defaults() -> None:
    settings = SceneSensorSettings.from_env_config({})

    assert settings.cameras == ()
    assert settings.enabled_cameras == ()
    assert settings.has_output_consumers is False


def test_scene_sensor_settings_use_camera_defaults() -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "main": {
                        "prim_path": "/World/Camera",
                        "output": {},
                    }
                }
            }
        }
    )

    camera = settings.cameras[0]
    assert camera.enabled is True
    assert camera.pose_xyz == (0.0, 0.0, 0.0)
    assert camera.pose_rpy == (0.0, 0.0, 0.0)
    assert camera.resolution == (640, 480)
    assert camera.frequency == 30.0
    assert camera.env_ids is None
    assert camera.modalities == ("rgb",)
    assert camera.clipping_range == (0.01, 5.0)
    assert camera.output.foxglove_live_host == "127.0.0.1"
    assert settings.has_output_consumers is False


@pytest.mark.parametrize("host", ("127.0.0.1", "127.0.0.2", "::1", "localhost"))
def test_camera_live_output_accepts_explicit_loopback_hosts(host: str) -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "main": {
                        "prim_path": "/World/Camera",
                        "output": {"foxglove_live_host": host},
                    }
                }
            }
        }
    )

    assert settings.cameras[0].output.foxglove_live_host == host


@pytest.mark.parametrize("host", ("0.0.0.0", "::", "192.0.2.10", "example.invalid"))
def test_camera_live_output_rejects_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        SceneSensorSettings.from_env_config(
            {
                "sensors": {
                    "cameras": {
                        "main": {
                            "prim_path": "/World/Camera",
                            "output": {"foxglove_live_host": host},
                        }
                    }
                }
            }
        )


def test_camera_output_consumer_requires_enabled_camera_and_real_sink() -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "disabled_writer": {
                        "enabled": False,
                        "prim_path": "/World/DisabledCamera",
                        "output": {"save_dir": "logs/disabled"},
                    },
                    "topic_only": {
                        "prim_path": "/World/TopicOnlyCamera",
                        "output": {"foxglove_topic_prefix": "/camera/topic-only"},
                    },
                }
            }
        }
    )

    assert settings.has_output_consumers is False


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
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0},
                        }
                    }
                }
            },
            "sensors.cameras.bad.intrinsics.cy is required",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "intrinsics": {
                                "fx": 0.0,
                                "fy": 600.0,
                                "cx": 320.0,
                                "cy": 240.0,
                            },
                        }
                    }
                }
            },
            "sensors.cameras.bad.intrinsics.fx must be positive",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "env_ids": [],
                        }
                    }
                }
            },
            "sensors.cameras.bad.env_ids must be non-empty",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "env_ids": None,
                        }
                    }
                }
            },
            "sensors.cameras.bad.env_ids must be a non-empty sequence of integers",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "env_ids": [1.0],
                        }
                    }
                }
            },
            "sensors.cameras.bad.env_ids[0] must be an integer",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "env_ids": [0, 0],
                        }
                    }
                }
            },
            "sensors.cameras.bad.env_ids cannot contain duplicates",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "env_ids": [True],
                        }
                    }
                }
            },
            "sensors.cameras.bad.env_ids[0] must be an integer",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "env_ids": [-1],
                        }
                    }
                }
            },
            "sensors.cameras.bad.env_ids[0] must be nonnegative",
        ),
        (
            {
                "sensors": {
                    "cameras": {
                        "bad": {
                            "prim_path": "/World/Camera",
                            "env_id": 0,
                        }
                    }
                }
            },
            "sensors.cameras.bad contains unsupported keys: env_id",
        ),
    ]

    for config, expected_message in invalid_configs:
        try:
            SceneSensorSettings.from_env_config(config)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"SceneSensorSettings accepted {config!r}")
