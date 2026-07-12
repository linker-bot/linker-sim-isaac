from __future__ import annotations

import pytest

from linkerbot_sim.configs.profiles import load_env_profile_yaml
from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.sensors.camera.observer import start_offline_camera_output
from linkerbot_sim.sensors.camera.runtime import (
    create_sensor_camera_runtime,
    initialize_sensor_camera_runtimes,
)
from linkerbot_sim.tiled.scene.cameras import tiled_sensor_camera_settings
from linkerbot_sim.tiled.config import TiledEnvConfig


class _ValidPrim:
    def IsValid(self) -> bool:
        return True


class _FakeStage:
    def GetPrimAtPath(self, path: str) -> _ValidPrim:
        del path
        return _ValidPrim()


class _FakeCamera:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []

    def initialize(self, *, attach_rgb_annotator: bool = True) -> None:
        self.calls.append(f"initialize:{attach_rgb_annotator}")

    def set_clipping_range(self, *, near_distance: float, far_distance: float) -> None:
        del near_distance, far_distance
        self.calls.append("clipping_range")

    def add_rgb_to_frame(self) -> None:
        self.calls.append("rgb_annotator")

    def add_distance_to_image_plane_to_frame(self) -> None:
        self.calls.append("depth_annotator")


@pytest.mark.parametrize(
    "profile_name",
    (
        "scene2_tiled",
        "scene3_tiled",
        "scene_exec_tiled",
        "scene_plan_tiled",
    ),
)
def test_bundled_tiled_camera_scope_matches_per_env_overrides(
    profile_name: str,
) -> None:
    env_config = load_env_profile_yaml(profile_name)
    sensors = SceneSensorSettings.from_env_config(env_config)
    tiled_config = TiledEnvConfig.from_env_config(env_config)

    expanded = tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)

    assert [camera.name for camera in expanded.cameras] == ["env_000_world_rgbd"]


def test_tiled_sensor_camera_settings_expand_per_env_cameras() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": True,
                        "env_ids": [0, 1],
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
                        "env_ids": [0],
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
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "prim_path": "/World/WorldRGBD",
                        "env_ids": [0],
                    }
                }
            }
        }
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


def test_tiled_sensor_camera_settings_rejects_pose_override_outside_scope() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "prim_path": "/World/WorldRGBD",
                        "env_ids": [0],
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
                "per_env": [
                    {
                        "env_id": 1,
                        "cameras": {
                            "world_rgbd": {
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

    with pytest.raises(
        ValueError,
        match=(
            r"env_id 1 at cameras\.world_rgbd\.pose is outside "
            r"sensors\.cameras\.world_rgbd\.env_ids"
        ),
    ):
        tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)


def test_tiled_camera_scope_selects_resources_independently_of_inspection() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": True,
                        "env_ids": [1, 2],
                        "prim_path": "/World/WorldRGBD",
                        "modalities": ["rgb", "depth"],
                        "output": {
                            "save_dir": "logs/cameras/world_rgbd",
                            "foxglove_topic_prefix": "/cameras/world_rgbd",
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
                "num_envs": 4,
                "diagnostics": {"inspect_env_ids": [3]},
            },
        }
    )

    expanded = tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)

    assert [camera.name for camera in expanded.cameras] == [
        "env_001_world_rgbd",
        "env_002_world_rgbd",
    ]
    assert [camera.prim_path for camera in expanded.cameras] == [
        "/World/envs/env_1/WorldRGBD",
        "/World/envs/env_2/WorldRGBD",
    ]
    assert [camera.output.save_dir for camera in expanded.cameras] == [
        "logs/cameras/world_rgbd/env_001",
        "logs/cameras/world_rgbd/env_002",
    ]
    assert [camera.output.foxglove_topic_prefix for camera in expanded.cameras] == [
        "/cameras/world_rgbd/env_001",
        "/cameras/world_rgbd/env_002",
    ]
    assert all(camera.env_ids is None for camera in expanded.cameras)


def test_tiled_camera_scope_limits_prim_annotator_and_output_resources(
    tmp_path,
) -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": True,
                        "env_ids": [1],
                        "prim_path": "/World/WorldRGBD",
                        "modalities": ["rgb", "depth"],
                        "output": {"save_dir": "world_rgbd"},
                    }
                }
            }
        }
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 3,
                "diagnostics": {"inspect_env_ids": [0, 2]},
            },
        }
    )
    expanded = tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)

    runtimes = tuple(
        create_sensor_camera_runtime(
            stage=_FakeStage(),
            settings=settings,
            camera_type=_FakeCamera,
            array_factory=tuple,
            quat_from_rpy=lambda rpy: ("quat", rpy),
        )
        for settings in expanded.enabled_cameras
    )
    initialize_sensor_camera_runtimes(runtimes)
    output = start_offline_camera_output(
        runtimes,
        path_resolver=lambda value: tmp_path / value,
    )
    assert output is not None
    try:
        assert [runtime.prim_path for runtime in runtimes] == [
            "/World/envs/env_1/WorldRGBD"
        ]
        assert runtimes[0].camera.calls == [
            "initialize:False",
            "clipping_range",
            "rgb_annotator",
            "depth_annotator",
        ]
        assert [camera.name for camera in output.observer.cameras] == [
            "env_001_world_rgbd"
        ]
        assert (tmp_path / "world_rgbd" / "env_001" / "metadata.jsonl").is_file()
        assert not (tmp_path / "world_rgbd" / "env_000").exists()
        assert not (tmp_path / "world_rgbd" / "env_002").exists()
    finally:
        assert output.close() is True


def test_tiled_camera_requires_explicit_scope() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": True,
                        "prim_path": "/World/WorldRGBD",
                    }
                }
            }
        }
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {"tiled": {"enabled": True, "num_envs": 3}}
    )

    with pytest.raises(
        ValueError,
        match=r"sensors\.cameras\.world_rgbd\.env_ids is required for a tiled profile",
    ):
        tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)


@pytest.mark.parametrize("camera_name", ("first", "disabled", "second"))
def test_tiled_camera_requires_scope_even_when_disabled(camera_name: str) -> None:
    camera = {"prim_path": f"/World/{camera_name.title()}"}
    if camera_name == "disabled":
        camera["enabled"] = False
    sensors = SceneSensorSettings.from_env_config(
        {"sensors": {"cameras": {camera_name: camera}}}
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {"tiled": {"enabled": True, "num_envs": 2}}
    )

    with pytest.raises(
        ValueError,
        match=rf"sensors\.cameras\.{camera_name}\.env_ids is required",
    ):
        tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)


def test_tiled_enabled_camera_requires_explicit_scope() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": True,
                        "prim_path": "/World/WorldRGBD",
                    }
                }
            }
        }
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {
            "tiled": {"enabled": True, "num_envs": 2},
        }
    )

    with pytest.raises(
        ValueError,
        match=r"sensors\.cameras\.world_rgbd\.env_ids is required for a tiled profile",
    ):
        tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)


def test_tiled_disabled_camera_requires_explicit_scope() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": False,
                        "prim_path": "/World/WorldRGBD",
                    }
                }
            }
        }
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {
            "tiled": {"enabled": True, "num_envs": 2},
        }
    )

    with pytest.raises(
        ValueError,
        match=r"sensors\.cameras\.world_rgbd\.env_ids is required for a tiled profile",
    ):
        tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)


def test_tiled_camera_scope_rejects_out_of_range_env_with_full_path() -> None:
    sensors = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world_rgbd": {
                        "enabled": True,
                        "env_ids": [2],
                        "prim_path": "/World/WorldRGBD",
                    }
                }
            }
        }
    )
    tiled_config = TiledEnvConfig.from_env_config(
        {
            "tiled": {"enabled": True, "num_envs": 2},
        }
    )

    with pytest.raises(
        ValueError,
        match=r"sensors\.cameras\.world_rgbd\.env_ids contains out-of-range env id",
    ):
        tiled_sensor_camera_settings(sensors, tiled_config=tiled_config)
