from __future__ import annotations

from dataclasses import replace

import pytest

from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.sensors.camera.config import (
    SensorCameraIntrinsicsSettings,
    SensorCameraOutputSettings,
    SensorCameraSettings,
)


def _camera(**overrides: object) -> SensorCameraSettings:
    values: dict[str, object] = {
        "name": "wrist_rgbd",
        "prim_path": "/World/Robot/Wrist/WristRGBD",
        "parent_prim_path": "/World/Robot/Wrist",
        "pose_xyz": (0.0, 0.0, 0.08),
        "pose_rpy": (0.1, 0.2, 0.3),
        "resolution": (640, 480),
        "frequency": 30.0,
        "modalities": ("rgb", "depth"),
        "clipping_range": (0.01, 5.0),
        "intrinsics": SensorCameraIntrinsicsSettings(
            fx=615.0,
            fy=616.0,
            cx=320.0,
            cy=240.0,
        ),
        "output": SensorCameraOutputSettings(
            save_dir="logs/cameras/wrist_rgbd",
            foxglove_topic_prefix="/cameras/wrist_rgbd",
            foxglove_live_host="127.0.0.2",
            foxglove_live_port=8770,
            foxglove_mcap_path="logs/cameras/wrist_rgbd.mcap",
        ),
    }
    values.update(overrides)
    return SensorCameraSettings(**values)  # type: ignore[arg-type]


def test_typed_scene_sensor_settings_preserve_camera_values() -> None:
    settings = SceneSensorSettings(cameras=(_camera(),))

    assert len(settings.enabled_cameras) == 1
    assert settings.has_output_consumers is True
    camera = settings.cameras[0]
    assert camera.name == "wrist_rgbd"
    assert camera.pose_xyz == (0.0, 0.0, 0.08)
    assert camera.pose_rpy == (0.1, 0.2, 0.3)
    assert camera.resolution == (640, 480)
    assert camera.modalities == ("rgb", "depth")
    assert camera.intrinsics is not None
    assert camera.intrinsics.matrix() == (
        (615.0, 0.0, 320.0),
        (0.0, 616.0, 240.0),
        (0.0, 0.0, 1.0),
    )
    assert camera.output.foxglove_live_host == "127.0.0.2"


def test_scene_sensor_settings_defaults_and_consumer_selection() -> None:
    assert SceneSensorSettings().enabled_cameras == ()
    assert SceneSensorSettings().has_output_consumers is False

    disabled = replace(_camera(), enabled=False)
    topic_only = SensorCameraSettings(
        name="topic_only",
        prim_path="/World/TopicOnlyCamera",
        output=SensorCameraOutputSettings(foxglove_topic_prefix="/camera/topic-only"),
    )
    settings = SceneSensorSettings(cameras=(disabled, topic_only))

    assert settings.enabled_cameras == (topic_only,)
    assert settings.has_output_consumers is False


@pytest.mark.parametrize("host", ("127.0.0.1", "127.0.0.2", "::1", "localhost"))
def test_camera_live_output_accepts_explicit_loopback_hosts(host: str) -> None:
    assert (
        SensorCameraOutputSettings(foxglove_live_host=host).foxglove_live_host == host
    )


@pytest.mark.parametrize("host", ("0.0.0.0", "::", "192.0.2.10", "example.invalid"))
def test_camera_live_output_rejects_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        SensorCameraOutputSettings(foxglove_live_host=host)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"prim_path": "World/Camera", "parent_prim_path": None}, "prim_path"),
        ({"resolution": (640, 0)}, "resolution"),
        ({"frequency": 0.0}, "frequency"),
        ({"modalities": ("rgb", "thermal")}, "unsupported modality"),
        ({"modalities": ("rgb", "rgb")}, "duplicates"),
        ({"clipping_range": (1.0, 0.5)}, "clipping_range"),
        (
            {
                "prim_path": "/World/Sensors/WristRGBD",
                "parent_prim_path": "/World/Robot/Wrist",
            },
            "must be under parent_prim_path",
        ),
        ({"env_ids": ()}, "non-empty tuple"),
        ({"env_ids": (0, 0)}, "duplicates"),
        ({"env_ids": (True,)}, "integer"),
        ({"env_ids": (-1,)}, "nonnegative"),
    ),
)
def test_typed_camera_rejects_invalid_runtime_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _camera(**overrides)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"fx": 0.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"fx": 1.0, "fy": float("nan"), "cx": 0.0, "cy": 0.0},
    ),
)
def test_typed_camera_rejects_invalid_intrinsics(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SensorCameraIntrinsicsSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"save_dir": ""},
        {"foxglove_topic_prefix": "camera/main"},
        {"foxglove_live_port": 0},
        {"foxglove_mcap_path": ""},
    ),
)
def test_typed_camera_rejects_invalid_output(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SensorCameraOutputSettings(**kwargs)  # type: ignore[arg-type]


def test_scene_sensor_settings_reject_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        SceneSensorSettings(cameras=(_camera(), _camera()))


def test_mirror_camera_scope_rejects_replicated_selector() -> None:
    settings = SceneSensorSettings(cameras=(_camera(env_ids=(0, 2)),))

    with pytest.raises(ValueError, match="env_ids.*replicated environments"):
        settings.validate_mirror_camera_scope()
