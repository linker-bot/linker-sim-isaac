from __future__ import annotations

import pytest

from linkerbot_sim.configuration.common import ConfigurationError
from linkerbot_sim.configuration.catalog import load_mirror_config
from linkerbot_sim.configuration.scenes import CameraSettings
from linkerbot_sim.mirror.scene_assembly import _mirror_scene_runtime_settings


def _camera_mapping() -> dict[str, object]:
    return {
        "id": "main",
        "parent_prim_path": "/World",
        "prim_path": "/World/MainCamera",
        "pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        "resolution": [640, 480],
        "frequency_hz": 30.0,
        "modalities": ["rgb", "depth"],
        "clipping_range_m": [0.01, 5.0],
    }


def test_canonical_mirror_camera_intrinsics_reach_sensor_runtime_settings() -> None:
    config = load_mirror_config("physx_cpu")
    camera = config.scene.cameras[0]

    assert camera.intrinsics is not None
    assert (
        camera.intrinsics.fx,
        camera.intrinsics.fy,
        camera.intrinsics.cx,
        camera.intrinsics.cy,
    ) == (307.5, 308.0, 160.0, 120.0)

    sensor = _mirror_scene_runtime_settings(
        config.scene,
        camera_output=config.outputs.camera,
    ).sensors.cameras[0]
    assert sensor.intrinsics is not None
    assert sensor.intrinsics.matrix() == (
        (307.5, 0.0, 160.0),
        (0.0, 308.0, 120.0),
        (0.0, 0.0, 1.0),
    )


def test_mirror_camera_intrinsics_are_optional_as_a_complete_mapping() -> None:
    camera = CameraSettings.from_mapping(_camera_mapping(), label="scene.cameras[0]")

    assert camera.intrinsics is None


@pytest.mark.parametrize(
    ("intrinsics", "expected"),
    [
        ({"fx": 615.0, "fy": 616.0, "cx": 320.0}, "cy"),
        ({"fx": 0.0, "fy": 616.0, "cx": 320.0, "cy": 240.0}, "fx"),
    ],
)
def test_mirror_camera_rejects_invalid_intrinsics(
    intrinsics: dict[str, float], expected: str
) -> None:
    mapping = _camera_mapping()
    mapping["intrinsics"] = intrinsics

    with pytest.raises(ConfigurationError, match=expected):
        CameraSettings.from_mapping(mapping, label="scene.cameras[0]")
