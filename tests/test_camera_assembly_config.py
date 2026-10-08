from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.configuration.catalog import (
    _ConfigurationGraphReader,
    _resolve_camera_profiles,
)
from linkerbot_sim.configuration.common import ConfigurationError
from linkerbot_sim.sensors.camera.config import SensorCameraSettings
from linkerbot_sim.sensors.camera.runtime import (
    _camera_to_usd_quaternion,
    _resolve_parent_link,
)

ROOT = Path(__file__).resolve().parents[1]


def test_nominal_profiles_preserve_separate_streams_and_rates():
    config = load_mirror_config("camera_workstation")
    cameras = {c.camera_id: c for c in config.scene.cameras}
    assert cameras["left_rgb"].resolution == (1280, 800)
    assert cameras["left_rgb"].frequency_hz == 60
    assert cameras["left_depth"].frequency_hz == 30
    assert cameras["left_rgb"].parent_link != cameras["left_depth"].parent_link
    assert cameras["side"].resolution == (1920, 1080)
    assert cameras["side"].frequency_hz == 60
    assert cameras["top"].resolution == (1280, 720)
    assert cameras["left_depth"].clipping_range_m[0] < 0.25
    assert config.sources["camera.0"].name == "gemini335l_rgb_60.yaml"


def test_camera_leaf_cannot_silently_override_scene_values():
    reader = _ConfigurationGraphReader(ROOT / "configs")
    with pytest.raises(ConfigurationError, match="one writer"):
        _resolve_camera_profiles(
            {
                "cameras": [
                    {"camera_profile": "gemini335l_rgb_60", "resolution": [1, 1]}
                ]
            },
            reader,
        )


def test_mount_resolves_only_one_named_link_in_the_selected_instance():
    stage = Usd.Stage.CreateInMemory()
    for path in (
        "/World/Left/asset/camera_rgb_optical",
        "/World/Right/other/camera_rgb_optical",
    ):
        UsdGeom.Xform.Define(stage, path)
    settings = SensorCameraSettings(
        name="left",
        prim_path="/World/Left/Sensor",
        parent_prim_path="/World/Left",
        parent_link="camera_rgb_optical",
        pose_axes="opencv",
    )
    resolved = _resolve_parent_link(stage, settings)
    assert resolved.parent_prim_path == "/World/Left/asset/camera_rgb_optical"
    assert resolved.prim_path == resolved.parent_prim_path + "/Sensor"
    assert _resolve_parent_link(stage, resolved) == resolved
    with pytest.raises(ValueError, match="found 0"):
        _resolve_parent_link(stage, replace(settings, parent_link="missing"))
    UsdGeom.Xform.Define(stage, "/World/Left/duplicate/camera_rgb_optical")
    with pytest.raises(ValueError, match="found 2"):
        _resolve_parent_link(stage, settings)


def test_optical_identity_points_forward_in_opencv_coordinates():
    from scipy.spatial.transform import Rotation

    settings = SensorCameraSettings(
        name="sensor", prim_path="/World/Sensor", pose_axes="opencv"
    )
    q = _camera_to_usd_quaternion(settings)
    rotation = Rotation.from_quat([*q[1:], q[0]])
    np.testing.assert_allclose(rotation.apply([0, 0, -1]), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(rotation.apply([0, 1, 0]), [0, -1, 0], atol=1e-12)


@pytest.mark.parametrize("axes", ("ros", "USD", ""))
def test_unknown_camera_axis_conventions_fail(axes):
    with pytest.raises(ValueError, match="pose_axes"):
        SensorCameraSettings(name="sensor", prim_path="/World/Sensor", pose_axes=axes)
