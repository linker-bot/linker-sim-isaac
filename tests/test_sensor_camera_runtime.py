from __future__ import annotations

from linkerbot_sim.sensors.camera_config import SceneSensorSettings
from linkerbot_sim.sensors.camera_runtime import (
    create_sensor_camera_runtime,
    create_sensor_camera_runtimes,
    initialize_sensor_camera_runtimes,
)


class FakePrim:
    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid


class FakeStage:
    def __init__(self, valid_paths: set[str]) -> None:
        self._valid_paths = valid_paths

    def GetPrimAtPath(self, path: str) -> FakePrim:
        return FakePrim(path in self._valid_paths)


class FakeCamera:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, object]] = []

    def initialize(self) -> None:
        self.calls.append(("initialize", None))

    def set_clipping_range(
        self, *, near_distance: float, far_distance: float
    ) -> None:
        self.calls.append(("set_clipping_range", (near_distance, far_distance)))

    def set_focal_length(self, focal_length: float) -> None:
        self.calls.append(("set_focal_length", focal_length))

    def set_horizontal_aperture(
        self, horizontal_aperture: float, *, maintain_square_pixels: bool = True
    ) -> None:
        self.calls.append(
            (
                "set_horizontal_aperture",
                (horizontal_aperture, maintain_square_pixels),
            )
        )

    def set_vertical_aperture(
        self, vertical_aperture: float, *, maintain_square_pixels: bool = True
    ) -> None:
        self.calls.append(
            (
                "set_vertical_aperture",
                (vertical_aperture, maintain_square_pixels),
            )
        )

    def set_opencv_pinhole_properties(
        self, *, cx: float, cy: float, fx: float, fy: float
    ) -> None:
        self.calls.append(
            (
                "set_opencv_pinhole_properties",
                {"cx": cx, "cy": cy, "fx": fx, "fy": fy},
            )
        )

    def add_rgb_to_frame(self) -> None:
        self.calls.append(("add_rgb_to_frame", None))

    def add_distance_to_image_plane_to_frame(self) -> None:
        self.calls.append(("add_distance_to_image_plane_to_frame", None))

    def get_rgb(self, *, device: str | None = None) -> str:
        return f"rgb:{device}"

    def get_depth(self, *, device: str | None = None) -> str:
        return f"depth:{device}"


def test_create_sensor_camera_runtime_initializes_camera_wrapper() -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "wrist_rgbd": {
                        "prim_path": "/World/Robot/Wrist/WristRGBD",
                        "parent_prim_path": "/World/Robot/Wrist",
                        "pose": {
                            "xyz": [0.0, 0.0, 0.08],
                            "rpy": [0.1, 0.2, 0.3],
                        },
                        "resolution": [320, 240],
                        "frequency": 20.0,
                        "modalities": ["rgb", "depth"],
                        "clipping_range": [0.02, 4.0],
                        "intrinsics": {
                            "fx": 320.0,
                            "fy": 240.0,
                            "cx": 160.0,
                            "cy": 120.0,
                        },
                    }
                }
            }
        }
    )

    runtime = create_sensor_camera_runtime(
        stage=FakeStage({"/World/Robot/Wrist"}),
        settings=settings.cameras[0],
        camera_type=FakeCamera,
        array_factory=tuple,
        quat_from_rpy=lambda rpy: ("quat", rpy),
    )

    assert runtime.name == "wrist_rgbd"
    assert runtime.prim_path == "/World/Robot/Wrist/WristRGBD"
    assert runtime.camera.kwargs == {
        "prim_path": "/World/Robot/Wrist/WristRGBD",
        "name": "wrist_rgbd",
        "dt": 0.05,
        "resolution": (320, 240),
        "translation": (0.0, 0.0, 0.08),
        "orientation": ("quat", (0.1, 0.2, 0.3)),
    }

    initialize_sensor_camera_runtimes((runtime,))

    assert runtime.camera.calls == [
        ("initialize", None),
        ("set_focal_length", 1.0),
        ("set_horizontal_aperture", (1.0, False)),
        ("set_vertical_aperture", (1.0, False)),
        (
            "set_opencv_pinhole_properties",
            {"cx": 160.0, "cy": 120.0, "fx": 320.0, "fy": 240.0},
        ),
        ("set_clipping_range", (0.02, 4.0)),
        ("add_rgb_to_frame", None),
        ("add_distance_to_image_plane_to_frame", None),
    ]
    assert runtime.get_rgb(device="cpu") == "rgb:cpu"
    assert runtime.get_depth(device="cpu") == "depth:cpu"
    assert runtime.get_intrinsics_matrix() == (
        (320.0, 0.0, 160.0),
        (0.0, 240.0, 120.0),
        (0.0, 0.0, 1.0),
    )


def test_create_sensor_camera_runtimes_skips_disabled_cameras() -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "disabled": {
                        "enabled": False,
                        "prim_path": "/World/Camera",
                    }
                }
            }
        }
    )

    runtimes = create_sensor_camera_runtimes(
        stage=FakeStage(set()),
        sensors=settings,
    )

    assert runtimes == ()


def test_create_sensor_camera_runtime_rejects_missing_parent_prim() -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "wrist_rgbd": {
                        "prim_path": "/World/Robot/Wrist/WristRGBD",
                        "parent_prim_path": "/World/Robot/Wrist",
                    }
                }
            }
        }
    )

    try:
        create_sensor_camera_runtime(
            stage=FakeStage(set()),
            settings=settings.cameras[0],
            camera_type=FakeCamera,
            array_factory=tuple,
            quat_from_rpy=lambda rpy: ("quat", rpy),
        )
    except ValueError as exc:
        assert "parent_prim_path does not exist" in str(exc)
    else:
        raise AssertionError("missing parent prim accepted")
