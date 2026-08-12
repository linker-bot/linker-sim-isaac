from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import linkerbot_sim.sensors.camera.runtime as camera_runtime
from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.sensors.camera.config import (
    SensorCameraIntrinsicsSettings,
    SensorCameraSettings,
)
from linkerbot_sim.sensors.camera.runtime import (
    SensorCameraRuntime,
    _NewtonSyntheticDataCamera,
    _quat_from_extrinsic_xyz_rpy,
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


class FakeOpticalCamera:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_clipping_ranges(self, near: float, far: float) -> None:
        self.calls.append(("set_clipping_ranges", (near, far)))

    def set_focal_lengths(self, focal_length: float) -> None:
        self.calls.append(("set_focal_lengths", focal_length))

    def set_apertures(self, horizontal: float, vertical: float) -> None:
        self.calls.append(("set_apertures", (horizontal, vertical)))

    def get_focal_lengths(self):
        return FakeWarpArray([[2.0]])

    def get_apertures(self):
        return FakeWarpArray([[4.0]]), FakeWarpArray([[3.0]])


class FakeRtxCamera:
    def __init__(self, path: str, **kwargs: object) -> None:
        self.path = path
        self.kwargs = kwargs
        self.camera = FakeOpticalCamera()

    def get_world_poses(self):
        return (
            FakeWarpArray([[1.0, 2.0, 3.0]]),
            FakeWarpArray([[0.5, 0.5, -0.5, -0.5]]),
        )


class FakeCameraSensor:
    def __init__(
        self,
        authoring_camera: FakeRtxCamera,
        *,
        resolution: tuple[int, int],
        annotators: list[str],
    ) -> None:
        self.authoring_object = authoring_camera
        self.camera = authoring_camera.camera
        self.resolution = resolution
        self.annotators = annotators
        self.data_calls: list[str] = []

    def get_data(self, annotator: str):
        self.data_calls.append(annotator)
        return FakeWarpArray([[1.0]]), {"annotator": annotator}


class FakeWarpArray:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)
        self.numpy_calls = 0

    def numpy(self) -> np.ndarray:
        self.numpy_calls += 1
        return self._values.copy()


def _settings(*, resolution: tuple[int, int] = (320, 240)):
    return SensorCameraSettings(
        name="wrist_rgbd",
        prim_path="/World/Robot/Wrist/WristRGBD",
        parent_prim_path="/World/Robot/Wrist",
        pose_xyz=(0.0, 0.0, 0.08),
        pose_rpy=(0.1, 0.2, 0.3),
        resolution=resolution,
        frequency=20.0,
        modalities=("rgb", "depth"),
        clipping_range=(0.02, 4.0),
        intrinsics=SensorCameraIntrinsicsSettings(
            fx=320.0,
            fy=240.0,
            cx=resolution[0] / 2.0,
            cy=resolution[1] / 2.0,
        ),
    )


def _create_runtime(settings=None) -> SensorCameraRuntime:
    return create_sensor_camera_runtime(
        stage=FakeStage({"/World/Robot/Wrist"}),
        settings=settings or _settings(),
        physics_backend="physx",
        rtx_camera_type=FakeRtxCamera,
        camera_sensor_type=FakeCameraSensor,
        array_factory=np.asarray,
        quat_from_rpy=lambda _rpy: (1.0, 0.0, 0.0, 0.0),
    )


def test_create_sensor_camera_runtime_uses_experimental_rtx_api() -> None:
    runtime = _create_runtime()
    sensor = runtime.camera
    authoring = sensor.authoring_object

    assert runtime.name == "wrist_rgbd"
    assert runtime.prim_path == "/World/Robot/Wrist/WristRGBD"
    assert authoring.path == "/World/Robot/Wrist/WristRGBD"
    assert authoring.kwargs["tick_rate"] == 20.0
    assert np.asarray(authoring.kwargs["translations"]).shape == (1, 3)
    assert np.asarray(authoring.kwargs["orientations"]).shape == (1, 4)
    assert np.asarray(authoring.kwargs["orientations"]).tolist() == [
        [0.5, 0.5, -0.5, -0.5]
    ]
    assert authoring.kwargs["schemas"] == ["OmniLensDistortionOpenCvPinholeAPI"]
    attributes = authoring.kwargs["attributes"]
    assert attributes["omni:lensdistortion:opencvPinhole:fx"] == 320.0
    assert tuple(attributes["omni:lensdistortion:opencvPinhole:imageSize"]) == (
        320,
        240,
    )

    assert sensor.resolution == (240, 320)
    assert sensor.annotators == ["rgb", "distance_to_image_plane"]
    assert sensor.camera.calls == [
        ("set_clipping_ranges", (0.02, 4.0)),
        ("set_focal_lengths", 1.0),
        ("set_apertures", (1.0, 1.0)),
    ]

    initialize_sensor_camera_runtimes((runtime,))
    assert sensor.camera.calls[-1] == ("set_apertures", (1.0, 1.0))
    assert runtime.get_intrinsics_matrix() == (
        (320.0, 0.0, 160.0),
        (0.0, 240.0, 120.0),
        (0.0, 0.0, 1.0),
    )

    runtime.get_rgb(device="cpu")
    runtime.get_depth(device="cpu")
    assert sensor.data_calls == ["rgb", "distance_to_image_plane"]


def test_non_square_resolution_swaps_only_camera_sensor_constructor_order() -> None:
    runtime = _create_runtime(_settings(resolution=(848, 480)))
    sensor = runtime.camera
    attributes = sensor.authoring_object.kwargs["attributes"]

    assert sensor.resolution == (480, 848)
    assert tuple(attributes["omni:lensdistortion:opencvPinhole:imageSize"]) == (
        848,
        480,
    )
    assert runtime.settings.resolution == (848, 480)


def test_runtime_copies_warp_annotator_data_to_cpu() -> None:
    runtime = _create_runtime()
    payload = FakeWarpArray(np.ones((2, 3, 4), dtype=np.uint8))
    runtime.camera.get_data = lambda annotator: (payload, {"name": annotator})

    result = runtime.get_modality_data("rgb", device="cpu", clone=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3, 4)
    assert payload.numpy_calls == 1
    assert result is not payload._values


def test_runtime_unbatches_pose_and_preserves_world_camera_axes() -> None:
    runtime = _create_runtime()

    position, orientation = runtime.get_world_pose()

    assert position == (1.0, 2.0, 3.0)
    assert orientation == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_rpy_conversion_matches_legacy_extrinsic_xyz_convention() -> None:
    assert _quat_from_extrinsic_xyz_rpy((0.1, 0.2, 0.3)) == pytest.approx(
        (0.9833474433, 0.0342707986, 0.1060205111, 0.1435721750)
    )


def test_runtime_derives_intrinsics_from_experimental_optical_api() -> None:
    settings = SensorCameraSettings(
        name="world",
        prim_path="/World/Camera",
        resolution=(800, 300),
    )
    sensor = FakeCameraSensor(
        FakeRtxCamera("/World/Camera"),
        resolution=(300, 800),
        annotators=["rgb"],
    )
    runtime = SensorCameraRuntime(settings=settings, camera=sensor)

    assert runtime.get_intrinsics_matrix() == (
        (400.0, 0.0, 400.0),
        (0.0, 200.0, 150.0),
        (0.0, 0.0, 1.0),
    )


def test_create_sensor_camera_runtimes_skips_disabled_cameras() -> None:
    settings = SceneSensorSettings(
        cameras=(
            SensorCameraSettings(
                name="disabled",
                enabled=False,
                prim_path="/World/Camera",
            ),
        )
    )

    runtimes = create_sensor_camera_runtimes(
        stage=FakeStage(set()),
        sensors=settings,
        physics_backend="physx",
    )

    assert runtimes == ()


def test_create_sensor_camera_runtimes_rolls_back_previous_camera(
    monkeypatch,
) -> None:
    settings = SceneSensorSettings(
        cameras=(
            SensorCameraSettings(name="first", prim_path="/World/FirstCamera"),
            SensorCameraSettings(name="second", prim_path="/World/SecondCamera"),
        )
    )
    events: list[str] = []
    primary = RuntimeError("second camera failed")

    class _FirstCamera:
        def close(self) -> None:
            events.append("close:first")
            raise ValueError("first camera cleanup failed")

    def create(*, settings, **_kwargs):
        events.append(f"create:{settings.name}")
        if settings.name == "second":
            raise primary
        return _FirstCamera()

    monkeypatch.setattr(camera_runtime, "create_sensor_camera_runtime", create)

    with pytest.raises(RuntimeError, match="second camera failed") as caught:
        create_sensor_camera_runtimes(
            stage=FakeStage(set()),
            sensors=settings,
            physics_backend="newton",
        )

    assert caught.value is primary
    assert events == ["create:first", "create:second", "close:first"]
    assert getattr(primary, "__notes__", ()) == [
        "previous camera cleanup also failed: ValueError: first camera cleanup failed"
    ]


def test_create_sensor_camera_runtimes_rejects_replicated_scope_in_mirror() -> None:
    settings = SceneSensorSettings(
        cameras=(
            SensorCameraSettings(
                name="scoped",
                prim_path="/World/Camera",
                env_ids=(0,),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=r"sensors\.cameras\.scoped\.env_ids is only valid for replicated environments",
    ):
        create_sensor_camera_runtimes(
            stage=FakeStage(set()),
            sensors=settings,
            physics_backend="physx",
        )


def test_create_sensor_camera_runtime_rejects_missing_parent_prim() -> None:
    settings = SensorCameraSettings(
        name="wrist_rgbd",
        prim_path="/World/Robot/Wrist/WristRGBD",
        parent_prim_path="/World/Robot/Wrist",
    )

    with pytest.raises(ValueError, match="parent_prim_path does not exist"):
        create_sensor_camera_runtime(
            stage=FakeStage(set()),
            settings=settings,
            physics_backend="physx",
            rtx_camera_type=FakeRtxCamera,
            camera_sensor_type=FakeCameraSensor,
            array_factory=np.asarray,
            quat_from_rpy=lambda _rpy: (1.0, 0.0, 0.0, 0.0),
        )


def test_direct_camera_constructor_cleans_partially_created_window(
    monkeypatch,
) -> None:
    events: list[str] = []
    settings = SensorCameraSettings(
        name="direct",
        prim_path="/World/DirectCamera",
    )

    class _Window:
        def destroy(self) -> None:
            events.append("destroy")

    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_define_camera_prim",
        lambda self: None,
    )

    def fail_after_window(self) -> None:
        self._window = _Window()
        raise RuntimeError("viewport API unavailable")

    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_create_viewport",
        fail_after_window,
    )

    with pytest.raises(RuntimeError, match="viewport API unavailable"):
        _NewtonSyntheticDataCamera(stage=object(), settings=settings)

    assert events == ["destroy"]


def test_direct_camera_constructor_defers_syntheticdata_initialization(
    monkeypatch,
) -> None:
    events: list[str] = []
    settings = SensorCameraSettings(
        name="direct",
        prim_path="/World/DirectCamera",
    )

    class _Window:
        def destroy(self) -> None:
            events.append("destroy")

    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_define_camera_prim",
        lambda self: events.append("camera_prim"),
    )

    def create_viewport(self) -> None:
        events.append("viewport")
        self._window = _Window()
        self._viewport = object()

    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_create_viewport",
        create_viewport,
    )

    camera = _NewtonSyntheticDataCamera(stage=object(), settings=settings)

    assert events == ["viewport"]
    assert camera._initialized is False
    camera.close()
    assert events == ["viewport", "destroy"]


def test_direct_camera_viewport_disables_only_its_camera_navigation(
    monkeypatch,
) -> None:
    calls: list[tuple[object, bool]] = []
    viewport = SimpleNamespace(updates_enabled=False, resolution=None)

    class _Window:
        viewport_api = viewport

    window = _Window()
    utility = ModuleType("omni.kit.viewport.utility")
    utility.create_viewport_window = lambda **_kwargs: window
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", utility)
    monkeypatch.setattr(
        camera_runtime,
        "set_viewport_camera_navigation_enabled",
        lambda actual_window, *, enabled: calls.append((actual_window, enabled)),
    )

    camera = _NewtonSyntheticDataCamera.__new__(_NewtonSyntheticDataCamera)
    camera._settings = SensorCameraSettings(
        name="world_rgbd",
        prim_path="/World/WorldRGBD",
        resolution=(640, 480),
    )
    camera._window = None
    camera._viewport = None

    camera._create_viewport()

    assert calls == [(window, False)]
    assert camera._window is window
    assert camera._viewport is viewport
    assert viewport.updates_enabled is True
    assert viewport.resolution == (640, 480)


def test_direct_camera_accepts_robot_parent_created_before_initialize(
    monkeypatch,
) -> None:
    events: list[str] = []
    parent_path = "/World/Robot/Wrist"
    settings = SensorCameraSettings(
        name="wrist",
        parent_prim_path=parent_path,
        prim_path=f"{parent_path}/Camera",
        modalities=("rgb",),
    )
    stage = FakeStage(set())
    viewport = SimpleNamespace(updates_enabled=False)

    class _Window:
        viewport_api = viewport

        def destroy(self) -> None:
            events.append("destroy")

    def reserve_viewport(self) -> None:
        events.append("viewport_reserved")
        self._window = _Window()
        self._viewport = viewport

    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_create_viewport",
        reserve_viewport,
    )
    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_define_camera_prim",
        lambda self: events.append("camera_prim"),
    )
    syntheticdata = ModuleType("omni.syntheticdata")
    syntheticdata._syntheticdata = SimpleNamespace(
        SensorType=SimpleNamespace(Rgb="rgb", DistanceToImagePlane="depth")
    )
    syntheticdata.sensors = SimpleNamespace(
        create_or_retrieve_sensor=lambda _viewport, sensor_type: events.append(
            f"sensor:{sensor_type}"
        ),
        disable_sensors=lambda _viewport, _types: events.append("sensors_disabled"),
    )
    monkeypatch.setitem(sys.modules, "omni.syntheticdata", syntheticdata)
    omni = sys.modules.get("omni")
    if omni is None:
        omni = ModuleType("omni")
        omni.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setattr(omni, "syntheticdata", syntheticdata, raising=False)

    runtime = create_sensor_camera_runtime(
        stage=stage,
        settings=settings,
        physics_backend="newton",
    )

    assert events == ["viewport_reserved"]
    stage._valid_paths.add(parent_path)
    runtime.camera.initialize()
    assert events == ["viewport_reserved", "camera_prim", "sensor:rgb"]
    assert str(viewport.camera_path) == settings.prim_path
    assert runtime.camera._initialized is True
    runtime.close()
    assert events[-2:] == ["sensors_disabled", "destroy"]


def test_direct_camera_rejects_parent_still_missing_at_initialize(
    monkeypatch,
) -> None:
    events: list[str] = []
    settings = SensorCameraSettings(
        name="wrist",
        parent_prim_path="/World/Robot/Wrist",
        prim_path="/World/Robot/Wrist/Camera",
    )
    viewport = SimpleNamespace(updates_enabled=False)

    class _Window:
        viewport_api = viewport

        def destroy(self) -> None:
            events.append("destroy")

    def reserve_viewport(self) -> None:
        events.append("viewport_reserved")
        self._window = _Window()
        self._viewport = viewport

    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_create_viewport",
        reserve_viewport,
    )
    monkeypatch.setattr(
        _NewtonSyntheticDataCamera,
        "_define_camera_prim",
        lambda self: events.append("camera_prim"),
    )

    runtime = create_sensor_camera_runtime(
        stage=FakeStage(set()),
        settings=settings,
        physics_backend="newton",
    )

    with pytest.raises(ValueError, match="parent_prim_path does not exist"):
        runtime.camera.initialize()
    assert events == ["viewport_reserved"]
    runtime.close()
    assert events == ["viewport_reserved", "destroy"]


def test_direct_camera_close_attempts_destroy_after_disable_failure(
    monkeypatch,
) -> None:
    events: list[str] = []
    syntheticdata = ModuleType("omni.syntheticdata")

    def fail_disable(_viewport: object, _types: object) -> None:
        events.append("disable")
        raise RuntimeError("disable failed")

    syntheticdata.sensors = SimpleNamespace(disable_sensors=fail_disable)
    monkeypatch.setitem(sys.modules, "omni.syntheticdata", syntheticdata)
    omni = sys.modules.get("omni")
    if omni is None:
        omni = ModuleType("omni")
        omni.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setattr(omni, "syntheticdata", syntheticdata, raising=False)

    class _Window:
        def destroy(self) -> None:
            events.append("destroy")

    camera = _NewtonSyntheticDataCamera.__new__(_NewtonSyntheticDataCamera)
    camera._closed = False
    camera._initialized = True
    camera._viewport = object()
    camera._window = _Window()
    camera._sensor_types = {"rgb": object()}
    camera._stage = object()

    with pytest.raises(RuntimeError, match="disable failed"):
        camera.close()

    assert events == ["disable", "destroy"]
    assert camera._closed is False
    camera.close()
    assert camera._closed is True


def test_direct_camera_render_selection_updates_owned_viewport() -> None:
    viewport = SimpleNamespace(updates_enabled=True)
    camera = _NewtonSyntheticDataCamera.__new__(_NewtonSyntheticDataCamera)
    camera._closed = False
    camera._viewport = viewport

    camera.set_render_active(False)
    assert viewport.updates_enabled is False
    camera.set_render_active(True)
    assert viewport.updates_enabled is True


def test_direct_camera_uses_four_ticks_for_hidden_product_history() -> None:
    assert _NewtonSyntheticDataCamera.render_update_count == 4


def test_direct_camera_is_owned_and_closed_by_returned_runtime(
    monkeypatch,
) -> None:
    events: list[str] = []
    camera = SimpleNamespace(close=lambda: events.append("close"))
    monkeypatch.setattr(
        camera_runtime,
        "_NewtonSyntheticDataCamera",
        lambda **_kwargs: camera,
    )
    runtime = camera_runtime._create_newton_camera_runtime(
        stage=object(),
        settings=_settings(),
    )
    assert runtime.camera is camera
    assert events == []
    runtime.close()
    assert events == ["close"]
