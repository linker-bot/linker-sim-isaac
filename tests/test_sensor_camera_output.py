from __future__ import annotations

import json
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from linkerbot_sim.sensors.camera_config import SceneSensorSettings
from linkerbot_sim.sensors.camera_frame import sample_camera_frames
from linkerbot_sim.sensors.camera_observer import CameraFrameObserver
from linkerbot_sim.sensors.camera_observer import start_camera_output
from linkerbot_sim.sensors.camera_recorder import (
    CameraFramePublisher,
    OfflineCameraFrameSink,
)
from linkerbot_sim.sensors.camera_runtime import SensorCameraRuntime


class _FakeCamera:
    def __init__(self) -> None:
        self.rgb = np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        self.depth = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    def get_rgb(self, *, device: str | None = None):
        del device
        return self.rgb

    def get_depth(self, *, device: str | None = None):
        del device
        return self.depth

    def get_intrinsics_matrix(self, *, device: str | None = None):
        del device
        return np.eye(3, dtype=float)

    def get_world_pose(self):
        return np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 0.0, 0.0, 0.0])


class _NotReadyThenReadyCamera(_FakeCamera):
    def __init__(self) -> None:
        super().__init__()
        self.ready = False

    def get_rgb(self, *, device: str | None = None):
        del device
        if not self.ready:
            return None
        return self.rgb

    def get_depth(self, *, device: str | None = None):
        del device
        if not self.ready:
            return np.asarray(())
        return self.depth


class _FakeWorld:
    def get_physics_dt(self) -> float:
        return 0.1


class _CollectingPublisher:
    def __init__(self) -> None:
        self.frames = []

    def publish(self, frame) -> None:
        self.frames.append(frame)


def _camera_runtime() -> SensorCameraRuntime:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "wrist_rgbd": {
                        "prim_path": "/World/Camera",
                        "frequency": 5.0,
                        "modalities": ["rgb", "depth"],
                    }
                }
            }
        }
    ).cameras[0]
    return SensorCameraRuntime(settings=settings, camera=_FakeCamera())


def _camera_runtime_with_camera(camera) -> SensorCameraRuntime:
    runtime = _camera_runtime()
    return SensorCameraRuntime(settings=runtime.settings, camera=camera)


def test_sample_camera_frames_normalizes_payload_and_metadata() -> None:
    frame_indices: dict[tuple[str, str], int] = {}
    frames = sample_camera_frames(
        _camera_runtime(),
        frame_indices=frame_indices,
        simulation_step=7,
        time_s=0.8,
    )

    assert [frame.modality for frame in frames] == ["rgb", "depth"]
    assert frames[0].data.dtype == np.uint8
    assert frames[0].data.tolist()[0][0] == [255, 0, 0]
    assert frames[1].data.dtype == np.float32
    metadata = frames[0].metadata(relative_path="rgb/000000.ppm")
    assert metadata["camera_name"] == "wrist_rgbd"
    assert metadata["relative_path"] == "rgb/000000.ppm"
    assert metadata["camera_position_world"] == [1.0, 2.0, 3.0]
    assert metadata["camera_orientation_world"] == [1.0, 0.0, 0.0, 0.0]


def test_sample_camera_frames_skips_not_ready_payloads_without_indexing() -> None:
    camera = _NotReadyThenReadyCamera()
    runtime = _camera_runtime_with_camera(camera)
    frame_indices: dict[tuple[str, str], int] = {}

    assert (
        sample_camera_frames(
            runtime,
            frame_indices=frame_indices,
            simulation_step=0,
            time_s=0.1,
        )
        == ()
    )
    assert frame_indices == {}

    camera.ready = True
    frames = sample_camera_frames(
        runtime,
        frame_indices=frame_indices,
        simulation_step=1,
        time_s=0.2,
    )

    assert [(frame.modality, frame.frame_index) for frame in frames] == [
        ("rgb", 0),
        ("depth", 0),
    ]


def test_offline_camera_frame_sink_writes_rgb_depth_and_metadata() -> None:
    frames = sample_camera_frames(
        _camera_runtime(),
        frame_indices={},
        simulation_step=0,
        time_s=0.1,
    )
    with TemporaryDirectory() as tmp_dir:
        sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=tmp_dir)
        try:
            for frame in frames:
                sink.publish(frame)
        finally:
            sink.close()

        root = Path(tmp_dir)
        rgb_path = root / "rgb" / "000000.ppm"
        depth_path = root / "depth" / "000000.npy"
        metadata_path = root / "metadata.jsonl"
        assert rgb_path.read_bytes().startswith(b"P6\n2 2\n255\n")
        assert np.load(depth_path).shape == (2, 2)
        rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["relative_path"] for row in rows] == [
            "rgb/000000.ppm",
            "depth/000000.npy",
        ]


def test_camera_frame_observer_samples_by_frequency() -> None:
    publisher = _CollectingPublisher()
    observer = CameraFrameObserver(
        cameras=(_camera_runtime(),),
        publisher=publisher,
    )

    observer.observe(_FakeWorld(), step=0, phase="a")
    observer.observe(_FakeWorld(), step=1, phase="b")
    observer.observe(_FakeWorld(), step=2, phase="c")

    assert [(frame.modality, frame.frame_index) for frame in publisher.frames] == [
        ("rgb", 0),
        ("depth", 0),
        ("rgb", 1),
        ("depth", 1),
    ]


def test_camera_frame_publisher_writes_in_background() -> None:
    frames = sample_camera_frames(
        _camera_runtime(),
        frame_indices={},
        simulation_step=0,
        time_s=0.1,
    )
    with TemporaryDirectory() as tmp_dir:
        sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=tmp_dir)
        publisher = CameraFramePublisher(sink=sink, max_queue_size=4)
        publisher.start()
        for frame in frames:
            publisher.publish(frame)
        publisher.close()

        metadata_path = Path(tmp_dir) / "metadata.jsonl"
        assert len(metadata_path.read_text(encoding="utf-8").splitlines()) == 2


def test_start_camera_output_returns_none_when_no_outputs() -> None:
    assert start_camera_output((_camera_runtime(),)) is None
