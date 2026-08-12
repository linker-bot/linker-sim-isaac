from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
import threading
import time
from typing import cast

import numpy as np
import pytest

import linkerbot_sim.sensors.camera.recorder as recorder_module
from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.sensors.camera.config import (
    SensorCameraOutputSettings,
    SensorCameraSettings,
)
from linkerbot_sim.sensors.camera.frame import CameraFrame, sample_camera_frames
from linkerbot_sim.sensors.camera.observer import (
    CameraFrameObserver,
    CameraPublisherSettings,
)
from linkerbot_sim.sensors.camera.observer import open_prepared_camera_output
from linkerbot_sim.sensors.camera.observer import prepare_camera_output
from linkerbot_sim.sensors.camera.observer import start_camera_output
from linkerbot_sim.sensors.camera.recorder import (
    CameraFrameQueueFullError,
    CameraFramePublisher,
    CameraOutputQuotaExceededError,
    CompositeCameraFrameSink,
    OfflineCameraFrameSink,
)
from linkerbot_sim.sensors.camera.runtime import SensorCameraRuntime
from linkerbot_sim.utils.output_paths import apply_output_path_plans


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


class _ExperimentalDepthCamera(_FakeCamera):
    def get_depth(self, *, device: str | None = None):
        del device
        return self.depth[:, :, np.newaxis]


class _FakeWorld:
    def get_physics_dt(self) -> float:
        return 0.1


class _CollectingPublisher:
    def __init__(self) -> None:
        self.frames = []

    def publish(self, frame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        pass


def _camera_settings(
    *,
    name: str = "wrist_rgbd",
    prim_path: str = "/World/Camera",
    frequency: float = 30.0,
    modalities: tuple[str, ...] = ("rgb",),
    save_dir: str | None = None,
) -> SensorCameraSettings:
    return SensorCameraSettings(
        name=name,
        prim_path=prim_path,
        frequency=frequency,
        modalities=modalities,
        output=SensorCameraOutputSettings(save_dir=save_dir),
    )


def _camera_runtime() -> SensorCameraRuntime:
    settings = _camera_settings(
        frequency=5.0,
        modalities=("rgb", "depth"),
    )
    return SensorCameraRuntime(settings=settings, camera=_FakeCamera())


def _camera_runtime_with_camera(camera) -> SensorCameraRuntime:
    runtime = _camera_runtime()
    return SensorCameraRuntime(settings=runtime.settings, camera=camera)


def _rgb_frame(*, value: int, simulation_step: int) -> CameraFrame:
    return CameraFrame(
        camera_name="wrist_rgbd",
        modality="rgb",
        frame_index=0,
        simulation_step=simulation_step,
        time_s=float(simulation_step) * 0.1,
        data=np.full((1, 1, 3), value, dtype=np.uint8),
    )


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


def test_sample_camera_frames_squeezes_experimental_depth_channel() -> None:
    runtime = _camera_runtime_with_camera(_ExperimentalDepthCamera())

    frames = sample_camera_frames(
        runtime,
        frame_indices={},
        simulation_step=0,
        time_s=0.1,
    )

    depth = next(frame for frame in frames if frame.modality == "depth")
    assert depth.data.shape == (2, 2)
    assert depth.data.dtype == np.float32


def test_offline_camera_frame_sink_writes_rgb_depth_and_metadata() -> None:
    frames = sample_camera_frames(
        _camera_runtime(),
        frame_indices={},
        simulation_step=0,
        time_s=0.1,
    )
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "frames"
        sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)
        try:
            for frame in frames:
                sink.publish(frame)
        finally:
            sink.close()

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


def test_offline_camera_frame_sink_restart_appends_metadata() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "frames"
        for value, simulation_step in ((10, 1), (20, 2)):
            sink = OfflineCameraFrameSink(
                camera_name="wrist_rgbd",
                save_dir=root,
                existing_data_policy="resume",
            )
            try:
                sink.publish(_rgb_frame(value=value, simulation_step=simulation_step))
            finally:
                sink.close()

        rows = [
            json.loads(line)
            for line in (root / "metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [row["simulation_step"] for row in rows] == [1, 2]


def test_offline_camera_frame_sink_restart_preserves_existing_payload() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "frames"
        first_sink = OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
        )
        try:
            first_sink.publish(_rgb_frame(value=10, simulation_step=1))
        finally:
            first_sink.close()

        first_row = json.loads(
            (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        first_payload_path = root / first_row["relative_path"]
        first_payload = first_payload_path.read_bytes()

        second_sink = OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
        )
        try:
            second_sink.publish(_rgb_frame(value=20, simulation_step=2))
        finally:
            second_sink.close()

        rows = [
            json.loads(line)
            for line in (root / "metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == 2
        assert [row["frame_index"] for row in rows] == [0, 1]
        assert rows[1]["relative_path"] != first_row["relative_path"]
        assert first_payload_path.read_bytes() == first_payload
        assert (root / rows[1]["relative_path"]).is_file()


def test_offline_camera_frame_sink_rejects_metadata_with_missing_payload() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        metadata = _rgb_frame(value=10, simulation_step=1).metadata(
            relative_path="rgb/000000.ppm"
        )
        (root / "metadata.jsonl").write_text(
            json.dumps(metadata) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing payload"):
            OfflineCameraFrameSink(
                camera_name="wrist_rgbd",
                save_dir=root,
                existing_data_policy="resume",
            )


@pytest.mark.parametrize(
    "metadata_line",
    (
        '{"camera_name":"wrist_rgbd","camera_name":"wrist_rgbd"}\n',
        '{"camera_name":"wrist_rgbd","value":NaN}\n',
        '{"camera_name":"wrist_rgbd","value":1e999}\n',
    ),
)
def test_camera_resume_rejects_ambiguous_or_non_finite_json(
    tmp_path: Path,
    metadata_line: str,
) -> None:
    root = tmp_path / "frames"
    root.mkdir()
    (root / "metadata.jsonl").write_text(metadata_line, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid camera metadata JSON"):
        OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
        )


def test_offline_camera_frame_sink_reserves_orphan_payload_index() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        orphan_path = root / "rgb" / "000000.ppm"
        orphan_path.parent.mkdir()
        orphan_payload = b"P6\n1 1\n255\n\x01\x02\x03"
        orphan_path.write_bytes(orphan_payload)

        sink = OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
        )
        try:
            sink.publish(_rgb_frame(value=20, simulation_step=2))
        finally:
            sink.close()

        row = json.loads((root / "metadata.jsonl").read_text(encoding="utf-8"))
        assert row["frame_index"] == 1
        assert row["relative_path"] == "rgb/000001.ppm"
        assert orphan_path.read_bytes() == orphan_payload


def test_camera_resume_rejects_incomplete_orphan_payload(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    orphan_path = root / "rgb" / "000000.ppm"
    orphan_path.parent.mkdir(parents=True)
    orphan_path.write_bytes(b"P6\n1 1\n255\n\x01")

    with pytest.raises(ValueError, match="incomplete or unreadable"):
        OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
        )


def test_camera_resume_rejects_unterminated_metadata_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)
    sink.publish(_rgb_frame(value=10, simulation_step=1))
    sink.close()
    metadata_path = root / "metadata.jsonl"
    original = metadata_path.read_bytes().rstrip(b"\n")
    metadata_path.write_bytes(original)

    with pytest.raises(ValueError, match="unterminated final JSON record"):
        OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
        )

    assert metadata_path.read_bytes() == original


def test_camera_resume_rejects_symlinked_modality_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "rgb").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
        )

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("modality", "format_field", "output_format"),
    (
        ("rgb", "rgb_format", "ppm"),
        ("rgb", "rgb_format", "png"),
        ("rgb", "rgb_format", "npy"),
        ("depth", "depth_format", "npy"),
        ("depth", "depth_format", "npz"),
    ),
)
def test_camera_resume_rejects_truncated_last_payload(
    tmp_path: Path,
    modality: str,
    format_field: str,
    output_format: str,
) -> None:
    root = tmp_path / f"{modality}-{output_format}"
    kwargs = {format_field: output_format}
    sink = OfflineCameraFrameSink(
        camera_name="wrist_rgbd",
        save_dir=root,
        **kwargs,
    )
    frame = next(
        frame
        for frame in sample_camera_frames(
            _camera_runtime(),
            frame_indices={},
            simulation_step=0,
            time_s=0.1,
        )
        if frame.modality == modality
    )
    sink.publish(frame)
    sink.close()
    row = json.loads((root / "metadata.jsonl").read_text(encoding="utf-8"))
    payload_path = root / row["relative_path"]
    payload = payload_path.read_bytes()
    payload_path.write_bytes(payload[: max(1, len(payload) // 2)])

    with pytest.raises(ValueError, match="incomplete or unreadable"):
        OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
            **kwargs,
        )


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


def test_camera_frame_observer_reset_clears_sampling_state() -> None:
    publisher = _CollectingPublisher()
    observer = CameraFrameObserver(
        cameras=(_camera_runtime(),),
        publisher=publisher,
    )

    observer.observe(_FakeWorld(), step=0, phase="before_reset")
    observer.reset()
    observer.observe(_FakeWorld(), step=0, phase="after_reset")

    assert [(frame.modality, frame.frame_index) for frame in publisher.frames] == [
        ("rgb", 0),
        ("depth", 0),
        ("rgb", 0),
        ("depth", 0),
    ]


def test_camera_frame_publisher_writes_in_background() -> None:
    frames = sample_camera_frames(
        _camera_runtime(),
        frame_indices={},
        simulation_step=0,
        time_s=0.1,
    )
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "frames"
        sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)
        publisher = CameraFramePublisher(sink=sink, max_queue_size=4)
        publisher.start()
        for frame in frames:
            publisher.publish(frame)
        publisher.close()

        metadata_path = root / "metadata.jsonl"
        assert len(metadata_path.read_text(encoding="utf-8").splitlines()) == 2


def test_offline_camera_sink_formats_and_metadata_flush_interval(
    tmp_path: Path,
) -> None:
    frames = sample_camera_frames(
        _camera_runtime(),
        frame_indices={},
        simulation_step=0,
        time_s=0.1,
    )
    sink = OfflineCameraFrameSink(
        camera_name="wrist_rgbd",
        save_dir=tmp_path / "frames",
        rgb_format="npy",
        depth_format="npz",
        metadata_flush_interval_frames=2,
    )
    try:
        sink.publish(frames[0])
        assert sink.metadata_path.stat().st_size == 0
        sink.publish(frames[1])
        assert sink.metadata_path.stat().st_size > 0
    finally:
        sink.close()

    assert np.load(tmp_path / "frames" / "rgb" / "000000.npy").dtype == np.uint8
    with np.load(tmp_path / "frames" / "depth" / "000000.npz") as payload:
        assert payload["data"].dtype == np.float32


def test_offline_camera_quota_removes_unindexed_payload(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    sink = OfflineCameraFrameSink(
        camera_name="wrist_rgbd",
        save_dir=root,
        max_bytes_per_camera=1,
    )

    with pytest.raises(CameraOutputQuotaExceededError, match="quota exceeded"):
        sink.publish(_rgb_frame(value=1, simulation_step=1))

    assert sink.next_frame_indices == {}
    assert sink.used_bytes == 0
    assert sink.metadata_path.read_bytes() == b""
    assert [path for path in root.rglob("*") if path.is_file()] == [sink.metadata_path]
    sink.close()


def test_offline_camera_payload_write_failure_removes_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)

    def fail_after_partial_write(save_dir: Path, *_args, **_kwargs) -> str:
        payload_path = save_dir / "rgb" / "000000.ppm"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(b"partial")
        raise OSError("simulated payload write failure")

    monkeypatch.setattr(
        recorder_module,
        "_write_frame_payload",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated payload write failure"):
        sink.publish(_rgb_frame(value=1, simulation_step=1))

    assert sink.metadata_path.read_bytes() == b""
    assert not (root / "rgb" / "000000.ppm").exists()
    assert sink.used_bytes == 0
    assert sink.next_frame_indices == {}
    sink.close()


def test_offline_camera_metadata_failure_rolls_back_record_and_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)
    sink.publish(_rgb_frame(value=1, simulation_step=1))
    committed_metadata = sink.metadata_path.read_bytes()
    committed_used_bytes = sink.used_bytes
    metadata_file = sink.metadata_file

    class PartialWriteThenFail:
        def tell(self) -> int:
            return metadata_file.tell()

        def write(self, data: bytes) -> int:
            metadata_file.write(data[: len(data) // 2])
            metadata_file.flush()
            raise OSError("simulated metadata write failure")

        def seek(self, offset: int) -> int:
            return metadata_file.seek(offset)

        def truncate(self, size: int | None = None) -> int:
            return metadata_file.truncate(size)

        def flush(self) -> None:
            metadata_file.flush()

        def close(self) -> None:
            metadata_file.close()

    sink.metadata_file = PartialWriteThenFail()

    with pytest.raises(OSError, match="simulated metadata write failure"):
        sink.publish(_rgb_frame(value=2, simulation_step=2))

    assert sink.metadata_path.read_bytes() == committed_metadata
    assert (root / "rgb" / "000000.ppm").is_file()
    assert not (root / "rgb" / "000001.ppm").exists()
    assert sink.used_bytes == committed_used_bytes
    assert sink.next_frame_indices == {"rgb": 1}
    sink.close()


def test_offline_camera_resume_rejects_existing_data_over_quota(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)
    sink.publish(_rgb_frame(value=1, simulation_step=1))
    sink.close()
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(CameraOutputQuotaExceededError, match="quota exceeded"):
        OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            existing_data_policy="resume",
            max_bytes_per_camera=1,
        )

    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_offline_camera_quota_accounts_for_payload_and_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    sink = OfflineCameraFrameSink(
        camera_name="wrist_rgbd",
        save_dir=root,
        max_bytes_per_camera=10_000,
    )
    sink.publish(_rgb_frame(value=1, simulation_step=1))
    accounted_bytes = sink.used_bytes
    sink.close()

    actual_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    assert accounted_bytes == actual_bytes
    assert sink.status() == {
        "type": "offline_camera",
        "camera_name": "wrist_rgbd",
        "save_dir": str(root),
        "used_bytes": actual_bytes,
        "max_bytes": 10_000,
        "remaining_bytes": 10_000 - actual_bytes,
    }


def test_offline_camera_quota_is_validated_before_path_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    with pytest.raises(ValueError, match="max_bytes_per_camera"):
        OfflineCameraFrameSink(
            camera_name="wrist_rgbd",
            save_dir=root,
            max_bytes_per_camera=0,
        )
    assert not root.exists()


def test_offline_camera_rejects_nonfinite_metadata_before_payload_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frames"
    sink = OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)

    with pytest.raises(ValueError, match="JSON compliant"):
        sink.publish(replace(_rgb_frame(value=1, simulation_step=1), time_s=np.nan))

    assert sink.metadata_path.read_bytes() == b""
    assert [path for path in root.rglob("*") if path.is_file()] == [sink.metadata_path]
    sink.close()


def test_offline_camera_existing_data_policies(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    root.mkdir()
    old = root / "old.bin"
    old.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="already exists"):
        OfflineCameraFrameSink(camera_name="wrist_rgbd", save_dir=root)
    assert old.read_bytes() == b"old"

    truncated = OfflineCameraFrameSink(
        camera_name="wrist_rgbd",
        save_dir=root,
        existing_data_policy="truncate",
    )
    truncated.close()
    assert root.is_dir()
    assert not old.exists()

    timestamped = OfflineCameraFrameSink(
        camera_name="wrist_rgbd",
        save_dir=root,
        existing_data_policy="timestamped_dir",
        timestamped_run_name="20260711T120000.000000Z",
    )
    try:
        assert timestamped.save_dir == root / "20260711T120000.000000Z"
    finally:
        timestamped.close()


@pytest.mark.parametrize(
    ("policy", "expected_value", "expected_dropped"),
    (("drop_oldest", 2, 1), ("drop_newest", 1, 1)),
)
def test_camera_frame_publisher_lossy_overflow_policies(
    policy: str,
    expected_value: int,
    expected_dropped: int,
) -> None:
    sink = _CollectingPublisher()
    publisher = CameraFramePublisher(
        sink=sink,
        max_queue_size=1,
        overflow_policy=policy,
    )

    publisher.publish(_rgb_frame(value=1, simulation_step=1))
    publisher.publish(_rgb_frame(value=2, simulation_step=2))
    queued = publisher.queue.get_nowait()

    assert queued is not None
    assert int(queued.data[0, 0, 0]) == expected_value
    assert publisher.status()["dropped_frames"] == expected_dropped
    publisher.close()


def test_camera_frame_publisher_error_policy_is_fail_fast() -> None:
    publisher = CameraFramePublisher(
        sink=_CollectingPublisher(),
        max_queue_size=1,
        overflow_policy="error",
    )
    publisher.publish(_rgb_frame(value=1, simulation_step=1))

    with pytest.raises(CameraFrameQueueFullError, match="capacity=1"):
        publisher.publish(_rgb_frame(value=2, simulation_step=2))

    assert publisher.status()["overflow_errors"] == 1
    publisher.close()


def test_camera_frame_publisher_block_policy_waits_without_dropping() -> None:
    class SlowSink:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.values: list[int] = []

        def publish(self, frame: CameraFrame) -> None:
            self.entered.set()
            self.release.wait(timeout=2.0)
            self.values.append(int(frame.data[0, 0, 0]))

        def close(self) -> None:
            pass

    sink = SlowSink()
    publisher = CameraFramePublisher(
        sink=sink,
        max_queue_size=1,
        overflow_policy="block",
        worker_poll_interval_s=0.01,
    )
    publisher.start()
    publisher.publish(_rgb_frame(value=1, simulation_step=1))
    assert sink.entered.wait(timeout=1.0)
    publisher.publish(_rgb_frame(value=2, simulation_step=2))
    blocked = threading.Thread(
        target=publisher.publish,
        args=(_rgb_frame(value=3, simulation_step=3),),
    )
    blocked.start()
    time.sleep(0.03)
    assert blocked.is_alive()

    sink.release.set()
    blocked.join(timeout=1.0)
    assert not blocked.is_alive()
    assert publisher.close(timeout_s=1.0)
    assert sink.values == [1, 2, 3]
    assert publisher.status()["dropped_frames"] == 0


def test_camera_frame_publisher_abort_discards_queued_frames() -> None:
    publisher = CameraFramePublisher(
        sink=_CollectingPublisher(),
        max_queue_size=2,
        shutdown_policy="discard",
    )
    publisher.publish(_rgb_frame(value=1, simulation_step=1))
    publisher.publish(_rgb_frame(value=2, simulation_step=2))

    assert publisher.close()
    assert publisher.status()["aborted_frames"] == 2
    with pytest.raises(Empty):
        publisher.queue.get_nowait()


def test_camera_frame_publisher_close_preserves_worker_failure_status() -> None:
    class FailingSink:
        def __init__(self) -> None:
            self.closed = False

        def publish(self, frame: CameraFrame) -> None:
            del frame
            raise OSError("disk full")

        def close(self) -> None:
            self.closed = True

    sink = FailingSink()
    publisher = CameraFramePublisher(
        sink=sink,
        worker_poll_interval_s=0.01,
    )
    publisher.start()
    publisher.publish(_rgb_frame(value=1, simulation_step=1))
    assert publisher.stop_event.wait(timeout=1.0)

    assert publisher.close(timeout_s=1.0) is True
    assert sink.closed is True
    assert publisher.status()["last_error"] == "OSError: disk full"
    with pytest.raises(RuntimeError, match="worker failed") as error:
        publisher.publish(_rgb_frame(value=2, simulation_step=2))
    assert isinstance(error.value.__cause__, OSError)


def test_camera_frame_publisher_successful_retry_clears_timeout_status() -> None:
    class BlockingSink:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def publish(self, frame: CameraFrame) -> None:
            del frame
            self.entered.set()
            self.release.wait(timeout=2.0)

        def close(self) -> None:
            pass

    sink = BlockingSink()
    publisher = CameraFramePublisher(
        sink=sink,
        worker_poll_interval_s=0.01,
    )
    publisher.start()
    publisher.publish(_rgb_frame(value=1, simulation_step=1))
    assert sink.entered.wait(timeout=1.0)

    assert publisher.close(timeout_s=0.0) is False
    assert publisher.status()["shutdown_timed_out"] is True
    sink.release.set()
    assert publisher.close(timeout_s=1.0) is True
    assert publisher.status()["shutdown_timed_out"] is False
    assert publisher.status()["sink_closed"] is True


def test_composite_camera_sink_closes_every_sink_and_retries_only_failure() -> None:
    calls: list[str] = []

    class RetrySink:
        def __init__(self) -> None:
            self.attempts = 0

        def publish(self, frame: CameraFrame) -> None:
            del frame

        def close(self) -> None:
            self.attempts += 1
            calls.append("retry")
            if self.attempts == 1:
                raise OSError("flush failed")

    class StableSink:
        def publish(self, frame: CameraFrame) -> None:
            del frame

        def close(self) -> None:
            calls.append("stable")

    composite = CompositeCameraFrameSink((RetrySink(), StableSink()))

    with pytest.raises(OSError, match="flush failed"):
        composite.close()
    assert calls == ["retry", "stable"]

    composite.close()
    assert calls == ["retry", "stable", "retry"]


def test_start_camera_output_returns_none_when_no_outputs() -> None:
    assert start_camera_output((_camera_runtime(),)) is None


def test_start_camera_output_injects_queue_and_shutdown_limits(tmp_path: Path) -> None:
    settings = _camera_settings(name="camera", save_dir="camera")
    camera = SensorCameraRuntime(settings=settings, camera=_FakeCamera())

    handle = start_camera_output(
        (camera,),
        path_resolver=lambda value: tmp_path / value,
        settings=CameraPublisherSettings(
            queue_size=7,
            overflow_policy="block",
            worker_poll_interval_s=0.02,
            existing_data_policy="error",
            shutdown_policy="discard",
            rgb_format="npy",
            depth_format="npz",
            metadata_flush_interval_frames=4,
            max_bytes_per_camera=123_456,
        ),
        shutdown_timeout_s=0.25,
    )

    assert handle is not None
    assert handle.publisher.queue.maxsize == 7
    assert handle.publisher.overflow_policy == "block"
    assert handle.publisher.worker_poll_interval_s == pytest.approx(0.02)
    assert handle.publisher.shutdown_policy == "discard"
    assert handle.publisher.shutdown_timeout_s == pytest.approx(0.25)
    sink = handle.publisher.sink
    assert isinstance(sink, OfflineCameraFrameSink)
    assert sink.rgb_format == "npy"
    assert sink.depth_format == "npz"
    assert sink.metadata_flush_interval_frames == 4
    assert sink.max_bytes_per_camera == 123_456
    assert handle.close()


def test_camera_output_prepare_is_read_only_until_joint_apply(
    tmp_path: Path,
) -> None:
    settings = _camera_settings(name="camera", save_dir="camera")
    camera = SensorCameraRuntime(settings=settings, camera=_FakeCamera())
    prepared = prepare_camera_output(
        (camera,),
        path_resolver=lambda value: tmp_path / value,
        settings=CameraPublisherSettings(),
        shutdown_timeout_s=0.25,
    )

    assert not (tmp_path / "camera").exists()
    apply_output_path_plans(prepared.path_plans)
    handle = open_prepared_camera_output(prepared)
    assert handle is not None
    assert (tmp_path / "camera" / "metadata.jsonl").is_file()
    assert handle.close()


def test_camera_output_validates_publisher_before_mutating_paths(
    tmp_path: Path,
) -> None:
    settings = _camera_settings(name="camera", save_dir="camera")
    camera = SensorCameraRuntime(settings=settings, camera=_FakeCamera())

    with pytest.raises(ValueError, match="max_queue_size"):
        start_camera_output(
            (camera,),
            path_resolver=lambda value: tmp_path / value,
            settings=CameraPublisherSettings(queue_size=0),
        )

    assert not (tmp_path / "camera").exists()

    with pytest.raises(ValueError, match="overflow_policy"):
        start_camera_output(
            (camera,),
            path_resolver=lambda value: tmp_path / value,
            settings=CameraPublisherSettings(
                overflow_policy=cast(str, []),
            ),
        )

    assert not (tmp_path / "camera").exists()


def test_open_prepared_camera_output_rolls_back_when_thread_start_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _camera_settings(name="camera", save_dir="camera")
    camera = SensorCameraRuntime(settings=settings, camera=_FakeCamera())
    prepared = prepare_camera_output(
        (camera,),
        path_resolver=lambda value: tmp_path / value,
        settings=CameraPublisherSettings(),
        shutdown_timeout_s=0.25,
    )
    apply_output_path_plans(prepared.path_plans)
    closed: list[str] = []

    class FakeSink:
        def publish(self, frame: CameraFrame) -> None:
            del frame

        def close(self) -> None:
            closed.append("sink")
            if len(closed) == 1:
                raise OSError("first close failed")

    monkeypatch.setattr(
        OfflineCameraFrameSink,
        "open_prepared",
        classmethod(lambda _cls, _plan: FakeSink()),
    )

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    with pytest.raises(RuntimeError, match="thread start failed"):
        open_prepared_camera_output(prepared)

    assert closed == ["sink", "sink"]


def test_start_camera_output_rejects_lossy_policy_for_offline_sink(
    tmp_path: Path,
) -> None:
    settings = _camera_settings(name="camera", save_dir="camera")
    camera = SensorCameraRuntime(settings=settings, camera=_FakeCamera())

    with pytest.raises(ValueError, match="offline camera output.*block.*error"):
        start_camera_output(
            (camera,),
            path_resolver=lambda value: tmp_path / value,
            settings=CameraPublisherSettings(overflow_policy="drop_oldest"),
        )

    assert not (tmp_path / "camera").exists()


def test_start_camera_output_preflights_all_paths_before_truncate(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    old = first / "old.bin"
    old.write_bytes(b"old")
    second = tmp_path / "second"
    second.mkdir()
    (second / "metadata.jsonl").write_text("not-json\n", encoding="utf-8")
    sensors = SceneSensorSettings(
        cameras=(
            _camera_settings(
                name="first",
                prim_path="/World/First",
                save_dir="first",
            ),
            _camera_settings(
                name="second",
                prim_path="/World/Second",
                save_dir="second",
            ),
        )
    )
    cameras = tuple(
        SensorCameraRuntime(settings=item, camera=_FakeCamera())
        for item in sensors.cameras
    )

    # Resume performs format/integrity validation for every target before any sink
    # is opened. The first directory must remain untouched when the second fails.
    with pytest.raises(ValueError, match="invalid camera metadata JSON"):
        start_camera_output(
            cameras,
            path_resolver=lambda value: tmp_path / value,
            settings=CameraPublisherSettings(
                overflow_policy="block",
                existing_data_policy="resume",
            ),
        )

    assert old.read_bytes() == b"old"


def test_start_camera_output_rejects_shared_offline_directory(tmp_path: Path) -> None:
    settings = SceneSensorSettings(
        cameras=(
            _camera_settings(
                name="left",
                prim_path="/World/LeftCamera",
                save_dir="shared",
            ),
            _camera_settings(
                name="right",
                prim_path="/World/RightCamera",
                save_dir="shared/../shared",
            ),
        )
    )
    cameras = tuple(
        SensorCameraRuntime(settings=camera, camera=_FakeCamera())
        for camera in settings.cameras
    )

    with pytest.raises(ValueError, match="save_dir must be unique"):
        start_camera_output(
            cameras,
            path_resolver=lambda value: tmp_path / value,
        )

    assert not (tmp_path / "shared" / "metadata.jsonl").exists()
