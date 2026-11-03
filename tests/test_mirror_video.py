"""CPU contract checks for the optional CUDA recording boundary."""

from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from linkerbot_sim.sensors.camera import video


class Backend:
    instances = []
    failure = None
    description = {"codec": "h264", "device": "cuda:0", "engine": "nvenc"}

    def __init__(self, resolution, fps, device, bitrate):
        self.args = resolution, fps, device, bitrate
        self.packet_count = 0
        self.inputs = []
        self.released = False
        self.instances.append(self)

    def open(self, path):
        self.path = path
        if self.failure == "open":
            raise RuntimeError("open failed")
        path.write_bytes(b"header")

    def validate(self, rgb):
        if rgb != "cuda-frame":
            raise ValueError("invalid GPU input")

    def encode(self, rgb):
        self.validate(rgb)
        self.inputs.append(rgb)
        if self.failure == "encode":
            raise RuntimeError("encode failed")
        return [{"data": b"encoded", "timestamp": len(self.inputs) - 1}]

    def mux(self, packet):
        if self.failure == "mux":
            raise RuntimeError("mux failed")
        with self.path.open("ab") as stream:
            stream.write(packet["data"])
        self.packet_count += 1

    def drain(self):
        if self.failure == "drain":
            raise RuntimeError("drain failed")
        return []

    def finish(self):
        if self.failure == "finish":
            raise RuntimeError("finish failed")
        with self.path.open("ab") as stream:
            stream.write(b"trailer")
        self.release()

    def release(self):
        self.released = True


@pytest.fixture
def backend(monkeypatch):
    Backend.instances = []
    Backend.failure = None
    monkeypatch.setattr(video, "_CudaEncoder", Backend)
    return Backend


def recorder(tmp_path, **kwargs):
    return video.MirrorVideoRecorder(
        tmp_path / "recording", camera_name="top", resolution=(640, 320), **kwargs
    )


def capture(index, time_s=None):
    return {
        "native_frame_id": index + 1,
        "render_snapshot_index": index + 1,
        "native_frame_source": "native",
        "render_product_path": "/Render/top",
        "physics_time_s": index / 60 if time_s is None else time_s,
    }


def write(writer, index=0, **kwargs):
    values = {"time_s": index / 60, "capture_metadata": capture(index)}
    writer.write("cuda-frame", **(values | kwargs))


def test_records_true_times_and_completes_only_after_drain(tmp_path, backend):
    metadata = {"simulator_commit": "a" * 40}
    with recorder(tmp_path, metadata=metadata) as writer:
        metadata["simulator_commit"] = "changed"
        for index in range(3):
            write(writer, index)
        assert not (tmp_path / "recording/recording.json").exists()
    path = writer.close()
    result = json.loads(path.read_text())
    assert result["frame_count"] == 3 and result["duration_s"] == 3 / 60
    assert result["first_time_s"] == 0 and result["last_time_s"] == 2 / 60
    assert result["metadata"]["simulator_commit"] == "a" * 40
    rows = [
        json.loads(row)
        for row in path.with_name("frames.jsonl").read_text().splitlines()
    ]
    assert [row["time_s"] for row in rows] == [0, 1 / 60, 2 / 60]
    assert [row["frame_index"] for row in rows] == [0, 1, 2]
    assert backend.instances[-1].released
    with pytest.raises(RuntimeError, match="closed"):
        write(writer, 3)


@pytest.mark.parametrize(
    "options",
    [
        {"fps": 0},
        {"fps": True},
        {"fps": 241},
        {"cuda_device": -1},
        {"cuda_device": True},
        {"max_frames": 0},
        {"max_bytes": -1},
        {"bitrate": 0},
        {"metadata": {"bad": float("nan")}},
        {"metadata": {"large": "x" * 70000}},
    ],
)
def test_invalid_options_do_not_open_resources(tmp_path, backend, options):
    with pytest.raises(ValueError):
        recorder(tmp_path, **options)
    assert backend.instances == [] and not (tmp_path / "recording").exists()


@pytest.mark.parametrize(
    "resolution", [(1, 320), (640, 0), (True, 320), (640,), (640, 320, 3)]
)
def test_invalid_resolution(tmp_path, backend, resolution):
    with pytest.raises(ValueError):
        video.MirrorVideoRecorder(
            tmp_path / "recording", camera_name="top", resolution=resolution
        )
    assert backend.instances == []


def test_paths_never_overwrite_and_reject_symlinks(tmp_path, backend):
    directory = tmp_path / "recording"
    directory.mkdir()
    (directory / "keep").write_text("evidence")
    with pytest.raises(FileExistsError):
        recorder(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        recorder(link)
    assert (directory / "keep").read_text() == "evidence"
    assert backend.instances == []


@pytest.mark.parametrize(
    "fault",
    [
        "nan",
        "negative",
        "stale",
        "id",
        "snapshot",
        "source",
        "product",
        "gap",
        "duplicate",
        "metadata",
    ],
)
def test_bad_sample_poisoning_prevents_completed_evidence(tmp_path, backend, fault):
    writer = recorder(tmp_path)
    write(writer)
    sample = capture(1)
    time_s = 1 / 60
    if fault == "nan":
        time_s = float("nan")
    elif fault == "negative":
        time_s = -1
    elif fault == "stale":
        sample["physics_time_s"] = 0
    elif fault == "id":
        sample["native_frame_id"] = 1
    elif fault == "snapshot":
        sample["render_snapshot_index"] = 1
    elif fault == "source":
        sample["native_frame_source"] = "different"
    elif fault == "product":
        sample["render_product_path"] = "/Render/other"
    elif fault in ("gap", "duplicate"):
        time_s = 2 / 60 if fault == "gap" else 0
        sample["physics_time_s"] = time_s
    else:
        sample["large"] = "x" * 70000
    with pytest.raises(ValueError):
        write(writer, 1, time_s=time_s, capture_metadata=sample)
    assert backend.instances[-1].released and writer._index.closed
    with pytest.raises(RuntimeError, match="failed"):
        writer.close()
    assert not (tmp_path / "recording/recording.json").exists()


@pytest.mark.parametrize("fault", ["encode", "mux", "drain", "finish"])
def test_codec_failure_closes_resources_without_completion(tmp_path, backend, fault):
    writer = recorder(tmp_path)
    if fault in ("drain", "finish"):
        write(writer)
    backend.failure = fault
    with pytest.raises(RuntimeError, match="failed"):
        writer.close() if fault in ("drain", "finish") else write(writer)
    assert backend.instances[-1].released and writer._index.closed
    assert not (tmp_path / "recording/recording.json").exists()


def test_initialization_failure_releases_codec(tmp_path, backend):
    backend.failure = "open"
    with pytest.raises(RuntimeError, match="open failed"):
        recorder(tmp_path)
    assert backend.instances[-1].released


def test_frame_quota_and_empty_close(tmp_path, backend):
    writer = recorder(tmp_path, max_frames=1)
    write(writer)
    with pytest.raises(ValueError, match="quota"):
        write(writer, 1)
    other = tmp_path / "empty"
    other.mkdir()
    writer = recorder(other)
    with pytest.raises(ValueError, match="empty"):
        writer.close()
    assert backend.instances[-1].released


def test_byte_quota_before_encode_and_after_final_trailer(
    tmp_path, backend, monkeypatch
):
    writer = recorder(tmp_path, max_bytes=1)
    with pytest.raises(ValueError, match="quota"):
        write(writer)
    assert backend.instances[-1].inputs == []
    other = tmp_path / "second"
    other.mkdir()
    writer = recorder(other, max_bytes=1500)
    write(writer)
    original = backend.finish

    def large_trailer(self):
        original(self)
        with self.path.open("ab") as stream:
            stream.write(b"x" * 2000)

    monkeypatch.setattr(backend, "finish", large_trailer)
    with pytest.raises(ValueError, match="quota"):
        writer.close()
    assert not (other / "recording/recording.json").exists()


def test_wrong_thread_cannot_write_close_or_destroy_owner(tmp_path, backend):
    writer = recorder(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for action in (lambda: write(writer), writer.close, writer.abort):
            with pytest.raises(RuntimeError, match="thread"):
                pool.submit(action).result(timeout=2)
    assert not backend.instances[-1].released
    write(writer)
    writer.close()


def test_context_body_failure_aborts_without_completion(tmp_path, backend):
    with pytest.raises(LookupError):
        with recorder(tmp_path) as writer:
            write(writer)
            raise LookupError("caller failed")
    assert backend.instances[-1].released
    assert not (tmp_path / "recording/recording.json").exists()


def test_missing_or_invalid_native_identity(tmp_path, backend):
    writer = recorder(tmp_path)
    bad = capture(0)
    bad["native_frame_id"] = True
    with pytest.raises(ValueError, match="native_frame_id"):
        write(writer, capture_metadata=bad)


def test_public_facade_remains_import_safe():
    from linkerbot_sim.mirror import MirrorVideoRecorder

    assert MirrorVideoRecorder is video.MirrorVideoRecorder


@pytest.mark.parametrize(
    "device,dtype,shape",
    [
        ("cpu", "uint8", (320, 640, 3)),
        ("cuda:1", "uint8", (320, 640, 3)),
        ("cuda:0", "float32", (320, 640, 3)),
        ("cuda:0", "uint8", (320, 640, 4)),
    ],
)
def test_backend_rejects_wrong_payload(device, dtype, shape):
    class Tensor:
        pass

    encoder = video._CudaEncoder.__new__(video._CudaEncoder)
    encoder._torch = SimpleNamespace(Tensor=Tensor, uint8="uint8")
    encoder._warp = SimpleNamespace(array=type("WarpArray", (), {}))
    encoder._device, encoder._width, encoder._height = "cuda:0", 640, 320
    tensor = Tensor()
    tensor.device, tensor.dtype, tensor.shape = device, dtype, shape
    with pytest.raises(ValueError, match="CUDA uint8"):
        encoder.validate(tensor)


def test_backend_rejects_packet_reordering():
    encoder = video._CudaEncoder.__new__(video._CudaEncoder)
    encoder.packet_count = 0
    with pytest.raises(RuntimeError, match="consecutive"):
        encoder.mux({"timestamp": 1})


def test_atomic_completion_failure_keeps_incomplete_evidence(
    tmp_path, backend, monkeypatch
):
    writer = recorder(tmp_path)
    write(writer)

    def fail(*_):
        raise OSError("publish failed")

    monkeypatch.setattr(video.os, "link", fail)
    with pytest.raises(OSError, match="publish failed"):
        writer.close()
    assert backend.instances[-1].released
    assert not (tmp_path / "recording/recording.json").exists()


def test_nonzero_epoch_rate_and_float32_clock_precision(tmp_path, backend):
    import struct

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    with recorder(tmp_path, fps=30) as writer:
        for index in range(3):
            now = f32(60.0 + index / 30)
            write(writer, index, time_s=now, capture_metadata=capture(index, now))
    result = json.loads(writer.close().read_text())
    assert result["duration_s"] == 0.1
    assert result["first_time_s"] == 60.0


def test_index_io_error_releases_encoder(tmp_path, backend):
    writer = recorder(tmp_path)
    original = writer._index

    def fail(_):
        raise OSError("index I/O failed")

    writer._index = SimpleNamespace(write=fail)
    with pytest.raises(OSError, match="index I/O"):
        write(writer)
    assert original.closed and backend.instances[-1].released
    assert not (tmp_path / "recording/recording.json").exists()


def test_missing_encoded_packets_cannot_complete(tmp_path, backend):
    writer = recorder(tmp_path)
    write(writer)
    backend.instances[-1].packet_count = 0
    with pytest.raises(RuntimeError, match="packet count"):
        writer.close()
    assert backend.instances[-1].released
    assert not (tmp_path / "recording/recording.json").exists()
