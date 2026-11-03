"""Bounded, synchronous CUDA RGB -> NVENC -> MP4 recording for Mirror.

The caller owns sampling and Kit. This writer never steps physics or downloads
pixels. A fixed-rate video accepts only consecutive physical sample times; its
JSONL index retains native exposure identity separately from playback timestamps.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from importlib import import_module
import json
import math
import os
from pathlib import Path
from threading import get_ident
from typing import Any


class MirrorVideoRecorder:
    """Record one camera's owned CUDA RGB frames, with synchronous backpressure.

    Use on its creating thread and close before the CUDA context/Kit exits.
    Any write/finalization failure makes the recording unusable; partial files
    remain for diagnosis. Only ``recording.json`` denotes a completed recording.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        camera_name: str,
        resolution: tuple[int, int],
        fps: int = 60,
        cuda_device: int = 0,
        bitrate: int = 12_000_000,
        max_frames: int = 3600,
        max_bytes: int = 536_870_912,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError("camera_name must be nonempty")
        if len(resolution) != 2 or any(
            type(value) is not int or value <= 0 or value % 2 for value in resolution
        ):
            raise ValueError("resolution must contain two positive even integers")
        for name, value in (
            ("fps", fps),
            ("bitrate", bitrate),
            ("max_frames", max_frames),
            ("max_bytes", max_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if fps > 240:
            raise ValueError("fps must not exceed 240")
        if type(cuda_device) is not int or cuda_device < 0:
            raise ValueError("cuda_device must be a nonnegative integer")
        directory = Path(output_dir)
        if not directory.is_absolute() or ".." in directory.parts:
            raise ValueError("output_dir must be absolute without '..'")
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError("output_dir must not traverse symlinks")
        if directory.exists():
            raise FileExistsError(directory)
        if not directory.parent.is_dir():
            raise ValueError("output_dir parent must exist")
        context = json.loads(_json(dict(metadata or {})))
        self._owner = get_ident()
        self._state = "open"
        self._directory = directory
        self._fps = fps
        self._max_frames = max_frames
        self._max_bytes = max_bytes
        self._count = 0
        self._encoded_bytes = 0
        self._metadata_bytes = 0
        self._first_time: float | None = None
        self._last_time: float | None = None
        self._last_capture: dict[str, Any] | None = None
        self._files = ExitStack()
        self._backend = _CudaEncoder(resolution, fps, cuda_device, bitrate)
        self._description = {
            "schema": "linker-sim.mirror-video.v1",
            "camera_name": camera_name,
            "resolution": list(resolution),
            "fps": fps,
            "encoder": self._backend.description,
            "metadata": context,
            "metadata_source": "caller-declared",
            "max_frames": max_frames,
            "max_bytes": max_bytes,
            "timing": "constant-rate-consecutive-physics-samples",
            "pixel_semantics": "lossy-display-rgb; not depth or calibrated sensor evidence",
        }
        try:
            directory.mkdir(mode=0o700)
            self._index = self._files.enter_context(
                (directory / "frames.jsonl").open("xb")
            )
            self._backend.open(directory / "video.mp4")
        except BaseException:
            self.abort()
            raise

    def write(
        self, rgb: Any, *, time_s: float, capture_metadata: Mapping[str, object]
    ) -> None:
        """Encode one real sample; CUDA uint8 HWC3 Torch or Warp data only."""

        self._require_open()
        try:
            if self._count >= self._max_frames:
                raise ValueError("video frame quota exceeded")
            if isinstance(time_s, bool) or not math.isfinite(time_s) or time_s < 0:
                raise ValueError("time_s must be finite and nonnegative")
            capture = json.loads(_json(dict(capture_metadata)))
            for name in ("native_frame_id", "render_snapshot_index"):
                value = capture.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"capture lacks {name}")
                if self._last_capture is not None and value <= self._last_capture[name]:
                    raise ValueError(f"capture {name} must increase")
            for name in ("native_frame_source", "render_product_path"):
                if not isinstance(capture.get(name), str) or not capture[name]:
                    raise ValueError(f"capture lacks {name}")
                if (
                    self._last_capture is not None
                    and capture[name] != self._last_capture[name]
                ):
                    raise ValueError(f"capture {name} changed")
            if capture.get("physics_time_s") != time_s:
                raise ValueError("capture physics time differs from sample time")
            if self._first_time is not None:
                expected = self._first_time + self._count / self._fps
                # PhysX reports a float32 time. Allow its precision, but never a
                # missing/duplicated sampling interval, even in long episodes.
                tolerance = min(0.25 / self._fps, 1e-6 + abs(expected) * 2e-7)
                if time_s <= self._last_time or abs(time_s - expected) > tolerance:
                    raise ValueError(
                        "video requires consecutive fixed-rate physical samples"
                    )
            row = (
                _json(
                    {
                        "frame_index": self._count,
                        "time_s": time_s,
                        "playback_time_s": self._count / self._fps,
                        "capture": capture,
                    }
                )
                + b"\n"
            )
            self._check_quota(extra=len(row))
            self._metadata_bytes += len(row)
            self._consume(self._backend.encode(rgb))
            self._index.write(row)
            self._index.flush()
            if self._first_time is None:
                self._first_time = time_s
            self._last_time = time_s
            self._last_capture = capture
            self._count += 1
            self._check_quota()
        except BaseException:
            self.abort()
            raise

    def close(self) -> Path:
        """Drain and release the codec, then exclusively publish completion metadata."""

        self._check_owner()
        complete = self._directory / "recording.json"
        if self._state == "closed":
            return complete
        self._require_open()
        try:
            if not self._count:
                raise ValueError("cannot finalize an empty video")
            self._consume(self._backend.drain())
            if self._backend.packet_count != self._count:
                raise RuntimeError(
                    "encoded packet count differs from captured frame count"
                )
            self._backend.finish()
            self._files.close()
            manifest = (
                _json(
                    {
                        **self._description,
                        "frame_count": self._count,
                        "first_time_s": self._first_time,
                        "last_time_s": self._last_time,
                        "duration_s": self._count / self._fps,
                        "video": "video.mp4",
                        "frames": "frames.jsonl",
                    }
                )
                + b"\n"
            )
            self._check_quota(extra=len(manifest))
            pending = self._directory / "recording.json.partial"
            with pending.open("xb") as stream:
                stream.write(manifest)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(pending, complete)
            pending.unlink()
            self._state = "closed"
            return complete
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Release resources without marking a failed/incomplete recording complete."""

        self._check_owner()
        if self._state == "closed":
            return
        self._state = "failed"
        try:
            self._backend.release()
        finally:
            self._files.close()

    def __enter__(self) -> MirrorVideoRecorder:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    def _consume(self, packets) -> None:
        for packet in packets:
            self._check_quota(extra=len(packet["data"]))
            self._backend.mux(packet)
            self._encoded_bytes += len(packet["data"])

    def _check_quota(self, *, extra: int = 0) -> None:
        # The muxer's AVIO buffer/trailer may not yet be on disk. Check both
        # accepted compressed bytes and actual files, including the final trailer.
        disk_bytes = sum(
            path.stat().st_size for path in self._directory.iterdir() if path.is_file()
        )
        if (
            max(disk_bytes, self._encoded_bytes + self._metadata_bytes) + extra
            > self._max_bytes
        ):
            raise ValueError("video byte quota exceeded")

    def _check_owner(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError("video recorder requires its creating thread")

    def _require_open(self) -> None:
        self._check_owner()
        if self._state != "open":
            raise RuntimeError(f"video recorder is {self._state}")


def _json(value: object) -> bytes:
    data = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(data) > 65_536:
        raise ValueError("video metadata exceeds 64 KiB")
    return data


class _CudaEncoder:
    """Small adapter for the pinned PyNvVideoCodec 2.2.4 API; no host pixel path."""

    def __init__(self, resolution, fps, device, bitrate):
        self._torch = import_module("torch")
        self._warp = import_module("warp")
        try:
            self._codec = import_module("PyNvVideoCodec")
        except ImportError as exc:
            raise RuntimeError("GPU video requires the Simulator video extra") from exc
        if self._codec.__version__ != "2.2.4":
            raise RuntimeError("GPU video requires pinned PyNvVideoCodec 2.2.4")
        self._width, self._height = resolution
        self._fps = fps
        self._device = f"cuda:{device}"
        native_device = self._warp.get_device(self._device)
        caps = self._codec.GetEncoderCaps(gpuid=device, codec="h264")
        for dimension, value in (("width", self._width), ("height", self._height)):
            if not caps[f"{dimension}_min"] <= value <= caps[f"{dimension}_max"]:
                raise ValueError(f"{dimension} exceeds NVENC capabilities")
        self._rgba = self._torch.empty(
            (self._height, self._width, 4), dtype=self._torch.uint8, device=self._device
        )
        self._stream = self._torch.cuda.current_stream(device)
        self._encoder = self._codec.CreateEncoder(
            self._width,
            self._height,
            "ABGR",
            False,
            cudacontext=native_device.context,
            cudastream=self._stream.cuda_stream,
            codec="h264",
            preset="P4",
            tuning_info="ultra_low_latency",
            fps=fps,
            bf=0,
            bitrate=bitrate,
        )
        if self._encoder.GetNumBFrames() != 0:
            raise RuntimeError("video requires no B-frame reordering")
        self._muxer = None
        self.packet_count = 0
        self.description = {
            "library": "PyNvVideoCodec",
            "version": self._codec.__version__,
            "codec": "h264",
            "engine": "nvenc",
            "device": self._device,
            "input_format": "ABGR",
            "bitrate": bitrate,
            "preset": "P4",
            "b_frames": 0,
            "pixel_transfer": "cuda-only",
            "host_transfer": "compressed-bitstream-and-metadata",
        }

    def open(self, path):
        # MP4 raises tiny timescales (e.g. 60) internally. A >= 10 kHz timescale
        # preserves our explicit packet units and exact integer-rate timestamps.
        self._muxer = self._codec.FFmpegMuxer(
            str(path),
            self._codec.MEDIA_FORMAT.MP4,
            "h264",
            self._width,
            self._height,
            self._fps,
            1,
            1,
            self._fps * 10000,
            self._encoder.GetSequenceParams(),
        )

    def validate(self, rgb):
        if isinstance(rgb, self._warp.array):
            if not rgb.device.is_cuda or str(rgb.device) != self._device:
                raise ValueError("video requires native input on its CUDA device")
            tensor = self._warp.to_torch(rgb, requires_grad=False)
        elif isinstance(rgb, self._torch.Tensor):
            tensor = rgb
        else:
            raise TypeError("video requires a CUDA Torch tensor or Warp array")
        if (
            str(tensor.device) != self._device
            or tensor.dtype != self._torch.uint8
            or tuple(tensor.shape) != (self._height, self._width, 3)
        ):
            raise ValueError(
                "video input must be CUDA uint8 HWC3 at its fixed resolution"
            )
        return tensor

    def encode(self, rgb):
        tensor = self.validate(rgb)
        with self._torch.cuda.stream(self._stream):
            self._rgba[:, :, :3].copy_(tensor)
            self._rgba[:, :, 3].fill_(255)
        # ABGR is RGBA byte order on little-endian CUDA hosts. Keep the single
        # reusable surface alive until NVENC's device copy has completed.
        self._stream.synchronize()
        packets = self._encoder.Encode(self._rgba)
        self._stream.synchronize()
        return packets

    def mux(self, packet):
        index = packet["timestamp"]
        if index != self.packet_count:
            raise RuntimeError("NVENC output is not consecutive display order")
        if not self._muxer.Mux(
            packet["data"],
            index * 10000,
            index * 10000,
            10000,
            0,
            int(packet["picture_type"] in (2, 3)),
            index,
        ):
            raise RuntimeError("MP4 mux failed")
        self.packet_count += 1

    def drain(self):
        return self._encoder.EndEncode()

    def finish(self):
        self._muxer.Finalize()
        self.release()

    def release(self):
        # PyNvVideoCodec owns the AVFormatContext/NVENC handles and exposes no
        # close method. Dropping the sole references runs its native destructors.
        self._muxer = None
        self._encoder = None
        self._rgba = None
