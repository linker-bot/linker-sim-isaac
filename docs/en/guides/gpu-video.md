# Mirror GPU Video

Language: [English](gpu-video.md) | [中文](../../zh-CN/guides/gpu-video.md)

`MirrorVideoRecorder` is an optional, synchronous consumer of already captured CUDA
RGB images. It uses NVENC H.264 and MP4 for display video. It does not step physics,
change automatic camera scheduling, create Kit, or provide full state recovery.

## Installation

Add `--extra video` to the existing simulation installation command, retaining all
other extras you use. Do not combine `dev` with `simulation`:

```bash
uv sync --frozen --extra simulation --extra visualization --extra training --extra video
```

The video extra pins PyNvVideoCodec 2.2.4. Existing Isaac/Torch/Warp versions are
unchanged. NVENC-capable hardware and a compatible NVIDIA driver are required;
missing libraries, unsupported dimensions, or unavailable encoding fail explicitly.
There is no software encoding or host-pixel fallback. CPU contract tests do not
install this extra.

## Embedded Use

Import through the lazy public facade. This example assumes one camera named `top`,
640×320 RGB, a 60 Hz control period, and automatic outputs disabled. Use the actual
resolved camera settings and `config.compute.cuda_device` in an application.

```python
from linkerbot_sim.mirror import MirrorVideoRecorder

camera = next(c for c in runtime.scene_resources.sensor_cameras if c.name == "top")
with MirrorVideoRecorder(
    "/absolute/existing/parent/new-recording",
    camera_name=camera.name,
    resolution=camera.settings.resolution,
    fps=60,
    cuda_device=runtime.config.compute.cuda_device,
    max_frames=60,
    max_bytes=128 * 1024 * 1024,
) as video:
    for frame_index in range(60):
        if frame_index:
            runtime.step(render=False)
        rgb = runtime.render(camera_ids=(camera.name,))[camera.name]["rgb"]
        video.write(
            rgb,
            time_s=float(runtime.physics_runtime.simulation_time),
            capture_metadata=camera.get_capture_metadata(),
        )
# Close/download video resources before closing the CUDA context or Kit.
```

`write` accepts an owned Warp array or Torch CUDA tensor: uint8, H×W×3, on the
selected device. Inputs must be ready for consumption and remain immutable until
`write` returns. RGB-to-RGBA layout conversion, synchronization and encoding use the
GPU; only compressed packets and JSON metadata reach host I/O. The reusable input
surface and NVENC's internal buffers are retained until their work finishes.
H.264 is lossy; it is not a substitute for depth/mask NPY or calibrated sensor data.

## Time, Identity And Limits

- One writer fixes camera name, even positive dimensions, integer `fps` (1–240),
  CUDA device and bitrate. The defaults are 60 fps, CUDA 0, 12 Mbps, at most 3600
  frames and 512 MiB per output directory. Hardware dimension limits are checked.
- Pass native capture metadata with matching `physics_time_s`, increasing integer
  `native_frame_id`/`render_snapshot_index`, and fixed `native_frame_source` and
  `render_product_path`. The caller supplies the camera association and optional
  recording `metadata`; these are not independently verified provenance claims.
- Real times must follow `first_time + frame_index / fps`. Float32 physics-time
  precision is allowed, capped below one quarter of a sampling interval. Duplicate
  paused exposures, skipped intervals, reset time, or source changes fail; start a
  new recording for a new segment. No repeated frames or silently shortened gaps.
- Playback starts at zero; `frames.jsonl` keeps each original physical time and
  native identity. H.264 uses no B-frames; encoded packet order/count are checked.
  MP4 uses explicit integer timestamp units rather than treating a frame number as
  a timestamp after the container adjusts its time base.
- Each JSON metadata object/row is at most 64 KiB. Synchronous backpressure avoids
  an application queue and dropped frames. Compressed bytes plus metadata and
  actual file sizes are checked against `max_bytes`; final checks include the MP4
  trailer and completion metadata. Buffered writes/trailer can temporarily exceed
  the limit before failure is detected, and incomplete files are retained.

## Lifecycle And Failure

Create a fresh absolute output directory under an existing parent. Symlink paths
and existing targets are rejected. All calls use the creating thread; the caller
still owns the runtime. Encoding can delay simulation/UI service. Native codec
calls are not preemptible; use the existing process supervisor for wall-clock
limits and stuck GPU processes.

`close()` drains NVENC, validates packet count, finalizes MP4 and releases codec/file
resources. Only then is `recording.json` atomically published. Repeated successful
close returns the same marker. An empty recording cannot be completed.

A failed `write`/`close` poisons the writer and releases resources. `abort()` or a
context-manager body exception releases them without a completion marker. Retry
in a new directory; do not treat a playable partial MP4 as accepted evidence.
Files are `video.mp4`, `frames.jsonl` and, on success, `recording.json` with schema
`linker-sim.mirror-video.v1`. This is process completion, not a power-loss transaction.

## Validation Scope

Run `just test-mirror-video` and `just quality` without CUDA. Real acceptance must
also inspect decoded frames, count, resolution, playback rate/duration and the
physical-time index; successful construction or a codec name alone is insufficient.
The [implementation plan](../../plans/MIRROR_GPU_VIDEO.zh-CN.md) records the tested
version and limits. This feature does not claim 60 wall-clock FPS or continuous
sensor calibration, and does not change Kaleidoscope or actual Tasks.
