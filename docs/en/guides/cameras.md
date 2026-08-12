# Cameras

Language: [English](cameras.md) | [中文](../../zh-CN/guides/cameras.md)

Cameras are a Mirror-only capability. Kaleidoscope training scenes and modes reject
camera, renderer, viewport, and sensor-output fields before runtime construction. Its
separate human viewport can display one selected environment, but still excludes
cameras, SyntheticData, Replicator, recording, and image observations.

## Two Configuration Owners

Camera geometry belongs to the Mirror scene:

```yaml
cameras:
  - id: world_rgbd
    parent_prim_path: /World
    prim_path: /World/WorldRGBD
    pose:
      xyz: [0.08, 0.0, 0.08]
      rpy: [0.0, 1.1, 0.0]
    resolution: [320, 240]
    frequency_hz: 20.0
    modalities: [rgb, depth]
    clipping_range_m: [0.01, 5.0]
    intrinsics:
      fx: 307.5
      fy: 308.0
      cx: 160.0
      cy: 120.0
```

Encoding and queue policy belong to the output profile. This is the complete
`outputs.camera` section; a full output file also contains the required sibling
`render`, `logging`, and `telemetry` sections:

```yaml
camera:
  enabled: true
  save_root: logs/cameras
  foxglove_live_host: 127.0.0.1
  foxglove_live_port: null
  foxglove_mcap_path: null
  queue_size: 128
  overflow_policy: block
  worker_poll_interval_s: 0.1
  existing_data_policy: timestamped_dir
  shutdown_policy: drain
  rgb_format: png
  depth_format: npz
  metadata_flush_interval_frames: 1
  max_bytes_per_camera: 10737418240
  shutdown_timeout_s: 2.0
```

When camera output is enabled, the selected scene must contain at least one camera.

## Coordinate And Path Rules

- Poses use metres and XYZ Euler radians.
- `parent_prim_path` and `prim_path` are absolute.
- The camera prim must be below its declared parent namespace.
- Resolution is `[width, height]` with positive integers.
- Intrinsics use OpenCV pinhole pixel units; `fx` and `fy` are positive and all four
  fields must be supplied together. Omitting the whole mapping preserves Isaac defaults
  for backward compatibility, but calibrated scenes should configure it explicitly.
- Clipping satisfies `0 < near < far` in metres.
- Modalities are unique strings.

## Render Transaction

`RenderCoordinator` owns the whole render transaction. Timeline and physics-advancing
`hold_step` idle paths first call `physics.step(render=False)` and then `render_only()`,
which advances the renderer without reading cameras. The shared post-step observer is
the only component that captures and publishes those completed physics steps according
to frequency and backpressure policy, so one tick cannot perform two readbacks.

With `idle_physics_policy: pause`, no physics step or post-step observer runs. When its
wall-clock render cadence expires, the owner loop explicitly calls
`MirrorRuntime.render()`, which uses `render_frame(capture=True)` and returns the current
frame immediately. Application code calling `runtime.render()` has the same explicit
capture semantics.

For PhysX a transaction is one concrete-runtime `render()` call. For Newton the
coordinator calls `pre_render()` exactly once to synchronize the owner stream and
publish one immutable physics snapshot, then calls the pure `render_update()` operation
as many times as each hidden camera product requires. Multiple cameras are activated
one viewport at a time and all activation states are restored even if an update fails.
These renderer-only updates reuse the same snapshot and never advance physics time.

The physics runtime does not own cameras. `CameraBundle` owns camera handles and their
output sink; Mirror closes the bundle before the Isaac session.

## Backpressure

Choose overflow behavior deliberately:

- `block` preserves frames but can stall the producer;
- `drop_oldest` favors recent observation;
- `drop_newest` preserves queued history;
- `error` makes overload explicit.

Estimate raw bandwidth before increasing resolution or frequency. RGB at
`width * height * 3` bytes and float depth at `width * height * 4` bytes can exceed
disk or consumer throughput long before rendering saturates.

## Access From Python

`runtime.render()` returns the camera bundle's capture result when configured. Camera
handles normally expose `get_current_frame(clone=True)`; frames are copied before a
background consumer retains them.

Do not access render products from transport workers. All capture and stage interaction
must occur on the Mirror owner thread; only owned payloads may cross to output workers.

## Shutdown

The sink stops first, then camera handles close in reverse order. A timeout keeps the
bundle and session alive for retry. This prevents workers from dereferencing a stage
that has already been destroyed.

See [Outputs](../reference/outputs.md) and [Foxglove](foxglove.md).
