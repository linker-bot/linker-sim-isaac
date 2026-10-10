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

`runtime.step(render=True)` and motion execution permit scheduled acquisition after
each completed physical step. Only cameras due at their configured simulation-time
`frequency_hz` and having an output consumer enter a fresh-frame transaction. The
existing output observer owns the schedule and publishes each completed sample once.
Rates that do not divide the physics rate are quantized onto physical ticks without
long-term drift; missed historical samples are not synthesized. `render=False`
requests no rendering and cannot relabel old pixels with the new physical time.

`runtime.render()` explicitly returns a **new** frame from every configured camera.
Use `runtime.render(camera_ids=("left_rgb", "top"))` to capture only selected cameras.
Each selected camera returns all its configured modalities; empty, duplicate and
unknown selections fail before rendering. Every call acquires a new native frame,
even while paused. It never steps physics, publishes automatic outputs, or advances
the recording schedule. A failed transaction invalidates its selected capture metadata.

Sensor render products are inactive between captures; their windows may retain the
last image. Turning `outputs.camera.enabled` off disables automatic sampling, while
explicit capture remains available. Resolution, frequency and modalities remain
startup settings, selected through existing camera leaves or a complete inline scene
configuration. There is no runtime reconfiguration or new image JSON operation.

GUI service uses `scene.render_frequency_hz` as a wall-clock target, independently of
sensor sampling. Pause, estop and continuous query traffic still service the main
viewport without advancing physics or starting a sensor transaction. Motion provides
service opportunities at completed steps. Headless idle does not pump Kit without a
capture request. A capture also services GUI, avoiding an immediate duplicate update.
Blocking native calls, planning and reliable output backpressure can still delay GUI;
the target is not a guaranteed wall-clock FPS.

For CPU PhysX and Newton the coordinator calls `pre_render()` once to publish the
current physics poses, then pumps `render_update()` without advancing physics time.
Native cameras wait for a matching render-product completion whose Kit SWH frame
number is at least the first update after publication. Newton has no SWH stage-update
owner: it uses the native product frame number and a SyntheticData rational render-clock
barrier instead. That renderer clock is separate from physical simulation time. Warmup is bounded to 150 application
updates, matching SyntheticData’s render-simulation wait: asynchronous rendering
can take several app updates per completed frame. A timeout raises instead of returning old data. Newton keeps its minimum
four-update history budget. Selected products are activated together and share
renderer updates until each has completed a fresh frame; all return to the inactive
idle set even on failure. This uses the owner thread, not concurrent USD calls.
Configured frequency controls output sampling; explicit
`render()` requests a fresh frozen snapshot even when paused.

Headless PhysX with rendering enabled retains an active startup Hydra viewport; it
does not require a visible window or mechanical camera mesh. Mirror's PhysX Kit
disables independent sensor/TLAS clocks so paused pose changes reach the next capture.
With rendering disabled it still suppresses the startup viewport.

Recorded frame metadata adds `capture`: `native_frame_id` (`native_frame_source` is
`kit_swh` for PhysX or `kit_render_product_frame` for Newton), `render_product_path`, `render_snapshot_index`, and `physics_time_s`.
`frame_index` remains an output sequence number, not a native exposure ID. A physics
step without a matching render cannot relabel old pixels as a new observation.
Native IDs are local to the Kit process; snapshot indices are local to the coordinator.
Camera handles expose the same identity through `get_capture_metadata()`.

The focused regression is `scripts/smoke_mirror_camera.py`: first frame without
physics warmup, paused object set/restore, reset, product inactivity and fresh subset
capture. With `--record-root <empty-directory>`, every step permits automatic capture;
the supervisor decodes drained RGB/depth and checks native identities, physical time
and cadence. Select `--gui`, `--resolution 1920x1080`, `--cameras 4`, or
`--cameras 3 --frequencies 60 30 10 --steps 120` for focused variants.
`--modalities rgb` or `--modalities depth` tests a single configured modality.
A 640×320 diagnostic setting is also available; it is not a device calibration preset.

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

## Flanged hands and physical camera modules

`camera_workstation` is an optional Mirror mode/scene example at 120 Hz physics.
The existing L6/O6 left/right profiles now include the common flange and enable
self-collision. Select `ar5_08_{l6,o6}_gemini335l_{l,r}` for the flange, bracket and
Gemini 335L together. Select each side independently; remove its sensor entries when
using the flange-only profile. Rendering off does not remove installed hardware.
There is no bracket-only profile. Gravity, gains and armature retain their previous
settings; attachments inherit disabled default gravity, but their inertia is present.

The static `workstation_zed2i` object adds the top camera and stand. The example
places it on the centered pedestal mount; other scenes must provide their own
measured root transform. These asset optical frames and inertias are nominal CAD
registrations/estimates, not device calibration. The default
`AR5V2_L_flange_tcp` / `AR5V2_R_flange_tcp` coincides with the arm flange;
installing a hand or camera does not make it a grasp/contact point.

Use `parent_prim_path` for the robot/object instance root and `parent_link` for an
exact, unique imported link name. The importer hierarchy is resolved after assets
exist; missing/ambiguous links fail clearly. `prim_path` supplies the camera leaf
name, moved under that resolved link. `pose_axes: opencv` interprets the local pose
as X right, Y down, Z forward; the default `world` convention is X forward, Y left,
Z up. A zero optical-frame pose needs no extra rotation. Recorded capture metadata
includes the resolved parent, profile, model and calibration provenance. Returned
camera world poses retain the existing world-camera convention.

```yaml
- id: left_rgb
  camera_profile: gemini335l_rgb_60
  parent_prim_path: /World/Robots/left_arm
  parent_link: camera_rgb_optical
  prim_path: /World/Robots/left_arm/LeftRGB
  pose: {xyz: [0, 0, 0], rpy: [0, 0, 0]}
  pose_axes: opencv
```

Camera leaves live in `configs/cameras`. A setting has one writer: a selected leaf
cannot be silently overridden by an inline scene value. Copy/edit a leaf to change
specifications, or supply the complete inline configuration. The catalog records
all selected camera sources.

| Profile | Native resolution | Hz | Stream |
| --- | --- | --- | --- |
| `gemini335l_rgb_60` | 1280×800 | 60 | RGB |
| `gemini335l_depth_30` | 1280×800 | 30 | depth |
| `gemini335l_depth_60` | 848×480 | 60 | depth |
| `zed2i_1080p_30` | 1920×1080 | 30 | RGB + ideal depth |
| `zed2i_720p_60` | 1280×720 | 60 | RGB + ideal depth |
| `development_1080p_60` | 1920×1080 | 60 | RGB + ideal depth |

The nominal pinhole FOVs are Gemini RGB 94×68°, depth 90×65°, and ZED 2i 2.1 mm
110×70°. They do not simulate stereo matching, lens distortion, hardware exposure,
MinZ invalidity or sensor noise. Physical MinZ is not the renderer near plane.
Gemini RGB/depth use distinct asset optical frames, so matching dimensions do not
mean depth-to-color alignment. Real devices require profile-specific SDK calibration.
Development recording is a virtual camera specification and is not upscaled sensor
output. Hz denotes distinct simulation-time samples, not guaranteed wall-clock FPS.

With the current O6 + Gemini mounting, some straight-thumb sideways sweeps hit the
camera. This matches the hardware: retain the collision and coordinate bending with
sideways motion. Bending does not make every angle safe, especially on the left.
Validate the task's complete trajectory; the L6 assembly has a different hand shape.

The assembly regression is `scripts/smoke_mirror_assemblies.py`; select `--hand`,
`--wrists none|left|right|both`, `--profile`, `--gui`, and an empty `--record-root`.
It checks native image sizes, physical wrist following, paused restoration, reset,
and the unchanged gravity flags. The source asset mounting and collision rationale
is in the asset repository's `docs/camera-assemblies.md`.

The smoke also supports `--depth-hz 30|60`, `--top-hz 30|60` and
`--development-cameras` (four native 1080p cameras with both wrists). These select
the same catalog leaves; they do not upscale captured images. With `--record-root`,
the supervisor decodes the drained image/depth files after native shutdown and
checks dimensions, configured sample cadence, increasing native frame IDs and
matching capture/physics times.
