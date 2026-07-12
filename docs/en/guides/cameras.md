# Camera Types And Sensors

Language: [English](cameras.md) | [中文](../../zh-CN/guides/cameras.md)

This document explains the difference between GUI viewport settings and simulated
sensor cameras, including runtime data flow, switch semantics, output formats,
capacity planning, and Isaac Sim rendering warning diagnostics.

## Terminology

| Type | Recommended Name | Config Location | Purpose | Produces Image Data |
| --- | --- | --- | --- | --- |
| GUI view | viewport view / GUI viewport | `visuals.viewport` | Set the Isaac GUI viewing angle. | No |
| Simulated camera | sensor camera / RGB-D camera | `sensors.cameras` | Scene sensor that outputs RGB, depth, etc. | Yes |

The GUI viewport is for human observation only. It does not participate in control, planning, logging, or visual algorithm inputs.

## GUI Viewport

```yaml
visuals:
  viewport:
    enabled: true
    eye: [1.35, -1.65, 1.05]
    target: [0.0, -0.1, 0.42]
    prim_path: /OmniverseKit_Persp
```

Fields:

| Field | Meaning |
| --- | --- |
| `enabled` | Whether to set the GUI viewport after startup. |
| `eye` | View eye point in world coordinates, m. |
| `target` | View target in world coordinates, m. |
| `prim_path` | Camera prim path used by the Isaac viewport. |

## Sensor Camera

Sensor cameras are configured under env profile `sensors.cameras`:

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      parent_prim_path: /World
      prim_path: /World/WorldRGBD
      pose:
        xyz: [0.5, -0.6, 0.8]
        rpy: [0.0, 0.7, 0.0]
      resolution: [640, 480]
      frequency: 30.0
      modalities: [rgb, depth]
      clipping_range: [0.01, 5.0]
      intrinsics:
        fx: 615.0
        fy: 615.0
        cx: 320.0
        cy: 240.0
      output:
        save_dir: logs/cameras/world_rgbd
        foxglove_topic_prefix: /cameras/world_rgbd
        foxglove_live_host: 127.0.0.1
        foxglove_live_port: 8770
```

Fields:

| Field | Meaning |
| --- | --- |
| `enabled` | Whether to create and initialize this camera; automatic sampling in the canonical runtime also requires at least one active output sink. |
| `parent_prim_path` | Optional parent prim. `/World` means fixed world camera; robot link path means wrist/tool camera. |
| `prim_path` | Absolute USD prim path for the camera. |
| `pose.xyz` | Translation relative to parent prim, or world coordinates without a parent. |
| `pose.rpy` | Orientation relative to parent prim, rad. |
| `resolution` | Image width and height. |
| `frequency` | Sampling frequency in Hz. |
| `env_ids` | Resource scope used only by Tiled Scene; a nonempty duplicate-free integer list. Single Scene rejects this field. |
| `modalities` | Output types, for example `rgb` and `depth`. |
| `clipping_range` | Near/far clipping planes, m. |
| `intrinsics.fx` / `intrinsics.fy` | Optional explicit pinhole focal lengths in pixels; applied to the Isaac Camera. |
| `intrinsics.cx` / `intrinsics.cy` | Optional explicit pinhole principal point in pixels; configure together with `fx/fy`. |
| `output.save_dir` | Optional local output directory. |
| `output.foxglove_topic_prefix` | Optional Foxglove topic prefix. |
| `output.foxglove_live_host` | Optional Foxglove live loopback bind address; non-loopback values are rejected. |
| `output.foxglove_live_port` | Optional Foxglove live port. Camera ports should start at `8770`. |
| `output.foxglove_mcap_path` | Optional Foxglove MCAP output path. |

The camera live port is configured by the env profile. It is not the same as the interactive state-stream `--foxglove-live-port`.
Like every built-in listener, a camera live server accepts only `localhost` or a
numeric loopback address. Use an authenticated TLS reverse proxy or SSH tunnel
for remote viewing.

## Runtime Data Flow And Switch Boundaries

The complete path from an env profile to an output sink is:

```text
sensors.cameras
  -> SensorCameraSettings
  -> Isaac Camera prim + Replicator render product
  -> RTX render + RGB/depth annotator
  -> world.step(render=True)
  -> CPU ndarray sampled on the simulation thread
  -> bounded queue
  -> background file, Foxglove live, or MCAP sink
```

Creation order and thread ownership are explicit:

1. The env profile is parsed by ordinary Python without starting Isaac or creating a camera.
2. Each `enabled: true` entry creates an Isaac Camera wrapper and camera prim while the runtime scene is built.
3. After `world.reset()`, the Camera is initialized and its configured `modalities` are attached as annotators.
4. Any of `save_dir`, `foxglove_live_port`, or `foxglove_mcap_path` creates a camera output observer and forces render steps even in headless mode.
5. The observer samples by simulation time and `frequency` after `world.step()`. Isaac objects are read only on the main simulation thread. The background publisher only consumes captured arrays.
6. The output queue is bounded. Runtime `camera_output.overflow_policy` decides whether saturation blocks, fails fast, or drops a declared side of the queue.

The switches have distinct behavior:

| Setting | Runtime effect |
| --- | --- |
| `enabled: false` | Creates no Camera prim, render product, annotator, or output task. Prefer this for control/planning-only runs. |
| `enabled: true` | Creates and initializes the camera; it does not by itself enable file or network output. |
| `modalities` | Selects attached annotators and the frames produced at each sample. |
| `output.save_dir` | Enables local frame payloads and `metadata.jsonl`. |
| `output.foxglove_live_port` | Enables Foxglove live RawImage output. A host or topic prefix alone does not start output. |
| `output.foxglove_mcap_path` | Enables camera RawImage recording to MCAP. |
| `--gui` | Controls the GUI viewport. Camera output can still drive rendering without it. |
| `visuals.viewport.enabled` | Only sets the GUI view and never creates sensor images. |

`frequency` follows simulation time, not guaranteed wall-clock real time. If the
simulation runs slower than real time, wall-clock output FPS also falls. The
current implementation does not synthesize intermediate frames when one physics
step crosses multiple sampling periods.

## Runtime Output Policy

Sensor placement and sampling remain in the env profile. Process-wide queue,
encoding, file lifecycle, quota, and shutdown behavior belong to the runtime
profile:

```yaml
runtime:
  camera_output:
    queue_size: 128
    max_bytes_per_camera: 10737418240
    overflow_policy: block
    worker_poll_interval_s: 0.1
    existing_data_policy: error
    shutdown_policy: drain
    rgb_format: png
    depth_format: npz
    metadata_flush_interval_frames: 16
  shutdown:
    camera_publisher_timeout_s: 2.0
```

| Field | Behavior |
| --- | --- |
| `queue_size` | Maximum queued modality frames; one modality frame is one queue item. |
| `max_bytes_per_camera` | Independent byte quota for each local camera directory. |
| `overflow_policy` | `block` or `error` for persistent output; live-only output may also use either dropping policy. |
| `worker_poll_interval_s` | Poll interval used by the worker and by a blocked producer checking failure or shutdown. |
| `existing_data_policy` | File policy for local camera directories and camera MCAP files. |
| `shutdown_policy` | `drain` writes admitted frames; `abort` discards queued frames. |
| `rgb_format` | Local RGB payload: `ppm`, `png`, or `npy`. |
| `depth_format` | Local float32 depth payload: `npy` or compressed `npz`. |
| `metadata_flush_interval_frames` | Number of modality metadata rows between flushes. |
| `shutdown.camera_publisher_timeout_s` | Bounded worker join timeout. |

Any local directory or camera MCAP makes the output persistent and therefore
requires `overflow_policy: block` or `error`. The precise existing-data checks,
quota accounting, queue behavior, recovery rules, status counters, and shutdown
semantics are defined in [Outputs And Persistence](../reference/outputs.md).

## Tiled Scene Camera Expansion

In tiled envs, `sensors.cameras` keeps shared settings and `env_ids` selects the
envs where camera resources are actually created:

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      env_ids: [0, 1]
      prim_path: /World/WorldRGBD
      modalities: [rgb, depth]
      output:
        save_dir: logs/cameras/world_rgbd

tiled:
  enabled: true
  num_envs: 8
  diagnostics:
    inspect_env_ids: [7]
```

This profile creates Camera prims, render products, annotators, and output
channels only for envs 0 and 1. Envs 2 through 7 incur no camera rendering cost.
`inspect_env_ids: [7]` affects diagnostics only and never participates in camera
creation. Output directories and Foxglove topics receive `env_000`, `env_001`,
and similar suffixes.

`env_ids` must be a nonempty duplicate-free list of nonnegative integers, each
smaller than `tiled.num_envs`. A misspelled `env_id`, empty list, duplicate,
boolean, or out-of-range value is rejected with the full field path. The regular
Single Scene entrypoint rejects this tiled-only field as well, instead of silently
creating an unscoped camera.

Every camera in an env profile with `tiled.enabled: true` must provide `env_ids`, including an
`enabled: false` camera. A missing selector never expands to every env and never
warns and continues.

Per-env files may still override a declared camera pose with
`cameras.<name>.pose`, but only when that camera's scope contains the current env.
An override outside `env_ids` is rejected instead of being retained as inactive
configuration. Start with a small selected set when validating throughput, GPU
memory, and storage.

## Output Formats And Topics

All supported modalities can be stored locally. RGB and depth also have a
Foxglove image payload; segmentation modalities publish metadata only:

| Modality | Local payload | Foxglove live/MCAP | Info topic |
| --- | --- | --- | --- |
| `rgb` | `ppm`, `png`, or `npy` | `RawImage`, `rgb8` | JSON metadata |
| `depth` | `npy` or `npz` | `RawImage`, `32FC1` | JSON metadata |
| `semantic_segmentation` | `npy` | No image channel | JSON metadata |
| `instance_segmentation` | `npy` | No image channel | JSON metadata |

Local output example:

```text
logs/cameras/world_rgbd/
├── metadata.jsonl
├── rgb/
│   └── 000000.png
├── depth/
│   └── 000000.npz
├── semantic_segmentation/
│   └── 000000.npy
└── instance_segmentation/
    └── 000000.npy
```

The RGB and depth extensions follow `rgb_format` and `depth_format`. `png` is
lossless RGB compression, while `npz` stores depth under the `data` key. `npy`
preserves the normalized RGB8, float32 depth, or segmentation array directly.

Foxglove topics:

| Topic | Encoding |
| --- | --- |
| `/cameras/world_rgbd/rgb` | `RawImage`, `rgb8` |
| `/cameras/world_rgbd/depth` | `RawImage`, `32FC1` |
| `/cameras/world_rgbd/info` | JSON metadata |

`/info` receives one message for every sampled modality. It contains the frame
index, simulation step/time, shape, dtype, and optional intrinsics/world pose.
Local `metadata.jsonl` additionally contains `relative_path`. See
[Outputs And Persistence](../reference/outputs.md#camera-metadata) for the exact
record and commit contract.

## GUI And Headless

Sensor cameras do not depend on `--gui`. When camera output is configured, the runtime drives render updates so camera annotators can produce images in headless mode too. Empty startup frames are skipped.

`SINGLE_SCENE_INTERACTIVE_READY` only means that command transports and the queue
are ready. RTX, Fabric, Replicator, and SyntheticData may perform lazy first-frame
work on the next `world.step(render=True)`. Camera warnings can therefore appear
after READY without indicating that runtime creation failed.

## Performance And Capacity Planning

The bundled local formats are uncompressed binary PPM and float32 NPY. PNG/NPZ
can reduce disk size at additional worker CPU cost. Foxglove and MCAP use
uncompressed `RawImage`. Ignoring file headers, metadata, protocol, and queue
overhead, raw payload in one output direction is approximately:

```text
bytes_per_second
  = camera_count * width * height * frequency_hz
    * sum(bytes_per_pixel_per_modality)

rgb8    = 3 bytes/pixel
depth   = 4 bytes/pixel
```

One 640x480 camera at 30 Hz with `[rgb, depth]` produces:

```text
640 * 480 * 30 * (3 + 4)
  = 64,512,000 bytes/s
  ~= 61.5 MiB/s
  ~= 216 GiB/hour
```

This is one output direction. With both `save_dir` and Foxglove live enabled,
disk and network each carry roughly this payload, plus their own overhead. The
same settings expanded to 64 tiled envs reach about 3.84 GiB/s and 13.5 TiB/hour
of theoretical raw data. GPU readback, the bounded queue, disk, or network will
normally saturate first. A lossless profile then slows the simulation through
backpressure; a live-only dropping profile increments its dropped counter.

`max_bytes_per_camera` is a hard quota for each local camera namespace. It
includes payloads, metadata, and existing regular files. Quota compensation and
queue failure semantics are specified in
[Outputs And Persistence](../reference/outputs.md#capacity-and-quotas).

Recommended profiles by use case:

| Use case | Recommendation |
| --- | --- |
| Control, planning, regression tests | Set `enabled: false`; create no sensor camera. |
| One-camera interactive debugging | Lower `frequency`; enable only required modalities and one primary sink. |
| Dataset capture | Use headless mode, budget disk, and keep `overflow_policy: block` (or fail fast with `error`). |
| Large tiled runs | Keep cameras disabled; test with a small `num_envs` before increasing concurrency. |
| GPU vision algorithms | Keep frames device-side and copy to CPU only when file or network encoding actually requires it. |

## Isaac Sim Rendering Warnings

The warnings below come from the Isaac Sim 5.1 RTX/Fabric/SyntheticData path.
They are unrelated to robot mass, inertia, joint control, or cuRobo solving.
Check image output before deciding that they require action.

### `USD->Fabric: Unhandled array type string[]`

This is commonly followed by messages such as:

```text
[usdrt.population.plugin] Unhandled attribute type VtArray<std::string>
(prim attribute: omni:rtx:material:db:flattener:transmittance_color)
```

`reflection_roughness_constant` and `ior_constant` can appear too. These are
internal string-array metadata attributes emitted by the RTX MaterialDB
flattener. The current Fabric/USDRT population path cannot synchronize that type
and skips the attributes. This does not modify the PhysX stage and does not mean
that mass or collision data in repository assets is corrupt.

- Treat it as an expected one-time warning when RGB/depth and GUI materials are correct.
- Do not delete material data from robot or object assets merely to silence it; the named attributes are generated internally by RTX.
- If materials are actually missing, retain the complete Kit log and retest after an Isaac/Kit upgrade.

### `DLSS increasing input dimensions`

The default GUI runtime profile sets
`runtime.simulation_app.render.anti_aliasing_gui: 3`, which the launcher passes
to Kit as DLSS. DLSS renders at a lower internal size and upscales to the
render-product output. With the current Isaac version and a 640x480 camera, the
observed input is 320x240. Its height is below the plugin minimum of 300, so the
plugin raises the internal size:

```text
DLSS increasing input dimensions: Render resolution of (320, 240)
is below minimal input resolution of 300.
```

The camera output remains 640x480. This is normally an automatic quality/performance
adjustment, although actual cost may differ from an estimate based on 320x240.
Options are:

- Accept the one-time warning.
- Increase sensor resolution. Under the currently observed 2x scaling, 800x600 produces an internal size no smaller than 400x300.
- In the runtime YAML, set `runtime.simulation_app.render.anti_aliasing_gui` for GUI launches or `runtime.simulation_app.render.anti_aliasing_headless` for headless launches to the Kit antialiasing mode required by the workload. Env profiles do not accept a bare `anti_aliasing` field.

### `OgnSdPostRenderVarToHost ... counter-performant`

The recorder and Foxglove sinks consume NumPy CPU data, so current RGB/depth
sampling explicitly requests CPU frames. SyntheticData must read the GPU render
texture back to host and reports that a direct texture-to-host path is inefficient:

```text
OgnSdPostRenderVarToHost : rendervar copy from texture directly to host buffer
is counter-performant. Please use copy from texture to device buffer first.
```

This is a performance warning, not evidence of incorrect pixels. It is usually
acceptable for one low-rate camera. For high-rate or tiled capture:

1. Disable unneeded modalities and sinks.
2. Reduce `resolution` or `frequency`.
3. Let algorithms consume a GPU device buffer and delay or combine host copies.
4. When CPU file/network output is required, design explicit device staging, batched copies, and a documented drop/backpressure policy.

### When To Treat A Warning As A Failure

Do not globally suppress one-time initialization messages while interaction and
images remain correct. Investigate further when any of these occurs:

- RGB/depth stays empty or has an unexpected shape or dtype.
- The same warning repeats every frame and materially reduces throughput.
- `CAMERA_FRAME_PUBLISHER_FAILED`, CUDA OOM, renderer errors, or process termination appears.
- Frame indices in `metadata.jsonl` stop increasing or have unacceptable gaps.
- Camera render/readback work continues after that camera is disabled.

## Common Issues

RGB does not show in Foxglove
: Check local `logs/cameras/<name>/rgb/` first. If files exist, choose the RGB topic in the Foxglove Image panel.

Depth looks black
: Depth is `32FC1`; tune Image panel depth/color scale min/max.

Camera live port has no data
: Check that `sensors.cameras.<name>.enabled` is `true`, `output.foxglove_live_port` is set, and Foxglove connects to the camera port rather than the state-stream port.

Wrist camera does not follow the robot
: Check `parent_prim_path` points to the real robot link and `prim_path` is under that link path.

Rendering warnings appear immediately after READY
: READY precedes lazy first-frame rendering. Classify the message by plugin and the warning guidance above; log order alone does not make it a startup failure.

Control-only runs still consume substantial GPU or disk resources
: Inspect the selected env profile, not only `--gui`. Set `sensors.cameras.<name>.enabled` to `false`; disabling the GUI does not disable a sensor camera that has output configured.

Disk use grows rapidly during a long run
: The bundled `save_dir` formats are uncompressed PPM/NPY. Use the capacity formula above for every resolution, frequency, modality, and camera. Remove `save_dir` when local files are not required, lower the rate, or choose PNG/NPZ after measuring worker throughput.

The output directory already exists
: Choose the intended `runtime.camera_output.existing_data_policy`; validation and data-preservation behavior are defined in [Outputs And Persistence](../reference/outputs.md#existing-data-policies).
