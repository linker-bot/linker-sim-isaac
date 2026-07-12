# Outputs And Persistence

Language: [English](outputs.md) | [中文](../../zh-CN/reference/outputs.md)

This page is the reference for every built-in persistent or live output. It
defines who owns each destination, what is written, how existing paths are
handled, and what happens under pressure or during shutdown.

## Output Ownership And Matrix

Output destinations and process policies have different owners. Configure the
destination where the data source is declared, and configure shared writer
behavior in the runtime profile.

| Output | Entrypoint | Destination and content | Existing-data policy |
| --- | --- | --- | --- |
| Joint tracking CSV | Single Scene only | `configs/logging/*.yaml` | `runtime.output.csv_existing_file_policy` |
| State Foxglove live | Single Scene and Tiled Scene | `runtime.telemetry` | Not applicable; no file is created |
| State MCAP | Single Scene and Tiled Scene | `runtime.telemetry.mcap.path` | `runtime.output.mcap_existing_file_policy` |
| Camera local files | Single Scene and Tiled Scene | `sensors.cameras.<name>.output.save_dir` | `runtime.camera_output.existing_data_policy` |
| Camera Foxglove live | Single Scene and Tiled Scene | Camera topic prefix, live host, and live port in the env profile | Not applicable; no file is created |
| Camera MCAP | Single Scene and Tiled Scene | `sensors.cameras.<name>.output.foxglove_mcap_path` | `runtime.camera_output.existing_data_policy` |

Tiled Scene execution does not create joint tracking CSV files. State MCAP and camera
MCAP are separate sinks. Their paths belong respectively to
`runtime.telemetry.mcap.path` and each camera's env-profile output. The
`runtime.output` and `runtime.camera_output` sections own the corresponding
existing-data and shared writer policies, not those destinations.

The relevant runtime settings are:

```yaml
runtime:
  camera_output:
    queue_size: 128
    overflow_policy: block
    worker_poll_interval_s: 0.1
    existing_data_policy: error
    shutdown_policy: drain
    rgb_format: png
    depth_format: npz
    metadata_flush_interval_frames: 16
    max_bytes_per_camera: 10737418240

  telemetry:
    rate_hz: 30.0
    buffer_size: 1
    drop_policy: latest
    on_error: stop
    topics:
      joint_states: /joint_states
      scene: /scene
      state: /linkerbot/state
    mcap:
      path: logs/state.mcap
    foxglove_live:
      enabled: true
      host: 127.0.0.1
      port: 8767

  output:
    csv_existing_file_policy: error
    mcap_existing_file_policy: error

  shutdown:
    state_publisher_timeout_s: 2.0
    camera_publisher_timeout_s: 2.0
```

## Single Scene Joint CSV

Select a logging profile with `--logging-profile`. CSV output is enabled only
when that profile has both `logging.enabled: true` and a non-null
`logging.joint_tracking_path`:

```yaml
logging:
  enabled: true
  joint_tracking_path: logs/joint_tracking/run.csv
  flush_interval_s: 0.2
  interval_steps: 5
  log_actual_position: true
  log_actual_velocity: true
  log_command_position: true
  log_command_velocity: true
  log_command_effort: true
  log_action_effort: false
  log_measured_effort: false
  log_applied_effort: false
```

The configured path is a template. Single Scene creates one file per robot by adding
the numeric robot ID and stable label:

```text
configured: logs/joint_tracking/run.csv
robot 0, label left: logs/joint_tracking/run.0.left.csv
```

Every file starts with these columns:

| Column | Meaning |
| --- | --- |
| `step` | Physics step associated with the sample |
| `time_s` | Simulation time in seconds |
| `phase` | Current execution phase |
| `drive_update` | `true` when the drive target was refreshed for this sample |

Enabled measurements are then emitted for each driven joint, in joint order:

| Column pattern | Controlled by | Meaning |
| --- | --- | --- |
| `qd_<joint>_rad` | `log_command_position` | Commanded position |
| `q_<joint>_rad` | `log_actual_position` | Observed position |
| `vd_<joint>_rad_s` | `log_command_velocity` | Commanded velocity |
| `v_<joint>_rad_s` | `log_actual_velocity` | Observed velocity |
| `pos_err_<joint>_rad` | Both position fields | Command minus observed position |
| `vel_err_<joint>_rad_s` | Both velocity fields | Command minus observed velocity |
| `tau_cmd_<joint>` | `log_command_effort` | Cached effort command |
| `tau_action_<joint>` | `log_action_effort` | Effort action sent to Isaac |
| `tau_measured_<joint>` | `log_measured_effort` | Measured PhysX effort |
| `tau_applied_<joint>` | `log_applied_effort` | Applied PhysX effort |

Unavailable effort values are written as `nan`. Their physical dimension
depends on the PhysX joint type.

`interval_steps` writes when `step % interval_steps == 0`. The flush setting is
converted once using the physics step:

```text
flush_rows = max(1, round(flush_interval_s / physics_dt))
```

The writer flushes after `flush_rows` rows, not after that many physics steps.
With sampling decimation, an approximate simulation-time flush interval is
`flush_rows * interval_steps * physics_dt`.

CSV `resume` requires an existing file to have the exact configured header, a
terminated final record, valid CSV syntax, and the expected number of fields in
every row. Validation completes before append begins.

## State Foxglove And MCAP

State live and MCAP sinks publish the same selected state frame. Topic names and
payload switches belong to `runtime.telemetry`; their complete payload contract
is documented in [Realtime State Stream](../guides/telemetry.md).

`runtime.telemetry.rate_hz: 0` disables telemetry completely. A configured live
endpoint or MCAP path is not opened while the rate is zero. A live sink never
uses an existing-data policy because it creates no file.

State MCAP uses `runtime.output.mcap_existing_file_policy`. The Foxglove SDK
opens the resolved file with overwrite disabled. MCAP cannot be safely appended,
so `resume` is rejected before either the file or a concurrently configured live
sink is opened. Use `error`, `truncate`, or `timestamped_dir`.

## Camera Files And Foxglove

Each enabled camera can independently select local files, Foxglove live, camera
MCAP, or any combination. A configured camera with no active sink is still
created, but the canonical runtime does not schedule automatic frame output for
it.

| Modality | Local payload | Foxglove live and MCAP payload | `<prefix>/info` |
| --- | --- | --- | --- |
| `rgb` | `ppm`, `png`, or `npy` | `RawImage` with `rgb8` | JSON metadata |
| `depth` | `npy` or `npz` | `RawImage` with `32FC1` | JSON metadata |
| `semantic_segmentation` | `npy` | No image payload | JSON metadata |
| `instance_segmentation` | `npy` | No image payload | JSON metadata |

Segmentation frames are preserved locally as their native arrays. Foxglove
receives their metadata on `/info`, but no segmentation `RawImage` channel is
created.

A local directory can contain:

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

An NPZ depth payload stores its array under the `data` key. RGB and depth are
normalized to contiguous RGB8 and float32 arrays before output. Foxglove
`RawImage` is uncompressed even when the selected local file format is
compressed.

Multiple cameras may share a live host/port or camera MCAP path. They share the
underlying sink and retain distinct topic prefixes. Each local `save_dir`
remains a separate camera namespace.

## Camera Metadata

Local `metadata.jsonl` contains one strict JSON object per committed modality
frame. The row is appended only after its payload has been created and checked:

```json
{
  "camera_name": "world_rgbd",
  "modality": "depth",
  "frame_index": 12,
  "simulation_step": 480,
  "time_s": 2.0,
  "shape": [480, 640],
  "dtype": "float32",
  "relative_path": "depth/000012.npz",
  "intrinsics": [[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
  "camera_position_world": [0.5, -0.6, 0.8],
  "camera_orientation_world": [1.0, 0.0, 0.0, 0.0]
}
```

| Field | Presence |
| --- | --- |
| `camera_name`, `modality`, `frame_index` | Always |
| `simulation_step`, `time_s` | Always; simulation clock |
| `shape`, `dtype` | Always; stored frame array |
| `relative_path` | Local metadata only |
| `intrinsics` | Present when the camera API supplies it |
| `camera_position_world`, `camera_orientation_world` | Present when world pose is available |

Foxglove `<prefix>/info` sends the same metadata without `relative_path`, because
there is no local payload associated with the live or MCAP message.

`metadata_flush_interval_frames` counts modality rows, not complete multi-modal
camera samples. Closing the sink flushes remaining rows.

## Existing Data Policies

The three policy fields accept the same values, but each applies only to its
owned targets:

| Policy | Result |
| --- | --- |
| `error` | Fail when the final target already exists, including an empty directory. |
| `truncate` | Recheck every planned target, remove an existing target, and create a new empty file namespace. |
| `resume` | Reuse an existing target of the expected type after that sink validates its contents. |
| `timestamped_dir` | Resolve the target beneath a newly named UTC run directory. |

For a directory request, `timestamped_dir` resolves to
`requested/<UTC-run>/`. For a file request, it resolves to
`parent/<UTC-run>/filename`.

A CSV robot batch shares one run name, and one prepared camera batch shares one
run name across its local directories and camera MCAP files. A separately
prepared state MCAP may use a different run name. Consumers must use the
resolved paths reported for their sinks instead of assuming one process-wide
timestamp.

`resume` is sink-specific:

- CSV validates the exact header, final newline, CSV syntax, and row width.
- Local camera output validates every metadata row and referenced payload,
  scans unindexed payloads, and computes safe next indices.
- State MCAP and camera MCAP reject `resume` because the SDK cannot append
  safely.

Camera resume requires strict JSON, the expected camera owner, unique
modality/index/path tuples, an exact derived `relative_path`, and an existing
payload for every indexed row. The final metadata row must end with a newline.
Unindexed payloads are retained and their indices are reserved. The highest
payload for every modality must be completely readable; an incomplete or
unreadable orphan rejects resume.

## Joint Preflight And Path Safety

Single Scene validates its CSV targets, local camera directories, camera MCAP files,
and any additional state MCAP plan as one startup set. Tiled Scene validates camera
targets and state MCAP together. Validation happens before a sink is opened or
an existing target is truncated.

The final output target must not be a symbolic link. Local camera resume also
rejects symbolic links inside the output tree. Canonical targets in one startup
must be distinct and may not have an ancestor/descendant relationship. This
prevents one directory operation from deleting another sink's file.

Preflight and immediate revalidation reduce the path race window, but they are
not a cross-process lock. Do not start concurrent writers against the same
namespace. Applying several path plans is not a filesystem transaction: an I/O
failure can leave earlier changes applied and later changes unapplied.

Likewise, publishing one frame to several sinks is sequential, not atomic. An
earlier sink can accept a frame before a later sink fails. Status and metadata
should be used to reconcile such an interrupted run.

## Capacity And Quotas

`runtime.camera_output.max_bytes_per_camera` is applied independently to every
local camera directory. It includes all existing regular files, the next encoded
payload, and the metadata row that would index that payload.

The recorder must encode a payload before its exact size is known. If payload
plus metadata would exceed the quota, it deletes the new unindexed payload and
does not commit the metadata row. A cleanup failure is reported explicitly and
can leave an orphan that requires inspection. If metadata append fails, the
recorder truncates metadata to the previous offset and deletes the payload;
incomplete compensation is reported as a separate error.

For uncompressed RGB8 and float32 depth, estimate one output direction with:

```text
bytes_per_second
  = camera_count * width * height * frequency_hz
    * sum(bytes_per_pixel_per_modality)

rgb8  = 3 bytes/pixel
depth = 4 bytes/pixel
```

Local compression changes disk use and worker CPU cost. It does not reduce the
uncompressed Foxglove `RawImage` payload or the in-memory queue item.

## Queue And Error Policies

Camera frames and state snapshots use separate queues and intentionally
different pressure policies:

| Behavior | Camera output | State telemetry |
| --- | --- | --- |
| Queue item | One modality frame | One immutable state snapshot |
| Capacity | Positive `camera_output.queue_size` | Positive `telemetry.buffer_size` |
| Producer behavior | May block or fail according to policy | Never blocks the physics producer |
| Pressure policies | `block`, `error`, `drop_oldest`, `drop_newest` | `latest`, `drop_oldest`, `drop_newest` |
| Worker error | Fail-stop; remember first error and discard queued frames | `on_error: stop` or `continue` |

Camera `block` waits for space while periodically checking for worker failure or
shutdown. `error` raises immediately when full. `drop_newest` rejects the new
frame; `drop_oldest` evicts an admitted frame. Dropping policies are accepted
only when every camera sink is live-only. Any local directory or camera MCAP
requires `block` or `error` and is rejected at startup otherwise.

Telemetry `latest` replaces all pending snapshots with the newest one.
`drop_oldest` evicts the oldest snapshot only when full, and `drop_newest`
rejects the new snapshot when full. All three keep the simulation producer
non-blocking. `on_error: stop` halts publishing; `continue` records the error and
accepts later snapshots.

## Shutdown And Status

Camera `shutdown_policy: drain` writes every admitted frame before closing;
`abort` discards queued frames and increments `aborted_frames`. The join is
bounded by `runtime.shutdown.camera_publisher_timeout_s`.

Telemetry always stops admission, drains admitted snapshots, and then closes
live and MCAP sinks. Its join is bounded by
`runtime.shutdown.state_publisher_timeout_s`.

If either join times out, the thread and sink remain open. Closing a sink while
its worker may still write would be unsafe, so the owning runtime retains the
handle for status reporting and a later close attempt.

Camera status includes queue depth/capacity, selected policies, published,
dropped, aborted, and overflow-error counters, thread/sink/timeout flags,
`last_error`, and nested sink quota information when available. Telemetry status
includes buffer depth/capacity, dropped snapshots, error count, last published
sequence, last error, and thread/sink/timeout state. Inspect these fields before
treating process exit as proof that every queued item was persisted.

## Related Documentation

- [Camera Types And Sensors](../guides/cameras.md) covers camera placement,
  sampling, modalities, tiled scope, rendering, and capacity selection.
- [Realtime State Stream](../guides/telemetry.md) defines state topics, payloads,
  Single Scene/Tiled Scene sampling, and effort selection.
- [Foxglove Quick Reference](../guides/foxglove.md) provides minimal launch and
  connection examples.
- [Runtime Configuration](configuration.md) describes strict profile ownership
  and validation.
