# Output Reference

Language: [English](outputs.md) | [中文](../../zh-CN/reference/outputs.md)

Persistent output, rendering, cameras, logging, and telemetry belong to Mirror. The
Kaleidoscope runtime has no output profile or hidden capture switch; training code
must create explicit cold checkpoints or downstream metrics outside the environment.

## Configuration Shape

```yaml
outputs:
  render:
    enabled: true
    gui: false
    renderer: RaytracedLighting
    width: 640
    height: 480
    samples_per_pixel_per_frame: 1
  camera:
    enabled: true
    queue_size: 128
    overflow_policy: block
    rgb_format: ppm
    depth_format: npy
    shutdown_timeout_s: 2.0
  logging:
    enabled: true
    existing_data_policy: timestamped_dir
    hybrid_control_path: logs/hybrid_control/hybrid_force_position.csv
    log_hybrid_control: false
  telemetry:
    enabled: true
    rate_hz: 60.0
    buffer_size: 1
    include_hybrid_control: true
    shutdown_timeout_s: 2.0
```

The profile has exactly these four sections.

For `camera`, `logging`, and `telemetry`, disabling the section does not require
deleting its valid paths, ports, topics, or policy values. The strict parser still
checks retained values for type, range, and schema correctness, but scene assembly
does not resolve or preflight those paths, bind those ports, or create the disabled
sink. Enabling a section still requires its runtime consumers: at least one camera
destination, a logging path, or at least one telemetry endpoint respectively.

## Render Fields

| Field | Contract |
| --- | --- |
| `enabled` | Boolean render transaction switch. |
| `gui` | Boolean viewport/window request. |
| `renderer` | Nonempty Kit renderer name. |
| `width`, `height` | Positive output dimensions. |
| `samples_per_pixel_per_frame` | Positive sample count. |

Camera capture may require rendering even when an interactive GUI is disabled. For
Newton, one `render()` call already includes exactly one state-to-host-to-USD
publish followed by the Kit update; consumers must not invoke a separate pre-render
sync.

## Camera Output Fields

| Field | Contract |
| --- | --- |
| `enabled` | Enable the camera output path. When true, at least one destination and one scene camera are required; when false, retained destinations are ignored. |
| `queue_size` | Positive bounded frame capacity. |
| `overflow_policy` | `block`, `drop_oldest`, `drop_newest`, or `error`. |
| `rgb_format` | `ppm`, `png`, or `npy`. |
| `depth_format` | `npy` or `npz`. |
| `shutdown_timeout_s` | Positive bounded worker shutdown time. |

Camera geometry, pose, modality, frequency, and clipping planes belong to the scene
profile. Encoding, queueing, and shutdown belong here. Do not duplicate either set of
facts in the other profile.

## Logging Fields

`logging.enabled` is boolean. When true, `joint_tracking_path` is required. When
false, that path may remain in the profile but is not resolved or opened.
`existing_data_policy` is one of:

- `error`: fail if the destination already contains data;
- `truncate`: replace the maintained output set;
- `resume`: continue only when the writer can validate compatibility;
- `timestamped_dir`: create a new run directory.

The strict parser returns one frozen `LoggingOutputSettings` value. Mirror scene
assembly and `JointTrackingLogger` consume that same value directly, including its
sampling cadence, column switches, flush interval, and existing-data policy. There
is no second runtime logging configuration or per-field projection.

Use atomic metadata/manifests when adding a new writer. Never infer compatibility
from a filename alone.

`log_hybrid_control: true` additionally requires `hybrid_control_path` and the master
logging switch. It writes one bounded-width row at the common `interval_steps`
cadence. Vector fields are compact JSON arrays; scalar columns include request/robot,
step/tick/time, phase, tare and parameter generations, and Jacobian conditioning. The
hybrid CSV participates in the same preflight, existing-data policy, collision check,
and runtime close order as joint CSV, camera, and MCAP outputs.

## Telemetry Fields

| Field | Contract |
| --- | --- |
| `enabled` | Boolean publisher switch. When true, at least one live or MCAP endpoint and one message modality are required; when false, retained endpoints are ignored. |
| `rate_hz` | Nonnegative; must be positive when enabled. |
| `buffer_size` | Positive bounded handoff capacity. |
| `include_efforts` | When true, `joint_effort_field` must select a source other than `none`. When false, a retained valid source is ignored and runtime publishes no effort array. |
| `include_hybrid_control` | Publish the cached hybrid diagnostic on `topics.hybrid_control`; inactive payloads contain only `active: false`. |
| `shutdown_timeout_s` | Positive bounded sink shutdown time. |

Telemetry is an observation path and must not mutate simulation state. Sampling
occurs after a completed physics step. A slow sink must obey the configured bounded
handoff rather than block the Isaac owner thread without limit.

## Ownership And Shutdown

Mirror closes output workers, camera resources, and planners before controllers/views
and before `IsaacSession`. If a worker remains live at its timeout, shutdown reports
the live resource and retains the session. Destroying the stage while a camera or
publisher still references it is prohibited.

## Paths And Security

- Resolve user output paths before opening writers.
- Keep runtime data outside tracked source/config directories.
- Reject path traversal from identifiers used in filenames.
- Avoid pickle for interchange artifacts.
- Treat captured RGB/depth/state as potentially sensitive operational data.

See [Cameras](../guides/cameras.md), [Telemetry](../guides/telemetry.md), and
[Foxglove](../guides/foxglove.md).
