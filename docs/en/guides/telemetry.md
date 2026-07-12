# Realtime State Stream

Language: [English](telemetry.md) | [中文](../../zh-CN/guides/telemetry.md)

Single Scene and Tiled Scene runtimes can publish immutable simulation-state snapshots to a
Foxglove live server, an MCAP file, or both. Telemetry is observation-only: it
does not accept control commands and does not change simulation state.

## Install And Configure

Foxglove output requires the optional SDK dependency:

```bash
uv sync --all-extras
```

Process-level telemetry settings belong to `configs/runtime/*.yaml`. Explicit
CLI values override the selected runtime profile for that launch.

```yaml
runtime:
  telemetry:
    primary_env_id: 0
    rate_hz: 30.0
    buffer_size: 1
    drop_policy: latest
    on_error: stop
    include_joint_states: true
    include_state_json: true
    include_scene_markers: false
    include_efforts: false
    include_objects: false
    joint_effort_field: none
    topics:
      joint_states: /joint_states
      scene: /scene
      state: /linkerbot/state
    mcap:
      path: null
    foxglove_live:
      enabled: true
      host: 127.0.0.1
      port: 8767
```

Tiled Scene runtime profiles additionally use `selected_env_ids` and `publish_decimation`:

```yaml
runtime:
  telemetry:
    primary_env_id: 0
    selected_env_ids: [0, 2]
    publish_decimation: 2
```

`primary_env_id` must be one of the selected Tiled envs. Topic names must be
distinct absolute paths. `rate_hz: 0` disables telemetry completely, so neither
a configured live endpoint nor a configured MCAP path is opened.

## Endpoint Boundaries

Control TCP JSONL, control WebSocket, Foxglove state live, and camera live are
different services. They cannot share a port, and sending a control JSON object
to a Foxglove endpoint does not execute it.

Every built-in listener accepts only `localhost` or a numeric loopback address.
It provides neither authentication nor TLS. Remote access must terminate an
authenticated TLS proxy or SSH tunnel on the configured loopback endpoint.

Use a distinct port and distinct topic namespace for each concurrent runtime.
Camera live ports are configured per camera in the env profile; they are not the
state `--foxglove-live-port`.

## Sampling And Threading

Single Scene sampling runs after the shared `world.step()`. The simulation thread reads
all robots and optional objects into one immutable `StateSnapshot`; the
publisher thread never accesses Isaac or USD objects. Every robot in a snapshot
therefore has the same physics time.

Single Scene converts the requested rate to an integer physics-step interval:

```text
interval_ticks = max(1, round(1 / (physics_dt * rate_hz)))
actual_rate_hz = 1 / (physics_dt * interval_ticks)
```

A request above the physics frequency produces at most one snapshot per physics
step. Joint acceleration is a velocity difference between consecutive sampled
snapshots. It is unavailable on the first sample and the first sample after a
reset, where JSON uses `null`.

Tiled Scene telemetry reads a completed `get_state` result on the main thread and
freezes only `selected_env_ids`. `rate_hz` schedules idle-loop publication;
`publish_decimation` filters ordinary events by global step. `reset` and
`set_state` events bypass decimation, and successful interactive responses also
attempt to publish the current state. The background publisher consumes only
the frozen frame.

## Topics And Schemas

The same three configured topic names are used by Single Scene and Tiled Scene:

| Config field | Encoding | Single Scene | Tiled Scene |
| --- | --- | --- | --- |
| `topics.joint_states` | Foxglove `JointStates` | All robots | All robots from `primary_env_id` |
| `topics.scene` | Foxglove `SceneUpdate` | Runtime-object markers | Robot TCP markers and optional object markers from `primary_env_id` |
| `topics.state` | JSON | Full Single Scene snapshot | Full selected-env state |

Standard `JointStates` concatenates robots in stable runtime order. Names use
`<robot_label>/<joint_name>` so identical joint names on different robots do not
collide. Use the JSON topic when robot boundaries or all selected Tiled envs are
required.

All timestamps are simulation time. Joint position is rad, velocity is rad/s,
acceleration is rad/s2, and world position is m. Non-finite numeric values are
encoded as JSON `null`.

## Single Scene Telemetry

Start a live Single Scene stream with:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --gui \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --state-rate-hz 30 \
  --state-include-efforts --state-include-objects \
  --foxglove-joint-effort-field measured \
  --foxglove-live-port 8766
```

Record the same stream to MCAP by setting `--foxglove-mcap-path`; live and MCAP
may be enabled together. At least one of the two destinations is required to
create the state stream.

Important Single Scene CLI overrides are:

| Option | Meaning |
| --- | --- |
| `--state-rate-hz` | Non-negative target sample rate; `0` disables telemetry |
| `--state-include-efforts` | Sample commanded, measured, and applied effort arrays |
| `--state-include-objects` | Sample runtime-object root world poses |
| `--foxglove-live-host` / `--foxglove-live-port` | Loopback live endpoint |
| `--foxglove-mcap-path` | State MCAP destination |
| `--foxglove-joint-effort-field` | Standard JointStates effort source: `none`, `commanded`, `measured`, or `applied` |

When a Single Scene telemetry sink is enabled,
`include_scene_markers: true` requires `include_objects: true`, because scene
markers come from runtime objects.

The Single Scene JSON payload has this shape:

```json
{
  "step": 120,
  "time_s": 0.5041666667,
  "phase": "ik_offset",
  "robots": [
    {
      "robot_id": 0,
      "label": "ar5v2_l6v1_0",
      "joint_names": ["AR5V2_L_arm_joint_1"],
      "positions_rad": [0.12],
      "velocities_rad_s": [0.03],
      "accelerations_rad_s2": [0.4],
      "commanded_efforts": [null],
      "measured_efforts": [0.8],
      "applied_efforts": [0.75]
    }
  ],
  "objects": {
    "Tblock": {
      "prim_path": "/World/TBlock",
      "position_m": [0.15, 0.0, -0.4],
      "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

`phase` may be absent while idle. Effort arrays are `null` when collection is
disabled; unavailable individual values are also `null`. Failure to read an
optional effort API does not terminate the state stream.

## Tiled Scene Telemetry

Start a Tiled Scene stream with:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --env scene3_tiled \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --foxglove-live-port 8767 \
  --telemetry-env-ids 0,2 \
  --telemetry-rate-hz 10 \
  --telemetry-decimation 2
```

Important Tiled Scene CLI overrides are:

| Option | Meaning |
| --- | --- |
| `--telemetry-env-ids` | Nonempty, unique, in-range comma-separated env IDs |
| `--telemetry-primary-env-id` | Single selected env used by standard topics |
| `--telemetry-rate-hz` | Idle-loop publication rate; `0` disables all telemetry sinks |
| `--telemetry-decimation` | Positive global-step filter for ordinary events |
| `--telemetry-full-batch-json` | Enable the selected-env JSON topic |
| `--telemetry-joint-states` | Enable standard JointStates for the primary env |
| `--foxglove-live-port` | Enable the state live sink |
| `--foxglove-mcap-path` | Enable the state MCAP sink |

The custom JSON retains the selected-env dimension:

```json
{
  "event": "step",
  "step": 42,
  "time_s": 0.175,
  "env_ids": [0, 2],
  "state": {
    "robots": {
      "ar5v2_l6v1_0": {
        "joint_names": ["AR5V2_L_arm_joint_1"],
        "joint_positions": [[0.1], [0.2]],
        "joint_velocities": [[0.0], [0.0]],
        "tcp_positions_world": [[0.3, 0.0, 0.4], [6.3, 0.0, 0.4]],
        "tcp_orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
      }
    },
    "episode_steps": [42, 42],
    "episode_ids": [0, 0]
  },
  "trigger": {
    "event": "step",
    "accepted": true,
    "kind": "joint_delta_pos"
  }
}
```

The robot map key is the stable runtime label. `include_objects: false` omits
object state and object markers, but does not remove robot TCP markers.
`include_efforts: true` samples measured/applied effort; standard Tiled Scene
JointStates uses measured effort when available.

## Output Pressure And Persistence

Telemetry uses a bounded, non-blocking producer buffer. The default
`buffer_size: 1` and `drop_policy: latest` keep the newest snapshot. The other
choices are `drop_oldest` and `drop_newest`; `on_error` selects `stop` or
`continue` after a publisher error.

State MCAP uses `runtime.output.mcap_existing_file_policy` and rejects `resume`.
Normal close stops admission, drains accepted snapshots, then closes live/MCAP
sinks. The exact existing-data policies, joint path preflight, buffer behavior,
join timeout, and failure status are defined in
[Outputs And Persistence](../reference/outputs.md).

## Status And Troubleshooting

Single Scene and Tiled Scene expose telemetry status including `buffer_depth`,
`buffer_capacity`, `dropped_snapshots`, `error_count`,
`last_published_sequence`, `last_error`, `thread_alive`,
`shutdown_timed_out`, and `sink_closed`.

No state-stream startup event
: Confirm `rate_hz > 0` and configure at least one live port or MCAP path.

Foxglove cannot connect
: Confirm the client uses the Foxglove live port, not a control or camera port.
Check that the rate is positive and that any remote proxy terminates on the
configured loopback address.

JointStates effort is empty
: For Single Scene, enable effort sampling and select a non-`none` effort field. For
Tiled Scene, enable `include_efforts`. The current Isaac articulation wrapper must also
provide the requested effort.

The first acceleration is `null`
: This is expected before two sampled velocity frames exist. Reset clears the
velocity history.

Tiled Scene live data stops while idle
: Use `--idle-physics-policy hold_step` so the main loop continues to pump the
runtime, and use `--stdin-eof-policy keep_alive` for a long-running stdin launch.

For topic selection and connection steps, see
[Foxglove Quick Reference](foxglove.md). For control payloads, see
[Single Scene JSON](../reference/single-scene-json.md) and
[Tiled Scene JSON](../reference/tiled-scene-json.md).
