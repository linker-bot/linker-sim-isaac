# Tiled Scene Runtime And JSON Protocol

Language: [English](tiled-scene-json.md) | [中文](../../zh-CN/reference/tiled-scene-json.md)

This page owns the JSON transport, message, selector, lifecycle, and response
contracts for the independent Isaac Tiled Scene runtime. For checkout setup and a
complete first request, use the [Tiled Scene Quickstart](../getting-started/tiled-scene-quickstart.md).
For every launch option, effective default, and process marker, use the
[Tiled Scene CLI Reference](tiled-scene-cli.md). cuRobo algorithms and resource behavior are
covered by [Motion Planning](../guides/motion-planning.md).

## 1. Runtime Model

The Tiled Scene runtime clones `num_envs` homogeneous environments into one Isaac
stage. Every env has independent articulation/object state, episode counters,
and trajectory playback, while all envs share one physics clock:

```text
JSON request
  -> parse robot_id/env_ids on the main thread
  -> compute targets for selected envs
  -> compose a full-batch command target
  -> write every selected articulation for the same physics tick
  -> call world.step() once
```

One high-level `step` expands into a fixed number of physics ticks. The
`ee_linear_path` action can instead specify a logical `duration_s`; execution
rounds up to the physics-dt grid. Unselected envs do not stop global physics
time. They keep their existing command targets while advancing with the shared
World.

Asynchronous planner workers receive copied numpy current state, goals, and
selectors from the main thread. They do not access Isaac `World`, the USD
stage, articulation views, or PhysX handles. Planner completion does not advance
simulation. A result must enter `TiledTrajectoryBuffer` and then be played with
`step_trajectory`.

## 2. Transport Framing And Concurrency

Launch and endpoint activation are documented in the
[Tiled Scene CLI Reference](tiled-scene-cli.md). Once enabled, the control transports use
these protocol boundaries:

| Transport | Request framing | Direct response | Additional delivery |
|---|---|---|---|
| stdin | One JSON object per line | One JSON line on stdout | None |
| TCP JSONL | One JSON object per line | One response line on the same connection | None; poll lifecycle APIs |
| WebSocket | One JSON object per text message | JSON text on the same connection | Broadcast copies of processed responses |

Every request receives one direct response. A TCP client should keep its
connection open and alternate one request line with one response line; it must
not depend on one request per connection. Async planner state remains queryable
through `planner_status`.

Transport threads parse strict JSON and enqueue requests. Isaac, USD, and PhysX
access stays on the simulation main thread. TCP and WebSocket share the runtime
`max_connections` admission limit; stdin does not consume it. All inputs share
the bounded request queue and `max_message_bytes`, and every WebSocket has a
bounded event queue. Oversized, non-UTF-8, duplicate-key, `NaN`, infinite, or
trailing JSON input is rejected before main-thread dispatch.

## 3. Runtime Configuration Boundary

The Tiled Scene entrypoint requires an env profile with `tiled.enabled: true` and at least one
homogeneous env row. Clone count, layout, per-env pose/metadata overrides, camera
env scope, and collision filtering belong to the env profile; there is no CLI
override for `num_envs`. Planner selection, capacity, failure policy, playback
limits, and telemetry belong to the runtime profile.

Use the [Configuration Guide](../guides/configuration.md) for authoring workflows
and the [Configuration Reference](configuration.md) for exact accepted fields
and validation. Collision strategy and planning-world separation are owned by
[Collision Models](../guides/collision-models.md); planner algorithms and batch
resources are owned by [Motion Planning](../guides/motion-planning.md).

## 4. Messages And Selectors

### 4.1 Complete Message Index

<!-- tiled-message-index:start -->
| `type` | Execution | Purpose |
|---|---|---|
| `status` | synchronous | Runtime, env, and robot discovery |
| `step` | synchronous | Fixed-tick batched command action |
| `reset` | synchronous | Restore selected envs to initial state |
| `get_state` | synchronous | Read transient batched runtime state |
| `set_state` | synchronous | Write selected-env command state |
| `get_snapshot` | synchronous | Read one env as a persistent snapshot |
| `set_snapshot` | synchronous | Broadcast one snapshot to target envs |
| `clone_state` | synchronous | Clone one env inside the Tiled Scene runtime |
| `load_trajectory` | synchronous | Load an existing joint trajectory |
| `hand` | synchronous submission | Queue a sparse joint subtrack; append by default |
| `step_trajectory` | synchronous | Advance trajectory playback |
| `trajectory_status` | synchronous | Inspect the playback buffer |
| `clear_trajectory` | synchronous | Clear playback entries |
| `plan` | asynchronous submission | Copy state and enter the planner queue |
| `planner_status` | synchronous collection | Dispatch and collect planner results |
| `cancel_plan` | synchronous | Cancel queued/running planning requests |
| `clear_completed` | synchronous | Remove completed summaries |
| `quit` | synchronous | Request main-loop shutdown |
<!-- tiled-message-index:end -->

All input errors return:

```json
{"event":"rejected","error":"..."}
```

A rejected message does not stop the main loop.

### 4.2 Selector Rules

| Target | Field | Rule |
|---|---|---|
| One robot | `robot_id` | Nonnegative ID in the current session |
| Several robots | `robot_ids` | Nonempty duplicate-free ID array |
| Every robot | `robot_ids: "all"` | Only messages with multi-robot semantics |
| One env | `env_id` | Snapshot source only |
| Several envs | `env_ids` | Nonempty duplicate-free array; every env-scoped command requires it explicitly |
| Clone source | `source_env_id` | One env ID |
| Clone targets | `target_env_ids` | Nonempty env-ID array |

The following messages always require explicit `env_ids`; omission never means
every env:

<!-- tiled-env-ids-required-index:start -->
| `type` | Scope |
|---|---|
| `reset` | Selected env state |
| `get_state` | Selected env state |
| `set_state` | Selected env state |
| `set_snapshot` | Snapshot restore targets |
| `load_trajectory` | Playback targets |
| `step_trajectory` | Playback targets |
| `trajectory_status` | Playback rows queried |
| `clear_trajectory` | Playback rows cleared |
| `hand` | Hand-motion targets |
| `plan` | Planning problem rows |
| `step` | Control targets |
<!-- tiled-env-ids-required-index:end -->

`cancel_plan` does not require an env selector when `request_id` identifies one
request. Without `request_id`, its robot/env intersection form requires
`env_ids`.

When the scene has multiple robots, `step` and `step_trajectory` require an
explicit `robot_id` or `robot_ids`. `plan`, `load_trajectory`, and `hand` are
single-robot messages and require `robot_id` in a multi-robot scene. Public
control does not accept label, name, side, or role selectors. The response
boundary converts internal labels back into session robot IDs.

### 4.3 Common Responses And `quit`

Successful non-status responses generally contain:

| Field | Meaning |
|---|---|
| `event` | Interface-specific event such as `step`, `state`, or `plan_submitted` |
| `accepted` | Validation and synchronous operation/submission succeeded |
| `backend` | Runtime backend; `TiledSceneRuntime` reports `isaac` |
| `step`, `time_s` | Global physics step and simulation time; pure queue responses can omit them |
| `env_ids` | Envs explicitly selected by the request and validated by the runtime |
| `robot_id`, `robot_ids` | Session IDs produced by the public response boundary |

Selector, shape, or runtime errors use the common rejection envelope:

```json
{"event":"rejected","error":"env_ids contains out-of-range env id"}
```

Do not assume every rejection is side-effect-free. Only interfaces explicitly
documented as validating atomically guarantee that no state was written.

Exit request and response:

```json
{"type":"quit"}
```

```json
{"event":"quit","accepted":true}
```

`quit` sets the runtime exit event. The main loop stops after the current
message. It is not a pause and does not persist playback, planner cache, or
episode state.

The entrypoint then stops stdin, WebSocket, and TCP resources, closes telemetry,
and closes the runtime. Runtime close waits for planner workers, camera output,
and each owned IK context before closing the SimulationApp. Transport waits use
`runtime.shutdown.transport_timeout_s`; state and camera publishers use their
respective shutdown settings; planner workers use
`runtime.planner.resources.shutdown_timeout_s`. A timed-out child remains owned,
and `runtime.close()` is idempotent so an owner can retry without re-closing
already completed children. `TILED_SCENE_INTERACTIVE_EXIT` is printed from the final
cleanup block even when a resource timed out, so any preceding
`TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT` means shutdown was incomplete.

## 5. Status And Session Discovery

```json
{"type":"status"}
```

The following response is from a hypothetical four-env, one-robot profile. It
does not represent the bundled `scene3_tiled` scale: that profile currently has
64 envs and two robots, so its arrays and `robots[]` contain the corresponding
rows.

Representative response:

```json
{
  "event": "status",
  "backend": "isaac",
  "env": "demo_tiled",
  "num_envs": 4,
  "step": 0,
  "time_s": 0.0,
  "episode_steps": [0, 0, 0, 0],
  "episode_ids": [0, 0, 0, 0],
  "env_roots": [
    "/World/envs/env_0",
    "/World/envs/env_1",
    "/World/envs/env_2",
    "/World/envs/env_3"
  ],
  "env_origins": [
    [0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [6.0, 0.0, 0.0],
    [9.0, 0.0, 0.0]
  ],
  "runtime": {"inspect_env_ids": [0]},
  "per_env_metadata": [],
  "robots": [
    {
      "robot_id": 0,
      "label": "ar5v2_l6v1_0",
      "robot_profile": "ar5v2_l6v1_l",
      "kind": "arm_hand",
      "supports_planning": true,
      "count": 4,
      "num_dof": 30,
      "command_joints": ["..."],
      "ik_tcp_frame": "AR5V2_L_pinch_tcp"
    }
  ],
  "sensors": {"cameras": []}
}
```

`env_id` and `robot_id` are independent dimensions. Robot 0 describes one
robot definition whose batched articulation has `num_envs` rows. Robot IDs are
valid only for the current process.

`status` takes no selector and does not include `accepted`:

| Field | Meaning |
|---|---|
| `num_envs` | Number of rows in each batched articulation |
| `env_roots`, `env_origins` | USD root and world-frame translation origin for every env |
| `episode_steps`, `episode_ids` | Per-env episode-local steps and reset generation |
| `robots[].robot_id` | Session robot ID used by public commands |
| `robots[].command_joints` | Column order for step, trajectory, plan, and state commands |
| `robots[].supports_planning` | Valid cuRobo binding; the async linear backend can still work when false |
| `robots[].ik_tcp_frame` | Default TCP for synchronous `ee_*` actions |

The response always includes `per_env_metadata`, which is an empty array when
no env fragment defines metadata. Enabled providers and resources can also add
`transport`, `telemetry`, `planner`, and `camera_output` diagnostics. Read
status by field name rather than assuming the example is exhaustive.

## 6. Canonical `step` Actions

The only synchronous action envelope uses top-level `type="step"` and an
explicit `kind`. Ordinary actions share `values`; `ee_linear_path` can instead
use named absolute or relative target fields at the same top level:

```json
{
  "type": "step",
  "kind": "joint_delta_pos",
  "robot_ids": [0, 1],
  "env_ids": [0, 1],
  "values": [0.01, 0.2, 0.0],
  "decimation": 2,
  "interpolation": "smoothstep"
}
```

The outer `type` is `step`; `kind` and every action parameter are direct
top-level fields as shown above. There is no nested `action` mapping, and fields
not consumed by the selected `kind` are rejected.

### 6.1 Action Fields

| Field | Default | Meaning |
|---|---|---|
| `kind` | required | One of the seven actions below |
| `values` | required except hold and named linear path | `(D,)` or `(E,D)`; for `ee_linear_path`, compact world-frame offset |
| `decimation` | runtime execution default | Positive physics-tick count; for `ee_linear_path`, an explicit alternative to `duration_s` |
| `duration_s` | runtime planner request default | `ee_linear_path` only; positive logical duration, mutually exclusive with `decimation` |
| `sample_dt_s` | physics dt | `ee_linear_path` only; sequential batched-IK period |
| `interpolation` | runtime command default (`smoothstep` in bundled profiles) | `linear` or `smoothstep` |
| `tcp_frame_name` | robot default | Registered TCP for `ee_*` |
| `pose_reference_frame` | runtime command default (`env` in bundled profiles) | `env`, `base`, or `world` |
| `target_offset` | none | Named linear-path relative translation; exclusive with target position/values |
| `target_position` | none | Named linear-path absolute endpoint; exclusive with target offset/values |
| `orientation_mode` | runtime command default (`current` in bundled profiles) | Linear path only: `free`, `current`, or `target` |
| `target_orientation_quat_wxyz` | none | Required for `orientation_mode=target` |

For synchronous `ee_linear_path`, explicit JSON takes precedence over
`runtime.planner.request_defaults` and `runtime.execution.command_defaults`.
An explicit JSON `null` is not omission: `duration_s: null` and
`sample_dt_s: null` are rejected.

<!-- tiled-action-index:start -->
| `kind` | Row width | Semantics |
|---|---:|---|
| `hold` | no `values` | Keep the last command target |
| `joint_position_target` | `1..command_dim` | Absolute command-space prefix |
| `joint_delta_pos` | `1..command_dim` | Relative command-space prefix |
| `ee_pose_target` | 7 | `[x,y,z,qw,qx,qy,qz]` |
| `ee_delta_pos` | 3 | World-frame TCP translation delta, preserve orientation |
| `ee_delta_pose` | 6 or 7 | Translation plus rotvec, or translation plus absolute wxyz |
| `ee_linear_path` | 3 | TCP linear pose path with batched IK at each sampled waypoint |
<!-- tiled-action-index:end -->

Joint actions can write a command-space prefix. Uncovered columns keep their
current targets. Use `joint_names` in `load_trajectory` or `plan` for a
non-prefix subset; `step` has no name-scatter interface.

### 6.2 Common `step` Response

All seven actions complete target conversion, fixed-tick execution, and TCP
cache refresh before returning. Joint-action response:

```json
{
  "event": "step",
  "accepted": true,
  "backend": "isaac",
  "kind": "joint_position_target",
  "env_ids": [0, 1],
  "robot_ids": [0],
  "ticks": 20,
  "step": 20,
  "time_s": 0.0833333333,
  "episode_steps": [20, 20, 20, 20],
  "info": [
    {"robot_id": 0, "command_width": 2}
  ]
}
```

`ticks` is the number of physics ticks advanced by this action. Global
`step/time_s` and every env's `episode_steps` advance because all envs share one
World. Unselected envs keep their targets. `info[]` reports the actual command
width by robot ID. EE actions add `ik` and `ik_backend`. A per-env numerical IK
failure is represented inside `info[].ik`; it does not turn an otherwise valid
request into `rejected`, and the failed row keeps its seed.

### 6.3 Hold

```json
{
  "type": "step",
  "kind": "hold",
  "robot_ids": "all",
  "env_ids": [0, 1, 2, 3],
  "decimation": 2
}
```

`hold` rejects `values` and keeps each adapter's latest command target. On first
use, it holds the current joint positions. It still advances physics and is
useful for contact settling, GUI refresh, or telemetry.

### 6.4 Absolute Joint Target

```json
{
  "type": "step",
  "kind": "joint_position_target",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [[0.3, -0.4], [0.4, -0.4]],
  "decimation": 20,
  "interpolation": "linear"
}
```

A one-dimensional `values` row broadcasts to every selected env. For a matrix,
the first dimension must be 1 or `len(env_ids)`. Width D writes the first D
columns of `status.robots[].command_joints`; the suffix keeps its old target.
The action interpolates from current command position to the target over
`decimation` ticks.

### 6.5 Relative Joint Target

```json
{
  "type": "step",
  "kind": "joint_delta_pos",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [[0.05, 0.0], [-0.05, 0.0]],
  "decimation": 4,
  "interpolation": "smoothstep"
}
```

Each delta is based on selected-env command joint positions read when the
message executes. Shape and prefix rules match the absolute action.

### 6.6 Absolute TCP Pose

```json
{
  "type": "step",
  "kind": "ee_pose_target",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "values": [0.35, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "pose_reference_frame": "env",
  "decimation": 2
}
```

- `env`: position is env-local and receives each env origin.
- `base`: position and orientation are robot-base-local and convert to world.
- `world`: all selected envs share one world pose, generally useful only for
  inspecting a single env.

### 6.7 TCP Translation Delta

```json
{
  "type": "step",
  "kind": "ee_delta_pos",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "values": [0.0, 0.0, 0.01],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "decimation": 2
}
```

`values` is a world-frame TCP translation `[dx,dy,dz]`; orientation remains the
TCP orientation at command start. `pose_reference_frame` does not rotate this
compact delta. Use named `ee_linear_path.target_offset` for a frame-aware
offset.

### 6.8 TCP Pose Delta

Rotvec form:

```json
{
  "type": "step",
  "kind": "ee_delta_pose",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [0.0, 0.0, 0.01, 0.0, 0.0, 0.0872664626],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "decimation": 4
}
```

Seven-value form, where the final four values are an absolute target wxyz
quaternion rather than a quaternion delta:

```json
{
  "type": "step",
  "kind": "ee_delta_pose",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "decimation": 4
}
```

The six-value form is `[dx,dy,dz,rx,ry,rz]`, with rotvec in radians and
left-multiplied onto the current world-frame TCP orientation. Both forms use a
world-frame translation and accept `(E,6)` or `(E,7)` per-env rows.

Every `ee_*` action except `ee_linear_path` makes one batched cuRobo IK call.
With `failure_policy=hold_failed_env`, successful rows write IK solutions and
failed selected rows preserve seed/current targets. `info[].ik` reports
`ik_success`, `failed_env_ids`, `ik_position_error`, and `ik_orientation_error`
by robot ID. With `failure_policy=reject_request`, any selected failed row
rejects the complete synchronous request before any robot target or physics
write; the rejection response contains sorted `failed_env_ids`.

### 6.9 Fixed-Duration Batched TCP Line

Relative endpoint:

```json
{
  "type": "step",
  "kind": "ee_linear_path",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "target_offset": [0.0, 0.0, 0.10],
  "orientation_mode": "free",
  "duration_s": 0.4,
  "sample_dt_s": 0.02,
  "interpolation": "linear",
  "tcp_frame_name": "AR5V2_L_pinch_tcp"
}
```

Exactly one of `values`, `target_offset`, and `target_position` is required.
Named targets can broadcast one row or provide `(len(env_ids),3)` rows.

Absolute endpoint and target orientation:

```json
{
  "type": "step",
  "kind": "ee_linear_path",
  "robot_id": 0,
  "env_ids": [0, 1],
  "target_position": [0.35, 0.0, 0.4],
  "pose_reference_frame": "base",
  "orientation_mode": "target",
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "interpolation": "smoothstep"
}
```

Orientation modes:

- `free`: position-only IK.
- `current`: preserve the TCP orientation at command start.
- `target`: Slerp to required `target_orientation_quat_wxyz`.

When `target_orientation_quat_wxyz` is explicit and `orientation_mode` is
omitted, the request implies `target`, taking priority over the runtime command
default. An explicit `free` or `current` cannot be combined with a target
quaternion, and `target` without a quaternion is rejected.

`pose_reference_frame` interprets named position, offset direction, and target
orientation. `env` axes align with world and positions receive each env origin;
`base` converts with each robot base pose. Compact `values: [dx,dy,dz]` always
means world-frame offset and uses the resolved runtime orientation default.

`linear` uses equal-distance/equal-angle waypoints at equal time intervals.
`smoothstep` applies one smooth progress parameter to position and orientation
without changing the geometric line and Slerp path.

Timing rules:

1. Explicit `duration_s` and `decimation` are mutually exclusive. If both are
   omitted, `runtime.planner.request_defaults.duration_s` is injected. An
   explicit `decimation` suppresses that duration default and selects a fixed
   tick count.
2. IK waypoint count is `ceil(duration_s / sample_dt_s)`. The final waypoint is
   pinned to logical `duration_s`; exact divisibility is not required.
3. Physics ticks are `ceil(duration_s / physics_dt)`. At 100 Hz,
   `duration_s=0.405` executes 41 ticks and reports actual `duration_s=0.41`.
4. The runtime computes every IK waypoint before the first `world.step()`, uses
   the previous solution as the next seed, then resamples the sparse joint path
   to physics-tick endpoints with the same progress mode.
5. All selected envs execute the same tick count and actual duration. Unselected
   envs advance the shared World while holding their targets.
6. Under `hold_failed_env`, after an env's first IK failure that row holds its
   last successful target and other envs continue. Under `reject_request`, any
   selected failure rejects all robots before physics execution. Configuration,
   shape, or CUDA exceptions also reject the complete action before execution.

The response adds actual `duration_s`, effective `sample_dt_s`, and
`ik_waypoints`. `info[].ik` also contains:

| Field | Meaning |
|---|---|
| `ik_success` | Whether each env completed every waypoint |
| `ik_first_failure_step` | First failed waypoint, one-based; `-1` on success |
| `ik_completed_steps` | Waypoints completed before failure |
| `ik_position_error` | Maximum position error over attempted waypoints |
| `ik_orientation_error` | Maximum orientation error when supplied by the backend |

This action gives a uniform rollout horizon. It is not a collision-aware
trajectory optimizer and does not plan arcs or obstacle detours.

## 7. State, Reset, And Episodes

### 7.1 Reset

```json
{"type":"reset","env_ids":[0,1]}
```

```json
{
  "event": "reset",
  "accepted": true,
  "env_ids": [0, 1],
  "step": 120,
  "time_s": 0.5,
  "episode_steps": [0, 0, 37, 42],
  "episode_ids": [3, 3, 2, 2],
  "objects_reset": 4
}
```

Selected robot positions/velocities and object state return to startup values.
Their `episode_steps` reset to zero and `episode_ids` increment. Corresponding
trajectories are cleared and overlapping pending planner requests are cancelled.
Global `step/time_s` do not rewind, and reset does not call `world.step()`.
`objects_reset` counts successfully restored object/env pairs: object count
multiplied by the number of selected envs. It does not count an object's
internal prims or bodies. `env_ids` is mandatory; resetting every env requires
listing every ID explicitly.

### 7.2 `get_state`

```json
{"type":"get_state","env_ids":[0,1]}
```

Complete response shape:

```json
{
  "event": "state",
  "accepted": true,
  "backend": "isaac",
  "env_ids": [0, 1],
  "step": 120,
  "time_s": 0.5,
  "state": {
    "robots": [
      {
        "robot_id": 0,
        "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
        "joint_positions": [[0.1, 0.2], [0.3, 0.4]],
        "joint_velocities": [[0.0, 0.0], [0.0, 0.0]],
        "tcp_positions_world": [[0.35, 0.0, 0.4], [3.35, 0.0, 0.4]],
        "tcp_orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
      }
    ],
    "objects": {
      "Tblock": {
        "positions_world": [[0.2, 0.0, -0.4], [3.2, 0.0, -0.4]],
        "orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
      }
    },
    "episode_steps": [12, 15],
    "episode_ids": [2, 2]
  }
}
```

Use `fields` to select a subset:

```json
{
  "type": "get_state",
  "env_ids": [0],
  "fields": [
    "robots.ar5v2_l6v1_0.joint_positions",
    "objects.Tblock.positions_world",
    "episode_steps"
  ]
}
```

One-part names select top-level fields. Robot/object paths use
`robots.<internal_label>.<field>` or `objects.<object_name>.<field>`. The robot
label comes from `status.robots[].label`, not `robot_id`. Unknown fields are
ignored, so clients must verify that requested fields appear in the response.

Public `state.robots` is an array with `robot_id` and these fields:

- `joint_names`
- `joint_positions: (E,D)`
- `joint_velocities: (E,D)`
- `tcp_positions_world: (E,3)`
- `tcp_orientations_wxyz: (E,4)`

Object fields depend on the object runtime and expose root/body poses.
`get_state` is a debugging and telemetry shape, not a stable persistence format.

### 7.3 `set_state`

```json
{
  "type": "set_state",
  "env_ids": [0, 1],
  "state": {
    "robots": [
      {
        "robot_id": 0,
        "joint_positions": [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "joint_velocities": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
      }
    ],
    "episode_steps": [5, 8],
    "episode_ids": [2, 3]
  }
}
```

Successful response omits large arrays:

```json
{
  "event": "set_state",
  "accepted": true,
  "backend": "isaac",
  "env_ids": [0, 1],
  "step": 120,
  "time_s": 0.5
}
```

One robot-state row can broadcast to selected envs. Width must equal the full
command dimension ordered by `get_state.state.robots[].joint_names`. Public
input uses `robots[] + robot_id`; internal label-keyed maps are rejected.
Omitted robots or fields remain unchanged. A write clears selected-env
trajectories and cancels overlapping planner requests to prevent stale results.

Writable fields are `robots[].joint_positions`, `robots[].joint_velocities`,
`episode_steps`, and `episode_ids`. Episode values can be one broadcast value or
`len(env_ids)` values. Robot rows cannot repeat a `robot_id`. `set_state` does
not write object poses; use `set_snapshot` for persistent object restore.

## 8. Snapshots And Clone

The canonical payload, identity matching, shapes, units, restore result, and
transaction rules are owned by the [Snapshot Data And Restore Reference](snapshots.md).
This section defines only Tiled Scene envelopes and env selectors.

`get_snapshot` reads exactly one source env:

```json
{"type":"get_snapshot","env_id":0}
```

```json
{
  "event": "snapshot",
  "accepted": true,
  "backend": "isaac",
  "env_id": 0,
  "step": 120,
  "time_s": 0.5,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

The abbreviated body shows only the envelope. Broadcast the complete returned
snapshot to explicit target envs:

```json
{
  "type": "set_snapshot",
  "env_ids": [1, 2],
  "strict": true,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

`env_ids` is required, nonempty, unique, and range checked. `label_map` is
optional; `strict` defaults to true. Successful public response:

```json
{
  "event": "snapshot_restored",
  "accepted": true,
  "backend": "isaac",
  "robot_ids": [0],
  "objects": [],
  "env_ids": [1, 2],
  "partial": false,
  "step": 120,
  "time_s": 0.5
}
```

`set_snapshot` accepts only `type`, `snapshot`, `env_ids`, `label_map`, and
`strict`. The runtime broadcasts one source payload to every selected env and
returns object names once, not once per object-env pair.

Clone state inside one runtime:

```json
{
  "type": "clone_state",
  "source_env_id": 0,
  "target_env_ids": [1, 2, 3],
  "strict": true
}
```

```json
{
  "event": "state_cloned",
  "accepted": true,
  "backend": "isaac",
  "robot_ids": [0, 1],
  "objects": ["Tblock"],
  "env_ids": [1, 2, 3],
  "partial": false,
  "source_env_id": 0,
  "target_env_ids": [1, 2, 3],
  "step": 120,
  "time_s": 0.5
}
```

This is main-thread `get_snapshot(source) + set_snapshot(targets)`.
`get_snapshot` accepts `env_id`; `clone_state` identifies destinations with
`target_env_ids` and does not accept `label_map`. Both restore forms use the
transaction semantics defined by the canonical Snapshot reference.

## 9. Trajectory Buffer

### 9.1 Load A Trajectory

```json
{
  "type": "load_trajectory",
  "request_id": "trajectory-1",
  "source": "offline_planner",
  "robot_id": 0,
  "env_ids": [0, 1],
  "times": [0.0, 0.1, 0.2],
  "positions": [
    [0.0, 0.0],
    [0.1, 0.05],
    [0.2, 0.1]
  ],
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "replace": true,
  "queue": false
}
```

```json
{
  "event": "trajectory_loaded",
  "accepted": true,
  "backend": "isaac",
  "robot_id": 0,
  "env_ids": [0, 1],
  "samples": 3,
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "step": 120,
  "time_s": 0.5
}
```

| Field | Rule |
|---|---|
| `times` | Nonempty finite vector; strictly increasing for multiple samples; first value may be zero |
| `positions` | `(T,D)` broadcast or `(E,T,D)` / `(1,T,D)` |
| `joint_names` | Optional length-D unique subset of command joints |
| `request_id` | Optional ID stored in playback status |
| `source` | Optional source string; default `interactive` |
| `replace` | Default true; replace current/queued playback for selected envs |
| `queue` | Default false; when true, append takes precedence over replace |

Without `joint_names`, D columns write the command-space prefix and remaining
joints are filled from the target at load time. With names, columns scatter by
name and omitted columns still preserve the load-time target. Nested
`trajectory`, `joint_positions`, and `overlays` payloads are rejected.

### 9.2 Playback, Query, And Clear

Advance playback:

```json
{"type":"step_trajectory","robot_ids":[0],"env_ids":[0,1],"decimation":2}
```

```json
{
  "event": "trajectory_step",
  "accepted": true,
  "backend": "isaac",
  "env_ids": [0, 1],
  "robot_ids": [0],
  "ticks": 2,
  "step": 122,
  "time_s": 0.5083333333,
  "episode_steps": [22, 22, 20, 20],
  "trajectory": [
    {
      "robot_id": 0,
      "env_ids": [0, 1],
      "active_env_ids": [0, 1],
      "completed_env_ids": [],
      "idle_env_ids": [],
      "dt_s": 0.0041666667
    }
  ],
  "planner_ready": [],
  "planner_loaded": [],
  "load_rejected": []
}
```

`decimation` defaults to `--default-decimation` and must be positive. In a
multi-robot scene, select `robot_id/robot_ids`; `robot_ids="all"` advances every
robot. `planner_ready/planner_loaded` contain async results collected as a side
effect of this call.
Its `load_rejected` field uses the same per-request playback-admission structure
as `planner_status`; it is empty when every collected result loads successfully.

Query the buffer:

```json
{"type":"trajectory_status","robot_id":0,"env_ids":[0,1]}
```

```json
{
  "event": "trajectory_status",
  "accepted": true,
  "backend": "isaac",
  "step": 122,
  "time_s": 0.5083333333,
  "trajectory": {
    "num_envs": 64,
    "limits": {
      "max_queue_depth_per_env": 32,
      "max_samples_per_env": 100000,
      "max_duration_s_per_env": 3600.0,
      "overflow_policy": "reject"
    },
    "queued_trajectories": 2,
    "queued_samples": 6,
    "queued_duration_s": 0.4,
    "rejected_loads": 0,
    "rejected_loads_scope": "robot",
    "robots": [
      {
        "robot_id": 0,
        "count": 2,
        "queued_trajectories": 2,
        "queued_samples": 6,
        "queued_duration_s": 0.4,
        "rejected_loads": 0,
        "active_env_ids": [0, 1],
        "completed_env_ids": [],
        "envs": [
          {
            "env_id": 0,
            "request_id": "trajectory-1",
            "source": "offline_planner",
            "stage": "trajectory",
            "completed": false,
            "elapsed_s": 0.0083333333,
            "duration_s": 0.2,
            "progress": 0.0416666667,
            "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
            "samples": 3,
            "joint_track_names": [],
            "queue_length": 1,
            "queued_trajectories": 1,
            "queued_samples": 3,
            "queued_duration_s": 0.2
          },
          {
            "env_id": 1,
            "request_id": "trajectory-1",
            "source": "offline_planner",
            "stage": "trajectory",
            "completed": false,
            "elapsed_s": 0.0083333333,
            "duration_s": 0.2,
            "progress": 0.0416666667,
            "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
            "samples": 3,
            "joint_track_names": [],
            "queue_length": 1,
            "queued_trajectories": 1,
            "queued_samples": 3,
            "queued_duration_s": 0.2
          }
        ]
      }
    ]
  }
}
```

Clear selected entries:

```json
{"type":"clear_trajectory","robot_id":0,"env_ids":[0,1]}
```

```json
{
  "event": "trajectory_cleared",
  "accepted": true,
  "backend": "isaac",
  "cleared": [{"robot_id": 0, "env_ids": [0, 1]}],
  "step": 122,
  "time_s": 0.5083333333
}
```

Each `step_trajectory` request collects ready planner results once, before
entering its physics-tick loop. A future that finishes during that loop remains
for the next collection. Each tick then samples the head playback for every
selected env; envs without a trajectory hold their targets. The response
separates active, completed, and idle envs and reports this call's
ready/loaded/load-rejected planner summaries.

`trajectory_status` requires `env_ids`; its robot selector may be omitted to
query every robot for those envs. Its fixed `trajectory` payload includes the
configured limits plus aggregate `queued_trajectories`, `queued_samples`,
`queued_duration_s`, and cumulative `rejected_loads`. The accompanying
`rejected_loads_scope` is `buffer` without a robot selector and `robot` for a
robot-scoped query; an env selector does not narrow this request counter. Each
robot and env entry
also reports its queued capacity. Append checks existing plus new staged
playbacks; replace checks only the new sequence. A load is preflighted across
all selected envs and rejected atomically if any depth, sample, or duration
limit would be exceeded. No active trajectory is silently evicted. Each env
entry also contains request/source, stage, elapsed/duration/progress,
full-trajectory `joint_names`, and sparse `joint_track_names`. A normal full
trajectory has an empty `joint_track_names`. `clear_trajectory` likewise
requires `env_ids`; omit only the robot selector to clear every robot for those
envs.

### 9.3 Sparse `hand` Subtrack

```json
{
  "type": "hand",
  "request_id": "close-hand",
  "robot_id": 0,
  "env_ids": [0, 1],
  "duration_s": 0.2,
  "joint_positions": {
    "L6V1_L_hand_index_mcp_pitch": [0.7, 0.8]
  },
  "queue": true,
  "replace": false
}
```

```json
{
  "event": "hand_motion_queued",
  "accepted": true,
  "backend": "isaac",
  "motions": [
    {
      "robot_id": 0,
      "env_ids": [0, 1],
      "duration_s": 0.2,
      "joint_names": ["AR5V2_L_arm_joint_1", "L6V1_L_hand_index_mcp_pitch"],
      "joint_track_count": 1,
      "queued": true
    }
  ],
  "step": 122,
  "time_s": 0.5083333333
}
```

| `hand` field | Default/rule |
|---|---|
| `request_id` | Optional ID stored in playback status |
| `source` | `interactive_hand` |
| `robot_id` | Required in multi-robot scenes; exactly one robot |
| `env_ids` | required; nonempty, unique, and in range |
| `duration_s` | Required and nonnegative; zero is an immediate one-point target |
| `joint_positions` | Required nonempty joint-name mapping; scalar or one value per selected env |
| `queue` | true; append after existing playback |
| `replace` | false; true replaces instead of appending |

Each target is a scalar broadcast or an array of `len(env_ids)`. Every field is
top-level; nothing is inherited from a nested plan/request. `motions[]` reports
the robot, envs, duration, complete command-space joint list, one sparse track,
and whether it queued.

Internally, `PlaybackJointTrack` covers only named command columns. An appended
subtrack reads the current command target when playback actually begins, so an
old arm baseline captured at load time cannot overwrite the previous
trajectory's endpoint. `trajectory_status.joint_track_names` exposes covered
columns, but clients cannot submit internal `PlaybackJointTrack` objects or
overlays.

The convenience interface selects sparse columns by joint name. Clients should
send only joints confirmed as hand joints in status. It performs no cuRobo
planning and is not a strict same-tick arm/hand timeline. For complex synchronized
arm/hand motion, use Single Scene `group_tracks` or load a complete Tiled Scene command-space
trajectory.

## 10. Asynchronous `plan`

Every plan field is top-level and `kind` is mandatory. Submission copies the
selected-env targets at that instant; a worker does not keep reading changing
runtime state.

| Field | Default | Meaning |
|---|---|---|
| `request_id` | generated `plan-<uuid>` | Unique planner lifecycle/cancellation/cache ID |
| `robot_id` | optional only with one robot | Exactly one robot |
| `env_ids` | required | Nonempty unique planning rows |
| `kind` | required | `joint_position_target`, `joint_delta_pos`, or `linear_pose_path` |
| `duration_s` | runtime planner request default (`1.0` in bundled profiles) | Positive logical duration |
| `sample_dt_s` | physics dt | Planner output period |
| `avoid_collisions` | false | Requires complete cuRobo collision capability when true |
| `load_on_success` | true | Automatically load a successful result into playback |
| `replace` | true | Replace selected-env playback during automatic load |
| `source` | `interactive_plan` | Result/playback source string |

Every valid kind receives the same immediate submission shape. It confirms
entry into the planner manager, not GPU completion:

```json
{
  "event": "plan_submitted",
  "accepted": true,
  "backend": "isaac",
  "request_id": "plan-1",
  "robot_id": 0,
  "env_ids": [0, 1],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "segments": ["joint_position_target"],
  "load_on_success": true
}
```

### 10.1 Absolute Joint Goal

```json
{
  "type": "plan",
  "request_id": "plan-1",
  "robot_id": 0,
  "env_ids": [0, 1],
  "kind": "joint_position_target",
  "joint_positions": [[0.2, 0.1], [0.3, 0.1]],
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false,
  "load_on_success": true,
  "replace": true,
  "source": "policy"
}
```

### 10.2 Relative Joint Goal

```json
{
  "type": "plan",
  "request_id": "delta-1",
  "robot_id": 0,
  "env_ids": [0, 1],
  "kind": "joint_delta_pos",
  "joint_deltas": [0.1, 0.0],
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "duration_s": 0.8,
  "sample_dt_s": 0.02
}
```

`(D,)` broadcasts and `(E,D)` supplies one row per env. Without `joint_names`,
the payload writes a command prefix. With names, it scatters by name. Omitted
absolute columns retain current state; omitted delta columns use zero increment.

### 10.3 Linear TCP Path

Relative endpoint:

```json
{
  "type": "plan",
  "request_id": "path-1",
  "robot_id": 0,
  "env_ids": [0],
  "kind": "linear_pose_path",
  "target_offset": [0.0, 0.0, 0.1],
  "orientation_mode": "free",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

Absolute endpoint with orientation:

```json
{
  "type": "plan",
  "request_id": "path-2",
  "robot_id": 0,
  "env_ids": [0],
  "kind": "linear_pose_path",
  "target_position": [0.35, 0.0, 0.4],
  "orientation_mode": "target",
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "sample_dt_s": 0.02
}
```

Exactly one of `target_position` and `target_offset` is required. Orientation
modes are:

- `free`: constrain position only.
- `current`: preserve the starting orientation.
- `target`: Slerp to required wxyz target orientation.

The same explicit-quaternion rule applies here: a quaternion with no mode
implies `target`; explicit `free/current` with a quaternion and `target` without
a quaternion are rejected. Otherwise the runtime command default applies.

Async plan target position/offset accepts only `(3,)`, and orientation accepts
only `(4,)`; one goal applies to every selected env. This canonical async
interface does not define `pose_reference_frame`, and strict field validation
rejects a request that supplies it. Coordinates are interpreted directly in
the cuRobo context's robot-base-local frame. Convert world/env goals before
submission, or split per-env goals into separate requests. This differs from
synchronous `step/ee_linear_path`, whose named targets can broadcast `(E,3)`
rows and perform frame conversion.

Task-space paths currently call the planner facade per env rather than entering
joint batch. One JSON `plan` expresses one segment with fields at the request
top level.

## 11. Planner Lifecycle And Batch Scheduling

### 11.1 Query, Cancel, And Clear

Dispatch queued requests and wait/collect ready results:

```json
{"type":"planner_status","wait_timeout_s":0.1}
```

```json
{
  "event": "planner_status",
  "accepted": true,
  "backend": "isaac",
  "ready": [
    {
      "request_id": "plan-1",
      "robot_id": 0,
      "env_ids": [0, 1],
      "success": true,
      "status": "SUCCESS",
      "message": "",
      "samples": 51,
      "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
      "source": "policy",
      "load_on_success": true
    }
  ],
  "loaded": [
    {"request_id": "plan-1", "robot_id": 0, "env_ids": [0, 1]}
  ],
  "load_rejected": [],
  "planner": {
    "pending": [],
    "pending_count": 0,
    "completed_count": 1,
    "max_pending_requests": 64,
    "max_completed_results": 256,
    "max_batch_problems": 64,
    "oversize_request_policy": "split",
    "max_workers": 2,
    "running_batch_count": 0,
    "rejected_requests": 0,
    "split_requests": 0,
    "evicted_completed_results": 0,
    "shutdown_requested": false,
    "shutdown_timed_out": false,
    "queued_request_ids": [],
    "running_request_ids": [],
    "live_request_ids": [],
    "completed": [
      {
        "request_id": "plan-1",
        "robot_id": 0,
        "env_ids": [0, 1],
        "success": true,
        "status": "SUCCESS",
        "message": "",
        "samples": 51,
        "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
        "source": "policy",
        "load_on_success": true
      }
    ]
  }
}
```

`wait_timeout_s` defaults to zero for nonblocking collection. A positive value
waits at most that long for the first future and does not guarantee all pending
work completes. `ready` contains only results newly collected by this call;
`planner.completed` contains every retained summary. Large `times/positions`
arrays never appear in status; consume a successful trajectory through playback.
`load_rejected` is always an array. It lists successful results with
`load_on_success=true` that playback admission could not accept, one object per
request with `request_id`, `robot_id`, `env_ids`, `code`, and `error`. Capacity
violations use `code="playback_capacity_exceeded"`; other buffer validation
failures use `code="playback_load_rejected"`. One rejected load does not discard
later ready results; `loaded` and `load_rejected` partition this call's
auto-load attempts. Failed planner results and successful results with
`load_on_success=false` appear in neither list.

Cancel one request:

```json
{"type":"cancel_plan","request_id":"plan-1"}
```

```json
{
  "event": "plan_cancelled",
  "accepted": true,
  "backend": "isaac",
  "result": {
    "request_id": "plan-1",
    "accepted": true,
    "status": "cancel_requested",
    "future_cancelled": false
  }
}
```

Cancel the intersection selected by robot/env:

```json
{"type":"cancel_plan","robot_id":0,"env_ids":[0,1]}
```

This form returns an array of per-request results. Without `request_id`,
`env_ids` is required; omit `robot_id` to match every robot on those envs.
Omitting all three selectors is rejected. Queued work becomes `cancelled`
immediately. A running GPU future normally receives only `cancel_requested` and
produces a `CANCELLED` result when later collected.

Clear one, several, or every completed summary:

```jsonl
{"type":"clear_completed","request_id":"plan-1"}
{"type":"clear_completed","request_ids":["plan-1","plan-2"]}
{"type":"clear_completed"}
```

```json
{
  "event": "completed_cleared",
  "accepted": true,
  "backend": "isaac",
  "result": {
    "cleared": ["plan-1"],
    "missing": ["plan-2"],
    "count": 1
  }
}
```

Lifecycle rules:

- `plan` only queues the request and returns `plan_submitted`; it neither starts
  backend work nor waits for the GPU.
- `planner_status` dispatches queued requests, then waits up to
  `wait_timeout_s` and collects.
- `step_trajectory` also dispatches and collects once before its tick loop.
- Successful results with `load_on_success=true` enter playback automatically.
- Queued work cancels immediately; running GPU work usually cannot be forcibly
  interrupted and becomes `CANCELLED` when it returns.
- `cancel_plan` without request ID filters by required env and optional robot;
  omitting all selectors is rejected.
- `clear_completed` accepts `request_id` or `request_ids`; no IDs clears all.

`planner_status.planner` includes pending/completed counts, capacity, oversize
policy, cumulative rejection/split/eviction counters, shutdown state,
live/queued/running IDs, individual queued/running state, and completed summaries
without trajectory matrices. With `max_completed_results=0`, a newly ready result
still appears in the current response but is not retained.

### 11.2 Two-Level Batch Scheduling

```text
public request_id queue
  -> manager groups consecutive homogeneous requests in FIFO order
  -> one future per group, limited by planner-workers
  -> TiledCuroboPlanningBackend.plan_many merges request rows
  -> CuroboBatchJointProblem (no env/request fields)
  -> CuroboBatchJointPlanner -> cuRobo BatchMotionPlanner
  -> split row slices back into public request_id results
```

The manager batch key includes:

- Internal robot label corresponding to one public `robot_id`.
- Complete command joint names.
- `duration_s` and `sample_dt_s`.
- `avoid_collisions`.
- Joint-space segment kind and structure.

Only consecutive requests with identical keys merge. An incompatible request
ends the group; FIFO order is not rearranged around it. A group normally holds
at most 64 problem rows, counted as the sum of `len(env_ids)`, not request count.
When consecutive public requests would exceed the configured limit, the manager
ends the current group and dispatches another bounded future; those public
requests are not themselves split. For one oversized request,
`oversize_request_policy=split`
slices every env-shaped request/segment row into bounded backend invocations and
merges only fully successful, structurally consistent chunks back into the
original request ID. Any chunk failure fails the whole request, so no partial
trajectory is loaded. `oversize_request_policy=reject` rejects it before it
enters the pending queue. No backend invocation exceeds `max_batch_problems`.

The backend vstacks rows, creates one exclusive planner/context for the group,
and constructs a `CuroboBatchJointProblem` with no env/request metadata. When
real rows are fewer than cuRobo's fixed `BatchMotionPlanner.batch_size`, it pads
with the final row and hides padding results. Runtime profile validation ensures
explicit `max_batch_problems` does not exceed the selected cuRobo profile
capacity; manager-side splitting happens before backend entry.

`--planner-workers > 1` permits multiple batch futures, but every worker creates
an independent cuRobo context, CUDA graph/cache, and tensors. More workers can
increase throughput and GPU memory substantially; measure with one or two first.

## 12. End-To-End Workflows

The executable [Tiled Scene Quickstart](../getting-started/tiled-scene-quickstart.md) owns
the discovery, synchronous operation, result check, and shutdown workflow.
After that succeeds, use [Control And Trajectories](../guides/control-and-trajectories.md)
and [Motion Planning](../guides/motion-planning.md) for task-oriented trajectory
and asynchronous planning workflows, then return here for exact message fields.

The public response boundary converts internal robot-keyed mappings into
`robots[]`, `info[]`, or `trajectory[]`. Locate rows by `robot_id`; do not assume
an array contains only one robot.

## 13. Common Failures And Diagnostics

| Status/error | Cause and response |
|---|---|
| `rejected` selector range | Rediscover status and validate env/robot IDs |
| `values first dimension` | First dimension must be 1 or `len(env_ids)` |
| `robots is required` | Select robots explicitly for multi-robot step/playback |
| `COLLISION_UNSUPPORTED` | Missing spheres, checker, cache, or scene synchronization |
| `BATCH_UNAVAILABLE` | `batch_only` request cannot use the batch API |
| `BATCH_TOO_SMALL` | Merged rows exceed cuRobo batch size; reduce public request size or increase profile batch |
| `FAILED env rows [...]` | At least one real row failed; inspect target, seed, collision, and tolerances |
| `too many pending` | Collect/cancel work or raise the pending limit deliberately |
| `duplicate request_id` | ID remains pending or cached; choose a new ID or clear completed |
| Stale result cancellation | reset/set_state/set_snapshot overlapped the request's robot/env rows |
| GUI/Foxglove does not refresh | Use `--idle-physics-policy hold_step`; also use `--stdin-eof-policy keep_alive` for a long-running service |

## 14. Current Boundaries

- Batched IK does not fall back to a per-env IK loop.
- Synchronous `ee_linear_path` batches every IK waypoint across envs and
  resamples to full-batch physics targets; async `linear_pose_path` remains
  per-env sequential planning.
- One public request larger than `max_batch_problems` follows
  `oversize_request_policy`: the bundled `split` policy chunks env rows into
  bounded backend calls, while `reject` refuses it before queueing.
- With `multi_env=false`, one batch problem shares one collision world; differing
  env obstacles are not independently collision-aware.
- `avoid_collisions=true` never silently degrades.
- Trajectory payloads cannot submit partial-joint track objects; `hand` is the
  only sparse-subtrack convenience interface.
- Execution uses integer physics ticks. Nonintegral `duration_s/sample_dt_s`
  rounds IK count up, and nonintegral `duration_s/physics_dt` can extend actual
  execution by less than one physics tick.
- `get_state/set_state` is a runtime debugging shape. Use snapshots for
  persistent restore.
