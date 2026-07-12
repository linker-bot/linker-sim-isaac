# Single Scene Runtime And JSON Protocol

Language: [English](single-scene-json.md) | [中文](../../zh-CN/reference/single-scene-json.md)

This page owns the JSON transport, message, selector, lifecycle, and response
contracts for the regular, non-tiled `SingleSceneRuntime`. For checkout setup and a
complete first request, use the [Single Scene Quickstart](../getting-started/single-scene-quickstart.md).
For every launch option, effective default, and process marker, use the
[Single Scene CLI Reference](single-scene-cli.md). For cloned parallel environments, see the
[Tiled Scene JSON Reference](tiled-scene-json.md).

## 1. Runtime Boundary

`SingleSceneRuntime` owns one physical World with every robot and object declared by
the selected env profile. It has no cloned `env_id` dimension, and Single Scene does
not mean single robot: discovery may return any configured robot count. Launch,
EULA, effective configuration, endpoint activation, and process-marker details
are owned by the [Single Scene CLI Reference](single-scene-cli.md).

## 2. Transports And Concurrency Boundary

All three transports use `parse_interactive_motion_message()` and one shared
`InteractiveMotionQueue`:

| Transport | Framing | Immediate response | Later state changes |
|---|---|---|---|
| stdin | One JSON object per line | One JSON line on stdout | Poll with `status` |
| TCP JSONL | One JSON object per line | One response line on the same connection | Poll with `status` |
| WebSocket | One JSON object per text message | JSON text on the same connection | Also receives `running`, `done`, `failed`, and `cancelled` pushes |

All Isaac, USD, and PhysX access occurs on the simulation main thread. TCP and
WebSocket threads only parse JSON, enqueue work, and wait for responses. They
must not access `World`, the stage, or articulations directly. Snapshot requests
also enter the main-thread queue. The runtime
`interactive.snapshot_timeout_s` (30 seconds in the bundled profile) bounds only
the wait for the main thread to begin a request. A pre-execution timeout cancels
the request so it never touches runtime state; after execution begins, the
transport waits for its real success or failure result during normal operation.
Shutdown can interrupt that wait with the explicit `snapshot_running` response
described in Section 9.3.

Transport resources are bounded by the selected runtime profile. TCP and
WebSocket share one process-wide `max_connections` admission limit; stdin does
not consume it. `max_message_bytes` applies before JSON dispatch, request and
snapshot queues are bounded, and each WebSocket has a bounded event queue.
Oversized, non-UTF-8, duplicate-key, `NaN`, and infinite JSON input is rejected
without reaching Isaac. WebSocket event overflow follows the configured
reject-new transport policy and increments diagnostics; direct request responses
remain separate.

### 2.1 stdin Example

```bash
export OMNI_KIT_ACCEPT_EULA=Y
printf '%s\n' \
  '{"type":"status"}' \
  '{"type":"quit"}' \
| PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
    --env scene1
```

This pre-buffered pipe demonstrates stdin framing and orderly shutdown only.
Use interactive stdin, TCP, or WebSocket when a motion client must wait for a
command's terminal state before sending `quit`.

### 2.2 TCP JSONL Example

After starting the service, keep one connection open:

```bash
nc 127.0.0.1 8765
```

Send one request per line:

```jsonl
{"type":"status"}
{"type":"status","id":"move-1"}
```

TCP returns exactly one direct response per request. Motion completion events
are not inserted asynchronously into the connection; poll `status`.

### 2.3 WebSocket Example

```python
import asyncio
import json

import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:8766") as ws:
        await ws.send(json.dumps({"type": "status"}))
        print(json.loads(await ws.recv()))

        await ws.send(json.dumps({
            "type": "joint_delta",
            "id": "move-1",
            "robot_id": 0,
            "group": "arm",
            "joint_deltas": {"AR5V2_L_arm_joint_1": 0.05},
            "duration_s": 0.5,
        }))
        while True:
            event = json.loads(await ws.recv())
            print(event)
            if event.get("id") == "move-1" and event.get("state") in {
                "done", "failed", "cancelled"
            }:
                break


asyncio.run(main())
```

An `accepted` event can appear both as the direct WebSocket response and as a
queue broadcast. Make handling idempotent by keying on `event + id + state`,
not on message count.

## 3. First Request: Discover Session Identities

Start every new process connection with:

```json
{"type":"status"}
```

Representative response:

```json
{
  "event": "status",
  "commands": [],
  "current_id": null,
  "estop": false,
  "resetting": false,
  "last_reset": null,
  "config_fingerprint": "...",
  "robots": [
    {
      "robot_id": 0,
      "label": "ar5v2_l6v1_0",
      "robot_profile": "ar5v2_l6v1_l",
      "profile_fingerprint": "...",
      "kind": "arm_hand",
      "supports_planning": true,
      "supports_collision_aware_planning": false,
      "planning_joint_group": "arm",
      "joint_groups": {
        "arm": ["AR5V2_L_arm_joint_1"],
        "hand": ["L6V1_L_hand_index_mcp_pitch"],
        "passive": []
      }
    }
  ],
  "collision": {},
  "planning": {}
}
```

| Field | Contract |
|---|---|
| `robot_id` | Dense integer generated from the current env `robots[]` order |
| `label` | Stable configuration identity used for logs and snapshot matching |
| `robot_profile` | Profile name under `configs/robots/` |
| `profile_fingerprint` | Fingerprint of the loaded profile content |
| `kind` | `arm`, `hand`, or `arm_hand` |
| `supports_planning` | The robot has a valid cuRobo model and joint binding, so a cuRobo context can be created |
| `supports_collision_aware_planning` | The materialized context has collision capability for the current scene version |
| `planning_joint_group` | Currently `arm` or `null` |
| `joint_groups` | Explicit articulation order for arm, hand, and passive joints |

`robot_id` is not persistent. Rediscover it after process restart, env reorder,
or robot additions/removals. A request may carry `robot_label` as an identity
assertion, but cannot select a robot by label, side, role, or name alone.

`supports_planning=false` does not prevent the `linear` backend from executing
explicit arm joint interpolation. Task-space planning, collision checking, and
cuRobo C-space planning still depend on the capability fields above.

## 4. Command Index And State Machine

<!-- scene-message-index:start -->
| `type` | Purpose | Primary response |
|---|---|---|
| `plan_timeline` | Full multi-robot timeline | `accepted`, then lifecycle events |
| `hold` | One robot/group hold shorthand | `accepted` |
| `joint_goal` | Direct absolute joint target | `accepted` |
| `joint_delta` | Direct relative joint target | `accepted` |
| `joint_trajectory` | Direct sampled joint trajectory | `accepted` |
| `plan_cspace_goal` | Absolute C-space target through the selected planner | `accepted` |
| `plan_cspace_delta` | Relative C-space target through the selected planner | `accepted` |
| `ik_pose` | cuRobo TCP pose-goal trajectory planning | `accepted` |
| `ik_offset` | cuRobo relative TCP translation trajectory planning | `accepted` |
| `plan_linear_pose_path` | Sequential-IK TCP line path | `accepted` |
| `status` | Query all commands or one command | `status` |
| `cancel` | Cancel a pending/running command by ID | `cancel` |
| `cancel_current` | Interrupt the running command | `cancel_current` |
| `reset` | Main-thread-safe runtime reset | `reset`, then `reset_done` or `reset_failed` |
| `get_snapshot` | Read a runtime-neutral snapshot | `snapshot` |
| `set_snapshot` | Restore a snapshot | `snapshot_restored` or `snapshot_failed` |
| `estop` | Cancel the queue and stop the interaction loop | `estop` |
| `quit` | Normal shutdown | `quit` |
<!-- scene-message-index:end -->

### 4.1 Common Motion Responses

All shorthand segments and `plan_timeline` enter the same queue. The runtime
generates `move-<n>` when `id` is omitted, but production clients should supply
a unique ID within the session. A successful submission immediately returns:

```json
{
  "event": "accepted",
  "id": "timeline-1",
  "state": "pending",
  "queue_index": 0
}
```

`queue_index` is the zero-based position in the current pending subqueue, not a
global command sequence. WebSocket later broadcasts transitions; TCP and stdin
observe the same state through `status`:

```jsonl
{"event":"running","id":"timeline-1","state":"running"}
{"event":"done","id":"timeline-1","state":"done","steps":480}
{"event":"failed","id":"timeline-1","state":"failed","error":"planning failed: ..."}
{"event":"cancelled","id":"timeline-1","state":"cancelled","error":"interrupted","steps":231}
```

| Response field | Meaning |
|---|---|
| `event` | Current event; terminal values are `done`, `failed`, and `cancelled` |
| `id` | Request ID or generated runtime ID |
| `state` | `pending`, `running`, `done`, `failed`, or `cancelled` |
| `queue_index` | Present only in `accepted` |
| `error` | Failure reason; null or omitted on success |
| `steps` | Global simulation step at termination, not the command's own tick count |

JSON, field, or queue-submission failures do not create a command:

```json
{"event":"rejected","error":"robot_id is required"}
```

Timeline compilation is atomic before execution. If any robot, unit, group, or
segment fails planning, the complete command enters `failed`; already compiled
tracks do not start early.

### 4.2 `status` Request And Response

Without `id`, `status` returns every command and the complete runtime discovery
payload from section 3. With `id`, `commands` contains at most one row:

```jsonl
{"type":"status"}
{"type":"status","id":"timeline-1"}
```

Targeted response:

```json
{
  "event": "status",
  "commands": [
    {
      "id": "timeline-1",
      "state": "running",
      "error": null,
      "steps": null,
      "command_kind": "timeline"
    }
  ],
  "config_fingerprint": "...",
  "robots": [
    {"robot_id": 0, "label": "ar5v2_l6v1_0"}
  ]
}
```

An unknown ID returns an empty `commands` array, not `rejected`. In the full
response, `current_id`, `estop`, `resetting`, and `last_reset` describe the
running command, emergency-stop state, reset activity, and latest reset result.

## 5. Single-Segment Shorthand

A single-segment command requires `robot_id` and is normalized into a
single-track timeline. Shared fields are:

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | One canonical segment type below |
| `id` | recommended | Queue command ID |
| `robot_id` | yes | Robot ID in this session |
| `robot_label` | no | Assertion that the ID maps to this label |
| `group` | no | `arm` or `hand`; default `arm` |
| `duration_s` | yes | Finite nonnegative seconds, converted to integer ticks |
| `sample_dt_s` | no | Planning segments only; planner sample period, default physics dt |
| `coordination` | no | `independent` or `static_others`; default `independent` |
| `force_collision_refresh` | no | Force a collision-view rebuild for the current context |
| `phase` | no | Trajectory/telemetry phase name; defaults to the kind |
| `metadata` | no | JSON object passed to the compiled trajectory |

Type-specific canonical fields:

| `type` | Required payload | Optional payload and constraints |
|---|---|---|
| `hold` | none | No joint or task-space target |
| `joint_goal` | `joint_positions` | Complete ordered vector or joint-name-to-scalar mapping |
| `joint_delta` | `joint_deltas` | Complete ordered vector or joint-name-to-scalar mapping |
| `joint_trajectory` | `joint_positions` | `(T,D)` or joint-name-to-T-samples mapping; optional `times_s` |
| `plan_cspace_goal` | `joint_positions` | `sample_dt_s`, `avoid_collisions`, `force_collision_refresh` |
| `plan_cspace_delta` | `joint_deltas` | `sample_dt_s`, `avoid_collisions`, `force_collision_refresh` |
| `ik_pose` | `target_position`, `reference_frame` | `target_orientation_quat_wxyz`, TCP, sampling, collision fields |
| `ik_offset` | `offset`, `offset_frame` | `target_orientation_quat_wxyz`, TCP, sampling, collision fields |
| `plan_linear_pose_path` | exactly one of absolute position+frame or offset+frame | Optional target orientation, TCP, sampling, and collision fields |

`sample_dt_s` is accepted only by planning types. Do not mix
`joint_positions` and `joint_deltas`. Single Scene relative translation always uses
`offset`. Every successful example below receives the `accepted` response from
section 4.1.

### 5.1 Hold

```json
{
  "type": "hold",
  "id": "hold-arm",
  "robot_id": 0,
  "group": "arm",
  "duration_s": 0.2
}
```

`hold` does not call a planner or change the group target. It holds the command
captured at compilation for the requested duration. A zero-duration hold is a
valid no-op and can mark a timing boundary inside a unit.

### 5.2 Absolute Joint Target

A mapping changes only named joints; all other group joints keep their current
values:

```json
{
  "type": "joint_goal",
  "id": "close-finger",
  "robot_id": 0,
  "group": "hand",
  "joint_positions": {
    "L6V1_L_hand_index_mcp_pitch": 0.7
  },
  "duration_s": 0.8
}
```

Alternatively, provide one complete vector ordered according to
`status.robots[].joint_groups[group]`.

### 5.3 Relative Joint Target

```json
{
  "type": "joint_delta",
  "id": "arm-nudge",
  "robot_id": 0,
  "group": "arm",
  "joint_deltas": {
    "AR5V2_L_arm_joint_2": 0.2
  },
  "duration_s": 0.4
}
```

The delta is applied to the current group target when this segment is compiled.

### 5.4 Sampled Joint Trajectory

```json
{
  "type": "joint_trajectory",
  "id": "hand-curve",
  "robot_id": 0,
  "group": "hand",
  "joint_positions": {
    "L6V1_L_hand_index_mcp_pitch": [0.2, 0.5, 0.7]
  },
  "times_s": [0.1, 0.3, 0.6],
  "duration_s": 0.6
}
```

`joint_positions` can also be a `(samples, group_dim)` matrix. Every mapping
column must have the same sample count. When present, `times_s` must have that
length, contain positive values, and be strictly increasing. Without
`times_s`, samples are distributed uniformly over `duration_s`; when
`duration_s=0`, each sample receives one physics-dt interval.

### 5.5 Planned C-Space Targets

These commands use the backend selected by `--planner-backend`:

```json
{
  "type": "plan_cspace_goal",
  "id": "arm-plan-goal",
  "robot_id": 0,
  "group": "arm",
  "joint_positions": {
    "AR5V2_L_arm_joint_1": 0.2
  },
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

`curobo` uses the robot planning model. `linear` generates direct interpolation
from the current joints to the target; it does not validate joint limits, and
explicitly fails when `avoid_collisions=true`.

Complete ordered arm examples for two independent robots:

```jsonl
{"type":"plan_cspace_goal","id":"arm-plan-goal0","robot_id":0,"group":"arm","joint_positions":[1.64,1.2,-1.5707,1.57,-0.37,0.0,0.0],"duration_s":1.0,"avoid_collisions":false}
{"type":"plan_cspace_goal","id":"arm-plan-goal1","robot_id":1,"group":"arm","joint_positions":[1.2,-1.2,-1.5707,1.57,0.37,0.0,0.0],"duration_s":1.0,"avoid_collisions":false}
```

These are two JSONL commands, not one JSON array. Use one multi-robot timeline
when both robots must begin on the same tick.

The relative form changes the type and target field:

```json
{
  "type": "plan_cspace_delta",
  "id": "arm-plan-delta",
  "robot_id": 0,
  "group": "arm",
  "joint_deltas": {"AR5V2_L_arm_joint_1": -0.1},
  "duration_s": 1.0,
  "avoid_collisions": true
}
```

### 5.6 Absolute TCP Pose Planning

```json
{
  "type": "ik_pose",
  "id": "ik-absolute",
  "robot_id": 0,
  "group": "arm",
  "target_position": [0.35, 0.0, 0.40],
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "reference_frame": "world",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "avoid_collisions": false
}
```

The orientation is optional; omitting it constrains TCP position only.

### 5.7 Relative TCP Translation Planning

```json
{
  "type": "ik_offset",
  "id": "ik-up",
  "robot_id": 0,
  "group": "arm",
  "offset": [0.0, 0.0, 0.05],
  "offset_frame": "robot_base",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 0.8,
  "avoid_collisions": false
}
```

`ik_offset` changes position while preserving the starting TCP orientation.

The protocol names `ik_pose` and `ik_offset` do not mean the runtime returns one
standalone IK solution. `TimelinePlanningSession` converts the goal into robot
base coordinates and constructs `MotionRequest(goal_pose=...)`.
`CuroboMotionPlanner.plan()` then calls cuRobo `MotionPlanner.plan_pose()` to
produce an executable trajectory. Direct `CuroboInverseKinematics.solve()` is a
Python-facade operation, not the execution path for these Single Scene commands.

### 5.8 TCP Linear Path

Relative endpoint:

```json
{
  "type": "plan_linear_pose_path",
  "id": "tcp-line-up",
  "robot_id": 0,
  "group": "arm",
  "offset": [0.0, 0.0, 0.10],
  "offset_frame": "robot_base",
  "duration_s": 1.0,
  "avoid_collisions": false
}
```

Absolute endpoint:

```json
{
  "type": "plan_linear_pose_path",
  "id": "tcp-line-target",
  "robot_id": 0,
  "group": "arm",
  "target_position": [0.35, 0.0, 0.40],
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "reference_frame": "world",
  "duration_s": 1.2
}
```

Exactly one of `target_position` and `offset` is required. Without a target
orientation the path constrains position only. With one, orientation is Slerped
from the current pose. The backend samples TCP waypoints and warm-starts each IK
solve from the previous solution.

## 6. Complete Multi-Robot Timeline

The hierarchy is fixed:

```text
plan_timeline
  robot track                 different robots run from tick 0 in parallel
    motion unit               units for one robot run sequentially
      arm/hand group track    tracks in one unit share a start and run in parallel
        segment               segments in one group run sequentially
```

Top-level fields:

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | Must be `plan_timeline` |
| `id` | recommended | Queue ID for the whole timeline |
| `tracks` | yes | Nonempty robot-track array; each `robot_id` appears once |
| `coordination` | no | `independent` or `static_others`; `coupled` is rejected |
| `force_collision_refresh` | no | Force collision-view synchronization before planning |

A track has exactly one of two canonical shapes:

| Shape | Fields | Meaning |
|---|---|---|
| Single-group shorthand | `robot_id`, optional `robot_label/group`, `segments[]` | One serial group; `group` defaults to `arm` |
| Full units | `robot_id`, optional `robot_label`, `units[]` | Serial units, each allowing parallel arm/hand tracks |

Full unit structure:

```json
{
  "robot_id": 0,
  "robot_label": "ar5v2_l6v1_0",
  "units": [
    {
      "group_tracks": [
        {
          "group": "arm",
          "segments": [
            {"kind": "hold", "duration_s": 0.2}
          ]
        }
      ]
    }
  ]
}
```

Segments inside a timeline use `kind`; their fields are identical to the
single-segment types in section 5. For example, top-level
`{"type":"ik_pose",...}` becomes `{"kind":"ik_pose",...}` inside
`segments[]`. `group_tracks` and `segments` must be nonempty, and one unit
cannot contain two writers for the same group.

Complete example: robot 0 moves its arm while closing a finger, and robot 1
holds:

```json
{
  "type": "plan_timeline",
  "id": "coordinated-1",
  "coordination": "static_others",
  "force_collision_refresh": true,
  "tracks": [
    {
      "robot_id": 0,
      "robot_label": "ar5v2_l6v1_0",
      "units": [
        {
          "group_tracks": [
            {
              "group": "arm",
              "segments": [
                {
                  "kind": "ik_pose",
                  "target_position": [0.35, 0.0, 0.4],
                  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                  "reference_frame": "world",
                  "duration_s": 1.0,
                  "avoid_collisions": true
                }
              ]
            },
            {
              "group": "hand",
              "segments": [
                {"kind": "hold", "duration_s": 0.2},
                {
                  "kind": "joint_goal",
                  "joint_positions": {
                    "L6V1_L_hand_index_mcp_pitch": 0.7
                  },
                  "duration_s": 0.8
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "robot_id": 1,
      "group": "arm",
      "segments": [
        {"kind": "hold", "duration_s": 1.0}
      ]
    }
  ]
}
```

Immediate response:

```json
{
  "event": "accepted",
  "id": "coordinated-1",
  "state": "pending",
  "queue_index": 0
}
```

Every robot track provides exactly one of `units` and `segments`. A combined
`arm_hand` direct joint target supplied in complete command-space can be split
by explicit joint names into arm and hand tracks in one unit. Planning segments
can target only `arm`.

Single Scene and Tiled Scene planners consume shared canonical request/result contracts.
`curobo` supports joint space, IK, and TCP lines. `linear` supports
`plan_cspace_goal` and `plan_cspace_delta` as executable interpolation but does
not perform IK, collision avoidance, joint-limit validation, or constrained
optimization. `sample_dt_s` controls the planning grid; Single Scene execution always
resamples the result to the physics grid.

### 6.1 Integer-Tick Timing

- `duration_s` is rounded up to a number of physics-dt ticks.
- `sample_dt_s` does not change physics dt and defaults to physics dt.
- A nontrivial positive-duration motion occupies at least one tick.
- A zero-duration hold can be a no-op.
- Segments in one group are sequential; each starts from the previous endpoint.
- A unit ends with its longest group track; shorter groups hold their endpoints.
- The global timeline ends with its longest robot track; shorter robots hold.
- Each tick computes and applies all robot targets before one `world.step()`.
- The executor consumes compiled integer ticks and does not rederive progress
  from floating-point seconds.

## 7. Frames, Quaternions, And TCP Selection

Task-space requests do not infer a frame:

| Field | Values | Purpose |
|---|---|---|
| `reference_frame` | `world`, `env`, `robot_base`, `tcp` | Absolute target position and orientation |
| `offset_frame` | `world`, `env`, `robot_base`, `tcp` | Relative translation |
| `tcp_frame_name` | Frame registered in the current cuRobo model | Constrained TCP; omit for the robot default |

The only public orientation field is
`target_orientation_quat_wxyz`, ordered `[w,x,y,z]`. World and env targets are
converted to robot-base-local coordinates using the env origin and robot root
pose; position and orientation use one rigid transform. A `tcp` offset uses the
current TCP axes.

TCP names come from the materialized robot profile's
`default_tcp_frame/tool_frames/custom_tcps`. JSON cannot declare an unknown
cuRobo frame at request time.

## 8. Collision Planning And Multi-Robot Coordination

| `coordination` | Behavior |
|---|---|
| `independent` | Plan each robot independently without treating other tracks as coordinated motion |
| `static_others` | Freeze other robots' current geometry as static obstacles while planning one robot |
| `coupled` | Rejected because no dynamic coupled backend is implemented |

`avoid_collisions=true` requires collision spheres in the robot model, a scene
collision checker, adequate collision-cache capacity, and a context synchronized
to the current scene version. Missing capability produces an explicit error; it
does not silently fall back to collision-free-disabled planning.

`force_collision_refresh=true` is valid at timeline or segment level. Use it
after object movement, snapshot restore, or when diagnosing a stale collision
view. Forcing every request to refresh increases planning latency.

## 9. Control, Reset, And Snapshots

### 9.1 Cancellation And Exit

Cancel by command ID:

```json
{"type":"cancel","id":"timeline-1"}
```

```json
{"event":"cancel","id":"timeline-1","accepted":true}
```

`accepted=true` means a cancellable pending or running command was found.
Pending commands immediately become `cancelled`; running commands stop at a
later tick boundary and emit a terminal event. Unknown or terminal IDs return
`accepted=false`.

Interrupt the current running command without knowing its ID:

```json
{"type":"cancel_current"}
```

```json
{"event":"cancel_current","accepted":true}
```

With no running command, `accepted=false`. Other pending commands remain.

Emergency-stop and terminate the interaction loop:

```json
{"type":"estop"}
```

```json
{"event":"estop","accepted":true}
```

`estop` cancels all pending commands, requests prompt interruption of the
running command, and ends the loop. There is no resume command; restart the
process. WebSocket can additionally receive
`{"event":"estop","state":"cancelled"}` from the queue.

Normal exit:

```json
{"type":"quit"}
```

```json
{"event":"quit","accepted":true}
```

`quit` wakes the main loop and closes transports normally. It is not a
recoverable pause and does not promise to finish the running trajectory.

Shutdown is dependency ordered and bounded. The interaction loop first wakes
the command queue, interrupts the stdin reader, and stops TCP/WebSocket work,
then drains the state publisher. `SingleSceneRuntime.close()` closes retained async
resources, planning contexts, camera output, and CSV loggers before the
SimulationApp. `runtime.shutdown.transport_timeout_s`,
`state_publisher_timeout_s`, and `camera_publisher_timeout_s` control those
independent waits. A timed-out resource remains owned for a later close retry;
the SimulationApp is not closed underneath a live child resource. Treat any
`*_SHUTDOWN_TIMEOUT` line as an incomplete shutdown, even if the outer command
subsequently prints its final step count.

### 9.2 Reset

<!-- scene-reset-request:start -->
```json
{
  "type": "reset",
  "id": "reset-1",
  "clear_queue": true,
  "hold_after_reset": true
}
```
<!-- scene-reset-request:end -->

The direct response confirms that reset entered the main-thread queue:

<!-- scene-reset-response:start -->
```json
{
  "event": "reset",
  "accepted": true,
  "id": "reset-1",
  "clear_queue": true,
  "hold_after_reset": true
}
```
<!-- scene-reset-response:end -->

`clear_queue=true` cancels pending commands; a running command is always
interrupted. `hold_after_reset=true` performs a short hold so drive targets and
physics state settle. WebSocket receives terminal events; TCP and stdin poll
`status.last_reset`:

```jsonl
{"event":"reset_done","id":"reset-1","state":"done","step":120}
{"event":"reset_failed","id":"reset-1","state":"failed","error":"..."}
```

With `clear_queue=false`, pending work survives and continues after reset. Both
booleans default to true. Omitting `id` generates `reset-<n>`.

### 9.3 Snapshot Read And Restore

The canonical payload, identity matching, shapes, units, restore result, and
transaction rules are owned by the [Snapshot Data And Restore Reference](snapshots.md).
This section defines only the Single Scene message envelope and admission behavior.

The runtime reads the whole current scene and has no env selector:

```json
{"type":"get_snapshot","id":"snapshot-1"}
```

```json
{
  "event": "snapshot",
  "accepted": true,
  "backend": "isaac",
  "id": "snapshot-1",
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

The abbreviated body above shows the envelope only. A real response contains
the complete captured state. Pass that complete `snapshot` back to restore:

```json
{
  "type": "set_snapshot",
  "id": "restore-1",
  "strict": true,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

`set_snapshot` accepts only `type`, optional `id`, `snapshot`, optional
`label_map`, and optional `strict`; `strict` defaults to true. The success
envelope is:

```json
{
  "event": "snapshot_restored",
  "accepted": true,
  "backend": "isaac",
  "id": "restore-1",
  "robots": ["robot_0"],
  "objects": [],
  "env_ids": [],
  "partial": false
}
```

The configured snapshot timeout applies only before the main thread atomically
marks a request executing. Snapshot validation or write failures after that
point return `snapshot_failed` with `error`; they are never reported as timeouts
while continuing invisibly in the background. A pre-execution wait timeout
returns:

```json
{"event":"snapshot_timeout","accepted":false,"id":"snapshot-1"}
```

Shutdown is the only exception to waiting for a terminal result after execution
begins. When shutdown wins the race, a request that has not started is cancelled;
an executing request cannot honestly be reported as cancelled, so its waiter
returns immediately with `snapshot_running`:

```jsonl
{"event":"snapshot_cancelled","accepted":false,"reason":"shutdown","id":"snapshot-1"}
{"event":"snapshot_running","accepted":true,"state":"running","id":"restore-1"}
```

`snapshot_running` is not a success terminal state. It only says the runtime
accepted and began the operation before shutdown stopped the synchronous wait.
Clients must not treat it as `snapshot_restored` or automatically replay the
same mutation.

## 10. End-To-End Workflow

The executable [Single Scene Quickstart](../getting-started/single-scene-quickstart.md) owns
the complete discovery, submission, terminal polling, shutdown, and marker
verification workflow. Return here for exact protocol fields after that
workflow succeeds.

## 11. Common Rejection Causes

- Missing, out-of-range, or cross-session cached `robot_id`.
- `robot_label` does not match the current ID.
- A robot track provides both or neither of `units` and `segments`.
- Unknown group, target joints outside the group, or duplicate group writer.
- Task-space request lacks an explicit frame, uses a non-wxyz quaternion, or
  selects a TCP absent from the model.
- A hand-only robot or hand group requests cuRobo planning.
- `avoid_collisions=true` without complete collision capability.
- Duplicate command ID, or submission after estop/quit.
