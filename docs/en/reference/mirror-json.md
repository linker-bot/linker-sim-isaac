# Mirror JSON Protocol

Language: [English](mirror-json.md) | [中文](../../zh-CN/reference/mirror-json.md)

Mirror exposes three strict process protocols: the frozen `linkerbot.mirror.v1`, its
backward-compatible `linkerbot.mirror.v2` superset, and the hybrid-control
`linkerbot.mirror.v3` superset.
It is served over stdin JSONL, loopback TCP JSONL, and loopback WebSocket text frames.
All transports share one bounded admission queue and one Isaac owner-thread consumer.

## Request Envelope

Every request contains exactly four fields:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "client-unique-id",
  "operation": "runtime.status",
  "arguments": {}
}
```

| Field | Contract |
| --- | --- |
| `protocol` | Must equal `linkerbot.mirror.v1`, `linkerbot.mirror.v2`, or `linkerbot.mirror.v3`. Responses echo the accepted request version. |
| `request_id` | Nonempty string, no leading or trailing whitespace. It must be unique while present in pending, active, or retained terminal history. |
| `operation` | One of the 20 frozen v1, 23 v2, or 27 v3 operations below. |
| `arguments` | JSON object. Unknown fields are rejected by the selected operation. |

Duplicate JSON object keys, non-UTF-8 bytes, `NaN`, and infinities are rejected.
Fields are never coerced from strings or numbers into booleans, integers, or vectors.

## Response Envelope

Success:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "status-1",
  "ok": true,
  "result": {}
}
```

Failure:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "status-1",
  "ok": false,
  "error": {
    "code": "invalid_arguments",
    "message": "runtime.status contains unknown arguments"
  }
}
```

An error may include a JSON `details` object. Clients should branch on `ok`, then on
`error.code`; human-readable messages are not stable identifiers.

## Operation Inventory

| Operation | Arguments | Purpose |
| --- | --- | --- |
| `runtime.status` | none | Physics, scene, collision, shutdown, and queue status. |
| `runtime.reset` | optional `clear_queue`, `hold_after_reset` booleans | Transactionally restore the configured initial state and clear emergency stop. |
| `state.get` | none | Read an owned JSON representation of the current logical scene state. |
| `state.set` | required `state`; optional `strict` boolean | Transactionally write logical scene state through the Mirror state service. |
| `snapshot.get` | none | Capture the versioned scene snapshot. |
| `snapshot.set` | required `snapshot`; optional `label_map`, `strict` | Restore a snapshot. |
| `queue.cancel` | required `request_id` | Cancel a pending request or mark the matching active request for cooperative cancellation. |
| `queue.cancel_current` | none | Mark the active request for cooperative cancellation. |
| `runtime.estop` | none | Cancel queued work, mark active work for cancellation, and freeze idle physics. |
| `runtime.quit` | none | Request orderly event-loop shutdown. |
| `control.get_mode` | none; v2 only | Read the initial/active mode, generation, supported modes, and all-robot scope. |
| `control.set_mode` | required `mode`; optional `expected_generation`; v2 only | Transactionally switch every robot between complete motions. |
| `control.get_hybrid_parameters` | none; v3 only | Read the current hybrid gains, tuning limits, and generation. |
| `control.set_hybrid_parameters` | one or more gain fields; optional `expected_generation`; v3 only | Atomically update gains between motions. |
| `control.tare_wrench` | robot/TCP identity and `reference_frame`; v3 only | Measure and register a wrench tare generation. |
| `motion.plan_timeline` | timeline fields | Compile and execute synchronized robot/group tracks. |
| `motion.joint_goal` | one-segment fields | Move one joint group to mapped or profile-ordered target positions. |
| `motion.joint_delta` | one-segment fields | Apply mapped or profile-ordered joint offsets. |
| `motion.joint_trajectory` | one-segment fields | Execute mapped columns or profile-ordered sample rows at explicit times. |
| `motion.joint_effort` | one-segment fields; v2 only | Hold explicit group efforts for a duration and finish at zero effort. |
| `motion.plan_cspace_goal` | one-segment planning fields | Plan to mapped or profile-ordered joint positions. |
| `motion.plan_cspace_delta` | one-segment planning fields | Plan from mapped or profile-ordered joint offsets. |
| `motion.ik_pose` | one-segment task-space fields | Solve and execute a TCP pose goal. |
| `motion.ik_offset` | one-segment task-space fields | Solve and execute a TCP offset. |
| `motion.plan_linear_pose_path` | one-segment task-space fields | Plan and execute a straight TCP path. |
| `motion.hold` | `robot_id`, optional group, required duration | Hold a group for a fixed duration. |
| `motion.hybrid_force_position` | pose, wrench, six `force_axes`, tare and parameter generations; v3 only | Run explicit Cartesian force/position control for one robot. |

The original 20-operation v1 inventory is unchanged. v2 accepts every v1 operation
plus two mode operations and explicit joint effort. v3 accepts every v2 operation
plus four hybrid-control operations. A malformed payload whose protocol cannot be
trusted may receive a transport-level v1 failure; every accepted request receives a
response with its own protocol version.

## Runtime Operations

### Status

```json
{"protocol":"linkerbot.mirror.v1","request_id":"s1","operation":"runtime.status","arguments":{}}
```

Queue status includes `pending`, `active_request_id`, `terminal`, `capacity`, `closed`,
and `estopped`.

### Reset

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "reset-1",
  "operation": "runtime.reset",
  "arguments": {"clear_queue": true, "hold_after_reset": true}
}
```

Both options default to `true`. Clearing the queue returns the affected request IDs.
A successful reset clears emergency-stop state.

### Runtime Control Mode

The query is ordinary queued owner-thread work:

```json
{
  "protocol": "linkerbot.mirror.v2",
  "request_id": "mode-get-1",
  "operation": "control.get_mode",
  "arguments": {}
}
```

Its result contains `initial_mode`, `active_mode`, `generation`,
`supported_modes`, and `scope: "all"`. Switch all robots with an optional optimistic
generation check:

```json
{
  "protocol": "linkerbot.mirror.v2",
  "request_id": "mode-set-1",
  "operation": "control.set_mode",
  "arguments": {"mode": "velocity", "expected_generation": 0}
}
```

`mode` is exactly `position`, `velocity`, or `effort`. A successful real change
increments generation; selecting the active mode is idempotent and performs no engine
writes. A generation mismatch is rejected even for an otherwise idempotent request.
The operation is serialized between motions, rejected while emergency stop is latched,
and never rebuilds the runtime, physics world, planner, or collision context.

Each real change first neutralizes the old channel, applies every robot profile, then
neutralizes the new channel: current joint position for position mode, zero for
velocity, and zero for effort. A forward failure rolls every touched robot back in
reverse order. Rollback failure permanently fail-stops mutation until the runtime is
closed and recreated.

### Hybrid Parameters And Wrench Tare

Mirror v3 exposes hybrid gains as ordinary owner-queued state. Read the initial
profile values and their configured upper bounds:

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "hybrid-parameters-get-1",
  "operation": "control.get_hybrid_parameters",
  "arguments": {}
}
```

Update any nonempty subset between complete motions:

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "hybrid-parameters-set-1",
  "operation": "control.set_hybrid_parameters",
  "arguments": {
    "expected_generation": 0,
    "motion_stiffness": [180.0, 180.0, 220.0, 8.0, 8.0, 8.0],
    "motion_damping": [28.0, 28.0, 32.0, 1.8, 1.8, 1.8],
    "force_proportional": [0.2, 0.2, 0.3, 0.1, 0.1, 0.1],
    "force_integral": [0.4, 0.4, 0.6, 0.08, 0.08, 0.08]
  }
}
```

The fields correspond to Cartesian motion `Kp`, motion `Kd`, force `Kf`, and
force `Ki`. `posture_stiffness` and `posture_damping` are the scalar null-space
gains. Values must be finite, nonnegative, and no greater than the immutable
`tuning_limits` loaded from YAML. A real change increments the separate hybrid
parameter generation; an identical update is idempotent. The operation is serialized
in the same admission queue as motion, so it cannot interleave with an active loop.

Tare the selected physical TCP before starting hybrid motion:

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "hybrid-tare-left-1",
  "operation": "control.tare_wrench",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "reference_frame": "world"
  }
}
```

The result supplies `tare_generation`. Reset invalidates every tare. Gain updates do
not invalidate tare because the sensor/frame binding is unchanged.

### State

Read the current state:

```json
{"protocol":"linkerbot.mirror.v1","request_id":"state-1","operation":"state.get","arguments":{}}
```

The successful `result` is the owned JSON object returned by the configured Mirror
state service; it is not wrapped in another `state` field. The caller cannot mutate
runtime storage by changing the decoded response.

Write state:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "state-2",
  "operation": "state.set",
  "arguments": {
    "state": {
      "schema": "linkerbot.scene-snapshot.v1",
      "metadata": {
        "source_runtime": "mirror",
        "coordinate_frame": "scene-local",
        "info": {}
      },
      "robots": [],
      "objects": {}
    },
    "strict": true
  }
}
```

The example shows the state object's outer shape; a real request should send the
complete object returned by `state.get`. `state` is required and must be a JSON
object. `strict` is optional, defaults to `true`, and must be a JSON boolean; unknown
arguments are rejected before the adapter is called. No string, integer, or truthy
value is coerced to a boolean.

The production state adapter returns this success-result schema for `state.set`:

| Result field | Type | Contract |
| --- | --- | --- |
| `event` | string | Restore event name. |
| `accepted` | boolean | Whether the transactional restore was accepted. |
| `robots` | array of strings | Robot labels restored. |
| `objects` | array of strings | Object names restored. |
| `env_ids` | array of integers | Empty for the single-world Mirror runtime. |
| `partial` | boolean | Whether non-strict restore skipped unmatched state. |
| `message` | string, optional | Additional adapter detail when present. |

State access is ordinary queued work, never an out-of-band engine call. Ingress
threads only freeze and enqueue the request; the single runtime owner thread invokes
the state service. A `queue.cancel` can remove a pending state request. Once a
transactional `state.set` has entered its adapter, cancellation or emergency stop
cannot interrupt it halfway; it completes or rolls back atomically.

### Snapshot

Capture:

```json
{"protocol":"linkerbot.mirror.v1","request_id":"snap-1","operation":"snapshot.get","arguments":{}}
```

Restore:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "snap-2",
  "operation": "snapshot.set",
  "arguments": {
    "snapshot": {},
    "label_map": null,
    "strict": true
  }
}
```

Replace the empty object with the captured snapshot. `label_map`, when present, must
be an object; `strict` defaults to `true`.

## Motion Command Rules

All requests below target the built-in `mirror/scene3`. stdin and TCP use JSONL, so
the encoded request must occupy one line even though the examples are expanded for
readability. Robot IDs are session-local; scene3 uses its declared scene order:

Every example passes the current strict envelope and motion parser. IK and planning
success still depends on robot state, collision state, and reachability at submission
time. Task-space coordinates illustrate the wire shape; applications must derive
targets from calibration and current state.

| `robot_id` | `robot_label` | Robot profile |
| --- | --- | --- |
| `0` | `left_arm` | `ar5v2_l6v1_l` |
| `1` | `right_arm` | `ar5v2_l6v1_r` |

`robot_label` is an optional identity assertion, not an alternative selector.
`group` defaults to `arm`; the built-in arm-hand profiles also accept `hand`.

### Joint Mappings And Arrays

`joint_positions`, `joint_deltas`, and `joint_efforts` accept either form:

- a name mapping may contain only the joints that should change; omitted group joints
  retain their current command;
- a flat array must cover the whole group in the exact `joint_groups` order from the
  selected robot profile.

The scene3 arm array orders are:

```text
left_arm/arm:
  AR5V2_L_arm_joint_1, AR5V2_L_arm_joint_2, AR5V2_L_arm_joint_3,
  AR5V2_L_arm_joint_4, AR5V2_L_arm_joint_5, AR5V2_L_arm_joint_6,
  AR5V2_L_arm_joint_7

right_arm/arm:
  AR5V2_R_arm_joint_1, AR5V2_R_arm_joint_2, AR5V2_R_arm_joint_3,
  AR5V2_R_arm_joint_4, AR5V2_R_arm_joint_5, AR5V2_R_arm_joint_6,
  AR5V2_R_arm_joint_7
```

The left and right `hand` arrays both use index, middle, ring, pinky, thumb roll,
then thumb pitch; their full names use the `L6V1_L_` or `L6V1_R_` prefix. These are
profile-defined orders, never incidental USD articulation DOF orders.

For `motion.joint_trajectory`, a name mapping maps each joint to an equal-length
sample array. A matrix uses `[sample][group_joint]`, so every row must cover the full
group. `times_s` must have the same sample count and contain positive, strictly
increasing values.

### Units, Frames, And Common Fields

- Revolute joint positions and deltas use radians, positions and offsets use metres,
  and time uses seconds.
- Quaternion order is always `wxyz`.
- `reference_frame` and `offset_frame` accept only `world`, `env`, `robot_base`, or
  `tcp`.
- Direct hold, joint, and trajectory operations require `duration_s`. Planning
  operations may inherit `planning.request_defaults.duration_s`.
- Planning segments may override `duration_s`, `sample_dt_s`, `avoid_collisions`, and
  `force_collision_refresh`. `timeout_s` is forbidden on the wire.
- `coordination` belongs on a one-segment wrapper or at timeline root.
  `independent` excludes other robots from planning obstacles, `static_others` treats
  them as static obstacles, and `coupled` is currently rejected.
- `interpolation` is valid only for `joint_goal` and `joint_delta`, and accepts
  `linear` or `smoothstep`.

| Operation | Required payload | Main optional fields |
| --- | --- | --- |
| `motion.hold` | `robot_id`, `duration_s` | `robot_label`, `group` |
| `motion.joint_goal` | `robot_id`, `duration_s`, `joint_positions` | `interpolation` |
| `motion.joint_delta` | `robot_id`, `duration_s`, `joint_deltas` | `interpolation` |
| `motion.joint_trajectory` | `robot_id`, `duration_s`, `joint_positions`, `times_s` | no kind-specific fields |
| `motion.joint_effort` | `robot_id`, `duration_s`, `joint_efforts` | `robot_label`, `group`, `phase` |
| `motion.plan_cspace_goal` | `robot_id`, `joint_positions` | planning overrides |
| `motion.plan_cspace_delta` | `robot_id`, `joint_deltas` | planning overrides |
| `motion.ik_pose` | `robot_id`, `target_position` | orientation, TCP, frame, planning overrides |
| `motion.ik_offset` | `robot_id`, `offset` | orientation, TCP, frame, planning overrides |
| `motion.plan_linear_pose_path` | `robot_id` and exactly one target/offset form | orientation mode, TCP, frame, planning overrides |
| `motion.plan_timeline` | `tracks` | `coordination`, `force_collision_refresh` |

## Single-Segment Motion Examples

### 1. Hold The Left Arm: `motion.hold`

This holds the current arm command for one second while physics continues to step:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-hold-left-arm-1",
  "operation": "motion.hold",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 1.0
  }
}
```

### 2. Move The Left Arm Home With An Array: `motion.joint_goal`

The array must contain exactly seven arm values. This request moves to the configured
zero pose in two seconds without resetting objects or the hand:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-joint-goal-left-home-1",
  "operation": "motion.joint_goal",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 2.0,
    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "interpolation": "smoothstep"
  }
}
```

### 3. Apply A Partial Left-Arm Delta: `motion.joint_delta`

Only joints 1 and 4 change; omitted arm commands remain unchanged:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-joint-delta-left-1",
  "operation": "motion.joint_delta",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 1.0,
    "joint_deltas": {
      "AR5V2_L_arm_joint_1": 0.1,
      "AR5V2_L_arm_joint_4": -0.1
    },
    "interpolation": "smoothstep"
  }
}
```

### 4. Execute Sampled Joint Positions: `motion.joint_trajectory`

Each named joint has three samples, and omitted joints hold their current command:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-joint-trajectory-left-1",
  "operation": "motion.joint_trajectory",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 1.5,
    "joint_positions": {
      "AR5V2_L_arm_joint_1": [0.1, 0.2, 0.0],
      "AR5V2_L_arm_joint_2": [-0.1, -0.2, 0.0]
    },
    "times_s": [0.5, 1.0, 1.5]
  }
}
```

### 5. Apply Explicit Left-Arm Effort: `motion.joint_effort`

This v2-only command holds the requested efforts, clamps them to the active controller
profile limits, avoids planner/collision access, and writes zero effort at completion:

```json
{
  "protocol": "linkerbot.mirror.v2",
  "request_id": "motion-joint-effort-left-1",
  "operation": "motion.joint_effort",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 0.2,
    "joint_efforts": {
      "AR5V2_L_arm_joint_1": 2.5,
      "AR5V2_L_arm_joint_2": -1.0
    },
    "phase": "contact_push"
  }
}
```

### 6. Run Hybrid Force/Position Control: `motion.hybrid_force_position`

This v3-only request freezes generation 1 gains for its whole duration. The Z
translation is force-controlled; the other five Cartesian axes use explicit motion
impedance. A later request may choose a different `force_axes` array independently:

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "motion-hybrid-force-position-left-1",
  "operation": "motion.hybrid_force_position",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "duration_s": 0.5,
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "reference_frame": "world",
    "target_position": [0.35, 0.0, 0.25],
    "target_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
    "force_axes": [false, false, true, false, false, false],
    "target_wrench": [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
    "tare_generation": 1,
    "hybrid_parameter_generation": 1,
    "phase": "normal_force_hold"
  }
}
```

`force_axes` is request state, not a persistent tuning parameter. The motion snapshots
all six gain groups once during preflight. A parameter update queued after this request
cannot affect its loop; the next motion must carry the new generation. Raw PhysX
wrench is environment-on-tool, while `target_wrench` and returned feedback are
tool-on-environment.

### 7. Plan A C-Space Home Goal: `motion.plan_cspace_goal`

Unlike direct joint interpolation, this operation uses the MotionPlanner.
`static_others` includes the right arm as a static planning obstacle:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-plan-cspace-goal-left-1",
  "operation": "motion.plan_cspace_goal",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "coordination": "static_others",
    "duration_s": 2.0,
    "sample_dt_s": 0.02,
    "joint_positions": {
      "AR5V2_L_arm_joint_1": 0.0,
      "AR5V2_L_arm_joint_2": 0.0,
      "AR5V2_L_arm_joint_3": 0.0,
      "AR5V2_L_arm_joint_4": 0.0,
      "AR5V2_L_arm_joint_5": 0.0,
      "AR5V2_L_arm_joint_6": 0.0,
      "AR5V2_L_arm_joint_7": 0.0
    },
    "avoid_collisions": true,
    "force_collision_refresh": true
  }
}
```

### 6. Plan A C-Space Delta: `motion.plan_cspace_delta`

Omitted duration and sample period come from the planning profile:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-plan-cspace-delta-left-1",
  "operation": "motion.plan_cspace_delta",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "joint_deltas": {
      "AR5V2_L_arm_joint_2": 0.05,
      "AR5V2_L_arm_joint_4": -0.05
    },
    "avoid_collisions": false
  }
}
```

### 7. Solve An Absolute TCP Pose: `motion.ik_pose`

The position is interpreted in `reference_frame`; orientation uses `wxyz`.
`ik_pose` does not accept `orientation_mode`: supplying the quaternion constrains the
target orientation.

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-ik-pose-left-1",
  "operation": "motion.ik_pose",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "target_position": [0.35, 0.0, 0.25],
    "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "reference_frame": "robot_base",
    "duration_s": 2.0,
    "sample_dt_s": 0.02,
    "avoid_collisions": false
  }
}
```

### 8. Solve A TCP-Relative Offset: `motion.ik_offset`

This moves 3 cm along the current TCP Z axis and constrains position only:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-ik-offset-left-1",
  "operation": "motion.ik_offset",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "offset": [0.0, 0.0, 0.03],
    "offset_frame": "tcp",
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "duration_s": 1.0,
    "avoid_collisions": false
  }
}
```

### 9. Plan A Straight TCP Path: `motion.plan_linear_pose_path`

The relative form uses `offset` and must not also contain `target_position`. This
request moves 5 cm along TCP Z while maintaining the starting TCP orientation:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-linear-offset-left-1",
  "operation": "motion.plan_linear_pose_path",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "offset": [0.0, 0.0, 0.05],
    "offset_frame": "tcp",
    "orientation_mode": "current",
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "duration_s": 1.5,
    "sample_dt_s": 0.02,
    "avoid_collisions": true,
    "force_collision_refresh": true
  }
}
```

The absolute form uses `target_position` and `reference_frame`.
`orientation_mode: target` requires a target quaternion:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-linear-target-left-1",
  "operation": "motion.plan_linear_pose_path",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "target_position": [0.35, 0.0, 0.30],
    "reference_frame": "robot_base",
    "orientation_mode": "target",
    "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "duration_s": 2.0,
    "sample_dt_s": 0.02,
    "avoid_collisions": false
  }
}
```

`free` constrains position only, `current` maintains the starting TCP orientation,
and `target` uses the supplied target orientation.

## Multi-Robot Timeline Example

### 10. Synchronize Both Arms: `motion.plan_timeline`

Different robot tracks start at global tick zero. Group tracks in one `unit` start
together, while `segments` in one group track execute serially. This request moves
both arms and the left hand to their configured zero positions:

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-timeline-dual-home-1",
  "operation": "motion.plan_timeline",
  "arguments": {
    "coordination": "independent",
    "force_collision_refresh": false,
    "tracks": [
      {
        "robot_id": 0,
        "robot_label": "left_arm",
        "units": [
          {
            "group_tracks": [
              {
                "group": "arm",
                "segments": [
                  {
                    "kind": "joint_goal",
                    "duration_s": 2.0,
                    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "interpolation": "smoothstep"
                  }
                ]
              },
              {
                "group": "hand",
                "segments": [
                  {
                    "kind": "joint_goal",
                    "duration_s": 2.0,
                    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "interpolation": "smoothstep"
                  }
                ]
              }
            ]
          }
        ]
      },
      {
        "robot_id": 1,
        "robot_label": "right_arm",
        "group": "arm",
        "segments": [
          {
            "kind": "joint_goal",
            "duration_s": 2.0,
            "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "interpolation": "smoothstep"
          }
        ]
      }
    ]
  }
}
```

Timeline compilation is atomic: all structural validation and required planning
finish before execution. If any segment fails to compile, no part of the timeline
runs. A successful motion result contains `event: motion_completed`, the operation,
and the runtime's cumulative physics `steps`; `steps` is not a per-request tick count.

## Cancellation And Emergency Stop

`queue.cancel`, `queue.cancel_current`, `runtime.estop`, and `runtime.quit` are handled
out of band so they are not trapped behind ordinary work. Active motion observes a
cooperative cancellation flag at execution boundaries.

Emergency stop rejects new `motion.*` requests, cancels pending work, marks active
work for cancellation, and prevents idle physics stepping. While latched,
`state.get` remains available for inspection, but a new `state.set` fails with
`runtime_estopped`. A state transaction that started before the latch may finish or
roll back atomically; it never clears the latch.

The existing snapshot contract is intentionally unchanged: `snapshot.set` remains
available while emergency-stopped so a versioned cold snapshot can be restored while
physics is frozen, but it also does not clear the latch. Only a successful
`runtime.reset` clears emergency stop and permits motion and new `state.set` requests.

## Admission And Timeouts

- The queue is bounded; overflow returns `queue_capacity_exceeded`.
- A retained request ID cannot be reused; duplicates return
  `duplicate_request_id`.
- A transport response timeout atomically expires a request that is still pending, so
  it cannot be claimed and executed later. An already-active operation receives
  cooperative cancellation but may have crossed a mutation boundary; query status
  before retrying non-idempotent motion.
- Each connection preserves request/response order, but several ingress connections
  may submit concurrently.

See [Mirror CLI](mirror-cli.md), [Control And Trajectories](../guides/control-and-trajectories.md),
and [Snapshots](snapshots.md).
