# Control And Trajectory Guide

Language: [English](control-and-trajectories.md) | [中文](../../zh-CN/guides/control-and-trajectories.md)

This guide helps application users choose the correct control and trajectory path.
It does not redefine message fields; use the [Single Scene JSON reference](../reference/single-scene-json.md)
and [Tiled Scene JSON reference](../reference/tiled-scene-json.md) for exact schemas.

## Choose The Execution Path

| Need | Use | Runtime |
| --- | --- | --- |
| Hold, move joints, or execute a sampled joint curve for one robot/group | Single-segment Single Scene command | Single Scene |
| Start several robots and arm/hand groups on one shared tick axis | `plan_timeline` | Single Scene |
| Plan one Single Scene arm target through joint space or task space | Single Scene planning segment | Single Scene |
| Apply a fixed-tick command to selected cloned env rows | `step` | Tiled Scene |
| Load a known joint trajectory and advance it under explicit caller control | `load_trajectory` then `step_trajectory` | Tiled Scene |
| Append a sparse named hand-joint subtrack | `hand` | Tiled Scene |
| Plan without blocking physics, inspect completion, then load/play the result | `plan`, `planner_status`, `step_trajectory` | Tiled Scene |

Single Scene timeline and Tiled Scene playback solve different synchronization problems. Single Scene compiles
all tracks atomically before execution and applies every robot target before one World step.
Tiled Scene playback owns a queue per robot/env row and advances only when the caller sends
`step_trajectory`.

## Command Space And Joint Order

Public joint vectors are not arbitrary articulation arrays:

- Single Scene group vectors follow `status.robots[].joint_groups.arm` or `.hand`.
- Tiled Scene command vectors follow `status.robots[].command_joints`.
- Planning vectors follow the selected backend's planning-joint order.
- A `JointTrajectory` column order is exactly its `joint_names` order.

Named mappings and `joint_names` are preferable when a request intentionally controls a
subset. Prefix-based Tiled Scene `step` actions write the first `D` command columns and retain the
remaining targets. Controller/runtime logic expands active command joints to mimic followers;
clients must not send separate follower targets unless a documented interface asks for them.

Positions use radians for revolute joints and meters for prismatic joints. Velocities use the
corresponding units per second. Effort dimensions follow the PhysX joint type.

## Single Scene Timeline Model

Single Scene uses this hierarchy:

```text
timeline
  robot track                 parallel from global tick 0
    motion unit               serial within one robot
      arm/hand group track    parallel within one unit
        segment               serial within one group
```

One unit ends when its longest group track ends; a shorter group holds its terminal target.
The whole timeline ends when its longest robot track ends. Compilation is atomic: a planning
or validation failure prevents every track from starting.

Minimal coordinated request:

```json
{
  "type": "plan_timeline",
  "id": "arm-and-hand",
  "tracks": [
    {
      "robot_id": 0,
      "units": [
        {
          "group_tracks": [
            {
              "group": "arm",
              "segments": [
                {
                  "kind": "joint_goal",
                  "joint_positions": {"AR5V2_L_arm_joint_1": 0.2},
                  "duration_s": 0.5
                }
              ]
            },
            {
              "group": "hand",
              "segments": [
                {"kind": "hold", "duration_s": 0.5}
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

Use single-segment Single Scene commands for one track. Use `plan_timeline` whenever same-tick
coordination is part of the requirement; sending separate JSONL requests does not make them
start together.

## Tiled Scene Synchronous `step`

`step` converts targets and advances a fixed number of physics ticks before returning. Every
env-scoped request supplies explicit `env_ids`; multi-robot scenes also require the documented
robot selector.

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

Use `step` for closed-loop policies that produce the next target from the latest observation.
Joint actions support absolute and relative command-space prefixes. End-effector actions use
batched IK, and `ee_linear_path` computes all waypoints before the first physics write.

All envs share one World step. Unselected envs therefore advance in time while holding their
latest targets.

## Tiled Scene Trajectory Buffer

Use the buffer when trajectory samples already exist or when an asynchronous planner result
should be replayed later:

1. `load_trajectory` validates and stages all selected rows atomically.
2. `step_trajectory` advances playback by explicit physics ticks.
3. `trajectory_status` reports active, queued, completed, capacity, and rejection data.
4. `clear_trajectory` removes selected robot/env entries.

The buffer is bounded per env by queue depth, samples, and duration. `replace` validates only
the replacement sequence; append validates existing plus new content. A capacity failure
rejects the complete selected-env load instead of evicting an active trajectory.

The `hand` command is a sparse named-joint convenience path. It can append a hand subtrack
without overwriting the arm endpoint when playback begins. It is not a same-tick replacement
for a Single Scene arm/hand timeline.

## Tiled Scene Asynchronous Planning

An asynchronous request has a separate planning and playback lifecycle:

```text
plan submission
  -> queued planner request
  -> planner_status dispatch/collect
  -> optional atomic playback load
  -> step_trajectory playback
```

`plan` only queues work. `planner_status` and `step_trajectory` dispatch/collect ready work.
Successful results with `load_on_success=true` enter the trajectory buffer if its admission
checks pass. Always inspect `ready`, `loaded`, and `load_rejected`; planner success does not
guarantee playback capacity.

Use request IDs for cancellation and completed-result management. Reset, state restore, and
other overlapping mutations can cancel stale planning work for the affected robot/env rows.

## Timing And Interpolation

- Single Scene converts `duration_s` to `ceil(duration_s / physics_dt)` ticks.
- Single Scene `sample_dt_s` controls a planning grid, not World physics dt.
- Tiled Scene joint `step` uses positive `decimation` physics ticks.
- Tiled Scene `ee_linear_path` accepts either a logical `duration_s` or explicit `decimation`, not both.
- Tiled Scene trajectory sample times are finite and strictly increasing for multi-sample data.
- `linear` interpolation uses uniform progress; `smoothstep` eases progress without changing
  the requested geometric endpoints.

The response's actual tick count is authoritative. Do not reconstruct completion from wall
clock time or assume a decimal duration is exactly divisible by physics dt.

## Frames, Orientation, And TCP

Public positions use meters and quaternions use `wxyz` order. Frame fields are interface
specific:

- Single Scene task-space commands explicitly name `world`, `env`, `robot_base`, or the documented
  offset frame.
- Tiled Scene synchronous named end-effector targets resolve `env`, `base`, or `world`.
- Tiled Scene asynchronous linear pose goals are robot-base-local and do not accept a
  `pose_reference_frame` field.

`free` constrains position only, `current` preserves the starting orientation, and `target`
requires a target quaternion. Confirm the robot's registered `tcp_frame_name` through status;
do not infer a TCP from a link name.

## Control Modes

Single Scene controller profiles can configure position, velocity, or effort control using the
supported implicit, explicit, or direct method for each component. The selected runtime mode
and controller bundle must agree. Tiled Scene runtime currently accepts position control only and
rejects velocity or effort mode during configuration resolution.

Mimic followers remain controller-owned position drives. Planning normally targets the arm
group; hand motion remains direct command-space control.

## Failure And Recovery

- A rejected JSON request performs no command mutation.
- Single Scene timeline compilation is all-or-nothing before execution.
- Tiled Scene `reject_request` IK policy rejects before any selected robot target or physics write.
- Tiled Scene `hold_failed_env` preserves the last successful target for failed rows while reporting
  their diagnostics.
- A failed trajectory load does not partially fill selected env buffers.
- A runtime in fail-stop rejects later mutations; recreate it rather than retrying control.

Use `status`, `trajectory_status`, or `planner_status` for the lifecycle you selected. A
submission response proves admission, not terminal execution.

## Related Documentation

- [Single Scene JSON Reference](../reference/single-scene-json.md)
- [Tiled Scene JSON Reference](../reference/tiled-scene-json.md)
- [Motion Planning](motion-planning.md)
- [Configuration](configuration.md)
- [Known Constraints](../operations/constraints.md)
