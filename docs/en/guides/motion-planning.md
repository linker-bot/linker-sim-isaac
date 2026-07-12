# cuRobo Usage And Batch Scheduling

Language: [English](motion-planning.md) | [中文](../../zh-CN/guides/motion-planning.md)

This document describes the current cuRobo v0.8.0 integration, the executable
linear planner, `SingleSceneRuntime` planning, Tiled Scene synchronous IK, and Tiled Scene async
batch scheduling.

Use the [Configuration Reference](../reference/configuration.md) for every YAML
field, the [Python API Reference](../reference/python-api.md) for supported
facades and result types, and the [Tiled Scene JSON Reference](../reference/tiled-scene-json.md)
for the complete protocol field and status inventory. This guide owns how those
contracts compose into planning workflows.

## Backend Selection

Both runtime modes share project-level planning requests, results, and the
`runtime.planner.backend` owner. Single Scene additionally exposes a launch-specific
CLI override; Tiled Scene does not:

| Runtime path | Selection | Supported work |
|---|---|---|
| Single Scene `curobo` | `runtime.planner.backend: curobo` (`--planner-backend curobo` overrides) | Joint goals, pose-goal trajectories, TCP linear paths, and optional collision-aware planning |
| Single Scene `linear` | `runtime.planner.backend: linear` (`--planner-backend linear` overrides) | `plan_cspace_goal` and `plan_cspace_delta` only |
| Tiled Scene async `curobo` | `runtime.planner.backend: curobo` | Joint batch/per-env planning and per-env task-space paths |
| Tiled Scene async `linear` | `runtime.planner.backend: linear` | Executable joint-space interpolation |
| Tiled Scene synchronous EE action | Robot cuRobo IK binding | Batched `ee_*` actions, independent of the async planner selection |

The linear backend does not create a cuRobo solver and does not provide IK,
collision checking, joint-limit validation, or velocity/acceleration constrained
optimization. It is a deliberate joint interpolation policy, not a collision
planner.

`status.supports_planning` reports whether a robot has a valid cuRobo binding.
It does not report availability of the model-free linear policy.

## Configuration Layers

Algorithm defaults belong in `configs/curobo/<profile>.yaml`:

- CUDA device and tensor dtypes.
- IK and planner seed counts and tolerances.
- CUDA graph settings.
- Batch sizes and `multi_env`.
- Self-collision and scene collision cache capacity.
- The validated `task_bundle` and planner `warmup` lifecycle policy.

Project-owned cuRobo profiles accept only the current strict structure. Their
fixed mappings reject unknown fields with the full nested path, booleans and
numeric fields use strict YAML types, and invalid ranges fail before a CUDA
context is created. The four tensor dtype fields currently accept only the
project-tested `float32` combination.

Only `task_bundle: curobo_v0_8_default` is supported. Raw optimizer, rollout,
transition-model, or graph-planner file paths are rejected because those files
are cuRobo-version contracts. Context creation verifies that the installed
cuRobo version is exactly `0.8.0`; other `0.8.x` patch versions are not assumed
compatible with the selected bundle. `motion_planner.warmup`
defaults to `true`; `false` skips the explicit lazy-creation warmup and moves
the cold-start cost to the first real request.

Files below `configs/curobo/task/**/*.yml` are vendored cuRobo 0.8.0 resources.
The exact `curobo_v0_8_default` bundle owns and validates the complete file set.

Robot resources belong in `configs/robots/<robot>.yaml`:

```yaml
curobo:
  enabled: true
  planning_joint_group: arm
  robot:
    urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
    robot_config_path: assets/single_system/arm/AR5V2_L/AR5V2_L_curobo.yml
    flange_frame: AR5V2_L_arm_flan_link
    default_tcp_frame: AR5V2_L_pinch_tcp
    load_collision_spheres: true
```

The merge order is algorithm profile first, robot profile second, so explicit
robot values win. MJCF remains an Isaac simulation asset; cuRobo receives a
planning URDF and optional robot YAML. A hand-only profile must set
`curobo.enabled: false`.

Custom TCP URDFs are written below `runtime.paths.cache_root/curobo`. When that
runtime value is null, resolution uses `LINKERBOT_SIM_CACHE_ROOT`, then
`XDG_CACHE_HOME/linkerbot_sim`, then `~/.cache/linkerbot_sim`. A relative
runtime or environment cache root is expanded and resolved against the process
working directory. The repository `.cache` directory is never the fallback.

## SingleSceneRuntime

Start Single Scene mode with an explicit backend when the choice matters:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --planner-backend curobo --curobo-profile default \
  --tcp-jsonl-port 8765 --gui
```

A planned joint goal is backend-neutral at the JSON boundary:

```json
{
  "type": "plan_cspace_goal",
  "id": "joint-plan-1",
  "robot_id": 0,
  "group": "arm",
  "joint_positions": {"AR5V2_L_arm_joint_1": 0.2},
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

With `--planner-backend linear`, only the two `plan_cspace_*` kinds are
accepted and `avoid_collisions` must be false. Task-space kinds require cuRobo:

```json
{
  "type": "ik_pose",
  "id": "pose-1",
  "robot_id": 0,
  "target_position": [0.35, 0.0, 0.4],
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "reference_frame": "world",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

`sample_dt_s` is an optional positive planner output interval and defaults to
the Single Scene physics dt. It does not modify the World physics step. The compiled
trajectory is resampled to integer physics ticks before execution.

Runtime request values are resolved before a command enters the queue. Explicit
JSON takes precedence over `runtime.planner.request_defaults` and
`runtime.execution.command_defaults`. `duration_s`, `avoid_collisions`,
`force_collision_refresh`, and `coordination` are planner defaults; joint
`interpolation`, task-space frame, and linear-path `orientation_mode` are
command defaults. `coupled` is unsupported. The model-free linear planner
receives the resolved duration and uses the actual Single Scene physics dt when
`sample_dt_s` is absent.

`ik_pose` and `ik_offset` are Single Scene protocol kind names. They do not call the
single-solution `CuroboInverseKinematics.solve()` facade. The Single Scene compiler
transforms the target into robot-base coordinates, builds
`MotionRequest(goal_pose=...)`, and `CuroboMotionPlanner.plan()` invokes cuRobo
`MotionPlanner.plan_pose()` to produce an executable trajectory. Direct
single-target IK is available only through the Python facade.

## Tiled Scene Async Planner

The Tiled Scene entrypoint has no planner backend CLI override. The runtime profile
owns the backend, cuRobo profile, and joint batch mode:

```yaml
runtime:
  profiles:
    curobo: default
  planner:
    backend: curobo
    joint_batch_mode: auto
```

Planner selection belongs to the runtime profile.

`joint_batch_mode` accepts:

- `auto`: prefer `BatchMotionPlanner`, then use per-env planning when the
  request cannot use the batch path.
- `per_env`: disable joint batch planning.
- `batch_only`: fail with `BATCH_UNAVAILABLE` when batching is unavailable.

Submit a request with all fields at the top level:

```json
{
  "type": "plan",
  "request_id": "batch-plan-1",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "kind": "joint_position_target",
  "joint_positions": [0.2, 0.1],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false,
  "load_on_success": true,
  "replace": true
}
```

The main thread snapshots the selected current command rows, then returns
`plan_submitted`. Workers never access Isaac `World`, stage, articulation
views, or PhysX handles. Poll and play back with:

Omitted `duration_s`, `avoid_collisions`, `load_on_success`, and `replace`
come from `runtime.planner.request_defaults`; omitted `sample_dt_s` comes from
the actual physics dt. Explicit JSON fields always take precedence.

```jsonl
{"type":"planner_status","wait_timeout_s":0.1}
{"type":"step_trajectory","robot_id":0,"env_ids":[0,1,2,3],"decimation":4}
```

Successful results with `load_on_success=true` enter the per-env trajectory
buffer. A planner completion alone does not advance physics.

For async `kind="linear_pose_path"`, `target_position` or `target_offset` is one
three-element vector shared by the selected envs; the optional target
orientation is one wxyz quaternion. These values are interpreted directly in
the cuRobo robot-base-local frame. The canonical async plan API does not define
or apply `pose_reference_frame`; including that field is rejected rather than
silently ignored. It also does not accept per-env `(E,3)/(E,4)` targets. Convert world/env
targets before submission, or submit separate requests when envs need different
targets.

The Tiled Scene async API also rejects Single Scene-only `coordination` and
`force_collision_refresh`. Tiled Scene async requests remain atomic and each cuRobo
worker request owns an isolated context. Other unknown fields are rejected per
plan kind.

## Synchronous Tiled Scene Linear TCP Motion

`type="step", kind="ee_linear_path"` is a synchronous control action, not an
async `MotionPlanner` request. It accepts exactly one target representation:

- `target_offset`: a named relative target interpreted with
  `pose_reference_frame`.
- `target_position`: an absolute named target interpreted with
  `pose_reference_frame`.
- `values`: a compact world-frame offset form.

```json
{
  "type": "step",
  "kind": "ee_linear_path",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "target_offset": [0.0, 0.0, 0.1],
  "orientation_mode": "free",
  "duration_s": 0.4,
  "sample_dt_s": 0.02,
  "interpolation": "linear",
  "tcp_frame_name": "AR5V2_L_pinch_tcp"
}
```

`orientation_mode` is `free`, `current`, or `target`. `target` requires
`target_orientation_quat_wxyz`; `free` performs position-only IK. When the mode
is omitted but an explicit target quaternion is present, the parser infers
`target`. Otherwise the resolved
`runtime.execution.command_defaults.orientation_mode` applies (`current` in the
bundled default profile). Explicit action fields always win. The
[Tiled Scene JSON Reference](../reference/tiled-scene-json.md) owns the complete field and
failure-policy contract.

The action runs batched IK across selected envs at each sampled waypoint. Time
is sequential so the previous waypoint solution warm-starts the next one. The
sparse IK grid uses `ceil(duration_s / sample_dt_s)` waypoints and is then
resampled to `ceil(duration_s / physics_dt)` execution ticks. All IK completes
before physics advances. With `failure_policy: hold_failed_env`, an env freezes
after its first numerical IK failure while every env still executes the same
number of physics ticks. With `failure_policy: reject_request`, any numerical
failure atomically rejects the complete action before target-cache or physics
writes.

## Cross-Request Batch Scheduling

`TiledPlannerManager` groups consecutive FIFO joint-space requests only when
their batch keys match. The key includes robot identity, command joint names,
duration, sample dt, collision requirement, and segment structure. It does not
reorder requests across an incompatible item.

The manager counts problem rows, not request objects. The bundled runtime
profiles set `runtime.planner.resources.max_batch_problems: 64`.
`TiledCuroboPlanningBackend.plan_many()` stacks a bounded group into a
`CuroboBatchJointProblem`; the cuRobo batch core has no `env_id`, `request_id`,
source, or playback fields. Results are split back to the original request IDs
and env rows after planning.

Rows below the fixed cuRobo batch size are padded by repeating the last real
row. With the bundled `runtime.planner.oversize_request_policy: split`, one
public request above `max_batch_problems` is dispatched in bounded env-row
chunks and merged atomically back into the original request result. Set the
policy to `reject` to reject such a request before dispatch. No backend call may
exceed `max_batch_problems`; the resolved limit must also fit the selected
cuRobo batch capacity.

Each active cuRobo planner future owns its context, CUDA graph/cache, and tensors.
Increasing `--planner-workers` can increase throughput but also multiplies
memory use and warmup cost. Measure one or two workers before increasing it.

## Collision Capability

`avoid_collisions=true` is a strict requirement. The request fails instead of
silently falling back unless all of these are available:

- Robot collision spheres.
- A scene collision checker for the selected solver/planner.
- Enough configured `cuboid`/`mesh` cache capacity.
- A materialized collision view synchronized to the current scene version.

`multi_env=false` means a tiled batch shares one collision world. Do not assume
per-env obstacle independence when obstacle poses differ unless the backend has
materialized equivalent per-env worlds.

## Current Boundaries

- Tiled Scene joint-space planning can use `BatchMotionPlanner`; async
  `linear_pose_path` still plans each env separately with sequential warm-start
  IK.
- Synchronous `ee_linear_path` batches the env dimension at each waypoint but
  is not collision-aware graph search or trajectory optimization.
- Oversized public requests follow `runtime.planner.oversize_request_policy`;
  the bundled default splits them at `max_batch_problems`, while `reject`
  refuses them before dispatch.
- Public task-space quaternions are always wxyz.
- The validated Isaac/cuRobo stack requires the Warp API adapter.

## Code Index

- Shared requests/results: `src/linkerbot_sim/planning/`
- Linear backend: `src/linkerbot_sim/planning/linear_backend.py`
- cuRobo config/context: `src/linkerbot_sim/backends/curobo/config.py`,
  `context.py`
- cuRobo IK/planning: `inverse_kinematics.py`, `motion_planner.py`,
  `linear_pose_path.py`
- cuRobo batch core: `src/linkerbot_sim/backends/curobo/batch/`
- Tiled Scene integration: `src/linkerbot_sim/tiled/planning/backends/curobo.py`
- Async manager: `src/linkerbot_sim/tiled/planning/manager.py`
- Single Scene compiler: `src/linkerbot_sim/app/motion/timeline/compiler.py`
