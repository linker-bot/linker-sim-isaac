# Tiled Usage And Command Format

Language: [English](tiled-usage-and-command-format.md) | [中文](../../zh-CN/并行环境/Tiled%20并行环境使用方式与指令格式.md)

This document describes the Isaac Lab style tiled env runtime: how to start it, how profiles are structured, and how JSON commands are shaped.

Tiled mode means: one Isaac/PhysX scene contains multiple homogeneous env instances and advances them with synchronized command steps. It is not a parallel copy of the old single-arm/dual-arm motion runtime. Planning is attached around the runtime through trajectory buffers and an async planner manager. `TiledCommandAdapter` itself remains a synchronous step-control adapter.

## Capability Boundary

Supported:

- One `SimulationApp`, one `World`, one PhysX scene.
- Env roots such as `/World/envs/env_0 ... /World/envs/env_N`.
- Single-arm and dual-arm tiled envs.
- Same robot/object set in every env, with per-env overrides for same-name object poses.
- Batched articulation views for joint state read/write.
- Per-env `reset`, `get_state`, `set_state`, and target updates.
- `load_trajectory` and `step_trajectory` playback for batched joint trajectories.
- `plan`, `planner_status`, and `cancel_plan` for async planning.
- `before`, `sync`, and `after` hand overlays on loaded trajectories and ready planner results.
- `hand` / `dual_hand` as hand-only motions in the selected robot/env trajectory playback queue.
- stdin JSONL and TCP JSONL command transports.
- Optional Foxglove live / MCAP telemetry.

Not supported:

- Different object sets per env.
- Old runtime `cancel_current` / `estop` semantics.
- Running graph search or trajectory optimizer inside `TiledCommandAdapter`.
- A general old-style running/pending motion queue; tiled keeps a trajectory playback queue.
- Different physics step counts per env.
- WebSocket command transport for tiled interactive mode.

## Start

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --default-decimation 1
```

Ready output:

```text
TILED_INTERACTIVE_READY
```

TCP JSONL startup:

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --no-stdin \
  --tcp-jsonl-host 127.0.0.1 \
  --tcp-jsonl-port 9003
```

Client:

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 9003
```

## Tiled Env Profile

Directory-style profiles are recommended:

```text
configs/envs/scene3_tiled/
  base.yaml
  envs/
    env_000.yaml
    env_001.yaml
    env_002.yaml
    env_003.yaml
```

`base.yaml` stores shared env settings: world, visuals, robots, objects, shared
`sensors.cameras` parameters, and tiled topology.

Per-env files store differences: same-name object `root_pose` overrides and same-name
camera `pose` overrides:

```yaml
env_id: 1
objects:
  Tblock:
    root_pose:
      xyz: [0.12, 0.04, -0.4]
      rpy: [0.0, 1.5707, 0.18]
cameras:
  world_rgbd:
    pose:
      xyz: [0.08, 0.0, 0.08]
      rpy: [0.0, 1.1, 0.0]
```

The tiled runtime creates one camera per env, such as
`/World/envs/env_0/WorldRGBD`. Offline save directories and Foxglove topic prefixes
get an `env_000` suffix so multiple envs do not write to the same frame files or
topics.

Important tiled fields:

```yaml
tiled:
  enabled: true
  num_envs: 4
  base_env_path: /World/envs
  env_prefix: env
  spacing: 2.0
  per_env_config_dir: envs
  clone:
    filter_collisions: true
  runtime:
    inspect_env_ids: [0]
```

The env count is defined by YAML `tiled.num_envs`; the CLI does not override it.

## CLI Parameters

| Parameter | Meaning |
| --- | --- |
| `--env` | Env profile name, for example `scene3_tiled`. |
| `--gui` | Open Isaac GUI. |
| `--default-decimation` | Physics ticks for an action when the action omits `decimation`. |
| `--planner-backend` | Async planner backend: `linear` or `cumotion`. |
| `--planner-workers` | Number of async planner workers. Workers consume snapshots and do not access Isaac runtime. |
| `--max-pending-requests` | In-flight planner request limit. |
| `--stdin / --no-stdin` | Enable/disable stdin JSONL. |
| `--hold` | Keep current targets and continue idle GUI/Foxglove stepping; also keep the process alive after stdin EOF. |
| `--tcp-jsonl-host` | TCP JSONL host. |
| `--tcp-jsonl-port` | TCP JSONL port. |

Telemetry parameters:

| Parameter | Meaning |
| --- | --- |
| `--foxglove-live-host` | Foxglove live host. |
| `--foxglove-live-port` | Foxglove live port; recommended tiled port is `8767`. |
| `--foxglove-mcap-path` | MCAP output path. |
| `--telemetry-env-ids` | Comma-separated selected env ids. |
| `--telemetry-rate-hz` | Periodic telemetry rate. `0` disables periodic publish. |
| `--telemetry-topic-prefix` | Topic prefix, default `/tiled`. |

## JSON Rules

- All messages are JSON objects.
- Length unit is m.
- Angle unit is rad.
- Quaternion order is `wxyz`.
- `env_ids` selects env rows. Omitted means all envs.
- `robot` / `robots` selects logical robot names such as `left` and `right`.
- Physics advances all envs synchronously, even if only some env targets are updated.
- For `joint_position_target` and `joint_delta_pos`, the width of `values` is the action width for this command. A short vector writes the first K command joints of the selected robot. For example, `values:[[0.01,-0.01,0]]` writes only the first three command joints. To command seven joints, send seven columns, such as `values:[[0.01,-0.01,0,0,0,0,0]]`.
- `info.<robot>.command_width` in a step response reports the action width used by that command. It does not mean the robot has only that many command joints. Use `status` to inspect `robots.<name>.command_joints`.

Control messages:

```json
{"type":"status"}
{"type":"reset","env_ids":[0,2]}
{"type":"get_state","env_ids":[0],"fields":["robots.left.joint_positions","episode_steps"]}
{"type":"set_state","env_ids":[1],"state":{}}
{"type":"quit"}
```

Step actions:

```json
{"type":"step","kind":"hold","robots":["left"],"decimation":1}
{"type":"step","kind":"joint_position_target","env_ids":[0],"robots":["left"],"values":[[0,0,0,0,0,0,0]]}
{"type":"step","kind":"joint_delta_pos","env_ids":[0],"robots":["left"],"values":[[0.01,0,0,0,0,0,0]]}
{"type":"step","kind":"ee_pose_target","env_ids":[0],"robots":["left"],"position":[[0.1,0.0,0.2]],"orientation_wxyz":[[1,0,0,0]]}
```

Trajectory playback:

```json
{"type":"load_trajectory","robot":"left","env_ids":[0],"times":[0.0,0.5],"positions":[[[0,0,0,0,0,0,0]],[[0.1,0,0,0,0,0,0]]]}
{"type":"step_trajectory","robot":"left","decimation":4}
```

Async planning:

```json
{"type":"plan","kind":"joint_position_target","robot":"left","env_ids":[0],"joint_positions":[[0.2,0,0,0,0,0,0]],"duration_s":0.5}
{"type":"plan","kind":"joint_delta_pos","robot":"left","env_ids":[0],"joint_deltas":[[0.01,0,0,0,0,0,0]],"duration_s":0.5}
{"type":"plan","kind":"task_space_line","robot":"left","env_ids":[0],"target_offset":[0,0,0.05],"duration_s":1.0}
{"type":"planner_status","wait_timeout_s":0.1}
{"type":"cancel_plan","request_id":"plan-123"}
```

All tiled planning requests use `type:"plan"` and put the motion type in `kind`. The tiled protocol no longer accepts old top-level motion messages such as `cspace_goal`, `cspace_delta`, `task_space_line`, `task_space_arc`, `specified_path`, `plan_queue`, top-level `moves`, `move_type`, or the `side` robot alias. Use `robot` or `robots`.

Use `--planner-backend linear` for single-segment joint-space targets. Use `--planner-backend cumotion` for task-space line/arc and specified path support.

## Hand Overlay

Trajectory loading and ready async planner results can carry hand overlays:

- `before`: play hand trajectory before arm motion.
- `sync`: play hand trajectory in sync with arm motion.
- `after`: play hand trajectory after arm motion.

Overlays only write explicit hand joint names that exist in command space.
