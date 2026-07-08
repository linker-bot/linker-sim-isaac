# Interactive Simulation

Language: [English](interactive-simulation.md) | [中文](../../zh-CN/交互与运行/交互式仿真使用说明.md)

This document explains how to start single-arm, dual-arm, and tiled interactive simulations, and how to send JSON motion commands.

## Start

Single-arm interactive runtime for one AR5+L6. Single-arm messages may omit `side`; the parser defaults to `left`.

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/single_arm_interactive.py \
  --env scene1 \
  --gui \
  --foxglove-live-port 8765 \
  --state-include-objects
```

Common JSONL:

```json
{"type":"hand","joint_positions":[0.05,0.05,0.05,0.05,0.05,0.05],"duration_s":0.2}
{"type":"cspace_delta","joint_deltas":[0.01,0,0,0,0,0,0],"duration_s":0.5}
{"type":"ik_offset","offset":[0,0.01,0],"orientation_mode":"current","duration_s":0.5}
{"type":"quit"}
```

Dual-arm GUI runtime:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py --gui --hold
```

Ready messages:

```text
SINGLE_ARM_INTERACTIVE_READY
DUAL_ARM_INTERACTIVE_READY
```

## Tiled Runtime

The tiled runtime does not reuse the old dual-arm interactive execution queue. It starts a real Isaac tiled scene and writes batched joint targets through articulation views.

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --default-decimation 1
```

Common tiled JSONL:

```json
{"type":"status"}
{"type":"step","kind":"joint_delta_pos","env_ids":[1],"robots":["left"],"values":[[0.01,0,0]],"decimation":1}
{"type":"get_state","env_ids":[1],"fields":["robots.left.joint_positions","episode_steps"]}
{"type":"reset","env_ids":[1]}
{"type":"quit"}
```

Tiled `joint_position_target` and `joint_delta_pos` accept short vectors. The number of columns in `values` is the number of leading command joints written for the selected robot. For example, `values:[[0.01,0,0]]` controls only the first three command joints. Send seven columns when you want to command seven arm joints, for example `values:[[0.01,0,0,0,0,0,0]]`. Use `{"type":"status"}` to inspect the full command joint list.

The interactive CLI no longer exposes `--robots` or `--num-envs`. Robot selection is done per message with `robot` / `robots`; env selection is done with `env_ids`; the env count comes from YAML `tiled.num_envs`.

## Transports

### stdin JSONL

Enabled by default. Each line is one JSON object.

### TCP JSONL

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --tcp-jsonl-host 127.0.0.1 \
  --tcp-jsonl-port 9001
```

Client example:

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 9001
```

### WebSocket JSON

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --websocket-host 127.0.0.1 \
  --websocket-port 9002
```

WebSocket messages are JSON objects. This transport is useful for browser control panels.

## Foxglove Telemetry

Single-arm state stream: use port `8765`.
Dual-arm state stream: use port `8766`.
Tiled telemetry: use port `8767`.

State streams are observation-only. Command TCP/WebSocket transports still handle motion commands.

## Common Motion Rules

- Length unit: m.
- Angle unit: rad.
- RPY orientation order: `[roll, pitch, yaw]`.
- Quaternion order: `wxyz`.
- `side` is `left` or `right`.
- `duration_s` is the execution duration for this motion.
- `tcp_frame_name` may be omitted; the robot YAML default TCP is used.
- Motion commands execute serially even if several clients submit commands.

## Common Motion Types

| Type | Purpose |
| --- | --- |
| `hand` | Single-side hand joint target. |
| `dual_hand` | Synchronized hand targets for two sides. |
| `cspace_goal` | Absolute arm C-space target. |
| `cspace_delta` | Delta in arm C-space. |
| `ik_pose` | Absolute TCP pose target. |
| `ik_offset` | TCP offset from current pose. |
| `task_space_line` | TCP line segment. |
| `task_space_arc` | TCP arc segment. |
| `specified_path` | Explicit C-space, task-space, or composite path request. |
| `moves` | Queue several motion specs in order. |
| `reset` | Reset runtime state, call `world.reset()`, and clear stream/camera sampling caches. |

## Return Events

Accepted:

```json
{"event":"accepted","id":"cmd-1","state":"pending","queue_index":0}
```

Running:

```json
{"event":"running","id":"cmd-1","state":"running"}
```

Done:

```json
{"event":"done","id":"cmd-1","state":"done","steps":240}
```

Failed:

```json
{"event":"failed","id":"cmd-2","state":"failed","error":"..."}
```

## Troubleshooting

If the robot does not keep moving, confirm the ready message was printed. For GUI sessions, use `--hold` when you want the process to continue stepping after stdin EOF.

If WebSocket startup fails, make sure the Python environment has the `websockets` package installed.
