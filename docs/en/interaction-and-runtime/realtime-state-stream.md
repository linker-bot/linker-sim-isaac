# Realtime State Stream

Language: [English](realtime-state-stream.md) | [中文](../../zh-CN/交互与运行/实时状态流使用说明.md)

This document explains how to enable realtime state streams for single-arm and dual-arm interactive runtimes, and how to read the output.

## Boundary

The state stream is observation-only. Motion commands still go through stdin JSONL, TCP JSONL, or WebSocket JSON.

Supported:

- Foxglove live server.
- Foxglove MCAP recording.
- Standard `/joint_states`.
- Object markers on `/scene`.
- Full JSON state snapshot on `/linkerbot/state`.
- Optional commanded, measured, and applied effort arrays.
- Optional env runtime object root poses.

Not supported:

- Sending motion commands through Foxglove.
- Reusing a command transport port as a Foxglove live port.
- Reading Isaac articulation, PhysX views, or USD stage directly from a background thread.

## Start Commands

Single-arm state stream, recommended port `8765`:

```bash
PYTHONPATH=src python scripts/single_arm_interactive.py \
  --env scene1 \
  --gui \
  --foxglove-live-port 8765 \
  --state-include-objects
```

Dual-arm state stream, recommended port `8766`:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --state-rate-hz 60 \
  --state-include-objects
```

Write MCAP at the same time:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-mcap-path logs/interactive_state.mcap \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

## CLI Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--state-rate-hz` | `60.0` | State sampling rate. Values `<=0` disable sampling. A live port or MCAP path is still required to start output. |
| `--state-include-objects` | off | Sample env runtime object root poses. |
| `--state-include-efforts` | off | Read commanded, measured, and applied efforts. |
| `--foxglove-live-host` | `127.0.0.1` | Live server bind host. |
| `--foxglove-live-port` | unset | State stream live server port. |
| `--foxglove-mcap-path` | unset | State stream MCAP output path. |
| `--foxglove-joint-effort-field` | `none` | Which effort semantic to write into standard `JointStates.effort`: `none`, `commanded`, `measured`, or `applied`. |

## Topics

| Topic | Encoding | Purpose |
| --- | --- | --- |
| `/joint_states` | Foxglove `JointStates` protobuf | Curves and joint-state panels. |
| `/scene` | Foxglove `SceneUpdate` protobuf | Object markers in the 3D panel. |
| `/linkerbot/state` | JSON | Full state snapshot with step/time, robot states, effort arrays, object poses, and phase. |

Full topic details and troubleshooting are in [Foxglove Data](../telemetry-and-sensors/foxglove-data.md).

## Effort Semantics

Do not mix the three effort fields:

- `commanded_efforts`: effort values the Python controller wanted to send.
- `measured_efforts`: generalized forces measured or computed by PhysX.
- `applied_efforts`: actuator/drive effort currently recorded by Isaac articulation runtime.

The standard Foxglove `JointStates` schema has only one `effort` field, so `/joint_states` can only carry one selected semantic. Read `/linkerbot/state` when you need all three.
