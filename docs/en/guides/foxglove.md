# Foxglove Quick Reference

Language: [English](foxglove.md) | [中文](../../zh-CN/guides/foxglove.md)

Install the optional Foxglove SDK with `uv sync --all-extras`. Built-in live
servers bind only to loopback and provide no authentication or TLS.

## Single Scene

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --gui \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --foxglove-live-port 8766 \
  --state-rate-hz 30 --state-include-objects
```

In Foxglove, open a connection to `ws://127.0.0.1:8766`. Use Plot or Raw
Messages for `/joint_states`, Raw Messages for `/linkerbot/state`, and a 3D
panel for `/scene` when `include_scene_markers` is enabled in the runtime
profile.

JointStates names use `<robot_label>/<joint_name>`. The JSON state preserves
robot boundaries and contains session `robot_id` plus stable `label`.

## Tiled Scene

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --env scene3_tiled \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --foxglove-live-port 8767 \
  --telemetry-env-ids 0,2 --telemetry-rate-hz 10
```

Connect to `ws://127.0.0.1:8767`. The JSON state keeps both requested envs;
standard JointStates and SceneUpdate use only `telemetry.primary_env_id`.

## Topic Quick Reference

Actual state topic names come from `runtime.telemetry.topics`. Camera prefixes
come from each `sensors.cameras.<name>.output.foxglove_topic_prefix`.

| Topic | Encoding | Content |
| --- | --- | --- |
| `/joint_states` | Foxglove `JointStates` | Standard joint arrays |
| `/scene` | Foxglove `SceneUpdate` | Scene/runtime markers |
| `/linkerbot/state` | JSON | Full Single Scene or selected-env state |
| `<camera-prefix>/rgb` | `RawImage`, `rgb8` | RGB frame |
| `<camera-prefix>/depth` | `RawImage`, `32FC1` | Float32 depth frame |
| `<camera-prefix>/info` | JSON | Metadata for every camera modality |

Segmentation modalities publish `/info` metadata but do not create a RawImage
channel. Camera live endpoints are configured in the env profile and can use a
different port from state live.

## Endpoint And Recording Boundaries

Control TCP JSONL, control WebSocket, state Foxglove live, and camera Foxglove
live are different protocols. Assign different ports. A Foxglove connection is
observation-only and cannot execute control JSON.

`--foxglove-mcap-path` records state telemetry. Camera MCAP paths are configured
per camera in the env profile. Live and MCAP can be enabled together.

For complete state schemas and sampling behavior, see
[Realtime State Stream](telemetry.md). For camera configuration and modality
behavior, see [Camera Types And Sensors](cameras.md). Existing-file, queue,
quota, and shutdown semantics are centralized in
[Outputs And Persistence](../reference/outputs.md).
