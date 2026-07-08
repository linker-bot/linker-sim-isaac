# Foxglove Data

Language: [English](foxglove-data.md) | [中文](../../zh-CN/传感器与遥测/Foxglove%20数据使用说明.md)

This document explains how single-arm, dual-arm, and tiled interactive runtimes publish simulation state through Foxglove live servers and MCAP, and how simulated sensor cameras publish RGB/depth images.

For the quick single-arm/dual-arm state-stream entrypoint, see [Realtime State Stream](../interaction-and-runtime/realtime-state-stream.md). For tiled telemetry commands, see [Tiled Usage And Command Format](../tiled-environments/tiled-usage-and-command-format.md).

## Output Types

The current implementation has two Foxglove output families:

- Interactive state streams: started by CLI parameters on `scripts/single_arm_interactive.py`, `scripts/dual_arm_interactive.py`, or `scripts/tiled_env_interactive.py`.
- Sensor camera images: enabled by env profile `sensors.cameras.<name>.output`.

Supported:

- Foxglove live server.
- Foxglove MCAP files.
- Robot joint position, velocity, and differential acceleration.
- Optional commanded, measured, and applied effort arrays.
- Optional env runtime object root poses.
- Sensor camera RGB `RawImage`.
- Sensor camera depth `RawImage`.
- Local RGB/depth/metadata output.

Not supported:

- Sending motion commands through Foxglove.
- Reusing command TCP/WebSocket ports as Foxglove live ports.
- Reading Isaac stage/articulation data from background threads.
- Overriding sensor camera output paths or ports through interactive script CLI. Camera output is env-profile based.
- Combining RGB and depth into one video topic.

## Recommended Ports

| Output | Port | Notes |
| --- | --- | --- |
| Single-arm state stream | `8765` | Single-arm interactive runtime. |
| Dual-arm state stream | `8766` | Dual-arm interactive runtime. |
| Tiled telemetry | `8767` | Tiled interactive telemetry. |
| Camera RawImage | `8770` and above | Configured in env profile camera output. |

Do not reuse these ports for command transports. Example command ports are `9001` for dual-arm TCP JSONL, `9002` for dual-arm WebSocket JSON, and `9003` for tiled TCP JSONL.

## Start Live Server

Dual-arm example:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --state-rate-hz 60 \
  --state-include-objects
```

With effort:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

Foxglove Desktop:

```text
Open connection -> Foxglove WebSocket -> ws://127.0.0.1:8766
```

## MCAP

MCAP-only:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-mcap-path logs/interactive_state.mcap \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

Live and MCAP can be enabled at the same time.

State stream topics:

| Topic | Encoding | Purpose |
| --- | --- | --- |
| `/joint_states` | Foxglove `JointStates` protobuf | Joint curves and panels. |
| `/scene` | Foxglove `SceneUpdate` protobuf | Object root-position markers. |
| `/linkerbot/state` | JSON | Full project state snapshot. |

## Sensor Camera Images

Camera output is configured in env profile, not by `--foxglove-live-port`:

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      parent_prim_path: /World
      prim_path: /World/WorldRGBD
      modalities: [rgb, depth]
      output:
        save_dir: logs/cameras/world_rgbd
        foxglove_topic_prefix: /cameras/world_rgbd
        foxglove_live_host: 127.0.0.1
        foxglove_live_port: 8770
```

Camera topics:

| Topic | Encoding | Purpose |
| --- | --- | --- |
| `/cameras/world_rgbd/rgb` | Foxglove `RawImage`, `rgb8` | Color image. |
| `/cameras/world_rgbd/depth` | Foxglove `RawImage`, `32FC1` | Float32 depth image. |
| `/cameras/world_rgbd/info` | JSON | Frame index, shape, dtype, intrinsics, world pose. |

Local output structure:

```text
logs/cameras/world_rgbd/
├── metadata.jsonl
├── rgb/
│   └── 000000.ppm
└── depth/
    └── 000000.npy
```

## Effort Fields

- `commanded_efforts`: controller-side desired effort.
- `measured_efforts`: PhysX generalized force.
- `applied_efforts`: Isaac articulation actuator/drive effort.

Foxglove standard `JointStates` has only one `effort` field, so use `/linkerbot/state` when you need all three.

## Troubleshooting

`handshake failed` usually means a client connected with the wrong protocol. In Foxglove, choose `Foxglove WebSocket` and use `ws://127.0.0.1:<port>`.

No data:

- Check the runtime printed `SINGLE_ARM_INTERACTIVE_READY`, `DUAL_ARM_INTERACTIVE_READY`, or `TILED_INTERACTIVE_READY`.
- Check `--foxglove-live-port` or `--foxglove-mcap-path`.
- Check `--state-rate-hz > 0`.
- Check port conflicts.
- For cameras, check `sensors.cameras.<name>.enabled` and `output.foxglove_live_port`.
