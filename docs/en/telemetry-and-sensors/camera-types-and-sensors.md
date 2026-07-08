# Camera Types And Sensors

Language: [English](camera-types-and-sensors.md) | [中文](../../zh-CN/传感器与遥测/相机类型与传感器设置.md)

This document explains the difference between GUI viewport camera settings and simulated sensor cameras.

## Terminology

| Type | Recommended Name | Config Location | Purpose | Produces Image Data |
| --- | --- | --- | --- | --- |
| GUI view | viewport view / GUI viewport | `visuals.viewport` | Set the Isaac GUI viewing angle. | No |
| Simulated camera | sensor camera / RGB-D camera | `sensors.cameras` | Scene sensor that outputs RGB, depth, etc. | Yes |

The GUI viewport is for human observation only. It does not participate in control, planning, logging, or visual algorithm inputs.

The historical `visuals.camera` field has been renamed to `visuals.viewport`. Current config parsing rejects `visuals.camera` and asks users to migrate.

## GUI Viewport

```yaml
visuals:
  viewport:
    enabled: true
    eye: [1.35, -1.65, 1.05]
    target: [0.0, -0.1, 0.42]
    prim_path: /OmniverseKit_Persp
```

Fields:

| Field | Meaning |
| --- | --- |
| `enabled` | Whether to set the GUI viewport after startup. |
| `eye` | View eye point in world coordinates, m. |
| `target` | View target in world coordinates, m. |
| `prim_path` | Camera prim path used by the Isaac viewport. |

## Sensor Camera

Sensor cameras are configured under env profile `sensors.cameras`:

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      parent_prim_path: /World
      prim_path: /World/WorldRGBD
      pose:
        xyz: [0.5, -0.6, 0.8]
        rpy: [0.0, 0.7, 0.0]
      resolution: [640, 480]
      frequency: 30.0
      modalities: [rgb, depth]
      clipping_range: [0.01, 5.0]
      output:
        save_dir: logs/cameras/world_rgbd
        foxglove_topic_prefix: /cameras/world_rgbd
        foxglove_live_host: 127.0.0.1
        foxglove_live_port: 8770
```

Fields:

| Field | Meaning |
| --- | --- |
| `enabled` | Whether to create and sample this camera. |
| `parent_prim_path` | Optional parent prim. `/World` means fixed world camera; robot link path means wrist/tool camera. |
| `prim_path` | Absolute USD prim path for the camera. |
| `pose.xyz` | Translation relative to parent prim, or world coordinates without a parent. |
| `pose.rpy` | Orientation relative to parent prim, rad. |
| `resolution` | Image width and height. |
| `frequency` | Sampling frequency in Hz. |
| `modalities` | Output types, for example `rgb` and `depth`. |
| `output.save_dir` | Optional local output directory. |
| `output.foxglove_live_port` | Optional Foxglove live port. Camera ports should start at `8770`. |
| `output.foxglove_mcap_path` | Optional Foxglove MCAP output path. |

The camera live port is configured by the env profile. It is not the same as the interactive state-stream `--foxglove-live-port`.

In tiled envs, `sensors.cameras` keeps shared camera settings. Each child env can
override the camera position with `cameras.<name>.pose` in `envs/env_XXX.yaml`.
The runtime creates one camera per env and appends an `env_000` suffix to local
output directories and Foxglove topic prefixes.

## Output

Local output example:

```text
logs/cameras/world_rgbd/
├── metadata.jsonl
├── rgb/
│   └── 000000.ppm
└── depth/
    └── 000000.npy
```

Foxglove topics:

| Topic | Encoding |
| --- | --- |
| `/cameras/world_rgbd/rgb` | `RawImage`, `rgb8` |
| `/cameras/world_rgbd/depth` | `RawImage`, `32FC1` |
| `/cameras/world_rgbd/info` | JSON metadata |

## GUI And Headless

Sensor cameras do not depend on `--gui`. When camera output is configured, the runtime drives render updates so camera annotators can produce images in headless mode too. Empty startup frames are skipped.

## Common Issues

`visuals.camera was renamed to visuals.viewport`
: Update env profiles from `visuals.camera` to `visuals.viewport`.

RGB does not show in Foxglove
: Check local `logs/cameras/<name>/rgb/` first. If files exist, choose the RGB topic in the Foxglove Image panel.

Depth looks black
: Depth is `32FC1`; tune Image panel depth/color scale min/max.

Wrist camera does not follow the robot
: Check `parent_prim_path` points to the real robot link and `prim_path` is under that link path.
