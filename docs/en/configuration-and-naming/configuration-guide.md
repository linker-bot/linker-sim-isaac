# Configuration Guide

Language: [English](configuration-guide.md) | [中文](../../zh-CN/配置与命名/配置使用说明.md)

This document explains how the YAML profiles under `configs/` are split and how runtime entrypoints select them. For complete field templates, see the `example.yaml` files in each config directory.

## General Rules

Profile arguments use file stems, not YAML paths:

```bash
PYTHONPATH=src python scripts/single_arm_interactive.py \
  --env scene1 \
  --cumotion-profile default \
  --logging-profile default_logger
```

This loads:

```text
configs/envs/scene1.yaml
configs/cumotion/default.yaml
configs/logging/default_logger.yaml
```

Profile names must be simple file stems and must not contain `/` or `\`. Relative paths in YAML files, such as `assets/...` and `logs/...`, are resolved relative to the repository root.

## Directory Responsibilities

| Directory | Purpose | Selected By |
| --- | --- | --- |
| `configs/envs/` | Scene profile: world frequency, gravity, solver, lights, GUI viewport, robot instances, object instances. | `--env <name>` |
| `configs/robots/` | Robot profile: one Isaac articulation, physical overrides, cuMotion model resources. | Referenced by env profiles. |
| `configs/objects/` | Object profile: reusable object asset paths, import options, runtime physics. | Referenced by env profiles. |
| `configs/controllers/` | Arm/hand control modes, gains, limits, and mimic follower drives. | Fixed runtime files. |
| `configs/cumotion/` | cuMotion algorithm profile: IK tolerances, planner pipeline, graph search, trajectory generation. | `--cumotion-profile <name>` |
| `configs/logging/` | CSV joint-tracking logger profile. | `--logging-profile <name>` |

## Env Profiles

An env profile describes how a scene is assembled. It references robot and object profiles instead of embedding their asset paths.

Typical shape:

```yaml
env:
  name: scene1
  gravity_z: -9.81
  add_ground: false
  ground_height: 0.0
  physics_frequency: 240.0
  render_frequency: 60.0

robots:
  single:
    robot_profile: ar5v2_l6v1_l
    root_pose:
      xyz: [0.0, 0.09, 0.0]
      rpy: [-1.5707, 0.0, 0.0]

objects:
  - name: workstation_armbase
    object_profile: workstation_armbase
    root_pose:
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
```

Important boundaries:

- `env.ground_height` is the z height of the Isaac default ground plane in meters. It only takes effect when `env.add_ground` is `true`.
- `root_pose` belongs to the env because one robot/object profile can be placed differently in different scenes.
- `robots.single` is used by single-arm entrypoints; `robots.dual.left/right` is used by dual-arm entrypoints.
- `objects[]` stores instance name, object profile name, optional runtime handle, and pose only.

## Robot Profiles

A robot profile describes one Isaac articulation plus cuMotion planning resources:

```yaml
robot:
  name: ar5v2_l6v1_l
  asset_type: mjcf
  asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
  prim_path: /World/AR5V2_L6V1_L

cumotion:
  xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
  flange_frame: AR5V2_L_arm_flan_link
```

Do not put robot placement in a robot profile. Put it under the env robot instance.

## Object Profiles

Object profiles describe reusable environment objects. Current runtime objects support:

- `kind: rigid` with `source: usd` or `source: urdf`.
- `kind: dynamic_chain` with `source: usd`, currently used for capsule rope assets.

Environment objects do not support `source: mjcf`; MJCF is only a robot asset type.

## Controller Profiles

Controller configuration is currently fixed to:

```text
configs/controllers/arm_controller.yaml
configs/controllers/hand_controller.yaml
```

Use `--control-mode` to select `position`, `velocity`, or `effort`.

## cuMotion Profiles

cuMotion profiles contain algorithm defaults only. Keep robot resources such as `xrdf_path`, `urdf_path`, and `flange_frame` in robot profiles.

Merge boundary:

```text
configs/cumotion/*.yaml  <  configs/robots/*.yaml  <  runtime motion arguments
```

## Logging Profiles

Logging profiles control CSV enablement, output path, flush interval, sample decimation, and column groups.

If you need a different output path or sampling interval, copy a logging profile and select it with `--logging-profile <name>`.

## Common Checks

Config-only checks:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_env_profile_directory.py \
  tests/test_controller_configs.py \
  tests/test_robot_loader_import_config.py -q
```

Dual-arm motion semantics:

```bash
PYTHONPATH=src python -m pytest tests/test_dual_arm_motion_test.py -q
```
