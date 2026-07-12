# Asset Naming

Language: [English](naming.md) | [中文](../../zh-CN/development/naming.md)

This document defines the identities used by assets, profiles, scene instances,
runtime protocols, sensors, and outputs. These names are related but are not
interchangeable.

## General Principles

- Use stable ASCII names and underscores for repository-owned asset and entity
  names. Avoid spaces, temporary words such as `new` or `final`, and hyphens
  that an importer may normalize to underscores.
- Preserve an upstream URDF/MJCF name when changing it would break a mesh,
  joint, controller, or planning reference.
- Keep physical hardware identity separate from reusable profile identity,
  per-scene instance identity, and session-only numeric IDs.
- Keep every joint, link, body, TCP, camera, path, and topic name consistent
  with the layer that owns it; do not infer one identity from another.

## Identity Layers

| Identity | Owner | Example | Contract |
| --- | --- | --- | --- |
| Asset | `assets/` directory and primary file prefix | `AR5V2_L6V1_L` | Physical model and variant |
| Profile | Selector under `configs/<group>/` | `ar5v2_l6v1_l` | Reusable validated configuration |
| Robot instance | Env `robots[].label` | `left_arm` | Stable scene and snapshot matching identity |
| Robot session ID | Env `robots[]` list order | `robot_id: 0` | Dense public selector for the current process only |
| Object instance | Env `objects[].name` | `Tblock` | Stable scene object identity |
| Camera | Env `sensors.cameras` mapping key | `world_rgbd` | Camera frame and output namespace |

## Profile Names

For ordinary profiles, the selector is the YAML file stem under
`configs/<group>/`. A directory-style env profile uses its directory name and
loads `base.yaml`. Profile selectors are one safe stem, not a path.

Examples:

```text
configs/envs/scene1.yaml             -> scene1
configs/envs/scene3_tiled/base.yaml  -> scene3_tiled
configs/robots/ar5v2_l6v1_l.yaml     -> ar5v2_l6v1_l
configs/logging/default_logger.yaml -> default_logger
```

CLI fields such as `--runtime-profile`, `--env`, and profile references inside
YAML use these selectors. Do not pass `configs/.../*.yaml` where a profile name
is required. Profile names are case-sensitive filesystem identities; preserve
the spelling already present in the checkout.

## Hardware And Asset Names

`AR5V2` and `L6V1` are the actual arm and hand hardware family/version
identities. `L` and `R` are physical variants, not runtime selectors:

```text
AR5V2_L
AR5V2_R
L6V1_L
L6V1_R
AR5V2_L6V1_L
AR5V2_L6V1_R
```

The current asset tree keeps the hardware identity in the directory and primary
file prefix:

```text
assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
assets/single_system/hand/L6V1_L/L6V1_L.xml
assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
assets/single_system/arm/AR5V2_L/AR5V2_L_curobo.yml
```

Robot profile stems use the current configuration convention, for example
`ar5v2_l`, `l6v1_l`, and `ar5v2_l6v1_l`. Do not add an unrecognized asset
revision suffix or treat `V1` / `V2` as a configuration schema version.

## Joint, Link, Body, And TCP Names

Repository robot entities retain a complete hardware/category prefix:

```text
AR5V2_L_arm_joint_1
AR5V2_L_arm_link7
AR5V2_L_arm_flan_link
L6V1_L_hand_base_link
L6V1_L_hand_thumb_cmc_roll
L6V1_L_hand_index_dip
L6V1_L_hand_couple_index
```

The `flan_link` spelling is part of the actual asset identity and must not be
silently normalized. Joint, link, body, actuator, mesh, mimic/equality, and TCP
names must remain consistent between:

- MJCF/URDF files.
- Mesh filenames and references.
- Robot profile joint and rigid-body groups.
- controller active/follower joints.
- cuRobo URDF, robot YAML, collision configuration, and tool frames.
- Motion targets, snapshots, tests, and command examples.

Custom TCP frames are named in the robot profile, for example
`AR5V2_L_tool_tcp` or `AR5V2_L_pinch_tcp`, and are defined relative to the
configured `flange_frame`. cuRobo may materialize a derived URDF with those
fixed links below the resolved cuRobo cache directory. The root comes from
`runtime.paths.cache_root`, `LINKERBOT_SIM_CACHE_ROOT`, `XDG_CACHE_HOME`, or the
user cache directory, in that order. The derived file is not a primary asset
and must not replace the checked-in URDF.

## Robot Runtime Identity

Each env `robots[]` row selects a reusable `robot_profile` and defines a scene
instance. `label` must be unique and match `[A-Za-z0-9_]+`; if omitted it becomes
`<robot_profile>_<robot_id>`. Its default USD path is
`/World/Robots/<label>`.

`robot_id` is generated from zero-based `robots[]` order. IDs are dense and are
not configurable in YAML. Public Single Scene and Tiled Scene control protocols use the
session `robot_id`; discover it from `status` after every process start or env
reorder. Do not cache it as persistent hardware identity and do not select a
robot by `L` / `R`.

The label remains the stable internal, telemetry, and snapshot matching key.
When a flattened output needs globally unique joint names, prefix the unchanged
asset joint name with that label:

```text
left_arm/AR5V2_L_arm_joint_1
right_arm/AR5V2_R_arm_joint_1
```

## Object Identity

Object asset, profile, and scene instance names are separate. Current examples
include:

```text
asset:    workstationV1_armbase, capsuleropeV1_default, TblockV1_default
profile:  workstation_armbase, capsule_rope, TblockV1_default
instance: workstation, rope, Tblock
```

An env `objects[].name` is unique within the scene and is used by snapshots and
Tiled per-env pose overrides. It starts with a letter or underscore and then
uses letters, digits, or underscores. `object_profile` is the reusable profile
selector. `runtime_handle` is an optional interaction alias; it does not rename
the profile, asset, or USD prim. Instance names, runtime handles, and effective
prim paths must be unique; a handle cannot collide with another object's name.

The env instance owns `prim_path` and `root_pose`. If `prim_path` is omitted,
the runtime derives `/World/Objects/<name>`. Asset source/path, import options,
physics, planning collision, and a supported asset-internal `root_path` remain
owned by `configs/objects/<object_profile>.yaml`.

## Camera And Output Names

The key under `sensors.cameras` is the camera name carried by `CameraFrame`,
offline metadata, and output routing. It must be nonempty and cannot contain a
path separator. `prim_path` is a separate absolute USD identity; when
`parent_prim_path` is set, the camera prim must be under that parent.

For Tiled cameras, per-env pose overrides refer to the base camera key. Runtime
expansion gives each camera an `env_NNN_<name>` identity and appends the same
`env_NNN` segment to any configured local `save_dir` and Foxglove topic prefix:

```text
base camera: world_rgbd
runtime camera: env_000_world_rgbd
local output: logs/cameras/world_rgbd/env_000/
topic prefix: /cameras/world_rgbd/env_000
```

Within one camera output directory, `metadata.jsonl` indexes deterministic
payload names such as `rgb/000000.png` and `depth/000000.npz`. Foxglove camera
channels append `/rgb`, `/depth`, and `/info` to the configured prefix.

Other output names keep their configured owners. Single Scene joint CSV treats the
configured path as a template and adds the session robot ID and label, for
example `run.0.left_arm.csv`. State topic names come directly from
`runtime.telemetry.topics`; they are not inferred from robot or asset names. See
[Outputs And Persistence](../reference/outputs.md) for all destination owners.

## Validation

Validate the full profile graph without starting Isaac:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_tiled_scene
```

Run identity-sensitive tests after a rename:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_system_configs.py \
  tests/test_robot_profile_schema.py \
  tests/test_object_instances.py \
  tests/test_sensor_camera_config.py \
  tests/test_tiled_cameras.py -q
```

Useful stale-name scan:

```bash
rg "AR5-V2|L6-V1|capsule-rope" assets configs src scripts tests README.md docs
```

After renaming, verify:

- Primary URDF/MJCF/XML files parse and every referenced mesh exists.
- Asset paths, profile references, env labels/names, and prim paths remain unique.
- Joint groups, controller mappings, trajectory targets, TCP frames, cuRobo
  descriptions, snapshots, and examples are updated together.
- Camera keys, Tiled per-env overrides, output directories, and topic prefixes
  still resolve to the intended namespace.
