# Asset Naming

Language: [English](asset-naming.md) | [中文](../../zh-CN/配置与命名/资产命名规范.md)

This document summarizes the naming rules used for assets, joints, links, bodies, and configuration profiles.

## General Principles

- Use stable, machine-readable names for assets and profiles.
- Keep physical side (`L` / `R`, `left` / `right`) explicit.
- Keep the same conceptual name across assets, configs, tests, and docs.
- Avoid spaces in file and directory names under assets and configs.
- Preserve upstream names in imported URDF/MJCF where changing them would break mesh or joint references.

## Profile Names

Profile names are file stems under `configs/<group>/`.

Examples:

```text
configs/envs/scene1.yaml        -> scene1
configs/cumotion/default.yaml   -> default
configs/logging/default_logger.yaml -> default_logger
```

Do not pass a YAML path to CLI options that expect a profile name.

## Robot Names

Robot profile names should identify the hardware combination and side:

```text
ar5v2_l6v1_l
ar5v2_l6v1_r
```

The display or asset name may keep the original uppercase convention, for example `AR5V2_L6V1_L`.

## Joint And Link Names

Joint and link names must remain consistent between:

- MJCF/URDF files.
- robot YAML joint groups.
- controller active/follower joints.
- cuMotion XRDF/URDF descriptions.
- TCP frame definitions.
- tests and command examples.

For dual-arm state streams, joint names may be prefixed with logical side, such as:

```text
left/AR5V2_L_arm_joint_1
right/AR5V2_R_arm_joint_1
```

## Object Names

Object profile names describe reusable assets:

```text
workstation_armbase
capsuleropeV1_default
TblockV1_default
```

Env object instances use `objects[].name` and can place the same profile in different scenes.

## Validation

Useful scans:

```bash
rg "AR5-V2|L6-V1|capsule-rope" assets configs src scripts tests README.md docs
```

After renaming, verify:

- URDF/MJCF/XML files still parse.
- Referenced mesh files still exist.
- Joint groups, trajectory targets, TCP frames, and IK descriptions are updated together.
- Config and motion tests pass.
