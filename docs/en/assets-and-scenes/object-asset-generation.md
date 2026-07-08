# Object Asset Generation

Language: [English](object-asset-generation.md) | [中文](../../zh-CN/资产与场景/物体资产生成指南.md)

This document explains how to generate object USD/USDA assets used by env profiles.

## Capsule Rope

The flexible rope builder lives at:

```text
tools/object_assets/flexible/rope/
```

Run:

```bash
PYTHONPATH=src python tools/object_assets/flexible/rope/build_asset.py
```

The builder reads its YAML config, creates capsule/cuboid link geometry, adds the required USD/PhysX schema, and writes the generated asset under `assets/flexible_env_objects/`.

Env profiles reference the generated asset through object profiles in `configs/objects/`.

## T Block

The rigid T block builder lives at:

```text
tools/object_assets/rigid/tblock/
```

Run:

```bash
PYTHONPATH=src python tools/object_assets/rigid/tblock/build_asset.py
```

The generated `TblockV1_default` follows the object naming rules described in [Asset Naming](../configuration-and-naming/asset-naming.md).

## Runtime Boundary

Runtime scripts reference already-generated USD/USDA files. They do not rebuild object assets automatically.

If you edit a builder config, rerun the corresponding `build_asset.py` before launching a runtime that uses that object.

## Preview

Use Isaac Sim to preview generated assets before adding them to a scene. See [USD Asset Preview](usd-asset-preview.md).

## Configuration Checks

Run lightweight config checks after changing object profiles:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_env_profile_directory.py \
  tests/test_controller_configs.py \
  tests/test_robot_loader_import_config.py \
  tests/test_sensor_camera_config.py -q
```
