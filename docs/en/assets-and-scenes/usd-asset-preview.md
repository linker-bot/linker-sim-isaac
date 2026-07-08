# USD Asset Preview

Language: [English](usd-asset-preview.md) | [中文](../../zh-CN/资产与场景/USD%20资产预览指南.md)

This document lists lightweight ways to preview generated `.usd` / `.usda` assets in Isaac Sim.

## Basic Preview

Use the relevant builder first:

```bash
PYTHONPATH=src python tools/object_assets/rigid/tblock/build_asset.py
```

Then open the generated USD/USDA in Isaac Sim or through a small preview script if one is available in your environment.

## What To Inspect

- The stage contains the expected root prim.
- Visual geometry is correctly positioned and scaled.
- Collision geometry exists and roughly matches the intended shape.
- Rigid bodies or joints have the expected mass, inertia, and solver settings.
- Materials and colors make the asset distinguishable in the scene.
- The asset can be referenced by the corresponding object profile.

## Troubleshooting

Missing mesh
: Check relative mesh paths inside the USD/USDA and the generated asset directory.

Unexpected scale
: Check units and authored transforms in the builder config.

Object falls or explodes
: Check rigid body mass/inertia, collision approximation, solver iteration counts, and whether the scene intended the object to be static.
