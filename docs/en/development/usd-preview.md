# USD Asset Preview

Language: [English](usd-preview.md) | [中文](../../zh-CN/development/usd-preview.md)

This document lists lightweight ways to preview generated `.usd` / `.usda` assets in Isaac Sim.

## Basic Preview

Use the relevant builder first:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python \
  tools/object_assets/rigid/tblock/build_asset.py
```

Then start the GUI:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
isaacsim isaacsim.exp.full
```

Open the generated USD/USDA with `File -> Open`. This repository does not contain
an `open_stage.py` helper, and the asset path must not be passed as Kit's
experience argument.

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
