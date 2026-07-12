# Object Asset Generation

Language: [English](object-assets.md) | [中文](../../zh-CN/development/object-assets.md)

This document explains how to generate object USD/USDA assets used by env profiles.

## Complete Builder Option Table

Both supported `build_asset.py` entrypoints accept the same options:

| Option | Default | Contract |
| --- | --- | --- |
| `--help` | n/a | Print argparse help and exit before launching Kit. |
| `--config PATH` | The entrypoint's colocated `config.yaml` | Load the asset-generation YAML from the supplied path. |
| `--output PATH` | Omitted | Override `object.asset_path`; when omitted, write to the path declared by the generation config. |

Paths are resolved from the checkout. A successful rope build prints
`BUILD_CAPSULE_ROPE_ASSET_OK`; a successful T block build prints
`BUILD_T_BLOCK_ASSET_OK`. Both commands require accepted Kit EULA terms and close
their headless `SimulationApp` after writing the asset.

## Generation And Runtime Boundaries

Object assets cross four separately owned layers:

| Layer | Location | Owns |
| --- | --- | --- |
| Generation config | `tools/object_assets/<rigid|flexible>/<name>/config.yaml` | Intrinsic geometry, mass, damping, joints, collision filtering, and visual color |
| Builder entrypoint | `tools/object_assets/<rigid|flexible>/<name>/build_asset.py` | Headless Kit startup and creation of a fresh USD/PhysX asset |
| Runtime object profile | `configs/objects/*.yaml` | Asset source/path, import options, runtime physics, planning collision, and an asset-internal `root_path` where supported |
| Scene instance | `configs/envs/*.yaml` `objects[]` | Reusable profile selection plus per-scene `name`, `prim_path`, and `root_pose` |

Run the entrypoints from the checkout root in a Python environment containing
the Isaac/Omni extensions. `builder.py` is a library module and does not write an
asset by itself. Both entrypoints launch headless Kit, create or replace the
exact output file, and close Kit; they do not import a robot or execute motion.
Runtime entrypoints only load an already-generated asset and never rebuild it.

## Capsule Rope

Default generation input:

```text
tools/object_assets/flexible/rope/config.yaml
```

Default output:

```text
assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

This explicit command uses the default input and output:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python tools/object_assets/flexible/rope/build_asset.py \
  --config tools/object_assets/flexible/rope/config.yaml \
  --output assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

The generation YAML uses `object.asset_path` and the asset-internal
`object.root_path`, then defines intrinsic structure under `rope`:

- `segments`, `length`, and `radius` define the chain resolution and size.
  A null radius derives a positive radius from length and segment count.
- `center`, `total_mass`, and `shape` place and size the local chain. Supported
  segment shapes are `capsule` and `cuboid`.
- `endpoint_box_mass`, `endpoint_box_size`, and the endpoint/segment linear and
  angular damping fields define the rigid bodies.
- `bend_*`, `twist_*`, and `lock_twist` define D6 joint limits, stiffness, and
  damping. A null twist limit is derived from the segment count.
- `disable_adjacent_collisions` controls collision filtering between connected
  bodies.
- `endpoint_color` and `rope_color` define the asset's visual materials.

The runtime profile is
`configs/objects/capsule_rope.yaml`. It declares a `dynamic_chain` USD object,
references the generated `asset_path`, and owns contact material and solver
overrides. Its `root_path` must equal the generated asset's
`object.root_path` (`/CapsuleRope`).

## T Block

Default generation input:

```text
tools/object_assets/rigid/tblock/config.yaml
```

Default output:

```text
assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda
```

Build with:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python tools/object_assets/rigid/tblock/build_asset.py \
  --config tools/object_assets/rigid/tblock/config.yaml \
  --output assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda
```

The `tblock` section owns:

- `center`, the asset-local origin offset.
- `stem_size` / `stem_offset` and `cap_size` / `cap_offset`, the two compound
  cuboids in asset-root `[x, y, z]` coordinates, in metres.
- `total_mass`, the mass on the compound rigid-body root.
- `linear_damping` and `angular_damping`, the authored rigid-body damping.
- `color`, the RGB visual material in the inclusive range `[0, 1]`.

The runtime profile is `configs/objects/TblockV1_default.yaml`. It references
the generated USD and owns static/dynamic behavior, contact material, and the
backend-neutral planning collision shape. A rigid USD profile does not use
`root_path`; the scene instance owns its stage path and world pose.

## Add The Asset To A Scene

1. Set the generation YAML `object.asset_path` to the intended output.
2. Run the corresponding `build_asset.py` and confirm its success marker.
3. Point `configs/objects/<profile>.yaml` at that same `asset_path`. For a
   dynamic chain, keep the runtime and generated `root_path` identical.
4. Add an env `objects[]` row that names the `object_profile` and supplies the
   required `root_pose`. Set `prim_path` only when the derived path is unsuitable.
5. Validate the profile graph, then launch the scene.

For example:

```yaml
objects:
  - name: tblock
    object_profile: TblockV1_default
    runtime_handle: tblock
    root_pose:
      xyz: [0.0, -0.45, 0.12]
      rpy: [0.0, 0.0, 0.0]
```

The object profile owns the asset source/path, import options, physics, and
planning collision. The env instance owns `prim_path` and `root_pose`; an
omitted path becomes `/World/Objects/<name>`. `runtime_handle` is an optional
interaction alias, not an asset or profile name.

## Preview In Isaac Sim

Preview the generated layer before wiring it into a scene:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
isaacsim isaacsim.exp.full
```

Use `File -> Open` to open either generated `.usda`. Check the dimensions,
orientation, local origin, default/root prim (`/CapsuleRope` or `/TBlock`),
rigid-body/collision/mass schemas, joints for the rope, and visual materials.
Previewing verifies the asset layer only; it does not validate an object profile,
env placement, runtime physics override, or planning collision. See
[USD Asset Preview](usd-preview.md) for the full inspection checklist.

## Verification

Run the offline entrypoint, config, and instance tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_object_asset_entrypoints.py \
  tests/test_object_instances.py \
  tests/test_system_configs.py -q
```

Validate the configured dependency graph without starting Isaac:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
```

Then run an import smoke test for a scene that references the asset:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py --env scene1 --gui
```

## Troubleshooting

`builder.py` exits without creating a file
: Run the sibling `build_asset.py`; `builder.py` only defines reusable builder
  functions.

`ModuleNotFoundError: linkerbot_sim`
: Run from the checkout root and set `PYTHONPATH=src` as shown above.

`pxr`, `PhysxSchema`, or another USD schema is unavailable
: Use the Python environment containing Isaac/Omni extensions, accept the Kit
  EULA, and invoke `build_asset.py` so the headless `SimulationApp` is started.

The build succeeds but the runtime cannot find the object prim
: Check the generated file and the runtime object profile use the same
  `asset_path`, then check the env instance `prim_path`. For a dynamic chain,
  also match the generated and runtime `root_path`.

Changing generation YAML has no effect in simulation
: Rerun `build_asset.py` and confirm its output is the file referenced by the
  runtime object profile. Runtime launch never regenerates the asset.
