# Object Assets

Language: [English](object-assets.md) | [中文](../../zh-CN/development/object-assets.md)

Offline builders create reusable USD assets. Runtime profiles reference the generated
files and apply deployment-specific material, solver, static/dynamic, or import
settings. Runtime startup must never regenerate geometry.

## Ownership Split

| Fact | Owner |
| --- | --- |
| Intrinsic geometry, compound-body structure, mass, authored damping, visual material | Tool-side generation YAML and builder |
| Generated USD path and root prim | Tool-side `object` section; referenced consistently by runtime asset profile |
| Runtime common contact material and static behavior | `configs/objects/` profile `object.physics` |
| PhysX combine mode and dynamic-chain solver tuning | `configs/objects/` profile `object.physics.physx` |
| Instance identity, world prim path, and root pose | Selected product-namespaced scene profile under `configs/scenes/<product>/` |
| Environment count, USD path naming, and origins | Kaleidoscope mode-root `environments` mapping |

## Builders

T-block:

```bash
PYTHONPATH=src .venv/bin/python \
  tools/object_assets/rigid/tblock/build_asset.py
```

Capsule rope:

```bash
PYTHONPATH=src .venv/bin/python \
  tools/object_assets/flexible/rope/build_asset.py
```

Both builders start a headless Kit application because USD/PhysX schema registration
is required while authoring. They save a USD/USDA asset and do not run a manipulation
task.

## Complete Builder Option Table

| Builder | Option | Default | Meaning |
| --- | --- | --- | --- |
| T-block | `--config PATH` | `tools/object_assets/rigid/tblock/config.yaml` | Tool-side generation YAML. |
| T-block | `--output PATH` | `object.asset_path` | Override the generated asset destination. |
| Capsule rope | `--config PATH` | `tools/object_assets/flexible/rope/config.yaml` | Tool-side generation YAML. |
| Capsule rope | `--output PATH` | `object.asset_path` | Override the generated asset destination. |

Paths are resolved from the repository workspace. An output override does not rewrite
the runtime object profile; update references deliberately after reviewing the asset.

## T-Block Schema

The generation file contains `object` and `tblock` sections. The T-block section owns
local center, stem/cap sizes and offsets, total mass, nonnegative damping, and RGB
color in `[0, 1]`. Sizes are positive XYZ lengths in metres. The builder authors one
compound rigid body with two collider/visual cuboids.

## Capsule Rope Schema

The generation file contains `object` and `rope` sections. The rope section owns
segment count and length, optional radius, total and endpoint mass, segment shape,
damping, bending/twist parameters, adjacent-collision policy, endpoint dimensions,
and colors.

At least two segments are required. Length, radius, masses, and endpoint dimensions
are positive. Supported segment shapes are `capsule` and `cuboid`. A null radius or
twist limit derives a deterministic value from length/count.

## Runtime Physics Leaves

Keep standard contact coefficients under `object.physics.material`: static friction,
dynamic friction, and restitution. Put `friction_combine_mode` under
`object.physics.physx.material`. Dynamic-chain position and velocity solver iterations
belong under `object.physics.physx.solver`.

PhysX combines the common material with that engine-specific leaf. Newton consumes
only the standard material and never imports or reports the PhysX leaf as a skipped
normal setting. The former flat combine-mode and rope solver paths are invalid rather
than deprecated aliases.

## Validation Workflow

1. Change the tool-side generation YAML.
2. Run the builder in the Isaac environment.
3. Inspect the USD hierarchy, default prim, units, visual material, collision APIs,
   mass, and local bounds.
4. Preview it using [USD Preview](usd-preview.md).
5. Run Mirror contact/planning checks if the asset will appear in reality replay.
6. Run Kaleidoscope environment-isolation/contact/memory smoke tests if the asset will appear
   in a vector task.
7. Commit the YAML, generated asset when repository policy requires it, and matching
   runtime profile change together.

Do not compensate for a bad intrinsic asset by scattering instance-specific scale or
mass overrides across scenes.
