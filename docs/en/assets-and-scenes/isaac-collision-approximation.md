# Isaac Collision Approximation

Language: [English](isaac-collision-approximation.md) | [中文](../../zh-CN/资产与场景/Isaac%20碰撞近似配置.md)

This document describes the collision approximation fields used when importing assets into Isaac.

## Where It Lives

Robot importer options live in robot profiles:

```yaml
robot:
  import:
    collision_approximation: convex_decomposition
    self_collision: false
```

Object importer options live in object profiles:

```yaml
object:
  import:
    collision_approximation: convex_decomposition
```

Do not put importer collision approximation settings in controller YAML.

## Common Values

Common Isaac values include:

- `none`
- `convex_hull`
- `convex_decomposition`
- `mesh_simplification`
- `sdf`

The exact support depends on the Isaac importer and asset type. Prefer existing project defaults unless you are validating a specific collision behavior.

## Runtime Meaning

Importer collision approximation affects the collision geometry generated when Isaac imports an asset. It is not the same as cuMotion's planning collision model.

cuMotion uses URDF/XRDF and its own robot/world descriptions. Changing Isaac importer collision approximation does not automatically update cuMotion planning geometry.

## Checks

After changing collision approximation:

- Launch a GUI smoke test and inspect generated collision prims.
- Verify root poses and fixed-base behavior still match the scene.
- Run relevant config and motion tests.
