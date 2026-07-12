# Collision Models Guide

Language: [English](collision-models.md) | [中文](../../zh-CN/guides/collision-models.md)

The project has three independent collision layers. Configure the layer that owns the
behavior you need; enabling one layer does not imply that either of the others is active.

| Layer | Purpose | Primary owner |
| --- | --- | --- |
| Simulation collision | PhysX contact, friction, penetration response, and rigid-body motion | Imported USD/URDF/MJCF colliders plus robot/object physics profiles |
| Planning collision | Robot self-collision and obstacle checks used by cuRobo | Robot cuRobo model, object `planning_collision`, and the Scene collision registry |
| Tiled env filtering | Prevent contact between different cloned env roots | Env `tiled.clone` configuration |

## Simulation Collision

Isaac importers materialize colliders from robot and rigid-object assets. The project-level
`collision_approximation` field selects how imported geometry becomes collision geometry;
valid choices and asset-format restrictions are documented in
[Collision Approximation](../development/collision-approximation.md).

Object profiles own runtime material, static/dynamic behavior, solver iterations, and import
options. Env profiles own each object instance path and pose. A planning obstacle does not
create a PhysX collider, and disabling planning collision does not disable physical contact.

Use simulation collision when the requirement is contact behavior: grasping, settling,
friction, restitution, penetration, or whether an object moves after impact.

## Planning Collision

cuRobo collision-aware requests require all of the following:

- A robot profile with a valid cuRobo planning model and collision geometry.
- A materialized cuRobo context with collision-query capability.
- A Single Scene or Tiled Scene planning path that supports `avoid_collisions=true`.
- A planning world containing the obstacles relevant to that request.

Rigid object profiles may expose a simplified planning shape:

```yaml
planning_collision:
  shape: cuboid
  size: [0.04, 0.2, 0.22]
  xyz: [-0.02, 0.0, 0.11]
  rpy: [0.0, 0.0, 0.0]
  padding: 0.0
  enabled: true
```

This shape is an explicit planning approximation. It does not replace the object's authored
PhysX compound collider. Supported canonical shapes are the ones accepted by the object
profile parser; cuRobo adapters materialize them into the fixed `cuboid`/`mesh` cache model.

Single Scene runtime registers object and robot geometry providers, captures one immutable planning
snapshot per planning transaction, and excludes the target robot from its own obstacle list.
With `coordination="static_others"`, other robots can be included as frozen obstacles.
Dynamic state changes mark the collision registry dirty; `force_collision_refresh` requests
an explicit current view.

`avoid_collisions=true` is strict. Missing robot spheres, unavailable world collision, an
unsupported request path, or insufficient backend capability fails the request. The runtime
does not silently execute a collision-unaware trajectory.

Use planning collision when the requirement is path feasibility, obstacle avoidance, or robot
self-collision during IK/planning. It does not simulate contact forces.

## Tiled Scene Inter-Env Filtering

Cloned envs share one PhysX scene. Without inter-env filtering, colliders from neighboring env
roots can contact each other when spacing or geometry overlaps.

```yaml
tiled:
  clone:
    filter_collisions: true
    collision_filter_strategy: collision_groups
    collision_root_path: /World/collisions
    physics_scene_path: null
    global_collision_paths: auto
    extra_global_collision_paths: []
```

`physics_scene_path: null` auto-discovers the stage's single
`UsdPhysics.Scene`. Discovery fails when the stage contains none or more than
one; for a multi-scene stage, set the field to the required absolute prim path.

`collision_groups` creates one collision group per env and a shared global group. With the
inverted PhysX filter, an env contacts itself and declared global ground/fixtures, but not other
envs. Its authored relation count grows linearly with env count.

`filtered_pairs` authors pair filters directly and has quadratic scaling. It is an explicit
alternative for environments where collision groups cannot be used. Group-only fields are
rejected unless filtering and the `collision_groups` strategy are active.

Inter-env filtering changes physical contact only. It does not give the planner an obstacle
world and does not enable cuRobo collision checking.

## Choosing Global Paths

`global_collision_paths: auto` scans the supported ground locations outside env roots. An
explicit list replaces auto discovery; `extra_global_collision_paths` appends shared fixtures.
Every configured path must be a valid absolute prim path outside cloned env roots. Treating an
env-local prim as global would reintroduce cross-env coupling.

## Diagnostics

| Symptom | Check |
| --- | --- |
| Robot passes through an object in simulation | Imported collider, `physics.static`, material, collision approximation, and stage pose |
| Physical contact works but planning ignores the object | Object `planning_collision`, Scene collision registry, and cuRobo capability |
| `avoid_collisions=true` is rejected | Robot collision model, context capability, request kind, and cache/world availability |
| Different Tiled envs touch each other | `filter_collisions`, strategy, global paths, spacing, and authored group diagnostics |
| Ground stops colliding after filtering | Ground/global prim is missing from global collision paths |
| Planning view is stale after state mutation | Registry invalidation and `force_collision_refresh` |

## Related Documentation

- [Motion Planning](motion-planning.md)
- [Configuration](configuration.md)
- [Collision Approximation](../development/collision-approximation.md)
- [Tiled Scene JSON Reference](../reference/tiled-scene-json.md)
- [Known Constraints](../operations/constraints.md)
