# Collision Models

Language: [English](collision-models.md) | [中文](../../zh-CN/guides/collision-models.md)

Three collision concepts must remain separate: physical contact, replicated-world
isolation, and planner avoidance.

## Physical Contact

The physics backend resolves contact between scene bodies. Both products can require
physical contact for manipulation. For example, Kaleidoscope's T-block reward depends
on contact resolved by the selected PhysX CUDA or Newton backend even though
the product has no planner collision world.

Disabling PhysX scene-query support does not disable contact dynamics. Newton
also resolves task contact without constructing a planning collision/query world.

## Replicated Environment Isolation

Kaleidoscope creates many homogeneous environments. Isolation is fixed by the
selected physics engine rather than exposed as a public profile selector:

- the PhysX builder always enables environment IDs for its GridCloner replicas; and
- the Newton builder always creates independent Newton-runtime worlds.

The internal replication implementations remain separate because the engines have
different topology and ownership models. What was removed is the false ability to
mix an engine with an incompatible public replication profile.

Isolation prevents bodies in different environments from contacting each other. It
is required for valid vector-task physics but does not provide obstacle avoidance or
path queries.

Validate isolation after scene assembly and inspect representative environments at
the beginning, middle, and end of the index range in GPU smoke tests.

## Planner Collision World

Mirror owns collision geometry providers and per-robot planning contexts. They build
a consistent query representation for cuRobo planning and can be refreshed after
state changes.

Approximation quality, mesh/cuboid conversion, robot envelopes, and obstacle identity
must be explicit. See [Collision Approximation](../development/collision-approximation.md).

## Product Matrix

| Capability | Mirror | Kaleidoscope |
| --- | --- | --- |
| Physical contacts | Yes | Yes |
| Cross-environment isolation | Not applicable | Required |
| Scene queries for planning | Yes | Disabled |
| cuRobo collision world | Yes | No |
| Per-request `avoid_collisions` | Yes | No |
| Collision-aware batch IK | Optional Mirror path | No |

## Refresh Rules

Mirror invalidates planning collision state after:

- a physics step;
- `set_state`;
- snapshot restore;
- reset.

The planner refreshes according to `planning.request_defaults` or an explicit request.
That profile owns refresh policy only; cuRobo collision capability and cache capacity
come from `curobo.motion_planner`. A forced refresh can be expensive, so do not
duplicate it for every segment when one consistent timeline snapshot is sufficient.

## Diagnostics

When contact behavior is wrong, first determine which layer is failing:

1. inspect physical shapes, transforms, and contact reports;
2. for Kaleidoscope, verify environment origins and isolation authoring;
3. for Mirror planning, inspect provider geometry and freshness separately from
   physical contact;
4. compare robot collision envelopes with the visual and articulation geometry.

Do not fix a planner approximation by changing physical collision shapes without a
separate dynamics review.

The warehouse floor in the `mirror/scene3` scene (file
`configs/scenes/mirror/scene3.yaml`, identity `scene.id: scene3`) makes this boundary explicit. Its coplanar
source mesh is visual-only, while the wrapper asset authors an invisible analytic
plane at the same local height for physical contact. The scene therefore keeps
`add_ground: false`, avoiding a second overlapping ground surface, and PhysX plus
Newton CPU/CUDA consume the same analytic collider.
