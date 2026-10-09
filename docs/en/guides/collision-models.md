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

### Explicit body seams in Newton

Newton's pinned USD importer reads collider-level filtered pairs but does not expand
body-level pairs emitted by the MJCF importer. The Simulator now projects each explicit
body pair to only the shapes owned by those two bodies before prototype replication.
Descendant rigid links remain collidable. PhysX keeps the original USD relationships.
This preserves named assembly seams such as AR5 link5–link7 without disabling the hand
or camera, and applies equally to Mirror and Kaleidoscope.

## Mounted Hands, Fixtures and Carried Objects

The eight flanged L6/O6 profiles declare `robot.planning_collision.mounted_urdf`.
Mirror reads each imported collider in its named link frame and evaluates the full
URDF with the current measured joints, including hand followers. During an arm plan
this hand shape is frozen. A collision-aware timeline rejects hand changes before
that plan ends; execute the hand change first and plan from a new snapshot. With
`static_others`, commands that move other robots before a plan ends are likewise
rejected. Holds/current-position goals and changes after the plan remain allowed.
Direct simultaneous control remains available; these checks do not certify
physical pose holding, dynamic obstacles, or coupled multi-robot motion. Volume-covering spheres replace the old generic gripper
sphere at the flange; they move with each candidate arm configuration. The old AR5
YAML already contained a coarse L6 envelope, not articulated finger geometry.

This is a conservative approximation of collider boxes, with a 0.03 m maximum cell
edge. It can reject a narrow but physically feasible passage. Self-contact *inside*
the frozen hand/flange/camera assembly is not checked by the flange envelope; Tasks
must validate hand-shape transitions and the current O6 thumb/camera limitation.
The current AR5 cuRobo YAML excludes the flange/TCP against own-arm links 4, 6
and 7. Because the envelope shares that flange frame, these exclusions also apply
to the hand, camera and carried object. `ignored_own_arm_links` in mounted coverage
reports the actual configured names; it does not claim complete own-arm coverage.
Tasks must check those interactions separately. Removing the exclusions requires
first improving the coarse arm spheres and validating installation contacts.
Carried-object contact is checked against all mounted links except explicitly named
`touch_links`. Neither check is a whole-body hand-motion planner.

The workstation and top camera profiles use `object.planning_collision.source:
colliders` for static objects. Dynamic objects retain explicit shapes following
the live root pose; dynamic-chain collider extraction is not provided. Every imported
collider gets its own bound; empty spaces between parts
are preserved. Geometry names are `<object name>/<path below the instance root>`.
Providers publish world coordinates; the registry converts them to the target
robot's base before cuRobo queries. Other robots' arm spheres and mounted boxes enter
`static_others` snapshots. `independent` deliberately omits other robots.

`scene.planning_contacts` lists exact `robot_label`, `link_name`, `geometry_name`
triples. The example scenes allow only fixed-base installation contacts. cuRobo 0.8
cannot express a link/world-object exclusion, so the named geometry is omitted from
optimization and checked against every non-exempt link before a successful path is
returned. This may reject a candidate that needs further planning around that part;
it does not disable the part globally or change physical contact filters.

Paths are checked at endpoints and intermediate samples with at most 0.02 rad
between joint samples, capped at 10,000 samples. This is sampled validation, not
continuous collision detection. `MotionResult.diagnostics.coverage` and the planning
registry's latest coverage report expose sample counts, minimum sampled clearance,
model fingerprint, approximation and contact allowances. Direct joint commands do
not invoke this planner validation. The canonical direct IK profile remains
collision-unaware; motion planning owns avoidance.

Use the Mirror Python facade after establishing a physical grasp:

```python
runtime.attach_planning_object("Tblock", 0, touch_links=("hand_lh_index_distal",))
# Plan and execute the arm motion while maintaining that grasp.
runtime.detach_planning_object("Tblock")
```

This declares planning geometry; it creates no physical joint or grasp controller.
Each snapshot freezes the current object-to-flange transform from live state, so a
changed grip or object position updates the next plan. The object is removed from
its owner's world view, included in the candidate flange envelope and still visible
to other robots. Detach returns it to the live world view. Snapshot/state capture
preserves declarations by robot label; restore validates them before physics writes,
and reset clears them. Geometry or attachment asset changes require a new runtime.

Hand/payload model changes destroy and lazily rebuild all affected cuRobo solvers,
including graph and validation models. Arm motion alone reuses the mounted model.
The Mirror cuboid cache is 128 entries, sufficient for the example's other mounted
robot and fixtures; larger scenes must set an explicit adequate capacity.

`just smoke-mirror-planning-geometry` checks native hand/camera-only obstacles,
state refresh, carried mid-path collisions and attachment lifecycle. These smokes
use planning fixtures and do not claim to demonstrate a successful physical grasp.

For a task phase, use exact link/geometry pairs and plan **and execute** inside the scope:

```python
with runtime.planning_contact_scope(0, (("hand_lh_index_distal", "Tblock"),)):
    # Run the task's grasp-approach planning and execution here.
    ...
```

Leaving the scope restores the previous planning policy, including on exceptions;
nested scopes preserve outer allowances. Other links remain checked against the
object. This does not change physical contact filters or grant persistent permissions
through reset/snapshot restore. Carried-object `touch_links` are a separate declaration.

## Pinned USD Parser Workaround

Kit's OpenUSD 25.11 collider parser can race when several meshes belong to one rigid
body (`_FinalizeCollisionDescs` appends to a shared vector). This caused native
allocator crashes while loading the decomposed assets. Isaac MJCF/URDF conversion
and Newton USD parsing now temporarily use one USD Work thread; the effective
limit is checked and restored even after an exception. Kit locks the public Work
setter through `PXR_WORK_THREAD_LIMIT`. On this pinned Linux runtime the guard
therefore calls the exported `WorkImpl_SetConcurrencyLimit(unsigned)` from the
already loaded Kit extension. It never loads a second USD, patches a binary or
changes environment variables. A missing symbol or ineffective limit fails import
instead of silently running the unsafe parser. Physics, rendering and rollout parallelism are
unchanged after import. The guard is specific to the bundled 25.11 version.

Upstream [OpenUSD fix ed857d77c95b](https://github.com/PixarAnimationStudios/OpenUSD/commit/ed857d77c95b27dd9cf0919e8ed36366cf3301e6)
replaces the shared vector write with a thread-safe gather and serial merge. Retire
this workaround when upgrading the bundled runtime to a verified fixed parser.
This allocator crash is separate from Hydra `_MarkInstancerDirty` diagnostics.
