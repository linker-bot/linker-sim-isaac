# Naming And Ownership

Language: [English](naming.md) | [中文](../../zh-CN/development/naming.md)

Names communicate product scope and resource ownership. New code and documentation
must use the current public names directly; do not add compatibility aliases.

## Product Names

**Mirror** is the reality-mapping product: one simulation world that reflects a
physical workspace and provides interactive planning, collision, camera, telemetry,
and transport capabilities.

**Kaleidoscope** is the reinforcement-learning product: many homogeneous environments
viewed through one GPU-native vector interface.

Use the names as proper nouns in prose and lowercase package/profile values in code:

- `linkerbot_sim.mirror`, `mode: mirror`;
- `linkerbot_sim.kaleidoscope`, `mode: kaleidoscope`.

## Layer Names

| Layer | Naming rule |
| --- | --- |
| Product facade | Product noun at package root |
| Composition root | `bootstrap`, `create_*_runtime`, or `make_*_env` |
| Runtime | Product-prefixed owner type |
| Infrastructure | Backend/engine term under `linkerbot_sim.isaac` |
| Pure domain | Capability noun without a product prefix when genuinely shared |
| Framework integration | Framework name under `linkerbot_sim.training` |

Do not call a product object `manager` when it owns the complete lifecycle; use
`Runtime`, `Session`, `Owner`, or a capability-specific service. Reserve `Adapter` for
a boundary that changes an interface or representation without taking ownership of
the underlying engine.

## Identity Terms

- A scene selector is a catalog lookup key namespaced by product, such as
  `mirror/scene3` or `kaleidoscope/tblock_push`.
- A scene file is its path under the configuration root, such as
  `configs/scenes/mirror/scene3.yaml`; it is provenance, not a selector.
- `scene.id` is the unqualified stable scene identity and matches the file basename,
  such as `scene3`.
- `task.id` identifies a task profile and matches its basename.
- A Mirror `robot_id` is a session-local numeric selector; `robot_label` is the stable
  configured identity assertion.
- Kaleidoscope `env_ids` are CUDA row selectors, not persistent world identities.
- Object `name` is the configured logical identity; `prim_path` is the USD namespace.
- `request_id` is a client-generated protocol identity, not a motion or robot ID.

## Units And Frames

Suffix configuration and DTO fields when the unit is not otherwise fixed:

- `_m`, `_m_s`, `_s`, `_hz`, `_rad`;
- `_xyz`, `_rpy`, `_quat_wxyz`;
- `_mib` for binary-megabyte memory gates.

Use `world`, `env`, `robot_base`, and `tcp` exactly for task-space frames. Do not use
ambiguous `local` without naming the owning local frame.

## State Terms

- **state** is the current mutable runtime field set;
- **snapshot** is an owned point-in-time copy suitable for restore;
- **clone state** copies selected Kaleidoscope rows within one runtime;
- **checkpoint** is an explicit persistent cold artifact;
- **reset** restores task/configured initial semantics, not an arbitrary snapshot.

A Mirror snapshot is an owned data copy only; Mirror exposes no clone operation and
does not create another world.

## Configuration Names

Profile references use extension-free category-relative stems such as
`physx/cuda` or `kaleidoscope/tblock_push_v1`. Every path component excludes dots and
backslashes; `/` is the only namespace separator. A leaf key names the fact it owns.
Avoid generic `default` inside a leaf when a product-qualified name communicates its
scope.

Scene references additionally require the matching product namespace. Thus
`mirror/scene3` resolves to `configs/scenes/mirror/scene3.yaml` with
`scene.id: scene3`; a Mirror root cannot select or symlink to a scene under
`kaleidoscope/`.

CUDA device aliases such as `active_gpu`, `physics_gpu`, or policy-specific device
numbers are prohibited. The canonical field is `compute.cuda_device`.

## Protocol Names

Mirror operations use a capability prefix and verb, for example
`motion.joint_goal`, `snapshot.get`, and `runtime.status`. Additions require a protocol
version decision; do not introduce undocumented synonyms.

Kaleidoscope methods use Python verbs (`step`, `reset_idx`, `clone_state`) because it
has no wire protocol.

## Files And Tests

Use lowercase snake case for Python files and tests, and kebab case for user-facing
Markdown filenames. Tests should name the invariant, not an implementation accident.
Configuration fixtures should identify product, capability, and backend where useful.
