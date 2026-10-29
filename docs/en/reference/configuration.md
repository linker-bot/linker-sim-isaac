# Configuration Reference

Language: [English](configuration.md) | [中文](../../zh-CN/reference/configuration.md)

Mirror and Kaleidoscope use a strict profile graph under `configs/`. The configuration catalog
does not perform last-writer-wins merging: a mode root names canonical leaf profiles,
and each leaf owns one category of facts.

## Catalog API

```python
from linkerbot_sim.configuration import (
    load_kaleidoscope_config,
    load_mirror_config,
)

mirror = load_mirror_config("physx_cpu")
mirror_newton_cpu = load_mirror_config("newton_cpu")
mirror_newton_cuda = load_mirror_config("newton_cuda")
kaleidoscope = load_kaleidoscope_config("physx_cuda")
kaleidoscope_newton = load_kaleidoscope_config("newton_cuda")
```

Both functions accept a profile stem, an explicit path below the matching mode
directory, and an optional `configs_root`. They return frozen typed configuration
graphs with a read-only `sources` mapping.

## Resolved Closure, Provenance, And Fingerprints

The selected root must close the entire graph: a mode references leaves and a scene
references robot and object profiles. The physics engine derives the default controller
bundle; a robot-profile default or scene-instance override may replace it. Controller
precedence is scene instance, robot profile, then the physics-derived default.

The returned root freezes instance `resolved_profile` values, the read-only
`controller_bundles`, and `sources`. Runtime builders consume these resolved objects
and never resolve the same names again under the repository default. A custom
`configs_root` must therefore provide every referenced mode, leaf, robot, object, and
controller file. The retired `configs/envs` schema is not a configuration entry point.

`sources` records absolute provenance for diagnostics but is excluded from semantic
fingerprints. The configuration layer owns one canonical payload and fingerprint used
by validation and Kaleidoscope snapshot compatibility. Relocating identical content
to another absolute root preserves the fingerprint; changing effective robot, object,
or controller content changes it.

The catalog:

- rejects duplicate YAML keys at any depth;
- accepts only safe YAML values and string mapping keys;
- rejects missing and unknown fields;
- rejects absolute profile references, `..`, empty path components, backslashes, and
  dots in any selector component (including explicit `.yaml` suffixes);
- requires a scene selector in the namespace of the selected product, such as
  `mirror/scene3` or `kaleidoscope/tblock_push`;
- resolves symbolic links and prevents escape from the configuration root; scene
  symlinks must additionally remain inside the selected product namespace;
- checks cross-profile invariants before Kit or CUDA starts.

Validate a graph from the command line:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile newton_cuda
```

## Directory Ownership

| Directory | Root key | Owner |
| --- | --- | --- |
| `configs/modes/mirror/` | `mode`, `compute`, `profiles` | Mirror composition and shared CUDA device |
| `configs/modes/kaleidoscope/` | `mode`, `compute`, `environments`, `profiles` | Kaleidoscope composition, CUDA device, environment count, paths, and origin |
| `configs/scenes/mirror/` | `scene` | Mirror world topology, instances, and optional visual facts |
| `configs/scenes/kaleidoscope/` | `scene` | Headless Kaleidoscope environment-template topology and instances |
| `configs/physics/` | `physics` | `engine`, `execution`, solver, and engine-specific capacity facts |
| `configs/control/mirror.yaml` | `control` | Mirror command, idle-step, wall-clock pacing, interface, and request-default contract |
| `configs/control/hybrid_force_position.yaml` | `hybrid_force_position` | Explicit Cartesian hybrid-control gains, fixed safety limits, and runtime tuning bounds |
| `configs/curobo/` | `curobo` | FK/IK capacity and optional MotionPlanner numerical capability |
| `configs/planning/` | `planning` | Backend-neutral Mirror request defaults only; it does not select or size a numerical backend |
| `configs/tasks/` | `task` | Kaleidoscope action, observation, reward, reset, and termination |
| `configs/outputs/` | `outputs` | Mirror render, camera, logging, and telemetry |
| `configs/training/` | `training` | Downstream trainer settings; not part of environment construction |
| `configs/visualization/kaleidoscope.yaml` | `viewport` | Launch-only single-environment Kaleidoscope viewport; outside training semantics |
| `configs/robots/` | asset-specific | Reusable robot asset facts |
| `configs/objects/` | asset-specific | Reusable object asset facts |
| `configs/controllers/` | controller bundle | Engine-specific asset drive gains; the default bundle is derived from `physics.engine` |

Every top-level profile group has one same-named schema owner under
`linkerbot_sim.configuration`: single-file groups use modules such as `scenes.py`, while
nested groups use packages such as `modes/`, `tasks/`, `training/`, and `visualization/`.
Only `catalog.py`, `common.py`, and `fingerprint.py` are configuration infrastructure
rather than profile groups. Scene visual primitives belong to `scenes.py`; the
`visualization` package contains only the Kaleidoscope launch profile corresponding to
`configs/visualization/kaleidoscope.yaml`.

A scene has three related but distinct names. The mode root stores the namespaced
selector (`mirror/scene3`), the catalog resolves that selector to a file path
(`configs/scenes/mirror/scene3.yaml`), and the file stores the unqualified stable
identity (`scene.id: scene3`). The same rule gives selector
`kaleidoscope/tblock_push`, path
`configs/scenes/kaleidoscope/tblock_push.yaml`, and `scene.id: tblock_push`.
Flat selectors and cross-product scene references are invalid because the two scene
schemas are not interchangeable.

Mirror logging has no separate profile directory. `outputs.logging` is its sole
configuration owner. The runtime configuration tree also contains no selectable
`example.yaml` templates; the maintained profiles below and this reference document
are the schema examples.

## Mirror Mode Root

```yaml
mode: mirror
compute:
  cuda_device: 0
profiles:
  scene: mirror/scene3
  physics: physx/cpu
  control: mirror
  curobo: mirror
  planning: mirror
  outputs: mirror_default
```

The keys above are exact. Every Mirror root declares `compute.cuda_device`. PhysX CPU
does not use it for simulation, but cuRobo IK/planning and rendering consume the same
canonical device. A Newton root changes only the physics reference; the Mirror
root derives the `newton` controller bundle from `physics.engine` while sharing one
control profile:

```yaml
mode: mirror
compute:
  cuda_device: 0
profiles:
  scene: mirror/scene3
  physics: newton/cuda
  control: mirror
  curobo: mirror
  planning: mirror
  outputs: mirror_default
```

Mirror accepts PhysX/CPU, Newton/CPU, and Newton/CUDA. For either CPU execution,
`physics_device` is `cpu`, while `compute.cuda_device` still selects the GPU used by RTX
and cuRobo. Newton/CUDA derives `cuda:{compute.cuda_device}`. Both Newton variants derive
one world in the Mirror session projection; the reusable runtime itself remains capable
of managing multiple worlds.

The optional `profiles.hybrid_control` slot is accepted only by Mirror. The maintained
composition selects a dedicated hybrid scene at 60 Hz commands / 240 Hz feedback / 480 Hz physics and PhysX CPU:

```yaml
mode: mirror
compute:
  cuda_device: 0
profiles:
  scene: mirror/scene3_hybrid
  physics: physx/cpu
  control: mirror
  hybrid_control: hybrid_force_position
  curobo: mirror
  planning: mirror
  outputs: mirror_default
```

Omitting `hybrid_control` leaves all v3 hybrid operations unsupported and fail-closed.
Selecting it requires initial position mode, PhysX CPU, sufficient scene frequency,
arm gravity compensation, physical TCP metadata, arm `effort+direct`, and hand/default
`position+implicit` controller profiles.

## Kaleidoscope Mode Root

```yaml
mode: kaleidoscope
compute:
  cuda_device: 0
environments:
  num_envs: 256
  base_env_path: /World/envs
  env_prefix: env
  origin_xyz: [0.0, 0.0, 0.0]
profiles:
  scene: kaleidoscope/tblock_push
  physics: physx/cuda
  task: kaleidoscope/tblock_push_v1
```

The root contains exactly `mode`, `compute`, `environments`, and `profiles`. `scene`,
`physics`, and `task` are the three required profile references. Kaleidoscope has no
control profile or resolved control object: its action contract belongs to the task,
and its default controller bundle is derived from `physics.engine`. An optional
`profiles.curobo` reference is allowed only when the selected task uses an end-effector
or linear action; the canonical `joint_control` roots must omit it. `environments` contains exactly `num_envs`,
`base_env_path`, `env_prefix`, and `origin_xyz`. Renderer, camera, transport, planner, planning, playback,
and telemetry keys are rejected recursively.

The optional viewport is a separate launch profile rather than a mode slot:

```yaml
viewport:
  selected_env: 0
  render_every_n_steps: 1
  width: 1280
  height: 720
  window_width: 1440
  window_height: 900
  renderer: RaytracedLighting
  anti_aliasing: 0
  samples_per_pixel_per_frame: 1
  denoiser: false
  visuals: { ... }
```

`KaleidoscopeViewportSettings` strictly validates the root and nested scene visuals.
`make_viewport_env()` selects it with `viewport_profile="kaleidoscope"`, or accepts an
already loaded object as `viewport`; Gymnasium `render_mode="human"` uses
`viewport_profile`. It is not attached to `KaleidoscopeConfig`; window, lighting, or
cadence changes therefore
do not change episode snapshot/clone fingerprints. `selected_env` must be valid for
the effective environment count and is the only renderer-facing world. Both viewport
Kits exclude cameras, SyntheticData, Replicator, recording, and image observations;
training physics ticks remain `render=False`.

The bundled Newton alternative is:

```yaml
mode: kaleidoscope
compute:
  cuda_device: 0
environments:
  num_envs: 256
  base_env_path: /World/envs
  env_prefix: env
  origin_xyz: [0.0, 0.0, 0.0]
profiles:
  scene: kaleidoscope/tblock_push
  physics: newton/cuda
  task: kaleidoscope/tblock_push_v1
```

`compute.cuda_device` is the only configured GPU index. The selected physics engine,
Torch, Warp interop, cuRobo, and the trainer derive their device from this value. Do
not add `active_gpu`, `physics_gpu`, policy-device, or another CUDA index to leaf
profiles.

An end-effector or linear composition adds the numerical backend at the mode root,
not inside the task:

```yaml
profiles:
  scene: kaleidoscope/tblock_push
  physics: physx/cuda
  task: kaleidoscope/tblock_push_v1
  curobo: kaleidoscope_batch_ik
```

The task still owns only action semantics and never selects a backend. The catalog
requires this optional reference for every EE/linear action and rejects it for
`joint_control` and `joint_delta`, so a joint-only environment cannot allocate an unused
cuRobo context.

## Scene Profiles

Common instance shape:

```yaml
robots:
  - label: left_arm
    robot_profile: ar5_08_l6_l
    root_pose:
      xyz: [0.0, 0.09, 0.0]
      rpy: [-1.5707, 0.0, 0.0]
objects:
  - name: Tblock
    object_profile: TblockV1_default
    prim_path: /World/TBlock
    root_pose:
      xyz: [0.15, 0.0, -0.4]
      rpy: [0.0, 1.5707, 0.0]
```

Robot labels, object names, and object prim paths must be unique. Poses use metres
and XYZ Euler radians. Object prim paths are absolute.

Mirror scene fields are exactly:

- `id`, `description`, `gravity_z`, `add_ground`, `ground_height`;
- `control_frequency_hz`, `physics_frequency_hz`, `render_frequency_hz`;
- `planning_startup` (`lazy` or `prewarm`);
- `robots`, `objects`, `cameras`, `viewport`, and `lights`.

`planning_startup: lazy` defers every planning context and planner until its first
request. `prewarm` creates each planning-capable robot's `interactive` slot-zero
context in robot-ID order, synchronizes one shared initial collision snapshot, and
materializes its MotionPlanner before `MIRROR_INTERACTIVE_READY`. The numerical
`motion_planner.warmup` switch remains owned by the cuRobo profile and runs while that
planner is materialized; the scene policy controls when materialization happens.

Each camera declares `id`, parent and child absolute prim paths, pose, `[width,
height]`, positive frequency, unique modalities, and `[near, far]` clipping metres.

Kaleidoscope scene fields stop at `physics_frequency_hz`, `robots`, and `objects`.
There is no render frequency, camera, viewport, light, or replication count in this
schema. Its scene describes one environment template only.

The catalog expands every robot/object profile and effective controller bundle from the
same configuration root before Kit starts. A Kaleidoscope scene may contain any
number of static rigid objects, but
must contain exactly one non-static rigid object; that object must be
`task.dynamic_object`. Dynamic-chain objects are rejected because the current
`object.*` snapshot schema owns one rigid pose and velocity only.

For both products, `scene.id` must match the selected scene profile basename.

## Physics Profiles

The physics-leaf catalog keeps a regular engine/execution layout:

```text
configs/physics/
  physx/{cpu,cuda}.yaml
  newton/{cpu,cuda}.yaml
```

Each canonical leaf is independently strict and schema-valid. Product roots then enforce
their narrower capability matrices: Mirror composes `physx/cpu`, `newton/cpu`, or
`newton/cuda`; Kaleidoscope composes only the CUDA leaves.

### PhysX CPU

```yaml
physics:
  engine: physx
  execution: cpu
  solver_type: PGS
```

`solver_type` is `PGS` or `TGS`. This variant belongs to Mirror.

### PhysX CUDA

```yaml
physics:
  engine: physx
  execution: cuda
  solver_type: PGS
  use_fabric: true
  enable_scene_query_support: false
  memory: { ... }
```

This variant belongs to Kaleidoscope. Fabric must be enabled and scene-query support
must be disabled because the product has no planning collision world. Engine allocation
capacity is not a project configuration field.

The process-level `memory` mapping is the complete `GpuMemoryBudget`; missing fields
do not receive defaults:

| Field | Type and range | Gate semantics |
| --- | --- | --- |
| `max_simulator_process_mib` | Positive integer MiB | Maximum memory attributed by NVML to the simulator PID, including Kit, PhysX, Torch, and native CUDA allocators |
| `min_free_floor_mib` | Positive integer MiB | Absolute free-device-memory floor at prelaunch, post-warmup, and both steady samples |
| `min_free_fraction_after_warmup` | Number in `(0, 1]` | Free-device-memory fraction required at post-warmup and both steady samples |
| `max_steady_growth_mib` | Nonnegative integer MiB | Maximum simulator-PID growth from steady baseline to steady final |

The auditor maps `compute.cuda_device` to NVML by CUDA UUID, requires the PID to be
visible after warmup, and reports Torch allocated/reserved for diagnostics. NVML
process memory remains the budget owner. Run either maintained entrypoint:

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

This gate accepts only the Kaleidoscope `physx_cuda` profile. Newton per-world contact,
Jacobian, and world-count capacities are a separate contract and cannot be inferred
from these four fields.

### Newton CPU

`configs/physics/newton/cpu.yaml` declares `engine: newton` and `execution: cpu` and
is parsed as `NewtonCpuSettings`. Mirror's `newton_cpu` composition runs one world with
Newton collision + MuJoCo Warp on CPU, no CUDA stream or graph, and the same Newton contact pipeline as CUDA. The root
`compute.cuda_device` remains mandatory because cuRobo and optional RTX rendering still
consume that GPU. Kaleidoscope rejects Newton CPU because its training contract is
CUDA-resident.

### Newton CUDA

```yaml
physics:
  engine: newton
  execution: cuda
  nconmax_per_world: 200
  njmax_per_world: 1200
  # remaining capacity and integration fields are required by the canonical leaf
```

The `newton/cpu` and `newton/cuda` leaves declare separate executions. The latter declares `engine: newton`,
`execution: cuda`, per-world
contact/Jacobian capacities, CUDA graph use,
solver iterations, line search, constraint solver, contact gap and matching, and
solver/collision determinism. It inherits the device from the mode root. Mirror and Kaleidoscope share
`newton/cuda`; Mirror derives one world while Kaleidoscope derives the runtime world
count from final `environments.num_envs`. Neither loads the Isaac Newton extension. Robot
gravity policy is authored as `mjc:gravcomp` before model finalization; runtime
per-link gravity changes are unsupported. `newton_cuda` is both an explicit Mirror or
Kaleidoscope selector and the resolved CUDA runtime kind; the implementation module is
simply `isaac.physics.newton`.

### Shared Newton pipeline defaults

CPU and CUDA both use Newton collision detection and MuJoCo Warp dynamics. The
`contact_pipeline` selector (`auto` / `mujoco` / `newton`) and native MuJoCo CPU
implementation are removed; old configuration keys are rejected, not aliased.
CPU remains a single-world Mirror option and executes eagerly. CUDA additionally
supports graph execution and Kaleidoscope independent worlds. Solver persistent
state uses the same Warp float32 representation on both devices.

| Setting in both Newton leaves | Default | Meaning |
| --- | --- | --- |
| `iterations` / `line_search_iterations` | 100 / 50 | Upper bounds, not a fixed number of iterations per step |
| `solver_deterministic` | false | MuJoCo Warp solver determinism disabled |
| `collision_deterministic` | true | Newton collision ordering deterministic |
| `contact_matching` | disabled | No persistent matching; `latest` and `sticky` remain explicit choices |
| `default_contact_gap_m` | 0.0001 | Fallback detection gap (0.1 mm), not allowed penetration |

Explicit asset gap values, **including zero**, take precedence over the fallback.
The default is applied before import; shapes are not overwritten after import.
Read back the resulting native `geom_gap` when validating an asset. These are the
selected starting settings; performance tuning must be evaluated alongside task
outcomes. Asset geometry and joint limits are unchanged by these runtime settings.

### AR5/O6 contact policy

Both Newton leaves include `ar5_o6_contact`. The policy is applied once before
model finalization when **any AR5/O6 assembly**, including camera variants, is
present. It also applies to mixed O6/L6 scenes and every replicated world.
Without a supported O6 assembly the policy is inactive. PhysX is unaffected.
Set `ar5_o6_contact: null` in a custom physics leaf to disable this override.

| Setting | Default | Scope |
| --- | --- | --- |
| `cone` / `impratio` | `elliptic` / `100` | Entire solver scene, including other robots and object/table contacts |
| `condim` | `4` for all three roles below | Each configured collider; normal, two sliding and one torsional constraint; no rolling constraints |
| `solref` | `(0.004, 1)` | Normal contact reference on configured colliders |
| `friction_gain` | `1000` | Newton tangential gain `kf` on configured colliders; used to generate the friction reference |
| `margin_m` / `restitution` | `0` / `0` | Configured colliders |
| `geom_priority` / `geom_solmix` | `0` / `1` | Equal material priority and equal impedance mixing weights |
| `gap` | Existing asset value, otherwise `0.0001 m` | Separate import policy; this material override does not rewrite it |

| Role | Colliders | Sliding / torsional / rolling friction | `solimp` |
| --- | --- | --- | --- |
| `hand` | All links of the selected O6 hands, including palms | `(4, 0, 0)` | `(0.95, 0.95, 0.001, 0.5, 2)` |
| `object` | Dynamic scene objects and dynamic chains | `(1, 0.005, 0)` | `(0.9, 0.95, 0.001, 0.5, 2)` |
| `station` | Selected AR5 arms/accessories, static objects, table and remaining scene colliders | `(1, 0, 0)` | `(0.9, 0.95, 0.001, 0.5, 2)` |

Other robots retain their own collider material values. They still use the global
cone/impratio, and contacts with a configured object use both sides' materials.
This is a scene material policy, not an override limited to O6–object contact pairs:
object/table contacts also change. Covered material values override the imported
values; geometry, inertia, collision filtering and explicit asset **gap** are preserved.

With equal priority, the Newton bridge takes the componentwise maximum friction
and maximum dimension, and blends impedance using `geom_solmix`. O6–object contact
therefore has sliding friction **4**, torsional friction **0.005 m**, `condim=4`, and
`solimp=(0.925,0.95,0.001,0.5,2)`. Zero authored torsional friction is bounded by the
upstream numerical floor (`1e-5 m`) when it is needed; it is not a calibrated pad radius.
The raw normal reference is `(0.004,1)`, subject to the solver's timestep safety
bound (at 480 Hz the time constant is at least `2/480 s`). Friction reference is
computed separately from `kf`, body weights and `impratio`; it is not copied from
normal `solref`. A larger `impratio` does not guarantee zero slip.

Newton collisions remain in use on CPU/CUDA. No MuJoCo `MULTICCD` switch, contact
matching, iteration budget, drive gain, geometry or asset file is changed by this
policy. Older experiments that forced every gap to 0.0001 m are not an exact
reproduction of assets with explicit gap zero. Runtime diagnostics expose activation,
material shape counts and the solver cone/impratio. `just smoke-newton-contact`
checks actual CPU/CUDA material mixing in addition to contact integration.

### Newton contact integration

Mirror scenes explicitly set `control_frequency_hz: 60.0` and
`physics_frequency_hz: 480.0`. Both PhysX and Newton execute eight native steps
per target update, each lasting 1/480 s. Custom frequencies must have an integer
physics/control ratio, e.g. 120/600 or 20/240 Hz; invalid ratios fail at load time.
The dedicated hybrid-force entry retains 60 Hz commands / 240 Hz feedback / 480 Hz physics. Rendering and camera rates remain independent.

Kaleidoscope keeps its existing action decimation: the scene defaults to 480 Hz
physics and `task.action.physics_ticks_per_action: 8`, giving 60 Hz policy actions.
Each runtime step there is one physical step; no hidden Newton subdivision is added.

Remove the old Newton leaf keys `substeps` and `max_substep_dt_s` from custom
profiles. Counts are derived at composition, with no automatic timestep cap.
The internal runtime `get_physics_dt()` / `physics_dt_s` is the duration of one
runtime step (Mirror control tick or Kaleidoscope physical tick). Newton diagnostics
include `effective_substeps` and `substep_dt_s`; PhysX World's native physics dt
is the substep dt. Simulation frequencies are distinct from wall-clock throughput.
Contacts and native drives update on every physical step, while targets are held
between control updates. More physical steps cost throughput.

### Independent command, hybrid feedback and physics clocks

| Clock | Configuration | Default | Meaning |
| --- | --- | --- | --- |
| Mirror commands | `scene.control_frequency_hz` | 60 Hz | Ordinary joint/trajectory target updates; held between updates |
| Hybrid feedback | `hybrid_force_position.feedback_frequency_hz` | 240 Hz | Fresh state/wrench read and controller calculation on every feedback tick |
| Native physics | `scene.physics_frequency_hz` | 480 Hz | Actual solver integration, independent of wall-clock speed |
| Rendering | `scene.render_frequency_hz` | 60 Hz | Rendering cadence; does not add physical steps |
| Planning samples | `request_defaults.sample_dt_s` in the planning profile | 0.02 s (50 Hz) | Timestamped trajectory samples, resampled onto the command grid |

These are fixed, independently configurable values, not adaptive rates. A hybrid
profile requires positive integer ratios **feedback/command** and **physics/feedback**.
Ordinary profiles require an integer physics/command ratio. Invalid combinations fail
before Kit startup; no frequency is silently raised, rounded or clamped.

Default 60/240/480 Hz means four feedback calculations and eight physical steps per
command period. Changing commands to 120 Hz leaves feedback at 240 Hz. Raising physics
to 720 or 960 Hz leaves feedback at 240 Hz, with three or four physical steps per
feedback update. Commands at 480 Hz with feedback still at 240 Hz are rejected;
explicitly configure a compatible feedback/physics pair. Hybrid 60/240/600 Hz is also
rejected, although ordinary 120/600 Hz is valid.

Hybrid profiles rename the old `minimum_physics_frequency_hz` key to
`minimum_feedback_frequency_hz` (200 Hz); this existing safety floor checks the
**feedback** rate. The new `feedback_frequency_hz` is required. Filter cutoff and
effort ramp-down length are validated against that rate. Gains, the 20 Hz filter and
all feedback counts are unchanged. At 240 Hz, 30 warmup ticks last 0.125 s, 120 tare
samples take 0.5 s, contact enter/exit counts of 3/5 take 12.5/20.8 ms, and 36 ramp-down
ticks take 0.15 s. Changing only commands or physics preserves these durations.
If explicitly increasing feedback to 480 Hz, double those counts to preserve durations
(including stale-sensor counts); the original 36-tick unload is rejected because it
cannot meet the configured effort-rate bound. Custom feedback gains/filter behavior
must be revalidated.

Targets are held without an added interpolation layer; each feedback tick reads
new state and computes new effort. The existing hybrid API executes a complete
request with a frozen target: this change does **not** add streaming target updates
inside that request. Ordinary joint requests continue to use their existing trajectory
sampling. Hybrid and ordinary control advance the same physics owner without changing
its native timestep. Output timestamps use simulation time; output `step` identifies
a completed sampling boundary, not a uniform physical interval across both modes.
Planning keeps `sample_dt_s: 0.02`; timestamped trajectory points are resampled onto
the control grid and need not have an integer frequency ratio with control.

Newton leaves still require `default_contact_time_constant_s` (finite positive
seconds). This contact fallback is separate from the integration timestep.

The canonical 0.01 s contact fallback maps to Newton `ke=1/t²`, `kd=2/t`
(critical damping), yielding MuJoCo `solref=(t, 1)`. It is applied to the builder
before USD import. Authored per-shape Newton/MJC stiffness and damping retain the
normal upstream resolver precedence. For MJCF robots, a temporary source copy
sets the main geom default only when `solref` is omitted, before Isaac converts
it to USD. MuJoCo resolves includes and class inheritance; explicit geom or
inherited values remain authoritative. Source assets and PhysX imports are
unchanged. The temporary source preserves the configured 10 ms response, or a
task-selected fallback, through Isaac 6.1 conversion instead of silently replacing
it with MuJoCo's implicit 20 ms default.
This is a numerical contact response, not a
joint drive gain or a calibrated force-space material law. CPU and CUDA use the
same conversion. MuJoCo's `refsafe` still limits a
positive time constant to at least twice the actual internal timestep.

`diagnostics()` reports the outer and internal timestep, configured and effective
substeps, and the default contact response. Finite steps and compliant contacts can
still penetrate during impacts; validate the chosen configuration on the task's
velocities, masses and contact geometry. Recheck learned policies after changing it.

## Robot And Object Physics Leaves

A robot profile separates backend-neutral gravity policy from PhysX-only asset facts:

```yaml
robot:
  physics:
    gravity:
      default: false
      arm: false
      hand: false
    material:
      contact_static_friction: 0.8
      contact_dynamic_friction: 0.6
      contact_restitution: 0.0
    physx:
      material:
        friction_combine_mode: average
      rigid_body:
        linear_damping: 0.0
        angular_damping: 0.1
      joint:
        friction: 0.5
        follower_friction: 0.5
      solver:
        arm: {position_iterations: 32, velocity_iterations: 4}
        hand: {position_iterations: 32, velocity_iterations: 4}
```

Both engines consume `gravity` and `material`; only a PhysX composition projects the `physx` leaf.
Newton still receives backend-neutral `UsdPhysics.DriveAPI` seeds from its
engine-specific controller bundle, but it neither reads nor emits skip warnings for
PhysX combine mode, damping, joint-friction, or solver fields. For MJCF, source
`frictionloss` takes precedence over configured joint friction in PhysX. Newton's
upstream `SchemaResolverMjc` consumes the importer-authored `mjc:frictionloss`
directly.

Controller profiles own only control-law values that are actually consumed:
stiffness, damping, maximum force, effort limit, and follower drive seeds.
`joint_friction` is no longer a controller field and the parser does not recreate a
default. The retired `robot.physics.solver` path is invalid; the sole path is
`robot.physics.physx.solver`.

Object contact coefficients are standard USD material facts. PhysX extensions use a
separate leaf:

```yaml
object:
  physics:
    material:
      static_friction: 0.8
      dynamic_friction: 0.6
      restitution: 0.0
    physx:
      material:
        friction_combine_mode: average
```

A dynamic-chain object may also declare PhysX solver tuning in that leaf:

```yaml
object:
  physics:
    physx:
      solver:
        position_iterations: 48
        velocity_iterations: 4
```

Newton projects only the common object material, does not import `PhysxSchema`, and
does not report a valid PhysX leaf as compatibility loss. The retired
`physics.material.friction_combine_mode`, `physics.solver_position_iterations`, and
`physics.solver_velocity_iterations` paths fail strict validation; there are no
compatibility aliases.

## Environments And Backend Replication

```yaml
environments:
  num_envs: 256
  base_env_path: /World/envs
  env_prefix: env
  origin_xyz: [0.0, 0.0, 0.0]
```

`environments.num_envs` is the sole persistent configuration owner for the environment
count. An explicit `num_envs` environment-construction argument may override this default
without creating a second profile. The other three fields own stable USD path naming and
the base origin. There is no `profiles.replication` slot and no `configs/replication/`
directory.

Replication is still implemented internally, but it is inseparable from the physics
engine. The PhysX builder always uses GridCloner with 3.0 m spacing,
`replicate_physics=true`, `copy_from_source=true`, and `enable_env_ids=true`. The
Newton builder always uses the multi-world manager, zero spacing, and one
separate world per environment. Only the Newton session projection derives
`world_count`, and it derives it exclusively from the final `num_envs`; physics leaves
never declare a second world count.

The fixed isolation mechanisms prevent contact across environments. They do not enable motion
planning or avoidance. Physical robot/object contact remains enabled in both backends.

## Control, cuRobo, And Planning

Kaleidoscope has no `profiles.control` slot or `control` object. Its task fixes the
action and position-target semantics, while `physics.engine` derives the default
`physx` or `newton` controller bundle. The Newton bundle supplies its lower arm/hand
gains and zero follower drives; native equality constraints remain the only follower
mechanism.

The single `configs/control/mirror.yaml` profile selects Mirror's position, velocity,
or effort mode and owns idle stepping and `sync_simulation_to_wall_clock`,
admission and terminal-history capacities, stdin EOF behavior, response and queue-poll
timeouts, message/connection limits, startup/shutdown timeouts, joint interpolation,
default pose frame, and orientation mode. Its controller bundle is not configurable;
the physics engine derives it.

`sync_simulation_to_wall_clock: true` makes idle and motion execution share one pacing
clock at `scene.control_frequency_hz`. It does not alter physics dt. If a tick is late,
Mirror rebases the next deadline instead of burst-running missed ticks, so an overloaded
runtime may be slower than real time. Set the field to `false` to run physics as fast as
the host allows. The canonical Mirror control profile enables synchronization.

`configs/control/hybrid_force_position.yaml` is a separate optional profile. Its
`motion`, `force`, and `posture` sections provide the initial explicit Cartesian gains.
`tuning` gives immutable per-field maxima for owner-queued updates. `tare`, `contact`,
`limits`, filter cutoff, supported frame, allowed force axes, maximum duration, and
minimum frequency are construction-time safety facts and cannot be changed through
the wire protocol. Runtime updates use their own generation and do not alter the
semantic configuration fingerprint.

A `configs/curobo/*.yaml` profile is explicitly a numerical-capacity profile. Its root
contains `kinematics` with positive `max_batch_size`/`seed_count`, `collision_check`,
and `use_cuda_graph`. The backend fixes the validated cuRobo 0.8.0 task bundle and all
four numerical dtypes to `float32`; neither is a YAML choice. The mode root remains the
only CUDA-index owner.

Mirror's `curobo: mirror` profile must additionally contain `motion_planner`.
`kinematics.max_batch_size` belongs only to FK/IK; it does not size MotionPlanner.
Mirror's MotionPlanner context is fixed to one request (`max_batch_size=1`), while the
`motion_planner` section owns warmup, CUDA graph use, IK/trajopt seed counts, collision
capability, and collision-cache preallocation. Under the pinned runtime,
`motion_planner.use_cuda_graph` must be `false`; the IK graph may remain enabled.
Kaleidoscope's
`curobo: kaleidoscope_batch_ik` profile must omit `motion_planner`, disable kinematics
collision checks, and cover the final environment count with
`kinematics.max_batch_size`. The canonical profile omits
`kinematics.collision_cache`, but validation also permits a well-formed dormant cache
when `collision_check: false`. Runtime projects either form to an empty backend cache
and allocates none. Mirror's planner keeps `collision_check: true`, so its planner
cache remains required. The same rule applies to an optional MotionPlanner cache:
enabled collision checking requires it; disabled checking may omit or retain it, and
runtime ignores any retained value.

`configs/planning/mirror.yaml` is deliberately backend-neutral. It contains only
`planning.request_defaults`: `duration_s`, `sample_dt_s`, `timeout_s`,
`avoid_collisions`, `force_collision_refresh`, and `coordination: independent`.
The wire-level planning overrides are `duration_s`, `sample_dt_s`,
`avoid_collisions`, and `force_collision_refresh`; `coordination` can be overridden
only at the one-segment wrapper or timeline top level. `timeout_s` is not a wire
field: every request uses `planning.request_defaults.timeout_s`. This profile is not
cuRobo solver capacity or a backend selector. If avoidance defaults to true, the
selected cuRobo profile must provide planner collision capability and cache capacity.

## Kaleidoscope Task

A task profile contains exactly:

- `id`, `dynamic_object`, and unit `heading_axis`;
- one fixed `action` variant;
- observation switches;
- reward coefficients;
- termination/horizon thresholds;
- reset randomization ranges.

`task.id` must match the profile basename, and `dynamic_object` must name the selected
scene's unique non-static rigid object.

Action variants are `joint_control`, `joint_delta`, `ee_delta_position`, `ee_delta_pose`,
`ee_pose_position`, `ee_pose_full`, `ee_linear_path_position`, and
`ee_linear_path_full`. Each variant has an exact field set. Linear variants require a
fixed waypoint count and progress mode. Failure policy is fixed by variant; there is
no backend, planner, profile reference, or avoidance field in the task action. The
mode composition supplies the conditional `profiles.curobo` reference described above.

`joint_control` requires `position_delta_scale_rad`, `velocity_scale_rad_s`,
`effort_limit_fraction`, `clip`, and `physics_ticks_per_action`. It is the canonical
variant and supports all three runtime control modes without changing action shape.
Position-reference variants require `reference_velocity_limit_rad_s` for bounded
velocity conversion and reject effort mode. Initial position mode, active mode, and
generation are runtime state, not YAML selectors and not semantic configuration
fingerprint inputs.

## Mirror Outputs

The output profile contains exactly `render`, `camera`, `logging`, and `telemetry`.
Hybrid diagnostics add `logging.hybrid_control_path`/`log_hybrid_control` and
`telemetry.include_hybrid_control`/`topics.hybrid_control`; they do not add another
top-level output section.
See [Outputs](outputs.md) for field-level behavior. Kaleidoscope has no output profile
reference.

## Trainer Profiles

Files under `configs/training/` configure a downstream consumer. They do not alter
physics or environment construction. `device_source: environment` means the trainer
inherits the canonical environment device and must not add another GPU index.
