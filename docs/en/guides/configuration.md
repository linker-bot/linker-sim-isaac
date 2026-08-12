# Configuration Guide

Language: [English](configuration.md) | [中文](../../zh-CN/guides/configuration.md)

Treat configuration as a typed dependency graph, not as a stack of mutable overlays.
Start with a mode root, follow its profile references, and change the leaf that owns
the fact.

## Choose A Composition

Mirror roots live under `configs/modes/mirror/`:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
```

Kaleidoscope roots live under `configs/modes/kaleidoscope/`:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile newton_cuda
```

The validator runs before Kit and prints the selected sources and a deterministic
fingerprint. Put that fingerprint in run metadata so a result can be tied back to its
exact graph.

## Close The Entire Graph

A mode load continues beyond the directly referenced leaves:

```text
mode -> leaf profiles -> scene robot/object profiles -> effective controller bundles
```

Controller selection precedence is the scene instance `controller_profile`, then the
robot-profile default, then the default derived from `physics.engine`. The frozen root
contains each instance's `resolved_profile` and a read-only `controller_bundles`
mapping. Runtime builders consume only those resolved objects; they do not look up the
same names again under the repository's default root.

Therefore a custom `configs_root` must contain the full referenced `robots/`,
`objects/`, and `controllers/` closure. Missing files fail before Kit starts. The
read-only `sources` mapping records provenance for diagnostics, but absolute paths do
not enter the semantic fingerprint. Configuration owns one canonical fingerprint
implementation shared by the validator and Kaleidoscope snapshot compatibility:
moving identical content to another absolute root preserves the fingerprint, while a
semantic robot, object, or controller change updates it.

## Change The Owning Leaf

| Desired change | Edit |
| --- | --- |
| Robot/object instance pose | Selected scene profile |
| Robot/object asset property | Referenced robot/object profile |
| Robot joint friction, rigid-body damping, material combine mode, or per-body solver iterations | Robot profile `robot.physics.physx` leaf |
| Object common contact coefficients | Object profile `object.physics.material` |
| Object combine mode or dynamic-chain solver iterations | Object profile `object.physics.physx` leaf |
| Mirror backend | Mirror mode's `profiles.physics` reference |
| Kaleidoscope backend | Kaleidoscope mode's `profiles.physics` reference |
| Kaleidoscope GPU | Mode root `compute.cuda_device` |
| PhysX process-memory budget | `configs/physics/physx/cuda.yaml` `physics.memory` |
| Environment count and path naming | Kaleidoscope mode-root `environments` mapping |
| Action semantics | Task profile |
| Default asset drive bundle | Derived from `physics.engine`; robot/scene profiles may override it; controller profiles do not own joint friction |
| cuRobo IK batch capacity plus MotionPlanner warmup, seeds, CUDA graph, collision capability, and cache capacity | Selected `configs/curobo/` profile |
| Mirror wire defaults for duration, sampling, avoidance, refresh, and coordination, plus the fixed per-request planner timeout | `configs/planning/mirror.yaml` |
| Camera geometry | Mirror scene profile |
| Camera encoding/queue | Mirror output profile |
| Trainer hyperparameters | Training profile |
| Kaleidoscope debug viewport and selected environment | `configs/visualization/kaleidoscope.yaml` |

Do not copy a leaf field into the mode root to override it. Create a focused new leaf
profile and point a new mode root at it.

## Add A Mirror Profile

1. Reuse or create a scene under `configs/scenes/mirror/` with render, camera,
   viewport, and light fields explicitly present. Reference it with a namespaced
   selector such as `mirror/scene3`; keep `scene.id` equal to the file basename.
2. Select `physx/cpu`, `newton/cpu`, or `newton/cuda`; both Newton leaves project to the project runtime.
3. Reuse the single `control: mirror`; create cuRobo, planning, and output leaves as needed. The cuRobo leaf owns
   numerical capability; the planning leaf owns only backend-neutral request defaults.
4. Add a root under `configs/modes/mirror/` containing only the profile references
   and the shared `compute.cuda_device`.
5. Validate both the new root and the existing bundled roots.
6. Run the relevant backend smoke test before treating the composition as deployable.

Mirror scene profiles may describe visual facts. Its cuRobo profile owns planner
collision capability and cache capacity, while the planning profile only decides the
default request policy. `planning.request_defaults.avoid_collisions: true` therefore
requires `curobo.motion_planner.collision_check: true`.

`curobo.kinematics.max_batch_size` sizes FK/IK only. Mirror's MotionPlanner is fixed
to one request at a time; its leaf still owns warmup, graph, IK/trajopt seeds,
collision capability, and cache capacity. On the wire, planning segments may override
`duration_s`, `sample_dt_s`, `avoid_collisions`, and `force_collision_refresh`, while
`coordination` is a wrapper/timeline-level override. They cannot override `timeout_s`;
the planner always uses `planning.request_defaults.timeout_s`.

## Add A Kaleidoscope Task

1. Create a headless template scene under `configs/scenes/kaleidoscope/`. Reference
   it with a `kaleidoscope/<stem>` selector, keep `scene.id` equal to `<stem>`, and do
   not add render frequency, cameras, viewport, or lights.
2. Keep exactly one non-static rigid object and name it in the task. Other objects
   must be static rigid objects; dynamic chains are outside the state schema.
3. Freeze one action variant in the task profile. The task does not select a numerical
   backend or carry a profile reference.
4. For an end-effector or linear action, add `profiles.curobo: kaleidoscope_batch_ik`
   to the mode root. That profile must omit `motion_planner`, set
   `kinematics.collision_check: false`, and cover the final effective environment
   count with `kinematics.max_batch_size`. The canonical profile omits
   `kinematics.collision_cache`; a retained valid cache is accepted but discarded
   before backend construction, so neither form allocates it. A joint-only
   `joint_control` or `joint_delta` composition must omit `profiles.curobo` and loads no
   cuRobo context.
5. Declare `num_envs`, `base_env_path`, `env_prefix`, and `origin_xyz` once in the
   Kaleidoscope mode-root `environments` mapping.
6. Select either PhysX/CUDA with Fabric and scene-query support disabled, or
   Newton/CUDA. The engine determines the internal replication plan; it is not a
   separate profile choice.
7. Validate the graph, then run GPU residency, composition, and real-physics smoke
   gates at representative environment counts.

The catalog recursively rejects renderer, camera, transport, planning, playback,
and telemetry keys in the Kaleidoscope closure. Its optional cuRobo profile is a narrow
kinematics-only numerical capability, not a planner exception. A rejected field should
be removed, not renamed to bypass the check.

The optional human viewport is intentionally outside that closure. Selecting the
standalone `KaleidoscopeViewportSettings` profile always opens the human-viewer window;
the profile owns the selected environment, render cadence, window/renderer settings,
and scene visuals. `make_viewport_env()` either selects it
through `viewport_profile="kaleidoscope"` or accepts an already loaded object through
`viewport`; Gymnasium `render_mode="human"` selects it through `viewport_profile`. It
does not enter the training configuration or
episode snapshot/clone fingerprint. Both backends render only `selected_env`; physics
ticks remain `render=False`, and callers invoke `env.render()` explicitly. This
profile does not enable cameras, SyntheticData, Replicator, recording, or image
observations.

This removes planning collision queries and avoidance, not physical contact. Both
PhysX CUDA and Newton still resolve rigid-body contacts; manipulation rewards
such as T-block pushing depend on them.

### Configure The PhysX GPU Memory Gate

The PhysX leaf's process-level `memory` mapping is the complete
`GpuMemoryBudget`:

| Field | Meaning |
| --- | --- |
| `max_simulator_process_mib` | Absolute NVML memory limit for the simulator PID |
| `min_free_floor_mib` | Minimum free MiB retained on the selected device at every audit phase |
| `min_free_fraction_after_warmup` | Minimum free fraction after warmup and at both steady samples; range `(0, 1]` |
| `max_steady_growth_mib` | Maximum PID-memory growth from steady baseline to steady final; zero is valid |

Do not move these facts into the mode root, task, or training profile, and do
not substitute Torch allocator counters for process-level NVML sampling. After strict
validation, run the maintained recipe or its underlying entrypoint on the target GPU:

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

The script accepts only the Kaleidoscope `physx_cuda` profile. Newton has
a separate world-capacity smoke.

The bundled compositions are intentionally explicit:

| Mode profile | Physics leaf | Internally fixed replication implementation | Runtime ownership |
| --- | --- | --- | --- |
| `physx_cuda` | `physx/cuda` | GridCloner; 3.0 m spacing; `replicate_physics=true`; `copy_from_source=true`; `enable_env_ids=true` | PhysX CUDA tensor pipeline and Fabric |
| `newton_cuda` | `newton/cuda` | Multi-world; zero spacing; one separate world per environment | Project-owned Newton |

The shared `newton/cuda` leaf owns per-world contact/Jacobian capacities, CUDA graph
selection, substeps, iterations, line search, constraint solver, and contact
pipeline. Kaleidoscope derives the runtime world count only from the final
`environments.num_envs`; the Newton builder always uses separate worlds and
`physics.engine` derives the `newton` controller bundle. It selects the project-owned
Newton runtime, not an Isaac Newton extension configuration.

## Device Ownership

Write the GPU index once:

```yaml
mode: kaleidoscope
compute:
  cuda_device: 0
```

The catalog injects this value into the concrete PhysX or Newton composition.
Torch, Warp interop, cuRobo, tensor views, task buffers, skrl memory, and policy
construction must consume the resolved device. A leaf-level device value is a
configuration bug even when it happens to match.

## Environment Count Overrides

Mode-root `environments.num_envs` is the canonical configured count. The Python factory accepts a
positive `num_envs` override for controlled experiments:

```python
env = make_torch_env(profile="physx_cuda", num_envs=32)
```

Use that runtime argument only for deliberate scaling tests. Record the effective
count beside the configuration fingerprint; do not add a second count to the task or
scene.

## Exactness And YAML Hygiene

- Use `.yaml` for new graph files.
- Keep each file rooted at its declared category key.
- Use YAML booleans, not `0`, `1`, or strings.
- Keep numeric values finite.
- Use absolute USD paths where the schema requires them.
- Do not use duplicate keys, aliases to escape ownership, or path traversal.
- Keep profile references extension-free and relative to their category.

Unknown fields are errors. This is intentional: a misspelled safety or device field
must not be silently ignored.

## Training Configuration

Training profiles are downstream of the environment:

```yaml
training:
  framework: skrl
  algorithm: final_observation_ppo
  device_source: environment
  rollout_length: 32
  mini_batches: 8
  learning_epochs: 5
  learning_rate: 0.0003
  discount_factor: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
```

They may choose optimization parameters but may not redefine the environment's CUDA
device, action space, physics frequency, reward, or termination semantics.

For every field and cross-profile invariant, see the
[Configuration Reference](../reference/configuration.md).
