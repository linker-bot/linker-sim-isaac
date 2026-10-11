# Kaleidoscope API Reference

Language: [English](kaleidoscope-api.md) | [中文](../../zh-CN/reference/kaleidoscope-api.md)

Kaleidoscope is a process-local CUDA vector environment with a renderer-free training
path and an explicit human viewport boundary. It does not expose a JSON server or
remote procedure protocol. Import public objects from
`linkerbot_sim.kaleidoscope`; framework adapters live under
`linkerbot_sim.training.skrl`.

## Construction

```python
from linkerbot_sim.kaleidoscope import make_torch_env

env = make_torch_env(
    profile="physx_cuda",
    num_envs=256,
)
```

`make_torch_env(*, config=None, profile="physx_cuda", num_envs=None,
runtime_factory=None)` loads a strict configuration unless `config` is supplied. A
`num_envs` override must be a positive integer. Production construction accepts
PhysX/CUDA and project-owned Newton/CUDA; their internal runtime kinds remain
`physx_cuda` and `newton_cuda`. `runtime_factory` is an injection seam
for tests.

| Profile | Physics leaf | Internally derived environment realization | Kit selected by the factory | Physics owner |
| --- | --- | --- | --- | --- |
| `physx_cuda` | `physx/cuda` | GridCloner with environment IDs | `linkerbot_sim.kaleidoscope.physx_cuda.python.kit` | PhysX CUDA/Fabric replicated scene |
| `newton_cuda` | `newton/cuda` | Multi-world with separate worlds | `linkerbot_sim.kaleidoscope.newton.python.kit` | Project `NewtonRuntime`, one world per environment |

Both training compositions are headless and GPU-native and expose the same public
tensor contract. Newton is selected before construction and is not the Isaac Newton
extension; `physics.engine` derives its Newton-specific controller bundle, and a live environment
cannot switch backends.

The bundled roots use `joint_control` and omit `profiles.curobo`. For an end-effector or
linear action, the task still contains only action semantics; its mode root must add
`profiles.curobo: kaleidoscope_batch_ik`. The catalog rejects that reference for
joint-only actions and requires it for every EE/linear action.

Public environment attributes:

| Attribute | Meaning |
| --- | --- |
| `num_envs` | Replicated environment count. |
| `device` | Canonical CUDA device shared by physics, task, state, and actions. |
| `action_dim` | Fixed action width selected by the task profile. |
| `observation_dim` | Fixed flattened observation width. |
| `supports_articulated_collision_queries` | Whether this constructed environment has the PhysX articulated-query capability. |
| `articulated_collision_query_specs` | Immutable mapping from queryable robot labels to their runtime command metadata. |

## Native Torch Environment

### `reset`

```python
observations, info = env.reset(seed=None, options=None)
```

Resets every row and returns CUDA observations plus a CUDA info mapping. A seed is a
cold deterministic reseed boundary. Dynamic reset options are rejected.

### `reset_idx`

```python
observations, info = env.reset_idx(env_ids)
```

Resets selected rows. `env_ids` must be a one-dimensional CUDA `torch.int64` tensor on
`env.device`, with unique in-range values.

### Portable Reset Scene Descriptions

A reset scene description is a JSON-compatible fresh-episode record with no
PhysX- or Newton-private state. It contains robot joint positions, velocities and
targets, the dynamic object pose and velocity, the task goal, and the RNG
continuation. Generate and store it without Torch, CUDA, Isaac, PhysX, or Newton:

```python
from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope import (
    load_reset_scene_description,
    sample_reset_scene_description,
    save_reset_scene_description,
)

config = load_kaleidoscope_config("physx_cuda")
description = sample_reset_scene_description(
    config,
    seed=7,
    sample_id=12,
    reset_index=4,
)
path = save_reset_scene_description(description, "runs/reset-12-4.json")
loaded = load_reset_scene_description(path)
```

`sample_id` identifies an independent deterministic RNG stream. `reset_index`
directly addresses one reset in that stream; it does not replay earlier resets. The
current sampler supports `tblock_push_v1`. Saving writes canonical UTF-8 JSON
atomically and rejects an existing destination unless `replace=True`. Loading checks
the portable schema and numeric values but does not start a simulation.

Not storing backend-private state does not promise cross-backend restore. The
compatibility fingerprint covers the complete Kaleidoscope configuration, including
the `physx_cuda` or `newton_cuda` selection. Version 1 promises restore in another
process with the same semantic configuration; cross-backend restore is outside the
Issue #20 v1 contract.

Apply the loaded description to exactly one native Torch environment row:

```python
env = make_torch_env(config=config, num_envs=4)
observations, info = env.reset_from_description(loaded, env_id=2)
```

Before any control or physics write, the runtime requires the semantic configuration
fingerprint, scene/task IDs, ordered robot labels and profiles, command-joint names,
dynamic-object identity, and derived RNG metadata to match. It then writes task
goal/RNG state, robot/object state and action state, forwards physics, refreshes the
selected row, and initializes its episode observation. The next ordinary reset of
that row continues from the stored RNG key and counter rather than sampling twice.

This method is an explicit CPU-to-CUDA cold path, not a training-step API. It is not
available through `KaleidoscopeTrainingPort` or the Gymnasium reset options. Read the
JSON before calling it; file paths are deliberately not accepted. Any apply failure
makes the runtime fail-stop, even when compatibility rejection occurs before a
physics write. Close and recreate the environment after such an error.

### `step`

```python
observations, rewards, terminated, truncated, info = env.step(actions)
```

`actions` must be finite CUDA `torch.float32` with shape
`(num_envs, action_dim)`. The returned leading dimension is `num_envs`. Every done row
must be reset before the next native step. The native/debug entry performs one
synchronous scalar readback so that this recoverable lifecycle error is raised before
physics advances. The skrl adapter uses the tokenized SAME_STEP entry instead; that
training path does not execute this guard and keeps observations, actions, resets, and
rollout data on CUDA.

### Articulated Arm-and-Hand Collision Query

This explicit diagnostic API checks caller-provided joint samples without advancing
physics or moving the active robot:

```python
import torch

from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope import (
    CollisionHitCategory,
    CollisionQueryTarget,
    make_torch_env,
)

config = load_kaleidoscope_config("physx_cuda")

env = make_torch_env(config=config, num_envs=2)
try:
    env.reset(seed=123)
    if not env.supports_articulated_collision_queries:
        raise RuntimeError("articulated collision query is unavailable")
    robot_label, query_spec = next(
        iter(env.articulated_collision_query_specs.items())
    )

    def check_candidates(candidate_q: torch.Tensor):
        # candidate_q comes from a policy, IK solver, sampler, or external
        # planner and has shape [selected_envs, samples, command_joints].
        if candidate_q.shape[-1] != query_spec.command_dim:
            raise ValueError("candidate command width does not match query spec")
        env_ids = torch.arange(
            candidate_q.shape[0],
            device=env.device,
            dtype=torch.int64,
        )
        return env.query_articulated_collisions(
            robot_label=robot_label,
            joint_positions=candidate_q,
            env_ids=env_ids,
            targets=(
                CollisionQueryTarget.TABLE,
                CollisionQueryTarget.DYNAMIC_OBJECT,
                CollisionQueryTarget.OTHER_ROBOT,
            ),
        )
finally:
    env.close()
```

The candidate tensor contract is strict:

- `joint_positions` is finite CUDA `float32` with shape `[K, S, J]` on
  `env.device`; `S >= 1` and a single pose uses `S = 1`;
- `K` equals the length of `env_ids`;
- `J == query_spec.command_dim` and columns follow
  `query_spec.command_joint_names` exactly;
- every value must be inside the real articulation command limits; and
- `env_ids` is unique, in-range CUDA `int64` with shape `[K]`.

Each `ArticulatedCollisionQuerySpec` publishes the exact post-import
`command_joint_names`, `command_dim`, `joint_lower`, and `joint_upper` used by the
query. The mapping, dataclass, and nested tuples are immutable. Reading it does not
run a query, synchronize CUDA, or advance physics. For `newton_cuda` or an
environment without this capability, the property is an immutable empty mapping.
The query consumes positional columns and does not reorder named values.

`targets` must be a non-empty tuple without duplicates. `TABLE` selects the static
workstation, `DYNAMIC_OBJECT` selects the task's unique dynamic object (the T-block in
the canonical scene), and `OTHER_ROBOT` selects the other robot in the same
environment. The current robot, query-only shadows, and other environments are
filtered out. This first API does not check collisions within the queried
articulation itself.

The result owns three contiguous CUDA tensors:

| Field | Shape | Meaning |
| --- | --- | --- |
| `collision_mask` | `[K, S]` bool | Whether each discrete sample overlaps a selected target. |
| `first_hit_sample` | `[K]` int64 | First true sample index in each row; `-1` means no hit. |
| `first_hit_category` | `[K]` int64 | `CollisionHitCategory`; `NONE` is used when the sample index is `-1`. |

The stable category codes are `NONE=0`, `TABLE=1`, `DYNAMIC_OBJECT=2`, and
`OTHER_ROBOT=3`. If several categories hit at the first colliding sample, the lowest
code is reported. `collision_mask` remains the complete per-sample Boolean result.

Samples are checked independently. The API does not interpolate between them and is
not continuous collision detection. It also does not generate candidates, execute an
action, or call cuRobo. Candidate samples come from the caller.

Only `physx_cuda` currently supplies this capability. `newton_cuda` reports
`supports_articulated_collision_queries == False` and raises
`CollisionQueryUnavailableError` if called. Capability discovery does not bypass
lifecycle rules: a query is accepted only while the Runtime is idle, on its owner
thread, and outside an outstanding SAME_STEP transaction.

The call is a synchronous slow path. It deliberately copies the selected candidate
values to the host, evaluates the full articulated arm-and-hand FK, updates private
query-only PhysX shadows, performs scene overlaps, and returns owned CUDA results.
Do not call it implicitly from every skrl step. `KaleidoscopeTrainingPort`, Gymnasium,
and skrl do not expose this method.

### Runtime Control Mode

The native environment starts in position mode. Between complete `step` calls, query
or switch every robot in every environment without recreating the environment:

```python
state = env.get_control_mode()
change = env.set_control_mode(
    "velocity",
    expected_generation=state.generation,
)
```

`ControlModeState` reports the immutable `initial_mode`, current `active_mode`,
monotonic `generation`, action-dependent `supported_modes`, and `scope="all"`.
Selecting the active mode is idempotent and does not increment generation; an expected
generation mismatch is always rejected first.

The switch is accepted only while the runtime is idle. It is rejected during step,
reset, another switch, close, or either the issued or stepped phase of an outstanding
SAME_STEP token. A real switch freezes current joint positions, writes the old-mode
neutral target, applies every robot profile, writes the new-mode neutral target, and
synchronizes before committing host state. Position neutral is current `q`; velocity
and effort neutral are zero. A forward failure rolls touched robots back in reverse
order. If rollback also fails, the environment permanently fail-stops and must be
closed and recreated.

The fixed action variant controls which modes are legal:

| Action variant | Position | Velocity | Effort |
| --- | --- | --- | --- |
| `joint_control` | yes | direct bounded target | direct profile-limited target |
| `joint_delta` and all EE/linear variants | yes | bounded derivative of the position reference | no |

Switching changes joint command units, not action shape or the configured action
variant. `KaleidoscopeTrainingPort`, Gymnasium, and skrl deliberately expose no mode
setter; training stays in the initial position mode.

### `close`

`close()` is idempotent. It closes the action term and IK resources, tensor views and
task, then the unique `IsaacSession`. A child close failure prevents premature session
destruction and can be retried.

## GPU State API

The environment delegates these methods to an internal transactional state service:

```python
state = env.get_state(env_ids=None, fields=None, clone=True)
env.set_state(state, env_ids=None)
snapshot = env.snapshot(env_ids=None, fields=None)
env.restore_snapshot(snapshot, target_env_ids=None)
env.clone_state(source_env_ids, target_env_ids, include_rng=True, fields=None)
```

Both backends bind the same core canonical fields. Task/history/counter/RNG fields are
then appended by the assembled task:

| Field | Shape and meaning |
| --- | --- |
| `robot.q` | `(N, Q_full)`, all articulation DOFs from every robot, concatenated in scene order. |
| `robot.qd` | `(N, Q_full)`, velocities for the same full articulation DOFs. |
| `robot.target` | `(N, Q_controlled)`, active-mode engine target: rad, rad/s, or effort according to snapshot mode metadata. |
| `robot.position_reference` | `(N, Q_controlled)`, persistent position reference in radians in every mode. |
| `object.pose_local_wxyz` | `(N, 7)`, environment-local XYZ position followed by a `wxyz` quaternion. |
| `object.com_velocity` | `(N, 6)`, object center-of-mass linear and angular velocity. |

Newton also binds the backend-private `solver.persistent` matrix. It stores
per-world SolverMuJoCo TIME, ACT, and WARMSTART state on CUDA and participates in
default get/set, snapshot/restore, reset, and clone operations. PhysX has no matching
field. Complete snapshot fingerprints therefore differ by backend and cannot be
restored across PhysX and Newton, even though the core fields above are shared.

These `object.*` fields own the selected scene's only non-static rigid object. Strict
configuration rejects a second dynamic rigid object or any dynamic chain, so
snapshot and clone cannot silently omit another dynamic object's state.

For example, a selective read uses the canonical names directly:

```python
state = env.get_state(
    env_ids,
    fields=("robot.q", "object.pose_local_wxyz"),
)
```

### Selectors

An omitted selector means all rows and creates a CUDA `arange`. An explicit selector
must already be CUDA `int64` on the canonical device. The API does not hide host-to-
device transfers with `as_tensor`.

### `get_state`

Returns a field-to-CUDA-tensor mapping. The default `clone=True` returns owned storage;
`clone=False` may return runtime-owned storage for a full selection and must not be
mutated or retained across runtime operations.

### `set_state`

The leading payload dimension must equal the selector length, dtype and trailing
shape must match each canonical field, and finite fields reject non-finite values.
All fields are preflighted before any engine writer runs. If an engine writer raises,
the state API and runtime become fail-stop and must be closed and recreated.

### `snapshot` And `restore_snapshot`

`snapshot` returns a `KaleidoscopeEpisodeSnapshot` whose selector and fields own
cloned storage on the same GPU. Restoration targets the captured IDs by default or
an equal-length explicit selector. Schema 2 stores `control_mode` and
`control_generation`. Restore requires the runtime to already be in the same mode and
never switches it automatically; generation is diagnostic and is not restored.
Legacy schema 1 is position-only and derives a missing `robot.position_reference`
from `robot.target`. The selected physics runtime is forwarded afterward.

The observation field `command_target_error` always means
`position_reference - actual_q` in radians. It is not velocity or effort tracking
error. Velocity actions advance the shadow reference by decision duration; effort
actions anchor it to actual `q` at the start of each decision.

### `clone_state`

Copies `K` source rows to `K` target rows entirely on the GPU. The two selectors must
have equal lengths and cannot overlap. RNG key/counter fields are copied by default,
which makes the first rollout after cloning reproducible. `include_rng=False` excludes
registered RNG fields; non-cloneable fields are always skipped. The selected physics
runtime is forwarded once after the transaction.

## Snapshots Versus Checkpoints

A `KaleidoscopeEpisodeSnapshot` is a hot, same-process GPU object. For deliberate
persistence, use the cold boundary:

```python
from linkerbot_sim.kaleidoscope.checkpoint import (
    load_kaleidoscope_checkpoint,
    save_kaleidoscope_checkpoint,
)

path = save_kaleidoscope_checkpoint(snapshot, "runs/checkpoint.npz")
restored = load_kaleidoscope_checkpoint(path, device=env.device)
```

Saving performs an explicit device-to-host transfer and writes compressed NPZ without
pickle. Loading performs an explicit host-to-device transfer. Neither operation is
called by `step` or `reset`.

## Fixed Action Variants

The task profile freezes one action variant at construction:

| Mode | Per-robot input | Execution |
| --- | --- | --- |
| `joint_control` | controlled-joint width | Position delta, bounded velocity, or profile-limited effort according to active control mode. |
| `joint_delta` | controlled-joint width | Accumulate, scale, clip, and clamp joint targets. |
| `ee_delta_position` | XYZ delta | Device-batched IK. |
| `ee_delta_pose` | XYZ plus rotation delta | Device-batched IK. |
| `ee_pose_position` | XYZ target | Device-batched IK. |
| `ee_pose_full` | XYZ plus `wxyz` target | Device-batched IK. |
| `ee_linear_path_position` | XYZ target | Fixed-count synchronous waypoints and batched IK. |
| `ee_linear_path_full` | XYZ plus `wxyz` target | Fixed-count synchronous waypoints and batched IK. |

Every action component must be finite. The `wxyz` slice used by `ee_pose_full` and
`ee_linear_path_full` must additionally have norm greater than `1e-8` for every
environment; the runtime normalizes valid quaternions on CUDA. The unbounded Box is a
flat interoperability description and does not make a zero quaternion valid.

Task action variants never contain a backend or profile reference. End-effector and
linear compositions select `curobo: kaleidoscope_batch_ik` in the mode root. That
numerical profile must omit `motion_planner`, set
`kinematics.collision_check: false`, and cover the effective environment count. Its
canonical YAML omits `kinematics.collision_cache`; a valid retained value is accepted
but not passed to the backend, so no collision cache is allocated. They do not
construct a planner or collision world. Failures hold the affected target and
truncate or penalize according to the frozen variant policy. There is no asynchronous
batch trajectory-planning API.

## Gymnasium Adapter

```python
from linkerbot_sim.kaleidoscope import make_gymnasium_env

env = make_gymnasium_env(
    profile="physx_cuda",
    num_envs=64,
    autoreset_mode="disabled",
)
```

`GymnasiumKaleidoscopeAdapter` is a Gymnasium 1.3 `VectorEnv`. It accepts and returns
NumPy arrays, so every step crosses the device boundary. Action shape is
`(num_envs, action_dim)` with finite `float32` values. For `joint_control` and
`joint_delta`, the Box
bounds are `[-task.action.clip, task.action.clip]`; EE and linear modes use unbounded
Box limits because their values are raw metres, rotation vectors, or quaternions. Full
pose quaternion slices remain subject to the finite, nonzero-norm contract above.

Supported autoreset modes:

- `disabled`: caller resets done rows;
- `same_step`: done rows are reset before returning, while `final_obs` and
  `_final_obs` preserve terminal data.

With `disabled`, perform a partial reset through the VectorEnv reset contract before
the next step:

```python
done = terminated | truncated
observations, reset_info = env.reset(options={"reset_mask": done})
```

`reset_mask` must be a Boolean NumPy array with shape `(num_envs,)`. The adapter does
not expose the native CUDA-only `reset_idx` method.

The adapter rejects `next_step`. Its default `render_mode=None` constructs the
renderer-free training environment. `render_mode="human"` constructs the explicit
viewport environment described below; rendering remains an explicit method call.

Registration is explicit and idempotent:

```python
from linkerbot_sim.kaleidoscope import register_gymnasium_envs

register_gymnasium_envs()
```

The registered ID is `linkerbot/TBlockPush-Kaleidoscope-v1`.

## Human Viewport Boundary

```python
from linkerbot_sim.kaleidoscope import make_viewport_env

env = make_viewport_env(
    profile="physx_cuda",
    viewport_profile="kaleidoscope",
    num_envs=4,
)
try:
    observations, info = env.reset()
    # Every internal physics tick in env.step(actions) still uses render=False.
    env.render()
finally:
    env.close()
```

`make_viewport_env` accepts an already loaded `KaleidoscopeViewportSettings` as
`viewport`, or loads `configs/visualization/kaleidoscope.yaml` through the default
`viewport_profile="kaleidoscope"`. Selecting that profile always creates the human
viewer window; it owns `selected_env`, render cadence, window/renderer settings, and
scene visuals. It is not part of `KaleidoscopeConfig`, so display changes do not alter episode
snapshot/clone fingerprints.

PhysX and Newton select their corresponding `*_viewport.python.kit` experiences.
Only `selected_env` is synchronized into renderer-facing USD; every other environment
continues in the GPU physics batch without per-world RTX display state. `env.step()`
never renders implicitly: all physics ticks use `step(render=False)`, and callers
invoke `env.render()` according to `render_every_n_steps`. A render-only call does not
advance simulation time.

This is a human viewport, not a sensor API. The viewport Kits continue to exclude
cameras, SyntheticData, Replicator, recording, image observations, and telemetry.

## skrl CUDA Path

`SkrlTorchAdapter` consumes only the public `KaleidoscopeTrainingPort`. Its
`begin_same_step`, `step_same_step`, and `complete_same_step` handshake copies the
terminal transition before resetting done rows and returns dense CUDA info tensors.
Neither the training port nor this adapter exposes `set_control_mode`.

Use it with:

- `CudaRolloutMemory`, which builds selectors and mini-batches on CUDA;
- `FinalObservationPPO`, which bootstraps truncated rows from final observations and
  keeps PPO update data on device.

Before allocating rollout memory or the agent, the trainer factory requires both
policy and value models to use exactly `env.device`. Their Gymnasium Box
`observation_space`, `state_space`, and `action_space` must also match the environment
in type, shape, dtype, and bounds.

The integration pins skrl 2.1.0 and source-fingerprints replaced upstream methods.
Version drift is a hard error requiring a new audit.

## Deliberate Non-API

Kaleidoscope exposes no camera or image-observation API, trajectory planner,
avoidance query, planning collision world, playback queue, transport, telemetry sink,
or runtime backend switch. Training mode configuration rejects those capabilities and
renderer keys. Human display is available only through the separate launch-only
viewport profile and does not add SyntheticData, Replicator, or recording.
