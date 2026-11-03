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

The default Newton AR5/O6 `newton_compliant` profile additionally binds the float64
`controller.virtual_shaft` matrix (virtual references, velocity history, interval
endpoints and command substep phase). Preserve its dtype in state transfers; it
participates in snapshot, restore, clone and selected reset.

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

## Offline Reset Sampling And Portable Initial States

For `tblock_push_v1`, export the actual imported nominal pose once from an idle
native environment. This cold operation reads nominal joints to the host and
hashes configuration/root assets; it neither samples RNG nor advances physics.
Do not infer nominal joints as zero or duplicate asset import logic in a task.

```python
from pathlib import Path
from linkerbot_sim.kaleidoscope import reset_json

template = env.get_reset_template()
Path("reset-template.json").write_text(reset_json(template), encoding="utf-8")
```

The saved template can then be sampled in a CPU-only process with no Torch, Kit,
CUDA, physics session, or local asset files:

```python
import json
from pathlib import Path
from linkerbot_sim.kaleidoscope import (
    KaleidoscopeResetTemplate, sample_reset_description, reset_json,
)

template = KaleidoscopeResetTemplate.from_mapping(
    json.loads(Path("reset-template.json").read_text(encoding="utf-8"))
)
description = sample_reset_description(
    template, seed=42, source_env_ids=(9, 2), reset_index=3,
)
Path("reset.json").write_text(reset_json(description), encoding="utf-8")
```

`source_env_ids` identify logical RNG streams, independent of batch size and
destination rows. `reset_index=0` is the first reset after reseeding. The sampler
uses the same integer RNG/range arithmetic and draw count as ordinary GPU reset;
float32 geometry is compared with a numerical tolerance, not a bitwise promise.

In another compatible native environment, restore each description row into the
corresponding selected destination row:

```python
import json
from pathlib import Path
import torch
from linkerbot_sim.kaleidoscope import KaleidoscopeResetDescription

description = KaleidoscopeResetDescription.from_mapping(
    json.loads(Path("reset.json").read_text(encoding="utf-8"))
)
observations, info = env.restore_reset_description(
    description, target_env_ids=torch.tensor([1, 0], device=env.device),
)
```

Omitting `target_env_ids` selects all environment rows, whose count must match the
description. The normal reset transaction resets velocities, targets, controller
and action histories, and task bookkeeping; it refreshes state without advancing
physics. Selected rows inherit the saved RNG key and post-reset counter, so their
next ordinary reset continues the original logical stream. Unselected rows remain
unchanged. Invalid payloads/configurations are rejected before writes; a failure
after reset writing begins puts the runtime in fail-stop, as ordinary reset does.
An outstanding SAME_STEP transaction forbids this cold operation.

Compatibility requires the same task, backend, robot/scene assets, resolved
configuration and active control mode. A process, CUDA device index, or environment
count may differ. The mode is checked, never switched implicitly. Root asset bytes
are hashed; referenced meshes/textures are not recursively bundled or hashed, so
installing the same complete pinned asset revision remains a caller requirement.
Descriptions are versioned new-episode inputs, not arbitrary-time contact/solver
checkpoints, collision-free seed guarantees, or cross-backend state conversion.
Existing GPU snapshots and portable cold checkpoints retain their separate APIs.

Run `just smoke-kaleidoscope-reset` for independent producer/consumer processes on
PhysX and Newton, using different batch sizes and destination rows.

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
