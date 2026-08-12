# State, Snapshots, And Cloning

Language: [English](snapshots.md) | [中文](../../zh-CN/reference/snapshots.md)

Mirror and Kaleidoscope both support state capture and restore, but their data types
serve different workloads. They are intentionally not interchangeable.

## Comparison

| Property | Mirror scene snapshot | Kaleidoscope episode snapshot |
| --- | --- | --- |
| Scope | One reality-mapped world | Selected replicated environment rows |
| Representation | Owned JSON-compatible mapping | Owned CUDA tensors |
| Transport | `snapshot.get` / `snapshot.set` and Python | Python only |
| Typical use | Replay, inspection, cross-process storage | Branching rollouts, curriculum, evaluation rollback |
| Clone operation | None; every capture is already owned | GPU row-to-row `clone_state` |
| Persistence | JSON-compatible application boundary | Explicit NPZ cold checkpoint |

## Mirror State

`MirrorRuntime.get_state()` returns an owned mapping. `set_state(state, strict=True)`
deep-copies the input before delegating to the engine-aware transactional adapter and
marks collision data dirty.

The JSON protocol exposes `state.get` and `state.set` for the current runtime state,
and separately exposes versioned `snapshot.get` and `snapshot.set` for durable replay
and identity compatibility checks. State mutation is transactional but is not a
substitute for the versioned snapshot schema.

## Mirror Snapshot

The schema identifier is `linkerbot.scene-snapshot.v1`. Capture returns an owned
mapping so the caller cannot mutate runtime storage through an alias:

```python
snapshot = runtime.capture_snapshot()
runtime.restore_snapshot(snapshot, strict=True)
```

The JSON equivalents are `snapshot.get` and `snapshot.set`. A restore may provide a
string-to-string `label_map` when identities are deliberately remapped. With
`strict=True`, missing, extra, incompatible, and ambiguous entities fail instead of
being silently skipped.

Mirror scene snapshots deliberately retain logical cross-engine restore between
PhysX and Newton. A Newton-origin snapshot additionally carries SolverMuJoCo TIME,
ACT, WARMSTART, and the simulation clock, which are restored exactly when the target
is Newton. A PhysX-origin snapshot has no such payload; restoring it into Newton
resets those fields to the committed baseline and sets the Newton clock to zero
instead of retaining the target runtime's previous integration history. This reset
participates in the same compensation transaction as robot and object writes. The
next step is therefore determined by the snapshot's logical state and an explicit
baseline, but it is not a continuation of the source engine's private solver path.

Mirror does not expose a clone operation because it owns only one scene. Repeated
captures return independent mappings; they do not create or fork a physics world.

Capture writes `linkerbot.snapshot.control_mode` into `metadata.info`, containing the
active mode and source generation. Restore requires the runtime to already be in that
mode and never changes its mode or generation. Legacy snapshots use per-joint target
mode metadata; if neither form exists they are accepted only in position mode.

## Kaleidoscope State

The environment binds canonical runtime buffers by field name. Every buffer has
leading dimension `num_envs`, lives on one CUDA device, and may have an engine writer.

```python
state = env.get_state(
    env_ids,
    fields=("robot.q", "object.pose_local_wxyz"),
)
env.set_state(state, env_ids)
```

Field names are defined by the assembled task and views; unknown names are rejected.
Payloads must match the bound dtype and trailing shape. All fields are preflighted
before writes begin. A writer exception fail-stops the runtime; close and recreate it
rather than attempting partial compensation.

The shared PhysX CUDA/Newton core schema is:

| Field | Meaning |
| --- | --- |
| `robot.q` | Full articulation DOFs for all robots concatenated in scene order. |
| `robot.qd` | Velocities for those same full articulation DOFs. |
| `robot.target` | Active-mode command target for controlled joints. |
| `robot.position_reference` | Position reference in radians, independent of active mode. |
| `object.pose_local_wxyz` | Environment-local XYZ and `wxyz` orientation. |
| `object.com_velocity` | Object center-of-mass linear and angular velocity. |

Task/history/counter/RNG bindings extend this core schema. A snapshot captures the
registered fields, so callers must not substitute world-space object pose or a
controlled-joint-only vector for `robot.q`.

Newton additionally registers `solver.persistent`. It is a per-environment
CUDA matrix containing the SolverMuJoCo integration state that affects the next step:
TIME, ACT, and WARMSTART (`qacc_warmstart`). It is engine-owned, writable, resettable,
and cloneable, so default state reads, snapshots, restores, and `clone_state` preserve
it without a host round trip. PhysX does not expose this field.

## GPU Episode Snapshot

```python
snapshot = env.snapshot(env_ids)
assert snapshot.device == env.device
assert snapshot.count == env_ids.numel()
```

`KaleidoscopeEpisodeSnapshot` schema version is integer `2`. Its `env_ids` and every
field are CUDA tensors with the same leading count and device. The mapping is
immutable and the capture path clones storage. `snapshot.clone()` produces another
independent GPU copy.

Schema 2 stores `control_mode` and source `control_generation`. Restore is permitted
only when the current runtime already has the same mode; it never switches mode and
never restores generation. Schema 1 remains a position-only compatibility input and
derives a missing `robot.position_reference` from `robot.target`.

The shared core field names do not make complete snapshots backend-neutral. The
compatibility fingerprint includes the resolved configuration and complete field
schema, including backend-private fields such as Newton `solver.persistent`.
Consequently a snapshot captured from PhysX cannot be restored into Newton, or vice
versa; select the backend before construction and keep a snapshot within that exact
runtime/configuration contract.

Restore to the captured rows:

```python
env.restore_snapshot(snapshot)
```

Restore to an equal-width selector:

```python
env.restore_snapshot(snapshot, target_env_ids=target_ids)
```

## GPU State Cloning

```python
env.clone_state(
    source_env_ids,
    target_env_ids,
    include_rng=True,
    fields=None,
)
```

Cloning copies selected canonical fields entirely on device and forwards the selected
PhysX CUDA or Newton runtime once after the transaction. Requirements:

- both selectors are CUDA `int64` on the environment device;
- both contain unique, in-range IDs;
- lengths are equal;
- source and target sets do not overlap.

Registered RNG key/counter fields are included by default so cloned rows begin from
the same logical random stream. Use `include_rng=False` only when intentional
divergence is required.

## Cold Checkpoints

```python
from linkerbot_sim.kaleidoscope.checkpoint import (
    load_kaleidoscope_checkpoint,
    save_kaleidoscope_checkpoint,
)

save_kaleidoscope_checkpoint(snapshot, "runs/episode.npz")
snapshot = load_kaleidoscope_checkpoint("runs/episode.npz", device=env.device)
```

This is the only maintained host-serialization boundary for a Kaleidoscope episode
snapshot. It uses compressed NPZ with `allow_pickle=False`. Saving and loading are
explicitly cold and must not run in the training step/reset hot path.

## Consistency Rules

- Capture only at a completed runtime boundary, not while an engine writer is active.
- Do not retain borrowed tensors returned with `clone=False` across a step, reset, or
  state write.
- Do not treat transport timeout as proof that a Mirror restore did not execute.
- Never restore a snapshot from a different schema or device without using the
  documented conversion boundary.
