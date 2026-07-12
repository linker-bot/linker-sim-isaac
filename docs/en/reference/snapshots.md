# Snapshot Data And Restore Reference

Language: [English](snapshots.md) | [中文](../../zh-CN/reference/snapshots.md)

This page is the sole detailed owner of the `linkerbot.snapshot` data structure,
target matching, restore result, and transaction semantics. The
[Single Scene JSON Reference](single-scene-json.md) and [Tiled Scene JSON Reference](tiled-scene-json.md)
own only their message envelopes, selectors, and response differences.

A snapshot describes one Single Scene runtime state or one Tiled env state. The Tiled env batch
exists only in `set_snapshot.env_ids` and `clone_state.target_env_ids`; it is not
part of the snapshot body.

## 1. Complete JSON Structure

This example includes robot state, object root and child-body state, metadata,
and every optional motion field. Every number must be finite. Prefer passing the
complete body returned by `get_snapshot` without reconstructing it.

```json
{
  "schema": "linkerbot.snapshot",
  "metadata": {
    "source_runtime": "tiled_scene",
    "source_env_id": 0,
    "step": 120,
    "time_s": 0.5,
    "coordinate_frame": "env-local",
    "info": {
      "per_env": {
        "replay_id": "case_001"
      }
    }
  },
  "robots": [
    {
      "label": "robot_0",
      "robot_id": 0,
      "robot_profile": "ar5v2_l6v1_l",
      "asset_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "joint_names": ["arm_joint_1", "arm_joint_2"],
      "joint_positions": [0.1, -0.2],
      "joint_velocities": [0.0, 0.05],
      "command_joint_names": ["arm_joint_1", "arm_joint_2"],
      "command_targets": [0.12, -0.18]
    }
  ],
  "objects": {
    "rope": {
      "name": "rope",
      "object_profile": "capsule_rope",
      "positions_local": [0.3, 0.0, 0.1],
      "orientations_wxyz": [1.0, 0.0, 0.0, 0.0],
      "linear_velocities": [0.0, 0.0, 0.0],
      "angular_velocities": [0.0, 0.0, 0.0],
      "body_names": ["segment_0", "segment_1"],
      "body_positions_local": [[0.3, 0.0, 0.1], [0.4, 0.0, 0.1]],
      "body_orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      "body_linear_velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      "body_angular_velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    }
  }
}
```

The top level supports only these fields:

| Field | Type | Rule |
| --- | --- | --- |
| `schema` | required string | Exactly `linkerbot.snapshot` |
| `metadata` | optional object | Source and coordinate diagnostics; omitted means empty source information |
| `robots` | required array | One entry per robot; may be empty, but cannot be a label-keyed object |
| `objects` | optional object | Keys are stable object names; capture responses always emit this field |

Top-level and robot entries reject unknown fields. Put metadata extensions in
`info`; use only the documented object fields because `as_dict()` does not
preserve other content. JSON transports also reject duplicate keys, nonstandard
numbers, and trailing content.

## 2. Metadata

| Field | Type | Meaning |
| --- | --- | --- |
| `source_runtime` | string | Diagnostic producer name; regular captures use `single_scene` or `tiled_scene`, while the in-memory debug adapter uses `tiled_scene_debug` |
| `source_env_id` | optional integer | Tiled Scene source env; normally absent for Single Scene |
| `step` | optional integer | Runtime step at capture |
| `time_s` | optional finite number | Simulation time at capture, in seconds |
| `coordinate_frame` | string | Local convention for object poses; captures use `scene-local` or `env-local` |
| `info` | JSON object | Diagnostics such as profile fingerprint, robot labels, or per-env metadata |

Metadata does not participate in robot or object identity matching. The Tiled Scene
adapter restores object local poses relative to each target env origin; the
Single Scene adapter uses scene-local poses. Robot joint state is independent of
`coordinate_frame`.

## 3. Robot Entry

| Field | Type and shape | Rule |
| --- | --- | --- |
| `label` | required nonempty string | Stable source identity; unique within the snapshot |
| `robot_id` | required nonnegative integer | Source-session diagnostic ID; never selects a restore target; unique in the snapshot |
| `robot_profile` | optional string | Must match when both sides provide it |
| `asset_fingerprint` | optional string | Must match when both sides provide it |
| `joint_names` | required nonempty string array | Unique names define position/velocity order |
| `joint_positions` | required finite number `[J]` | Same length as `joint_names` |
| `joint_velocities` | required finite number `[J]` | Same length as `joint_names` |
| `command_joint_names` | optional string array | Unique names define command-target order; capture emits it |
| `command_targets` | optional finite number `[C]` | Requires nonempty, equal-length `command_joint_names` |

Joint values use native articulation DOF units: rad/rad/s for revolute joints
and m/m/s for prismatic joints. A snapshot stores only controller-managed
command joints, not uncontrolled DOFs. Restore maps by name and never assumes
that source and target columns have the same order.

## 4. Object And Body Entry

| Field | Type and shape | Unit and rule |
| --- | --- | --- |
| `name` | nonempty string | Matches the outer `objects` key; parser uses that key when omitted |
| `object_profile` | optional string | Must match when both sides provide it |
| `positions_local` | finite number `[3]` | m, in the frame named by `metadata.coordinate_frame` |
| `orientations_wxyz` | finite number `[4]` | Nonzero `wxyz` quaternion; normalized while parsing |
| `linear_velocities` | optional finite number `[3]` | m/s |
| `angular_velocities` | optional finite number `[3]` | rad/s |
| `body_names` | optional string array | Unique child rigid-body names; capture emits it |
| `body_positions_local` | finite number `[B,3]` | Required with nonempty `body_names`, in m |
| `body_orientations_wxyz` | finite number `[B,4]` | Required with nonempty `body_names`; normalized row by row |
| `body_linear_velocities` | optional finite number `[B,3]` | m/s |
| `body_angular_velocities` | optional finite number `[B,3]` | rad/s |

A regular rigid object may use `body_names=[]` and restore only its root. A
dynamic-chain object must retain every body pose; otherwise child-body state
cannot be proven complete. Velocity fields are structurally optional, but
linear and angular components should be supplied together. Single Scene live rigid
views reject missing required velocities; Tiled Scene object views write omitted
velocities as zero. Zero quaternions and all nonfinite arrays are rejected.

## 5. Identity, `label_map`, `strict`, And `partial`

Robot identity defaults to exact `label` matching; source `robot_id` never
selects a target. `label_map` is an explicit `source label -> target label` JSON
object. It cannot be empty, both labels must exist, and target labels cannot be
repeated. When present, only listed source robots are processed. Objects cannot
be renamed and always match by `name`.

Matching proceeds as follows:

1. Resolve robot labels or `label_map`.
2. Require equal `robot_profile` and `asset_fingerprint` when both sides provide them.
3. Build joint and command-joint indices by name rather than array position.
4. Match objects by name, then dynamic-object bodies by body name.
5. Finish every check and capture rollback state before the first runtime write.

`strict=true` requires equal joint-name sets for each mapped robot and equal
body-name sets for each dynamic object; order may differ. `strict=false` writes
only the intersection of those name sets, but an empty intersection still
fails. It does not ignore a missing object, a profile/fingerprint mismatch, or
an invalid `label_map`.

Single Scene object restore applies the generated body-name index mapping. The current
Tiled Scene writer consumes body arrays directly in target-view column order without
applying that mapping. Tiled Scene dynamic-object restore therefore also requires
identical source/target `body_names` order and cannot rely on body reordering or
`strict=false` body-subset writes.

`partial` means that a complete robot or object entry from the snapshot did not
enter the mapping, for example because an explicit `label_map` selected only
some source robots. It does not currently become true merely because a mapped
robot or body writes a name intersection under `strict=false`. A caller cannot
infer from `partial=false` that every source array element was written.

## 6. Capture Differences

| Behavior | Single Scene | Tiled Scene |
| --- | --- | --- |
| Capture scope | Current complete Single Scene | One required `env_id` |
| Metadata frame | `scene-local` | `env-local` |
| Metadata clock | Adapter omits step/time | Includes source env and runtime step/time |
| Robot arrays | One row per runtime robot | One row per robot after removing the env batch dimension |
| Object poses | Scene-local root/body | Env-local root/body after removing the batch dimension |

Single Scene capture stores current physical joint positions as `command_targets`;
Tiled Scene capture stores the actual `target_positions`. Both capture joint
position/velocity, object roots, and available child-body/velocity state.
Python objects copy NumPy data; JSON output contains independent arrays. Asset
fingerprints include both normalized asset path and file content, so equal
content at different paths can still mismatch. Single Scene object capture reads a
world transform while restore writes a local transform; arbitrary prim
hierarchies with non-identity parents have no strict round-trip guarantee.

## 7. Restore Result

The common successful Python adapter result is:

```json
{
  "event": "snapshot_restored",
  "accepted": true,
  "robots": ["robot_0"],
  "objects": ["rope"],
  "env_ids": [1, 2],
  "partial": false
}
```

| Field | Meaning |
| --- | --- |
| `event` | `snapshot_restored` for restore; the Tiled Scene clone envelope changes it to `state_cloned` |
| `accepted` | true after the adapter completes restore |
| `robots` | Written target labels in Python/Single Scene; Tiled Scene JSON converts this to `robot_ids` |
| `objects` | Written target object names |
| `env_ids` | Empty for Single Scene; every target env ID for Tiled Scene |
| `partial` | Entry-level mapping summary defined in Section 5 |
| `message` | Present only for nonempty diagnostic text |

Single Scene execution errors return `snapshot_failed`, `accepted=false`, `id`, and
`error`. The Tiled Scene message boundary converts an exception to
`{"event":"rejected","error":"..."}`. Python facades raise parsing,
matching, or runtime exceptions directly.

## 8. Compensating Transactions, Rollback, And Fail-Stop

Snapshot restore is not a native PhysX atomic transaction. Before the first
write, an adapter parses data, matches targets, and captures independent
rollback state for every target. It registers a compensating action before
each setter and attempts those actions in reverse write order after failure.

- Single Scene rolls back robot position/velocity, controller cache, and object state.
  Observer-cache reset and collision-registry invalidation are irreversible steps.
- Tiled Scene separately saves each target env's position/velocity, command target,
  adapter/TCP cache, and object state. Successful restore clears those envs'
  trajectories and cancels intersecting planner requests.
- If every compensation succeeds and failure preceded every irreversible step,
  the runtime remains usable and the original exception reaches the caller.
- If any compensation fails, or failure follows an irreversible step, the
  runtime records its first fatal reason, requests exit, and rejects later state
  mutations. The caller must destroy and rebuild the runtime.

A target setter may already have changed physics. A client must not blindly
repeat a restore after receiving an indeterminate terminal state.

## 9. Single Scene Messages And Admission

Single Scene has no env selector:

```json
{"type":"get_snapshot","id":"snapshot-1"}
```

```json
{
  "type": "set_snapshot",
  "id": "restore-1",
  "strict": true,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

`label_map` is optional and `strict` defaults to true. This example shows only
the envelope; use a complete captured snapshot for restore. Success adds
`backend="isaac"` and the request `id`.

Single Scene transport puts get/set requests in a dedicated main-thread queue.
`runtime.interactive.snapshot_timeout_s` limits only the wait before the main
thread atomically marks a request executing; bundled profiles use 30 seconds.

```json
{"event":"snapshot_timeout","accepted":false,"id":"snapshot-1"}
```

Once executing, the caller waits past the admission deadline for a definite
terminal result. Shutdown is the only early return: an unstarted request is
`snapshot_cancelled`; an executing request is `snapshot_running`, which is not
a success state and does not authorize automatic replay.

```jsonl
{"event":"snapshot_cancelled","accepted":false,"reason":"shutdown","id":"snapshot-1"}
{"event":"snapshot_running","accepted":true,"state":"running","id":"restore-1"}
```

## 10. Tiled Scene Messages And Clone

Tiled Scene capture accepts one source env. Restore broadcasts one snapshot to an
explicit, nonempty, unique, in-range target env list:

```jsonl
{"type":"get_snapshot","env_id":0}
{"type":"set_snapshot","env_ids":[1,2],"strict":true,"snapshot":{"schema":"linkerbot.snapshot","robots":[],"objects":{}}}
```

A successful get response contains `backend="isaac"`, `env_id`, `step`,
`time_s`, and `snapshot`. A successful set response contains `backend="isaac"`,
`robot_ids`, `objects`, `env_ids`, `partial`, `step`, and `time_s`.

`clone_state` performs one `get_tiled_scene_snapshot(source)` on the main thread and
then writes every target with the same transaction semantics. It does not
accept `label_map`:

```json
{"type":"clone_state","source_env_id":0,"target_env_ids":[1,2],"strict":true}
```

Its response event is `state_cloned` and additionally returns `source_env_id`
and `target_env_ids`. The source is not otherwise modified, but the current
selector accepts it in the target list and would write the same snapshot back.

## 11. Python Facade

Data models and matching functions are `pure`. Adapters that access a live
runtime must run on the main thread that owns the Isaac runtime. Import the
documented surface from `linkerbot_sim.snapshots`:

```python
from linkerbot_sim.snapshots import (
    SimulationSnapshot,
    check_snapshot_compatibility,
    clone_tiled_env_state,
    get_snapshot,
    set_snapshot,
)

snapshot = get_snapshot(runtime, env_id=0)
payload = snapshot.as_dict()
parsed = SimulationSnapshot.from_mapping(payload)
result = set_snapshot(runtime, parsed, env_ids=[1, 2], strict=True)
```

`get_snapshot(runtime, env_id=None)` dispatches by runtime shape; Tiled Scene requires
`env_id`. `set_snapshot(runtime, snapshot, env_ids=None, label_map=None,
strict=True)` also dispatches, and Tiled Scene requires `env_ids`. Explicit adapters
are `get_single_scene_snapshot`, `set_single_scene_snapshot`, `get_tiled_scene_snapshot`,
`set_tiled_scene_snapshot`, and `clone_tiled_env_state`. Target descriptors and pure
matching are exposed through `single_scene_target_descriptor`,
`tiled_scene_target_descriptor`, `check_snapshot_compatibility`, and
`require_snapshot_compatibility`.

## 12. Use Checklist

- Save the complete `snapshot` object rather than only joint positions.
- Stop external workflows that depend on old state before restore.
- Omit `label_map` unless robot labels were intentionally renamed.
- Keep `strict=true` by default. When using name intersections, inspect actual
  target state rather than relying only on `partial`.
- Do not edit source `robot_id` to select a target; use stable labels or `label_map`.
- Treat `snapshot_running`, rollback errors, and fatal mutation as states that
  require runtime rebuild.
