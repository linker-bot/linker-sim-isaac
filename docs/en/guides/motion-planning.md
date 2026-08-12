# Motion Planning

Language: [English](motion-planning.md) | [中文](../../zh-CN/guides/motion-planning.md)

Full trajectory planning and collision avoidance belong exclusively to Mirror.
Kaleidoscope retains batched kinematics and synchronous linear end-effector actions,
but it does not construct a planner, planning collision world, or asynchronous
trajectory service.

## Mirror Planning Stack

Mirror's motion owner contains:

1. strict request parsing;
2. timeline compilation;
3. per-robot cuRobo planning contexts;
4. collision geometry providers and refresh ownership;
5. owner-thread execution on one integer physics-tick axis.

The two selected leaves have deliberately different owners. `configs/planning/mirror.yaml`
contains only backend-neutral request defaults: duration, sampling period, timeout,
avoidance, refresh behavior, and independent coordination. `configs/curobo/mirror.yaml`
selects the concrete numerical capability and owns IK batch capacity, MotionPlanner
seeds, CUDA graph use, collision capability, and cache preallocation.

Mirror may enable the IK CUDA graph, but its MotionPlanner CUDA graph must remain
disabled. The project's pinned cuRobo 0.8 runtime does not enable the experimental
global solver-graph reset by default, and Mirror does not depend on that process-wide
switch while sharing one planner across pose and cspace requests. `MirrorConfig`
rejects an invalid profile during startup instead of deferring the failure to the
first planning request.

## Planning Operations

| Operation | Goal |
| --- | --- |
| `motion.plan_cspace_goal` | Named absolute joint goal |
| `motion.plan_cspace_delta` | Named joint offset from current state |
| `motion.ik_pose` | TCP position and optional orientation |
| `motion.ik_offset` | TCP positional offset and optional orientation |
| `motion.plan_linear_pose_path` | Straight TCP path to an absolute or relative target |
| `motion.plan_timeline` | Compose planning and direct segments across robots/groups |

Planning fields may include `sample_dt_s`, `avoid_collisions`, and
`force_collision_refresh`; planning kinds may also override `duration_s`.
`coordination` is a wrapper/timeline-level override. Omitted fields inherit
`planning.request_defaults`, but `timeout_s` is configuration-only and always comes
from that profile. This policy never resizes or switches the cuRobo backend.

## Collision Refresh

Mirror marks collision state dirty after physics steps, state writes, snapshot
restores, and reset. A planning request can force refresh before planning. Geometry
providers capture a consistent planning snapshot; they do not let a worker read a
partially changing USD stage.

`avoid_collisions: false` disables planner avoidance for that request. It does not
disable physical contacts in the simulation.

## Coordination

The maintained planning configuration defaults to `coordination: independent`.
Timeline alignment still synchronizes segment start ticks, but each robot's plan is
solved independently and other robots are excluded from its planning obstacles.
`static_others` is the explicit alternative: each planned robot treats the other
robots in the captured scene snapshot as static collision obstacles. There is no
coupled multi-robot optimizer; `coupled` is rejected before planning.

## IK Versus Planning

Mirror IK operations are timeline segments: solve a task-space goal, compile the
result, then execute it. The low-level kinematics facade can also be used by embedded
applications, but it does not replace the timeline lifecycle or collision refresh.

Kaleidoscope's `DeviceBatchIKSolver` is a different product path. It consumes fixed
CUDA tensor batches for a configured action and never creates a motion planner.

## Kaleidoscope Linear Motion

A Kaleidoscope linear action:

- interpolates a fixed waypoint count on CUDA;
- performs batched IK without collision checking;
- writes fixed tick targets synchronously;
- holds from the first failed waypoint;
- reports a dense failure mask and applies task penalty/truncation.

It does not return a reusable trajectory, accept a planner request, search around an
obstacle, or allocate per-environment planning worlds. This keeps memory bounded and
the training action shape stable.

## Memory Implications

Large-scale planning would multiply seeds, graph/search buffers, trajectories,
collision geometry, and per-environment query state. Those allocations can dominate
physics and task tensors under either backend. Kaleidoscope avoids that entire
resource closure; only an end-effector action that needs kinematics allocates the
bounded batch IK context.

Mirror plans one request at a time against one interactive world. The cuRobo profile
makes IK batch capacity and MotionPlanner seed/cache allocation explicit. The planner
context is fixed at `max_batch_size=1`, while `kinematics.max_batch_size` applies only
to IK. The planning profile makes the non-overridable request timeout explicit;
neither fact has a second owner.

## Failure Handling

- Structural request errors fail before planning.
- Planning failure produces an error response and no partial timeline execution.
- Execution cancellation stops cooperatively; it is not automatic rollback.
- A planner close timeout prevents `IsaacSession` destruction.
- A Kaleidoscope IK failure affects only the configured rows but follows the frozen
  hold/penalty/truncation policy.

See [Mirror JSON And Motion Examples](../reference/mirror-json.md),
[Control And Trajectories](control-and-trajectories.md),
[Collision Models](collision-models.md).
