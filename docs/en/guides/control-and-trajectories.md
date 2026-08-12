# Control And Trajectories

Language: [English](control-and-trajectories.md) | [中文](../../zh-CN/guides/control-and-trajectories.md)

Mirror and Kaleidoscope use different control models. Mirror accepts discrete
interactive requests that compile into synchronized timelines. Kaleidoscope consumes
one fixed-shape action tensor per decision and advances a fixed number of physics
ticks. Only Mirror owns a control profile; Kaleidoscope action semantics come from its
task, and its physics engine derives the controller bundle used to prepare all supported
runtime joint-control modes.

## Mirror Timeline Model

Mirror represents motion as:

```text
timeline
  robot track
    motion unit (same start tick)
      group track (arm, hand, ...)
        serial segments
```

Use a one-segment operation for a direct move or hold. Use
`motion.plan_timeline` when robots or joint groups must share one tick axis.
Complete, parser-validated requests for every motion operation are maintained in
[Mirror JSON](../reference/mirror-json.md).

Supported segment kinds are:

- `hold`;
- `joint_goal` and `joint_delta`;
- `joint_trajectory` with explicit sample times;
- `joint_effort` in the v2 protocol;
- `plan_cspace_goal` and `plan_cspace_delta`;
- `ik_pose` and `ik_offset`;
- `plan_linear_pose_path`.

The compiler validates the entire request and completes required planning before
execution. A compilation error produces no partial movement.

## Runtime Joint-Control Modes

Joint-control mode is runtime state, not an action variant. Mirror v2 and the native
`TorchKaleidoscopeEnv` can switch all robots and all environments between `position`,
`velocity`, and `effort` after one complete motion/decision and before the next. The
switch keeps the session, physics runtime, robot views, action term, task, IK, and
planner owners intact. It is rejected during motion execution or an active SAME_STEP
transaction.

Mirror v1 remains frozen and has no mode operations. Mirror v2 adds
`control.get_mode`, `control.set_mode`, and `motion.joint_effort`. Position and velocity
modes accept position-derived timelines; effort mode accepts hold and explicit bounded
effort segments only. A real switch increments `generation`; an idempotent switch does
not write the engine or increment it.

Every real switch first neutralizes the old channel, applies all prepared controller
profiles transactionally, and then writes the new neutral target: current `q` for
position and zero for velocity or effort. A fully compensated failure leaves the old
mode usable. Failed compensation makes the runtime fail-stop; close and recreate it.

## Hybrid Force/Position Control

The dedicated `physx_cpu_hybrid` Mirror profile runs at 240 Hz. During one
`motion.hybrid_force_position` request, the selected robot arm temporarily uses
direct effort drives for every arm joint. Cartesian axes with `force_axes=false` use
explicit pose impedance (`Kp/Kd`); axes with `force_axes=true` use explicit force PI
(`Kf/Ki`). The hand remains position/implicit, and every other robot remains in its
existing mode. Per-axis implicit arm position drives are not mixed with explicit arm
effort drives because PhysX exposes the drive mode at the joint, not at a Cartesian
axis.

`force_axes` belongs to each motion request and may change from one request to the
next. The six gain groups are runtime tuning state. Mirror v3 serializes
`control.get_hybrid_parameters` and `control.set_hybrid_parameters` through the same
owner queue as motion. A motion requires `hybrid_parameter_generation` and freezes
exactly one immutable gain snapshot during preflight. Later updates cannot change an
active loop; the next motion must use the returned new generation. YAML keeps the
non-adjustable filter, contact, sensor, effort, rate, displacement, and safety limits.

A successful `control.tare_wrench` for the same robot, physical TCP, and world frame
is also required. Raw PhysX feedback is environment-on-tool; tare subtraction and
filtering precede the sign conversion exposed to targets, results, CSV, and telemetry
as tool-on-environment. Reset invalidates tare. Completion, cancellation, or failure
ramps effort to zero, restores the original position controller, and hands over at the
final measured joint position. Restore failure is fatal and requires closing and
recreating the runtime.

## Timing

Mirror durations are converted to integer physics ticks. Direct joint and hold kinds
require a positive `duration_s`. Planning kinds may inherit the positive default from
`planning.request_defaults`. `sample_dt_s` controls the planning/interpolation grid,
not the physics clock. Wire-level planning overrides are `duration_s`, `sample_dt_s`,
`avoid_collisions`, and `force_collision_refresh`; `coordination` is overridden at
the one-segment wrapper or timeline top level. `timeout_s` is not accepted on the
wire: every planning request uses `planning.request_defaults.timeout_s`. These request
defaults do not select or configure the cuRobo numerical backend. Its
`kinematics.max_batch_size` belongs to IK only; the Mirror MotionPlanner remains a
single-request context while retaining configured seeds, graph, collision capability,
and cache capacity.

`joint_trajectory.times_s` must describe finite, ordered samples compatible with the
named joint arrays. The executor never changes the configured physics frequency to
fit a trajectory.

`control.sync_simulation_to_wall_clock` controls execution pacing, not trajectory
retiming. When enabled, idle hold steps and motion timelines use the same wall-clock
deadline sequence. The first tick after startup or reset is immediate; later ticks wait
only for the remaining physics interval. A late tick rebases that sequence, so Mirror
does not issue a burst of catch-up steps. When disabled, both paths advance without
wall-clock sleeps while retaining the same physics dt. `idle_physics_policy: pause`
still stops simulation time because there are no physics ticks to pace.

## Joint Identity

Name mappings may target only part of a joint group. A flat JSON array must cover the
whole group and follows the robot profile's explicit `joint_groups` order. Neither
form depends on incidental USD articulation DOF order; never infer one robot's order
from a URDF, asset file, or another robot.

`robot_id` is a session-local nonnegative integer. `robot_label`, when supplied, is an
identity assertion and must match discovery; it is not a fallback selector.

## Task-Space Frames

Task-space operations accept `world`, `env`, `robot_base`, or `tcp` as documented by
the field. Positions use metres and quaternions use `wxyz`. Offset fields and absolute
targets are distinct and cannot be inferred from vector magnitude.

Mirror linear motion is a planning operation and can use collision avoidance. It is
not the same implementation as Kaleidoscope's synchronous fixed-waypoint action.

## Cancellation And Stop

Mirror motion checks a cooperative cancellation predicate while executing. A client
can cancel a request by ID, cancel the active request, or emergency-stop the runtime.
Emergency stop also freezes idle stepping and blocks new motion until reset.

Cancellation is bounded but not transactional rollback. Capture a snapshot before a
motion when the application requires explicit rollback semantics.

## Kaleidoscope Decision Model

Kaleidoscope freezes the action variant when the environment is created. The action
variant and runtime joint-control mode are separate: switching control mode never
changes action shape or `physics_ticks_per_action`. Every step
accepts a CUDA `float32` tensor with shape `(num_envs, action_dim)`. One decision
produces a fixed sequence of `physics_ticks_per_action` typed targets, writes each
target to the selected PhysX/Newton channel, and steps without rendering.

The canonical `joint_control` variant preserves the former joint-delta position
behavior:

```text
target <- clamp(target + scale * clip(action, -1, 1), lower, upper)
```

The anchor is the previous command target, not noisy measured joint position. Reset
re-anchors the selected rows to their reset joint command.

In velocity mode, `joint_control` maps the bounded action directly to rad/s. In effort
mode, it maps the action to the configured fraction of each controller profile's effort
limit. Existing `joint_delta` and EE/linear variants support position and bounded
position-reference derivatives in velocity mode, but reject effort mode. Gymnasium,
skrl, and `KaleidoscopeTrainingPort` do not expose the mode setter; training rollout
semantics remain fixed at the initial position mode.

Schema 2 snapshots record control mode and source generation. Restore never switches
mode and must target the same active mode; generation is diagnostic and is not restored.

## Batched IK And Linear Actions

End-effector action variants run device-batched IK for every environment. Linear
variants create a fixed number of synchronous waypoints using `linear` or
`smoothstep` progress, solve them in batches, and execute the resulting fixed tick
targets.

These operations intentionally provide no graph search, trajectory optimization,
asynchronous job queue, or avoidance. A failed row holds from the documented failure
point and receives the configured penalty/truncation behavior.

## Done And Reset

The native environment requires explicit `reset_idx(done_ids)` before the next
`step`. The skrl adapter uses a generation token to preserve terminal observations,
reset done rows in the same decision, and expose post-reset observations without a
missed reset. Gymnasium offers `disabled` or `same_step` autoreset at its NumPy
boundary.

See [Mirror JSON And Motion Examples](../reference/mirror-json.md) and
[Kaleidoscope API](../reference/kaleidoscope-api.md).
