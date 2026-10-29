# Control And Trajectories

Language: [English](control-and-trajectories.md) | [中文](../../zh-CN/guides/control-and-trajectories.md)

Mirror and Kaleidoscope use different control models. Mirror accepts discrete
interactive requests that compile into synchronized timelines. Kaleidoscope consumes
one fixed-shape action tensor per decision and advances a fixed number of physics
ticks. Only Mirror owns a control profile; Kaleidoscope action semantics come from its
task, and its physics engine derives the controller bundle used to prepare all supported
runtime joint-control modes.

## Flange And Task TCP Frames

The AR5 arm-hand profiles use `AR5V2_L_flange_tcp` and `AR5V2_R_flange_tcp` as
their default TCP names in both products. Each is a zero-translation,
zero-rotation frame relative to the configured arm `flange_frame`. It does not
identify a fingertip, pinch point, or calibrated contact point, including when a
physical flange or wrist camera is installed.

Requests that previously selected `AR5V2_L_pinch_tcp` or `AR5V2_R_pinch_tcp`
must select the corresponding `flange_tcp` name; the old names are not retained
as aliases. Omitting `tcp_frame_name` selects the configured default as before.
Numerical poses, physical binding, IK targets, and wrench reference points are
unchanged by this rename. The standalone arm profile keeps its existing
`AR5V2_L_tool_tcp` name.

For an actual tool/contact point, define a separately named frame in
`curobo.robot.custom_tcps` with its measured `xyz` (m) and `rpy` (rad) relative to
the specified `parent_frame`. Select it through `default_tcp_frame` or the
operation's supported TCP selector. A fixed offset only describes the hand shape
for which it was defined; changing finger joints may change the contact point.
Hybrid force/position control and tare must use the configured physical TCP.

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

The dedicated `physx_cpu_hybrid` Mirror profile defaults to 60 Hz commands / 240 Hz feedback / 480 Hz physics. During one
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

## Newton AR5/O6 compliant drive

Newton automatically selects the compliant drive for AR5/O6, including Gemini
camera assemblies. The bundle is named `newton_compliant`; no scene override is
required and the old direct-PD path is not selectable for these assemblies.
It supports **position control only** on CPU and CUDA. Other robot combinations
retain their existing drive settings; PhysX is unchanged.

The bundle has ordinary `arm_controller.yaml` / `hand_controller.yaml` native
output settings and a `newton_drive.yaml` leaf for virtual motors and drive
dynamics. Its parameter vectors follow the robot profile's declared arm/hand
joint-group order and are bound to exact joint names before import; USD traversal
and articulation ordering cannot swap thumb and finger gains. This name binding
applies to every controller bundle. O6 hand arrays use: `thumb_cmc_pitch`,
`thumb_cmc_yaw`, `index_mcp_pitch`, `middle_mcp_pitch`, `ring_mcp_pitch`,
`pinky_mcp_pitch`, prefixed with `hand_lh_` or `hand_rh_`. Default complete command
vectors concatenate declared arm then hand names; native solver indices are mapped,
not reordered.

| Joint group | Native Kp / Kd | Native cap (N·m) | Passive damping (N·m·s/rad) |
| --- | --- | --- | --- |
| AR5 J1–J7 | 16000 / 614.224, 613.711, 603.791, 600.333, 604.137, 604.749, 603.668 | 200 | 0 |
| O6 thumb yaw | 600 / 24 | 10 | 0 |
| O6 thumb flexion | 2500 / 40 | 60 | 4 |
| O6 four-finger MCP | 1250 / 40 | 60 | 4 |
| O6 followers | No independent actuator; native equality | No independent motor cap | 0.4 |

The hand virtual motor has Kp=600 and a 1 rad/s maximum shaft speed. Its
**advancing torque** is 3 N·m for thumb yaw and 5 N·m for flexion; these are
separate from the native holding/output caps above. Viscosity is 20 for thumb yaw
and flexion, 4.2 for four-finger flexion; output damping is respectively 20, 40,
and 40. The backward-Euler shaft update permits motion only toward the commanded
angle, caps free speed and displacement, and stalls against output load instead
of accumulating an unbounded reference. It reads active-joint motion; native
joint equality transmits follower contact reactions. Physical q/qd are never
clamped or overwritten by the controller.

At the default **60 Hz command / 480 Hz physics**, the virtual shaft advances once
per command, while the previous/current shaft reference is linearly interpolated
across eight native steps. The current `output_damping / Kp * shaft_velocity`
offset remains applied during interpolation. Mirror's command rate comes from the
scene; Kaleidoscope's comes from physics frequency divided by
`physics_ticks_per_action`. Valid custom integer ratios retain one motor update
per command, rather than accelerating the motor with the physics rate.

Arm references use cubic Hermite interpolation between successive command
positions and supplied velocities, with velocity feedforward. A direct position
target has zero endpoint velocity; a trajectory should supply its sampled
velocity. This is a generic per-interval interpolation, not the experimental
T-block phase-specific PCHIP/retiming policy. Immediately before each native
step, a target guard limits estimated arm PD output to 199 N·m, below the 200 N·m
native cap. This estimate is not a native force measurement or a guarantee on
post-integration constraint reactions.

Robot bodies and mounted accessories have gravity compensation; objects retain
gravity. Arm/hand armature is 0.01/0.001, joint Coulomb friction is zero, and native
follower equality uses `solref=(0.002,1)`,
`solimp=(0.999,0.999,0.001,0.5,2)`. These drive-profile overrides apply only to
selected robot roots. Geometry, mass/inertia, joint limits, contact/friction
parameters remain unchanged in this step. CPU/CUDA pipeline, gap fallback and
solver defaults are documented in the [shared Newton settings](../reference/configuration.md#shared-newton-pipeline-defaults).

Run `just smoke-newton-drive` to verify public Mirror motion commands on CPU and
CUDA: declared hand order, arm velocity forwarding, eight physics substeps per
command, hand-reference interpolation, native simulation time and history restore.

Controller history, including the command substep phase, participates in reset,
snapshot, restore and selected-world clone. Cold position resets initialize the
virtual shafts from measured joint positions. CPU/CUDA share float64 history and
identical equations; CUDA execution uses the physics owner's stream and graph.
Kaleidoscope exposes this float64 state as `controller.virtual_shaft`; it must not
be downcast when saving a snapshot. Requested targets remain separate from the
native targets generated at each physics step.
