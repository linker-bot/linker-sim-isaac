# Python Facade Reference

Language: [English](python-api.md) | [中文](../../zh-CN/reference/python-api.md)

This page defines the in-process Python interfaces on which application and
algorithm code may depend. It is for callers that deliberately own Python
objects inside this checkout. Process clients should use the
[Single Scene JSON reference](single-scene-json.md) or [Tiled Scene JSON reference](tiled-scene-json.md)
instead.

Only the import paths and entries named below are part of this Python boundary.
The existence of another module, a non-underscored name, or a package
`__all__` entry is not by itself an interface commitment. Concrete return types
that are described as opaque are for passing to another documented entry, not
for depending on their implementation attributes.

## 1. Checkout And Runtime Prerequisites

This project is a checkout application, not an installable SDK. Run from the
repository root on Linux x86-64 with Python 3.11:

```bash
uv sync --all-extras
export PYTHONPATH=src
```

After accepting the applicable NVIDIA/Kit EULA, set this only for commands that
start Isaac Sim:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

The labels used throughout this page have precise meanings:

| Label | Requirement |
| --- | --- |
| `pure` | Import and use do not require Kit/Isaac startup. A function may still read or write a path when its contract says so. |
| `Isaac main thread` | Create, read, mutate, step, and close the object on the thread that owns `SimulationApp`, World, stage, and articulation/PhysX views. |
| `cuRobo/CUDA` | Requires the project-pinned cuRobo, Torch, Warp, a usable configured CUDA device, and explicit resource cleanup. It does not imply an Isaac stage is required. |

All facades in this page can themselves be imported in the ordinary project
environment. Importing them does not import `omni`, `isaacsim`, `pxr`, `torch`,
or `curobo`; those dependencies are loaded at the documented call boundary.
This import smoke test is therefore valid before Kit startup:

```python
from linkerbot_sim import REPO_ROOT
from linkerbot_sim.app.interactive.single_scene import run_single_scene_interactive_motion
from linkerbot_sim.app.interactive.tiled_scene import TiledSceneRuntime
from linkerbot_sim.backends.curobo import CuroboConfig
from linkerbot_sim.controllers import JointController
from linkerbot_sim.execution import ExecutionRuntime
from linkerbot_sim.objects import ObjectProfileConfig
from linkerbot_sim.planning import MotionRequest
from linkerbot_sim.robots import JointGroupLayout
from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.snapshots import SimulationSnapshot
```

Importability does not relax the call labels. In particular, resolving the two
runtime symbols above is light, but constructing or running them is still an
`Isaac main thread` operation.

Common numeric conventions are metres, seconds, radians, radians per second,
and public quaternions in `wxyz` order. Joint array order is always established
by an explicit `joint_names`, command-joint list, or articulation `dof_names`;
never infer it from an asset file.

## 2. Facade Summary

| Import path | Label | Ownership |
| --- | --- | --- |
| `linkerbot_sim` | `pure` | Repository root only |
| `linkerbot_sim.planning` | `pure` | Backend-neutral requests, results, frames, collision DTOs, and the linear backend |
| `linkerbot_sim.backends.curobo` | mixed `pure` / `cuRobo/CUDA` | cuRobo config, context, solvers, mappings, collision world, and adapters |
| `linkerbot_sim.controllers` | mixed `pure` / `Isaac main thread` | Controller settings, complete DOF targets, and articulation control |
| `linkerbot_sim.execution` | `Isaac main thread` when run | Command-space execution steps over an existing World |
| `linkerbot_sim.objects` | mixed `pure` / `Isaac main thread` | Object profile parsing and stage insertion |
| `linkerbot_sim.robots` | `pure` | Robot kind, joint groups, and planning capability diagnostics |
| `linkerbot_sim.sensors` | `pure` | Camera configuration selection, not frame acquisition |
| `linkerbot_sim.snapshots` | mixed `pure` / `Isaac main thread` | Snapshot schema, compatibility, capture, restore, and clone |
| `linkerbot_sim.app.interactive.single_scene` | `Isaac main thread` | Single Scene CLI and canonical interactive loop |
| `linkerbot_sim.app.interactive.tiled_scene` | mixed | Tiled Scene runtime, protocol parsing/dispatch, and transport ownership |

## 3. Repository Root

`from linkerbot_sim import REPO_ROOT` is the complete top-level facade.
`REPO_ROOT` is an absolute `pathlib.Path` derived from the source location; it
does not depend on the current working directory and does not assert that a
particular child path exists. Import domain names from their own facades, not
from `linkerbot_sim` transitively.

## 4. Backend-Neutral Planning

Import from `linkerbot_sim.planning`. Every entry in this section is `pure`.
Data classes copy or normalize only where their implementation states so; treat
NumPy inputs as caller-owned unless the class validation explicitly copies them.

### 4.1 Protocols And Names

| Entry | Signature and contract |
| --- | --- |
| `PlannerBackendName` | `Literal["curobo", "linear"]`. |
| `PlanningRequest` | `MotionRequest | LinearPosePathRequest`. |
| `PlannerBackend` | Runtime-checkable protocol: `joint_names() -> Sequence[str]` and `plan(request: PlanningRequest) -> MotionResult`. |
| `normalize_planner_backend` | `(value: object) -> PlannerBackendName`; trims and lowercases, uses `curobo` for an empty value, and raises `ValueError` for any other name. |
| `BatchIKBackend` | Protocol method `solve(*, target_positions, target_orientations_wxyz, seeds, tcp_frame_name) -> BatchIKResult`; rows are environments and all arrays must have the same `N`. |
| `OrientationMode` | `Literal["free", "current", "target"]`; it controls whether a linear TCP segment ignores orientation, holds its start orientation, or reaches a supplied target quaternion. |

### 4.2 Requests And Geometry

| Entry | Constructor / public methods | Shape, units, frame, and rejection |
| --- | --- | --- |
| `PoseTarget` | `(position, orientation=None)` | Position `(3,)` in the caller/backend-agreed task frame, m; optional orientation `(4,)`, `wxyz`. Validation occurs when the containing request is validated. |
| `IKRequest` | `(target_position, target_orientation=None, tcp_frame_name=None, warm_start_ik_cspace_seed=None, position_tolerance=None, orientation_tolerance=None, avoid_collisions=False)`; `validate_structure()` | Target `(3,)` m, quaternion `(4,)` `wxyz`, seed `(C,)` in backend C-space order, tolerances nonnegative (m/rad). Empty frame names, non-finite values, wrong widths, and invalid tolerances raise `ValueError`; reachability is a result, not a structural exception. |
| `MotionRequest` | `(current_q, goal_q=None, goal_pose=None, tcp_frame_name=None, duration_s=None, sample_dt_s=None, avoid_collisions=False)`; `validate_structure()` | `current_q` and `goal_q` are `(C,)` in `backend.joint_names()` order, revolute values rad. Exactly one of `goal_q` and `goal_pose` is required. Duration is finite and nonnegative; sample interval is finite and positive. Structural errors raise `ValueError`. |
| `TcpLineSegment` | `(start_position=None, target_position=None, target_offset=None, orientation_mode="free", target_orientation=None)` | Vectors are `(3,)` m in the request task frame. Exactly one endpoint form is required. `target` mode requires one `(4,)` `wxyz` quaternion; other modes reject it. An explicit start must match the previous endpoint when the cuRobo path is sampled. |
| `TcpPoseSequenceSegment` | `(poses: tuple[PoseTarget, ...], blend_radius=0.0)` | Requires at least one fully oriented pose. The current linear path implementation requires `blend_radius == 0`; finite negative or positive values are rejected by request validation. |
| `TaskSpacePath` | `(segments: tuple[...])` | Nonempty ordered sequence of `TcpLineSegment` and `TcpPoseSequenceSegment`. |
| `LinearPosePathRequest` | `(current_q, path, tcp_frame_name=None, duration_s=None, sample_dt_s=None, avoid_collisions=False)`; `validate_structure()` | `current_q` is `(C,)`; task positions use the cuRobo robot-base planning frame at this facade. `sample_dt_s` must be supplied before cuRobo sampling, normally from physics dt. Invalid structure raises `ValueError`. |
| `CollisionObject` | `(name, shape, pose, size, enabled=True, padding=0.0)`; `pose_matrix()`, `padded_size()` | `pose` is homogeneous `(4,4)`. `cuboid` size is `(x,y,z)`, `sphere` is `(radius,)`, `capsule` is `(radius,length)`, all m. Padding grows both cuboid sides or the radius. Backend conversion rejects unsupported or nonpositive geometry. |
| `FrameTransformer` | `(world_from_robot_base, world_from_env, robot_base_from_tcp=None)`; `from_root_pose(...)`, `pose_to_robot_base(...)`, `offset_to_robot_base(...)` | Transforms are homogeneous `(4,4)` and named `target_from_source`. Reference frames are `world`, `env`, `robot_base`, and `tcp`; offsets rotate without translation. Missing TCP pose, unknown frame, or invalid homogeneous last row raises `ValueError`. |
| `PoseInRobotBase` | `(position, orientation_wxyz)` | Result of frame conversion: position `(3,)` m and optional `(4,)` `wxyz` orientation in robot-base coordinates. |

### 4.3 Results And Executable Linear Planning

| Entry | Signature and contract |
| --- | --- |
| `BatchIKResult` | `(joint_positions, success, position_error, orientation_error=None, status=())`; enforces `(N,C)`, `(N,)`, `(N,)`, optional `(N,)`, and `N` status strings. Wrong dimensions raise `ValueError`. Position error is m; orientation error uses the selected backend metric. |
| `PlanningDiagnostics` | `(status="", message="", metrics={})`; metrics are small printable numeric diagnostics, not solver or tensor objects. |
| `IKResult` | `(joint_positions, success, position_error, orientation_error=None, message="", status="", num_solutions=1)`; joint vector is in backend order. Always test `success` before using it. |
| `MotionResult` | `(path, trajectory, success, status, diagnostics=...)`; `path`, when present, is `(T,C)` in backend order. `trajectory` is normally the project `JointTrajectory`. Always test `success`. |
| `LinearPlannerBackend` | `(joint_names, *, default_duration_s=1.0, default_sample_dt_s=None)`; `joint_names()`, `plan(request)` | Generates deterministic joint interpolation only. Names must be nonempty and unique. It accepts joint-goal `MotionRequest`, requires a positive sample interval from request or constructor, and returns unsuccessful results for task-space or collision-aware requests. Width/default errors raise `ValueError`. It performs no IK, collision, joint-limit, velocity-limit, or acceleration-limit solving. |

Planning failures such as unreachable goals or unavailable collision capability
normally produce `success=False`. Malformed shapes, unknown names, and impossible
configuration contracts raise `ValueError`. See
[cuRobo Usage And Batch Scheduling](../guides/motion-planning.md) for backend
selection and collision behavior.

## 5. cuRobo Backend

Import from `linkerbot_sim.backends.curobo`. Importing this facade is `pure`;
constructing `CuroboContext` or using a real solver is `cuRobo/CUDA`. All joint
arrays are in the context's active C-space order unless a
`CuroboJointMapping` explicitly converts them.

### 5.1 Configuration And Profiles (`pure`)

| Entry | Constructor / public methods | Contract |
| --- | --- | --- |
| `SUPPORTED_CUROBO_DTYPES` | `frozenset({"float32"})` | Exact dtype names accepted by current config parsing. |
| `CuroboTaskBundle` | Validated bundle DTO; use `CuroboTaskBundle.named(value)` and `validate_curobo_version(value)` | Only the project-owned named task file set is accepted. An unsupported bundle raises `ValueError`; an incompatible installed cuRobo release raises `RuntimeError`. Do not construct raw task-path combinations. |
| `CuroboTcpFrame` | `(frame_name, parent_frame, xyz, rpy)`; `from_mapping(data, *, default_parent_frame, label)` | Fixed TCP relative to parent; `xyz (3,)` m and `rpy (3,)` rad. Names must be nonempty and vectors finite. |
| `CuroboDeviceConfig` | `(device="cuda:0", tensor_dtype="float32", collision_geometry_dtype="float32", collision_gradient_dtype="float32", collision_distance_dtype="float32")`; `from_mapping(data)` | Parsing rejects unknown fields and dtypes outside `SUPPORTED_CUROBO_DTYPES`. It does not probe CUDA until context construction. |
| `CuroboRobotConfig` | `(robot_config_path=None, urdf_path=None, base_link=None, flange_frame=None, tool_frames=(), default_tcp_frame=None, custom_tcp_frames=(), load_collision_spheres=True)`; `from_mapping`, `validate`, `resolved_tool_frames` | At least a robot YAML or URDF is required. Paths are checkout-resolved; frame uniqueness and static structure are checked before context creation, model membership during materialization/context creation. |
| `CuroboIkConfig` | IK seeds, tolerances, optimizer settings, regularization weights, `max_batch_size`, `multi_env`, `max_goalset`, self-collision, and collision-cache capacities; `from_mapping`, `validate` | Defaults include 32 seeds, `0.002` m position tolerance, `0.01` rad orientation tolerance, batch 256, and CUDA graph enabled. Types, finite ranges, positive capacities, and known keys are strict. |
| `CuroboMotionPlannerConfig` | Planner warmup, IK/trajopt seeds, tolerances, CUDA graph, batch/goal limits, self-collision, and cache capacities; `from_mapping`, `validate` | Defaults include 32 IK seeds, 4 trajopt seeds, batch 256, and the same tolerances. Strict numeric and capacity validation. |
| `CuroboConfig` | `(robot, task_bundle=..., device=..., ik=..., motion_planner=...)`; `from_mapping(data)`, `validate()` | Parses the complete current backend mapping only when cuRobo is enabled for the supported arm planning group. Unknown fields and inconsistent robot/TCP/capacity settings raise `ValueError`. |
| `validate_curobo_profile` | `(data, *, source="<curobo profile>") -> dict` | Strictly validates project algorithm profile data and returns an independent top-level dictionary; errors include `source`. |
| `load_curobo_profile` | `(path) -> dict` | Reads YAML, then applies the same strict validation; file/YAML errors propagate. |
| `merged_robot_config_with_curobo_profile` | `(robot_config, curobo_profile, *, profile_source=...) -> dict` | Returns a new deep merge in which robot configuration has precedence. Inputs are not mutated. |
| `robot_curobo_config` | `(robot_config, *, curobo_profile=None, robot_source=..., curobo_profile_source=...) -> CuroboConfig` | Performs the validated merge and parses the real backend config; `ValueError` includes both sources. |
| `resolve_curobo_cache_dir` | `(cache_root=None, *, environ=None) -> Path` | Priority is explicit root, `LINKERBOT_SIM_CACHE_ROOT`, `XDG_CACHE_HOME/linkerbot_sim`, then `~/.cache/linkerbot_sim`; appends `curobo` and returns an absolute path without creating it. |
| `materialize_curobo_config` | `(config, *, cache_root=None) -> CuroboConfig` | If custom TCPs exist, atomically writes a content-addressed URDF below the cache and returns a replaced immutable config. Requires a source URDF; XML, permission, validation, and I/O failures propagate. No custom TCP means the same config is returned. |
| `default_tcp_frame_name` | `(config) -> str | None` | Explicit default TCP, otherwise the first resolved tool frame. |
| `resolve_tcp_frame_name` | `(context, *, tcp_frame_name=None, default_tcp_frame_name=None, label="tcp_frame_name") -> str` | Resolution priority is explicit argument, caller default, then context config. Missing/empty/unknown frames raise `ValueError`. This helper is `pure` with a duck-typed context. |

The complete YAML ownership remains in the
[configuration reference](configuration.md); these classes define the Python
representation, not a second YAML field owner.

### 5.2 Context, Solvers, And Collision (`cuRobo/CUDA`)

| Entry | Constructor / public methods | Resource, shape, and result contract |
| --- | --- | --- |
| `import_curobo_module` | `() -> module` | Loads the pinned `curobo` package after Warp compatibility checks. Missing cuRobo or a transitive dependency becomes actionable `RuntimeError`. It does not own a context. |
| `CuroboContext` | `(config, *, cache_root=None)` | Probes configured Torch/CUDA, materializes TCPs, loads kinematics, and lazily creates `ik_solver`, `motion_planner`, or `batch_motion_planner`. `joint_names()` and `frame_names()` define array/frame order. `compute_tcp_poses((N,C), tcp_frame_name=...) -> ((N,3),(N,4))` returns base-frame m and `wxyz`. `sync_collision_world(collision_objects=())` replaces the context's current world snapshot and updates already-created solvers; `clear_collision_world()` performs the corresponding empty-world sync. Capability and factory methods use that same context. Always call `close()`; a close error leaves the failed solver owned so close can be retried. |
| `CollisionCapability` | DTO fields describe robot spheres, scene checker, supported cache types, required/configured capacities, and scene synchronization. `available` and `missing_requirements` are properties. | `available` is true only when the robot model, checker, capacity, synchronized scene, and materialized fingerprint are all present. |
| `CuroboCollisionWorld` | `(context, collision_objects=())`; `sync`, `update_solvers`; count properties | Rebuilds a complete cuRobo scene snapshot and updates already-created solvers. Context owns its current world; callers should not mutate `scene_cfg`. Invalid shape/dimension or cache overflow raises `ValueError`. |
| `make_curobo_scene_cfg` | `(context, collision_objects) -> cuRobo Scene` | Converts enabled cuboids directly and conservatively bounds spheres/capsules as cuboids. Pose is `[x,y,z,qw,qx,qy,qz]`; dimensions are m. Validates configured cache capacity. Returned third-party object is opaque. |
| `CuroboJointMapping` | Use `from_joint_names(*, cspace_joint_names, command_joint_names)`; properties `cspace_width`, `command_width`; `command_to_cspace`, `cspace_to_command` | Converts `(N,D)` arrays by exact names. Missing names, bad widths, or row mismatch raise `ValueError`; non-C-space command columns are copied from `base_command_positions`. The class itself is `pure`. |
| `CuroboForwardKinematics` | `(context)`; `joint_names`, `frame_names`, `compute_pose(q, frame)`, `compute_position`, `compute_orientation` | `q` is `(C,)`; output position `(3,)` m, orientation `(4,)` `wxyz`, and rotation `(3,3)`, all in cuRobo base frame. The full pose result is an opaque value object; use its three named attributes. Unknown frame/backend errors propagate. |
| `CuroboInverseKinematics` | `(context, *, tcp_frame_name=None)`; `joint_names`, `frame_names`, `solve(IKRequest) -> IKResult` | Creates the lazy IK solver. Per-request tolerance overrides are rejected; configure them in the profile. Collision capability failure returns `status="COLLISION_UNSUPPORTED"`; normal solve failure returns `success=False`. Bad frame/seed width raises `ValueError`. |
| `CuroboBatchIKSolver` | `(context, *, tcp_frame_name=None, command_joint_names=None)`; `solve(...)`, `compute_tcp_poses(...)` | Targets are `(N,3)` m and optional `(1,4)` or `(N,4)` `wxyz`; seeds are `(N,C)` or `(N,D)` with mapping. Returns `BatchIKResult`; unsuccessful rows retain their seed positions and carry `success=False`. Width/frame/context defects raise `ValueError` or `RuntimeError`. |
| `CuroboMotionPlanner` | `(context, *, tcp_frame_name=None)`; `planner`, `joint_names`, `plan(request)`, `close()` | Lazily creates the scalar planner. Joint/pose goals and linear pose paths return `MotionResult`; unavailable requested collision returns an unsuccessful result. Returned paths are `(T,C)` rad and trajectories use seconds. `close()` closes the shared context, so do not share it with another owner that must remain live. |
| `plan_linear_pose_path` | `(context, request, *, tcp_frame_name) -> MotionResult` | Samples each segment at `sample_dt_s`, linearly interpolates m positions and Slerps `wxyz`, then solves sequential IK with the preceding solution as seed. Returns unsuccessful results for invalid/unsupported paths, collision deficiency, or failed sample. The frame is cuRobo robot-base local. |

### 5.3 Trajectory Adapter (`pure` For Result-Like Inputs)

`joint_trajectory_from_curobo` has the signature
`joint_trajectory_from_curobo(result_or_trajectory, *, joint_names,
sample_dt=None, phase="trajectory") -> JointTrajectory` reads a cuRobo-like
interpolated trajectory, trajectory, or JointState object. Positions must reduce
to one `(T,C)` trajectory and `len(joint_names) == C`; explicit times are used
when available, otherwise positive `sample_dt` is required. Velocities and
accelerations retain the same shape. Missing attributes, batch dimensions,
width mismatch, invalid times, or non-finite values raise `ValueError` or
`AttributeError`. It allocates no CUDA resource by itself.

## 6. Controllers

Import from `linkerbot_sim.controllers`. The facade import and settings/target
types are `pure`; `JointController` is an `Isaac main thread` integration.

| Entry | Signature and contract |
| --- | --- |
| `ControlMode` | `Literal["position", "velocity", "effort"]`. |
| `ControlMethod` | `Literal["implicit", "explicit", "direct"]`; supported pairs are position/velocity with implicit or explicit, and effort with direct. |
| `ComponentControlSettings` | `(mode="position", method="implicit", stiffness=(1000,), damping=(50,), max_force=100, effort_limit=None, joint_friction=0.5, follower_stiffness=(50000,), follower_damping=(50,), follower_max_force=None, follower_joint_friction=None)`. Each joint parameter may be a scalar, exact-length sequence, or exact name map. `active_effort_limit(s)` resolves limits; mismatches/non-finite values raise `ValueError`. |
| `JointControlSettings` | `(default=..., arm=None, hand=None)`; `component(name, *, component=None)` returns the arm/hand setting or `default`. |
| `ControlTargets` | `(positions, velocities, efforts)`; constructs independent finite one-dimensional copies with identical shape `(D,)`. Units are articulation-native: revolute rad/rad/s and PhysX effort, prismatic m/m/s and force. Shape or finite-value errors raise `ValueError`. |
| `JointController` | `(robot, *, joint_names, settings, mimic_path=None, component_mapping=None, native_mimic=False)` after articulation finalization. Exposes `command_indices`, `follower_indices`, `driven_indices`, `command_joint_names`, and `runtime_follower_indices`. `configure_runtime()` writes modes/gains/limits; `build_control_targets(command_*, base_positions=None)` maps command `(C,)` to full `(D,)`; `targets_from_full_state` validates full `(D,)`; `apply_targets(ArticulationAction, targets)` issues grouped actions. Missing joints/files, invalid widths/settings, absent controller capabilities, and non-finite targets raise `ValueError`, `RuntimeError`, or file/XML errors. |

Call `configure_runtime()` before the first action. Command space excludes mimic
followers; Python-driven followers are recomputed from actual master
state each frame, while native URDF mimic followers receive no duplicate action.
The controller borrows the articulation and does not close it.

## 7. Command Execution

Import from `linkerbot_sim.execution`. Construction of DTOs is `pure`; every
`run` and `execute_*` call is `Isaac main thread` and borrows all resources.

| Entry | Signature and contract |
| --- | --- |
| `ExecutionRuntime` | `(articulation, simulation_world, articulation_action_type, joint_controller, simulation_app, render_enabled, drive_logger=None, state_observer=None, camera_observer=None)`. It is a non-owning bundle; caller closes World/app/loggers/observers. |
| `ExecutionStep` | Protocol with `phase: str` and `run(runtime, step: int) -> int`. The returned value is the cumulative completed physics-step count. |
| `SmoothCommandPositionTargetStep` | `(start_command, target_command, duration, phase, base_positions=None, should_stop=None)`; command arrays are `(C,)` in controller command order. Smoothstep duration is quantized by physics dt to at least one step. |
| `CommandPositionTrajectoryStep` | `(trajectory, should_stop=None)`; trajectory rows are already sampled one per physics step, omit the initial sample, and use controller command-joint columns. No resampling occurs here. |
| `HoldCommandPositionTargetStep` | `(target_command, duration, phase, base_positions=None, should_stop=None)`; positive duration is quantized to steps, nonpositive duration runs until app exit or cancellation. Supply a terminating callback when no app is present. |
| `SwitchControlModeStep` | `(settings, phase="switch_control_mode")`; updates controller settings/configuration without advancing physics and returns the input step. |

The function forms are:

- `execute_smooth_command_position_target`:
  `execute_smooth_command_position_target(*, articulation, simulation_world,
articulation_action_type, joint_controller, start_command, target_command,
duration, phase, simulation_app, render_enabled, step, base_positions=None,
should_stop=None, drive_logger=None, state_observer=None,
camera_observer=None) -> int`.
- `execute_command_position_trajectory`:
  `execute_command_position_trajectory(*, articulation, simulation_world,
articulation_action_type, joint_controller, trajectory, simulation_app,
render_enabled, step=0, should_stop=None, drive_logger=None,
state_observer=None, camera_observer=None, hold=False) -> int`.
- `execute_command_position_hold`:
  `execute_command_position_hold(*, articulation, simulation_world,
articulation_action_type, joint_controller, target_command, duration, phase,
simulation_app, render_enabled, step, base_positions=None, should_stop=None,
drive_logger=None, state_observer=None, camera_observer=None) -> int`.

Cancellation is checked between physics steps and raises a `RuntimeError`
subclass carrying `.step`. If World stepping succeeded but an observer/logger
failed, a `RuntimeError` subclass also carries the already-completed `.step`;
resume from that value and never replay the sample. Other action, shape, World,
and observer errors propagate.

## 8. Objects

Import from `linkerbot_sim.objects`. Parsing and DTOs are `pure`; functions that
receive a USD `stage` are `Isaac main thread`.

| Entry | Signature and contract |
| --- | --- |
| `ObjectSceneInstanceConfig` | `(name, object_profile, root_pose, runtime_handle=None, prim_path=None)`; use `from_mapping(data, *, index)` and the read-only `default_prim_path` / `effective_prim_path` properties. Names/handles must be nonempty and prim paths absolute; placement is m/rad through `RootPoseConfig`. |
| `ObjectProfileConfig` | `(profile_name, name, kind, source, asset_path, ...)`; prefer `from_profile(name)` or `from_mapping(data, *, profile_name, source=None)`. Represents one strictly validated rigid or dynamic-chain profile. |
| `ObjectMaterialConfig` | `(static_friction=None, dynamic_friction=None, restitution=None, friction_combine_mode=None)`; `from_mapping(..., label=...)`, `has_overrides()`. Coefficients are finite/nonnegative, restitution at most 1, combine mode one of the supported PhysX names. |
| `CapsuleRopeConfig` | `(asset_path=..., prim_path="/World/CapsuleRope", root_path="/CapsuleRope", physics=...)`; `from_mapping`, `asset_file`, `validate`. Prefer profile parsing because the nested physics type is not a separate facade entry. |
| `RigidObjectPlanningCollisionConfig` | `(shape, size, xyz=(0,0,0), rpy=(0,0,0), enabled=True, padding=0)`; simplified planner geometry only. Sizes/offsets/padding are m, rotations rad; shape is cuboid/sphere/capsule. |
| `RigidObjectPhysicsConfig` | `(static=False, material=None)`; describes runtime PhysX overrides. |
| `RigidObjectConfig` | `(name, asset_type, asset_path, prim_path, root_pose=..., physics=..., planning_collision=None, urdf_drive_type="none", import_config=...)`; prefer `rigid_objects_from_env_config` so dependent importer types are validated consistently. |
| `AddedRigidObject` | `(name, asset_type, asset_path, prim_path, imported_path, static)`; immutable summary returned after stage insertion. |
| `validate_object_profile` | `(data, *, source="<object profile>", profile_name="object") -> dict`; strict current profile validation with source-qualified `ValueError`. |
| `load_object_profile` | `(path) -> dict`; reads and validates YAML. |
| `object_scene_instances_from_env_config` | `(env_config) -> tuple[ObjectSceneInstanceConfig, ...]`; validates scene instance identity and uniqueness. |
| `expanded_object_mapping` | `(instance, profile=None) -> dict`; combines placement/identity with the referenced profile into a new mapping. Loads the profile when omitted. |
| `rigid_objects_from_env_config` | `(config) -> tuple[RigidObjectConfig, ...]`; expands only `kind=rigid` instances. |
| `add_capsule_rope_reference` | `(stage, config) -> dict[str, object]`; references an existing USD and returns collected named prim handles. Missing asset/prim and USD errors propagate. |
| `apply_capsule_rope_runtime_physics` | `(stage, config) -> {"collision_prims": int, "rigid_bodies": int}`; applies configured material/solver overrides to an already referenced rope. |
| `add_rigid_objects` | `(stage, objects) -> tuple[AddedRigidObject, ...]`; references USD or invokes the Isaac URDF importer, applies pose/physics, and rejects missing assets, unsupported types, target conflicts, or invalid stage state. |

Object runtime functions do not generate assets. Use the offline builders in
[Object Assets](../development/object-assets.md); distinguish PhysX and planning
geometry with [Collision Models](../guides/collision-models.md).

## 9. Robot Metadata And Capability

Import from `linkerbot_sim.robots`. Every entry is `pure`; articulation names may
be passed in as ordinary sequences.

| Entry | Signature and contract |
| --- | --- |
| `RobotKind` | String enum: `ARM="arm"`, `HAND="hand"`, `ARM_HAND="arm_hand"`; `RobotKind.parse(value)`, `has_arm`, and `has_hand`. Invalid values raise `ValueError`. |
| `robot_kind_from_profile` | `(profile) -> RobotKind`; requires the canonical top-level robot mapping and `robot.kind`. |
| `PlanningBindingConfig` | `(enabled, planning_joint_group, has_robot_model)`; use `from_profile(profile, *, kind)`. Enabled planning requires an arm group and nonempty model; a hand-only robot cannot enable it. |
| `PlanningCapability` | `(kind, backend_enabled, planning_joint_group, kinematics_binding_valid, arm_joint_mapping_valid, reasons=())`; property `supports_planning`, method `require(operation="planning")`. `require` raises diagnostic `RuntimeError`; this capability does not prove scene-collision availability. |
| `JointGroup` | `(name, joint_names)`; `from_mapping(name, data)`, `indices_in(dof_names, *, allow_all=True) -> int ndarray`. Exact name matching and order are preserved; missing names or invalid input raise `ValueError`. |
| `JointGroupLayout` | `(command_joint_names, arm=(), hand=(), passive=())`; `resolve(*, kind, command_joint_names, joint_groups, planning_joint_names=())`, `validate_kind`, `validate_planning_joints`, `names(group)`, `indices(group)`. Groups must be unique, disjoint, exhaustive over command joints, and consistent with robot kind; planning joints must equal the arm set. |

## 10. Sensor Configuration

Import `SceneSensorSettings` from `linkerbot_sim.sensors` (`pure`):

`SceneSensorSettings(cameras=())` groups parsed camera settings.
`from_env_config(config)` strictly parses `sensors.cameras`;
`enabled_cameras` filters enabled entries; `has_output_consumers` reports any
enabled file/Foxglove sink; and `validate_single_scene_camera_scope()` rejects a
Tiled Scene `env_ids` selector in Single Scene mode. Configuration errors raise `ValueError`.

This facade does not create a camera, acquire a frame, or publish it. Camera
construction/sampling requires `Isaac main thread` and render-enabled World
steps. File output uses the runtime camera-output policy. Foxglove live or MCAP
also requires `uv sync --all-extras` (the `foxglove-sdk` visualization extra),
a configured sink, and a loopback bind for built-in live servers. RGB and depth
publish RawImage; segmentation modalities are stored locally as NumPy arrays
and publish metadata only. See [Camera Types And Sensors](../guides/cameras.md),
[Foxglove](../guides/foxglove.md), and
[Outputs And Persistence](outputs.md).

## 11. Snapshots

Import from `linkerbot_sim.snapshots`. Schema and compatibility objects are
`pure`. Runtime capture, descriptor construction, restore, dispatch, and clone
are `Isaac main thread` because they read or mutate runtime/PhysX state.

### 11.1 Data And Compatibility (`pure`)

| Entry | Signature and contract |
| --- | --- |
| `SNAPSHOT_SCHEMA` | Exact discriminator string `linkerbot.snapshot`. |
| `SnapshotMetadata` | `(source_runtime="", source_env_id=None, step=None, time_s=None, coordinate_frame="local", info={})`; `from_mapping`, `as_dict`. Optional numeric values must be finite; Tiled Scene capture uses `env-local`, Single Scene uses `scene-local`. |
| `RobotSnapshot` | `(label, robot_id, joint_names, joint_positions, joint_velocities, robot_profile=None, asset_fingerprint=None, command_joint_names=(), command_targets=None)`; `from_mapping`, `as_dict`. State arrays are finite `(J,)`; targets `(C,)`; revolute rad/rad/s, prismatic m/m/s. Names establish order and must be unique. |
| `ObjectSnapshot` | `(name, positions_local, orientations_wxyz, object_profile=None, linear_velocities=None, angular_velocities=None, body_names=(), body_*=None)`; `from_mapping`, `as_dict`. Root is `(3,)` m plus normalized `(4,)` `wxyz`; velocity `(3,)` m/s and rad/s. Body matrices are `(B,3)` / `(B,4)` and required for named bodies. |
| `SimulationSnapshot` | `(robots, objects={}, metadata=..., schema=SNAPSHOT_SCHEMA)`; `from_mapping`, `as_dict`. Mapping keys must equal stable labels/names and robot IDs are unique. Invalid discriminator, unknown top-level/robot fields, shapes, names, or non-finite values raise `ValueError`. |
| `SnapshotRestoreResult` | `(accepted, event="snapshot_restored", robots=(), objects=(), env_ids=(), partial=False, message="")`; `as_dict()`. `partial` is an entry-level indicator for omitted snapshot robot/object entries; finer-grained compatibility semantics are not encoded by this flag. |
| `RobotTargetDescriptor` | `(label, joint_names, robot_profile=None, asset_fingerprint=None, command_joint_names=())`; immutable matching input. |
| `ObjectTargetDescriptor` | `(name, object_profile=None, body_names=())`; immutable matching input. |
| `SnapshotTargetDescriptor` | `(runtime_kind, robots, objects={})`; describes target identities without dynamic state. Runtime adapters use `single_scene` or `tiled_scene`. |
| `JointMapping` | `(source_indices, target_indices, names)`; paired one-dimensional integer mappings with equal length. |
| `RobotCompatibilityMapping` | `(source_label, target_label, joints, command_joints=None)`. |
| `ObjectCompatibilityMapping` | `(source_name, target_name, bodies=None)`. |
| `SnapshotCompatibilityResult` | `(compatible, issues, robot_mappings={}, object_mappings={}, partial=False)`; `partial` has the same entry-level meaning described above. |
| `SnapshotCompatibilityError` | `ValueError` subclass raised by required compatibility checks/restores. |
| `check_snapshot_compatibility` | `(snapshot, target, *, label_map=None, strict=True) -> SnapshotCompatibilityResult`; computes mappings without writing a runtime. |
| `require_snapshot_compatibility` | Same arguments/result; raises `SnapshotCompatibilityError` when incompatible. |

### 11.2 Runtime Adapters (`Isaac main thread`)

| Entry | Signature and contract |
| --- | --- |
| `single_scene_target_descriptor` / `tiled_scene_target_descriptor` | `(runtime) -> SnapshotTargetDescriptor`; read stable profiles, fingerprints, joint/body names, not dynamic state. |
| `get_single_scene_snapshot` | `(runtime) -> SimulationSnapshot`; captures one complete Single Scene with any number of robots. |
| `get_tiled_scene_snapshot` | `(runtime, *, env_id) -> SimulationSnapshot`; captures exactly one env and removes the batch row. Invalid env raises `ValueError`/`IndexError`. |
| `get_snapshot` | `(runtime, *, env_id=None) -> SimulationSnapshot`; dispatches by canonical runtime shape and requires `env_id` for Tiled Scene. Unknown runtime raises `ValueError`. |
| `set_single_scene_snapshot` | `(runtime, snapshot_or_mapping, *, label_map=None, strict=True) -> SnapshotRestoreResult`. |
| `set_tiled_scene_snapshot` | `(runtime, snapshot_or_mapping, *, env_ids, label_map=None, strict=True) -> SnapshotRestoreResult`; broadcasts one logical snapshot to a nonempty, unique, in-range env selection. |
| `set_snapshot` | `(runtime, snapshot_or_mapping, *, env_ids=None, label_map=None, strict=True) -> SnapshotRestoreResult`; dispatches and requires `env_ids` for Tiled Scene. |
| `clone_tiled_env_state` | `(runtime, *, source_env_id, target_env_ids, strict=True) -> SnapshotRestoreResult`; captures source through the same adapter, then restores each target. |

Complete payload fields, strict/non-strict matching, `partial` interpretation,
restore transactions, and exceptions are owned by the
[Snapshot Data And Restore Reference](snapshots.md).

## 12. Single Scene Interactive Runtime

Import from `linkerbot_sim.app.interactive.single_scene`. Both callable entries are
`Isaac main thread` when invoked.

| Entry | Signature and lifecycle |
| --- | --- |
| `main` | `(argv: Sequence[str] | None = None) -> None`; resolves the selected Single Scene runtime profile, supports effective-config output before Kit startup, constructs the runtime, runs the loop, and closes it in `finally`. It prints status to stdout. Prefer the script entrypoint for a subprocess. |
| `run_single_scene_interactive_motion` | `(runtime, *, stdin_enabled=True, tcp_jsonl_host=None, tcp_jsonl_port=None, websocket_host=None, websocket_port=None, state_stream_config=None, start_step=0, planner_backend="curobo", policy=None, interactive_settings=None, execution_settings=None, planner_settings=None, shutdown_settings=None) -> int`; runs the canonical Single Scene queue/timeline loop and returns cumulative global steps. |

The caller of `run_single_scene_interactive_motion` owns the already-created
`SingleSceneRuntime` and must call `runtime.close()` after the loop, even on error.
The loop owns and stops the transports/state stream it starts, but a timed-out
resource is retained by the runtime for a later close retry. Runtime mutation,
snapshot handling, timeline compilation that accesses runtime state, camera
sampling, and World stepping remain on the owner thread. Foxglove output has
the optional dependency and loopback prerequisites described in Section 10.
Exact messages and terminal events are in the
[Single Scene JSON reference](single-scene-json.md).

## 13. Tiled Scene Interactive Runtime And Transports

Import from `linkerbot_sim.app.interactive.tiled_scene`. Parsing and queue/transport
ownership do not require Isaac; runtime construction, dispatch, and loop
execution do.

### 13.1 Runtime (`Isaac main thread`, and `cuRobo/CUDA` when selected)

| Entry | Signature and lifecycle |
| --- | --- |
| `TiledSceneRuntime` | Use `create(*, env_name, env_config, simulation_app, camera_output_settings, shutdown_settings, default_decimation, controller_bundle="default", planner_workers=2, max_pending_requests=64, max_completed_results=256, max_batch_problems=64, oversize_request_policy="split", failure_policy="hold_failed_env", cache_root=None, planner_request_defaults=None, command_defaults=None, playback_settings=None, planner_shutdown_timeout_s=30.0, planner_backend="curobo", curobo_profile="default", joint_batch_mode="auto", additional_output_path_plans=())`. It creates World/scene, selected cameras, IK/planner services, buffers, and initial state. |
| `main` | `(argv=None) -> None`; resolves a Tiled Scene runtime profile, constructs runtime/transports/telemetry, runs the loop, and performs bounded shutdown. Prefer the script for subprocess use. |
| `handle_tiled_interactive_message` | `(message: Mapping[str, object], runtime) -> dict`; validates the current Tiled Scene message and selectors before side effects, invokes exactly one synchronous runtime operation, translates internal labels to public robot IDs, and converts expected errors to a JSON-compatible `rejected` response. It does not schedule onto the owner thread. |
| `run_interactive_loop` | `(runtime, *, telemetry, request_queue, telemetry_rate_hz, idle_physics_policy="pause", idle_step_duration_s=None, queue_poll_timeout_s=0.1, event_publisher=None, transport_status_provider=None) -> None`; sole queue consumer, serially dispatches on the owner thread, performs configured idle stepping and telemetry, publishes the response, then releases request admission. |

The runtime's public methods are `status()`, `robot_name_for_id(id)`,
`idle_step()`, `reset(env_ids)`, `step_action(action, *, env_ids,
robot_names=None)`, `get_state(*, env_ids, fields=None,
include_efforts=False)`, `set_state(state, *, env_ids)`,
`get_snapshot(*, env_id)`, `set_snapshot(snapshot, *, env_ids,
label_map=None, strict=True)`, `clone_state(*, source_env_id,
target_env_ids, strict=True)`, `load_trajectory(trajectory, *, env_ids,
robot_name=None)`, `step_trajectory(*, env_ids, robot_names=None,
decimation=None)`, `submit_plan(message, *, env_ids, robot_name=None)`,
`submit_hand_motion(message, *, env_ids, robot_name=None)`,
`planner_status(*, wait_timeout_s=0)`, `cancel_plan(...)`,
`clear_completed(...)`, `trajectory_status(...)`, `clear_trajectory(...)`, and
`close() -> bool`. State/action arrays retain selected-env rows `(E,...)` and
use the runtime robot's command-joint order. Detailed payloads and selector
rules belong to the [Tiled Scene JSON reference](tiled-scene-json.md).

Call `close()` until it returns true before releasing Kit. It closes planner,
camera output, cuRobo/IK resources, and `SimulationApp` in dependency order;
false means at least one bounded close timed out and ownership is retained.

### 13.2 Pure Parsing And Queue Admission

| Entry | Signature and contract |
| --- | --- |
| `parse_args` | `(argv=None) -> argparse.Namespace`; parses the Tiled Scene CLI without constructing Isaac. `argparse` reports invalid CLI values through `SystemExit`. |
| `parse_tiled_action` | `(message, *, planner_defaults=None, command_defaults=None) -> TiledCommandAction`; strictly parses only a canonical `type="step"` message, resolves configured defaults, validates finite arrays/enums/shape, and returns an action object for `runtime.step_action`. Invalid input raises `ValueError`. |
| `BoundedInteractiveRequestQueue` | `(*, capacity: int)`; `put`, `get`, `task_done`, `full`, `record_rejection`, `status`. Capacity covers each data request from admission through response delivery, not only queued items. The sole consumer must pair `get()` and `task_done()` in order on the same thread. Queue-full and ordinary `queue.Queue` errors apply. Control items do not consume data admission. |

The queue's item classes are transport implementation values and are not
independently constructible interfaces. Obtain them through the start functions
below and consume them through `run_interactive_loop`; do not manufacture them.

### 13.3 Transport Ownership (`pure`, background I/O only)

| Entry | Signature and lifecycle |
| --- | --- |
| `start_stdin_jsonl_reader` | `(request_queue, *, quit_on_eof, max_message_bytes=1048576, admission=None) -> reader handle`; starts an interruptible reader thread. Preserve the handle, then call `stop(timeout_s=...) -> bool` during shutdown and inspect `is_alive()` before releasing ownership; retry `stop` after a timeout. The transport validates framing, size, and UTF-8, then enqueues the text without calling runtime. Strict JSON parsing happens later in the owner-thread `run_interactive_loop`. |
| `start_tcp_jsonl_server` | `(request_queue, *, quit_event, host, port, max_message_bytes=1048576, max_connections=16, server_poll_interval_s=0.1, response_poll_interval_s=0.5, admission=None) -> ThreadingTCPServer`; bind must satisfy the project's loopback policy. Each line receives one response. |
| `stop_tcp_jsonl_server` | `(server, *, timeout_s=2.0) -> dict`; closes active connections and boundedly stops serve/shutdown/handler threads. Inspect the returned status; live resources remain owned and the call may be retried. |
| `start_websocket_server` | `(request_queue, *, quit_event, host, port, max_message_bytes=1048576, max_connections=16, event_queue_capacity=256, server_poll_interval_s=0.1, response_poll_interval_s=0.5, startup_timeout_s=5.0, admission=None) -> WebSocketServerHandle`; starts a dedicated asyncio thread. On a startup error or timeout, the helper performs bounded cleanup before re-raising and annotates the original error if cleanup fails or times out. After a successful return, the caller owns the handle and uses `publish_event`, `status`, and bounded `stop`; inspect the stop status and retry while its thread remains live. |

The project-owned `main` composition shares one connection admission object
between TCP and WebSocket. That admission class has no documented public
constructor, so independently calling both start functions creates independent
connection caps; use `main` when one process-wide cap is required. The concrete
admission and server-handle classes are not separate facade entries, so callers
should use only the documented server/handle methods returned here.
All built-in listeners are unauthenticated loopback endpoints. Never call an
Isaac runtime from a handler thread; handlers enqueue pure data and the owner
thread dispatches it.

## 14. Documented Advanced Owner Paths

The modules in this section do not provide a unified package facade. Only the
fully qualified symbols below are dependable owner paths; no other name in the
same module is implied to be public.

### 14.1 Runtime Configuration (`pure`)

| Exact symbol | Signature and contract |
| --- | --- |
| `linkerbot_sim.configs.profiles.profile_path` | `(group, name) -> Path`; groups are `runtime`, `robot`, `env`, `object`, `curobo`, and `logging`. The name must be one safe file stem. It resolves directory-style envs to `base.yaml` but does not require the returned ordinary file to exist. |
| `linkerbot_sim.configs.profiles.load_profile_yaml` | `(group, name) -> dict`; loads the selected checkout profile. Env, robot, object, cuRobo, and logging groups invoke their domain validation; `runtime` returns strict-YAML data but does not parse the runtime schema, so use `load_runtime_profile` for that group. Invalid group/name/content raises `ValueError`; missing files raise `FileNotFoundError`. |
| `linkerbot_sim.configs.profiles.load_env_profile_yaml` | `(name) -> dict`; loads and validates either one env YAML or a directory `base.yaml` plus sorted per-env fragments. It returns the complete merged env mapping. |
| `linkerbot_sim.configs.runtime.RuntimeProfileConfig` | Use `RuntimeProfileConfig.from_mapping(data, *, profile_name=None, source_path=None)` and `as_dict()`. Parsing accepts only the current top-level `runtime` mapping, validates types/ranges/cross-fields, copies input data, and creates no runtime resource. |
| `linkerbot_sim.configs.runtime.ResolvedRuntimeConfig` | Result container with `config`, leaf `sources`, and deterministic `fingerprint`; `as_dict()` returns the effective mapping. Read-only properties expose `mode`, `profiles`, `simulation_app`, `execution`, `interactive`, `planner`, `playback`, `camera_output`, `telemetry`, `output`, `paths`, and `shutdown`. |
| `linkerbot_sim.configs.runtime.load_runtime_profile` | `(name) -> RuntimeProfileConfig`; safe-stem checkout lookup plus strict runtime parsing. |
| `linkerbot_sim.configs.runtime.resolve_runtime_config` | `(profile, *, cli_overrides, env_config, expected_mode=None) -> ResolvedRuntimeConfig`; applies only known non-`None` dotted overrides, validates selected profiles and mode/env cross-fields, and records provenance. It does not start Isaac or validate the complete dependency graph. |
| `linkerbot_sim.configs.validator.ValidatedProfileGraph` | Frozen result containing `runtime_profile`, parsed `profile`, `resolved`, and a read-only mapping of sorted dependency names. |
| `linkerbot_sim.configs.validator.validate_profile_graph` | `(*, runtime_profile, profile, resolved, env_config) -> ValidatedProfileGraph`; reads and validates env, robot, controller, object, cuRobo, and logging dependencies without Isaac/GPU/file-output creation. Missing dependencies raise `FileNotFoundError`; structural/capability/path conflicts raise `TypeError` or `ValueError`. |

Use the returned nested settings when calling runtime factories rather than
constructing partially validated setting DTOs by hand. YAML fields and overlay
priority remain owned by the [Configuration Reference](configuration.md).

### 14.2 Single Scene Factory And Parser

| Exact symbol | Label and contract |
| --- | --- |
| `linkerbot_sim.app.runtime.single_scene_runtime.create_single_scene_runtime` | `Isaac main thread`; `(*, env="scene1", env_config=None, simulation_app, camera_output_settings, shutdown_settings, output_settings=None, curobo_profile="default", logging_profile="default_logger", controller_bundle="default", control_mode="position", cache_root=None, hold_app=False, status_prefix=None, additional_output_path_plans=(), session_factory=..., profile_loader=..., controller_bundle_loader=...) -> SingleSceneRuntime`. The final three injectable factories are test/composition hooks. It validates all output plans before applying them, performs one World reset, rolls back acquired resources on failure, and transfers ownership to the result. |
| `linkerbot_sim.app.runtime.single_scene_runtime.SingleSceneRuntime` | Returned owning runtime. Dependable operations are the read-only `robots_by_id`, `robot_id_by_label`, `world`, and `config_fingerprint` properties plus `robot(robot_id)`, `status()`, and `close()`. `close()` returns a report with `stopped` and `live_resources`; retry when not stopped. Do not instantiate the dataclass directly. |
| `linkerbot_sim.app.interactive.single_scene.protocol.InteractiveMotionCommand` | `pure` frozen DTO returned by the parser. Branch on `kind`; optional ID, reset, snapshot, and timeline fields are meaningful only for their corresponding kind. |
| `linkerbot_sim.app.interactive.single_scene.protocol.parse_interactive_motion_message` | `pure`; `(message, *, planner_defaults=None, command_defaults=None) -> InteractiveMotionCommand`. It parses/normalizes one Single Scene command without modifying the input mapping, applies strict field/type/finite-vector checks and defaults, and has no queue/runtime side effect. Protocol defects raise `ValueError`; reachability and runtime robot IDs are checked later on the main thread. |

### 14.3 Trajectory Values And Builders (`pure`)

| Exact symbol | Signature and contract |
| --- | --- |
| `linkerbot_sim.trajectories.types.TrajectoryEval` | Frozen value with `(position, velocity, acceleration, jerk, effort)`, each shape `(C,)` in trajectory joint order. |
| `linkerbot_sim.trajectories.types.JointTrajectory` | Keyword constructor `times (N,)`, `positions (N,C)`, `joint_names (C,)`, optional same-shape derivative/effort matrices and `phases (N,)`. It copies finite matrices, requires nonempty strictly increasing times, fills omitted matrices with zero, and raises `ValueError` on mismatch. `domain()`, `eval(t)`, and `eval_all(t)` clamp finite queries to endpoints; `len()` is `N`. Time is s; revolute position/derivatives are rad-based. |
| `linkerbot_sim.trajectories.joint_trajectory_builder.joint_trajectory_from_positions` | `(*, times, positions, joint_names, phase="trajectory") -> JointTrajectory`; computes velocity, acceleration, and jerk by finite differences. It does not enforce robot limits or dynamic feasibility. |
| `linkerbot_sim.trajectories.retiming.trajectory_sample_times` | `(*, duration_s, sample_dt_s, include_start=False) -> ndarray`; returns positive tick times with at least one interval and optionally prepends zero. When `0 <= duration_s < sample_dt_s`, the horizon expands to one complete `sample_dt_s` tick; otherwise the final value equals `duration_s`, with a partial final tick when needed. Invalid finite/range values raise `ValueError`. |
| `linkerbot_sim.trajectories.retiming.retime_joint_trajectory` | `(trajectory, *, duration_s: float | None, sample_dt_s: float | None, start_position=None, phase=None, include_start=False) -> JointTrajectory`; when both timing values are present, resamples by cumulative joint-path progress and recomputes derivatives. If either timing value is `None`, `include_start=False` returns the original trajectory object unchanged, while `include_start=True` raises `ValueError`. It is geometric timing, not limit-aware optimization; invalid start width also raises `ValueError`. |

### 14.4 Logging Configuration (`pure`)

| Exact symbol | Signature and contract |
| --- | --- |
| `linkerbot_sim.logging.config.JointLoggingConfig` | Frozen Single Scene CSV settings. `should_write_step(step)` applies enable/decimation; `flush_interval_steps(physics_dt)` converts seconds to at least one step. Tiled Scene entrypoints do not consume this profile. |
| `linkerbot_sim.logging.config.normalize_logging_profile_name` | `(value) -> str`; accepts only a trimmed `[A-Za-z0-9][A-Za-z0-9_-]*` stem, otherwise `ValueError`. |
| `linkerbot_sim.logging.config.joint_logging_config_from_mapping` | `(data, *, source_path=None) -> JointLoggingConfig`; strict current field/type/range parser, with no file creation. |
| `linkerbot_sim.logging.config.load_joint_logging_profile` | `(name, *, logging_root=...) -> JointLoggingConfig`; safe lookup plus strict parsing; missing file raises `FileNotFoundError`. |
| `linkerbot_sim.logging.config.override_logging_config` | `(config, **updates) -> JointLoggingConfig`; ignores `None`, returns `dataclasses.replace`, and raises `TypeError` for unknown fields. It does not re-run mapping validation, so pass already validated values. |

### 14.5 Telemetry DTO And Config Owners (`pure`)

| Exact symbol | Signature and contract |
| --- | --- |
| `linkerbot_sim.telemetry.state_snapshot.RobotJointStateSnapshot` | One robot row: ID/label/names plus `(J,)` rad, rad/s, rad/s2 arrays and optional effort arrays. `effort_values(field)` accepts `none`, `commanded`, `measured`, `applied`; `as_dict()` maps non-finite effort samples to JSON null. Caller is responsible for equal widths. |
| `linkerbot_sim.telemetry.state_snapshot.ObjectPoseSnapshot` | `(name, prim_path, position_m (3,), orientation_wxyz (4,))`; `as_dict()` emits one world-pose row. |
| `linkerbot_sim.telemetry.state_snapshot.StateSnapshot` | `(step, time_s, robots, objects=(), phase=None)`; field-frozen cross-thread payload with JSON-compatible `as_dict()`. The dataclasses neither copy nor write-protect contained NumPy arrays: producers must supply detached arrays, and every caller must treat them as immutable after publication. It is realtime observation data, distinct from restorable `SimulationSnapshot`. |
| `linkerbot_sim.telemetry.state_snapshot.StateStream` | `(*, capacity=1, drop_policy="latest")`; thread-safe single-consumer bounded handoff. `publish` never blocks; `latest`, `wait_next`, `status`, `is_closed`, and `close(*, discard_pending=False) -> None` define its lifecycle. Invalid capacity/policy raises `ValueError`. It contains no Isaac objects. |
| `linkerbot_sim.telemetry.foxglove.FoxgloveTopicConfig` | `(joint_states="/joint_states", scene="/scene", state="/linkerbot/state")`; topic-name DTO only and does not load the optional SDK. |
| `linkerbot_sim.telemetry.foxglove.prepare_mcap_output` | `(path, *, existing_file_policy) -> OutputPathPlan | None`; plans but does not apply filesystem changes. `resume` is rejected with `ValueError`; treat the returned plan as opaque input to a runtime factory. |
| `linkerbot_sim.telemetry.tiled.config.TiledTelemetryConfig` | Validated selected-env/topic/buffer DTO. Selected IDs are nonempty, unique, nonnegative; primary ID must be selected; decimation/capacity/policies/timeouts are strict. Invalid values raise `ValueError`. |
| `linkerbot_sim.telemetry.tiled.config.parse_env_ids` | `(value: str) -> tuple[int, ...]`; parses comma-separated integers and rejects a blank string. Construct `TiledTelemetryConfig` afterward to reject an empty parsed tuple and enforce uniqueness, nonnegative IDs, and primary selection. Runtime composition separately validates every ID against `num_envs`. |

Foxglove logger/sink classes, Single Scene samplers, CSV writers, and Tiled Scene telemetry
sinks remain runtime-owned implementation. Configure them through the documented
runtime settings unless another exact symbol is added to this section.

## 15. Error And Ownership Rules

- Validate shapes, names, frames, finite numbers, and configuration before
  creating GPU or Isaac resources. Configuration/programming defects raise
  exceptions; expected planner inability is normally represented by an
  unsuccessful result or rejected response.
- NumPy row/column meaning is part of the contract. Preserve explicit env,
  joint, body, and sample dimensions instead of relying on broadcasting unless
  a method explicitly permits it.
- A class that accepts a context, World, stage, articulation, or runtime usually
  borrows it. Only methods explicitly named `create`, `start`, or returning an
  owning handle allocate a lifecycle that must be closed.
- Do not read Isaac objects from planner, transport, telemetry, camera-output,
  or file-writer threads. Capture immutable Python/NumPy data on the owner
  thread first.
- On snapshot/reset/state restore rollback failure, or on incomplete shutdown,
  stop mutations and recreate or finish closing the runtime. See
  [Known Risks And Design Constraints](../operations/constraints.md).
