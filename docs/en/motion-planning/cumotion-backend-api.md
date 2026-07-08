# cuMotion Backend API

Language: [English](cumotion-backend-api.md) | [中文](../../zh-CN/运动规划/cuMotion%20后端接口说明.md)

This document summarizes the project-facing cuMotion backend API. It is a customer-facing interface guide, not an internal implementation plan.

## Layer Boundary

The cuMotion backend handles the robot C-space described by cuMotion robot descriptions. It does not own full Isaac articulation command space, dexterous hand DOFs, or mimic follower expansion.

Callers are responsible for mapping cuMotion C-space trajectories back into the full controller command space by joint name.

## Main Modules

| Module | Responsibility |
| --- | --- |
| `context.py` | Lazy import, robot resource loading, robot description, kinematics, default/custom TCP materialization. |
| `forward_kinematics.py` | FK requests and results. |
| `inverse_kinematics.py` | IK and collision-free IK requests. |
| `motion_planner.py` | Motion planning facade for trajectory optimization, graph search, and specified paths. |
| `collision_world.py` | Primitive obstacle sync, world enable/disable/update, inspector wrappers. |
| `trajectory_adapter.py` | Convert cuMotion trajectories into project `JointTrajectory`. |
| `pose_adapter.py` | Convert project `wxyz` quaternions to cuMotion pose objects. |

## Context

`CuMotionContext` is the long-lived object for one robot model/profile. It loads robot YAML resources and prepares cuMotion state lazily.

Typical responsibilities:

- Load XRDF/URDF and kinematics.
- Expose `joint_names()` as the only source of C-space order.
- Materialize robot YAML `cumotion.custom_tcps` when custom TCP frames are configured.
- Own the current `CuMotionCollisionWorld` when obstacle sync is used.

## FK

Forward kinematics maps a C-space joint vector to a TCP pose.

Inputs:

- C-space joint positions in `context.joint_names()` order.
- TCP frame name, optional. If omitted, the default TCP from robot YAML is used.

Outputs:

- TCP position.
- TCP orientation in project `wxyz` order.
- Diagnostics if the backend cannot evaluate the request.

## IK

IK maps a target pose into a C-space joint vector.

Common request fields:

- `target_position`
- `target_orientation_wxyz`
- `tcp_frame_name`
- `seed_joint_positions`
- position/orientation tolerances
- optional collision-free mode

The wrapper validates request shape and model compatibility before calling cuMotion. If orientation is intentionally unconstrained, the request should say so explicitly rather than passing an invalid quaternion.

## Motion Planning

`CuMotionMotionPlanner.plan(...)` supports three major families:

| Pipeline | Use |
| --- | --- |
| `trajectory_optimization` | Default point-to-point planning. |
| `graph_search` | Conservative graph planner path followed by trajectory generation. |
| `specified_path` | Convert C-space, task-space, or composite paths into a C-space trajectory. |

Successful graph search and specified-path planning must produce a timed trajectory. Returning only a discrete path is not enough for project execution.

## Specified Paths

Specified path requests can describe:

- C-space waypoint paths.
- Task-space TCP line segments.
- Task-space TCP arc segments.
- Composite path specs.

The backend converts path specs to C-space and then runs trajectory generation to produce position/velocity/acceleration/jerk samples where supported by cuMotion.

## Collision World

Primitive collision obstacles are synchronized explicitly by the caller. The backend does not scan the Isaac stage automatically.

Supported operations include:

- Add/update/remove obstacles.
- Enable/disable world collision checks.
- Rebuild geometry when obstacle shape changes.
- Inspect world/robot distances through wrapper APIs.

## Trajectory Adapter

The adapter samples cuMotion trajectories into project `JointTrajectory` objects.

The adapter accepts real cuMotion trajectories and test doubles that expose compatible `eval_all(t)` behavior. It keeps C-space order by joint name and lets the runtime map values into full command space.

## Diagnostics

Planning results should carry diagnostics that are useful to callers:

- success/failure state
- error text
- number of waypoints
- planner pipeline
- sampled duration
- collision metrics when available

Future work may attach inspector results directly to `MotionResult.diagnostics.metrics`.

## Out Of Scope

The backend intentionally does not own:

- Dexterous hand open/close scripted targets.
- Mimic follower expansion.
- Full articulation command-space clipping.
- High-level action state machines.
- Foxglove or CSV logging.
