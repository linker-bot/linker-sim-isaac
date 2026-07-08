# cuMotion Motion Modes And Parameters

Language: [English](cumotion-motion-modes-and-parameters.md) | [中文](../../zh-CN/运动规划/cuMotion%20运动模式与参数示例.md)

This document lists the project-level motion modes exposed on top of cuMotion and the main Python/JSON fields used to configure them.

## Hand Motion

Hand motion is a first-class motion and can also be used as an overlay on arm motion.

Common spec shape:

```python
HandMoveSpec(
    side="left",
    joint_positions={"joint_name": 0.05},
    duration_s=0.3,
    phase="left_hand_close",
)
```

Overlay timing:

| Timing | Meaning |
| --- | --- |
| `before` | Play hand trajectory before arm motion. |
| `sync` | Play hand trajectory in sync with arm motion. |
| `after` | Play hand trajectory after arm motion. |

The same overlay model applies to IK pose, IK offset, C-space goal, C-space delta, TCP line, TCP arc, C-space waypoint, composite path, and advanced cuMotion move specs.

## Absolute IK Pose

Absolute IK pose means: move the TCP to an absolute target pose.

JSON example:

```json
{
  "type": "ik_pose",
  "side": "left",
  "position": [0.1, -0.2, 0.3],
  "orientation": [0.0, 0.0, 0.0],
  "duration_s": 1.0
}
```

Use `orientation_quat_wxyz` when the caller already has a quaternion. Do not send both RPY and quaternion fields for the same target.

## IK Offset

IK offset means: offset the current TCP pose by a translation and optional orientation mode.

```json
{
  "type": "ik_offset",
  "side": "left",
  "offset": [0.0, 0.02, 0.0],
  "orientation_mode": "current",
  "duration_s": 0.5
}
```

Orientation modes:

- `current`: keep the current TCP orientation.
- `target`: use `target_orientation` or `target_orientation_quat_wxyz`.
- `none`: constrain position only.

## Absolute C-Space Goal

```json
{
  "type": "cspace_goal",
  "side": "left",
  "joint_positions": [1.5, 1.2, -1.5707, 1.57, -0.37, 0.0, 0.0],
  "duration_s": 2.2,
  "phase": "left_cspace_goal"
}
```

Joint order follows the selected side's arm joint group.

## C-Space Delta

```json
{
  "type": "cspace_delta",
  "side": "left",
  "joint_deltas": [0.01, 0, 0, 0, 0, 0, 0],
  "duration_s": 0.5
}
```

The runtime reads the current arm state, adds deltas, and plans/plays the resulting target.

## Task-Space Line

Task-space line requests move the TCP along a straight segment.

```json
{
  "type": "task_space_line",
  "side": "left",
  "target_position": [0.12, -0.18, 0.28],
  "target_orientation": [0.0, 0.0, 0.0],
  "duration_s": 1.0
}
```

This maps to cuMotion specified-path task-space conversion when the backend supports it.

## Task-Space Arc

Task-space arc requests describe a TCP arc segment.

Typical fields:

- start or current TCP pose
- target position
- arc center or arc control point
- target orientation
- duration and sampling parameters

Use arc paths when the tool must avoid a straight-line approach or follow a curved task-space path.

## Specified Path

Specified path is the advanced form for explicit path families:

- C-space waypoint path.
- Task-space TCP path.
- Composite path.

The backend converts the path into a C-space path and then generates a timed trajectory.

## Planner Pipeline

Common planner choices:

| Mode | Use |
| --- | --- |
| `trajectory_optimization` | Default smooth point-to-point motion. |
| `graph_search` | Conservative path search before trajectory generation. |
| `specified_path` | Follow caller-provided C-space/task-space/composite path. |

## Units

- Positions: m.
- Angles: rad.
- RPY order: `[roll, pitch, yaw]`.
- Quaternions: `wxyz`.
- Durations: seconds.

## Boundary

cuMotion owns arm C-space planning. Dexterous hand DOFs, mimic followers, full command-space clipping, and high-level action sequencing stay in runtime/controller layers.
