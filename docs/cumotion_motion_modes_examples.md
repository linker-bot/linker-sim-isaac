# cuMotion Motion Modes And Parameter Examples

本文说明客户脚本如何用 Python 参数描述 TCP 和运动。运动参数属于临时实验输入，默认不放进 YAML。

## 通用约定

- 长度单位是 m。
- 角度单位是 rad。
- 四元数使用 `wxyz` 顺序。
- `side` 取值为 `left` 或 `right`。
- `duration_s` 是期望执行时长。
- `phase` 可选；不传时 `src` 会生成稳定默认名称。
- `tcp_frame_name` 可选时，双臂执行器会按 `side` 使用启动时传入的左右默认 TCP。
- task-space 目标支持绝对位置和相对 offset 两种写法。
- C-space/关节角目标支持绝对目标和相对 delta 两种写法。
- TCP 姿态目标支持“不约束姿态”、“保持当前姿态”和“给定目标姿态”三种语义，具体取决于运动模式。
- 所有 arm/cuMotion 运动都可以叠加 hand overlay。
- hand motion 不进入 cuMotion planner；它走 controller command-space，可以单手执行，也可以双手同步执行。

TCP 只描述相对末端/flange 的坐标变换，不包含 parent link：

```python
tcp = DualArmTcpSpec(
    left=CartesianTcpFrameSpec(
        frame_name="left_demo_tcp",
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
    ),
    right=CartesianTcpFrameSpec(
        frame_name="right_demo_tcp",
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
    ),
)
```

具体 parent/flange frame 由 scene 选中的左右 robot profile 的 `cumotion.flange_frame` 绑定。`src` 不计算 pinch、吸盘、相机等具体工具点；这些几何语义留在客户脚本侧。

## Hand Motion And Overlay

手部动作应该是一等 motion，同时也能作为 arm motion 的同步 overlay。

建议新增通用 spec：

```python
@dataclass(frozen=True)
class HandMoveSpec:
    side: Literal["left", "right"]
    joint_positions: Mapping[str, float] | tuple[float, ...]
    duration_s: float
    phase: str | None = None


@dataclass(frozen=True)
class DualHandMoveSpec:
    left: HandMoveSpec | None = None
    right: HandMoveSpec | None = None
    duration_s: float | None = None
    phase: str | None = None


@dataclass(frozen=True)
class CommandOverlaySpec:
    timing: Literal["sync", "before", "after"] = "sync"
    left_hand: HandMoveSpec | None = None
    right_hand: HandMoveSpec | None = None
```

手部 `joint_positions` 推荐优先支持 mapping，键是 controller command-space 关节名；也可以支持 tuple，表示按该侧 hand command joints 的约定顺序给值。

单手动作：

```python
HandMoveSpec(
    side="left",
    joint_positions={
        "L6V1_L_hand_index_mcp_pitch": 0.7,
        "L6V1_L_hand_thumb_cmc_pitch": 0.5,
    },
    duration_s=0.5,
    phase="left_hand_close",
)
```

双手同步动作：

```python
DualHandMoveSpec(
    left=HandMoveSpec(
        side="left",
        joint_positions={"L6V1_L_hand_index_mcp_pitch": 0.7},
        duration_s=0.5,
    ),
    right=HandMoveSpec(
        side="right",
        joint_positions={"L6V1_R_hand_index_mcp_pitch": 0.7},
        duration_s=0.5,
    ),
    duration_s=0.5,
    phase="dual_hand_close",
)
```

arm motion 叠加手部动作：

```python
IkOffsetMoveSpec(
    side="left",
    tcp_frame_name="left_demo_tcp",
    tcp_offset=(0.03, 0.0, 0.02),
    duration_s=1.0,
    phase="left_reach_with_hand",
    overlays=(
        CommandOverlaySpec(
            timing="sync",
            left_hand=HandMoveSpec(
                side="left",
                joint_positions={
                    "L6V1_L_hand_index_mcp_pitch": 0.7,
                    "L6V1_L_hand_thumb_cmc_pitch": 0.5,
                },
                duration_s=1.0,
            ),
        ),
    ),
)
```

overlay timing:

| `timing` | 含义 |
| --- | --- |
| `sync` | hand trajectory 与 arm trajectory 同步播放 |
| `before` | 先执行 hand trajectory，再执行 arm motion |
| `after` | arm motion 完成后再执行 hand trajectory |

这套 overlay 应适用于本文后续所有 arm/cuMotion motion：绝对 IK、IK offset、绝对 C-space goal、C-space delta、TCP line、TCP arc、C-space waypoint、Composite path 和高级 `CumotionMoveSpec`。

## Absolute IK Pose

绝对 IK pose 表示“让 TCP 到达指定的绝对位置，可选约束目标姿态”。

当前可以用高级 `CumotionMoveSpec + IKRequest` 表达；后续交互协议中应把它作为 `ik_pose` 一等消息类型暴露出来。

```python
CumotionMoveSpec(
    execution="selected_side",
    side="left",
    tcp_frame_name="left_demo_tcp",
    duration_s=1.0,
    phase="left_ik_pose",
    request=IKRequest(
        target_position=np.asarray([0.35, -0.20, 0.40], dtype=float),
        target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        tcp_frame_name="left_demo_tcp",
        avoid_collisions=False,
    ),
)
```

带同步手部 overlay：

```python
CumotionMoveSpec(
    execution="selected_side",
    side="left",
    tcp_frame_name="left_demo_tcp",
    duration_s=1.0,
    phase="left_absolute_ik_with_hand",
    request=IKRequest(
        target_position=np.asarray([0.35, -0.2, 0.4], dtype=float),
        target_orientation=None,
        tcp_frame_name="left_demo_tcp",
    ),
    overlays=(
        CommandOverlaySpec(
            timing="sync",
            left_hand=HandMoveSpec(
                side="left",
                joint_positions={"L6V1_L_hand_index_mcp_pitch": 0.7},
                duration_s=1.0,
            ),
        ),
    ),
)
```

参数：

| 参数 | 含义 |
| --- | --- |
| `target_position` | TCP 绝对目标位置 `(x, y, z)` |
| `target_orientation` | 可选 TCP 绝对目标姿态，wxyz 四元数；不传则只约束位置 |
| `position_tolerance` | 可选位置容差 |
| `orientation_tolerance` | 可选姿态容差 |
| `avoid_collisions` | 可选 collision-aware IK |

## IK Offset

`IkOffsetMoveSpec` 表示“从当前 TCP pose 出发，给 TCP 位置加一个相对位移，然后求 IK”。

适合快速做小幅接近、抬升、平移测试。

```python
IkOffsetMoveSpec(
    side="left",
    tcp_frame_name="left_demo_tcp",
    tcp_offset=(0.05, 0.0, 0.02),
    duration_s=1.0,
    phase="left_ik_offset",
)
```

参数：

| 参数 | 含义 |
| --- | --- |
| `side` | 运动哪一侧机械臂 |
| `tcp_frame_name` | 使用哪个 TCP；可省略时按 `side` 使用默认 TCP |
| `tcp_offset` | 当前 TCP 位置的相对位移 `(x, y, z)` |
| `duration_s` | 执行时长 |
| `phase` | 可选日志/轨迹阶段名 |

当前 `IkOffsetMoveSpec` 只暴露位置 offset。更完整的接口应该允许传入目标姿态，例如：

```python
IkOffsetMoveSpec(
    side="left",
    tcp_frame_name="left_demo_tcp",
    tcp_offset=(0.05, 0.0, 0.02),
    target_orientation=(1.0, 0.0, 0.0, 0.0),
    duration_s=1.0,
)
```

实现时建议给 `IkOffsetMoveSpec` 增加：

- `target_orientation: tuple[float, float, float, float] | None = None`
- `orientation_mode: Literal["current", "target", "none"] = "current"`

其中 `current` 表示保持当前 TCP 姿态，`target` 表示使用 `target_orientation`，`none` 表示只约束位置。

## Absolute C-Space Goal Plan

绝对 C-space goal 表示“把选定侧 arm joints 规划到给定绝对角度”。

当前可以用高级 `CumotionMoveSpec + MotionRequest(goal_q=...)` 表达完整双臂 C-space 目标；客户脚本层更直观的接口应新增一等 spec，例如 `CSpaceGoalPlanMoveSpec`：

```python
CSpaceGoalPlanMoveSpec(
    side="right",
    tcp_frame_name="right_demo_tcp",
    joint_positions=(0.2, -0.5, 0.3, -1.0, 0.1, 0.2, 0.0),
    duration_s=1.2,
    phase="right_cspace_goal",
)
```

参数：

| 参数 | 含义 |
| --- | --- |
| `joint_positions` | 选定侧 arm C-space 的绝对目标关节角；可以少于 arm joints，未给出的尾部关节保持当前值 |
| `side` | 运动哪一侧机械臂 |
| `tcp_frame_name` | planner 使用的 TCP frame |
| `duration_s` | 执行时长 |
| `phase` | 可选日志/轨迹阶段名 |

## C-Space Delta Plan

`CSpaceDeltaPlanMoveSpec` 表示“在当前 arm C-space 上叠加关节增量，然后调用目标式 planner”。

适合测试 graph search / trajectory optimization 这类目标式规划 pipeline。

```python
CSpaceDeltaPlanMoveSpec(
    side="right",
    tcp_frame_name="right_demo_tcp",
    joint_deltas=(0.12, -0.08, 0.06, -0.04, 0.03, -0.02, 0.01),
    duration_s=1.2,
    phase="right_cspace_delta",
)
```

参数：

| 参数 | 含义 |
| --- | --- |
| `side` | 运动哪一侧机械臂 |
| `tcp_frame_name` | planner 使用的 TCP frame；目标是 C-space 时也可用于 planner 默认 frame |
| `joint_deltas` | 相对当前选定侧 arm C-space 的关节增量；可以少于 arm joints，未给出的尾部关节保持不变 |
| `duration_s` | 执行时长 |
| `phase` | 可选日志/轨迹阶段名 |

## TCP Line Path

`SpecifiedPathMoveSpec + TaskSpacePath + TcpLineSegment` 表示显式指定 TCP 直线路径。

```python
SpecifiedPathMoveSpec(
    side="left",
    tcp_frame_name="left_demo_tcp",
    path=TaskSpacePath(
        segments=(
            TcpLineSegment(
                target_offset=(0.0, 0.0, 0.08),
                orientation_mode="none",
            ),
        ),
    ),
    duration_s=1.2,
    phase="left_tcp_line",
)
```

`TcpLineSegment` 参数：

| 参数 | 含义 |
| --- | --- |
| `target_offset` | 相对当前 tracked TCP 起点的终点偏移 |
| `target_position` | 绝对终点位置；和 `target_offset` 二选一 |
| `orientation_mode` | `none` 只约束位置；`current` 保持当前姿态；`target` 使用 `target_orientation` |
| `target_orientation` | `orientation_mode="target"` 时的目标姿态，wxyz 四元数 |
| `start_position` | 可选一致性断言；不参与实际路径构造 |

## TCP Arc Path

`SpecifiedPathMoveSpec + TaskSpacePath + TcpArcSegment` 表示显式指定 TCP 圆弧路径。

```python
SpecifiedPathMoveSpec(
    side="right",
    tcp_frame_name="right_demo_tcp",
    path=TaskSpacePath(
        segments=(
            TcpArcSegment(
                target_offset=(0.0, 0.05, 0.0),
                intermediate_offset=(0.0, 0.03, 0.02),
                arc_mode="three_point",
                constant_orientation=True,
            ),
        ),
    ),
    duration_s=1.6,
    phase="right_tcp_arc",
)
```

`TcpArcSegment` 参数：

| 参数 | 含义 |
| --- | --- |
| `target_offset` | 相对当前 tracked TCP 起点的圆弧终点偏移 |
| `target_position` | 绝对圆弧终点；和 `target_offset` 二选一 |
| `arc_mode` | `tangent` 或 `three_point` |
| `intermediate_offset` | 三点圆弧的相对中间点；`arc_mode="three_point"` 时需要 |
| `intermediate_position` | 三点圆弧的绝对中间点；和 `intermediate_offset` 二选一 |
| `constant_orientation` | 是否沿圆弧保持姿态 |
| `target_orientation` | 可选终点姿态，wxyz 四元数 |

注意：圆弧会通过 cuMotion task-space path conversion 转成 C-space path。路径几何过大、经过奇异位形、碰撞或关节限制时，conversion 可能失败；这时应缩小 offset、调整中间点或放宽/调整 profile 中的 specified-path conversion 参数。

## C-Space Waypoint Path

`SpecifiedPathMoveSpec + CSpaceWaypointPath` 表示直接给出一组 C-space waypoint。

```python
SpecifiedPathMoveSpec(
    side="right",
    tcp_frame_name="right_demo_tcp",
    path=CSpaceWaypointPath(
        waypoints=(
            current_q,
            current_q + delta_q,
        ),
    ),
    duration_s=1.0,
    phase="right_cspace_waypoints",
)
```

注意：`CSpaceWaypointPath` 的 waypoint 顺序必须和后端 `joint_names()` 一致。普通客户脚本通常优先使用绝对 C-space goal、`CSpaceDeltaPlanMoveSpec` 或 task-space path；直接传 C-space waypoint 更适合高级调试。

## Composite Path

`CompositePath` 可以把 C-space 子路径和 task-space 子路径组合起来。

```python
SpecifiedPathMoveSpec(
    side="left",
    tcp_frame_name="left_demo_tcp",
    path=CompositePath(
        parts=(
            TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_offset=(0.0, 0.0, 0.03),
                        orientation_mode="current",
                    ),
                ),
            ),
            CompositePathPart(
                path=TaskSpacePath(
                    segments=(
                        TcpLineSegment(
                            target_offset=(0.02, 0.0, 0.0),
                            orientation_mode="none",
                        ),
                    ),
                ),
                transition_mode="linear_task_space",
            ),
        ),
    ),
    duration_s=2.0,
    phase="left_composite_path",
)
```

## 高级 CumotionMoveSpec

`CumotionMoveSpec` 允许直接传项目侧 `IKRequest`、`MotionRequest` 或 `SpecifiedPathRequest`。

```python
CumotionMoveSpec(
    execution="selected_side",
    side="left",
    tcp_frame_name="left_demo_tcp",
    duration_s=1.0,
    phase="left_absolute_ik",
    request=IKRequest(
        target_position=np.asarray([0.35, -0.2, 0.4], dtype=float),
        target_orientation=None,
        tcp_frame_name="left_demo_tcp",
        avoid_collisions=False,
    ),
)
```

双臂执行中：

- `execution="selected_side"`：只采用选定侧的解，另一侧保持。
- `execution="dual_cspace"`：高级模式，直接使用完整双臂 C-space 结果。

绝对 C-space 目标也可以通过 `MotionRequest(goal_q=...)` 直接表达：

```python
CumotionMoveSpec(
    execution="dual_cspace",
    tcp_frame_name="right_demo_tcp",
    duration_s=1.5,
    phase="dual_cspace_goal",
    request=MotionRequest(
        current_q=current_dual_q,
        goal_q=target_dual_q,
        tcp_frame_name="right_demo_tcp",
    ),
)
```

## 一个完整动作序列

```python
steps = run_dual_arm_cumotion_motion(
    runtime,
    tcp=tcp,
    moves=(
        IkOffsetMoveSpec(
            side="left",
            tcp_frame_name="left_demo_tcp",
            tcp_offset=(0.03, 0.0, 0.02),
            duration_s=1.0,
        ),
        SpecifiedPathMoveSpec(
            side="left",
            tcp_frame_name="left_demo_tcp",
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_offset=(0.0, 0.0, 0.05),
                        orientation_mode="none",
                    ),
                ),
            ),
            duration_s=1.2,
        ),
        CSpaceDeltaPlanMoveSpec(
            side="right",
            tcp_frame_name="right_demo_tcp",
            joint_deltas=(0.1, -0.06, 0.04),
            duration_s=1.2,
            overlays=(
                CommandOverlaySpec(
                    timing="sync",
                    right_hand=HandMoveSpec(
                        side="right",
                        joint_positions={"L6V1_R_hand_index_mcp_pitch": 0.5},
                        duration_s=1.2,
                    ),
                ),
            ),
        ),
        DualHandMoveSpec(
            left=HandMoveSpec(
                side="left",
                joint_positions={"L6V1_L_hand_index_mcp_pitch": 0.0},
                duration_s=0.5,
            ),
            right=HandMoveSpec(
                side="right",
                joint_positions={"L6V1_R_hand_index_mcp_pitch": 0.0},
                duration_s=0.5,
            ),
            duration_s=0.5,
            phase="dual_hand_open",
        ),
    ),
    cumotion_profile="default",
)
```
