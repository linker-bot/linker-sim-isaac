# cuMotion Motion Planner 设计报告

## 1. 设计目标

`motion_planner` 的目标是为动作脚本层提供统一的运动规划入口，屏蔽 cuMotion 后端不同规划 API 的差异。

动作脚本层只需要描述：

- 当前关节状态
- 目标关节 / 目标 TCP 位置 / 目标 TCP 位姿
- 或者显式指定一条路径几何形状
- 使用哪种规划 pipeline
- 轨迹生成参数

后端负责：

- 选择 cuMotion 对应规划接口
- 生成 C-space `path` 或直接生成 `trajectory`
- 统一转换成项目内部 `MotionResult`

目标是支持三种运动生成方式：

1. `trajectory_optimization`: Trajectory Optimization，目标式运动生成的默认路线。
2. `graph_search`: Graph-Based Path Planning / MotionPlanner + Trajectory Generation，作为显式选择的搜索路线。
3. `specified_path`: 用户指定路径族 / Path Generation + Trajectory Generation。

---

## 2. 当前实现概览

当前 `src/linkerbot_sim/backends/cumotion/motion_planner.py` 主要实现的是
`graph_search`：

```text
Graph-Based Path Planning / MotionPlanner
+
CSpaceTrajectoryGenerator
```

当前主流程：

```text
MotionRequest
    |
    v
CuMotionMotionPlanner.plan()
    |
    v
创建 MotionPlannerConfig
    |
    v
create_motion_planner(config)
    |
    v
根据目标类型调用 planner 方法
    |
    v
读取 results.path / results.interpolated_path
    |
    v
可选调用 CSpaceTrajectoryGenerator 做时间参数化
    |
    v
MotionResult
```

也就是说，文档里的三 pipeline 设计是目标架构；当前代码还没有完整拆出
`specified_path` 和 `trajectory_optimization` pipeline。后续大改时，目标默认路线应从当前
已实现的 `graph_search` 切换到 `trajectory_optimization`；`graph_search` 保留为显式选择
或由动作脚本层显式发起第二次规划。

---

## 3. 当前支持的目标类型

当前 `MotionRequest` 支持三类目标，适合 `graph_search` 和
`trajectory_optimization` 这类“目标式请求”。`specified_path` 不应强行塞进
`MotionRequest.goal_q / goal_pose`，它需要独立的路径请求结构。

### 3.1 C-space 目标

条件：

```python
request.goal_q is not None
```

调用：

```python
planner.plan_to_cspace_target(
    current,
    goal,
    self.generate_interpolated_path,
)
```

含义：

```text
从当前 C-space 构型规划到目标 C-space 构型。
```

适合：

- 已经知道目标关节角
- 预设姿态
- IK 已经求好的结果
- 回零位

---

### 3.2 TCP 位置目标

条件：

```python
request.goal_pose is not None
request.goal_pose.orientation is None
```

调用：

```python
planner.plan_to_translation_target(
    current,
    translation,
    self.generate_interpolated_path,
)
```

官方说明：

```text
Attempt to find path to task space translation target using JtRRT.
```

含义：

```text
只要求 TCP 到达目标位置，不约束姿态。
```

适合：

- 靠近某个空间点
- 姿态无所谓的移动

---

### 3.3 TCP 位姿目标

条件：

```python
request.goal_pose is not None
request.goal_pose.orientation is not None
```

调用：

```python
planner.plan_to_pose_target(
    current,
    pose_target,
    self.generate_interpolated_path,
)
```

官方说明：

```text
Attempt to find path to task space pose target using JtRRT.
```

含义：

```text
同时约束 TCP 的位置和姿态。
```

适合：

- 抓取
- 对准
- 插入
- 需要末端姿态的操作

---

## 4. Pipeline 默认碰撞语义

顶层配置只暴露 `planning_pipeline`，碰撞语义放在各 pipeline 配置中。

碰撞语义由 `planning_pipeline` 自身决定：

| Pipeline | 默认碰撞语义 | 说明 |
|---|---|---|
| `graph_search` | 考虑环境障碍 | 使用当前 `collision_world.world_view` 创建 `MotionPlannerConfig` |
| `specified_path` | 不考虑环境障碍 | Path Generation 本身是 collision-unaware；如需安全性，应做后验碰撞检查 |
| `trajectory_optimization` | 考虑环境障碍 | 使用当前 `collision_world.world_view` 创建 `TrajectoryOptimizerConfig` |

这样动作脚本层只需要选择 pipeline，就能决定主要碰撞语义：

```yaml
planning_pipeline: graph_search
```

表示：

```text
使用图搜索路径规划，默认考虑当前环境障碍。
```

```yaml
planning_pipeline: specified_path
```

表示：

```text
使用指定几何路径生成，默认不搜索避障，也不把环境障碍作为规划约束。
```

```yaml
planning_pipeline: trajectory_optimization
```

表示：

```text
使用轨迹优化，默认考虑当前环境障碍。
```

如果需要“忽略障碍的 graph search”或“specified path 后验碰撞检查”，应作为对应 pipeline 的高级选项，而不是顶层 `collision_mode`。

---

## 5. 默认 pipeline 策略

目标式运动请求默认使用：

```yaml
planning_pipeline: trajectory_optimization
```

默认选择 `trajectory_optimization` 的原因是：

- 它直接输出 `Trajectory`，不用先生成离散 path 再二次时间参数化。
- 它更适合把终端目标、路径约束、碰撞约束和轨迹平滑性放在同一个优化问题中处理。
- 对抓取、靠近、抬升、wiggle 等任务，轨迹连续性通常比“先搜索一条折线路径”更重要。

`graph_search` 的定位是：

- 作为显式选择的图搜索 pipeline。
- 作为调试路径可达性、粗略避障路径和参数对比的工具。

facade 不做自动回退。原因是 optimizer 失败和 graph search 成功代表两种不同质量和约束语义；
需要保守路线时，动作脚本层应显式选择 `graph_search` 或自行发起第二次规划。

配置：

```yaml
planning_pipeline: graph_search
```

这样 diagnostics 只记录实际执行的 pipeline，不混合多条路线的状态。

---

## 6. `generate_interpolated_path` 的意义

`generate_interpolated_path` 只对当前的 Graph-Based `MotionPlanner` 有意义。

cuMotion `MotionPlanner.Results` 里有：

```text
path
interpolated_path
path_found
```

其中：

```text
path              稀疏搜索路径
interpolated_path 插值后的较密路径
```

当前代码逻辑是：

```python
names = ("interpolated_path", "path") if prefer_interpolated else ("path",)
```

也就是说：

- `generate_interpolated_path=True`
  - 调用 planner 时要求生成 `interpolated_path`
  - 项目优先消费 `interpolated_path`
  - 如果没有，则回退到 `path`
- `generate_interpolated_path=False`
  - 只消费 `path`

### 注意

`interpolated_path` 是 `MotionPlanner.Results` 的字段。

它不属于：

- `TrajectoryOptimizer`
- `PathGeneration`
- `CSpaceTrajectoryGenerator`

所以多 pipeline 下不要把它作为顶层平铺参数继续扩展命名，例如：

```python
generate_interpolated_path_for_graph_search
```

使用分组参数，让字段保持短名，并通过所属分组表达作用域：

```yaml
graph_search:
    generate_interpolated_path: true
```

也就是说：

```text
graph_search.generate_interpolated_path 仅对 graph_search pipeline 生效。
```

---

## 7. 三种规划 pipeline 的设计

把 `motion_planner` 抽象成三条可选 pipeline。

---

## 8. Pipeline A：Graph-Based Path Planning + Trajectory Generation

### 8.1 对应 cuMotion API

```python
cumotion.MotionPlanner
cumotion.MotionPlannerConfig
cumotion.create_motion_planner(...)
cumotion.CSpaceTrajectoryGenerator
```

核心调用：

```python
planner.plan_to_cspace_target(...)
planner.plan_to_translation_target(...)
planner.plan_to_pose_target(...)
```

后处理：

```python
generator.generate_trajectory(...)
generator.generate_time_stamped_trajectory(...)
```

### 8.2 流程

```text
current_q + target + world_view
    |
    v
MotionPlanner
    |
    v
path / interpolated_path
    |
    v
CSpaceTrajectoryGenerator
    |
    v
Trajectory
```

### 8.3 特点

优点：

- 可以自动搜索避障路径
- 适合只知道起点和终点的任务
- 当前项目已有基础实现，重构后作为独立 pipeline 保留

缺点：

- 先找 `path`，再做时间参数化，路径搜索和平滑轨迹生成是两阶段
- `time_optimal` 在 dense waypoint 下可能出现速度变化频繁
- `interpolated_path` waypoint 过多时可能影响轨迹质量

### 8.4 适用场景

- 常规避障运动
- 从当前关节到目标关节
- 从当前状态到 TCP 目标位置
- 从当前状态到 TCP 目标位姿

---

## 9. Pipeline B：Specified Path Family + Trajectory Generation

这一路线顶层统一命名为 `specified_path`，但它不是一个单一具体实现，而是一组
“调用方显式指定路径几何”的子类型。

它的核心语义不是“给一个目标让 planner 自己搜索”，而是：

```text
调用方显式指定要走的路径几何形状。
```

因此 `specified_path` 应作为总类保留在顶层 pipeline 中，再在内部细分。
不要把每一种指定路径都提升为顶层 pipeline，因为它们的共同本质都是：

```text
路径由调用方给定，不由 graph planner 搜索。
```

### 9.1 对应 cuMotion API

```python
cumotion.CSpacePathSpec
cumotion.TaskSpacePathSpec
cumotion.CompositePathSpec
cumotion.LinearCSpacePath
cumotion.TaskSpacePath
cumotion.convert_task_space_path_spec_to_cspace(...)
cumotion.convert_composite_path_spec_to_cspace(...)
cumotion.CSpaceTrajectoryGenerator
```

### 9.2 流程

```text
用户指定 path family
    |
    v
cspace_waypoints / task_space_segments / composite
    |
    v
Path Generation / Path Conversion
    |
    v
C-space path
    |
    v
CSpaceTrajectoryGenerator
    |
    v
Trajectory
```

### 9.3 子类型

`specified_path` 内部至少分成三类：

| 子类型 | 输入 | 中间处理 | 说明 |
|---|---|---|---|
| `cspace_waypoints` | 一组 C-space waypoint | 直接生成 C-space path | 最简单的指定路径；`[start_q, goal_q]` 是它的退化特例 |
| `task_space_segments` | TCP 直线、平移、旋转、圆弧、位姿序列 | 通过 IK/path conversion 转成 C-space path | `TcpLineSegment` 属于这一类 |
| `composite` | C-space 段和 task-space 段混合 | 转成统一 C-space path | 适合复杂多段任务 |

结构关系：

```text
specified_path
    |
    |-- cspace_waypoints
    |      |
    |      v
    |   CSpacePathSpec / LinearCSpacePath
    |
    |-- task_space_segments
    |      |
    |      |-- tcp_line
    |      |-- tcp_arc
    |      |-- tcp_rotation
    |      |-- tcp_pose_sequence
    |      v
    |   TaskSpacePathSpec -> convert_task_space_path_spec_to_cspace(...)
    |
    |-- composite
           |
           v
        CompositePathSpec -> convert_composite_path_spec_to_cspace(...)
```

### 9.4 特点

优点：

- 路径形状可控
- 适合 TCP 直线、圆弧、组合路径
- 比图搜索结果更可解释

缺点：

- Path Generation 本身是 collision-unaware
- 不会主动搜索避障路径
- 需要用户明确指定路径段
- `MotionRequest` 需要扩展 path spec 描述能力

### 9.5 `cspace_waypoints`

`cspace_waypoints` 是指定路径族里最轻量的一类。调用方直接给出一组按后端
C-space 关节顺序排列的 waypoint：

```python
CSpaceWaypointPath(
    waypoints=(
        start_q,
        mid_q,
        goal_q,
    )
)
```

后端可以把它转成：

```text
CSpacePathSpec
    -> LinearCSpacePath
    -> C-space waypoint path
    -> CSpaceTrajectoryGenerator
```

如果只有两个 waypoint：

```text
[start_q, goal_q]
```

它就等价于“关节空间直连 + trajectory generator”。如果目标来自单点 IK：

```text
TCP target -> IK -> goal_q -> [start_q, goal_q]
```

这仍然是 `cspace_waypoints` 的退化特例，而不是第四种本质 pipeline。

### 9.6 `task_space_segments`

`task_space_segments` 描述 TCP 在任务空间中的路径几何，例如：

- TCP 直线移动
- TCP 平移段
- TCP 原地旋转
- TCP 圆弧
- TCP 位姿序列

理想实现可以映射到 cuMotion 官方接口：

```text
TaskSpacePathSpec
    -> TaskSpacePath
    -> convert_task_space_path_spec_to_cspace(...)
    -> C-space path
    -> CSpaceTrajectoryGenerator
```

### 9.7 `TcpLineSegment` 作为 task-space segment

TCP 直线不是典型的：

```text
起点 + 终点，让 planner 自己搜索路径
```

而是：

```text
明确指定 TCP 要沿一条直线走。
```

因此它直接建模为 `TaskSpacePath` 的一个 segment：

```python
SpecifiedPathRequest(
    current_q=current_q,
    tcp_frame_name="pinch_tcp",
    duration_s=duration_s,
    path=TaskSpacePath(
        segments=(
            TcpLineSegment(
                target_position=target_position,
                orientation_mode="current",
            ),
        )
    ),
)
```

后端映射到 cuMotion 官方路径接口：

```python
TaskSpacePathSpec.add_linear_path(...)
convert_task_space_path_spec_to_cspace(...)
CSpaceTrajectoryGenerator.generate_trajectory(...)
```

旧的 `TcpLineRequest` / `plan_tcp_line_joint_path(...)` 逐点 IK 辅助不再保留；动作脚本层需要 TCP
直线时直接发 `SpecifiedPathRequest(TaskSpacePath(TcpLineSegment(...)))`。

### 9.8 `composite`

`composite` 用来拼接 C-space 段和 task-space 段，例如：

```text
先关节空间绕到预备姿态
-> TCP 沿直线靠近
-> TCP 沿圆弧绕过某个工艺点
-> 关节空间回到安全姿态
```

理想实现可以映射到：

```text
CompositePathSpec
    -> convert_composite_path_spec_to_cspace(...)
    -> C-space path
    -> CSpaceTrajectoryGenerator
```

这类请求最适合复杂多段任务，但也最需要清晰的段间过渡语义。

### 9.9 适用场景

- TCP 直线移动
- 沿圆弧移动
- 指定 C-space waypoint 路径
- 工艺路径
- 已知安全路径的重复执行

---

## 10. Pipeline C：Trajectory Optimization

### 10.1 对应 cuMotion API

```python
cumotion.TrajectoryOptimizer
cumotion.TrajectoryOptimizerConfig
cumotion.create_trajectory_optimizer(...)
```

核心调用：

```python
optimizer.plan_to_cspace_target(...)
optimizer.plan_to_task_space_target(...)
optimizer.plan_to_task_space_goalset(...)
```

### 10.2 流程

```text
current_q + target + world_view + constraints
    |
    v
TrajectoryOptimizer
    |
    v
Trajectory
```

### 10.3 特点

优点：

- 直接输出 `Trajectory`
- 不需要 `path -> trajectory` 两阶段转换
- 理论上更适合平滑、约束一致的轨迹
- 可以把目标、路径约束、碰撞约束放进优化问题

缺点：

- 接入复杂度高于 `MotionPlanner`
- 需要单独处理 `TrajectoryOptimizer` 的 target / constraint 类型
- 失败状态比 `MotionPlanner` 更多
- 调参复杂度更高

### 10.4 适用场景

- 希望直接得到平滑轨迹
- 对轨迹连续性要求更高
- 希望减少 graph path 和 time parameterization 之间的割裂
- 复杂约束轨迹

---

## 11. 为什么不保留第四种模式

“起始关节 + 目标关节 + trajectory generator”可以看成 `specified_path.cspace_waypoints`
的退化特例：

```text
[start_q, goal_q]
    |
    v
CSpaceTrajectoryGenerator
    |
    v
Trajectory
```

如果目标来自 TCP 位姿或位置，也只是多了一步单点 IK：

```text
TCP target
    |
    v
IK -> goal_q
    |
    v
[start_q, goal_q]
    |
    v
CSpaceTrajectoryGenerator
```

它不应提升为独立顶层 pipeline，因为它没有新的规划语义；它只是最简单的指定 C-space
waypoint path。

---

## 12. 对外抽象

`MotionRequest` 只描述当前状态和目标；碰撞策略由 motion planner pipeline 配置表达。

对外主抽象只保留一个 pipeline 选择：

```python
planning_pipeline: Literal[
    "graph_search",
    "specified_path",
    "trajectory_optimization",
]
```

### 12.1 `planning_pipeline`

负责：

```text
使用哪条规划路线，并隐含该路线的默认碰撞语义。
```

可选值：

```text
graph_search
specified_path
trajectory_optimization
```

对应默认碰撞语义：

| `planning_pipeline` | 默认碰撞语义 |
|---|---|
| `graph_search` | 使用当前环境障碍 |
| `specified_path` | 不使用环境障碍作为规划约束 |
| `trajectory_optimization` | 使用当前环境障碍 |

如果要支持调试用途的覆盖项，例如 graph search 忽略环境障碍，应放在 pipeline 分组内：

```yaml
graph_search:
    use_environment_obstacles: false
```

这属于高级覆盖项，不是主抽象。

---

## 13. 文件结构

把 facade、pipeline 实现、配置模型拆分清楚：

```text
src/linkerbot_sim/backends/cumotion/
    motion_planner.py                    # facade / 统一入口
    motion_planner_config.py             # cuMotion motion planner 分组配置 dataclass
    graph_motion_planner.py              # Graph-Based MotionPlanner pipeline
    trajectory_optimizer_planner.py      # TrajectoryOptimizer pipeline
    specified_path_planner.py            # Specified Path / PathGeneration pipeline
    trajectory_generation.py             # CSpaceTrajectoryGenerator 封装
    trajectory_sampler.py                # cuMotion Trajectory -> 项目 JointTrajectory
    pose_adapter.py                      # pose / quaternion / matrix 适配
```

职责如下：

| 文件 | 职责 |
|---|---|
| `motion_planner.py` | facade，统一入口，根据 pipeline 分发 |
| `motion_planner_config.py` | 定义 `MotionPlannerBackendConfig` 及各分组配置 |
| `graph_motion_planner.py` | MotionPlanner graph-search 逻辑 |
| `trajectory_optimizer_planner.py` | TrajectoryOptimizer pipeline |
| `specified_path_planner.py` | PathSpec / PathGeneration pipeline |
| `trajectory_generation.py` | 统一封装 `CSpaceTrajectoryGenerator` 时间参数化 |
| `trajectory_sampler.py` | cuMotion Trajectory 转项目 JointTrajectory |
| `pose_adapter.py` | pose / quaternion / matrix 适配 |

---

## 14. `MotionResult` 统一输出

三条 pipeline 最终都应该统一返回：

```python
MotionResult
```

字段语义保持为：

```python
MotionResult(
    path=...,
    trajectory=...,
    success=...,
    status=...,
    diagnostics=...,
)
```

不同 pipeline 的字段含义可以是：

| Pipeline | `path` | `trajectory` |
|---|---|---|
| `graph_search` | planner path / interpolated_path | 成功时必须由 generator 生成 |
| `specified_path` | path spec 转出的 C-space waypoints | 成功时必须由 generator 生成 |
| `trajectory_optimization` | 可选，从 trajectory 采样得到 | optimizer 直接返回 |

对于 `trajectory_optimization`，因为它直接返回 `Trajectory`，可以：

```text
trajectory = results.trajectory()
joint_path = None
```

这是推荐默认语义：`trajectory` 是主输出，`joint_path` 不是必需产物。

如果动作脚本层需要离散采样或做日志诊断，可以：

```text
按 physics_dt 或诊断采样间隔从 trajectory eval 出 joint_path
```

但采样得到的 `joint_path` 应标记为 diagnostic path，不能再当作原始优化器输出。

---

## 15. 参数设计

使用分组参数形式，不把所有参数平铺到 `CuMotionMotionPlanner.__init__` 或 YAML 顶层。

这样可以避免类似：

```python
generate_interpolated_path_for_graph_search
```

这种过长命名。

使用：

```yaml
graph_search:
    generate_interpolated_path: true
```

字段名由分组限定作用域，语义更清楚。

### 15.1 顶层通用参数

```yaml
planning_pipeline: trajectory_optimization
```

顶层只放 pipeline 选择项：

- `planning_pipeline`: 选择运动生成路线，并隐含默认碰撞语义；默认值为 `trajectory_optimization`

不要把 trajectory、graph search、optimizer 的细节参数放在顶层。

### 15.2 Graph Search 专属参数

```yaml
graph_search:
  generate_interpolated_path: true
  motion_planner_params:
    step_size: 0.02
```

对应：

```python
MotionPlannerConfig.set_param(...)
```

### 15.3 Trajectory Generation 参数

`trajectory_generation` 只服务于会产生 C-space path 的 pipeline：

- `graph_search`
- `specified_path`

`trajectory_optimization` 的主输出是 `Trajectory`；`CSpaceTrajectoryGenerator` 服务于会产生
C-space path 的 pipeline。

```yaml
trajectory_generation:
  mode: time_optimal
  interpolation_mode: cubic_spline
  limits:
    velocity: [...]
    acceleration: [...]
    jerk: [...]
  solver_params:
    ...
```

对应：

```python
CSpaceTrajectoryGenerator
```

### 15.4 Trajectory Optimization 参数

```yaml
trajectory_optimization:
  config_path: ...
  params:
    ...
```

对应：

```python
TrajectoryOptimizerConfig.set_param(...)
```

optimizer 失败时直接返回失败结果。需要 graph-search 作为保守路线时，由动作脚本层显式切换
`planning_pipeline: graph_search` 后重新规划。

### 15.5 Specified Path 参数

```yaml
specified_path:
  family: task_space_segments
  validate_collision_after_generation: false
  task_space_segments:
    default_conversion:
      max_iterations: 100
      min_position_deviation: 0.001
  cspace_waypoints:
    interpolation: linear
  composite:
    transition_mode: free
```

这里的 `family` 只描述默认指定路径族。真实路径几何仍应放在请求对象中，而不是全部放进
planner 配置。配置只保存转换、后验检查、段间过渡等策略参数。

`specified_path` 对应的请求建模为独立的 specified-path 请求，不继续塞进只描述单一目标的 `MotionRequest`。

### 15.6 完整配置示例

```yaml
planning_pipeline: trajectory_optimization

graph_search:
    generate_interpolated_path: true
    motion_planner_params:
        step_size: 0.02

trajectory_generation:
    mode: time_optimal
    interpolation_mode: cubic_spline
    limits:
        velocity: [...]
        acceleration: [...]
        jerk: [...]
    solver_params: {}

trajectory_optimization:
    config_path: null
    params: {}

specified_path:
    family: task_space_segments
    validate_collision_after_generation: false
    cspace_waypoints:
        interpolation: linear
    task_space_segments:
        default_conversion: {}
    composite:
        transition_mode: free
```

### 15.7 Python 侧配置模型

使用强类型 dataclass 配置，而不是在 planner 内使用松散 dict 作为主要配置模型。

配置入口：

```python
@dataclass(frozen=True)
class MotionPlannerBackendConfig:
    planning_pipeline: PlanningPipeline = "trajectory_optimization"
    graph_search: GraphSearchConfig = field(default_factory=GraphSearchConfig)
    trajectory_generation: TrajectoryGenerationConfig = field(
        default_factory=TrajectoryGenerationConfig
    )
    trajectory_optimization: TrajectoryOptimizationConfig = field(
        default_factory=TrajectoryOptimizationConfig
    )
    specified_path: SpecifiedPathConfig = field(default_factory=SpecifiedPathConfig)
```

各分组配置示例：

```python
@dataclass(frozen=True)
class GraphSearchConfig:
    generate_interpolated_path: bool = True
    use_environment_obstacles: bool = True
    motion_planner_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    mode: Literal["time_optimal", "time_stamped"] = "time_optimal"
    interpolation_mode: Literal["linear", "cubic_spline"] = "cubic_spline"
    limits: Mapping[str, Any] = field(default_factory=dict)
    solver_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryOptimizationConfig:
    config_path: Path | None = None
    use_environment_obstacles: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecifiedPathConfig:
    family: Literal[
        "cspace_waypoints",
        "task_space_segments",
        "composite",
    ] = "task_space_segments"
    validate_collision_after_generation: bool = False
    cspace_waypoints: Mapping[str, Any] = field(default_factory=dict)
    task_space_segments: Mapping[str, Any] = field(default_factory=dict)
    composite: Mapping[str, Any] = field(default_factory=dict)
```

这样比字符串字典更容易做类型检查、默认值管理和配置校验。

`CuMotionMotionPlanner` 的构造参数收敛为：

```python
class CuMotionMotionPlanner:
    def __init__(
        self,
        context,
        *,
        config: MotionPlannerBackendConfig | None = None,
    ) -> None:
        ...
```

其中 `config=None` 时使用 `context.config.motion_planner` 或默认 `MotionPlannerBackendConfig()`。

---

## 16. 请求模型

区分“目标式请求”和“指定路径请求”。

### 16.1 目标式请求

`MotionRequest` 适合继续表达：

- `current_q + goal_q`
- `current_q + goal_pose.position`
- `current_q + goal_pose.position + goal_pose.orientation`

它主要供以下 pipeline 使用：

- `graph_search`
- `trajectory_optimization`

`MotionRequest` 不需要再携带 `mode="collision_aware" / "collision_unaware"`。是否使用环境障碍由所选 pipeline 的默认语义决定。

### 16.2 指定路径请求

`specified_path` 不应强行塞进 `MotionRequest.goal_q / goal_pose`，而应使用独立请求结构，例如：

```python
@dataclass(frozen=True)
class SpecifiedPathRequest:
    current_q: np.ndarray
    tcp_frame_name: str | None = None
    path: CSpaceWaypointPath | TaskSpacePath | CompositePath
    duration_s: float | None = None
```

其中 `path` 是指定路径族的具体几何描述：

```python
@dataclass(frozen=True)
class CSpaceWaypointPath:
    waypoints: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TaskSpacePath:
    segments: tuple[TaskSpaceSegment, ...]


@dataclass(frozen=True)
class CompositePath:
    parts: tuple[CSpaceWaypointPath | TaskSpacePath, ...]
```

`TaskSpaceSegment` 再细分为具体路径段：

```python
TaskSpaceSegment = (
    TcpLineSegment
    | TcpArcSegment
    | TcpRotationSegment
    | TcpPoseSequenceSegment
)
```

TCP 直线收敛成 `TaskSpacePathSegment` 的一种：

```python
@dataclass(frozen=True)
class TcpLineSegment:
    start_position: np.ndarray | None = None
    target_position: np.ndarray | None = None
    target_offset: np.ndarray | None = None
    orientation_mode: Literal["current", "target", "none"] = "current"
    target_orientation: np.ndarray | None = None
```

这样 `specified_path` pipeline 可以统一处理 TCP 直线、圆弧、C-space waypoint 和 composite path。它默认不把环境障碍作为规划约束；若需要安全保证，应通过后验碰撞检查或动作脚本层预先保证路径安全。

### 16.3 请求与 pipeline 的对应关系

| 请求类型 | 适用 pipeline | 说明 |
|---|---|---|
| `MotionRequest(current_q, goal_q)` | `graph_search` / `trajectory_optimization` | 目标式 C-space 请求 |
| `MotionRequest(current_q, goal_pose)` | `graph_search` / `trajectory_optimization` | 目标式 task-space 请求 |
| `SpecifiedPathRequest(path=CSpaceWaypointPath(...))` | `specified_path.cspace_waypoints` | 调用方直接指定关节路径 |
| `SpecifiedPathRequest(path=TaskSpacePath(...))` | `specified_path.task_space_segments` | 调用方指定 TCP 路径段 |
| `SpecifiedPathRequest(path=CompositePath(...))` | `specified_path.composite` | 调用方混合指定 C-space/task-space 段 |

---

## 17. 风险与注意事项

### 17.1 不同 pipeline 的输出层级不同

```text
MotionPlanner 输出 path
PathGeneration 输出 path
TrajectoryOptimizer 输出 trajectory
```

所以统一封装时不要强行要求所有 pipeline 都有 `interpolated_path`。

### 17.2 `interpolated_path` 只属于 Graph Search

不要让：

```python
graph_search.generate_interpolated_path
```

影响：

```text
trajectory_optimization
path_generation
```

否则语义会混乱。

### 17.3 Pipeline 碰撞语义不要重复建模

不要同时暴露：

```yaml
planning_pipeline: specified_path
collision_mode: collision_aware
```

这种组合会制造语义冲突，因为 `specified_path / path_generation` 本身不是避障搜索规划器。

如果确实需要覆盖默认行为，应放在 pipeline 分组内，并明确它是高级选项，例如：

```yaml
graph_search:
    use_environment_obstacles: false

specified_path:
    validate_collision_after_generation: true
```

### 17.4 Specified Path / PathGeneration 默认不避障

如果用 PathGeneration 生成 TCP 直线、C-space waypoint path 或 composite path，仍需要额外碰撞检查。

否则路径可能穿过障碍物。

### 17.5 Trajectory Generation 是 collision-unaware

`CSpaceTrajectoryGenerator` 只负责时间参数化和运动学限制。

它不会重新规划避障。

因此：

```text
输入 path 是否安全
```

必须由前面的 planner 或额外 collision check 保证。

---

## 18. 最终架构

```text
MotionRequest / PathRequest
    |
    v
CuMotionMotionPlanner facade
    |
    |-- graph_search
    |      |
    |      v
    |   MotionPlanner
    |      |
    |      v
    |   path / interpolated_path
    |      |
    |      v
    |   CSpaceTrajectoryGenerator
    |
    |-- specified_path
    |      |
    |      v
    |   CSpaceWaypointPath / TaskSpacePath / CompositePath
    |      |
    |      v
    |   PathSpec / PathGeneration / Path Conversion
    |      |
    |      v
    |   C-space path
    |      |
    |      v
    |   CSpaceTrajectoryGenerator
    |
    |-- trajectory_optimization
           |
           v
        TrajectoryOptimizer
           |
           v
        Trajectory
```

统一输出：

```python
MotionResult
```

---

## 19. 实施计划

本节作为代码修改的施工顺序；以实用性、清晰边界和测试可验证性为准。

### 19.1 Phase 1：配置和 facade 骨架

目标：

- 建立 motion planner 分组配置模型。
- 让 `CuMotionMotionPlanner` 作为 facade 分发 pipeline。
- 让 `MotionRequest` 只表达当前状态和目标。

修改内容：

- 新增 `src/linkerbot_sim/backends/cumotion/motion_planner_config.py`。
- 新增 dataclass：
  - `MotionPlannerBackendConfig`
  - `GraphSearchConfig`
  - `TrajectoryGenerationConfig`
  - `TrajectoryOptimizationConfig`
  - `SpecifiedPathConfig`
- 修改 `planning.requests.MotionRequest`：
  - 保留 `current_q`、`goal_q`、`goal_pose`、`tcp_frame_name`、`duration_s`
- 修改 `CuMotionContext.make_motion_planner(...)`：
  - 只接收 `config: MotionPlannerBackendConfig | None = None`
  - graph search、trajectory generation 和 optimizer 参数都通过分组配置传入
- 修改 `CuMotionMotionPlanner`：
  - 只作为 facade
  - 根据 `config.planning_pipeline` 分发到具体 pipeline

验收标准：

- `MotionRequest` 的构造参数只包含状态、目标、TCP frame 和时长。
- `context.make_motion_planner(...)` 只接收 `tcp_frame_name` 和 `config`。
- 默认 `MotionPlannerBackendConfig().planning_pipeline == "trajectory_optimization"`。
- 调用点使用 `MotionPlannerBackendConfig` 分组配置。

### 19.2 Phase 2：抽出 trajectory generation

目标：

- 把 `CSpaceTrajectoryGenerator` 封装从 graph planner 中独立出来。
- 让 `graph_search` 和 `specified_path` 共享同一个时间参数化入口。

修改内容：

- 新增 `src/linkerbot_sim/backends/cumotion/trajectory_generation.py`。
- 提供函数或类：

```python
generate_cspace_trajectory(
    context,
    joint_path: np.ndarray,
    config: TrajectoryGenerationConfig,
    duration_s: float | None,
)
```

- 支持：
  - `mode="time_optimal"`
  - `mode="time_stamped"`
  - `interpolation_mode="linear" | "cubic_spline"`
  - `limits`
  - `solver_params`

验收标准：

- graph-search pipeline 通过共享 trajectory generation helper 生成时间参数化轨迹。
- trajectory generation 对非法 limit key 给出清晰 `ValueError`。
- graph-search / specified-path 成功时强制返回 `trajectory`；不再提供禁用 trajectory generation 的参数。

### 19.3 Phase 3：graph_search pipeline

目标：

- 把当前 `motion_planner.py` 中的 graph planner 逻辑移动到独立 pipeline。

修改内容：

- 新增 `src/linkerbot_sim/backends/cumotion/graph_motion_planner.py`。
- 实现：

```python
plan_graph_search(context, request: MotionRequest, config: MotionPlannerBackendConfig) -> MotionResult
```

- 使用：
  - `GraphSearchConfig.generate_interpolated_path`
  - `GraphSearchConfig.use_environment_obstacles`
  - `GraphSearchConfig.motion_planner_params`
  - `TrajectoryGenerationConfig`

验收标准：

- `goal_q` 调用 `plan_to_cspace_target(...)`。
- `goal_pose.position` 调用 `plan_to_translation_target(...)`。
- `goal_pose.position + orientation` 调用 `plan_to_pose_target(...)`。
- `graph_search.generate_interpolated_path` 只影响 graph-search pipeline。
- `MotionResult.diagnostics.message` 或扩展后的 metadata 中记录实际 pipeline。
- `MotionResult.diagnostics.metrics` 至少包含以下数值：
  - `num_waypoints`
  - `num_collision_objects`
  - `path_length`

### 19.4 Phase 4：trajectory_optimization pipeline

目标：

- 接入默认 pipeline。
- 目标式请求优先走 optimizer。

修改内容：

- 新增 `src/linkerbot_sim/backends/cumotion/trajectory_optimizer_planner.py`。
- 实现：

```python
plan_trajectory_optimization(
    context,
    request: MotionRequest,
    config: MotionPlannerBackendConfig,
) -> MotionResult
```

- 支持：
  - `goal_q`
  - `goal_pose.position`
  - `goal_pose.position + orientation`
- 创建 optimizer config：
  - 有 `TrajectoryOptimizationConfig.config_path` 时从文件创建
  - 否则创建默认 config
  - 应用 `TrajectoryOptimizationConfig.params`
- 使用 `TrajectoryOptimizationConfig.use_environment_obstacles` 选择当前 world 或 empty world。
- 失败即返回 optimizer 的失败结果。

验收标准：

- facade 默认调用 trajectory optimizer。
- optimizer 成功时：
  - `MotionResult.trajectory` 是 optimizer trajectory
  - `MotionResult.path` 默认为 `None`
  - diagnostics 记录实际 pipeline 为 `trajectory_optimization`
- optimizer 失败时：
  - `success=False`
  - 不调用 graph-search

### 19.5 Phase 5：specified_path

目标：

- 建立 specified-path 请求模型。
- 使用 cuMotion 官方 PathSpec/path conversion 完成 `cspace_waypoints`、`task_space_segments` 和 `composite` 三类指定路径。

修改内容：

- 在 `planning.requests` 中新增：
  - `SpecifiedPathRequest`
  - `CSpaceWaypointPath`
  - `TaskSpacePath`
  - `CompositePath`
  - `TcpLineSegment`
  - `TcpRotationSegment`
  - `TcpArcSegment`
  - `TcpPoseSequenceSegment`
  - `CompositePathPart`
- 新增 `src/linkerbot_sim/backends/cumotion/specified_path_planner.py`。
- 新增 `src/linkerbot_sim/backends/cumotion/path_spec_adapter.py`。
- 支持的请求：

```python
SpecifiedPathRequest(path=CSpaceWaypointPath(...))
SpecifiedPathRequest(path=TaskSpacePath(...))
SpecifiedPathRequest(path=CompositePath(...))
```

- `CSpaceWaypointPath` 映射到 `CSpacePathSpec` + `LinearCSpacePath`。
- `TaskSpacePath` 映射到 `TaskSpacePathSpec` + `convert_task_space_path_spec_to_cspace(...)`。
- `CompositePath` 映射到 `CompositePathSpec` + `convert_composite_path_spec_to_cspace(...)`。
- 不静默 fallback 到 `tcp_line.py` 逐点 IK。
- 不保留独立 `tcp_line.py` / `MoveTcpLineConfig` 路线。

验收标准：

- `CSpaceWaypointPath` 至少要求 2 个 waypoint。
- 所有 waypoint 宽度必须等于 `context.expected_cspace_width`。
- 输出 `path` 等于请求指定的 C-space waypoints。
- 成功时强制生成 `trajectory`。
- `TaskSpacePath` / `CompositePath` 能通过官方 path conversion 生成 C-space path。

### 19.6 Phase 6：动作脚本层调用

目标：

- 让动作脚本层使用 facade 和分组配置模型。

修改内容：

- `pinch_grasp.py`：
  - 用 `MotionPlannerBackendConfig` 构造 motion planner 配置。
  - 默认关节目标阶段走 `trajectory_optimization`。
  - 如需要图搜索路线，在动作配置中显式设置 `planning_pipeline: graph_search`。
  - 从 approach 点下沉到 grasp 点的 TCP 直线使用 `SpecifiedPathRequest(TaskSpacePath(TcpLineSegment(...)))`。

验收标准：

- 项目内的运动规划请求使用 `MotionRequest` / `SpecifiedPathRequest`。
- 项目内的 motion planner 构造使用 `MotionPlannerBackendConfig`。
- 动作配置中的 motion planning 配置使用分组结构。

## 20. 测试计划

### 20.1 单元测试

必须新增或更新以下测试：

- `test_motion_planner_config_defaults_to_trajectory_optimization`
- `test_motion_request_rejects_mode_keyword`
- `test_facade_dispatches_to_trajectory_optimizer_by_default`
- `test_trajectory_optimizer_failure_returns_failure_directly`
- `test_trajectory_optimizer_config_rejects_fallback_pipeline`
- `test_graph_search_uses_graph_config_and_trajectory_generation`
- `test_trajectory_generation_rejects_unknown_limit_keys`
- `test_specified_path_cspace_waypoints_generates_joint_path`
- `test_specified_path_cspace_requires_start_match`
- `test_specified_path_tcp_line_none_orientation_uses_add_translation`
- `test_specified_path_tcp_rotation_uses_add_rotation`
- `test_specified_path_three_point_arc_uses_official_arc_api`
- `test_specified_path_composite_converts_to_cspace`

### 20.2 Fake cuMotion 覆盖

测试 fake 应覆盖这些后端方法：

- `create_default_motion_planner_config(...)`
- `create_motion_planner_config_from_file(...)`
- `create_motion_planner(...)`
- `create_cspace_trajectory_generator(...)`
- `create_default_trajectory_optimizer_config(...)`
- `create_trajectory_optimizer_config_from_file(...)`
- `create_trajectory_optimizer(...)`
- optimizer results 的：
  - `status()`
  - `trajectory()`
  - `target_index()`

### 20.3 集成验收

至少运行：

```bash
pytest tests/test_cumotion_motion_planner.py
pytest tests/test_cumotion_context.py
pytest tests/test_pinch_grasp_motion_planning.py
pytest tests/test_system_configs.py
```

如果真实 cuMotion 环境可用，再增加一个 smoke：

```text
current_q -> goal_q
planning_pipeline=trajectory_optimization
```

成功标准：

- 默认 pipeline 是 `trajectory_optimization`。
- graph-search 只能通过显式配置被调用。
- specified-path 的 C-space waypoint 可用。
- 未实现的 task-space/composite path 明确失败。

---

## 21. 结论

把这三种方式作为 `motion_planner` 的可选运动方式是合适的。

定位如下：

```text
graph_search:
    显式选择的搜索路线，自动搜索避障 path，再做时间参数化；默认考虑当前环境障碍。

specified_path:
    用户指定路径族，内部细分为 cspace_waypoints、task_space_segments 和 composite。
    所有子类型最终都应转换成 C-space path，再做时间参数化；默认不做避障搜索。
    TCP 直线通过 TaskSpacePath(TcpLineSegment(...)) 表达。

trajectory_optimization:
    目标式请求的默认路线，直接优化出 trajectory。
    适合对轨迹平滑性和约束一致性要求更高的任务；默认考虑当前环境障碍。
    失败时直接返回失败结果，不自动调用其它 pipeline。
```

最终顶层配置只需要：

```text
planning_pipeline    使用哪种运动生成路线，默认 trajectory_optimization
```

是否使用环境障碍由 pipeline 默认语义决定。这样 `motion_planner` 会更清晰，也避免 `collision_mode` 与 pipeline 能力边界冲突。
