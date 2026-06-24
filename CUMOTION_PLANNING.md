# cuMotion 统一解算架构设计

本文档重新定义本项目的运动计算文件架构：**抛弃所有 Lula 路径，所有 IK、FK、路径规划、避障、轨迹生成和规划诊断都只依赖 cuMotion**。

这不是“cuMotion 优先、其它后端兜底”的迁移方案，而是单一解算栈方案。后续代码中不再保留 Lula solver、Lula 配置、`auto` 后端选择、legacy 参数别名或兼容 fallback。cuMotion 不可用时应直接失败并给出明确安装/环境错误。

## 核心决策

- 唯一解算后端是 cuMotion。
- 机器人模型统一使用 XRDF + URDF。
- 所有调用 cuMotion Python API 的代码集中在 `backends/cumotion/`。
- 任务层不直接调用 cuMotion API，只提交项目内部请求对象。
- `ik/` 不再表示“可切换后端的 IK 包”；目标状态下要么只保留通用数据类型，要么整体并入 `planning/`。
- 删除 `lula_solver.py`、`solver_factory.py` 中的 Lula/auto 分支、配置文件里的 `lula:` 段和脚本里的 `--ik-backend lula/auto`。

目标依赖方向：

```text
tasks/
  -> planning/
      -> backends/cumotion/
          -> cumotion

trajectories/
controllers/
logging/
visualization/
  <- planning results
```

`tasks/` 描述“要做什么”，`planning/` 负责“用哪条规划流程做”，`backends/cumotion/` 负责“如何调用 cuMotion 做计算”。

## 当前问题

当前实现已经可以通过 cuMotion 求解 `pinch_grasp` 的离散 TCP 目标，但文件职责仍带有旧的多后端设计痕迹：

- `make_ik_solver()` 在 cuMotion 和 Lula 之间切换。
- `backend="auto"` 会在 cuMotion 不可用时退回 Lula。
- `configs/robots/*` 中仍保留 `lula:` 兼容配置。
- `pinch_grasp.py` 直接围绕 IK waypoint 编排，后续如果加入避障 planner 容易继续膨胀。
- cuMotion robot model、kinematics、IK config、后续 collision world 和 motion planner 还没有统一上下文。

新架构要解决的是：项目只有一个真实解算核心 cuMotion，但要把 **逆运动学、正运动学、笛卡尔路径、碰撞世界、运动规划、轨迹适配** 拆成清晰文件。

## 目标文件架构

推荐目标结构如下：

```text
source/manipulation_project/
├── backends/
│   ├── __init__.py
│   └── cumotion/
│       ├── __init__.py
│       ├── config.py               # cuMotion 默认参数、配置合并、校验
│       ├── context.py              # robot_description、kinematics、TCP、共享资源缓存
│       ├── pose_adapter.py         # numpy pose/quaternion <-> cuMotion Pose3/Rotation3
│       ├── forward_kinematics.py   # FK、frame 查询、joint/frame 名称
│       ├── inverse_kinematics.py   # 单点/批量/连续 IK，可选目标构型避障
│       ├── motion_planner.py       # cuMotion 路径规划、避障规划
│       ├── collision_world.py      # 项目 CollisionObject -> cuMotion world
│       ├── scene_sync.py           # Isaac stage/env 对象 -> CollisionObject
│       ├── trajectory_adapter.py   # cuMotion 输出 -> JointTrajectory/TrajectoryPoint
│       └── diagnostics.py          # 规划耗时、误差、失败原因、距离指标
├── planning/
│   ├── __init__.py
│   ├── requests.py                 # IKRequest、MotionRequest、CartesianPathRequest
│   ├── results.py                  # IKResult、MotionResult、PlanningDiagnostics
│   ├── targets.py                  # JointTarget、PoseTarget、CartesianWaypoint
│   ├── collision_objects.py        # 后端外部统一障碍物数据结构
│   ├── pipeline.py                 # 高层 PlanningPipeline
│   ├── strategies.py               # ik_waypoints/cartesian_ik/collision_aware
│   └── time_parameterization.py    # 规划路径到时间轨迹，可复用 trajectories/
├── trajectories/
│   ├── base.py                     # TrajectoryPoint、JointTrajectory
│   ├── interpolation.py
│   └── joint_trajectory.py
└── tasks/
    ├── move_tcp_to_pose.py         # 构造请求，不直接求解
    ├── move_tcp_line.py            # 构造 CartesianPathRequest
    └── pinch_grasp.py              # 抓取阶段编排，不内置 planner 细节
```

### 旧文件去向

| 当前文件 | 目标处理 |
| --- | --- |
| `ik/cumotion_solver.py` | 移到 `backends/cumotion/inverse_kinematics.py` |
| `ik/lula_solver.py` | 删除 |
| `ik/solver_factory.py` | 改成 cuMotion-only 兼容入口，拒绝 `auto`/`lula` |
| `ik/ik_request.py` | 移到 `planning/requests.py` |
| `ik/ik_result.py` | 移到 `planning/results.py` |
| `ik/tcp_urdf_builder.py` | 保留或移到 `backends/cumotion/tcp_urdf_builder.py`，因为它服务 cuMotion URDF frame |

`trajectories/` 继续作为轨迹容器和插值工具，不直接依赖 cuMotion。cuMotion 的结果转换放在 `backends/cumotion/trajectory_adapter.py`。

## cuMotion 后端层

`backends/cumotion/` 是唯一允许 `import cumotion` 的地方。外层代码只接触项目内部 request/result 数据结构。

### `context.py`

`CuMotionContext` 是所有 cuMotion 计算的共享入口，负责缓存机器人模型和默认配置。

职责：

- 接收 XRDF、URDF、默认 TCP frame 和 solver 配置。
- 调用 `cumotion.load_robot_from_file()`。
- 保存 `robot_description`、`kinematics`。
- 提供 `joint_names()`、`frame_names()`、`has_frame()`。
- 创建 IK solver、motion planner、collision world adapter。
- 管理默认 seeds、容差、迭代次数、姿态权重。

示意接口：

```python
class CuMotionContext:
    def __init__(self, config: CuMotionConfig):
        ...

    def joint_names(self) -> list[str]:
        ...

    def frame_names(self) -> list[str]:
        ...

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None) -> CuMotionInverseKinematics:
        ...

    def make_motion_planner(self, collision_world=None) -> CuMotionMotionPlanner:
        ...
```

### `config.py`

统一处理配置合并和校验。建议定义：

```python
@dataclass(frozen=True)
class CuMotionConfig:
    xrdf_path: Path
    urdf_path: Path
    flange_frame: str
    default_tcp_frame: str
    cspace_seeds: np.ndarray | None = None
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.75
    ccd_max_iterations: int = 180
    bfgs_max_iterations: int = 80
    orientation_weight: float = 0.25
```

配置层不再出现 `backend`、`auto`、`lula_robot_description`、`lula_base_urdf`。

### `inverse_kinematics.py`

只负责逆运动学计算，不管理任务阶段，也不直接同步 Isaac 场景。它可以选择是否使用外部传入的 collision world，让 IK 解本身避开环境障碍。

职责：

- 将 `IKRequest` 转成 cuMotion target pose。
- 应用 warm start、seeds、容差和姿态权重。
- 返回统一 `IKResult`。
- 支持 sequential IK：上一点成功解作为下一点 warm start。
- 支持 batch/waypoint IK 的结果诊断。
- 支持 `avoid_collisions` 开关：关闭时只做几何 IK；开启时将 `collision_objects` 转成 cuMotion world，并要求输出构型满足碰撞约束。

不应该做：

- 读取 Isaac stage。
- 生成抓取阶段。
- 管理动态障碍物刷新。
- 做规划失败重采样策略。
- 输出控制器命令。

需要注意：obstacle-aware IK 只保证“求出的目标构型/离散 waypoint 构型”满足碰撞约束，不保证机器人从当前构型运动到该构型的连续路径无碰撞。需要连续路径级避障时仍然使用 `motion_planner.py`。

### `forward_kinematics.py`

封装正运动学和 frame/joint 查询，让外层不碰 cuMotion kinematics 对象。

职责：

- `compute_fk(joint_positions, frame_name)`。
- `joint_names()`、`frame_names()`。
- frame 存在性校验。
- 用于 IK 后验误差计算、日志和可视化。

### `motion_planner.py`

负责 cuMotion 路径规划和避障规划。

支持目标：

- 当前关节状态到目标关节状态。
- 当前关节状态到目标 TCP pose。
- 通过多个 TCP waypoint 的任务。
- collision-aware move。

输入输出：

```text
MotionRequest(current_q, goal_q/goal_pose, collision_world, constraints)
-> CuMotionMotionPlanner.plan()
-> MotionResult(joint_path, status, diagnostics)
```

### `collision_world.py`

负责把项目内部障碍物转换成 cuMotion world。任务层和 env 层不直接操作 cuMotion world 对象。

第一批支持：

- ground plane；
- table box；
- rope endpoint box；
- rope segment capsule/box 近似；
- 用户配置的 box、sphere、capsule、mesh。

项目内部数据结构：

```python
@dataclass(frozen=True)
class CollisionObject:
    name: str
    shape: str
    pose: np.ndarray
    size: tuple[float, ...]
    enabled: bool = True
    padding: float = 0.0
```

### `scene_sync.py`

从 Isaac stage 或 env 对象读取当前场景，生成 `CollisionObject` 列表。

职责：

- 读取 ground/table 的静态尺寸和 pose。
- 读取 rope endpoint 和 rope segment pose。
- 把抓取后携带物体转成 attached object 或临时障碍物。
- 每次规划前刷新 collision world。

这个文件只做“场景到项目数据”的同步，不调用 cuMotion planner。

### `trajectory_adapter.py`

负责将 cuMotion 规划结果转换成项目现有轨迹类型。项目 `JointTrajectory` 应按 cuMotion
`Trajectory` 语义存储 `times`、`positions`、`velocities`、`accelerations` 和 `jerks` 采样矩阵，
同时保留迭代 `TrajectoryPoint` 的执行接口。

职责：

- joint path 或 cuMotion `Trajectory.eval_all()` -> `JointTrajectory` 矩阵。
- planner 时间戳/速度/加速度/jerk -> `TrajectoryPoint` 兼容视图。
- 没有时间信息时调用 `planning/time_parameterization.py`。
- 保留 phase、diagnostics 和 joint name 顺序。

## planning 层

`planning/` 是任务和 cuMotion 后端之间的项目计算层。它不引入其它求解器，只提供稳定的数据结构和流程组织。

### 请求与结果

建议统一几类 request/result：

```python
@dataclass(frozen=True)
class IKRequest:
    target_position: np.ndarray
    target_orientation: np.ndarray | None
    tcp_frame_name: str
    warm_start: np.ndarray | None = None
    position_tolerance: float | None = None
    orientation_tolerance: float | None = None
    avoid_collisions: bool = False
    collision_objects: tuple[CollisionObject, ...] = ()

@dataclass(frozen=True)
class MotionRequest:
    current_q: np.ndarray
    goal_q: np.ndarray | None = None
    goal_pose: PoseTarget | None = None
    tcp_frame_name: str | None = None
    collision_objects: tuple[CollisionObject, ...] = ()
    mode: str = "collision_aware"

@dataclass(frozen=True)
class MotionResult:
    joint_path: np.ndarray | None
    trajectory: JointTrajectory | None
    success: bool
    status: str
    diagnostics: PlanningDiagnostics
```

### 规划策略

策略只决定“怎么组织 cuMotion 计算”，不切换后端。

- `ik_waypoints`：离散 TCP 点逐个 IK，适合当前 `pinch_grasp` 的 approach/grasp/lift/wiggle；可用 `avoid_collisions` 要求每个 waypoint 解避开障碍。
- `cartesian_ik`：笛卡尔线段采样后 sequential IK，适合直线接近、抬升、摆动；可用 `avoid_collisions` 做离散采样点碰撞约束。
- `collision_aware`：构建 collision world 后调用 cuMotion planner。
- `joint_move`：当前关节状态到目标关节状态。

示意：

```text
PlanningPipeline.plan(request)
  -> choose strategy by request.mode
  -> strategy builds cuMotion calls through CuMotionContext
  -> returns MotionResult
```

`mode` 是流程选择，不是 backend 选择。

## 任务层边界

任务层只负责业务逻辑：

- 计算目标点、抓取姿态和阶段顺序。
- 选择 TCP frame。
- 收集当前关节状态。
- 收集或请求 collision objects。
- 构造 `IKRequest` / `MotionRequest`。
- 执行 `MotionResult.trajectory`。
- 记录日志和可视化 marker。

任务层不应该：

- import `cumotion`。
- 自己创建 cuMotion world。
- 包含 Lula/auto/backend fallback 逻辑。
- 把 IK seeds、planner retry、collision object 转换写死在任务脚本中。

## 配置设计

机器人配置只描述 cuMotion 机器人资源和默认解算参数：

```yaml
motion:
  robot_description: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  base_urdf: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
  flange_frame: AR5V2_L_arm_flan_link
  default_tcp_frame: pinch_tcp
  solver:
    position_tolerance: 0.005
    orientation_tolerance: 0.75
    ccd_max_iterations: 180
    bfgs_max_iterations: 80
    orientation_weight: 0.25
    cspace_seeds:
      - [-1.57, 0.8, 0.0, 0.8, 0.0, 0.0, 0.0]
```

任务配置只描述任务策略和覆盖项：

```yaml
planning:
  mode: ik_waypoints
  tcp_frame: pinch_tcp
  max_attempts: 5
  position_tolerance: 0.003
  orientation_tolerance: 0.5
  avoid_collisions: true
  allow_partial_success: false
```

环境配置描述碰撞世界来源：

```yaml
collision_world:
  include_ground: true
  include_table: true
  include_rope: true
  rope_as: capsules
  padding: 0.01
```

命令行参数也应去掉后端选择。推荐只保留规划模式覆盖：

```text
--planning-mode ik_waypoints|cartesian_ik|collision_aware|joint_move
```

## 需要删除的 Lula 痕迹

代码：

- `source/manipulation_project/ik/lula_solver.py`
- `make_ik_solver()` 的 `lula` 和 `auto` 分支
- `is_cumotion_available()` 驱动 fallback 的逻辑
- `PinchGraspTask` 中 `lula_robot_description` / `lula_base_urdf` 参数
- scripts 中 `--ik-backend` 的 `lula` / `auto` choices

配置：

- `configs/robots/*` 中的 `lula:` 段
- 旧的 `lula_robot_description` / `lula_base_urdf` 键
- README 和文档中“保留 Lula 兼容后端”的描述

资产：

- 若确认不再使用，可归档或删除 `*_lula_robot_description.yaml`。
- 如果暂时保留文件用于历史追溯，代码和配置也不应再引用它。

测试：

- 删除 Lula fallback 测试。
- 新增 cuMotion 缺失时 fail-fast 的测试。
- 新增配置中出现 `lula` 键时报错或测试失败。

## 迁移步骤

### 第一阶段：断开 Lula

- 删除 `lula_solver.py`。
- 修改 solver 创建入口：只允许 cuMotion。
- 删除 `backend="auto"` 语义；cuMotion 不可用就报错。
- 清理 `configs/robots/*` 的 `lula:` 段。
- 清理脚本参数和 README。

### 第二阶段：移动 cuMotion IK/FK 到后端层

- 新建 `source/manipulation_project/backends/cumotion/`。
- 将 `ik/cumotion_solver.py` 移到 `backends/cumotion/inverse_kinematics.py`。
- 新增 `backends/cumotion/forward_kinematics.py`，承载 FK、frame 查询和 IK 后验误差计算。
- 新增 `config.py` 和 `context.py`。
- 让任务通过 `PlanningPipeline` 或 `CuMotionContext` 创建逆运动学/正运动学组件。

### 第三阶段：重建 planning 数据结构

- 新建 `planning/requests.py`、`planning/results.py`、`planning/targets.py`。
- 将现有 `IKRequest` / `IKResult` 迁移进去。
- 定义 `MotionRequest` / `MotionResult`。
- 保持 `trajectories/base.py` 作为执行轨迹容器。

### 第四阶段：拆出抓取规划策略

- 将 `pinch_grasp.py` 中的 IK waypoint 求解流程抽到 `planning/strategies.py`。
- `pinch_grasp.py` 只生成 approach/grasp/lift/wiggle 目标和执行阶段。
- 支持 `ik_waypoints` 和 `cartesian_ik` 两种 IK 策略，并允许通过 `avoid_collisions` 开关启用离散构型避障。

### 第五阶段：加入 collision world

- 新增 `planning/collision_objects.py`。
- 新增 `backends/cumotion/collision_world.py`。
- 从 ground/table/rope 配置生成静态或准静态障碍物。
- 提供独立 demo：`scripts/run_collision_aware_move.py`。

### 第六阶段：接入 cuMotion motion planner

- 新增 `backends/cumotion/motion_planner.py`。
- 支持 `goal_q`、`goal_pose` 和 waypoint request。
- 输出 `MotionResult`，再由 `trajectory_adapter.py` 转成 `JointTrajectory`。
- 日志记录规划耗时、失败原因、最终误差和最小障碍物距离。

### 第七阶段：动态场景同步

- 新增 `scene_sync.py`。
- 每次规划前从 Isaac stage/env 刷新 rope endpoint、rope segment 和 attached object。
- 抓取后把被携带对象纳入规划约束。

## 验证计划

- 配置测试：内置 robot config 不允许出现 `lula`、`auto`、legacy IK key。
- 导入测试：外层 `planning/`、`tasks/` 不直接 import `cumotion`。
- cuMotion 环境测试：能加载 XRDF + URDF，能列出 joints 和 frames。
- IK smoke：固定 TCP pose 得到成功解，并校验 FK 误差。
- obstacle-aware IK 测试：同一目标在障碍物开关关闭/开启时有可解释的成功或失败结果。
- sequential IK 测试：笛卡尔采样点连续求解，关节跳变低于阈值。
- collision world 测试：ground/table/rope object 能转换成 cuMotion world。
- planner smoke：从当前 q 到目标 pose 生成非空 joint path。
- headless 任务测试：`pinch_grasp` 通过 `PlanningPipeline` 生成可执行轨迹。

## 禁止事项

- 不再新增任何 Lula 相关代码。
- 不再使用 `backend="auto"`。
- 不在任务层 import `cumotion`。
- 不在 `CuMotionInverseKinematics.solve()` 中塞场景同步、动态障碍物刷新和任务重试逻辑；它只消费外部传入的 collision world。
- 不把 obstacle-aware IK 当成完整路径避障；连续路径避障必须走 `motion_planner.py`。
- 不让 `pinch_grasp.py` 继续承载通用 planner 能力。
- 不把 Isaac stage prim 遍历、collision object 转换和 cuMotion planner 调用写在同一个函数里。
- 不假设所有任务使用同一个 TCP；TCP 必须来自任务请求或配置。

## 结论

项目后续的运动计算应收敛成一条清晰链路：

```text
Task target
-> Planning request
-> Planning strategy
-> CuMotionContext
-> cuMotion inverse kinematics / forward kinematics / planner / collision world
-> MotionResult
-> JointTrajectory
-> controller execution
```

Lula 不再是兼容层，也不再是 fallback。cuMotion 是唯一解算核心；文件架构围绕 cuMotion 做资源复用、规划能力扩展和任务层解耦。
