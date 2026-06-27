# LinkerHand Simulation

这是一个基于 Isaac Sim / Isaac Lab 的机械臂、灵巧手和绳体操作仿真工程。当前主线围绕
AR5 机械臂、LinkerHand L6 灵巧手、capsule/cuboid 近似绳体，以及 cuMotion 后端运动生成
展开，用于验证 TCP 定义、IK/FK、路径规划、mimic 关节同步、PhysX 参数和 scripted pinch
grasp 流程。

项目的核心原则是分层清楚：

- `configs/` 描述资产、控制器、环境、对象、日志和 cuMotion profile。
- `scripts/pinch_grasp.py` 是自包含动作入口，抓取目标、阶段时长和手型直接写在脚本中。
- `planning/` 提供后端无关请求和结果数据结构。
- `backends/cumotion/` 是唯一直接适配 cuMotion Python API 的层。
- `trajectories/` 只保存项目内部关节轨迹容器和从位置矩阵构造导数的工具。
- `execution/` 只按物理步播放已经生成好的目标或轨迹。
- `controllers/` 把项目目标转换成 Isaac articulation action，并负责 mimic follower 下发。

## 当前能力

- 资产导入：支持 AR5、L6、AR5+L6 组合 MJCF/URDF/XRDF/USD 资产。
- 绳体对象：支持由配置生成 capsule/cuboid 刚体链 USD，并在场景中引用。
- cuMotion 后端：提供 FK、IK、collision-free IK、trajectory optimization、graph search、specified path、task-space path conversion 和 C-space trajectory generation。
- TCP：支持法兰 TCP、已有工具 TCP、临时 fixed TCP，以及基于闭合手型计算的 thumb/index 夹捏中心 TCP。
- 轨迹语义：cuMotion `Trajectory` 会采样成项目 `JointTrajectory`；graph search 和 specified path 成功时必须生成 trajectory，不能只返回离散 path。
- 控制器：支持 position、velocity 和 effort 控制；position/velocity 可选 Isaac implicit drive 或 Python explicit effort。
- Mimic 关节：解析 MJCF `equality/joint` 的 `polycoef`，运行时按实际 master 状态刷新 follower 目标。
- 日志：支持关节目标、实际位置/速度、command effort、action effort、PhysX measured/applied effort 的 CSV 记录。
- 可视化/遥测：提供 Isaac viewport 辅助、debug draw、marker，以及可选 Foxglove MCAP/WebSocket 输出。

## 目录结构

```text
.
├── assets/
│   ├── mesh/                 # arm/hand/env object mesh
│   ├── single_system/        # 单体机器人资产，例如 AR5V2_L、L6V1_L
│   ├── combined_system/      # 组合机器人资产，例如 AR5V2_L6V1_L
│   ├── static_env_objects/   # 静态环境对象
│   └── dynamic_env_objects/  # 动态对象资产，例如 capsuleropeV1_default
├── configs/
│   ├── controllers/          # arm/hand 控制模式、增益、摩擦和 PhysX 覆盖
│   ├── cumotion/             # cuMotion 默认 profile 和详细参数示例
│   ├── envs/                 # empty/table/rope 场景、物理步频、solver
│   ├── logging/              # 关节跟踪日志配置
│   ├── objects/              # 可生成对象配置，例如 capsule rope
│   └── robots/               # 机器人资产、关节组、cuMotion XRDF/URDF
├── docs/                     # cuMotion 接口、规划设计和历史方案文档
├── scripts/                  # Isaac Sim 运行入口和资产生成入口
├── source/manipulation_project/
│   ├── app/                  # SimulationApp 启动
│   ├── assets/               # 资产导入、USD/PhysX 覆盖、solver 设置
│   ├── backends/cumotion/    # cuMotion context、FK/IK、planner、path/trajectory adapter
│   ├── controllers/          # 控制器配置解析和 runtime controller
│   ├── envs/                 # World 和场景构建
│   ├── execution/            # 目标/轨迹执行步骤
│   ├── logging/              # CSV 日志
│   ├── objects/              # 绳体对象资产生成和引用
│   ├── planning/             # 后端无关请求、结果、碰撞对象
│   ├── robots/               # 关节组、mimic/equality、状态工具
│   ├── tcp/                  # TCP frame 和夹捏中心计算
│   ├── telemetry/            # Foxglove、MCAP、WebSocket
│   ├── trajectories/         # JointTrajectory 容器和 builder
│   ├── utils/                # 配置、路径、旋转、数学、计时工具
│   └── visualization/        # camera、debug draw、marker
├── tests/                    # 不启动 Isaac Sim 的轻量测试
├── ASSET_NAMING_CONVENTIONS.md
├── pyproject.toml
└── README.md
```

## 环境约定

示例命令默认使用仓库内 `env_isaaclab/` Python 环境，并从仓库根目录运行：

```bash
PYTHONPATH=source env_isaaclab/bin/python <command>
```

完整依赖通过 `pyproject.toml` 管理；Isaac Sim、cuMotion、torch 版本应与当前 Isaac Python
环境匹配。`simulation` extra 记录了当前代码期望的主要运行依赖：

- `isaacsim[all]==5.1.0.0`
- `cumotion==1.1.0`
- `torch==2.7.0`

安装项目依赖示例：

```bash
env_isaaclab/bin/python -m pip install -e ".[dev,visualization,simulation]"
```

如果环境已经由 Isaac Lab 管理，也可以只安装缺失的 Python 包。cuMotion/Isaac 导入失败时，
后端会在创建实际 context 或启动脚本时直接报错；纯配置解析和大部分单元测试不需要启动 Isaac。

## 快速运行

生成或更新 capsule rope USD：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
```

只导入绳体和 AR5+L6，并快速保持初始姿态：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/pinch_grasp.py --no-grasp --short-smoke
```

运行 GUI pinch grasp demo：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/pinch_grasp.py --gui
```

常用覆盖参数：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/pinch_grasp.py \
  --endpoint right \
  --control-mode position \
  --physics-frequency 240 \
  --gui
```

日志默认写入 `logs/joint_tracking/`。如果只想快速验证导入、控制器和日志，可以使用
`--no-grasp --short-smoke`；如果希望最终姿态持续到窗口关闭，可以增加 `--hold --gui`。

## 主流程

`scripts/pinch_grasp.py` 是当前最完整的运行入口。典型执行顺序如下：

1. 加载 cuMotion profile、robot、controller、env、rope 和 logging YAML。
2. 将 `configs/cumotion/default.yaml` 的默认值合入 robot 配置，并把 `cumotion.motion_planner` 作为脚本内置动作的 planner 默认值。
3. 启动 Isaac `SimulationApp`，创建 World、重力、physics/render dt。
4. 引用已生成的 capsule rope USD。
5. 导入 AR5+L6 组合 MJCF，并应用 drive、摩擦、solver iteration、重力等 runtime 覆盖。
6. 创建 `JointController`，按 `--control-mode` 选择 position、velocity 或 effort 主动控制。
7. 根据闭合手型和 MJCF body 链计算 pinch TCP，并通过临时 URDF 创建 cuMotion context。
8. 使用 cuMotion 求解 approach、grasp、lift、wiggle 等阶段的 IK 和运动轨迹。
9. 把 cuMotion C-space 结果按关节名映射回 Isaac 完整 DOF。
10. `execution.steps` 按 physics dt 播放完整 DOF 轨迹或 smooth target，并写 CSV 日志。

动作脚本不会在执行过程中实时做 task-space conversion。所有 IK、指定路径转换和 trajectory
generation 都在规划阶段完成；执行阶段只按时间采样已经生成的 `JointTrajectory`。

## 配置系统

配置文件使用固定顶层结构。常见入口如下：

- `configs/robots/ar5v2_l6v1_l.yaml`：机器人资产、主动关节、cuMotion URDF/XRDF/frame。
- `configs/controllers/arm_controller.yaml` 和 `configs/controllers/hand_controller.yaml`：主动关节和 follower 的控制参数。
- `configs/envs/rope_scene.yaml`：物理步频、渲染步频、重力、solver iteration。
- `configs/objects/capsule_rope.yaml`：绳体几何、质量、材质、D6 joint、USD 输出路径。
- `configs/cumotion/default.yaml`：项目默认 cuMotion profile。
- `configs/cumotion/default.example.yaml`：详细字段说明，不作为默认运行配置。
- `scripts/pinch_grasp.py`：自包含 pinch grasp 动作参数和执行流程。

cuMotion 相关配置的优先级是：

```text
configs/cumotion/default.yaml
  < configs/robots/*.yaml
  < scripts/pinch_grasp.py 内置动作参数
```

实际合并分两条线：

- `cumotion` profile 先作为 robot config 的默认值；robot YAML 继续提供或覆盖 `xrdf_path`、`urdf_path`、`flange_frame`、`kinematics` 等机器人级字段。
- `cumotion.motion_planner` profile 作为 `PinchGraspActionConfig.motion_planning` 的默认值；抓取目标、阶段时长和手型固定在 `scripts/pinch_grasp.py` 中。

配置合并是递归 mapping merge。列表和标量按覆盖值整体替换，不做逐项合并，这对关节列表、
轨迹点和手型目标更安全。

### cuMotion 配置

`CuMotionConfig` 顶层只保存机器人描述和 frame：

```yaml
cumotion:
  xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
  flange_frame: AR5V2_L_arm_flan_link
  custom_tcp_frame: null
```

IK/FK 参数放在 `kinematics` 下：

```yaml
cumotion:
  kinematics:
    ik:
      cspace_seeds: null
      position_tolerance: 0.002
      orientation_tolerance: 0.01
      ccd_max_iterations: 180
      bfgs_max_iterations: 80
      orientation_weight: 0.25
      collision_free_params: {}
    fk: {}
```

路径级规划参数放在 `motion_planner` 下：

```yaml
cumotion:
  motion_planner:
    planning_pipeline: trajectory_optimization
    trajectory_optimization:
      use_environment_obstacles: true
      params: {}
    graph_search:
      generate_interpolated_path: false
      use_environment_obstacles: true
      motion_planner_config_path: null
      motion_planner_params: {}
    trajectory_generation:
      mode: time_optimal
      interpolation_mode: cubic_spline
      limits: {}
      solver_params: {}
    specified_path:
      family: task_space_segments
      validate_collision_after_generation: false
      cspace_waypoints: {}
      task_space_segments: {}
      composite: {}
```

`trajectory_generation.enabled` 已被移除。graph search 和 specified path 只要成功，就必须生成
cuMotion `Trajectory`；如果只得到离散 path 而无法生成 trajectory，应视为规划失败并调整配置
或 pipeline。

更完整的字段说明见 `configs/cumotion/default.example.yaml`。

## cuMotion 后端

`source/manipulation_project/backends/cumotion/` 是 cuMotion 的适配层。主要文件职责如下：

- `context.py`：加载 XRDF/URDF，创建 robot description、kinematics、collision world，并暴露 `joint_names()` / `frame_names()`。
- `forward_kinematics.py`：封装 FK 和 frame 查询。
- `inverse_kinematics.py`：封装单点 IK 和 collision-free IK。
- `motion_planner.py`：按 `MotionPlannerBackendConfig.planning_pipeline` 分发到具体 pipeline。
- `trajectory_optimizer_planner.py`：调用 cuMotion `TrajectoryOptimizer`，成功时 `trajectory` 是主产物，`path` 通常为 `None`。
- `graph_motion_planner.py`：调用 graph `MotionPlanner` 得到 C-space path，再强制生成 trajectory。
- `specified_path_planner.py`：消费调用方指定的 C-space/task-space/composite path，转换成 C-space path 后强制生成 trajectory。
- `path_spec_adapter.py`：把项目 `TaskSpacePath`、`CSpaceWaypointPath`、`CompositePath` 转为 cuMotion 官方 PathSpec。
- `trajectory_generation.py`：封装 `CSpaceTrajectoryGenerator`，对 joint path 做时间参数化。
- `trajectory_sampler.py`：把 cuMotion `Trajectory.eval_all(t)` 采样成项目 `JointTrajectory`。
- `tcp_context.py`：按需写临时 URDF，把调用方生成的 TCP 作为 fixed link 装进 cuMotion context。
- `tcp_urdf_builder.py`：纯 URDF 写入工具，动作脚本不直接管理临时文件。

### MotionResult 语义

所有路径级规划统一返回 `planning.results.MotionResult`：

```python
MotionResult(
    path=<np.ndarray | None>,
    trajectory=<cuMotion Trajectory | None>,
    success=<bool>,
    status=<str>,
    diagnostics=<PlanningDiagnostics>,
)
```

`path` 是离散 C-space waypoint 矩阵，主要用于诊断、路径长度、末端构型和动作脚本回填末点。
`trajectory` 是带时间参数化的后端轨迹对象，才是执行前需要采样的主数据。

不同 pipeline 的输出约定：

- `trajectory_optimization`：成功时主要返回 cuMotion `Trajectory`；`path` 通常保持 `None`。
- `graph_search`：先得到离散 C-space path，再生成 trajectory；成功结果必须两者都有。
- `specified_path`：先把调用方指定路径转换成 C-space path，再生成 trajectory；成功结果必须两者都有。

这些数据都保持 cuMotion C-space 关节顺序，不会自动扩展成 Isaac 完整 DOF。动作脚本必须使用
`motion_planner.joint_names()` 和 Isaac `robot.dof_names` 做名称映射。

## Trajectory 和执行频率

项目内部 `JointTrajectory` 位于 `source/manipulation_project/trajectories/types.py`。它保存：

- `times`: shape `(N,)`
- `positions`: shape `(N, dof)`
- `velocities`: shape `(N, dof)`
- `accelerations`: shape `(N, dof)`
- `jerks`: shape `(N, dof)`
- `efforts`: shape `(N, dof)`
- `phases`: 每个采样点所属阶段
- `joint_names`: 每列对应的关节名

cuMotion 轨迹进入项目执行层通常经历三步：

1. cuMotion pipeline 返回后端 `Trajectory`。
2. `trajectory_sampler.joint_trajectory_from_cumotion(...)` 按 `sample_dt` 或显式 `times` 采样。
3. 动作脚本把 C-space 列回填到完整 Isaac articulation DOF，并构造完整 DOF `JointTrajectory`。

如果轨迹采样频率和 physics step 频率不同，执行层以 physics dt 为准。`execute_full_joint_trajectory(...)`
会在每个物理步调用 `trajectory.eval_all(time_s)` 做线性插值，然后下发该时刻的位置、速度和
effort。换句话说，轨迹不是按原采样点逐点硬播放，而是按仿真时钟求值。

`trajectories/joint_trajectory_builder.py` 只用于从已有位置矩阵构造 `JointTrajectory`，并通过有限
差分补速度、加速度和 jerk。它不负责 cuMotion path conversion，也不负责运动规划。

## Pinch Grasp

`scripts/pinch_grasp.py` 是当前主要动作入口。它的输入是机器人 articulation、rope 配置、MJCF
路径、cuMotion config 和脚本内置的 `PinchGraspActionConfig`。主要步骤：

1. 展开闭合手型中的 MJCF mimic follower 目标。
2. 沿 thumb/index body 链计算闭合手型下两指尖位置。
3. 取两指尖几何中点作为 `pinch_tcp`，并挂到 `flange_frame` 下。
4. 通过 `make_cumotion_context(...)` 写临时 URDF，让 cuMotion 能识别该 TCP frame。
5. 计算 approach、grasp、lift、wiggle 的 TCP 目标。
6. 对 approach、lift、wiggle 等阶段求 IK 和关节空间规划。
7. 对接近端块的短距离下沉使用 specified path `TcpLineSegment`，规划期先做 task-space conversion。
8. 把机械臂 C-space 结果映射回完整 DOF，手部目标用脚本内置稀疏关节目标覆盖。
9. 生成 `SmoothJointTargetStep`、`FullJointTrajectoryStep`、`HoldJointTargetStep` 组成的执行序列。

动作参数集中在 `PinchGraspActionConfig`、`DEFAULT_PRE_PINCH_HAND_TARGETS` 和
`DEFAULT_CLOSED_PINCH_HAND_TARGETS`。`target_rpy` 使用固定轴 XYZ 顺序，单位 rad；脚本内部会
转换成项目统一的 wxyz 四元数。

## 控制器和 Mimic

`JointController` 的输入目标有两类：

- 完整 DOF 目标：长度等于 Isaac `articulation.num_dof`，顺序是 `robot.dof_names`。
- command-space 目标：只包含主动命令关节，不包含 mimic follower。

无论使用哪种入口，controller 都会在下发前刷新 follower：

```text
follower_target = polycoef(actual_master_position)
follower_velocity = d(polycoef)/dq * actual_master_velocity
```

这里故意使用实际 master 状态，而不是目标 master 状态。这样主动关节还没有跟上命令时，
follower 不会提前闭合或超前运动，能减少接触抖动和 mimic 误差。

主动关节支持三种模式：

- `position + implicit`：下发 position/velocity target，由 PhysX drive 计算力矩。
- `position + explicit`：Python 侧按位置/速度误差计算 effort。
- `velocity + implicit`：下发 velocity target，由 PhysX velocity drive 计算力矩。
- `velocity + explicit`：Python 侧按速度误差计算 effort。
- `effort + direct`：直接下发 effort command。

Follower 始终使用 Isaac position drive，不随主动控制模式改变。

## 绳体对象

绳体配置在 `configs/objects/capsule_rope.yaml`，核心结构如下：

```yaml
object:
  name: capsuleropeV1_default
  asset_path: assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
  prim_path: /World/CapsuleRope
  root_path: /CapsuleRope

rope:
  segments: 18
  length: 0.75
  radius: null
  center: [0.4, -0.55, 0.05]
  total_mass: 0.2
  shape: capsule
```

修改绳体几何、质量、材质或 joint 参数后，需要重新生成 USD：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
```

生成脚本会启动 headless SimulationApp，因为 USD/PhysX schema 写入依赖 Isaac/Omni 扩展注册。
运行 pinch grasp 时只是引用已经生成好的 USD，不会每次重新生成绳体资产。

## 坐标、单位和命名

- 距离单位为 m，质量单位为 kg，时间单位为 s。
- 所有关节角和 YAML 中的 RPY 使用 rad。
- 项目边界统一使用 wxyz 四元数，即 `[w, x, y, z]`。
- SciPy 内部使用 xyzw，转换封装在 `utils/rotations.py`。
- 配置中的 RPY 使用固定轴 XYZ 顺序。
- cuMotion C-space 顺序由 XRDF/URDF 决定，不能假设等于 Isaac articulation DOF 顺序。
- 正式资产、关节、link/body、配置引用不使用连字符 `-`，避免 Isaac importer 自动改名。

资产命名细节见 `ASSET_NAMING_CONVENTIONS.md`。

## 日志和调试

关节跟踪日志使用 `configs/logging/default_logger.yaml`，默认输出到 `logs/joint_tracking/`。
常用列名：

- `qd_*` / `q_*`：命令位置和实际位置。
- `vd_*` / `v_*`：命令速度和实际速度。
- `tau_cmd_*`：语义 effort command；implicit drive 下通常为 `nan`。
- `tau_action_*`：控制器实际下发给 Isaac 的 effort action。
- `tau_measured_*`：PhysX measured effort。
- `tau_applied_*`：Isaac applied effort。

默认不记录读取成本较高的 effort 字段。可以通过命令行打开：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/pinch_grasp.py \
  --log-measured-effort \
  --log-applied-effort \
  --log-action-effort
```

Foxglove 输出位于 `source/manipulation_project/telemetry/foxglove.py`，支持离线 MCAP、本地
WebSocket live server、`JointStates` 曲线和 `SceneUpdate` marker。

## 验证

语法检查：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m compileall -q source scripts tests
```

运行轻量测试：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m pytest -q tests
```

检查 YAML 是否可解析：

```bash
PYTHONPATH=source env_isaaclab/bin/python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path("configs").rglob("*.yaml")):
    yaml.safe_load(path.read_text(encoding="utf-8"))
print("yaml ok")
PY
```

提交前建议额外运行：

```bash
git diff --check
```

当前轻量测试重点覆盖 controller 配置、关节控制、mimic 解析、TCP frame、cuMotion context、
IK、motion planner、trajectory adapter、JointTrajectory builder、配置加载、pinch grasp
规划逻辑、日志和 Foxglove logger。它们不替代 Isaac GUI/物理接触验证，但可以快速发现配置、
数据结构和后端适配层的回归。
