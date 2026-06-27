# LinkerHand Simulation

这是一个基于 Isaac Sim / Isaac Lab 的机械臂、灵巧手和绳体操作仿真工程。项目当前围绕 AR5 机械臂、LinkerHand L6 灵巧手和 capsule/box 近似绳体搭建，用于验证 cuMotion 运动解算、TCP 定义、mimic 关节同步、PhysX 参数和基础抓取流程。

## 当前能力

- 资产导入：支持 AR5、L6、AR5+L6 组合 MJCF/URDF/XRDF/USD 资产。
- 运动解算：统一使用 cuMotion，当前提供 FK、几何 IK、obstacle-aware IK、路径级 MotionPlanner、碰撞世界适配和 cuMotion 轨迹适配。
- 控制器：支持位置、速度和 effort 控制；位置/速度可选 Isaac implicit drive 或 Python 显式 effort 计算。
- Mimic 关节：解析 MJCF `equality/joint` 的 `polycoef` 多项式关系，并在软件层同步 follower drive 目标。
- TCP：支持法兰 TCP、自定义固定 TCP、thumb/index 闭合夹捏中心 TCP。
- 任务脚本：提供关节目标 smoke、TCP 笛卡尔直线运动和 AR5+L6 绳端夹捏抓取 demo。
- 日志和可视化：支持关节跟踪 CSV，Foxglove MCAP/WebSocket 可选用于调试回放。

## 目录结构

```text
.
├── assets/
│   ├── mesh/                 # 设备 mesh，按 arm/hand 和系统名分目录
│   ├── single_system/        # 单体 URDF/MJCF/XRDF/USD 资产
│   ├── combined_system/      # 复合系统资产
│   ├── static_env_objects/   # 静态环境对象资产
│   └── dynamic_env_objects/  # 动态对象资产，例如 capsule rope
├── configs/
│   ├── controllers/          # arm/hand 控制、材料、刚体参数
│   ├── envs/                 # empty/table/rope 场景
│   ├── logging/              # CSV 日志配置
│   ├── objects/              # 可生成对象配置，例如 capsule rope
│   ├── robots/               # 机器人资产、关节组、cuMotion XRDF/URDF
│   └── trajectories/         # joint target 和 pinch grasp 配置
├── scripts/                  # Isaac Sim 运行入口
├── source/manipulation_project/
│   ├── app/                  # SimulationApp 启动
│   ├── assets/               # 资产导入、USD/PhysX 覆盖、solver 设置
│   ├── backends/cumotion/    # cuMotion 后端、FK/IK、碰撞世界、轨迹适配
│   ├── controllers/          # 控制器配置解析和 runtime controller
│   ├── envs/                 # World 和场景构建
│   ├── logging/              # CSV 和关节跟踪日志
│   ├── objects/              # 绳体对象资产生成和引用
│   ├── planning/             # 解算请求、结果、碰撞对象数据结构
│   ├── robots/               # 关节组、mimic/equality、状态容器
│   ├── tasks/                # 关节目标和 pinch grasp 任务流程
│   ├── telemetry/            # Foxglove、MCAP、WebSocket 等外部遥测输出
│   ├── tcp/                  # TCP frame 和夹捏中心计算
│   ├── trajectories/         # 矩阵轨迹容器和插值
│   ├── utils/                # 配置、路径、旋转、数学、计时工具
│   └── visualization/        # Isaac viewport、debug draw 和本地 marker
├── tests/                    # 不启动 Isaac Sim 的轻量测试
├── ASSET_NAMING_CONVENTIONS.md
├── docs/
│   ├── cumotion_interface.md
│   ├── motion_planner_design.md
│   └── specified_path_final_plan.md
├── pyproject.toml
└── README.md
```

## 环境约定

- 使用 Isaac Sim / Isaac Lab 对应的 Python 环境运行脚本，示例默认环境目录为 `env_isaaclab/`。
- 从仓库根目录运行脚本，并设置 `PYTHONPATH=source`。
- 依赖统一声明在 `pyproject.toml`；`requirements.txt` 安装项目的完整依赖集合。
- cuMotion、Isaac Sim 和 torch 版本与当前 Isaac Python 环境对齐；后端导入失败时会直接报安装错误。

安装完整依赖：

```bash
env_isaaclab/bin/python -m pip install -r requirements.txt
```

## 快速运行

生成或更新 capsule rope USD 资产：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
```

导入 AR5+L6 和绳体并保持初始姿态：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_pinch_grasp.py --no-grasp --short-smoke
```

运行绳端夹捏抓取 demo：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_pinch_grasp.py --gui
```

常用覆盖参数：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_pinch_grasp.py \
  --endpoint right \
  --physics-frequency 240 \
  --gui
```

运行日志默认写入 `logs/joint_tracking/`。

## 架构边界

当前运动计算链路如下：

```text
config/script
-> task
-> planning request/result
-> backends/cumotion
-> cuMotion
-> trajectory/controller/logging
```

`tasks/` 描述任务目标和执行阶段，例如关节目标或 pinch grasp。`planning/` 提供后端无关的数据结构，例如 `IKRequest`、`MotionRequest`、`IKResult`、`CollisionObject`。`backends/cumotion/` 是唯一直接调用 cuMotion Python API 的层，负责机器人模型加载、FK、IK、碰撞世界和轨迹适配。

### cuMotion 后端

核心入口位于 `source/manipulation_project/backends/cumotion/`：

- `context.py`：`CuMotionConfig` 和 `CuMotionContext`，加载 XRDF + URDF，缓存 robot description 和 kinematics。
- `forward_kinematics.py`：封装 FK、frame 查询和关节名查询。
- `inverse_kinematics.py`：封装单点 IK；`IKRequest.avoid_collisions=True` 时使用 cuMotion collision-free IK。
- `motion_planner.py`：封装路径级 `MotionRequest`；支持关节目标、TCP translation/pose 目标，并通过 cuMotion `MotionPlanner` 做连续路径级避障。
- `collision_world.py`：把项目 `CollisionObject` 转成 cuMotion `World` obstacle。
- `pose_adapter.py`：在项目 numpy pose/quaternion 和 cuMotion pose 类型之间转换。
- `tcp_urdf_builder.py`：复制 URDF 并临时追加 fixed TCP link/joint。
- `trajectory_adapter.py`：将 cuMotion `Trajectory.eval_all()` 采样成项目 `JointTrajectory`。

`CuMotionConfig` 的主要字段按模块分组：机器人描述和 frame 保持在顶层，FK/IK 参数放在
`kinematics`，路径级 planner 参数放在 `motion_planner`。

```python
from manipulation_project.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionIkConfig,
    CuMotionKinematicsConfig,
)

CuMotionConfig(
    xrdf_path="assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf",
    urdf_path="assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
    flange_frame="AR5V2_L_arm_flan_link",
    custom_tcp_frame="pinch_tcp",
    kinematics=CuMotionKinematicsConfig(
        ik=CuMotionIkConfig(
            position_tolerance=0.005,
            orientation_tolerance=0.75,
            ccd_max_iterations=180,
            bfgs_max_iterations=80,
            orientation_weight=0.25,
        )
    ),
)
```

### Planning 数据结构

`source/manipulation_project/planning/` 当前提供稳定的数据结构：

- `IKRequest`：单个 TCP 目标，包含位置、可选姿态、warm start、容差、碰撞开关和障碍物。
- `MotionRequest`：路径级请求结构，包含当前关节、目标关节或目标 TCP pose、碰撞对象和模式字段。
- `IKResult` / `MotionResult`：保存求解输出、成功状态、误差、状态和诊断。
- `CollisionObject`：后端无关障碍物描述，支持 `box/cuboid`、`sphere`、`capsule` 的尺寸和 padding。

obstacle-aware IK 约束的是“求出的目标构型”或离散 waypoint 构型；`CuMotionMotionPlanner` 则通过 cuMotion `MotionPlanner` 为 `MotionRequest` 提供连续路径级避障。`MotionRequest.mode` 默认使用碰撞感知规划，传入 `geometric`/`collision_unaware` 等模式时会忽略环境障碍物。

### TCP

TCP 相关代码位于 `source/manipulation_project/tcp/`：

- `flange_tcp.py`：直接使用法兰 link 作为 TCP。
- `custom_tcp.py`：在任意父 frame 下定义固定 `xyz/rpy` 偏移。
- `pinch_tcp.py`：读取 MJCF body 链，在闭合手型下计算 thumb tip 和 index tip 中点。

Pinch grasp 会先根据闭合手型计算夹捏中心，再通过 `backends/cumotion/tcp_context.py`
装配带临时 TCP URDF 的 `CuMotionContext`。底层 `tcp_urdf_builder.py` 仍负责纯 URDF 写入，
但任务层不直接管理临时文件。

### Trajectory

`JointTrajectory` 位于 `source/manipulation_project/trajectories/base.py`，内部按 cuMotion 轨迹语义保存矩阵：

- `times`: shape `(N,)`
- `positions`: shape `(N, dof)`
- `velocities`: shape `(N, dof)`
- `accelerations`: shape `(N, dof)`
- `jerks`: shape `(N, dof)`
- `phases`: 每个采样行的阶段名

关节目标轨迹由 `trajectories/joint_trajectory.py` 生成，支持 `linear`、`smoothstep` 和 `smootherstep` 插值。控制器执行时直接读取矩阵采样行。

## 配置

项目配置采用固定顶层结构。解析函数会要求当前结构存在，例如 robot 配置必须包含 `robot`，rope 配置必须包含 `object` 和 `rope`，抓取配置必须包含 `grasp`，轨迹配置必须包含 `trajectory`，环境配置必须包含 `env`。

### 机器人

`configs/robots/ar5v2_l6v1_l.yaml` 描述组合资产、控制关节和 cuMotion 机器人资源：

```yaml
robot:
  name: ar5v2_l6v1_l
  asset_type: mjcf
  asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
  prim_path: /World/AR5V2_L6V1_L

controlled_joints:
  - AR5V2_L_arm_joint_1
  - AR5V2_L_arm_joint_2

cumotion:
  xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
  flange_frame: AR5V2_L_arm_flan_link

tcp:
  type: flange
  frame_name: AR5V2_L_arm_flan_link
```

### cuMotion profile

`configs/cumotion/` 提供后端通用 profile 和带注释的配置示例：

```text
configs/cumotion/
├── default.yaml
└── example.yaml
```

`scripts/run_pinch_grasp.py` 默认加载 `configs/cumotion/default.yaml`。`example.yaml` 不作为
默认运行配置，而是详细说明每个字段可以怎么写、缺省时走什么行为。加载顺序为：

```text
configs/cumotion/default.yaml
  < configs/robots/*.yaml
  < configs/trajectories/*.yaml
```

其中 profile 的 `cumotion` 段作为机器人级后端默认值，robot YAML 继续负责覆盖
`xrdf_path`、`urdf_path` 和 `flange_frame`；profile 的 `cumotion.motion_planner` 段作为
planner 默认值，运行 pinch grasp 时会映射为 `grasp.motion_planning` 的默认值。trajectory
YAML 的 `grasp.motion_planning` 继续拥有最高优先级。

### 控制器

`configs/controllers/arm_controller.yaml` 和 `configs/controllers/hand_controller.yaml` 分别配置：

- `position_control.method`
- `position_control.active_joints`
- `position_control.follower_joints`
- `velocity_control.method`
- `velocity_control.active_joints`
- `velocity_control.follower_joints`
- `effort_control.method`
- `effort_control.active_joints`
- `effort_control.follower_joints`
- `physx.material`
- `physx.rigid_body`

`active_joints` 描述主动命令空间关节：`position_control.method` 和 `velocity_control.method` 支持 `implicit` / `explicit`，`explicit` 会在 Python 侧读取实际关节状态并计算 effort；`effort_control.method` 当前为 `direct`，直接下发 effort command。`follower_joints` 的语义不随主动模式变化，mimic follower 关节始终读取 master 实际角度，并用 Isaac position drive 跟随，因此每个模式下都应配置 `stiffness`、`damping`、`max_force` 和 `joint_friction`。

脚本通过 `--controller-config configs/controllers` 读取目录，目录内必须包含 `arm_controller.yaml` 和 `hand_controller.yaml`。`scripts/run_pinch_grasp.py` 可用 `--control-mode position|velocity|effort` 选择主动关节控制模式。

### 环境和 solver

`configs/envs/*.yaml` 使用：

```yaml
env:
  name: rope_scene
  gravity_z: -9.81
  add_ground: true
  physics_frequency: 200.0
  render_frequency: 60.0

solver:
  type: TGS
  apply_scope: arm_hand
  arm_position_iterations: 24
  arm_velocity_iterations: 4
  hand_position_iterations: 24
  hand_velocity_iterations: 4
```

`solver` 用于覆盖机器人 articulation 的 PhysX solver iteration。外部自定义场景可以省略 `solver`，脚本会保留 PhysX 默认值。

### 绳体对象

`configs/objects/capsule_rope.yaml` 包含资产输出和绳体参数：

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

修改绳体参数后，需要重新运行 `scripts/build_capsule_rope_asset.py` 生成 USD。

### 任务配置

`configs/trajectories/joint_target.yaml` 定义稀疏关节目标：

```yaml
trajectory:
  type: joint_target
  duration: 2.0
  sample_hz: 200.0
  interpolation: smoothstep
  targets:
    AR5V2_L_arm_joint_1: 0.4
```

`configs/trajectories/pinch_grasp.yaml` 定义抓取目标、阶段时长和手型：

```yaml
grasp:
  endpoint: left
  target_world_offset: [0.02, 0.0, 0.03]
  target_rpy: [0.0, 2.007128639793479, -1.5707963267948966]
  use_orientation: true
  approach_distance: 0.10
  lift_height: 0.4
  tcp_frame_name: pinch_tcp
```

cuMotion IK/FK 求解器默认参数可放在 `configs/cumotion/default.yaml` 的
`cumotion.kinematics` 段，并由具体 robot YAML 覆盖：

```yaml
cumotion:
  kinematics:
    ik:
      position_tolerance: 0.002
      orientation_tolerance: 0.01
      ccd_max_iterations: 180
      bfgs_max_iterations: 80
      orientation_weight: 0.25
    fk: {}
```

## 执行流程

### 关节目标

`source/manipulation_project/execution/steps.py` 提供唯一的执行步骤层：

1. `SmoothJointTargetStep`：在两个完整 DOF 目标之间做 smoothstep 过渡，适合手部开合等简单阶段。
2. `FullJointTrajectoryStep`：播放已经规划并映射到完整 DOF 顺序的 `JointTrajectory`。
3. `execute_command_joint_trajectory(...)`：播放 controller command-space 轨迹，适合简单 demo 或外部已采样命令轨迹。
4. `HoldJointTargetStep`：保持某个完整 DOF 目标一段时间。

### Pinch Grasp

`scripts/run_pinch_grasp.py` 的流程：

1. 读取 AR5+L6、rope、grasp、controller、env 配置。
2. 引用 capsule rope USD。
3. 导入 AR5+L6 组合 MJCF，并按 `--control-mode` 选择当前 runtime controller 配置。
4. 用闭合手型和 MJCF body 链计算 thumb/index 夹捏中心 TCP。
5. 通过 `make_cumotion_context(...)` 创建带 pinch TCP 的 cuMotion context。
6. 基于 context 创建 `CuMotionInverseKinematics` 和 motion planner wrapper。
7. 求解 approach、grasp、lift、wiggle 的 TCP IK。
8. 将 IK 结果写入完整 articulation DOF 目标，并同步 mimic follower。
9. 分阶段执行 prep、move、approach、close、lift、wiggle、hold，并记录 CSV。

## Mimic 关节

LinkerHand L6 的 follower 关节通过 MJCF `equality/joint` 表达。`robots/mimic.py` 解析 `polycoef`：

```text
dependent = a0 + a1 * master + a2 * master^2 + ...
```

运行时不依赖 importer 的实时硬约束同步 follower，而是在软件层根据 master 实际状态更新 follower position drive 目标；速度目标通过多项式导数计算。

## 坐标、姿态和单位

- 项目对外统一使用 wxyz 四元数，即 `[w, x, y, z]`。
- SciPy 内部使用 xyzw，转换封装在 `utils/math_utils.py` 和 `utils/rotations.py`。
- 配置中的 RPY 使用固定轴 XYZ 顺序，即外旋 XYZ；在 SciPy 中对应小写 `"xyz"`。
- 距离单位为 m，所有角度配置统一为 rad，关节速度为 rad/s。

## 资产和命名

当前正式命名不使用连字符 `-`，避免 Isaac importer 把 `-` 转成 `_` 后造成配置关节名和 runtime DOF 名不一致。更多规则见 `ASSET_NAMING_CONVENTIONS.md`。

当前默认资产：

- 左臂：`assets/single_system/arm/AR5V2_L/`
- 右臂：`assets/single_system/arm/AR5V2_R/`
- 左手：`assets/single_system/hand/L6V1_L/`
- 右手：`assets/single_system/hand/L6V1_R/`
- 左臂+左手组合：`assets/combined_system/AR5V2_L6V1_L/`
- 绳体对象：`assets/dynamic_env_objects/capsuleropeV1_default/`

右侧 AR5/L6 当前提供单体 URDF 和 mesh；组合运行资产需要继续生成对应 MJCF/XRDF 描述。

## 日志和 Foxglove

关节跟踪日志使用 CSV，默认路径在 `logs/joint_tracking/`，默认配置为 `configs/logging/default_logger.yaml`。所有控制模式都可以按需记录实际位置、实际速度、PhysX measured effort 和 Isaac applied effort；命令列则按控制语义记录 position / velocity / effort 目标。`measured_effort` 和 `applied_effort` 读取相对更贵，默认关闭，可以在 YAML 中打开，或运行 `scripts/run_pinch_grasp.py` 时传 `--log-measured-effort` / `--log-applied-effort`。

常用列名：

- `qd_*` / `q_*`：命令位置和实际位置。
- `vd_*` / `v_*`：命令速度和实际速度。
- `tau_cmd_*`：语义 effort command；implicit drive 下通常为 `nan`。
- `tau_action_*`：控制器实际下发给 Isaac 的 effort action，需打开 `action_effort`。
- `tau_measured_*` / `tau_applied_*`：PhysX measured effort 和 Isaac applied effort。

采样频率可以通过 `logging.interval_steps` 或 `--log-interval-steps` 降低。`telemetry/foxglove.py` 提供可选 Foxglove 输出：

- 离线 MCAP；
- 本地 WebSocket live server；
- `JointStates` 曲线；
- TCP、轨迹点和调试点的 `SceneUpdate` marker。

Foxglove 依赖已经包含在完整依赖安装中：

```bash
env_isaaclab/bin/python -m pip install -r requirements.txt
```

最小示例：

```python
from manipulation_project.telemetry.foxglove import FoxgloveLogger

with FoxgloveLogger.open_mcap("logs/debug.mcap") as logger:
    logger.log_joint_state(
        joint_names=["joint_1"],
        positions=[0.1],
        velocities=[0.0],
        time_s=0.0,
    )
```

## 验证

语法检查：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m compileall -q source scripts tests
```

检查 YAML 配置：

```bash
PYTHONPATH=source env_isaaclab/bin/python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path("configs").rglob("*.yaml")):
    yaml.safe_load(path.read_text(encoding="utf-8"))
print("yaml ok")
PY
```

运行轻量测试：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m pytest -q
```

当前轻量测试覆盖控制器配置、显/隐式关节控制、插值、cuMotion 轨迹适配、MJCF mimic 解析、配置加载、pinch TCP 计算、临时 TCP URDF 写入和 Foxglove logger 基本行为。
