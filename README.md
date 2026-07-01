# LinkerHand Simulation

这是一个基于 Isaac Sim / Isaac Lab 的机械臂、灵巧手和绳体操作仿真工程。当前主线围绕
AR5 机械臂、LinkerHand L6 灵巧手、capsule/cuboid 近似绳体，以及 cuMotion 后端运动生成
展开，用于验证 TCP 定义、IK/FK、路径规划、mimic 关节同步、PhysX 参数和 scripted pinch
grasp 流程。

项目的核心原则是分层清楚：

- `configs/` 随项目一起版本管理，作为项目内默认资产、控制器、环境、对象、日志和 cuMotion profile 的伴随配置层；代码可以依赖当前 schema，但具体资产和参数实例应尽量留在配置文件里。
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
│   ├── controllers/          # arm/hand 控制模式、增益、限幅和 follower drive
│   ├── cumotion/             # cuMotion 默认 profile 和详细参数示例
│   ├── dual_arm/             # 双臂动作/规划语义，例如 C-space 分区和 TCP frame 名
│   ├── envs/                 # scene、物理步频、solver、机器人/对象实例摆放
│   ├── logging/              # 关节跟踪日志配置
│   ├── objects/              # 可生成对象配置，例如 capsule rope
│   └── robots/               # Isaac 机器人资产、物理覆盖、cuMotion 资源
├── docs/                     # cuMotion 接口、Isaac 碰撞说明和风险记录
├── scripts/                  # Isaac Sim 运行入口和资产生成入口
├── src/linkerbot_sim/
│   ├── app/                  # 环境运行参数、SimulationApp/World 会话装配
│   ├── assets/               # 资产导入、USD/PhysX 覆盖、solver 设置
│   ├── backends/cumotion/    # cuMotion context、FK/IK、planner、path/trajectory adapter
│   ├── controllers/          # 控制器配置解析和 runtime controller
│   ├── envs/                 # World 和场景构建
│   ├── execution/            # 目标/轨迹执行步骤
│   ├── logging/              # CSV 日志
│   ├── objects/              # 绳体对象资产生成和引用
│   ├── planning/             # 后端无关请求/结果、碰撞对象、双臂 C-space 分区
│   ├── robots/               # 关节组、mimic/equality、状态工具
│   ├── tcp/                  # TCP frame 和夹捏中心计算
│   ├── telemetry/            # Foxglove、MCAP、WebSocket
│   ├── trajectories/         # JointTrajectory 容器、builder 和 command-space 轨迹组装
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
PYTHONPATH=src env_isaaclab/bin/python <command>
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

## 运行模式

常用入口按“依赖重量”和“验证目标”分成几类。先用 dry-run/smoke 确认配置，再运行 Isaac
GUI 或完整动作，会更容易定位问题。

| 模式 | 入口 | 启动 Isaac | 需要 cuMotion | 主要用途 |
| --- | --- | --- | --- | --- |
| 生成绳体 USD | `scripts/build_capsule_rope_asset.py` | 是，headless | 否 | 根据 `configs/objects/capsule_rope.yaml` 写 USD/PhysX schema |
| 单臂导入保持 | `scripts/pinch_grasp.py --no-grasp --short-smoke` | 是 | 否 | 验证 AR5+L6、rope、controller、logging 基础链路 |
| 单臂 pinch grasp | `scripts/pinch_grasp.py` | 是 | 是 | 完整 TCP、IK、motion planner、trajectory、execution 流程 |
| 双臂 motion dry-run | `scripts/dual_arm_motion_test.py --dry-run` | 否 | 否 | 校验 scene runtime、cuMotion 和 dual-arm profile 引用 |
| 双臂 cuMotion motion | `scripts/dual_arm_motion_test.py` | 是 | 是 | 执行脚本内 Python 参数定义的 IK、TCP 直线、TCP 圆弧和 C-space planner 动作 |

常用命令：

```bash
# 生成或更新 capsule rope USD
PYTHONPATH=src env_isaaclab/bin/python scripts/build_capsule_rope_asset.py

# 只导入绳体和 AR5+L6，并快速保持初始姿态
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py --no-grasp --short-smoke

# 运行 GUI pinch grasp demo
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py --gui

# 只校验双臂 scene runtime、cuMotion 和 dual-arm profile，不启动 Isaac
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py --dry-run

# 导入左右 AR5+L6，并执行脚本内定义的双臂 cuMotion 动作
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py
```

`pinch_grasp.py` 常用参数：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --endpoint right \
  --control-mode position \
  --gui
```

日志默认写入 `logs/joint_tracking/`。如果只想快速验证导入、控制器和日志，可以使用
`--no-grasp --short-smoke`；如果希望最终姿态持续到窗口关闭，可以增加 `--hold --gui`。

## 调用结构

### 单臂 Pinch Grasp

`scripts/pinch_grasp.py` 是当前最完整的运行入口。调用链按阶段分为“配置加载、场景导入、规划、
执行”四层：

```text
pinch_grasp.py
  load_yaml(...) + merged_robot_config_with_cumotion_profile(...)
  EnvRuntimeSettings.from_env_config(...)
  create_simulation_session(...)
  add_rigid_objects(...) + add_capsule_rope_reference(...)
  import_execution_robot_to_stage(...)
  world.reset()
  finalize_robot_controller(...)
  make_pinch_tcp_transform(...) + make_cumotion_context(...)
  CuMotionInverseKinematics / CuMotionMotionPlanner
  trajectory_sampler.joint_trajectory_from_cumotion(...)
  trajectories.command_trajectory.command_trajectory_from_arm_trajectory(...)
  execution.steps.*Step.run(...)
```

典型执行顺序：

1. 加载 cuMotion profile、robot、controller、env、rope 和 logging YAML。
2. 将 `configs/cumotion/default.yaml` 的算法默认值合入 robot config；robot config 只覆盖 URDF/XRDF/frame 资源字段。
3. 解析 env runtime settings，启动 Isaac `SimulationApp`，创建 World、重力、physics/render dt。
4. 引用 env `objects[]` 和已生成的 capsule rope USD。
5. 导入 AR5+L6 组合 MJCF，并在 `world.reset()` 前应用 drive、摩擦、solver iteration、重力等 USD 覆盖。
6. `world.reset()` 后创建 `JointController`，按 `--control-mode` 选择 position、velocity 或 effort 主动控制。
7. 根据闭合手型和 MJCF body 链在脚本侧计算 pinch TCP 相对末端的 `xyz/rpy`，并通过临时 URDF 创建 cuMotion context。
8. 使用 cuMotion 求解 approach、grasp、lift、wiggle 等阶段的 IK 和运动轨迹。
9. 把 cuMotion C-space 结果按关节名映射回 controller command-space。
10. `execution.steps` 按 physics dt 播放 command-space 目标或轨迹，并写 CSV 日志。

动作脚本不会在执行过程中实时做 task-space conversion。所有 IK、指定路径转换和 trajectory
generation 都在规划阶段完成；执行阶段只按时间采样已经生成的 `JointTrajectory`。

### 双臂 C-space 和 TCP 配置

双臂规划基础设施把机器人模型资源配置和双臂语义配置合并，用于表达“一个 14-DOF cuMotion
context，多个可选 TCP frame”：

```text
configs/envs/scene2.yaml             # dual scene robot instances + root_pose
configs/robots/ar5v2_l6v1_l.yaml          # left Isaac import + cuMotion model resources
configs/robots/ar5v2_l6v1_r.yaml          # right Isaac import + cuMotion model resources
configs/dual_arm/ar5v2_l6v1_dual.yaml     # arm_joints + flange_frame + tcp_frame + MJCF path
configs/cumotion/default.yaml             # IK/planner algorithm defaults
  -> prepare_cumotion_config_from_robot_config(...)
  -> DualArmJointPartitions.from_joint_names(...)
  -> selected_side_goal(...)
  -> optional make_cumotion_context(..., tcp=(left_tcp, right_tcp))
```

这条链路当前主要由轻量测试覆盖：`tests/test_dual_arm_selectable_tcp.py` 验证左右分区、
selected-side goal 和双臂轨迹拆分；`tests/test_dual_cumotion_urdf.py` 验证双臂 URDF/XRDF
生成路径和左右 flange frame 传递。

### 双臂 cuMotion Motion Test

`scripts/dual_arm_motion_test.py` 使用脚本内 Python 参数定义 TCP 和动作序列，通用的 cuMotion
profile 加载、TCP 注入、IK、路径规划和双臂轨迹执行都封装在 `src`：

```text
load_dual_robot_runtime_config(...)
  -> create_dual_robot_runtime(...)
  -> DualRobotAppRuntime
  -> run_dual_arm_cumotion_motion(tcp=..., moves=(...))
  -> make_cumotion_context(...)
  -> IK / specified path / C-space planner
  -> DualCommandPositionTrajectoryStep.run(...)
```

`--dry-run` 只校验 scene runtime、cuMotion 和 dual-arm profile 引用，不启动 Isaac。

## 双臂双手协作

当前双臂实现把“规划模型”和“Isaac 执行模型”分开：cuMotion 看到一个融合的 14-DOF robot
description，Isaac stage 里仍导入左右两个 AR5+L6 articulation。`configs/envs/scene2.yaml`
用 `robots.dual.left/right.robot_profile` 选择左右单臂 robot profile，并用
`robots.dual.left/right.root_pose` 描述左右安装位姿；`configs/robots/ar5v2_l6v1_l.yaml` 和
`configs/robots/ar5v2_l6v1_r.yaml` 分别提供左右 Isaac 资产和 cuMotion 单臂 XRDF/URDF/flange；
`configs/dual_arm/ar5v2_l6v1_dual.yaml` 保存左右 arm C-space、flange/TCP frame 和用于计算 pinch
TCP 的组合 MJCF 路径。运行时会根据左右单臂资源和 scene `root_pose` 生成缓存的
融合 URDF/XRDF，最终 C-space 是 14 个机械臂关节：

```text
left_arm_7 + right_arm_7
```

双臂没有默认 TCP；IK/planning 必须显式选择左侧或右侧 TCP。典型做法是在同一个 cuMotion context
中注入 `left_pinch_tcp` 和 `right_pinch_tcp`，每次规划/运动阶段传入一个 `tcp_frame_name`。
当前 selected-side 规则是：单 TCP IK 返回完整 `q[14]` 后，只采纳选定侧 7 个 arm joints，
另一侧保持当前目标；同一侧可以连续执行多个阶段，不要求左右轮流。pinch grasp 的预夹/闭合手型
仍留在动作脚本中，避免机器人配置混入任务语义。

执行侧使用双机器人结构：`robots.dual.left/right` 分别描述 Isaac 中导入的左右 articulation。未写
`controlled_joints` 时执行层默认选择全部主动 DOF，并剔除 mimic follower。`execution.dual_steps`
会在每个 physics step 前先下发左右 controller 目标，再统一调用一次 `world.step()`；融合 cuMotion
轨迹执行前按关节名拆回左右 arm command columns，手部目标由动作阶段补齐。

当前不声称自动免碰撞。融合 XRDF 保留 14-DOF C-space 和后验检查入口，但真正可靠的避碰还需要有效
collision spheres、正确的 self-collision mask、collision-aware IK/planner，以及轨迹采样后的碰撞检查。

## 配置系统

配置文件使用固定顶层结构。常见入口如下：

- `configs/robots/ar5v2_l6v1_l.yaml`：左侧单机器人 Isaac 导入、可选 controlled joint selector、机器人物理属性，以及 cuMotion URDF/XRDF/frame 资源。
- `configs/robots/ar5v2_l6v1_r.yaml`：右侧单机器人 Isaac 导入、可选 controlled joint selector、机器人物理属性，以及 cuMotion URDF/XRDF/frame 资源。
- `configs/dual_arm/ar5v2_l6v1_dual.yaml`：双臂规划语义，包括左右 arm C-space 关节顺序、flange/TCP frame 和组合 MJCF 路径。
- `configs/controllers/arm_controller.yaml` 和 `configs/controllers/hand_controller.yaml`：主动关节和 mimic follower 的控制参数。
- `configs/envs/scene1.yaml`：单臂 scene，包含 World 物理步频、渲染步频、重力、scene solver type、`robots.single` 和已有环境资产。
- `configs/envs/scene2.yaml`：双臂 scene，包含 World 设置、`robots.dual.left/right` 和已有环境资产。
- `configs/objects/capsule_rope.yaml`：绳体运行时对象 profile，引用已生成 USD 并提供接触材质和 solver iteration 覆盖。
- `configs/cumotion/default.yaml`：项目默认 cuMotion 算法 profile；脚本可换 profile，但不在脚本中散落算法默认值。
- `configs/logging/default_logger.yaml`：关节跟踪 CSV 输出和记录列。
- `scripts/pinch_grasp.py`：自包含 pinch grasp 动作参数和执行流程。

每个配置目录都提供了可复制改名的示例文件：

- `configs/robots/example.yaml`：单机器人配置模板。
- `configs/dual_arm/example.yaml`：双臂语义配置模板。
- `configs/controllers/example.yaml`：控制器 profile 字段说明。
- `configs/envs/example.yaml`：环境、solver、`robots.single` / `robots.dual` 和 `objects[]` 字段说明。
- `configs/objects/example.yaml`：运行时对象 profile 模板。
- `configs/cumotion/example.yaml`：cuMotion IK/planner profile 字段说明。
- `configs/logging/example.yaml`：CSV 日志配置模板。

cuMotion 相关配置的优先级是：

```text
configs/cumotion/default.yaml
  < configs/robots/*.yaml
  < scripts/pinch_grasp.py 内置动作参数
```

实际合并分两条线：

- `cumotion` profile 先作为 robot config 的默认值；robot YAML 只提供单 articulation 模型资源字段：`xrdf_path`、`urdf_path`、`flange_frame`。双臂由 scene 的 `robots.dual.left/right` 选择两个单臂 robot profile，runtime 再结合左右 `root_pose` 生成临时双臂 URDF/XRDF。
- `cumotion.motion_planner` profile 直接解析成动作脚本使用的 `MotionPlannerBackendConfig`；抓取目标、阶段时长和手型固定在 `scripts/pinch_grasp.py` 中。

配置合并是递归 mapping merge。列表和标量按覆盖值整体替换，不做逐项合并，这对关节列表、
轨迹点和手型目标更安全。

### 配置边界

当前约定是“资源、算法、动作语义分开放”：

`configs/` 是项目自带的伴随配置层，不是完全外置的临时输入。项目默认机器人、默认场景、
默认控制器和默认 cuMotion profile 都可以放在这里，并随代码一起提交。脚本中可以保留“默认选
哪个 scene”的名称，但不应把可替换的资产路径、root pose、solver、controller gain 或
planner 参数散落硬编码在动作逻辑里。

- `configs/robots/*.yaml`：Isaac 导入、可选 controlled joint selector、机器人刚体
  重力策略，以及 cuMotion 模型资源字段。这里不放 IK/planner 算法参数，也不放 pinch grasp 手型。
- `configs/cumotion/*.yaml`：cuMotion 算法 profile，包括 `kinematics`、`motion_planner`、solver
  params、trajectory generation limits 等。
- `configs/dual_arm/*.yaml`：双臂动作/规划语义，例如左右 arm C-space 关节顺序、TCP frame 名、
  用于计算 pinch TCP 的组合 MJCF 路径。这里不导入 Isaac，也不配置 cuMotion 算法。
- `scripts/pinch_grasp.py`：动作目标、阶段顺序、默认预夹/闭合手型、抓取策略。动作语义留在脚本，
  避免机器人资产配置变成任务配置。
- `configs/controllers/*.yaml`：控制器运行时参数，包括 position/velocity/effort 模式、implicit
  drive 或 explicit effort、stiffness/damping/max effort、follower 参数。
- `configs/robots/*.yaml`：机器人资产、重力、材料、刚体阻尼，以及 cuMotion 模型资源。
- `configs/envs/*.yaml`：World 物理步频、渲染步频、重力、scene solver type、`robots.single`
  或 `robots.dual.left/right` 机器人实例选择和安装位姿，以及 `objects[]` 场景物体摆放。`objects[]` 引用已有 USD/URDF
  资产，不生成对象本身。
- `configs/objects/*.yaml`：运行时对象 profile，例如已有 USD/URDF 资产路径、导入参数、接触材质
  和对象级 solver 覆盖。对象在世界中的 `root_pose` 仍由 env `objects[]` 决定；capsule rope
  等资产生成参数放在 `tools/assets/configs/*.yaml`。
- `configs/logging/*.yaml`：CSV 日志开关、输出路径、采样降频和需要记录的列。

### Isaac 导入碰撞近似

项目只把一个高层字段暴露给机器人和环境对象导入：

```yaml
robot:
  import:
    collision_approximation: convex_decomposition

objects:
  - name: workstation_armbase
    asset_type: urdf
    import:
      collision_approximation: convex_decomposition
```

当前项目只支持两种 snake_case 值：`convex_decomposition` 和 `convex_hull`。

这个字段只作用在 Isaac importer 阶段：

- MJCF 导入时映射到 importer 的 convex decomposition 开关。
- URDF 导入时映射到 importer 的 `convex_decomp` 开关。
- `asset_type: usd` 的环境对象是直接 reference 已有 USD，碰撞几何已经写在资产里，不会重新跑 importer。
- 它不改变 cuMotion XRDF/URDF 规划模型，也不表示 PhysX 运行时又做了一次新的碰撞分解。

如果资产里已经手工准备了凸碰撞 mesh，这个字段仍只是告诉 importer 如何处理需要导入的 mesh 碰撞。
更完整的 Isaac/USD/PhysX 层说明见 `docs/isaac_collision_approximation.md`。

### 机器人重力策略

机器人刚体是否受重力影响写在 robot YAML，而不是 env YAML：

```yaml
robot:
  physics:
    gravity:
      default: false
      arm: false
      hand: false
```

`false` 表示导入后禁用对应刚体重力，`true` 表示保留重力。`default` 用于未能按名称分到
`arm`/`hand` 的刚体，`arm` 和 `hand` 可分别覆盖。双机器人运行时分别读取左右单 robot profile
里的 `robot.physics.gravity`。

脚本不再提供额外的重力开关；需要调整机器人重力时直接修改 robot YAML。

### 机器人 PhysX 覆盖

机器人接触材质和刚体阻尼也是机器人资产物理属性，写在 robot YAML，而不是 controller YAML：

```yaml
robot:
  physics:
    physx:
      material:
        contact_static_friction: 0.8
        contact_dynamic_friction: 0.6
        contact_restitution: 0.0
      rigid_body:
        linear_damping: 0.0
        angular_damping: 0.1
```

controller YAML 仍负责控制模式、gain、限幅和 follower drive；导入阶段会把 controller 生成的
drive/friction seed 与 robot YAML 的材料/刚体阻尼合并后写入 USD。

### 机器人 Solver Iteration

机器人刚体的 solver iteration 也是机器人资产物理属性，写在 robot YAML；env YAML 只保留
scene 级 `solver.type`：

```yaml
robot:
  physics:
    solver:
      arm:
        position_iterations: 32
        velocity_iterations: 4
      hand:
        position_iterations: 32
        velocity_iterations: 4
```

双机器人运行时分别读取左右单 robot profile 里的 `robot.physics.solver`。

### cuMotion 配置

Robot YAML 的 `cumotion` 顶层只保存机器人描述和 frame：

```yaml
cumotion:
  xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
  flange_frame: AR5V2_L_arm_flan_link
  custom_tcp_frame: null
```

`flange_frame` 是基础机械臂末端 frame。`custom_tcp_frame` 只在基础 URDF/XRDF 已经自带工具
TCP frame，并且希望默认用它做 IK/FK/planning 时填写。pinch grasp 的 pinch TCP 通常由脚本
根据手指几何临时写入 URDF，因此不要在 robot YAML 里手写 `left_pinch_tcp` 或
`right_pinch_tcp`。

双臂配置运行时生成融合 URDF/XRDF，不在 YAML 中直接写单个 `xrdf_path/urdf_path/flange_frame`：

```yaml
cumotion:
  left:
    xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
    urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
    flange_frame: AR5V2_L_arm_flan_link
  right:
    xrdf_path: assets/single_system/arm/AR5V2_R/AR5V2_R.xrdf
    urdf_path: assets/single_system/arm/AR5V2_R/AR5V2_R.urdf
    flange_frame: AR5V2_R_arm_flan_link
  output_dir: .cache/cumotion
  robot_name: dual_ar5v2
  parent_link: world
```

`output_dir` 控制自动生成的双臂 URDF/XRDF 缓存目录。`robot_name` 是生成 URDF 的
`<robot name="...">`，`parent_link` 是融合模型里承载左右臂固定安装位姿的虚拟根 link；
它不是 USD 的 `/World` prim。除非要和外部工具的命名约定对齐，这两个字段通常保持默认。

IK/FK 算法参数放在 `configs/cumotion/*.yaml` 的 `kinematics` 下：

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

路径级规划参数放在 `configs/cumotion/*.yaml` 的 `motion_planner` 下：

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

更完整的字段说明见 `configs/cumotion/example.yaml`。

## 核心接口职责

下面这些接口是当前代码里最常被脚本、测试和后端共享的边界。读代码时可以先按这张表定位。

### 配置和资产

| 接口 | 位置 | 作用 |
| --- | --- | --- |
| `load_yaml(path)` | `utils/config.py` | 读取仓库相对 YAML，并保证顶层是 mapping。 |
| `deep_merge(base, override)` | `utils/config.py` | 递归合并 profile 和 robot/action 配置；列表和标量整体替换。 |
| `RobotAssetConfig` | `assets/robot_loader.py` | 描述单个机器人资产如何导入 Isaac stage：asset type/path、prim path、name。 |
| `RobotGravityPolicy` | `assets/robot_loader.py` | 描述 robot YAML 中 default/arm/hand 的刚体重力策略。 |
| `RobotPhysxOverrides` | `assets/robot_loader.py` | 描述 robot YAML 中 default/arm/hand 的接触材质和刚体阻尼覆盖。 |
| `RobotSceneInstanceConfig` | `assets/robot_loader.py` | 描述 env 中 `robots.single` 或 `robots.dual.left/right` 的 robot profile 引用和 scene root pose。 |
| `RobotExecutionConfig` | `assets/robot_loader.py` | 单个 articulation 的执行配置：资产、scene root pose、controlled joint selector。 |
| `DualRobotExecutionConfig` | `assets/robot_loader.py` | 从左右两个单 robot profile 组装双 articulation 执行配置。 |
| `import_robot_asset(...)` | `assets/robot_loader.py` | 调 Isaac importer，把 MJCF/URDF 放进当前 USD stage 并返回 articulation root。 |
| `apply_root_pose(...)` | `assets/robot_loader.py` | 把导入后的机器人根 prim 放到配置指定的世界位姿。 |
| `apply_robot_usd_overrides(...)` | `assets/usd_overrides.py` | 写入 USD 层 drive seed、摩擦、材质、follower drive 初值。 |
| `apply_robot_gravity_policy(...)` | `assets/usd_overrides.py` | 按 robot 重力策略写入 rigid body `disableGravity`。 |
| `scene_solver_settings(...)` / `robot_solver_settings(...)` / `apply_solver_iteration_overrides(...)` | `assets/solver_overrides.py` | 分别读取 env scene solver type 和 robot solver iteration，并写到 scene 或机器人刚体。 |

### 运行时装配

| 接口 | 位置 | 作用 |
| --- | --- | --- |
| `EnvRuntimeSettings.from_env_config(...)` | `app/runtime_settings.py` | 从 env YAML 解析 physics/render frequency、世界 gravity 和 ground。 |
| `create_simulation_session(...)` | `app/simulation_session.py` | 启动 Isaac app，创建 World/stage，并返回 Isaac runtime type handle。 |
| `import_execution_robot_to_stage(...)` | `execution/setup.py` | reset 前导入机器人，并应用 root pose、USD/PhysX、solver 和重力覆盖。 |
| `finalize_robot_controller(...)` | `execution/setup.py` | reset 后清零速度、创建 `JointController`，并配置 runtime gain。 |

### cuMotion Backend

| 接口 | 位置 | 作用 |
| --- | --- | --- |
| `CuMotionConfig` | `backends/cumotion/context.py` | cuMotion context 的模型资源和默认 kinematics 参数容器。robot YAML 只提供资源字段，算法默认来自 `configs/cumotion/*.yaml`。 |
| `CuMotionContext` | `backends/cumotion/context.py` | 加载 robot description/kinematics，维护 collision world，创建 FK/IK/planner wrapper。 |
| `make_cumotion_context(...)` | `backends/cumotion/tcp_context.py` | 可选地把自定义 TCP 写进临时 URDF，再创建 `CuMotionContext`。 |
| `CuMotionInverseKinematics` | `backends/cumotion/inverse_kinematics.py` | 把项目 `IKRequest` 转成几何 IK 或 collision-free IK，并返回 `IKResult`。 |
| `CuMotionMotionPlanner` | `backends/cumotion/motion_planner.py` | 按 `planning_pipeline` 分发到 trajectory optimization、graph search 或 specified path。 |
| `MotionPlannerBackendConfig` | `backends/cumotion/motion_planner_config.py` | motion planner profile 的 dataclass 结构，按 pipeline 分组保存参数。 |
| `prepare_cumotion_config_from_robot_config(...)` | `backends/cumotion/dual_urdf.py` | 单臂直接解析资源；双臂按 env root pose 生成缓存 URDF，并融合左右 XRDF。 |
| `joint_trajectory_from_cumotion(...)` | `backends/cumotion/trajectory_sampler.py` | 把 cuMotion `Trajectory.eval_all(t)` 采样成项目 `JointTrajectory`。 |

### Planning 和 Trajectory

| 接口 | 位置 | 作用 |
| --- | --- | --- |
| `IKRequest` / `MotionRequest` / `SpecifiedPathRequest` | `planning/requests.py` | 后端无关请求模型。脚本表达目标，backend 负责适配到 cuMotion 类型。 |
| `IKResult` / `MotionResult` | `planning/results.py` | 后端无关结果模型。`MotionResult.trajectory` 是执行前采样的主数据。 |
| `CollisionObject` | `planning/collision_objects.py` | 项目侧环境障碍物描述，后端转换成 cuMotion collision world。 |
| `DualArmJointPartitions` | `planning/dual_arm_cspace_partition.py` | 根据融合 C-space 关节名生成左右索引分区。 |
| `selected_side_goal(...)` | `planning/dual_arm_cspace_partition.py` | 在双臂 14-DOF 解中只采纳选定侧，另一侧保持 base_q。 |
| `split_dual_arm_trajectory_to_commands(...)` | `planning/dual_arm_cspace_partition.py` | 把融合 14-DOF arm trajectory 拆成左右 controller command-space trajectory。 |
| `JointTrajectory` | `trajectories/types.py` | 项目统一关节轨迹容器，保存 times、positions、velocities、accelerations、jerks、efforts、phases、joint_names。 |
| `joint_trajectory_from_positions(...)` | `trajectories/joint_trajectory_builder.py` | 从位置矩阵构造 `JointTrajectory`，用有限差分补导数。 |
| `command_trajectory_from_arm_trajectory(...)` | `trajectories/command_trajectory.py` | 把 arm-only trajectory 嵌入 controller command-space；非机械臂列按 start/target 插值。 |

### 控制和执行

| 接口 | 位置 | 作用 |
| --- | --- | --- |
| `JointController` | `controllers/joint_controller.py` | 把 command/full DOF target 转为 Isaac `ArticulationAction`；运行时刷新 mimic follower。 |
| `JointControlSettings` | `controllers/types.py` | 描述控制模式、method、gain、effort limit 等 runtime 控制参数。 |
| `ExecutionRuntime` / `RobotSideRuntime` / `DualRobotRuntime` | `execution/runtime.py`, `execution/dual_runtime.py` | 把 articulation、controller、world 和 action type 组合成执行步骤需要的上下文。 |
| `SmoothCommandPositionTargetStep` | `execution/steps.py` | 在 command-space 中平滑插值到目标。 |
| `CommandPositionTrajectoryStep` | `execution/steps.py` | 播放已经采样好的 command-space `JointTrajectory`。 |
| `HoldCommandPositionTargetStep` | `execution/steps.py` | 保持 command-space 目标若干步或直到 GUI 关闭。 |
| `DualCommandPositionTargetStep` | `execution/dual_steps.py` | 双臂执行步骤；每个 physics step 前先下发左右目标，再统一 `world.step()`。 |

### TCP 和对象

| 接口 | 位置 | 作用 |
| --- | --- | --- |
| `TcpTransform` | `backends/cumotion/tcp_frame.py` | 客户侧 TCP 输入；只描述相对末端/flange 的笛卡尔变换，后端再绑定到具体 flange frame。 |
| `make_pinch_tcp_transform(...)` | `scripts/pinch_grasp.py` | 脚本侧根据 MJCF 和闭合手型计算 thumb/index 夹捏中心 TCP 变换。 |
| `write_tcp_urdf_with_frames(...)` | `backends/cumotion/tcp_urdf_builder.py` | 把一个或多个 TCP fixed link 写入临时 URDF。 |
| `CapsuleRopeConfig` | `objects/dynamic_chain/capsule_rope.py` | 解析 capsule rope 运行时 profile，引用已生成 USD 并应用运行时物理覆盖。 |

## cuMotion 后端

`src/linkerbot_sim/backends/cumotion/` 是 cuMotion 的适配层。主要文件职责如下：

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

项目内部 `JointTrajectory` 位于 `src/linkerbot_sim/trajectories/types.py`。它保存：

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
3. `command_trajectory_from_arm_trajectory(...)` 把 arm-only 轨迹嵌入 controller command-space，
   手部等非机械臂列按 start/target 插值。
4. 执行层按 command-space 或 full DOF 目标下发给 `JointController`。

如果轨迹采样频率和 physics step 频率不同，执行层以 physics dt 为准。`execute_full_joint_trajectory(...)`
会在每个物理步调用 `trajectory.eval_all(time_s)` 做线性插值，然后下发该时刻的位置、速度和
effort。换句话说，轨迹不是按原采样点逐点硬播放，而是按仿真时钟求值。

`trajectories/joint_trajectory_builder.py` 只用于从已有位置矩阵构造 `JointTrajectory`，并通过有限
差分补速度、加速度和 jerk。它不负责 cuMotion path conversion，也不负责运动规划。

`trajectories/command_trajectory.py` 不接触 Isaac，也不调用 cuMotion。它只负责“轨迹矩阵组装”：
把后端输出的机械臂 C-space 轨迹写入 command-space 对应列，同时让手部或其它主动关节在同一
时间网格上保持或缓慢过渡。

## Pinch Grasp

`scripts/pinch_grasp.py` 是当前主要动作入口。它的输入是机器人 articulation、rope 配置、MJCF
路径、cuMotion config 和脚本内置动作常量。主要步骤：

1. 展开闭合手型中的 MJCF mimic follower 目标。
2. 沿 thumb/index body 链计算闭合手型下两指尖位置。
3. 取两指尖几何中点作为 `pinch_tcp` 相对末端的 `xyz/rpy` 变换。
4. 通过 `make_cumotion_context(...)` 写临时 URDF，让 cuMotion 能识别该 TCP frame。
5. 计算 approach、grasp、lift、wiggle 的 TCP 目标。
6. 对 approach、lift、wiggle 等阶段求 IK 和关节空间规划。
7. 对接近端块的短距离下沉使用 specified path `TcpLineSegment`，规划期先做 task-space conversion。
8. 把机械臂 C-space 结果映射回完整 DOF，手部目标用脚本内置稀疏关节目标覆盖。
9. 生成 `SmoothCommandPositionTargetStep`、`CommandPositionTrajectoryStep`、`HoldCommandPositionTargetStep` 组成的执行序列。

动作参数集中在 `run_pinch_grasp_action(...)` 和脚本层 helper 里；默认手型由
`default_pre_pinch_hand_targets(...)`、`default_closed_pinch_hand_targets(...)` 生成。它们属于
pinch_grasp 任务语义，不放进机器人 YAML。姿态目标使用固定轴 XYZ 顺序，单位 rad；脚本内部会转换成项目统一的
wxyz 四元数。

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
PYTHONPATH=src env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
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
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --log-measured-effort \
  --log-applied-effort \
  --log-action-effort
```

Foxglove 输出位于 `src/linkerbot_sim/telemetry/foxglove.py`，支持离线 MCAP、本地
WebSocket live server、`JointStates` 曲线和 `SceneUpdate` marker。

## 验证

语法检查：

```bash
PYTHONPATH=src env_isaaclab/bin/python -m compileall -q src scripts tests
```

运行轻量测试：

```bash
PYTHONPATH=src env_isaaclab/bin/python -m pytest -q tests
```

检查 YAML 是否可解析：

```bash
PYTHONPATH=src env_isaaclab/bin/python - <<'PY'
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
