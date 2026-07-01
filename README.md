# LinkerHand Simulation

这是一个基于 Isaac Sim / Isaac Lab 的机械臂、灵巧手和绳体操作仿真工程。当前工程围绕
AR5 机械臂、LinkerHand L6 灵巧手、capsule/cuboid 近似绳体，以及 cuMotion 后端运动生成
展开，用于验证 TCP 定义、IK/FK、路径规划、mimic 关节同步、PhysX 参数、单臂抓绳和双臂
脚本化/交互式运动流程。

项目主线遵循几个边界：

- `configs/` 随代码一起版本管理，保存默认 robot、env、object、controller、logging 和
  cuMotion profile。脚本可以依赖这些 schema，但具体资产路径、安装位姿、solver、增益和
  planner 参数应优先留在配置里。
- `scripts/` 是可直接运行的入口。动作目标、阶段顺序、实验用 TCP 和临时 move 序列可以留在
  脚本里，避免把任务语义塞进机器人资产配置。
- `app/runtime/` 负责 Isaac app、World、stage、机器人导入、对象导入和 controller 装配。
- `app/motion/` 负责把客户侧 motion spec 转成 cuMotion 请求、轨迹和 command-space 执行。
- `app/interactive/` 负责 JSON 协议、队列和 stdin/TCP/WebSocket transport。
- `backends/cumotion/` 是唯一直接适配 cuMotion Python API 的层。
- `execution/` 只按 physics step 播放已经生成好的目标或轨迹，不做 IK 或重新规划。
- `controllers/` 把项目目标转换成 Isaac articulation action，并在每帧刷新 mimic follower。

## 当前能力

- 资产导入：机器人执行侧支持 MJCF/URDF，cuMotion 规划侧使用 URDF/XRDF，环境和动态对象支持
  USD/URDF 引用或导入。
- 单臂运行：支持导入保持、scripted pinch grasp、pinch TCP 计算、IK、规划、轨迹采样和 CSV 日志。
- 双臂运行：Isaac stage 中左右两个 AR5+L6 articulation 独立导入，cuMotion 侧融合成一个
  14-DOF arm C-space。
- 交互运动：双臂 runtime 支持 stdin JSONL、TCP JSONL 和 WebSocket JSON 提交运动命令。
- 绳体对象：支持由 tools 配置生成 capsule/cuboid 刚体链 USD，并由 env `objects[]` 引用到场景。
- cuMotion 后端：提供 FK、IK、collision-free IK、trajectory optimization、graph search、
  specified path、task-space path conversion 和 C-space trajectory generation。
- TCP：支持法兰 TCP、已有工具 TCP、临时 fixed TCP，以及基于闭合手型计算的 thumb/index
  夹捏中心 TCP。
- 轨迹语义：cuMotion `Trajectory` 会采样成项目 `JointTrajectory`；graph search 和
  specified path 成功时必须生成 trajectory，不能只返回离散 path。
- 控制器：支持 position、velocity 和 effort 控制；position/velocity 可选 Isaac implicit drive
  或 Python explicit effort。
- Mimic 关节：解析 MJCF `equality/joint` 的 `polycoef`，运行时按实际 master 状态刷新 follower
  目标。
- 日志和遥测：支持关节目标、实际位置/速度、command effort、action effort、PhysX
  measured/applied effort 的 CSV 记录；可选 Foxglove MCAP/WebSocket 输出。

## 目录结构

```text
.
├── assets/
│   ├── mesh/                 # arm/hand/env object mesh
│   ├── single_system/        # 单体机器人资产，例如 AR5V2_L、AR5V2_R、L6V1_L
│   ├── combined_system/      # 组合机器人资产，例如 AR5V2_L6V1_L/R
│   ├── static_env_objects/   # 静态环境对象，例如 workstationV1_armbase/tablebase
│   └── dynamic_env_objects/  # 动态对象资产，例如 capsuleropeV1_default
├── configs/
│   ├── controllers/          # arm/hand 控制模式、增益、限幅和 follower drive
│   ├── cumotion/             # cuMotion IK/planner profile
│   ├── envs/                 # scene、physics/render frequency、solver、机器人/对象实例摆放
│   ├── logging/              # 关节跟踪 CSV 日志配置
│   ├── objects/              # 运行时对象 profile，引用已有 USD/URDF
│   └── robots/               # Isaac 机器人资产、物理覆盖、cuMotion 资源
├── docs/                     # cuMotion 接口、交互协议、碰撞近似、风险记录和方案文档
├── scripts/                  # Isaac Sim 运行入口和资产生成入口
├── src/linkerbot_sim/
│   ├── app/                  # runtime/motion/interactive 高层装配
│   ├── assets/               # 资产导入、USD/PhysX 覆盖、solver 设置
│   ├── backends/cumotion/    # cuMotion context、FK/IK、planner、path/trajectory adapter
│   ├── controllers/          # 控制器配置解析和 JointController
│   ├── envs/                 # World、viewport、灯光和 visual settings
│   ├── execution/            # 单臂/双臂目标和轨迹执行步骤
│   ├── logging/              # CSV logger 和 effort logger
│   ├── objects/              # rigid object 和 dynamic chain 运行时导入
│   ├── planning/             # 后端无关请求/结果、碰撞对象、双臂 C-space 分区
│   ├── robots/               # 关节组、mimic/equality、状态工具
│   ├── tcp/                  # TCP frame 和夹捏中心计算
│   ├── telemetry/            # Foxglove、MCAP、WebSocket
│   ├── trajectories/         # JointTrajectory 容器、builder 和 command-space 轨迹组装
│   ├── utils/                # 配置、路径、旋转、数学、计时工具
│   └── visualization/        # camera、debug draw、marker
├── tests/                    # 尽量不启动 Isaac Sim 的轻量测试
├── tools/assets/             # 离线资产生成工具，例如 capsule rope USD builder
├── ASSET_NAMING_CONVENTIONS.md
├── pyproject.toml
└── README.md
```

## 环境约定

示例命令默认从仓库根目录运行，并使用仓库内 `env_isaaclab/` Python 环境：

```bash
PYTHONPATH=src env_isaaclab/bin/python <command>
```

项目采用 src-layout。`scripts/pinch_grasp.py`、`scripts/dual_arm_motion_test.py` 和
`scripts/build_capsule_rope_asset.py` 会自行把 `src/` 放进 `sys.path`，但测试、交互片段和
临时 Python 命令仍建议显式设置 `PYTHONPATH=src`。

完整依赖通过 `pyproject.toml` 管理。`simulation` extra 记录了当前代码期望的主要仿真依赖：

- `isaacsim[all]==5.1.0.0`
- `cumotion==1.1.0`
- `torch==2.7.0`

安装示例：

```bash
env_isaaclab/bin/python -m pip install -e ".[dev,visualization,simulation]"
```

如果 Isaac Lab 环境已经安装了 Isaac Sim、cuMotion 和 torch，也可以只安装缺失的普通 Python
依赖。cuMotion/Isaac 导入失败通常会在创建后端 context 或启动脚本时暴露；配置解析和大多数
单元测试不需要启动 Isaac。

## 快速开始

先生成或确认 capsule rope USD：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
```

只导入单臂场景、机器人、对象、controller 和 logger，不执行抓取：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py --no-grasp
```

打开 GUI 并运行单臂 pinch grasp demo：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py --gui
```

只校验双臂 scene runtime、cuMotion profile 和左右 robot profile 推导出的 cuMotion 语义，不启动 Isaac：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py --dry-run
```

导入左右 AR5+L6，并执行脚本内定义的双臂 cuMotion 动作：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py
```

启动双臂交互式 GUI runtime：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py \
  --gui --hold --interactive
```

启动后看到 `DUAL_ARM_INTERACTIVE_READY`，即可通过 stdin 输入 JSON motion，例如：

```json
{"type":"ik_offset","side":"left","offset":[0.02,0.0,0.02],"duration_s":1.0}
```

## 运行入口

| 模式 | 入口 | 启动 Isaac | 需要 cuMotion | 主要用途 |
| --- | --- | --- | --- | --- |
| 生成绳体 USD | `scripts/build_capsule_rope_asset.py` | 是，headless | 否 | 根据 `tools/assets/configs/capsule_rope.yaml` 写 USD/PhysX schema |
| 单臂导入保持 | `scripts/pinch_grasp.py --no-grasp` | 是 | 否 | 验证 env、objects、AR5+L6、controller 和 logging 基础链路 |
| 单臂 pinch grasp | `scripts/pinch_grasp.py` | 是 | 是 | 完整 TCP、IK、motion planner、trajectory、execution 流程 |
| 双臂 motion dry-run | `scripts/dual_arm_motion_test.py --dry-run` | 否 | 否 | 校验 scene runtime、cuMotion 和左右 robot profile 的规划语义 |
| 双臂 scripted motion | `scripts/dual_arm_motion_test.py` | 是 | 是 | 执行脚本内 Python 参数定义的 IK、TCP line、TCP arc 和 C-space planner 动作 |
| 双臂交互 motion | `scripts/dual_arm_motion_test.py --interactive` | 是 | 是 | 长生命周期 runtime，按 JSON 命令串行执行 motion |

`pinch_grasp.py` 常用参数：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --env scene1 \
  --cumotion-profile default \
  --logging-profile default_logger \
  --control-mode position \
  --grasp-world 0.025 -0.55 0.08 \
  --gui
```

`dual_arm_motion_test.py` 常用参数：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py \
  --env scene2 \
  --cumotion-profile default \
  --control-mode position \
  --gui --hold
```

`build_capsule_rope_asset.py` 常用参数：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/build_capsule_rope_asset.py \
  --config tools/assets/configs/capsule_rope.yaml \
  --output assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

## 配置 Profile

配置 profile 通过稳定名称加载，而不是任意路径。`load_profile_yaml(group, name)` 会解析到：

```text
configs/<group>/<name>.yaml
```

当前支持的 group：

| group | 目录 | 用途 |
| --- | --- | --- |
| `robot` | `configs/robots/` | 单个 Isaac articulation 的资产、受控关节、物理覆盖和 cuMotion 模型资源 |
| `env` | `configs/envs/` | World 设置、scene solver、机器人实例、对象实例和 root pose |
| `object` | `configs/objects/` | 已有 USD/URDF 对象的运行时 profile |
| `cumotion` | `configs/cumotion/` | IK/FK 和 motion planner 算法 profile |
| `logging` | `configs/logging/` | CSV 日志输出和记录列 |

每个配置目录都有可复制改名的 `example.yaml`。新增实验配置时，推荐复制示例或现有 profile，
改文件名后通过命令行传 profile 名：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --env my_scene \
  --cumotion-profile my_planner \
  --logging-profile my_logger
```

配置合并规则是递归 mapping merge。只有两边都是 mapping 时才继续递归；列表和标量会被覆盖值
整体替换。这对关节列表、轨迹点、手型目标和 solver 参数更安全。

### 配置边界

- `configs/robots/*.yaml`：描述一个 Isaac articulation。包含 `robot.asset_type`、
  `robot.asset_path`、`robot.prim_path`、导入碰撞近似、机器人刚体重力策略、材料/阻尼、solver
  iteration、可选 `controlled_joints`，以及 cuMotion 单臂 `xrdf_path`、`urdf_path`、
  `flange_frame`。这里不放 IK/planner 算法参数，也不放抓取动作参数。
- `configs/envs/*.yaml`：描述 scene。包含 World 物理步频、渲染步频、重力、是否添加默认地面、
  scene solver type、GUI camera/lights、`robots.single` 或 `robots.dual.left/right` 的 profile
  引用和安装位姿，以及 `objects[]` 对象实例摆放。
- `configs/objects/*.yaml`：描述运行时对象。包含对象类别、来源、资产路径、stage prim path、
  importer 参数、接触材质和对象级 solver 覆盖。对象在世界中的 `root_pose` 仍放在 env
  `objects[]`。
- `tools/assets/configs/*.yaml`：描述资产生成固有属性。例如 capsule rope 的段数、长度、质量、
  阻尼、关节限制、碰撞过滤和可视颜色。
- `configs/cumotion/*.yaml`：描述 cuMotion 算法参数，包括 `kinematics.ik` 和
  `motion_planner`。robot 模型资源仍放在 robot profile。
- `configs/controllers/*.yaml`：描述 position/velocity/effort 模式、implicit/explicit 方法、
  stiffness、damping、max force/effort limit 和 follower drive 参数。
- `configs/logging/*.yaml`：描述 CSV 是否启用、输出路径、flush 周期、采样降频和记录列。
- `scripts/pinch_grasp.py`：保存 pinch grasp 的动作目标、阶段顺序、默认预夹/闭合手型和抓取策略。

## 场景、对象和机器人

`configs/envs/scene1.yaml` 是默认单臂抓绳场景。它通过 `robots.single.robot_profile:
ar5v2_l6v1_l` 导入左侧 AR5+L6，并通过 `objects[]` 引用 workstation 和 capsule rope。

`configs/envs/scene2.yaml` 是默认双臂运动测试场景。它通过
`robots.dual.left/right.robot_profile` 分别引用 `ar5v2_l6v1_l` 和 `ar5v2_l6v1_r`，并用
`root_pose` 描述左右安装位姿。

env 中的对象实例只允许这些字段：

```yaml
objects:
  - name: rope
    object_profile: capsule_rope
    runtime_handle: rope
    root_pose:
      xyz: [0.1, -0.55, -0.4]
      rpy: [0.0, 0.0, 1.5707]
```

对象资产路径、静态/动态属性、导入参数和接触材质写在 `configs/objects/*.yaml`。这种拆分让同一个
对象 profile 可以在多个 scene 里用不同 root pose 实例化。

机器人 root pose 也只写在 env，不写在 robot profile：

```yaml
robots:
  single:
    robot_profile: ar5v2_l6v1_l
    root_pose:
      xyz: [0.0, 0.09, 0.0]
      rpy: [-1.5707, 0.0, 0.0]
```

这种方式可以把桌面、工装等静态环境物体和机器人 articulation 分开导入，避免把桌面和机械臂合并
到同一个 URDF 后引入 fixed joint、base link 和 importer fixed-base 语义混淆。相关风险见
`docs/known_risks.md`。

## Capsule Rope

绳体分为“生成配置”和“运行时对象配置”两层。

`tools/assets/configs/capsule_rope.yaml` 描述 USD 资产固有属性：

```yaml
object:
  name: capsuleropeV1_default
  asset_path: assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
  root_path: /CapsuleRope

rope:
  segments: 12
  length: 0.75
  radius: null
  center: [0.0, 0.0, 0.0]
  total_mass: 0.2
  shape: capsule
```

修改段数、长度、质量、阻尼、关节限制、端块尺寸或可视颜色后，需要重新生成 USD：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
```

`configs/objects/capsule_rope.yaml` 描述运行时如何引用这个 USD，以及接触材质和 solver iteration
覆盖：

```yaml
object:
  name: capsuleropeV1_default
  kind: dynamic_chain
  source: usd
  asset_path: assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
  prim_path: /World/CapsuleRope
  root_path: /CapsuleRope
  physics:
    material:
      static_friction: 0.7
      dynamic_friction: 0.5
      restitution: 0.0
      friction_combine_mode: average
    solver_position_iterations: 48
    solver_velocity_iterations: 4
```

运行 pinch grasp 或双臂 motion 时只引用已经生成好的 USD，不会每次重新生成绳体资产。

## 单臂 Pinch Grasp

`scripts/pinch_grasp.py` 是当前最完整的单臂运行入口。它通过
`create_single_robot_runtime(...)` 创建 Isaac runtime，然后执行脚本内置的 pinch grasp 动作。

简化调用链：

```text
pinch_grasp.py
  create_single_robot_runtime(...)
    load_profile_yaml(cumotion/env/robot/logging)
    merged_robot_config_with_cumotion_profile(...)
    EnvRuntimeSettings.from_env_config(...)
    create_simulation_session(...)
    runtime_objects_from_env_config(...) + add_runtime_objects(...)
    import_execution_robot_to_stage(...)
    world.reset()
    finalize_robot_controller(...)
    JointTrackingLogger(...)
  run_pinch_grasp_action(...)
    make_pinch_tcp_transform(...)
    make_cumotion_context(...)
    IK / graph_search / specified_path / trajectory_generation
    command_trajectory_from_arm_trajectory(...)
    execution.steps.*Step.run(...)
```

典型执行顺序：

1. 加载 cuMotion、env、robot、controller、object 和 logging profile。
2. 将 cuMotion profile 的默认值合入 robot config；robot config 提供模型资源字段。
3. 启动 Isaac `SimulationApp`，创建 World、stage、physics/render dt、camera 和 lights。
4. 根据 env `objects[]` 导入 workstation、capsule rope 等运行时对象。
5. 导入 AR5+L6 组合 MJCF，并在 `world.reset()` 前应用 root pose、drive seed、摩擦、材料、
   solver iteration 和机器人重力策略。
6. `world.reset()` 后清零速度，创建 `JointController`，按 `--control-mode` 选择 position、
   velocity 或 effort 主动控制。
7. 根据闭合手型和 MJCF body 链计算 thumb/index pinch TCP 相对 flange 的 fixed transform。
8. 通过 `make_cumotion_context(...)` 写临时 URDF，把 pinch TCP 作为 fixed link 注入 cuMotion。
9. 使用 cuMotion 求解 approach、grasp、lift、wiggle 等阶段的 IK 和运动轨迹。
10. 把 cuMotion C-space 结果按关节名映射回 controller command-space。
11. `execution.steps` 按 physics step 播放 command-space 目标或轨迹，并写 CSV 日志。

动作脚本不会在执行过程中实时做 task-space conversion。所有 IK、指定路径转换和 trajectory
generation 都在规划阶段完成；执行阶段只播放已经采样好的 `JointTrajectory`。

## 双臂和交互运动

双臂实现把规划模型和执行模型分开：

- Isaac stage 中导入左右两个 AR5+L6 articulation。
- cuMotion 侧根据左右单臂 URDF/XRDF 和 env `root_pose` 生成缓存的融合 URDF/XRDF。
- 融合模型的 arm C-space 是 14 个关节：`left_arm_7 + right_arm_7`。
- 手部 DOF 不进入 cuMotion C-space；手部目标由 command-space move 或 overlay 处理。

双臂配置链路：

```text
configs/envs/scene2.yaml
  robots.dual.left/right.robot_profile + root_pose
configs/robots/ar5v2_l6v1_l.yaml
configs/robots/ar5v2_l6v1_r.yaml
  Isaac asset + cuMotion single-arm URDF/XRDF/flange resource
configs/cumotion/default.yaml
  IK/planner algorithm defaults
```

双臂 selected-side 规划需要的左右 arm C-space 分区来自左右 robot profile 的
`cumotion.xrdf_path` 中的 `cspace.joint_names`；临时 TCP 的 parent frame 来自同一 robot
profile 的 `cumotion.flange_frame`。动作脚本默认 TCP 名由入口层构造的 `DualArmTcpSpec`
提供，不再有单独的 dual-arm profile。

`scripts/dual_arm_motion_test.py` 内置了一个测试用 `DualArmTcpSpec` 和一组 motion：

- 左侧 IK offset。
- 左侧 task-space line specified path。
- 右侧 task-space arc specified path。
- 右侧 C-space delta planner。

`--interactive` 会复用同一个双臂 runtime 和长生命周期 cuMotion execution session，通过 JSON
命令队列串行执行 motion。支持的 transport：

- stdin JSONL：默认启用，每行一个 JSON object。
- TCP JSONL：`--tcp-jsonl-host 127.0.0.1 --tcp-jsonl-port 8765`。
- WebSocket JSON：`--websocket-host 127.0.0.1 --websocket-port 8766`，需要安装 `websockets`。

TCP JSONL 示例：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py \
  --gui --hold --interactive \
  --tcp-jsonl-host 127.0.0.1 \
  --tcp-jsonl-port 8765
```

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 8765
printf '%s\n' '{"type":"ik_offset","side":"left","offset":[0.03,0,0.02],"duration_s":1.0}' | nc 127.0.0.1 8765
```

常用 JSON motion 类型：

| type | 作用 |
| --- | --- |
| `ik_offset` | 从当前 TCP pose 出发，对 TCP 位置加相对位移并求 IK |
| `ik_pose` | 指定 TCP 绝对目标位置和可选姿态 |
| `cspace_goal` | 把选定侧 arm joints 规划到绝对关节角目标 |
| `cspace_delta` | 在选定侧当前 arm joints 上叠加关节增量并规划 |
| `task_space_line` | 执行 TCP 直线 specified path |
| `task_space_arc` | 执行 TCP 圆弧 specified path |
| `hand` | 执行单手 command-space 目标 |
| `dual_hand` | 同步执行左右手 command-space 目标 |
| `hold` | 保持当前姿态一段时间 |
| `status` / `cancel` / `cancel_current` / `estop` / `quit` | 查询、取消、急停和退出 |

详细 JSON 字段、返回事件、批量命令和 overlay 示例见 `docs/interactive_simulation_usage.md`。

当前不声称双臂自动免碰撞。融合 XRDF 保留 14-DOF C-space 和后验检查入口，但真正可靠的避碰还
需要有效 collision spheres、正确 self-collision mask、collision-aware IK/planner，以及轨迹采样后
的碰撞检查。

## cuMotion 后端

`src/linkerbot_sim/backends/cumotion/` 是 cuMotion 的适配层。主要文件职责：

- `context.py`：加载 XRDF/URDF，创建 robot description、kinematics、collision world，并暴露
  `joint_names()` / `frame_names()`。
- `dual_urdf.py`：根据左右单臂资源和 env root pose 生成双臂 cuMotion URDF/XRDF 缓存。
- `forward_kinematics.py`：封装 FK 和 frame 查询。
- `inverse_kinematics.py`：封装单点 IK 和 collision-free IK。
- `motion_planner.py`：按 `MotionPlannerBackendConfig.planning_pipeline` 分发到具体 pipeline。
- `trajectory_optimizer_planner.py`：调用 cuMotion `TrajectoryOptimizer`。
- `graph_motion_planner.py`：调用 graph `MotionPlanner` 得到 C-space path，再生成 trajectory。
- `specified_path_planner.py`：消费调用方指定的 C-space/task-space/composite path，转换成
  C-space path 后生成 trajectory。
- `path_spec_adapter.py`：把项目 `TaskSpacePath`、`CSpaceWaypointPath`、`CompositePath` 转为
  cuMotion 官方 PathSpec。
- `trajectory_generation.py`：封装 `CSpaceTrajectoryGenerator`，对 joint path 做时间参数化。
- `trajectory_sampler.py`：把 cuMotion `Trajectory.eval_all(t)` 采样成项目 `JointTrajectory`。
- `tcp_context.py`：按需写临时 URDF，把调用方生成的 TCP 作为 fixed link 装进 cuMotion context。
- `tcp_urdf_builder.py`：纯 URDF 写入工具。

路径级规划统一返回 `planning.results.MotionResult`：

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
`joint_names()`、controller `command_joint_names` 和 Isaac `dof_names` 做名称映射。

## 轨迹和执行频率

项目内部 `JointTrajectory` 位于 `src/linkerbot_sim/trajectories/types.py`。它保存：

- `times`: shape `(N,)`
- `positions`: shape `(N, dof)`
- `velocities`: shape `(N, dof)`
- `accelerations`: shape `(N, dof)`
- `jerks`: shape `(N, dof)`
- `efforts`: shape `(N, dof)`
- `phases`: 每个采样点所属阶段
- `joint_names`: 每列对应的关节名

cuMotion 轨迹进入执行层通常经历三步：

1. cuMotion pipeline 返回后端 `Trajectory`。
2. `trajectory_sampler.joint_trajectory_from_cumotion(...)` 按 physics dt 或指定 sample dt 采样。
3. `command_trajectory_from_arm_trajectory(...)` 把 arm-only 轨迹嵌入 controller command-space。

当前 `execution.steps` 期望传入的 `JointTrajectory` 已经是“一行对应一个 physics step”的离散执行
矩阵，并且通常不包含首样本。执行层逐行播放，不再调用后端 `trajectory.eval_all(...)` 二次插值。

`trajectories/joint_trajectory_builder.py` 只用于从已有位置矩阵构造 `JointTrajectory`，并通过有限
差分补速度、加速度和 jerk。它不负责 cuMotion path conversion，也不负责运动规划。

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

主动关节支持：

- `position + implicit`：下发 position/velocity target，由 PhysX drive 计算力矩。
- `position + explicit`：Python 侧按位置/速度误差计算 effort。
- `velocity + implicit`：下发 velocity target，由 PhysX velocity drive 计算力矩。
- `velocity + explicit`：Python 侧按速度误差计算 effort。
- `effort + direct`：直接下发 effort command。

Follower 始终使用 Isaac position drive，不随主动控制模式改变。

## 物理和碰撞配置

机器人重力策略写在 robot YAML：

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
里的策略。

机器人材料、刚体阻尼和 solver iteration 也写在 robot YAML。controller YAML 只负责控制模式、
gain、限幅和 follower drive。

Isaac importer 层碰撞选项写在 `robot.import`：

- `convex_decomposition`：接触更贴合，成本更高。
- `convex_hull`：每个 mesh 一个凸包，更快但会填平凹陷。
- `self_collision`：是否开启机器人 articulation 内部 link 之间的 PhysX 自碰撞接触，默认
  `false`。

`collision_approximation` 只作用于 Isaac importer 把 MJCF/URDF mesh 写入 USD 的过程；
`self_collision` 只作用于 Isaac/PhysX 物理侧的 articulation 接触生成。它们都不改变
cuMotion XRDF/URDF 规划模型，也不表示运行时重新 cooking 碰撞体。更完整说明见
`docs/isaac_collision_approximation.md`。

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

关节跟踪日志使用 `configs/logging/default_logger.yaml`，默认输出到
`logs/joint_tracking/pinch_grasp.csv`。常用列名：

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

也可以覆盖输出路径和采样间隔：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --log logs/joint_tracking/my_run.csv \
  --log-interval-steps 1
```

Foxglove 输出位于 `src/linkerbot_sim/telemetry/foxglove.py`，支持离线 MCAP、本地 WebSocket live
server、`JointStates` 曲线和 `SceneUpdate` marker。

## 文档索引

- `docs/cumotion_interface.md`：cuMotion 后端接口、请求/结果、path conversion、trajectory adapter
  和当前封装评估。
- `docs/cumotion_motion_modes_examples.md`：不同 cuMotion motion mode 的示例和使用边界。
- `docs/interactive_simulation_usage.md`：双臂交互式 JSON 协议、transport、返回事件和命令示例。
- `docs/interactive_motion_runtime_plan.md`：交互式 motion runtime 的设计方案。
- `docs/isaac_collision_approximation.md`：Isaac importer 碰撞近似字段和 USD/PhysX 语义。
- `docs/known_risks.md`：已知风险，包括 URDF 静态环境物体 fixed-base/kinematic 叠加，以及桌面和
  机器人合并 URDF 的 fixed joint 风险。
- `ASSET_NAMING_CONVENTIONS.md`：资产、关节、link/body 和配置命名约定。

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

当前轻量测试重点覆盖 controller 配置、关节控制、mimic 解析、TCP frame、cuMotion context、IK、
motion planner、trajectory adapter、JointTrajectory builder、配置加载、pinch grasp 规划逻辑、
双臂 C-space/URDF、交互 runtime、日志和 Foxglove logger。它们不替代 Isaac GUI/物理接触验证，
但可以快速发现配置、数据结构和后端适配层的回归。
