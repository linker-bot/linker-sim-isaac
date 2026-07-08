# LinkerHand Simulation

语言版本：[中文](README.zh-CN.md) | [English](README.en.md)

这是一个基于 Isaac Sim / Isaac Lab 的机械臂、灵巧手和环境物体操作仿真工程。当前工程围绕
AR5 机械臂、LinkerHand L6 灵巧手、capsule/cuboid 近似绳体、T 形刚体块，以及 cuMotion 后端
运动生成展开，用于验证 TCP 定义、IK/FK、路径规划、mimic 关节同步、PhysX 参数、单臂交互、
双臂推块、双臂交互式运动和 tiled 并行环境流程。

项目主线遵循几个边界：

- `configs/` 随代码一起版本管理，保存默认 robot、env、object、controller、logging 和
  cuMotion profile。脚本可以依赖这些 schema，但具体资产路径、安装位姿、solver、增益和
  planner 参数应优先留在配置里。
- `scripts/` 是可直接运行的仿真/实验入口。动作目标、阶段顺序和临时 move 序列可以留在脚本里；
  TCP frame 的几何定义应放在 robot YAML 里，避免运行时改写机器人描述。
- `tools/object_assets/` 是静态/动态对象的离线资产生成入口。资产固有几何、质量、阻尼、关节
  限制和可视材质留在 tools 侧；运行时引用和物理覆盖留在 `configs/objects/`。
- `app/runtime/` 负责 Isaac app、World、stage、机器人导入、对象导入和 controller 装配。
- `app/motion/` 负责把客户侧 motion spec 转成 cuMotion 请求、轨迹和 command-space 执行。
- `app/interactive/` 负责 JSON 协议、队列和 stdin/TCP/WebSocket transport。
- `backends/cumotion/` 是唯一直接适配 cuMotion Python API 的层。
- `execution/` 只按 physics step 播放已经生成好的目标或轨迹，不做 IK 或重新规划。
- `controllers/` 把项目目标转换成 Isaac articulation action，并在每帧刷新 mimic follower。

## 当前能力

- 资产导入：机器人执行侧支持 MJCF/URDF，cuMotion 规划侧使用 URDF/XRDF，环境和动态对象支持
  USD/URDF 引用或导入。
- 单臂运行：支持交互式 hand/arm motion、YAML TCP、IK、规划、轨迹采样和 CSV 日志。
- 双臂运行：Isaac stage 中左右两个 AR5+L6 articulation 独立导入，cuMotion 侧融合成一个
  14-DOF arm C-space。
- 交互运动：双臂 runtime 支持 stdin JSONL、TCP JSONL 和 WebSocket JSON 提交运动命令。
- 环境对象：支持由 tools 配置生成 capsule/cuboid 刚体链 USD 和 T 形 compound rigid body USD，
  并由 env `objects[]` 引用到场景。
- cuMotion 后端：提供 FK、IK、collision-free IK、trajectory optimization、graph search、
  specified path、task-space path conversion 和 C-space trajectory generation。
- TCP：支持法兰 TCP、已有工具 TCP，以及 robot YAML `cumotion.custom_tcps` 声明的 fixed TCP。
- 轨迹语义：cuMotion `Trajectory` 会采样成项目 `JointTrajectory`；graph search 和
  specified path 成功时必须生成 trajectory，不能只返回离散 path。
- 控制器：支持 position、velocity 和 effort 控制；position/velocity 可选 Isaac implicit drive
  或 Python explicit effort。
- Mimic 关节：解析 MJCF `equality/joint` 的 `polycoef`，运行时按实际 master 状态刷新 follower
  目标。
- 日志和遥测：支持关节目标、实际位置/速度、command effort、action effort、PhysX
  measured/applied effort 的 CSV 记录；单臂、双臂和 tiled 交互 runtime 可选 Foxglove live
  server 或 MCAP 状态流。

## 目录结构

```text
.
├── assets/
│   ├── mesh/                 # arm/hand/env object mesh
│   ├── single_system/        # 单体机器人资产，例如 AR5V2_L、AR5V2_R、L6V1_L
│   ├── combined_system/      # 组合机器人资产，例如 AR5V2_L6V1_L/R
│   ├── rigid_env_objects/    # 刚体环境对象，例如 workstationV1_armbase/tablebase
│   └── flexible_env_objects/ # 柔性/链式对象资产，例如 capsuleropeV1_default
├── configs/
│   ├── controllers/          # arm/hand 控制模式、增益、限幅和 follower drive
│   ├── cumotion/             # cuMotion IK/planner profile
│   ├── envs/                 # scene、physics/render frequency、solver、机器人/对象实例摆放
│   ├── logging/              # 关节跟踪 CSV 日志配置
│   ├── objects/              # 运行时对象 profile，引用已有 USD/URDF
│   └── robots/               # Isaac 机器人资产、物理覆盖、cuMotion 资源
├── docs/                     # 文档入口；中文版在 zh-CN，英文版在 en
│   ├── README.md             # 文档语言入口
│   ├── zh-CN/                # 中文使用说明和接口文档
│   └── en/                   # English user and interface docs
├── scripts/                  # Isaac Sim 仿真/实验运行入口
├── src/linkerbot_sim/
│   ├── app/                  # runtime/motion/interactive 高层装配；interactive 下按 single_arm/dual_arm/tiled 拆分入口
│   ├── assets/               # 资产导入、USD/PhysX 覆盖、solver 设置
│   ├── backends/cumotion/    # cuMotion context、FK/IK、planner、path/trajectory adapter
│   ├── controllers/          # 控制器配置解析和 JointController
│   ├── envs/                 # World、viewport、灯光和 visual settings
│   ├── execution/            # 单臂/双臂目标和轨迹执行步骤
│   ├── logging/              # CSV logger 和 effort logger
│   ├── objects/              # rigid object 和 dynamic chain 运行时导入
│   ├── planning/             # 后端无关请求/结果、碰撞对象、双臂 C-space 分区
│   ├── robots/               # 关节组、mimic/equality、状态工具
│   ├── tcp/                  # TCP frame 相关工具
│   ├── telemetry/            # Foxglove、MCAP、WebSocket
│   ├── trajectories/         # JointTrajectory 容器、builder 和 command-space 轨迹组装
│   ├── utils/                # 配置、路径、旋转、数学、计时工具
│   └── visualization/        # GUI viewport 辅助
├── tests/                    # 尽量不启动 Isaac Sim 的轻量测试
├── tools/object_assets/      # 静态/动态对象离线资产生成工具，例如 rope/T block USD builder
├── README.md                 # 项目语言入口
├── README.en.md              # English README
├── README.zh-CN.md           # 中文 README
└── pyproject.toml
```

## 环境约定

示例命令默认从仓库根目录运行，并假设已经激活了包含 Isaac Sim、Isaac Lab、cuMotion 和项目
Python 依赖的环境：

```bash
PYTHONPATH=src python <command>
```

如果没有激活环境，可以把示例里的 `python` 替换成实际解释器路径，例如
`/path/to/venv/bin/python`。仓库不要求虚拟环境目录使用固定名称。

项目采用 src-layout。`scripts/single_arm_interactive.py`、`scripts/dual_arm_interactive.py`、`scripts/tiled_env_interactive.py`、
`tools/object_assets/flexible/rope/build_asset.py` 和 `tools/object_assets/rigid/tblock/build_asset.py`
会自行把 `src/` 放进 `sys.path`，但测试、交互片段和临时 Python 命令仍建议显式设置
`PYTHONPATH=src`。

完整依赖通过 `pyproject.toml` 管理。`simulation` extra 记录了当前代码期望的主要仿真依赖：

- `isaacsim[all]==5.1.0.0`
- `cumotion==1.1.0`
- `torch==2.7.0`

安装示例：

```bash
python -m pip install -e ".[dev,visualization,simulation]"
```

如果已有环境已经安装了 Isaac Sim、cuMotion 和 torch，也可以只安装缺失的普通 Python 依赖。
cuMotion/Isaac 导入失败通常会在创建后端 context 或启动脚本时暴露；配置解析和大多数单元测试
不需要启动 Isaac。

## 快速开始

先生成或确认环境物体 USD。`scene1` / `scene2` 使用 capsule rope；双臂默认 `scene3` 依赖 T block：

```bash
PYTHONPATH=src python tools/object_assets/flexible/rope/build_asset.py
PYTHONPATH=src python tools/object_assets/rigid/tblock/build_asset.py
```

启动单臂单手交互式 GUI runtime，单臂消息可省略 `side`：

```bash
PYTHONPATH=src python scripts/single_arm_interactive.py \
  --env scene1 \
  --gui \
  --foxglove-live-port 8765
```

启动双臂交互式 GUI runtime：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold
```

只校验双臂 motion 语义、cuMotion profile 和左右 robot profile 推导出的规划结构，不启动 Isaac：

```bash
PYTHONPATH=src python -m pytest tests/test_dual_arm_motion_test.py -q
```

启动 tiled 并行环境交互入口：

```bash
PYTHONPATH=src python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --default-decimation 1
```

启动双臂交互式 GUI runtime，并把实时状态发布给 Foxglove：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --state-rate-hz 60 \
  --state-include-objects
```

Foxglove Desktop 中选择 `Foxglove WebSocket`，连接 `ws://127.0.0.1:8766`。完整 topic、MCAP
和 effort 字段说明见 `docs/zh-CN/传感器与遥测/Foxglove 数据使用说明.md`。

启动后看到 `DUAL_ARM_INTERACTIVE_READY`，即可通过 stdin 输入 JSON motion，例如：

```json
{"type":"ik_offset","side":"left","offset":[0.02,0.0,0.02],"duration_s":1.0}
```

## 运行入口

| 模式 | 入口 | 启动 Isaac | 需要 cuMotion | 主要用途 |
| --- | --- | --- | --- | --- |
| 生成绳体 USD | `tools/object_assets/flexible/rope/build_asset.py` | 是，headless | 否 | 根据 `tools/object_assets/flexible/rope/config.yaml` 写 USD/PhysX schema |
| 生成 T 形块 USD | `tools/object_assets/rigid/tblock/build_asset.py` | 是，headless | 否 | 根据 `tools/object_assets/rigid/tblock/config.yaml` 写 USD/PhysX schema |
| 单臂交互 motion | `scripts/single_arm_interactive.py` | 是 | 是 | 长生命周期单 AR5+L6 runtime，按 JSON 命令串行执行 arm/hand motion |
| 双臂交互 motion | `scripts/dual_arm_interactive.py` | 是 | 是 | 长生命周期 runtime，按 JSON 命令串行执行 motion |
| Tiled 并行交互 | `scripts/tiled_env_interactive.py` | 是 | 可选 | 单 scene 多 env 的同步 command、trajectory buffer 和 async planner |
| 双臂 motion 语义测试 | `python -m pytest tests/test_dual_arm_motion_test.py -q` | 否 | 否 | 校验双臂 MoveSpec、TCP、specified path 和 C-space planner 数据语义 |

`single_arm_interactive.py` 常用参数：

```bash
PYTHONPATH=src python scripts/single_arm_interactive.py \
  --env scene1 \
  --cumotion-profile default \
  --logging-profile default_logger \
  --control-mode position \
  --gui
```

`dual_arm_interactive.py` 常用参数：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --env scene3 \
  --cumotion-profile default \
  --control-mode position \
  --gui --hold
```

`tiled_env_interactive.py` 常用参数：

```bash
PYTHONPATH=src python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --planner-backend linear \
  --tcp-jsonl-port 9003 \
  --hold
```

`flexible/rope/build_asset.py` 常用参数：

```bash
PYTHONPATH=src python tools/object_assets/flexible/rope/build_asset.py \
  --config tools/object_assets/flexible/rope/config.yaml \
  --output assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

`rigid/tblock/build_asset.py` 常用参数：

```bash
PYTHONPATH=src python tools/object_assets/rigid/tblock/build_asset.py \
  --config tools/object_assets/rigid/tblock/config.yaml \
  --output assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda
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
PYTHONPATH=src python scripts/single_arm_interactive.py \
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
  scene solver type、GUI viewport/lights、`robots.single` 或 `robots.dual.left/right` 的 profile
  引用和安装位姿，以及 `objects[]` 对象实例摆放。
- `configs/objects/*.yaml`：描述运行时对象。包含对象类别、来源、资产路径、stage prim path、
  importer 参数、接触材质和对象级 solver 覆盖。对象在世界中的 `root_pose` 仍放在 env
  `objects[]`。
- `tools/object_assets/<rigid|flexible>/<object>/`：每个对象一个生成工具文件夹，描述该对象资产的
  固有属性。例如 `flexible/rope` 保存 capsule rope 的段数、长度、质量、阻尼、关节限制、碰撞
  过滤和可视颜色；`rigid/tblock` 保存 T 形块的 cuboid 尺寸、质量、阻尼和可视颜色。
- `configs/cumotion/*.yaml`：描述 cuMotion 算法参数，包括 `kinematics.ik` 和
  `motion_planner`。robot 模型资源仍放在 robot profile。
- `configs/controllers/*.yaml`：描述 position/velocity/effort 模式、implicit/explicit 方法、
  stiffness、damping、max force/effort limit 和 follower drive 参数。
- `configs/logging/*.yaml`：描述 CSV 是否启用、输出路径、flush 周期、采样降频和记录列。
- `scripts/single_arm_interactive.py` / `scripts/dual_arm_interactive.py`：提供交互式 motion runtime，动作目标通过 JSON 或客户端程序提交。

## 场景、对象和机器人

`configs/envs/scene1.yaml` 是默认单臂抓绳场景。它通过 `robots.single.robot_profile:
ar5v2_l6v1_l` 导入左侧 AR5+L6，并通过 `objects[]` 引用 workstation 和 capsule rope。

当前内置 scene：

| scene | 入口默认 | 机器人 | 主要对象 | 用途 |
| --- | --- | --- | --- | --- |
| `scene1` | `single_arm_interactive.py` | 单左臂 AR5+L6 | workstation、capsule rope | 单臂交互和导入检查 |
| `scene2` | 手动选择 | 双 AR5+L6 | workstation、capsule rope | 双臂 rope 场景配置检查和动作测试 |
| `scene3` | `dual_arm_interactive.py` | 双 AR5+L6 | workstation、T block | 双臂推块/运动测试和交互 runtime |

双臂场景通过 `robots.dual.left/right.robot_profile` 分别引用 `ar5v2_l6v1_l` 和
`ar5v2_l6v1_r`，并用各自的 `root_pose` 描述左右安装位姿。双臂交互入口可通过
`scripts/dual_arm_interactive.py --env scene3` 读取 T block 场景；如需回到双臂 rope 场景，可显式传 `--env scene2`。

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
`docs/zh-CN/风险与约束/已知风险与设计约束.md`。

## Capsule Rope

绳体分为“生成配置”和“运行时对象配置”两层。

`tools/object_assets/flexible/rope/config.yaml` 描述 USD 资产固有属性：

```yaml
object:
  name: capsuleropeV1_default
  asset_path: assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
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
PYTHONPATH=src python tools/object_assets/flexible/rope/build_asset.py
```

`configs/objects/capsule_rope.yaml` 描述运行时如何引用这个 USD，以及接触材质和 solver iteration
覆盖：

```yaml
object:
  name: capsuleropeV1_default
  kind: dynamic_chain
  source: usd
  asset_path: assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
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

运行单臂或双臂 motion 时只引用已经生成好的 USD，不会每次重新生成绳体资产。

## Rigid T Block

`tools/object_assets/rigid/tblock/config.yaml` 描述 T 形块 USD 资产固有属性，默认生成到
`assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda`。

生成出的模型是一个根刚体 `/TBlock`，下面包含 `stem` 和 `cap` 两个 cuboid collision/visual
子块。`tblock.total_mass`、`linear_damping` 和 `angular_damping` 写在根刚体上；cuboid 的尺寸、
局部偏移和颜色来自 tools 侧配置。

```bash
PYTHONPATH=src python tools/object_assets/rigid/tblock/build_asset.py
```

运行时引用由 `configs/objects/TblockV1_default.yaml` 管理。当前默认 `static: false`，也就是作为
可被推动的动态刚体导入；接触材质、stage `prim_path` 和是否静态冻结都属于运行时对象 profile。
`configs/envs/scene3.yaml` 通过 `objects[]` 引用该 profile，并设置当前场景里的 root pose：

```yaml
objects:
  - name: Tblock
    object_profile: TblockV1_default
    root_pose:
      xyz: [0.15, 0.0, 0.0]
      rpy: [0.0, 1.5707, 0.0]
```

生成出的 `TblockV1_default` 遵循 `docs/zh-CN/配置与命名/资产命名规范.md` 的环境对象命名规则。修改
`tools/object_assets/rigid/tblock/config.yaml` 后需要重新运行 `build_asset.py`，运行脚本不会自动
刷新 `.usda`。

## 单臂交互运行

`scripts/single_arm_interactive.py` 是当前单臂 AR5+L6 的交互入口。它通过
`create_single_robot_runtime(...)` 创建 Isaac runtime，并通过 JSON 命令串行执行 arm/hand motion。

简化调用链：

```text
single_arm_interactive.py
  create_single_robot_runtime(...)
    load_profile_yaml(cumotion/env/robot/logging)
    EnvRuntimeSettings.from_env_config(...)
    create_simulation_session(...)
    runtime_objects_from_env_config(...) + add_runtime_objects(...)
    import_execution_robot_to_stage(...)
    world.reset()
    finalize_robot_controller(...)
  run_interactive_single_arm_motion(...)
    stdin/TCP/WebSocket transport
    InteractiveMotionQueue
    SingleArmCuMotionExecutionSession
    execution.steps.*Step.run(...)
```

常用命令：

```bash
PYTHONPATH=src python scripts/single_arm_interactive.py \
  --env scene1 \
  --cumotion-profile default \
  --logging-profile default_logger \
  --control-mode position \
  --gui \
  --foxglove-live-port 8765
```

启动后看到 `SINGLE_ARM_INTERACTIVE_READY`，即可发送 `hand`、`cspace_goal`、`cspace_delta`、
`ik_offset`、`ik_pose`、`task_space_line`、`task_space_arc` 等 JSON motion。详细协议见
`docs/zh-CN/交互与运行/交互式仿真使用说明.md`。

## 双臂和交互运动

双臂实现把规划模型和执行模型分开：

- Isaac stage 中导入左右两个 AR5+L6 articulation。
- cuMotion 侧根据左右单臂 URDF/XRDF 和 env `root_pose` 生成缓存的融合 URDF/XRDF。
- 融合模型的 arm C-space 是 14 个关节：`left_arm_7 + right_arm_7`。
- 手部 DOF 不进入 cuMotion C-space；手部目标由 command-space move 或 overlay 处理。

双臂配置链路：

```text
configs/envs/scene3.yaml
  robots.dual.left/right.robot_profile + root_pose
  objects[] -> workstation + TblockV1_default
configs/robots/ar5v2_l6v1_l.yaml
configs/robots/ar5v2_l6v1_r.yaml
  Isaac asset + cuMotion single-arm URDF/XRDF/flange resource
configs/objects/TblockV1_default.yaml
  T block USD runtime reference + material/static settings
configs/cumotion/default.yaml
  IK/planner algorithm defaults
```

双臂 selected-side 规划需要的左右 arm C-space 分区来自左右 robot profile 的
`cumotion.xrdf_path` 中的 `cspace.joint_names`；默认 TCP 和 custom TCP 来自同一 robot
profile 的 `cumotion.default_tcp_frame/custom_tcps`，并在双臂融合 cuMotion 配置中合并。
动作脚本不再传入 TCP 位姿，也不再维护单独的 dual-arm TCP profile。

`tests/test_dual_arm_motion_test.py` 覆盖了一组双臂 motion 语义：

- 左侧 IK offset。
- 左侧 task-space line specified path。
- 右侧 task-space arc specified path。
- 右侧 C-space delta planner。

这些动作主要验证双臂融合 cuMotion C-space、TCP frame 选择、轨迹拆分和左右 controller 执行链路；
不是完整的物体抓取策略。默认 `scene3` 中的 T block 用于检查刚体对象导入、接触材质和双臂场景
装配，具体推块策略仍由后续 motion 脚本或交互命令定义。

`scripts/dual_arm_interactive.py` 会复用同一个双臂 runtime 和长生命周期 cuMotion execution session，通过 JSON
命令队列串行执行 motion。支持的 transport：

- stdin JSONL：默认启用，每行一个 JSON object。
- TCP JSONL：`--tcp-jsonl-host 127.0.0.1 --tcp-jsonl-port 9001`。
- WebSocket JSON：`--websocket-host 127.0.0.1 --websocket-port 9002`，需要安装 `websockets`。

TCP JSONL 示例：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --tcp-jsonl-host 127.0.0.1 \
  --tcp-jsonl-port 9001
```

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 9001
printf '%s\n' '{"type":"ik_offset","side":"left","offset":[0.03,0,0.02],"duration_s":1.0}' | nc 127.0.0.1 9001
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

详细 JSON 字段、返回事件、批量命令和 overlay 示例见 `docs/zh-CN/交互与运行/交互式仿真使用说明.md`。

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
`docs/zh-CN/资产与场景/Isaac 碰撞近似配置.md`。

## 坐标、单位和命名

- 距离单位为 m，质量单位为 kg，时间单位为 s。
- 所有关节角和 YAML 中的 RPY 使用 rad。
- 项目边界统一使用 wxyz 四元数，即 `[w, x, y, z]`。
- SciPy 内部使用 xyzw，转换封装在 `utils/rotations.py`。
- 配置中的 RPY 使用固定轴 XYZ 顺序。
- cuMotion C-space 顺序由 XRDF/URDF 决定，不能假设等于 Isaac articulation DOF 顺序。
- 正式资产、关节、link/body、配置引用不使用连字符 `-`，避免 Isaac importer 自动改名。

资产命名细节见 `docs/zh-CN/配置与命名/资产命名规范.md`。

## 日志和调试

关节跟踪日志由 `configs/logging/default_logger.yaml` 控制。常用列名：

- `qd_*` / `q_*`：命令位置和实际位置。
- `vd_*` / `v_*`：命令速度和实际速度。
- `tau_cmd_*`：语义 effort command；implicit drive 下通常为 `nan`。
- `tau_action_*`：控制器实际下发给 Isaac 的 effort action。
- `tau_measured_*`：PhysX measured effort。
- `tau_applied_*`：Isaac applied effort。

默认不记录读取成本较高的 measured/applied effort。需要调整输出路径、采样间隔或 effort 列时，复制并修改 `configs/logging/default_logger.yaml`，再通过 `--logging-profile <name>` 选择新的 profile。

Foxglove 状态流位于 `src/linkerbot_sim/telemetry/` 和 `src/linkerbot_sim/app/interactive/state_stream.py`，
支持本地 live server、离线 MCAP、`JointStates` 曲线、`SceneUpdate` marker，以及
`/linkerbot/state` JSON 完整快照。使用方式和数据结构见 `docs/zh-CN/传感器与遥测/Foxglove 数据使用说明.md`。

## 文档索引

完整入口见 `docs/zh-CN/文档索引.md`。当前按主题分为：

- `docs/zh-CN/配置与命名/`：配置 profile 使用、资产和关节命名规范。
- `docs/zh-CN/交互与运行/`：单臂、双臂和 Tiled 交互入口、JSON 协议和实时状态流使用说明。
- `docs/zh-CN/并行环境/`：Tiled 并行环境使用方式、指令格式、trajectory buffer 和 async planner 接口。
- `docs/zh-CN/运动规划/`：cuMotion 后端接口和运动模式示例。
- `docs/zh-CN/传感器与遥测/`：Foxglove live/MCAP、状态流、相机图像和 sensor camera 设置。
- `docs/zh-CN/资产与场景/`：物体资产生成、Isaac 碰撞近似和 USD 资产预览。
- `docs/zh-CN/风险与约束/`：已知风险和设计约束。

## 验证

语法检查：

```bash
PYTHONPATH=src python -m compileall -q src scripts tests
```

运行轻量测试：

```bash
PYTHONPATH=src python -m pytest -q tests
```

检查 YAML 是否可解析：

```bash
PYTHONPATH=src python - <<'PY'
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
motion planner、trajectory adapter、JointTrajectory builder、配置加载、单臂 cuMotion 规划逻辑、
双臂 C-space/URDF、交互 runtime、日志和 Foxglove logger。它们不替代 Isaac GUI/物理接触验证，
但可以快速发现配置、数据结构和后端适配层的回归。
