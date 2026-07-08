# Isaac Lab 风格 Tiled Envs 改造计划书

**目标仓库:** `linkerhand/simulation`
**编写日期:** 2026-07-07
**目标读者:** 后续负责实际修改代码的大模型/工程师
**修改范围:** 新增独立的同步批量 step-control 仿真能力, 不破坏现有单臂、双臂、交互、cuMotion 工作流

## 0. 一句话目标

把当前“单 Kit 进程内一个场景实例”的 Isaac Sim runtime, 扩展为 Isaac Lab 风格的 tiled envs, 但把它定位成独立的同步批量 step-control runtime:

- 一个 `SimulationApp` / 一个 `World` / 一个 PhysX scene。
- `/World/envs/env_0 ... /World/envs/env_{N-1}` 下放置 N 份相同任务实例。
- 每个 env 有独立机器人 articulation、物体、初始状态和 episode 状态。
- 每个 physics step 先批量写入全部 env 的 action, 再只调用一次 `world.step()`。
- action 必须在固定 control step 内转成 batched joint target; 异步规划只允许作为外围 producer 输出 trajectory, 不进入 `TiledCommandAdapter` 或变长 MoveSpec 调度。
- 支持 headless 下单 GPU `num_envs=256+` 的目标路径。

本计划重点是改造路线, 不是立即保证 `256+` 对所有复杂场景都达标。AR5 + LinkerHand + 接触物体的成本较高, 应按 `4 -> 16 -> 64 -> 256` 分阶段验收。

## 0.1 本轮落地状态

当前分支已经先落地了一版真实 Isaac/PhysX tiled scene、实时并行脚本和 tiled 交互入口:

- `src/linkerbot_sim/tiled/scene.py`: 构建 `/World/envs/env_i` tiled scene。流程是导入 `env_0`、使用 `GridCloner` clone、按需创建 env 间 collision filtering、创建 batched `Articulation` view，并在 `world.reset()` 后解析 command joints。
- `scripts/tiled_env_realtime.py`: 启动真实 `SimulationApp` / `World`，构建 tiled scene，并用 batched articulation action 对多个 env 同步下发简单关节正弦目标。它只验证同步 step-control，不接路径规划。
- `scripts/tiled_env_interactive.py` / `src/linkerbot_sim/app/interactive/tiled.py`: 脚本保留薄 CLI 入口，主体交互 runtime 已迁入 `app/interactive`。正式 CLI 只启动真实 Isaac tiled scene，用 batched `Articulation` view 支持 `status/reset/step/get_state/set_state/quit`，并支持 `env_ids` 与 `robots` 裁剪。交互层已支持 `load_trajectory` / `step_trajectory` / `plan` / `planner_status` / `cancel_plan` / `clear_completed` 的外围轨迹和异步关节规划流程。Isaac runtime 已接入 `BatchedCuMotionIKSolver`，支持 `ee_pose_target` / `ee_delta_pos` / `ee_delta_pose` 的真实 batched IK 路径。纯 Python debug runtime 已降级为内部测试替身，不再作为用户可见 backend。
- `src/linkerbot_sim/telemetry/tiled.py`: 提供 tiled interactive 的轻量 Foxglove/MCAP sink。未传 `--foxglove-*` 时不会创建 sink；启用后发布 `/tiled/state` JSON，并为第一个 selected env 发布 `/tiled/env_XXX/joint_states` 标准 JointStates 和 `/tiled/env_XXX/scene` object/TCP marker。旧路径 `src/linkerbot_sim/tiled/telemetry.py` 已删除。
- `configs/envs/scene3_tiled/`: 目录型 tiled env profile 示例。`base.yaml` 保存共通 World/solver/camera/light/robot/object 集合，`envs/env_XXX.yaml` 只保存每个 env 的对象初始位姿差异。当前 AR5+LinkerHand MJCF fixed-base 示例默认 `replicate_physics=false`，优先保证 root joint anchor 的物理语义正确。
- `load_profile_yaml("env", name)`: 已兼容旧的 `configs/envs/<name>.yaml` 和新的 `configs/envs/<name>/base.yaml`。
- 旧导入路径 `linkerbot_sim.tiled.cumotion` 和 `linkerbot_sim.tiled.telemetry` 已删除；cuMotion tiled IK 使用 `linkerbot_sim.backends.cumotion.tiled_ik`，telemetry 使用 `linkerbot_sim.telemetry.tiled`。

当前能力边界:

- 支持 `robots.single` 和 `robots.dual.left/right`。
- 支持同构物体集合，也就是所有 env 拥有相同 object name/profile，只允许 per-env 覆盖已有对象的 `root_pose`。
- 不支持每个 env 拥有不同物体集合；如确需异构，应按 variant 分组后分别 tiled。
- 不支持在 `TiledCommandAdapter` 内做路径规划、异步队列或变长轨迹调度。
- 支持在 `TiledCommandAdapter` 外通过 `TiledPlannerManager` 生成关节空间 trajectory, 再由 trajectory buffer 在同步 command boundary 回放。
- `replicate_physics=true` 仍是 256+ 性能目标路径，但当前 MJCF fixed-base root joint 与 PhysX replication 不兼容；builder 会检测 `body0` 为空的 `rootJoint_*` 并自动关闭 replication。实时脚本仍提供 `--replicate-physics` / `--no-replicate-physics`，但最终 ready 行会打印 effective `replicate_physics`。
- tiled 交互模式已经有 debug/Isaac 两种脚本 backend，但它仍是轻量调试入口，还不是 `src/linkerbot_sim/tiled/runtime.py` 下的正式 runtime API。
- tiled Foxglove/MCAP 已接入交互脚本的 selected-env 调试路径；正式 tiled runtime、benchmark 和 evaluator API 尚未统一接入 telemetry。

## 1. 术语和语义约定

### 1.1 tiled envs

`tiled envs` 指在同一个 USD stage 中复制多个相同环境, 并通过空间偏移分隔。例如:

```text
/World/envs/env_0/AR5V2_L6V1_L
/World/envs/env_0/TBlock
/World/envs/env_1/AR5V2_L6V1_L
/World/envs/env_1/TBlock
...
```

每个 env 的局部坐标一致, 但 env root 有不同世界坐标偏移。策略和任务逻辑应尽量使用 env-local 语义; 只有写入 USD/PhysX 时才加上 `env_origin`。

### 1.2 一次 tick 全部推进

正确循环:

```python
batched_left_controller.apply_targets(left_targets)    # shape: (num_envs, dof)
batched_right_controller.apply_targets(right_targets)  # shape: (num_envs, dof)
world.step(render=False)
obs = batched_state_reader.read()
```

错误循环:

```python
for env_id in range(num_envs):
    controller[env_id].apply_targets(targets[env_id])
    world.step(render=False)  # 错: 第 0 个 env 已推进, 第 1 个 env 还没写 action
```

后续实现时必须保持“所有 action 写完 -> 一次 step -> 所有 state 一起读”的时序。

### 1.3 单 env runtime 与 tiled runtime 的关系

不要把现有 `SingleRobotRuntime` / `DualRobotAppRuntime` 直接硬改成 tiled 版本。新增并行 runtime, 保持旧脚本稳定:

- 旧 single-env runtime 继续服务 `scripts/pinch_grasp.py`、`scripts/dual_arm_interactive.py`、交互 JSON 协议和现有测试。
- 旧路径继续承载 cuMotion `MotionPlanner`、`MoveSpec`、交互队列、cancel/estop 等复杂异步工作流。
- 新路径只服务同步批量 rollout、MPC 或 RL 风格策略中的 step-based 控制。
- 新路径不做“并行版完整 motion runtime”; 它只把简单 action 同步转换成 batched joint target 并推进物理。

### 1.4 同步 command step

tiled runtime 的上层动作以 command step 为单位。每个 command step 固定包含:

```text
读取 batched state -> 计算 batched joint target -> 插值/保持固定 decimation 个 physics tick -> 读取 batched state
```

第一版只支持以下同步动作:

- `joint_position_target`: 直接给每个 env 的 command-space 关节目标。
- `joint_delta_pos`: 在当前 command-space 关节角上叠加增量。
- `ee_pose_target`: 末端 TCP 绝对目标位姿, 经批量 IK 得到关节目标。
- `ee_delta_pos`: 末端 TCP 平移微动, 保持当前 TCP 姿态, 经批量 IK 得到关节目标。
- `ee_delta_pose`: 末端 TCP 位姿微动, 经批量 IK 得到关节目标。
- `hold`: 保持当前或上一个 joint target。

明确不支持:

- 在 `TiledCommandAdapter` 热路径内执行 graph search、trajectory optimization、specified path planning。
- 把 planner request 当成 action 直接塞进 `step()`。
- `MoveSpec`、旧交互 JSON motion queue、cancel_current/estop。
- 每个 env 单独规划再以不同步数推进。

外围已支持:

- `load_trajectory` / `step_trajectory`: 离线或后台规划结果回放。
- `plan` / `planner_status` / `cancel_plan`: 外部 async planner manager 的关节空间规划调度。
- 默认 planner backend 是 linear joint trajectory；collision-aware cuMotion planner 需通过独立 backend/factory 装配。

## 2. 当前架构关键事实

下面是本计划基于当前代码读到的事实, 后续大模型修改前应先再次确认。

### 2.1 启动和基础 world

- `src/linkerbot_sim/app/runtime/simulation_session.py`
  - `create_simulation_session()` 启动单个 `SimulationApp`。
  - 创建单个 Isaac `World`。
  - 当前只暴露 `SingleArticulation` 类型和 `ArticulationAction` 类型。

- `src/linkerbot_sim/envs/scene_builder.py`
  - `build_world()` 配置 physics/render dt、重力、默认地面。
  - tiled envs 可以复用该 world 创建逻辑。

### 2.2 单/双机器人 runtime

- `src/linkerbot_sim/app/runtime/single_robot.py`
  - 按 env YAML 导入一台机器人和对象。
  - reset 后创建 `JointController`。
  - `ExecutionRuntime` 只持有一个 articulation。

- `src/linkerbot_sim/app/runtime/dual_robot.py`
  - 按 env YAML 导入左右两台机器人。
  - `DualRobotRuntime` 持有 left/right 两个 articulation。

- `src/linkerbot_sim/execution/dual_steps.py`
  - 已经实现“左右目标先下发, 再 `world.step()` 一次”的正确同步模式。
  - tiled runtime 应把这个模式推广到 `num_envs * sides`。

### 2.3 资产导入和 prim path

- `src/linkerbot_sim/assets/robot_loader.py`
  - `RobotAssetConfig.prim_path` 来自 robot profile, 当前是固定 `/World/...`。
  - `import_robot_asset()` 用该 `prim_path` 导入 MJCF/URDF。
  - `apply_root_pose()` 把 env YAML 的 root pose 写到导入 root prim。

- `src/linkerbot_sim/app/runtime/objects.py`
  - `RuntimeObjectConfig` 从 env object scene instance + object profile 合成。
  - object profile 也含固定 `prim_path`, 如 `/World/TBlock`。

固定 `/World/...` 是 tiled envs 最大结构性阻碍。所有 robot/object path 都必须支持 env-local namespace。

### 2.4 控制器

- `src/linkerbot_sim/controllers/joint_controller.py`
  - 当前 `JointController` 面向单个 articulation。
  - 目标数组是 `(num_dof,)`。
  - mimic follower 由 MJCF equality 解析后在每帧根据 master 状态展开。

tiled 版本不能创建 256 个 Python controller 然后逐个 apply。短期可以作为 smoke fallback, 但目标实现必须是 batched controller, 用 `(num_envs, num_dof)` 数组一次写入 articulation view。

### 2.5 本地 Isaac Sim 能力

当前 venv 有 Isaac Sim 5.1, 未发现独立 `isaaclab` 包。可使用 Isaac Sim 原生扩展:

- `isaacsim.core.cloner.GridCloner`
- `isaacsim.core.prims.Articulation`

`Articulation` view 支持按 regex 包装多个 articulation, 状态和 action API 形状为 `(num_envs, num_dof)`。因此第一阶段不需要引入完整 Isaac Lab, 可以先做“Isaac Lab 风格”的本地批量层。

## 3. 总体改造策略

### 3.1 推荐路线

采用“现有 Isaac Sim runtime + 原生 Cloner/View”的路线:

1. 先做最小 spike, 证明本项目资产可被 `GridCloner` 和 `Articulation` view 批量包装。
2. 抽出 env namespace/path rewrite 层。
3. 新增 tiled scene builder, 只构建 `env_0`, 再 clone。
4. 新增 batched articulation runtime 和 controller。
5. 新增 batched reset/state/action API。
6. 新增同步 step-control action 层, 支持关节目标/关节增量/末端微动 IK。
7. 后续如需对接 evaluator 或 gym-style wrapper, 也只包一层同步 step API, 不把 planner 接入 tiled runtime。

### 3.2 不推荐一开始做的事

- 不要一开始全仓迁移到 Isaac Lab API。当前项目已有大量 Isaac Sim 运行时、cuMotion、交互协议和测试, 全迁移风险大。
- 不要先支持相机 tiled rendering。相机会大幅增加显存和同步成本, 应放在 batched physics 稳定后。
- 不要先支持 rope tiled envs。动态链和复杂接触成本高, 第一阶段用刚体 TBlock/工装验证。
- 不要在 env 循环中调用 `world.step()`。
- 不要把现有 `MotionPlanner`、`MoveSpec`、路径规划、轨迹优化或交互队列接进 tiled runtime。
- 不要在一个 command step 中让不同 env 执行不同长度的轨迹。

### 3.3 兼容性原则

所有新增功能应挂在新模块或新函数上。现有默认 CLI 和脚本不应改变行为。旧 profile 中的固定 prim path 仍可用于单 env runtime; tiled runtime 在加载后通过 namespace overlay 改写 path, 不强迫所有 profile 立即改写。

## 4. 目标目录和新增模块

建议新增:

```text
src/linkerbot_sim/tiled/
  __init__.py
  config.py
  paths.py
  scene.py
  runtime.py
  articulation.py
  controller.py
  command.py
  ik.py
  reset.py
  state.py
  step_env.py

scripts/
  tiled_env_realtime.py
  tiled_env_benchmark.py

configs/envs/
  scene3_tiled/
    base.yaml
    envs/
      env_000.yaml
      env_001.yaml

tests/
  test_tiled_config.py
  test_tiled_paths.py
  test_tiled_controller.py
  test_tiled_state_shapes.py
```

如果后续决定和 `design_plan/gym_task_framework_proposal.zh-CN.md` 的 env 封装合并, `step_env.py` 可移动到 `src/linkerbot_sim/app/env.py`; tiled runtime 仍建议保留在 `tiled/`。该 env 封装只暴露同步 step-control API, 不承载 task reward/success, 也不承载异步规划。

## 5. 新配置规格

### 5.1 env YAML 扩展

在 env profile 中新增可选顶层 `tiled`:

```yaml
tiled:
  enabled: true
  num_envs: 256
  base_env_path: /World/envs
  env_prefix: env
  spacing: 2.0
  num_per_row: 16
  per_env_config_dir: envs
  clone:
    use_grid_cloner: true
    replicate_physics: false  # 当前 MJCF fixed-base 场景默认关闭；见 7.6
    copy_from_source: false
    enable_env_ids: false
    filter_collisions: true
    collision_root_path: /World/collisions
  runtime:
    use_batched_articulation_view: true
    inspect_env_ids: [0]
```

字段含义:

- `enabled`: 仅 tiled runtime 使用。旧 runtime 可忽略或拒绝。
- `num_envs`: 并行 env 数。
- `base_env_path`: env namespace 根路径。
- `env_prefix`: 子 env 命名前缀, 默认生成 `/World/envs/env_0`。
- `spacing`: 网格间距。必须大于机器人/物体最大运动半径的两倍。
- `num_per_row`: 网格每行数量; 缺省可用 `sqrt(num_envs)`。
- `per_env_config_dir`: 目录型 env profile 中保存 per-env YAML 的相对目录。
- `replicate_physics`: 用 PhysX replication 加速克隆。当前含 `body0` 为空 `rootJoint_*` 的 MJCF fixed-base robot 会在 builder 阶段自动关闭该项，避免 clone 后机器人 root anchor 错位。
- `filter_collisions`: 创建 collision groups, 防止 env 间碰撞。
- `enable_env_ids`: 高级选项。先默认 false; 需要 colocated env 时再启用。

### 5.1.1 目录型 tiled env profile

为了表达“所有 env 共享机器人、灯光、相机、solver 和物体集合, 但每个 env 的物体初始位姿不同”, tiled env 建议使用目录型 profile:

```text
configs/envs/scene3_tiled/
  base.yaml
  envs/
    env_000.yaml
    env_001.yaml
    env_002.yaml
    env_003.yaml
```

`base.yaml` 保存所有 env 的共通设置:

```yaml
env:
  name: scene3_tiled
solver:
  type: PGS
visuals: ...
sensors: ...
robots:
  dual:
    left: ...
    right: ...
objects:
  - name: workstation
    object_profile: workstation_armbase
    root_pose: ...
  - name: Tblock
    object_profile: TblockV1_default
    root_pose: ...
tiled:
  enabled: true
  num_envs: 4
  per_env_config_dir: envs
```

每个 `envs/env_XXX.yaml` 只写已有对象的 env-local `root_pose` 覆盖:

```yaml
env_id: 1
objects:
  Tblock:
    root_pose:
      xyz: [0.12, 0.04, -0.4]
      rpy: [0.0, 1.5707, 0.18]
metadata:
  replay_id: scene3_tiled_001
```

硬约束:

- 每个 env 的 object name/profile 集合必须和 `base.yaml.objects` 一致。
- per-env 文件不能新增、删除或替换 object profile, 只能覆盖已有对象的 `root_pose`。
- 这样 scene builder 可以先构建 `env_0`, 用 `GridCloner` 复制同构 stage, 再在 `world.reset()` 前逐 env 写入对象位姿差异。
- 如果未来需要完全不同的物体集合, 应按 variant 分成多个 tiled profile, 例如 `scene3_tiled_A` / `scene3_tiled_B`, 每个 profile 内部仍保持同构。

### 5.2 Python 配置类型

新增 `src/linkerbot_sim/tiled/config.py`:

```python
@dataclass(frozen=True)
class TiledCloneConfig:
    use_grid_cloner: bool = True
    replicate_physics: bool = True
    copy_from_source: bool = False
    enable_env_ids: bool = False
    filter_collisions: bool = True
    collision_root_path: str = "/World/collisions"

@dataclass(frozen=True)
class TiledRuntimeConfig:
    use_batched_articulation_view: bool = True
    inspect_env_ids: tuple[int, ...] = (0,)

@dataclass(frozen=True)
class TiledPerEnvConfig:
    env_id: int
    object_root_poses: dict[str, RootPoseConfig]
    metadata: Mapping[str, object] | None = None

@dataclass(frozen=True)
class TiledEnvConfig:
    enabled: bool = False
    num_envs: int = 1
    base_env_path: str = "/World/envs"
    env_prefix: str = "env"
    spacing: float = 2.0
    num_per_row: int | None = None
    per_env_config_dir: str | None = None
    per_env: tuple[TiledPerEnvConfig, ...] = ()
    clone: TiledCloneConfig = TiledCloneConfig()
    runtime: TiledRuntimeConfig = TiledRuntimeConfig()
```

验收:

- `TiledEnvConfig.from_env_config({})` 返回 disabled/num_envs=1。
- 非绝对 USD path 报错。
- `num_envs < 1` 报错。
- `spacing <= 0` 报错。
- `inspect_env_ids` 越界报错。
- `per_env.env_id` 越界或重复时报错。
- `per_env.objects.<name>.root_pose` 只能引用 base objects 中已经存在的对象; scene builder 阶段发现未知对象必须报错。

## 6. path 和 namespace 改造

### 6.1 问题

当前 robot/object profile 内写死:

```yaml
robot:
  prim_path: /World/AR5V2_L6V1_L

object:
  prim_path: /World/TBlock
```

tiled env 中应变成:

```text
/World/envs/env_0/AR5V2_L6V1_L
/World/envs/env_0/TBlock
```

### 6.2 新增 path helper

新增 `src/linkerbot_sim/tiled/paths.py`:

```python
def env_root_path(config: TiledEnvConfig, env_id: int) -> str:
    return f"{config.base_env_path}/{config.env_prefix}_{env_id}"

def make_env_local_prim_path(env_root: str, original_prim_path: str) -> str:
    # /World/AR5V2_L6V1_L -> /World/envs/env_0/AR5V2_L6V1_L
    # /World/Foo/Bar -> /World/envs/env_0/Foo/Bar
```

规则:

- 只接受绝对 USD path。
- 去掉开头 `/World/`, 把剩余后缀挂到 `env_root`。
- 如果输入已经在 `env_root` 下, 返回原值。
- 如果输入是 `/World`, 报错, 因为不能把整个 World 当资产路径。

### 6.3 不要直接修改原 profile 对象

推荐用 `dataclasses.replace()` 或局部 copy 派生 runtime config:

```python
robot_asset = replace(robot_execution.robot, prim_path=env_local_path)
robot_execution = replace(robot_execution, robot=robot_asset, root_pose=...)
```

这样旧 runtime 仍然使用原始 profile。

### 6.4 root pose 和 env origin

base env `env_0` 内部对象仍使用 scene YAML 的 `root_pose`。GridCloner 对 `env_0` root 加世界位移。不要把 env origin 直接加到每个 object/robot 的 root pose 上, 否则 clone 后会重复偏移。

规则:

- 构建 `env_0`: 使用原 scene root pose, 但 prim path 改到 `/World/envs/env_0/...`。
- clone: 对 env root `/World/envs/env_i` 加网格位移。
- robot prim 的 Xform 仍保持 env-local root pose; 不要把 `env_origin` 写回 robot prim。
- MJCF fixed-base robot 例外在 root joint anchor 上处理: importer 生成的 `rootJoint_*` 在 `body0` 为空时是 world anchor 语义, clone 后需要把每个 env 的 anchor 写成 `env_origin + robot.root_pose.xyz`, 但不能改 robot prim Xform。
- state API: 读世界坐标后, 如任务需要 env-local 坐标, 减去 `env_origins[env_id]`。

验收:

- `env_0` 的机器人相对 workstation 位姿与原 `scene3` 一致。
- `env_1` 与 `env_0` 的相对布局一致, 世界坐标相差一个 env origin。
- `world.reset()` 后各 env 的 fixed-base 机器人仍固定在各自 env root 下, 不会被 MJCF root joint 拉回同一个世界位置。

## 7. tiled scene 构建流程

新增 `src/linkerbot_sim/tiled/scene.py`。

### 7.1 单臂 tiled scene

流程:

1. 加载普通 env profile, robot profile, object profiles。
2. 创建 `SimulationSession`。
3. 定义 `/World/envs` scope。
4. 定义 `/World/envs/env_0` xform。
5. 把 robot/object 的 prim path 改写到 `env_0` 下。
6. 导入 `env_0` 的 robot/object。
7. 对 `env_0` 应用 USD/PhysX overrides。
8. 使用 `GridCloner` clone `env_0` 到 `env_1...env_N-1`。
9. clone 后为每个 env 的 MJCF fixed-base robot 重写 root joint world anchor。
10. 根据 `tiled.per_env` 对每个 env 已有 object 写入 env-local `root_pose` 差异。
11. 可选创建 collision groups。
12. 创建 batched articulation view。
13. `world.reset()`。
14. finalize articulation view, 解析 command joints。
15. 初始化 batched controller。

### 7.2 双臂 tiled scene

双臂场景同上, 但每个 env 内有 left/right 两个 articulation:

```text
/World/envs/env_0/AR5V2_L6V1_L
/World/envs/env_0/AR5V2_L6V1_R
```

创建两个 batched view:

```python
left_view = Articulation(prim_paths_expr="/World/envs/env_.*/AR5V2_L6V1_L/...")
right_view = Articulation(prim_paths_expr="/World/envs/env_.*/AR5V2_L6V1_R/...")
```

注意: MJCF importer 返回的 articulation root 可能不是 profile prim path 本身。构建 `env_0` 后需要记录 articulation root 相对 env root 的后缀, 再生成 regex。

### 7.3 per-env object pose 覆盖

目录型 tiled profile 的 per-env 文件只允许覆盖对象 `root_pose`。scene builder 的处理顺序是:

1. 用 `base.yaml.objects` 在 `/World/envs/env_0` 下导入完整对象集合。
2. `GridCloner` clone `env_0` 到所有 env root。
3. 对 `tiled.per_env` 中声明的对象, 找到对应 env 下的同名 prim path, 在 `world.reset()` 前调用 `apply_root_pose_to_prim()`。

位姿语义:

- per-env `root_pose` 是 env-local 坐标, 不需要手动加 `env_origin`。
- 因为对象 prim 位于 `/World/envs/env_i/...` 下, 其 translate/rotate 会自然相对于 env root 生效。
- env 间 object set 必须同构; 未在 base objects 中声明的 object name 一律报错。

### 7.4 clone 前还是 clone 后应用 overrides

优先选择:

1. 对 `env_0` 完成导入和所有 USD overrides。
2. 再 clone。

理由: inherited clone 可复用 source 修改, 物理复制也能减少重复解析。

如果发现某些 runtime API 必须 clone 后逐 prim 写入, 只能在 `world.reset()` 前对所有 env 遍历写 USD 属性, 但要把这类遍历限制在初始化阶段, 不要进入每 step。per-env object pose 覆盖就属于这种初始化阶段写入。

### 7.5 PhysX replication root path

开启 `replicate_physics=true` 时, `GridCloner.clone()` 的 `root_path` 必须传 clone path 前缀:

```python
base_env_path = "/World/envs"
env_prefix = "env"
prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]
root_path = "/World/envs/env_"  # 注意末尾下划线
```

不要传 `"/World/envs/env"`。Isaac 的 `generate_paths("/World/envs/env", N)` 虽然输入不带末尾下划线, 但内部给 PhysX replicator 保存的是 `"/World/envs/env_"`。如果传错, USD clone 是 `env_1`, PhysX replication 可能推导成 `env1`, 并在 `world.reset()` 附近触发 USD/PhysX 层崩溃。

验收:

- 对不含 MJCF world-fixed root joint 的资产，在 profile YAML 中设置 `tiled.num_envs: 2` 后，`scripts/tiled_env_realtime.py --env <profile> --steps 2 --no-realtime --replicate-physics` 可以完成 reset 和 step。
- 对当前 `scene3_tiled`，即使 CLI 传入 `--replicate-physics`，builder 也应打印 `PHYSICS_REPLICATION_DISABLED reason=mjcf_fixed_root_joint_without_body0`，ready 行中的 effective `replicate_physics=false`。
- left/right batched articulation view count 都等于 2。

### 7.6 MJCF fixed-base robot root joint anchor

MJCF importer 在 `fix_base=True` 时会生成 `rootJoint_*`。如果该 joint 的 `physics:body0` 为空, 它的 `localPos0/localRot0` 实际承担 world anchor 语义。单 env 场景里 `apply_root_pose()` 同时写 robot prim Xform 和 root joint anchor 是正确的；tiled env 里不能这样逐 env 直接重用。

tiled scene builder 应采用两层语义:

- robot prim Xform: 始终写 env-local `root_pose`, 由 `/World/envs/env_i` 的 Xform 提供世界偏移。
- MJCF root joint anchor: clone 后、`world.reset()` 前逐 env 写 world pose, 即 `xyz = env_origin + robot.root_pose.xyz`, `rpy = robot.root_pose.rpy`。

不要用完整 `apply_root_pose()` 处理 cloned robot, 否则 robot prim 会在 env root 偏移之外再加一次 `env_origin`。应提供或调用“只更新 MJCF fixed root joint anchor”的函数。

重要限制: PhysX replication 对 `body0` 为空的 cloned joint 会提示 `localPose wont be updated`，实测会导致 env_1 articulation world pose 不在目标 tile。当前 builder 应检测这类 joint 并关闭 replication。未来若要恢复 `replicate_physics=true` 的性能路径，需要先把 fixed-base root joint 改造成 replication 能正确更新的 env-local anchor 结构。

验收:

- 在 profile YAML 中设置 `tiled.num_envs: 2` 后，`scripts/tiled_env_realtime.py --env scene3_tiled --steps 2 --no-realtime` reset 后, 两个 env 的 left/right 机器人分别位于 `env_0/env_1`。
- 状态打印应包含 robot root pose override 数量; 双臂 `num_envs=2` 时应为 4。

### 7.7 collision filtering

如果 env 通过 spacing 分隔足够远, 理论上不需要过滤。但 256 env 时 broadphase 仍可能受影响, 建议实现可选过滤。

使用 `GridCloner.filter_collisions()`:

- `physicsscene_path`: 从 stage 中查找 `UsdPhysics.Scene`。
- `collision_root_path`: `/World/collisions`。
- `prim_paths`: 所有 env root paths。
- `global_paths`: 默认地面等需要所有 env 共享碰撞的路径; 当前 `scene1/scene3` 多数关闭 default ground, 可以为空。

验收:

- env 间相邻物体不会产生跨 env 接触。
- 关闭 filter 时仍能通过 spacing 正常运行。

## 8. batched articulation runtime

新增 `src/linkerbot_sim/tiled/articulation.py`。

### 8.1 类型

```python
@dataclass(frozen=True)
class TiledArticulationHandle:
    name: str
    view: object                    # isaacsim.core.prims.Articulation
    prim_paths_expr: str
    root_paths: tuple[str, ...]      # per env articulation root path
    asset_path: Path
    asset_type: str
    controlled_joints: tuple[str, ...]
    mjcf_path: Path | None

    @property
    def num_envs(self) -> int: ...

    @property
    def num_dof(self) -> int: ...

    @property
    def dof_names(self) -> tuple[str, ...]: ...
```

### 8.2 初始化

在 `world.reset()` 后:

```python
from isaacsim.core.prims import Articulation

view = Articulation(
    prim_paths_expr=regex,
    name=f"{logical_name}_view",
    reset_xform_properties=False,
)
world.scene.add(view)
world.reset()
# 或视 Isaac Sim API 要求: world.scene.add(view) 后 reset 时自动 initialize
```

需要通过 smoke test 确认 `Articulation` view 的初始化顺序。若 `world.scene.add(view)` 不适配, 使用 view 自身 `initialize()`。

### 8.3 shape 约定

所有 batched articulation state:

- `get_joint_positions()` -> `(num_envs, num_dof)`
- `get_joint_velocities()` -> `(num_envs, num_dof)`
- `set_joint_positions(positions)` 接受 `(num_envs, num_dof)`
- `set_joint_velocity_targets(velocities)` 接受 `(num_envs, num_dof)`

不要在公共 tiled API 中返回 `(num_dof,)`, 除非明确是单个 env 索引。

验收:

- `view.count == num_envs`。
- `view.num_dof` 与单 env articulation 一致。
- `view.dof_names` 与现有 `SingleArticulation.dof_names` 一致。

## 9. batched joint controller

新增 `src/linkerbot_sim/tiled/controller.py`。

### 9.1 目标

把当前 `JointController` 的核心语义扩展到 batch:

- command space 仍按 joint names 定义。
- follower/mimic 仍由 MJCF equality 解析。
- 输入 command positions/velocities/efforts shape 为 `(num_envs, num_command_dof)`。
- 输出 full target shape 为 `(num_envs, num_dof)`。
- 一次调用 articulation view API 写入全部 env。

### 9.2 数据结构

```python
@dataclass(frozen=True)
class BatchedControlTargets:
    positions: np.ndarray   # (N, D)
    velocities: np.ndarray  # (N, D)
    efforts: np.ndarray     # (N, D)

class BatchedJointController:
    def __init__(
        self,
        articulation_view,
        *,
        joint_names: list[str],
        settings: JointControlSettings,
        mjcf_path: str | Path | None,
    ) -> None: ...

    def configure_runtime(self) -> None: ...

    def build_control_targets(
        self,
        command_positions: np.ndarray | None = None,   # (N, C)
        command_velocities: np.ndarray | None = None,  # (N, C)
        command_efforts: np.ndarray | None = None,     # (N, C)
        *,
        base_positions: np.ndarray | None = None,      # (N, D)
    ) -> BatchedControlTargets: ...

    def apply_targets(self, targets: BatchedControlTargets) -> None: ...
```

### 9.3 follower 语义

当前单 controller 的 follower 目标基于 master 实际状态推导。batched 版应读取:

```python
current_positions = articulation_view.get_joint_positions()  # (N, D)
```

然后对每个 follower relation 以向量化方式写:

```python
full_positions[:, follower_idx] = multiplier * current_positions[:, master_idx] + offset
full_velocities[:, follower_idx] = multiplier * current_velocities[:, master_idx]
```

不要 Python `for env_id in range(N)`。可以对 follower relation 循环, 因为 relation 数量小。

### 9.4 runtime gains 和 modes

`Articulation` view 有批量 `set_joint_*` API, 但 articulation controller gain/mode API 是否与 `SingleArticulation` 完全一致需要 spike 确认。

实现顺序:

1. 先支持 implicit position control, 通过 `set_joint_position_targets()` / `set_joint_velocity_targets()` 跑通。
2. 再补 explicit effort/direct effort。
3. 最后补 per-DOF mode/gain/max effort 的完整一致性。

如果批量 view 的 controller API 不支持现有所有 per-DOF mode 操作, 可在初始化阶段对 `env_0` source USD 写 drive/gain, clone 后继承; 运行时只发 targets。

### 9.5 短期兼容 fallback

如果 batched controller API 卡住, 可临时实现 `LoopedTiledJointController`:

- 初始化时持有 N 个 `SingleArticulation`。
- 每 step 对 N 个 controller apply, 但仍只 `world.step()` 一次。

该 fallback 仅用于功能验证, 不满足 256+ 性能目标。代码中要标注 deprecated/temporary, benchmark 不以它为准。

验收:

- `build_control_targets()` 输入同一 command 给所有 env, 输出每行一致。
- 对所有 env 下发相同 hold target 100 步, 所有 env joint positions 在容差内一致。
- follower joints 不出现在 command space。

## 10. tiled runtime

新增 `src/linkerbot_sim/tiled/runtime.py`。

### 10.1 类型

```python
@dataclass
class TiledRobotRuntime:
    session: SimulationSession
    env_config: Mapping[str, object]
    tiled_config: TiledEnvConfig
    env_root_paths: tuple[str, ...]
    env_origins: np.ndarray                       # (N, 3)
    robot: TiledArticulationHandle
    controller: BatchedJointController
    command_adapter: TiledCommandAdapter
    object_handles: tuple[TiledObjectHandle, ...]
    objects: Mapping[str, TiledObjectHandle]
    render_enabled: bool

    def reset(self, env_ids: np.ndarray | None = None, *, seed: int | None = None) -> None: ...
    def step(self, action: TiledCommandAction, *, render: bool | None = None) -> TiledState: ...
    def close(self) -> None: ...

@dataclass
class TiledDualRobotRuntime:
    session: SimulationSession
    env_config: Mapping[str, object]
    tiled_config: TiledEnvConfig
    env_root_paths: tuple[str, ...]
    env_origins: np.ndarray
    left: TiledSideRuntime
    right: TiledSideRuntime
    object_handles: tuple[TiledObjectHandle, ...]
    objects: Mapping[str, TiledObjectHandle]
    render_enabled: bool

    def reset(self, env_ids: np.ndarray | None = None, *, seed: int | None = None) -> None: ...
    def step(
        self,
        *,
        left_action: TiledCommandAction | None,
        right_action: TiledCommandAction | None,
        render: bool | None = None,
    ) -> TiledState: ...
    def close(self) -> None: ...
```

### 10.2 create functions

```python
def create_tiled_robot_runtime(
    *,
    env: str,
    control_mode: str = "position",
    gui: bool = False,
    status_prefix: str | None = None,
) -> TiledRobotRuntime: ...

def create_tiled_dual_robot_runtime(...) -> TiledDualRobotRuntime: ...
```

env 数量由 profile YAML 中的 `tiled.num_envs` 决定，不提供 CLI/调用方 override。

### 10.3 command action 语义

新增 `src/linkerbot_sim/tiled/command.py`:

```python
@dataclass(frozen=True)
class TiledCommandAction:
    kind: str
    values: np.ndarray | None = None
    decimation: int | None = None
    interpolation: str = "smoothstep"
    tcp_frame_name: str | None = None
    pose_reference_frame: str = "env"
```

`kind` 第一版只接受:

- `hold`
- `joint_position_target`
- `joint_delta_pos`
- `ee_pose_target`
- `ee_delta_pos`
- `ee_delta_pose`

所有 action 第一维必须是 `num_envs`; 如果第一维是 1, 可以广播到所有 env。所有 action 必须在进入 physics tick 循环前转换成 batched joint target, shape 为 `(num_envs, num_command_dof)`。

`pose_reference_frame` 只对 `ee_pose_target` 生效:

- `env`: 默认, 目标位姿在每个 env 的局部坐标系下表达。转换到 USD/PhysX 世界坐标时加上对应 `env_origin`。
- `base`: 目标位姿在对应机器人 base 坐标系下表达。适合不希望上层关心 env origin 的控制器。
- `world`: 目标位姿直接在 USD world 坐标系下表达。该模式容易让多个 env 指向同一个世界点, 只建议调试或高级调用方使用。

### 10.4 step 语义

单臂:

```python
q_target = runtime.command_adapter.action_to_joint_target(action)
for q_cmd in runtime.command_adapter.interpolate_to(q_target):
    targets = runtime.controller.build_control_targets(command_positions=q_cmd)
    runtime.controller.apply_targets(targets)
    runtime.session.world.step(render=render_enabled)
return read_tiled_state(runtime)
```

双臂:

```python
left_q_target = left.command_adapter.action_to_joint_target(left_action)
right_q_target = right.command_adapter.action_to_joint_target(right_action)
for left_q_cmd, right_q_cmd in synchronized_interpolation(left_q_target, right_q_target):
    left_targets = left.controller.build_control_targets(command_positions=left_q_cmd)
    right_targets = right.controller.build_control_targets(command_positions=right_q_cmd)
    left.controller.apply_targets(left_targets)
    right.controller.apply_targets(right_targets)
    world.step(render=render_enabled)
return read_tiled_state(runtime)
```

注意:

- `decimation` 是 control step 到 physics tick 的固定展开倍数, 不是 env 循环。
- 同一个 `step()` 内所有 env 使用相同 `decimation`。
- `ee_delta_*` 在 physics tick 循环前完成 IK, 循环内只做 joint target 插值和 batched apply。
- IK 失败的 env 不应让整个 batch 崩溃; 建议该 env 保持上一帧 target, 并在 `TiledState.info` 或 step 返回诊断中暴露 `ik_success` mask。

### 10.5 render 语义

headless benchmark 必须 `render=False`。GUI 模式可以只观察 env 0, 但所有 env 仍参与物理。不要为每个 env 创建 viewport。

验收:

- `create_tiled_dual_robot_runtime(env="scene3", num_envs=4, gui=False)` 成功。
- 连续 `step()` 10 次只调用 10 次 `world.step()`。
- `runtime.close()` 关闭 app。

## 11. tiled objects 和 state

新增 `src/linkerbot_sim/tiled/state.py` 和必要的 object helper。

### 11.1 TiledObjectHandle

```python
@dataclass(frozen=True)
class TiledObjectHandle:
    name: str
    runtime_handle: str | None
    kind: str
    prim_paths: tuple[str, ...]          # len N
    local_prim_path_suffix: str
    view: object | None                  # RigidPrim/RigidPrimView, 后续补
```

初期可以不用 Isaac batched rigid view, 先通过 USD/PhysX API 获取 root pose。但为了 256+ 性能, 最终应使用 `RigidPrim`/`RigidPrimView` 一次读写。

### 11.2 TiledState

```python
@dataclass(frozen=True)
class TiledRobotJointState:
    joint_names: tuple[str, ...]
    positions: np.ndarray       # (N, D)
    velocities: np.ndarray      # (N, D)
    measured_efforts: np.ndarray | None
    applied_efforts: np.ndarray | None

@dataclass(frozen=True)
class TiledObjectState:
    name: str
    positions_world: np.ndarray     # (N, 3)
    orientations_wxyz: np.ndarray   # (N, 4)
    positions_local: np.ndarray     # (N, 3), positions_world - env_origins

@dataclass(frozen=True)
class TiledState:
    step: int
    time_s: float
    robots: Mapping[str, TiledRobotJointState]
    objects: Mapping[str, TiledObjectState]
    info: Mapping[str, np.ndarray] = field(default_factory=dict)
```

### 11.3 state dict

为了同步 rollout/MPC, runtime 必须能 get/set state:

```python
def get_state_dict(runtime) -> dict[str, np.ndarray]:
    return {
        "left_qpos": ...,
        "left_qvel": ...,
        "right_qpos": ...,
        "right_qvel": ...,
        "objects/Tblock/pose": ...,
        "objects/Tblock/velocity": ...,
        "step_count": ...,
    }

def set_state_dict(runtime, state: Mapping[str, np.ndarray], env_ids: np.ndarray | None = None) -> None:
    ...
```

要求:

- 所有数组第一维是 `num_envs` 或 selected `len(env_ids)`。
- 如果传入第一维是 1, 允许广播到 selected env_ids。
- set 后清零或恢复控制器内部目标缓存, 避免状态泄漏。

验收:

- `state = get_state_dict(); set_state_dict(state);` 后读出的 state 一致。
- 对 env 0 的 state 广播到 env 1..N, 执行相同动作 100 步后 state 一致。

## 12. reset 和 episode 管理

新增 `src/linkerbot_sim/tiled/reset.py`。

### 12.1 reset(env_ids)

`reset(env_ids=None)` 重置全部 env; `env_ids=[...]` 只重置部分 env。这里的 reset 是 vectorized runtime reset, 不是重新创建 USD stage, 也不是对 Isaac `World` 做局部 reset。步骤:

1. 规范化 env_ids。
2. 写机器人 joint positions 到初始 qpos。
3. 写机器人 joint velocities 为 0。
4. 写 object pose 到初始 pose + env origin。
5. 写 object linear/angular velocity 为 0。
6. 清空 batched controller 内部缓存。
7. 清空 selected env 的 action target、IK 诊断、episode step、episode id 等 runtime 缓存。
8. 返回 reset 后的 batched state, 并在 `info["reset_env_ids"]` 中标明本次被 reset 的 env。
9. 可选跑 settle steps。settle steps 仍是全 world step; 对未 reset env 会被推进, 因此默认不在 runtime.reset 内自动 settle。需要 settle 时应在 episode 初始化阶段统一 reset all env。

实现约束:

- 初始化阶段仍需要调用一次 `world.reset()` 来创建 PhysX view。
- runtime 的 `reset(env_ids=...)` 不应再调用 `world.reset()`，而应通过 batched articulation/object view 写回状态；否则无法支持部分 env reset。
- `reset(env_ids=None)` 也优先走同一套 state write 路径，保证全量 reset 和部分 reset 语义一致；只有 debug/hard reset 才重新 `world.reset()`。
- reset 后必须调用 `TiledCommandAdapter.reset()` 或等价 per-env cache reset，避免上一 episode 的 hold target 泄漏。

### 12.2 部分 reset 的注意事项

PhysX 一个 scene 中不能只推进部分 env。部分 reset 只是在同一全局时间下改写部分 env 的状态, 下一次 `world.step()` 仍推进全部 env。这符合 vectorized env 常见语义。

因此 tiled runtime 应区分两条时间轴:

- `global_step` / `time_s`: 整个 PhysX scene 的全局时间, 永远单调递增。
- `episode_step[env_id]`: 每个 env 自己的 episode 内步数, reset 该 env 时清零。

状态和 telemetry 必须同时携带这两类时间信息。不要试图让某个 env 的 Isaac 仿真时间单独归零。

### 12.3 deterministic seed

新增 `TiledSeedManager`:

```python
seed_for_env = base_seed + env_id
```

runtime 自身的 `reset()` 应恢复到 scene 默认状态, 不主动做任务随机化。下游 task 如需随机化, 应在 reset 后通过 `set_state_dict()` / object pose setter / joint state setter 显式写入, 并把采样值记录进 state dict, 方便复现。

验收:

- 两次 `reset(seed=123)` 后 state 完全一致。
- 下游用 `TiledSeedManager` 生成同一批随机化输入并通过 setter 写入后, state 完全一致。
- `reset(env_ids=[3])` 只改变 env 3。

### 12.4 reset API 形态

建议正式 runtime 暴露:

```python
class TiledResetOptions:
    env_ids: np.ndarray | None = None
    seed: int | None = None
    hard_world_reset: bool = False
    reset_controller_cache: bool = True
    return_state: bool = True

class TiledResetResult:
    env_ids: np.ndarray
    global_step: int
    episode_ids: np.ndarray
    episode_steps: np.ndarray
    state: TiledState | None
```

交互协议和 gym-style wrapper 都只调用这层 API, 不直接访问 articulation/object view。

## 13. 同步 step-control 接口和 IK 边界

本节定义 tiled runtime 对 evaluator/RL/MPC 暴露的最小接口。它可以被下游包装成 gym-style API, 但本仓库 tiled 层不放 task、reward、success 或异步 planner。

### 13.1 Step API

新增 `src/linkerbot_sim/tiled/step_env.py`:

```python
class TiledStepEnv:
    def __init__(
        self,
        *,
        env_profile: str,
        gui: bool = False,
        control_mode: str = "position",
        action_mode: str = "joint_position_target",
        control_frequency: float = 20.0,
    ) -> None: ...

    def reset(self, *, env_ids: np.ndarray | None = None) -> dict[str, np.ndarray]: ...
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: ...
    def get_state_dict(self) -> dict[str, np.ndarray]: ...
    def set_state_dict(self, state: Mapping[str, np.ndarray], env_ids: np.ndarray | None = None) -> None: ...
    def close(self) -> None: ...
```

输出:

- `obs`: dict of arrays/tensors, 第一维 `N`。
- `info["env_ids"]`: `(N,)`。
- `info["ik_success"]`: `(N,)`, 仅 IK action 填写。
- `info["action_mode"]`: 当前 action mode。

不输出:

- reward。
- terminated/truncated。
- success。
- task-specific metric。

这些都应由下游 task/evaluator 基于本仓库暴露的状态原语计算。

### 13.2 action mode

`TiledStepEnv(action_mode=...)` 第一版只接受:

| action mode | action shape | 含义 |
| --- | --- | --- |
| `joint_position_target` | `(N, C)` | command-space 绝对关节目标 |
| `joint_delta_pos` | `(N, C)` | command-space 关节增量 |
| `ee_pose_target` | `(N, 7)` | TCP 绝对目标位姿, `[x, y, z, qw, qx, qy, qz]` |
| `ee_delta_pos` | `(N, 3)` | TCP 位置增量, 姿态保持 |
| `ee_delta_pose` | `(N, 6)` 或 `(N, 7)` | TCP 位姿增量, 具体旋转参数需在实现前固定 |
| `hold` | `None` 或 `(N, 0)` | 保持当前 target |

不允许一个 `step()` 中混用不同 action mode。若上层需要左右臂不同 action mode, 应显式创建左右 action adapter, 但每侧仍必须在固定 control step 内得到 batched joint target。

绝对末端位姿的坐标系必须显式处理:

- 默认使用 env-local 位姿。也就是说同一个 `(x, y, z, quat)` action 广播到 N 个 env 后, 每个 env 都在自己的局部场景内到达相同目标。
- 可选支持 robot-base-local 位姿, 适合以机器人安装基座为参考的控制器。
- world 位姿只作为高级/debug 模式; 批量 env 下如果直接广播同一个 world 位姿, 多个 env 会指向同一个世界点, 通常不是想要的语义。

### 13.3 action decimation

高层 action 通常低于 physics 频率:

```text
physics_frequency = 240 Hz
control_frequency = 20 Hz
decimation = 12
```

`TiledStepEnv.step(action)` 应:

1. 把 action 转成 command targets。
2. 循环 `decimation` 次:
   - 对所有 env 写 targets。
   - `world.step()` 一次。
3. 读 obs/info。

注意: 这里的循环是 control decimation, 不是 env 循环。每次 physics tick 仍同时推进全部 env。

### 13.4 与 cuMotion 的边界

并行化方案只使用 cuMotion 的 FK/IK 能力, 不使用 cuMotion 规划能力。

允许:

- batched FK, 用于从 `(N, C)` 当前关节状态读取 TCP pose。
- batched geometric IK, 用于 `ee_pose_target` / `ee_delta_pos` / `ee_delta_pose`。
- 可选 batched collision-free IK, 但第一版默认关闭。

禁止:

- `MotionPlanner.plan(...)`。
- graph search。
- trajectory optimization。
- specified path conversion。
- 任何返回变长 trajectory 的接口。

当前 `CuMotionInverseKinematics` 是单请求、带内部 warm-start 状态的封装, 不应直接复用为 tiled IK hot path。tiled 方案应新增 `src/linkerbot_sim/tiled/ik.py`:

```python
@dataclass(frozen=True)
class BatchedIKResult:
    joint_positions: np.ndarray      # (N, C)
    success: np.ndarray              # (N,)
    position_error: np.ndarray       # (N,)
    orientation_error: np.ndarray | None
    status: tuple[str, ...]

class BatchedIKSolver:
    def solve(
        self,
        *,
        target_positions: np.ndarray,              # (N, 3)
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,                         # (N, C)
        tcp_frame_name: str,
    ) -> BatchedIKResult: ...
```

要求:

- seed 必须显式传入, shape 为 `(N, C)`。不要在 solver 内部用一个全局 warm-start 覆盖所有 env。
- 成功 env 使用 IK 解; 失败 env 保持上一帧 joint target 或当前 joint position, 并在 `success` mask 中标出。
- 必须使用 cuMotion batch/array IK API。运行环境缺少 batch API 时应直接报错并提示缺失能力, 不把 per-env loop 作为 tiled 性能路径或默认 fallback。
- warm-start 可以按 env 显式维护: 上层保存 `(N, C)` 的上一帧成功解或当前 joint target, 每次作为 seeds 传入 batch IK。solver 内部不要持有跨 env 混用的全局 warm-start。
- PushT 这类接触推动任务默认不要启用 collision-free IK 避开 TBlock, 否则 IK 可能主动躲开需要接触的物体。

### 13.5 与现有 motion runtime 的边界

tiled step-control 不替代现有 motion runtime。以下能力继续由旧路径负责:

- 绝对 IK pose 的高层命令协议。
- IK offset MoveSpec。
- C-space path / task-space path / line / arc。
- graph planner、trajectory optimizer、trajectory generation。
- 手部 overlay 的 before/after/sync 调度。
- 交互协议中的 queue、cancel、cancel_current、estop、status、quit。

如果未来确实需要“批量规划”, 应另开 planner 专用模块或下游 evaluator 阶段处理, 不要把它塞进 tiled runtime 的 `step()`。

### 13.6 异步 planner 层（已按三阶段落地）

当前代码已经支持在 tiled 场景中执行关节空间规划, 但 planner 仍然是 tiled runtime 外部的异步生产者, 再由同步 executor 在 command boundary 消费结果。这个边界没有改变 `TiledCommandAdapter`: adapter 仍只接受同步 step-control action, 不接受 planner request 或 `MoveSpec`。

已落地结构:

```text
TiledStepRuntime / Isaac World
  ↑ batched state snapshots
AsyncPlannerManager
  ├── worker_0: 独立 planner backend
  ├── worker_1: 独立 planner backend
  └── request/result tracking + cancel/stale result 标记
TiledTrajectoryBuffer
  ↓ 每个 command step 采样 ready trajectory -> batched joint targets
```

三阶段实现:

1. `src/linkerbot_sim/tiled/trajectory.py`: 提供 per-env/per-robot trajectory buffer, 支持离线规划结果载入、ready/idle/completed 状态和同步 step 采样。
2. `src/linkerbot_sim/tiled/planner.py`: 提供 `TiledPlannerManager`、`TiledPlanningRequest`、`TiledPlanningResult` 和默认 `LinearJointPlannerBackend`；cuMotion 专属 adapter 已拆到 `src/linkerbot_sim/backends/cumotion/tiled_planner.py`。
3. `src/linkerbot_sim/app/interactive/tiled/`: 提供 `load_trajectory`、`step_trajectory`、`trajectory_status`、`clear_trajectory`、`plan`、`planner_status`、`cancel_plan` 交互消息，并按 `protocol`、`transport`、`debug_runtime`、`isaac_runtime`、`planning`、`telemetry_loop` 等模块拆分。

边界规则:

- planner worker 只能读取上层传入的 state snapshot、目标和静态配置, 不直接访问 Isaac stage、World 或 PhysX runtime。
- 每个 worker 应拥有独立 cuMotion planner/context, 或使用明确线程安全的 planner batch API; 不要多个线程共享一个带内部状态的 planner 实例。
- 请求跟踪必须支持 cancel/stale result 丢弃, 防止 planner 落后于实时仿真后继续写旧轨迹。当前 `reset` / `set_state` 会取消命中 env 的 in-flight plan。
- planner 输出可以是变长轨迹, 但执行器必须在同步 command step 内把每个 env 的 ready trajectory 采样成统一 shape 的 batched joint target。
- 没有 ready trajectory 的 env 应 hold 上一帧 target 或进入显式等待状态; 不能因为某个 env 的 planner 未完成而阻塞 `world.step()`。
- 这一层可以多线程处理多个 planner 请求, 但它是“仿真外计算层”的并发, 不是 PhysX 多 scene 并发。Isaac/PhysX 仍然保持单 World、一次 step 推进全部 env。

结论:

- 异步路径规划已经可用于关节空间目标和离线轨迹回放, 但仍不属于 `TiledCommandAdapter` 热路径。
- 不要修改 `TiledCommandAdapter` 支持 planner request。
- 后续接入 collision-aware cuMotion graph/trajectory optimizer 时, 应通过 `linkerbot_sim.backends.cumotion.tiled_planner.CuMotionJointPlannerBackend` 或新的 planner backend 明确装配, 并保证每个 worker 独享 planner/context。

## 14. 脚本和 CLI

### 14.1 tiled realtime

已新增 `scripts/tiled_env_realtime.py`:

参数:

```text
--env scene3_tiled
--gui
--steps 0
--robots all
--drive-joints 3
--amplitude 0.12
--frequency 0.20
--phase-stride 0.35
--status-prefix TILED_REALTIME
--realtime / --no-realtime
--replicate-physics / --no-replicate-physics
--filter-collisions / --no-filter-collisions
```

行为:

1. 创建 `SimulationApp` / `World`。
2. 加载普通 `scene3.yaml` 或目录型 `scene3_tiled/base.yaml`。
3. 构建真实 Isaac/PhysX tiled scene。
4. `world.reset()` 后 finalize batched articulation view。
5. 用 batched `ArticulationActions` 同步驱动所有 env 的简单关节正弦目标。
6. 打印 env roots、env origins、object pose override 数量、collision filtering 状态和 step FPS。
7. 允许通过 CLI 临时覆盖 `tiled.clone.replicate_physics` 和 `tiled.clone.filter_collisions`, 用于隔离 Isaac/PhysX clone 问题。

验收输出示例:

```text
TILED_REALTIME_READY num_envs=4 roots=[...] origins=[...] objects=[...] object_pose_overrides=4 collision_filtering=True
TILED_REALTIME_ROBOT name=left count=4 num_dof=... command_joints=[...]
TILED_REALTIME_ROBOT name=right count=4 num_dof=... command_joints=[...]
TILED_REALTIME_STEP step=120 sim_time=0.500 wall_fps=...
```

### 14.2 tiled interactive

tiled 模式应考虑交互, 但它不是旧 `dual_arm_interactive.py` 的并行版本。它的定位是“同步 command-step 调试入口”“episode 控制入口”和“外部 trajectory/planner 调度入口”, 不承担旧 `MoveSpec` 队列或每 env 独立 motion runtime。

当前已把 `scripts/tiled_env_interactive.py` 变成薄入口，并把主体 runtime 迁入 `src/linkerbot_sim/app/interactive/tiled.py`。正式 CLI 只启动真实 tiled scene + batched `Articulation` view；纯 Python fake IK / fake state runtime 仅作为内部测试替身保留，不再暴露 `--backend debug`。

Isaac runtime 当前是交互 smoke/调试入口，不是旧 motion runtime 的并行版。它已支持关节类同步 action、局部 reset、状态读写、机器人选择、trajectory buffer 回放、外部异步关节规划，以及通过 `BatchedCuMotionIKSolver` 执行的 `ee_pose_target` / `ee_delta_pos` / `ee_delta_pose` batched IK action。该路径不回退到 per-env `solve_ik` loop。

交互协议只支持:

- `status`: 返回 global step、每个 env 的 episode step/id、env roots/origins、selected robot command joints。
- `reset`: 支持 `env_ids`；不传时 reset all env。
- `step`: 接收 `TiledCommandAction` 支持的同步动作。
- `load_trajectory` / `step_trajectory`: 载入并同步回放 ready trajectory。
- `trajectory_status` / `clear_trajectory`: 查看或清理 trajectory buffer。
- `plan` / `planner_status` / `cancel_plan` / `clear_completed`: 提交后台关节空间规划、收集 ready result、取消 in-flight request 或清理 completed 摘要缓存。
- `get_state`: 返回 batched state dict, 可通过 `env_ids` 和字段列表裁剪。
- `set_state`: 写回 selected env state, 用于 replay/MPC/debug。
- `quit`: 退出进程。

明确不支持:

- `MoveSpec`、line/arc path、specified path、graph planner。
- 每 env 异步命令队列、cancel current、不同 env 不同步数推进。
- 交互层直接访问 Isaac stage 或单独调用 `world.step()`。

协议示例:

```json
{"type": "reset", "env_ids": [0, 3]}
{"type": "step", "kind": "joint_delta_pos", "robots": ["left"], "values": [[0.01, 0, 0, 0, 0, 0, 0]], "decimation": 4}
{"type": "step", "kind": "joint_delta_pos", "values": [[0.005, 0, 0]], "env_ids": [1, 2], "robots": ["left"]}
{"type": "get_state", "env_ids": [0], "fields": ["robots.left.joint_positions", "episode_steps"]}
```

`env_ids` 在 `step` 中的含义是“只更新这些 env 的 action target, 其它 env hold 上一帧 target”。physics 仍然一次 step 推进所有 env。

### 14.3 tiled Foxglove / MCAP telemetry

tiled telemetry 应复用旧 runtime 的 `FoxgloveStateSink` / `FoxgloveLogger` 思路, 但不能直接把 `(num_envs, dof)` 全量塞进标准 `/joint_states`。当前已新增 `src/linkerbot_sim/telemetry/tiled.py`，先服务 `app/interactive/tiled.py` 的 selected-env 调试输出；旧 `src/linkerbot_sim/tiled/telemetry.py` 已删除:

```python
class TiledTelemetryConfig:
    selected_env_ids: tuple[int, ...] = (0,)
    publish_decimation: int = 10
    include_full_batch_json: bool = True
    include_standard_joint_states: bool = True
    foxglove_live_host: str = "127.0.0.1"
    foxglove_live_port: int | None = None
    foxglove_mcap_path: str | Path | None = None

class TiledTelemetrySink:
    def publish(self, state: TiledState) -> None: ...
    def close(self) -> None: ...
```

topic 建议:

- 已实现 `/tiled/state`: JSON, 保存 interactive state response、`env_ids`、`step/time_s` 和触发事件摘要。
- 待实现 `/tiled/summary`: JSON, 保存 aggregate 指标, 如 active env 数、reset env ids、IK 失败数、step FPS。
- 已实现 `/tiled/env_000/joint_states`: Foxglove 标准 `JointStates`, 只为第一个 selected env 发布, 方便曲线面板查看。
- 已实现 `/tiled/env_000/scene`: Foxglove `SceneUpdate`, 只为 selected env 发布 object marker / TCP marker。

默认策略:

- realtime benchmark 默认关闭 telemetry。
- interactive 默认只发布 selected env, 默认 `selected_env_ids=[0]`。
- MCAP 必须支持 decimation, 默认不记录 256 env 全量每步状态。
- reset 事件必须写入 `/tiled/state` 或 `/tiled/summary`, 便于回放时知道哪个 env 开始了新 episode。

验收:

- 不开启 telemetry 时不创建 Foxglove sink, 不影响 tiled runtime。
- 开启 MCAP 后能看到 reset、step 和 selected env joint state。
- `num_envs=256` 时默认 telemetry 不造成显著 step FPS 下降。

### 14.4 benchmark

新增 `scripts/tiled_env_benchmark.py`:

参数:

```text
--env scene3
--steps 1000
--warmup-steps 120
--headless
--disable-objects
--action-mode joint_position_target,ee_pose_target,ee_delta_pos
--csv logs/tiled_benchmark.csv
```

记录:

- num_envs
- scene
- robot dof
- object count
- physics dt
- wall time
- physics steps/s
- aggregate env steps/s = `num_envs * physics_steps/s`
- GPU memory, 如果可读
- CPU memory
- warnings/error count
- action mode
- IK solve time / success rate, 仅 IK action 记录

## 15. 测试计划

### 15.1 轻量单元测试, 不启动 Isaac

新增:

- `tests/test_tiled_config.py`
  - YAML parse。
  - 默认值。
  - 非法值。

- `tests/test_tiled_paths.py`
  - `/World/Foo` -> `/World/envs/env_0/Foo`
  - `/World/Foo/Bar` -> `/World/envs/env_0/Foo/Bar`
  - 已在 env root 下不重复嵌套。
  - 非绝对 path 报错。

- `tests/test_tiled_controller.py`
  - 用 fake articulation view 测 shape。
  - 输入 `(N, C)` 输出 `(N, D)`。
  - follower 映射向量化正确。

- `tests/test_tiled_state_shapes.py`
  - fake runtime state dict get/set shape。
  - broadcasting 语义。
  - reset result 同时携带 `global_step` 和 per-env `episode_step`。
  - partial reset 只清 selected env 的 episode/cache 信息。

- `tests/test_tiled_command.py`
- action mode 白名单。
- `(1, C)` 到 `(N, C)` 广播。
- decimation 固定且不依赖 env 数。
- `ee_pose_target` 默认 env-local, 并能拒绝未知 `pose_reference_frame`。
- `env_ids` 裁剪 step target 时, 未选中 env 默认 hold。

- `tests/test_tiled_ik.py`
  - fake batched IK solver 输入/输出 shape。
  - IK 失败 mask 不抛弃整个 batch。
  - seed shape 必须是 `(N, C)`。

- `tests/test_tiled_interactive.py`
  - JSON `status/reset/step/get_state/set_state/quit` 协议解析。
  - `reset(env_ids=[...])` 返回 selected env ids。
  - `plan` / `planner_status` / `step_trajectory` 协议打通外部 planner + trajectory buffer。
  - `MoveSpec` / 旧 motion queue 类型消息被明确拒绝。

- `tests/test_tiled_telemetry.py`
  - telemetry 未开启时不 import Foxglove SDK。
  - selected env topic 名称稳定, 例如 `/tiled/env_000/joint_states`。
  - MCAP/live sink 共用同一份 `TiledState` 映射逻辑。

### 15.2 Isaac smoke, 需要本地 Isaac

这些可以不进普通 pytest, 通过脚本运行:

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_realtime.py \
  --env scene3_tiled --steps 120 --no-realtime
```

真实 interactive smoke:

```bash
printf '%s\n' \
  '{"type":"status"}' \
  '{"type":"reset","env_ids":[0]}' \
  '{"type":"step","kind":"joint_delta_pos","robots":["left"],"values":[[0.01,0,0,0,0,0,0]],"decimation":2}' \
  '{"type":"quit"}' \
  | PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
      --env scene3_tiled
```

逐步扩大:

```bash
for n in 4 16 64 256; do
  # 先在 benchmark/profile YAML 中写入 tiled.num_envs=$n。
  PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_benchmark.py \
    --env scene3_tiled --steps 1000 --headless
done
```

### 15.3 回归测试

任何 tiled 改动后必须确认旧测试仍过:

```bash
PYTHONPATH=src pytest
```

旧单臂/双臂脚本不应因为新增 tiled 模块改变行为。

## 16. 分阶段 ticket

### P0. Spike: 原生 Cloner/View 可用性验证

目标: 不改主架构, 用单独脚本证明本项目资产可批量 clone 和 batched view。

任务:

1. 新增临时或正式 `scripts/tiled_env_realtime.py`。
2. 在 `env_0` 下导入 scene3 的左右机器人和 TBlock。
3. 用 `GridCloner` clone 4 份。
4. 用 `Articulation` view 包装 left/right。
5. 读取 `get_joint_positions()` shape。
6. 下发简单 position target, `world.step()` 10 次。

验收:

- `num_envs=4` 不崩溃。
- left/right view count 都是 4。
- joint position shape 是 `(4, D)`。

失败处理:

- 如果 MJCF importer 与 clone 有冲突, 先尝试 clone 完整 env root, 而不是逐 robot clone。
- 如果 `Articulation` regex 找不到 root, 记录 `env_0` articulation root 相对路径后生成明确路径列表。

### P1. 配置和 path namespace

目标: 建立不会污染旧 runtime 的 path rewrite 能力。

任务:

1. 新增 `tiled/config.py`。
2. 新增 `tiled/paths.py`。
3. 添加不启动 Isaac 的测试。
4. env 数量只从 profile YAML 的 `tiled.num_envs` 读取。
5. 支持目录型 env profile: `configs/envs/<name>/base.yaml` + `envs/env_XXX.yaml`。
6. 支持 per-env 对已有 object 的 `root_pose` 覆盖。

验收:

- 所有新增单元测试通过。
- 旧 runtime 没有导入 tiled 模块也能正常加载。
- `load_profile_yaml("env", "scene3_tiled")` 能返回合并后的 config。
- `TiledEnvConfig.per_env` 能解析 env id、对象名、root pose 和 metadata。

### P2. tiled scene builder

目标: 以正式模块方式构建 env_0 和 clones。

任务:

1. 新增 `tiled/scene.py`。
2. 抽出 robot/object config path override helper。
3. 创建 env root scope。
4. 导入 env_0 robot/object。
5. clone 到 N env。
6. 保存 env origins、env root paths、articulation root suffix。
7. clone 后、reset 前应用 per-env object pose overrides。
8. 可选 collision filtering。

验收:

- `create_tiled_dual_scene(env="scene3", num_envs=4)` 返回结构化 scene handles。
- stage 中存在所有 env root。
- 每个 env 下存在左右 robot 和 object。
- `scene3_tiled` 中每个 env 的 TBlock 使用各自 per-env YAML 里的 env-local pose。

### P3. batched articulation view

目标: 正式使用 `Articulation` view 读取/写入批量 joint state。

任务:

1. 新增 `tiled/articulation.py`。
2. 根据 env_0 articulation root suffix 生成 view regex 或 path list。
3. world reset 后初始化 view。
4. 实现 joint positions/velocities read/write helper。

验收:

- view count = num_envs。
- view dof names 与单 env 一致。
- set_joint_velocities zero 可执行。

### P4. batched controller

目标: 支持 batched command-space joint target、position hold 和 follower 展开。

任务:

1. 新增 `BatchedJointController`。
2. 复用 `resolve_joint_indices()`、`MimicFollowerTargetMapper` 或抽出 follower relation 数据。
3. 实现 `build_control_targets()`。
4. 实现 implicit position apply。
5. 单元测试 fake view。
6. smoke 脚本改用正式 controller。

验收:

- 所有 env 同 action 下最终 joint state 一致。
- follower 不接收外部 command。
- `num_envs=16` smoke 稳定。

### P5. tiled runtime

目标: 形成同步批量 step-control runtime API。

任务:

1. 新增 `TiledRobotRuntime` / `TiledDualRobotRuntime`。
2. 新增 create functions。
3. 实现 `step()`、`reset()`、`close()`。
4. `step()` 只接受 `TiledCommandAction`, 不接受 `MoveSpec` 或 planner request。
5. status_prefix 输出。
6. smoke/benchmark 使用 runtime。

验收:

- 修改 YAML `tiled.num_envs` 为 4/16 后，`scripts/tiled_env_realtime.py` 通过。
- `world.step()` 次数等于 physics tick 数, 不乘以 env 数。

### P6. state dict 和 partial reset

目标: 支持同步 rollout/MPC 的 state clone, 并提供真实 Isaac tiled runtime 的单 env/批量 reset。

任务:

1. 新增 `tiled/state.py`。
2. 实现 robot state get/set。
3. 实现 object pose get/set。
4. 实现 broadcasting。
5. 新增 `tiled/reset.py`。
6. 实现 `reset(env_ids=None)` 和 `reset(env_ids=[...])`。
7. 区分 `global_step` 和 per-env `episode_step` / `episode_id`。
8. reset 后清空 selected env 的 controller target、IK cache、action adapter cache。

验收:

- state round trip。
- env 0 state broadcast 到所有 env 后, 相同 action rollout 一致。
- `reset(env_ids=[1])` 后 env 1 回到初始状态, env 0/2/3 状态不被重写。
- partial reset 不调用 `world.reset()`。

### P7. 同步 action adapter 和 step env

目标: 把上层 action mode 稳定转换成 batched joint target, 并提供最小 `TiledStepEnv` 封装。

任务:

1. 新增 `tiled/command.py`。
2. 新增 `TiledCommandAdapter`。
3. 支持 `hold`、`joint_position_target`、`joint_delta_pos`。
4. 实现 action decimation 和 smoothstep/linear 插值。
5. 新增 `TiledStepEnv`, 只返回 obs/info。
6. 明确拒绝 reward/success/task metric, 并拒绝把 planner request 作为 `step()` action。

验收:

- `env.reset()` 返回第一维 N 的 obs。
- `env.step(action)` 推进 `decimation` 个 world step。
- `joint_position_target` 与 `joint_delta_pos` action shape 校验清晰。
- 传入 planner/MoveSpec 类型对象给 `step()` 会报错, 错误信息说明 planner 应走外部 manager + trajectory buffer。

### P8. batched IK action

目标: 支持 `ee_pose_target` / `ee_delta_pos` / `ee_delta_pose`, 并验证 cuMotion FK/IK 的批量路径是否足够快。

任务:

1. 已完成: 新增 `tiled/ik.py`。
2. 已完成: 实现 `BatchedIKSolver` 接口和 fake 测试。
3. 已完成: 接入 cuMotion `CollisionFreeIkSolver.solve_array` batch API, 不保留 per-env fallback。
4. 已完成: `ee_pose_target` 默认接受 env-local `[x, y, z, qw, qx, qy, qz]`, 并支持 world/base reference。
5. 已完成: `ee_delta_pos` 默认保持 TCP 姿态。
6. 已完成: IK 失败时保持上一帧 joint target, 并返回 `ik_success` mask。
7. 待 benchmark: `num_envs=4/16/64/256` 下的 IK solve time。

验收:

- `ee_pose_target` 的同一 env-local 目标广播到所有 env 后, 每个 env 的 TCP 都到达各自局部坐标系下的同一位姿。
- `ee_delta_pos` action 能让 TCP 按期望方向微动。
- `ik_success` shape 为 `(N,)`。
- `num_envs=256` 的 IK solve time 被记录; 若不能达标, 文档明确降级限制。

### P9. tiled interactive 和 telemetry

目标: 在不引入旧 motion runtime 异步复杂度的前提下, 支持 tiled command-step 交互调试、partial reset 控制和可选 Foxglove/MCAP 记录。

任务:

1. 已完成: 升级 `scripts/tiled_env_interactive.py`, 正式 CLI 只暴露真实 Isaac tiled runtime；debug runtime 保留为内部测试替身。
2. 已完成: 交互协议支持 `status/reset/step/get_state/set_state/quit`。
3. 已完成: `reset` 和 `step` 支持 `env_ids` 裁剪; 未选中 env 默认 hold。
4. 已完成: 新增 `telemetry/tiled.py`；删除 `tiled/telemetry.py` 兼容 re-export。
5. 已完成: 支持 selected env 的标准 `/tiled/env_XXX/joint_states`。
6. 部分完成: 支持 `/tiled/state` JSON topic；`/tiled/summary` 仍待补。
7. 已完成: 支持 MCAP 和 live server, 并提供 decimation。
8. benchmark 默认关闭 telemetry。

验收:

- 内部 debug runtime 不启动 Isaac, 单元测试覆盖协议和 shape。
- Isaac runtime 能在 `num_envs=2` 下执行 status/reset/step/get_state/set_state/quit。
- 开启 MCAP 后可以回放 selected env joint state 和 reset event。
- 不开启 telemetry 时不 import Foxglove SDK。

### P10. 性能优化和 256+

目标: 让 `num_envs=256` 成为可用目标。

任务:

1. benchmark scene3 no-camera/no-logging。
2. 识别瓶颈: Python controller、PhysX solver、contact count、object views、cuMotion/IK。
3. 移除 per-env Python loop。
4. 降低不必要 solver iteration 或提供 fast profile。
5. 验证 collision filtering。

验收:

- `num_envs=256` smoke 通过。
- benchmark 产出 CSV。
- 文档记录目标 GPU、Isaac 版本、吞吐和限制。

## 17. 性能注意事项

### 17.1 初始建议 profile

为了先跑通 256:

- `gui=False`
- `render=False`
- 禁用 sensor cameras。
- 禁用 Foxglove state stream。
- 禁用 joint CSV logger 或大幅降低频率。
- 先用 rigid object, 不用 rope。
- 先用 position implicit, 不用 Python explicit effort。
- 先用 PGS, solver iteration 使用现有默认; 如不稳定再调。

### 17.2 性能指标

必须同时记录:

- physics steps/s
- env steps/s
- memory
- clone/build time
- reset time
- action apply time
- IK solve time, 仅 `ee_delta_*` action 记录
- state read time

只看 env steps/s 容易掩盖初始化或 reset 瓶颈。

### 17.3 256+ 的真实含义

`num_envs=256+` 不等于所有功能都开时一定 256:

- AR5 + L6 手自碰撞/复杂接触会显著增加成本。
- 每 env 相机渲染会显著增加显存和同步。
- rope/dynamic chain 比 TBlock 贵很多。

因此文档和 CLI 帮助中应写清楚 benchmark 条件。

## 18. 风险和应对

### R1. MJCF importer 生成的 articulation root path 不稳定

应对:

- 构建 env_0 后用 `find_articulation_root()` 记录真实 root。
- 计算相对 env root suffix。
- clone 后用 suffix 拼出每个 env 的 root path。
- view 可用 path list 时优先 path list; regex 作为简化。

### R2. cloned inherited prim 的 USD overrides 不生效

应对:

- 所有 overrides 先写入 env_0, 再 clone。
- 如 clone 后仍缺属性, 添加 clone 后验证器, 检查 gravity/solver/material counts。
- 只在初始化阶段遍历修复。

### R3. batched view control mode API 不完整

应对:

- 第一版只支持 implicit position target。
- gain/mode 尽量在 USD 或 source env_0 runtime 初始化中写好。
- explicit effort/direct effort 作为后续 ticket。

### R4. cross-env collision

应对:

- spacing 先足够大。
- 添加 `filter_collisions` 选项。
- benchmark 开关对比。

### R5. Python per-env loop 限制 256+

应对:

- 单元测试允许 fake loop, 但 runtime hot path 禁止 env loop。
- benchmark 输出 action apply/state read 时间。
- code review 关注 `for env_id in range(num_envs)` 出现在 step path。

### R6. state reset 泄漏

应对:

- 所有 controller target、IK seed、action adapter 状态必须有 reset hook。
- `set_state_dict()` 后默认清 controller cache 和 IK 诊断缓存。
- 添加“broadcast same state -> same rollout”验收。

### R7. IK 失败或过慢

应对:

- IK action 必须返回 `ik_success` mask 和 error/status。
- 失败 env 默认保持上一帧 joint target, 不让整个 batch 崩溃。
- `ee_pose_target` / `ee_delta_pos` 第一版允许只在小 `num_envs` 下启用; `num_envs=256` 需要真实 benchmark 支撑。
- PushT/接触推动任务默认关闭 collision-free IK, 避免 IK 规避本该接触的物体。

### R8. 交互协议膨胀成第二套 motion runtime

应对:

- tiled interactive 只接受 `TiledCommandAction` 和 reset/state 操作。
- `plan` 只提交外部 planner manager, ready result 只进入 trajectory buffer；`TiledCommandAdapter` 不接 planner request。
- 明确拒绝 `MoveSpec`、旧 motion queue、cancel_current/estop 等旧 motion runtime 语义。
- 异步规划不得访问 Isaac stage/World/articulation view, 只能消费主线程复制出的 state snapshot。

### R9. telemetry 拖慢 256 env benchmark

应对:

- benchmark 默认关闭 Foxglove/MCAP。
- telemetry 默认 selected env + decimation。
- 完整 batch 只写 JSON summary 或低频 MCAP, 不默认每步写 256 份标准 `JointStates`。

## 19. 代码审查清单

每个实现 PR/commit 都检查:

- 是否保留旧 runtime 行为。
- 是否没有在 per physics tick 中按 env 调 `world.step()`。
- 是否没有修改用户现有 profile 的原始 dict/dataclass。
- 所有 batched arrays 第一维是否是 `num_envs`。
- env world pose 与 env-local pose 是否区分清楚。
- reset 是否清 joint velocity、object velocity、controller cache。
- `step()` 是否只接受同步 action, 没有接入 `MoveSpec` / planner request / 旧 motion queue。
- interactive 的 planner 是否仍通过外部 manager + trajectory buffer, 没有复刻旧 motion queue。
- Foxglove/MCAP 是否支持 selected env 和 decimation, 且默认不影响 benchmark。
- IK action 是否有失败 mask, 且失败 env 不影响其它 env。
- smoke 脚本是否可在 headless 下运行。
- 失败信息是否包含 env_id、prim_path、shape。

## 20. 最小可交付版本定义

MVP 不要求完整 step env 或 IK, 只要求:

1. YAML `tiled.num_envs: 4` 时，`scripts/tiled_env_realtime.py --env scene3_tiled` 通过。
2. left/right batched articulation view count = 4。
3. `joint_position_target` 或 `hold` 下所有 env joint state 一致。
4. 每 tick 只调用一次 `world.step()`。
5. 旧 pytest 通过。

MVP 之后再扩大到:

1. `num_envs=16/64`。
2. state dict 和 reset。
3. `joint_delta_pos` 和 `TiledStepEnv`。
4. `ee_pose_target` / `ee_delta_pos` / batched IK。
5. tiled interactive 的 Isaac runtime。
6. selected env Foxglove/MCAP telemetry。
7. `num_envs=256` benchmark。

## 21. 建议实施顺序摘要

1. P0 spike: 证明 Cloner + Articulation view 跑通本项目资产。
2. P1: 配置和 path rewrite。
3. P2: tiled scene builder。
4. P3: batched articulation view。
5. P4: batched controller。
6. P5: tiled runtime。
7. P6: state dict/reset。
8. P7: 同步 action adapter 和 step env。
9. P8: batched IK action。
10. P9: tiled interactive 和 telemetry。
11. P10: 性能优化和 256+ benchmark。

实现代理请按这个顺序推进。不要先做后面漂亮的 env wrapper、telemetry 或 planner 接入, 因为真正承重的是 P0-P6: namespace、clone、batched view、batched action apply、同步 step-control runtime 和可靠 reset。
