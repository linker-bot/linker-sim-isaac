# YAML 配置参考

语言：[中文](configuration.md) | [English](../../en/reference/configuration.md)

`linkerbot_sim.configuration.catalog` 是新配置图唯一 YAML I/O owner。所有 mapping 拒绝重复 key、
未知字段、隐式类型转换、路径逃逸和未消费字段；profile 在创建 SimulationApp 前完成组合与校验。
Profile selector 只允许以 `/` 分隔的不带点号相对 component；禁止反斜杠、绝对路径、`..`、空 component
和 `.yaml` 后缀。scene symlink 解析后仍必须位于所选产品的 namespace 根内。

## 解析闭包、来源与指纹

catalog 必须在调用方选择的同一个 `configs_root` 中解析完整闭包：mode 引用 leaf；scene 引用 robot/object
profile；physics engine 派生默认 controller bundle，robot profile 默认值和 scene instance override 可覆盖它。
controller 优先级固定为 scene instance > robot profile > physics 派生默认值。返回的 config 已冻结
`resolved_profile`、`controller_bundles` 和只读 `sources`，runtime 不允许按名称再次读取全局默认根。

自定义配置根必须提供所有被引用的 `modes/scenes/physics/.../robots/objects/controllers` 文件。旧
`configs/envs` schema 已删除，不是 scene 配置入口。`sources` 的绝对路径只用于 provenance，不参与
语义 fingerprint；validator、runtime 与 snapshot compatibility 共用 configuration 层的 canonical
payload/fingerprint。相同语义配置位于不同绝对根时指纹相同，有效 robot/object/controller 内容变化时
指纹变化。

## Mode root

### Mirror

```yaml
mode: mirror
compute:
  cuda_device: 0
profiles:
  scene: mirror/scene3
  physics: physx/cpu
  control: mirror
  curobo: mirror
  planning: mirror
  outputs: mirror_default
```

可选的 `profiles.hybrid_control` 只属于 Mirror。维护的组合使用 240 Hz 专用 scene 与 PhysX CPU：

```yaml
mode: mirror
compute:
  cuda_device: 0
profiles:
  scene: mirror/scene3_hybrid
  physics: physx/cpu
  control: mirror
  hybrid_control: hybrid_force_position
  curobo: mirror
  planning: mirror
  outputs: mirror_default
```

省略该 slot 时，v3 hybrid operation 会 fail closed 为未配置。选择它要求初始 position mode、PhysX
CPU、足够的 scene 频率、arm 重力补偿、物理 TCP metadata、arm `effort+direct` 与 hand/default
`position+implicit` controller profile。

Mirror 接受 PhysX/CPU、Newton/CPU 或 Newton/CUDA；四个 profile 都引用同一个 `control: mirror`，默认 controller bundle
由 `physics.engine` 派生，不在 control leaf 重复选择。所有 mode profile 都在根声明唯一
`compute.cuda_device`。PhysX CPU 的 `physics_device=cpu`，但 cuRobo IK/规划与 RTX 渲染仍消费根选择的
同一 GPU；Newton CPU 的 physics device 是 `cpu`，Newton CUDA 则由根设备派生。Mirror composition 在 session 投影时派生一个 world，
不会删减底层 multi-world engine 能力；physics leaf 不声明第二份 world 数。

### Kaleidoscope

```yaml
mode: kaleidoscope
compute:
  cuda_device: 0
environments:
  num_envs: 256
  base_env_path: /World/envs
  env_prefix: env
  origin_xyz: [0.0, 0.0, 0.0]
profiles:
  scene: kaleidoscope/tblock_push
  physics: physx/cuda
  task: kaleidoscope/tblock_push_v1
```

根字段只允许 `mode`、`compute`、`environments` 和 `profiles`。`scene/physics/task` 是三个必选
profile slot；Kaleidoscope 完全没有 `profiles.control` 或 resolved control 对象，action 语义属于 task，
默认 controller bundle 由 `physics.engine` 派生。只有 EE/直线 action 才允许并且必须增加可选的 `profiles.curobo`，canonical
`joint_control` mode 必须省略它。
`environments` 只允许 `num_envs/base_env_path/env_prefix/origin_xyz`。递归出现 render、camera、transport、planner/planning、playback 或
telemetry 字段都会失败。默认选择的 `physx_cuda` profile 使用上面的 PhysX CUDA/Fabric 组合；Newton alternative 为：

```yaml
mode: kaleidoscope
compute:
  cuda_device: 0
environments:
  num_envs: 256
  base_env_path: /World/envs
  env_prefix: env
  origin_xyz: [0.0, 0.0, 0.0]
profiles:
  scene: kaleidoscope/tblock_push
  physics: newton/cuda
  task: kaleidoscope/tblock_push_v1
```

所选 physics backend、Torch、Warp interop、cuRobo 与 trainer 的 device 均来自根
`compute.cuda_device`；leaf 不得重复声明 active GPU 或 policy device。

EE/直线 composition 在 mode root 选择数值后端，而不是让 task 选择：

```yaml
profiles:
  scene: kaleidoscope/tblock_push
  physics: physx/cuda
  task: kaleidoscope/tblock_push_v1
  curobo: kaleidoscope_batch_ik
```

task 仍只拥有 action 语义，不包含 backend/profile 字段。catalog 要求每个 EE/直线 action 都有这个可选
引用，并拒绝 `joint_control`/`joint_delta` 携带它，避免纯关节训练分配无用 cuRobo context。

## Profile owner

| 目录 | 唯一事实 |
| --- | --- |
| `configs/modes/` | 产品选择、profile 引用、唯一 compute device；Kaleidoscope 环境数、路径命名和原点 |
| `configs/scenes/mirror/` | Mirror reality scene 的 robot/object、重力、频率、planning 启动策略与可选视觉事实 |
| `configs/scenes/kaleidoscope/` | Kaleidoscope 无头单环境模板的 robot/object、重力与物理频率 |
| `configs/physics/` | `engine/execution`、solver、PhysX Fabric/显存门禁或 Newton per-world capacity |
| `configs/control/mirror.yaml` | Mirror command space、空闲步进、墙钟 pacing、interface 与请求缺省语义 |
| `configs/control/hybrid_force_position.yaml` | 显式笛卡尔混合控制初值、固定安全限幅与运行时 tuning 上限 |
| `configs/curobo/` | FK/IK 容量和可选 MotionPlanner 数值能力；不拥有 CUDA 卡号 |
| `configs/planning/` | 后端中立的 Mirror 请求默认策略；不选择或配置数值后端 |
| `configs/outputs/` | Mirror-only camera/logging/telemetry 生命周期 |
| `configs/tasks/` | observation、action、reward、termination、randomization |
| `configs/training/` | 下游算法与 rollout 参数；不拥有环境 device |
| `configs/visualization/kaleidoscope.yaml` | Kaleidoscope launch-only 单环境 viewport；不属于训练语义 |

每个 `configs/` 一级 profile 目录在 `linkerbot_sim.configuration` 下恰有一个同名 schema owner：
单文件分组使用 `scenes.py` 一类模块，嵌套分组使用 `modes/`、`tasks/`、`training/`、
`visualization/` 一类 package。只有 `catalog.py/common.py/fingerprint.py` 是配置基础设施，
不对应 profile 分组。通用场景视觉原语归 `scenes.py`；`visualization` package 只保留与
`configs/visualization/kaleidoscope.yaml` 对应的 Kaleidoscope 启动 schema。

Mirror 日志没有独立 profile 目录，`outputs.logging` 是唯一配置入口。运行时配置树也不再保留
`controllers/curobo/objects/robots` 下可被误选的 `example.yaml`；实际 profile 与本文就是维护示例。
Hybrid diagnostics 复用同一 outputs profile，通过
`logging.hybrid_control_path`/`log_hybrid_control` 与
`telemetry.include_hybrid_control`/`topics.hybrid_control` 配置，不增加顶层 output section。

Scene 有三个相关但不能混用的名称：mode root 保存带产品命名空间的 selector，例如
`mirror/scene3`；catalog 将其解析为文件路径 `configs/scenes/mirror/scene3.yaml`；文件内的稳定身份仍是
不带命名空间的 `scene.id: scene3`。Kaleidoscope 对应的是 selector
`kaleidoscope/tblock_push`、路径 `configs/scenes/kaleidoscope/tblock_push.yaml` 和
`scene.id: tblock_push`。旧平铺 selector 和跨产品 scene 引用一律无效，因为两种 scene schema 不可互换。

两份 scene 必须保持独立。前者包含相机/viewport/60 Hz 现实映像；后者是无视觉、240 Hz 的任务
prototype。环境数量与命名事实只在 Kaleidoscope mode root 声明，training 不能内联进 mode。

Mirror scene 必须声明 `planning_startup: lazy|prewarm`。`lazy` 把 context 与 planner 创建推迟到首次
规划请求；`prewarm` 则在 `MIRROR_INTERACTIVE_READY` 前按 robot ID 顺序创建每个支持规划机器人的
`interactive` slot 0 context、同步同一个初始 collision snapshot，并 materialize MotionPlanner。
cuRobo profile 的 `motion_planner.warmup` 仍独立决定 planner materialize 时是否执行数值预热；scene
字段只决定 materialize 的时机。

## Physics engine/execution

physics leaf 目录按 engine/execution 保持完整、规则的布局：

```text
configs/physics/
  physx/{cpu,cuda}.yaml
  newton/{cpu,cuda}.yaml
```

每个 canonical leaf 都能按 configuration schema 独立严格解析，产品根再收窄合法组合：

| 产品 | engine | execution | 关键约束 |
| --- | --- | --- | --- |
| Mirror | `physx` | `cpu` | 物理在 CPU；根 `compute` 供 cuRobo/RTX；完整 scene query 能力按 profile |
| Mirror | `newton` | `cpu` | MuJoCo CPU integration；一个 product world；无 CUDA stream/graph；根 `compute` 仍供 cuRobo/RTX |
| Mirror | `newton` | `cuda` | project-owned Newton runtime；一个 product world；可选 render sync |
| Kaleidoscope | `physx` | `cuda` | GPU pipeline、Fabric、scene query off；训练 Kit headless，可选 viewport Kit |
| Kaleidoscope | `newton` | `cuda` | project-owned multi-world Newton runtime；每个 env 一个独立 world；训练 Kit headless，可选 viewport Kit |

公开 physics leaf 使用正交字段，不使用 backend `kind`：

```yaml
physics:
  engine: physx
  execution: cpu  # Mirror；Kaleidoscope PhysX 使用 cuda
  solver_type: PGS
```

```yaml
physics:
  engine: newton
  execution: cuda
  nconmax_per_world: 200
  njmax_per_world: 1200
  # canonical leaf 还必须声明其余容量与积分字段
```

`newton/cpu` 与 `newton/cuda` 分别声明各自的 execution；Mirror 可组合两者，Kaleidoscope 只组合 CUDA。
Mirror 与 Kaleidoscope 共用 `newton/cuda` 的每 world 求解配置。Mirror 派生一个 world；Kaleidoscope
从 mode root `environments.num_envs` 的最终值派生 world 数。两者都由项目 runtime 直接拥有
Model/State/Control/Solver，不加载 Isaac Newton extension。机器人 profile 的 `physics.gravity` 在
PhysX 中写 `disableGravity`，在 Newton 中于 model finalize 前写 `mjc:gravcomp`；后者不支持
运行期逐 link 修改。

## Kaleidoscope viewport profile

`configs/visualization/kaleidoscope.yaml` 是与上述 mode graph 平行的 launch-only 配置：

```yaml
viewport:
  selected_env: 0
  render_every_n_steps: 1
  width: 1280
  height: 720
  window_width: 1440
  window_height: 900
  renderer: RaytracedLighting
  anti_aliasing: 0
  samples_per_pixel_per_frame: 1
  denoiser: false
  visuals: { ... }
```

根字段和嵌套 visual 都严格拒绝未知字段。`make_viewport_env()` 通过
`viewport_profile="kaleidoscope"` 选择该文件，也可用 `viewport` 直接传入已加载对象；Gymnasium human
render 使用 `viewport_profile`。`selected_env` 在最终
`num_envs` 上校验；只有该 world 进入 renderer-facing USD。该对象不附加到 `KaleidoscopeConfig`，所以只改
窗口、视角、灯光或 render
cadence 不会改变 episode snapshot/clone fingerprint。PhysX/Newton viewport Kit 均排除 camera、
SyntheticData、Replicator、录制和 image observation；训练 tick 仍固定 `render=False`。

catalog 会从当前配置根展开 scene 引用的 robot/object profile 和有效 controller bundles，并把来源登记到
只读 `sources`。Kaleidoscope 可包含多个静态 rigid，但必须恰好包含一个非静态 rigid，且其名称必须等于
`task.dynamic_object`；dynamic chain 会在 Kit 启动前失败。该约束保证唯一的 `object.*`
state/snapshot/clone 字段覆盖全部动态对象状态。

### PhysX CUDA GPU memory budget

PhysX 引擎分配容量不属于项目配置字段。`configs/physics/physx/cuda.yaml` 必须声明完整
`physics.memory`；对应的 `GpuMemoryBudget` 不提供缺字段默认值：

| 字段 | 类型与范围 | 门禁语义 |
| --- | --- | --- |
| `max_simulator_process_mib` | 正整数 MiB | NVML 归属当前 simulator PID 的最大进程显存，覆盖 Kit、PhysX、Torch 与其它原生 CUDA allocator |
| `min_free_floor_mib` | 正整数 MiB | prelaunch、post-warmup、steady baseline/final 四阶段的设备空闲显存绝对下限 |
| `min_free_fraction_after_warmup` | 浮点数 `(0, 1]` | post-warmup、steady baseline/final 三阶段的设备空闲比例下限 |
| `max_steady_growth_mib` | 非负整数 MiB | steady final 相对 steady baseline 的 simulator PID 显存增长上限 |

审计通过 CUDA UUID 把根 `compute.cuda_device` 映射到 NVML 设备，要求 warmup 后 PID 可见，并同时报告
Torch allocated/reserved 供诊断；NVML 进程值才是预算判定 owner。该门禁只接受 `physx_cuda` profile：

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

Newton 的 `nconmax_per_world`、`njmax_per_world` 和 world 数属于另一套容量合同，不能由这四个 PhysX
字段或该脚本推断。

## Robot 与 object 的物理 leaf

Robot profile 把后端中立重力策略和 PhysX 专属资产属性分开：

```yaml
robot:
  physics:
    gravity:
      default: false
      arm: false
      hand: false
    material:
      contact_static_friction: 0.8
      contact_dynamic_friction: 0.6
      contact_restitution: 0.0
    physx:
      material:
        friction_combine_mode: average
      rigid_body:
        linear_damping: 0.0
        angular_damping: 0.1
      joint:
        friction: 0.5
        follower_friction: 0.5
      solver:
        arm: {position_iterations: 32, velocity_iterations: 4}
        hand: {position_iterations: 32, velocity_iterations: 4}
```

`gravity` 与 `material` 由两个后端消费；`physx` leaf 只在 PhysX composition 中投影。Newton 仍会从
engine-specific controller bundle 获得通用 `UsdPhysics.DriveAPI` seed，但不会读取或告警跳过 PhysX
combine mode、阻尼、关节摩擦和 solver。MJCF 原生 `frictionloss` 在 PhysX 下优先于 profile 的 joint friction；Newton 则由
上游 `SchemaResolverMjc` 直接读取 importer author 的 `mjc:frictionloss`。

Controller profile 只拥有控制律真正消费的 stiffness、damping、max force、effort limit 和 follower
drive seed。`joint_friction` 不再是 controller 字段，也不会因字段缺失被 parser 补默认值。旧
`robot.physics.solver` 路径无效；唯一合法路径是 `robot.physics.physx.solver`。

Object profile 的接触系数属于通用 USD material，PhysX 扩展单独放在 `physics.physx`：

```yaml
object:
  physics:
    material:
      static_friction: 0.8
      dynamic_friction: 0.6
      restitution: 0.0
    physx:
      material:
        friction_combine_mode: average
```

Dynamic-chain object 还可以在同一 PhysX leaf 声明：

```yaml
object:
  physics:
    physx:
      solver:
        position_iterations: 48
        velocity_iterations: 4
```

Newton 只投影通用 object material，不导入 `PhysxSchema`，也不会把合法的 PhysX leaf 当成兼容性降级。
旧 `physics.material.friction_combine_mode`、`physics.solver_position_iterations` 和
`physics.solver_velocity_iterations` 均被 strict schema 拒绝，不提供兼容别名。

## Kaleidoscope action union

Action 由 `task.action.mode` 判别：

- `joint_control`：`position_delta_scale_rad`、`velocity_scale_rad_s`、
  `effort_limit_fraction`、`clip`、`physics_ticks_per_action`；canonical task 使用该 variant，固定 action
  shape 下支持 position/velocity/effort；
- `joint_delta`：`scale_rad`、`clip`、`physics_ticks_per_action`；禁止 IK/linear 字段；
- EE position/full-pose：由 mode root 的可选 `profiles.curobo` 提供 batch IK，失败策略固定 hold/penalty/truncate；
- EE linear position/full-pose：再声明 waypoint count/progress mode，同步执行；
- 所有 task variant 均禁止 backend、profile、planner、trajectory batching 和 collision avoidance 字段；
  EE/linear 所选 cuRobo profile 还必须省略 `motion_planner`、保持 `kinematics.collision_check=false`，
  canonical profile 省略 `kinematics.collision_cache`；也可保留合法 cache，但运行时会忽略。

## Environments 与后端复制实现

Kaleidoscope mode root 的 `environments` 只负责 `num_envs`、`base_env_path`、`env_prefix` 和
`origin_xyz`。`num_envs` 是唯一持久配置 owner；创建环境时的显式 `num_envs` 参数可以覆盖该
缺省值，但不会产生第二份 profile。公开 `profiles.replication` 和 `configs/replication/` 已经删除。

内部复制实现没有删除，而是与物理引擎固定绑定。PhysX builder 始终使用 GridCloner、3.0 m 间距、
`replicate_physics=true`、`copy_from_source=true`、`enable_env_ids=true`；Newton builder 始终使用
multi-world、零间距和彼此独立的 worlds。Newton `world_count` 只在 session projection 中从最终
`num_envs` 派生，physics leaf 不声明第二份 world 数。这里的隔离是物理上阻止不同 env 互相接触，
不是关闭环境内部的物理接触。
机器人与任务对象仍由 PhysX/Newton contact pipeline 计算真实接触，但不建立规划碰撞查询或避障。

EE/直线 action 对应的 mode 必须用 `profiles.curobo` 选择 `kaleidoscope_batch_ik` 一类 kinematics-only
profile。其 `kinematics.max_batch_size` 必须覆盖最终有效 `num_envs`，并保持 collision check 关闭、cache
不进入后端；canonical profile 选择省略该字段，保留合法值也不会分配 cache。不足时 scene assembly
在创建 solver 前失败。`joint_control`/`joint_delta` mode 则必须省略该引用。

## Control、cuRobo 数值能力与 Mirror planning 默认策略

Kaleidoscope 没有 control slot。task 固定 action variant、shape 与 tick 数，`physics.engine` 派生默认
`physx` 或 `newton` controller bundle。Newton bundle 提供较低的 arm/hand 增益与 follower 零 drive。
初始 position、active mode 与 generation 是 runtime state，不是 YAML selector，也不进入 semantic config
fingerprint。原生环境可在完整 decision 之间切换全局关节控制模式，但不会改变构造期 action variant。
Mirror 则唯一引用 `configs/control/mirror.yaml`；它拥有控制模式、空闲步进、
`sync_simulation_to_wall_clock`、interface 和请求缺省语义，controller bundle 同样由 physics 派生。

`sync_simulation_to_wall_clock: true` 让 idle 与 motion 共用一个按
`scene.physics_frequency_hz` 推进的墙钟同步器，但不会修改 physics dt。某个 tick 落后时，Mirror 会从
当前墙钟重新建立 deadline，不会突发补跑错过的 tick；因此负载过高时仿真可以慢于真实时间。设为
`false` 后不再等待墙钟，physics 会按机器可达到的最快速度推进。Canonical Mirror control profile 默认开启。

`configs/control/hybrid_force_position.yaml` 是独立可选 profile。`motion`、`force`、`posture` 保存显式
笛卡尔控制初值，`tuning` 保存 owner-queued 更新不可越过的固定逐字段上限。`tare`、`contact`、
`limits`、filter cutoff、支持 frame、允许的 force axes、最大时长和最低频率都是构造期安全事实，不能通过
wire 修改。运行时增益使用独立 generation，不改变语义配置 fingerprint。

`configs/curobo/*.yaml` 是唯一的 cuRobo 数值 profile。根 `curobo` 只声明 `kinematics` 的
IK batch/seed/CUDA graph/collision 开关与可选 MotionPlanner 数值能力；CUDA 卡号仍只来自 mode root。已验证的
cuRobo 0.8.0 task bundle 和四个 float32 dtype 由后端固定，不是 YAML 字段。
`kinematics.max_batch_size` 只限制 FK/IK，不参与 MotionPlanner 容量计算。Mirror 的
`curobo: mirror` 还必须声明 `motion_planner`；其 context 固定一次处理一个请求
（`max_batch_size=1`），该 section 只拥有 warmup、IK/trajopt seed、CUDA graph、collision capability
和 cache 预分配。项目固定 runtime 下
`motion_planner.use_cuda_graph` 必须为 `false`；IK 的 graph 可独立开启。Kaleidoscope profile 必须省略
整个 `motion_planner`。`collision_check=true` 时对应 `collision_cache` 必填；false 时可省略或保留
合法 cache，运行时都会投影为空且不分配。Mirror MotionPlanner 使用 `collision_check=true`，所以
planner cache 仍是必填容量。

`configs/planning/mirror.yaml` 只保存后端中立的 `planning.request_defaults`：`duration_s`、`sample_dt_s`、
`timeout_s`、`avoid_collisions`、`force_collision_refresh` 和 `coordination: independent`。wire 层允许
覆盖的 planning 字段只有 `duration_s`、`sample_dt_s`、`avoid_collisions` 和
`force_collision_refresh`；`coordination` 只能在单 segment wrapper 或 timeline 顶层覆盖。
`timeout_s` 不是 wire 字段，每个请求始终使用 `planning.request_defaults.timeout_s`。该 profile 不选择
cuRobo，也不拥有 solver 容量。默认开启避障时，所选 cuRobo profile 必须同时提供 planner collision
capability 和 cache 容量。

## 校验命令

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile newton_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode kaleidoscope --profile newton_cuda
```

命令输出所选 source path 与确定性 fingerprint，不启动 Isaac。自定义 profile 名只能是安全的
相对名称，不能包含 `..`、绝对路径或反斜杠。
