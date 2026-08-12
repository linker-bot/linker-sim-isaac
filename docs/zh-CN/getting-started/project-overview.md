# 项目概览

语言：[中文](project-overview.md) | [English](../../en/getting-started/project-overview.md)

LinkerHand Simulation 是面向 Isaac Sim 6.0.1 的 checkout application。它把两类执行目标拆成
两个互不导入的产品组合根，而不是用一个带大量 optional 字段的通用 runtime：

- **Mirror** 表示一个现实工作站在仿真中的映像。一个 World 中可以有多个机器人和对象；产品
  提供交互协议、共享 tick motion、cuRobo 规划、规划碰撞、相机、遥测、日志和持久快照。
- **Kaleidoscope** 表示同一任务 scene 的大量隔离变体。它可选择 PhysX CUDA/Fabric 或项目自有
  multi-world Newton；两个后端的训练入口都以 headless、GPU-native 方式向相同的 Torch CUDA 接口提供
  批量 reset/state/snapshot/clone、批量 IK 和同步直线动作。产品不拥有批量轨迹 planner、规划
  避障、相机、SyntheticData、Replicator、录制、transport 或遥测；显式调试入口可显示一个选中环境。

## 执行模型

| 边界 | Mirror | Kaleidoscope |
| --- | --- | --- |
| 稳定入口 | `linkerbot_sim.mirror` | `linkerbot_sim.kaleidoscope` |
| 环境形状 | 一个 World；一个或多个 robot/object | `num_envs` 个同构、隔离 env |
| physics | PhysX/CPU、Newton/CPU 或 Newton/CUDA | PhysX/CUDA 或 Newton/CUDA |
| 数据边界 | JSON/Python 与冷 NumPy/映射状态 | 原生 Torch CUDA；Gymnasium 才转 NumPy |
| 动作 | 关节、timeline、IK、直线、完整规划 | position target、批量 IK、同步直线 action |
| 状态 | scene state 与 versioned scene snapshot | GPU state、episode snapshot、env-to-env clone |
| 输出 | camera、CSV、MCAP、Foxglove | 训练 info tensor；无输出 worker |
| 碰撞 | 物理接触及 Mirror planning collision | 保留任务物理接触和 env 隔离；无规划碰撞/避障 |

Mirror 的“一个 World”不表示只能有一个机器人。Kaleidoscope 的“并行”也不表示每个 env
有一个机器人；scene prototype 可以包含多个机器人，但所有 env 必须保持同构。

## 正式 Kit 入口矩阵

产品工厂根据 strict composition 选择且只选择一个正式 Kit；调用方不传入任意 experience 路径，也不
手工叠加 physics/render extension：

| 产品 | engine / execution | 渲染闭包 | factory 选择的 Kit |
| --- | --- | --- | --- |
| Mirror | PhysX / CPU | 由 outputs profile 控制 | `apps/linkerbot_sim.mirror.physx.python.kit` |
| Mirror | Newton / CPU 或 CUDA | 关闭 | `apps/linkerbot_sim.mirror.newton.python.kit` |
| Mirror | Newton / CPU 或 CUDA | 开启 | `apps/linkerbot_sim.mirror.newton_render.python.kit` |
| Kaleidoscope | PhysX / CUDA | 训练 headless | `apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` |
| Kaleidoscope | Newton / CUDA | 训练 headless | `apps/linkerbot_sim.kaleidoscope.newton.python.kit` |
| Kaleidoscope | PhysX / CUDA | 显式单环境 viewport | `apps/linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit` |
| Kaleidoscope | Newton / CUDA | 显式单环境 viewport | `apps/linkerbot_sim.kaleidoscope.newton_viewport.python.kit` |

Mirror PhysX 使用同一个含 RTX 资源的 experience，并由 session render spec 决定是否实际渲染；Mirror
Newton 才按 render closure 分成 physics-only 与 Newton-render 两个 experience。Kaleidoscope
保留两个 physics-only 训练 Kit，显式 viewer 才选择对应 viewport Kit；viewport 只显示
`selected_env`，不增加 camera/SyntheticData/Replicator。
公开 mode profile 显式写出 execution：Mirror 为 `physx_cpu/newton_cpu/newton_cuda`，
Kaleidoscope 为 `physx_cuda/newton_cuda`。

## 所有权与调用流

```text
Mirror CLI / embedded API                    Torch / Gymnasium / skrl
        ↓                                              ↓
MirrorController → MirrorRuntime          TorchKaleidoscopeEnv → KaleidoscopeRuntime
                         \                  /
                          → IsaacSession ←
                                  ↓
                  concrete PhysicsRuntime
          PhysxRuntime(World) | NewtonRuntime(Model/State/Solver)
```

`IsaacSession` 直接拥有 SimulationApp、stage 和一个 concrete physics runtime。PhysX runtime
再拥有唯一 Isaac World；Newton runtime 直接拥有 Model/State/Control/Solver，不伪造 World。
产品资源先关闭，session 最后关闭。Kaleidoscope training package 不拥有 Isaac session，只消费
`KaleidoscopeTrainingPort`。

Mirror 的 Newton composition 在 session 投影时派生一个 world。Kaleidoscope 则从 mode root
`environments.num_envs` 的最终值派生 `world_count`，由项目 `NewtonRuntime` 为同构环境创建彼此隔离的 world；physics
leaf 不重复声明环境数，也不加载 `isaacsim.physics.newton`、`isaacsim.physics.newton.tensors` 或其它
Isaac Newton extension。

## 分层

- `configuration`：纯 Python、不可变的 mode 根和唯一 YAML catalog；不导入 Torch、Isaac 或产品。
- `isaac`：SimulationApp、stage、physics runtime 和 replicated scene 基础设施；不导入产品层。
- `backends.curobo`：按能力拆分的数值 backend。Kaleidoscope 只能使用 kinematics/IK 能力；
  planning/collision 只由 Mirror 组合。
- `mirror`、`kaleidoscope`：各自拥有状态机、use case 与资源关闭顺序，互不导入。
- `training.skrl`：只依赖 Kaleidoscope public training port。

完整逐模块 owner 见[源码模块图](../development/module-map.md)。

## 配置图

Mode 文件只保存 profile 引用和唯一 device 事实：

```text
configs/modes/mirror/{physx_cpu,newton_cpu,newton_cuda}.yaml
  ├─ compute.cuda_device
  ├─ scene selector mirror/scene3
  │    └─ scenes/mirror/scene3.yaml（scene.id: scene3）
  ├─ physics/physx/cpu.yaml、physics/newton/cpu.yaml 或 physics/newton/cuda.yaml
  ├─ control/mirror.yaml
  ├─ curobo/mirror.yaml
  ├─ planning/mirror.yaml
  └─ outputs/mirror_default.yaml

configs/modes/kaleidoscope/{physx_cuda,newton_cuda}.yaml
  ├─ compute.cuda_device
  ├─ environments.{num_envs,base_env_path,env_prefix,origin_xyz}
  ├─ scene selector kaleidoscope/tblock_push
  │    └─ scenes/kaleidoscope/tblock_push.yaml（scene.id: tblock_push）
  ├─ physics/physx/cuda.yaml 或 physics/newton/cuda.yaml
  ├─ tasks/kaleidoscope/tblock_push_v1.yaml
  └─ 可选 curobo/kaleidoscope_batch_ik.yaml（仅 EE/直线 action）

configs/visualization/kaleidoscope.yaml
  └─ launch-only selected env / window / renderer / scene visuals
```

Scene selector、文件路径与内部 identity 是三类事实：mode root 分别写 `mirror/scene3` 和
`kaleidoscope/tblock_push`；catalog 将它们解析到对应产品子目录；文件内只写 `scene.id: scene3` 或
`scene.id: tblock_push`。两个产品的 scene schema 互不兼容，因此不接受旧平铺 selector 或跨产品引用。
环境数量与路径命名只由 Kaleidoscope mode root 的 `environments` 持有，training profile 不折回 mode。
PhysX 固定派生 GridCloner/env IDs，Newton 固定派生 multi-world；这些仍是内部复制实现，不是公开
selector。配置规则见
[配置参考](../reference/configuration.md)。

Mirror 两个 engine 共用 `control/mirror.yaml`，默认 controller bundle 由 physics engine 派生；Kaleidoscope
没有 control profile 或 control 对象。`planning/mirror.yaml` 只拥有后端中立的请求默认策略；cuRobo
IK batch 容量，以及 MotionPlanner seed、CUDA graph、碰撞能力和 cache 容量统一归
`configs/curobo/`。MotionPlanner 固定一次处理一个请求，其 cache 容量仍显式配置。已验证的 0.8.0 task bundle 与
float32 dtype 由后端固定。Kaleidoscope task 不选择 backend：EE/直线 composition 在 mode root 增加
`profiles.curobo`，纯关节的 `joint_control`/`joint_delta` composition 必须省略它。

## GPU 与冷数据边界

Kaleidoscope 的 observation、action、reward、done、selector、状态、snapshot、clone、RNG 及
skrl rollout memory 必须位于 `compute.cuda_device`。skrl tokenized 训练热路径不调用 `.cpu()`、
`.numpy()`、`.tolist()`、`.item()` 或 `nonzero()`。native/debug `step` 刻意同步读取一次 done scalar，
使未 reset 的可恢复误用在 physics 推进前失败。只有以下显式边界可以读取或离开 GPU：

- native/debug pending-reset guard：每拍最多一次 scalar，只服务直接调试调用；
- `GymnasiumKaleidoscopeAdapter`：整批转换成 NumPy，换取 Gymnasium 生态兼容；
- persistent checkpoint：用户显式请求磁盘保存/加载时执行 CPU 序列化。
- human viewport：只在 `env.render()` 时把选中 world 同步到 renderer-facing USD；不进入训练 step。

Mirror 的 scene snapshot 是版本化冷状态，不可与 Kaleidoscope episode snapshot 混用。

## Workspace 与环境

仓库依赖 checkout 内的 `configs/`、`scripts/`、资产和 cuRobo task，不构建 wheel。仿真环境：

```bash
uv sync --extra simulation --extra visualization --extra training
export OMNI_KIT_ACCEPT_EULA=Y
```

CPU 文档/配置测试使用独立 `.venv-dev`，避免 `usd-core` 与 Kit `pxr` 混装：

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev --extra visualization
```

## 运行与关闭

Mirror 的 stdin、TCP JSONL 和 WebSocket 只允许 loopback；项目不提供认证或 TLS。后台 ingress
只解析请求，任何 USD/Isaac mutation 都回到 owner thread。关闭顺序固定为：

1. 停止 ingress/admission；
2. 关闭 outputs、camera、planner；
3. 关闭 controller 与 view；
4. 关闭 `IsaacSession`。

Kaleidoscope 没有后台 service。`TorchKaleidoscopeEnv.close()` 关闭 task/view，再关闭 session。
任何 engine setter 失败会让 runtime fail-stop，不能在 canonical buffer 与物理状态可能分叉时继续训练。

## 下一步

- [选择 Mirror 或 Kaleidoscope](choose-runtime-and-api.md)
- [Mirror 快速入门](mirror-quickstart.md)
- [Kaleidoscope 快速入门](kaleidoscope-quickstart.md)
- [状态、快照与克隆](../reference/snapshots.md)
- [约束与安全边界](../operations/constraints.md)
