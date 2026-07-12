# 项目总览

语言：[中文](project-overview.md) | [English](../../en/getting-started/project-overview.md)

LinkerHand Simulation 是一个基于 checkout workspace 运行的 Isaac Sim 应用，用于机器人控制、
运动规划、并行环境执行、状态捕获和传感器输出。项目提供两套执行契约不同的 runtime：

- Single Scene 模式运行一个物理 World，可以协调所选 env profile 声明的任意数量机器人。
- Tiled Scene 模式构建克隆环境，并对显式选择的环境行批量执行机器人和对象操作。

两种模式都支持严格 YAML 配置、robot ID 发现、JSON 控制、cuRobo 集成、canonical snapshot、
Foxglove/MCAP 遥测和传感器相机。Single Scene 还提供共享 tick 的多机器人 timeline 和逐机器人关节跟踪
CSV；Tiled Scene 还提供 batch step action、逐环境状态操作、trajectory buffer 和具有资源上限的异步规划。

选择入口或 Python 集成面之前，先阅读[选择 Runtime 与接口](choose-runtime-and-api.md)。

## 执行模型

| 边界 | Single Scene | Tiled Scene |
| --- | --- | --- |
| Runtime owner | `SingleSceneRuntime` | `TiledSceneRuntime` |
| 拓扑 | 一个 World，包含配置的机器人和对象实例 | 一个 source env 克隆为 `tiled.num_envs` 行 |
| 机器人数量 | 同一场景中的一个或多个机器人 | 每个克隆环境中的一个或多个机器人 |
| 选择器 | 会话级 `robot_id` | 显式 `env_ids`，以及接口要求的会话级 `robot_id` 或 `robot_ids` |
| 同步执行 | 整数 tick robot timeline，每个 tick 共享一次 `world.step()` | batch 固定 tick `step` action 和 trajectory-buffer 回放 |
| 规划 | Single Scene compiler 使用 `curobo` 或关节空间 `linear` backend | 末端 action 使用同步 batch IK，另有独立异步 planner manager |
| 状态操作 | Single Scene reset 和 Single Scene snapshot 读取/恢复 | 逐环境 reset、调试状态、snapshot 广播和 env-to-env clone |
| 关节 CSV | 通过所选 logging profile 支持 | Tiled Scene 入口不创建该 logger |

Single Scene 不等于单机器人。Single Scene 所选的 env profile 可以包含多个机器人实例，一条 timeline 可以让它们
从共同 tick 0 协同执行。Single Scene 没有克隆环境的 `env_id` 维度。

Tiled Scene 不是 `SingleSceneRuntime` 的包装层。它有独立的 runtime 类、scene builder、command adapter、
state shape、trajectory buffer 和 planner manager。一个 Tiled env 仍可包含多个机器人；
`env_ids` 选择克隆行，robot ID 选择这些行中的机器人。

## 共享边界

当数据语义一致时，两套 runtime 复用以下领域契约：

- Runtime profile 拥有进程策略、所选领域 profile、资源上限、遥测、输出生命周期和关闭超时。
- Env、robot、controller、object、cuRobo 和 logging profile 使用严格仓库 schema，并在 Isaac
  启动前校验。
- 控制使用会话级数字 robot ID，持久匹配使用稳定 label、profile 和 fingerprint。
- Planning request/result 在进入 `linear` 或 cuRobo adapter 前使用 backend-neutral DTO。
- `linkerbot.snapshot` 是 Single Scene 和 Tiled Scene adapter 共用的 canonical 逻辑快照 schema。
- 坐标值使用 m 和 rad；公开四元数统一使用 `wxyz`。
- 状态遥测、相机输出和文件 target 使用有界队列和显式输出策略。

共享这些契约并不表示 runtime-specific JSON 可以互换。必须使用所选 runtime 的消息 schema。

## 独立边界

以下能力明确由各 runtime 独立拥有：

- Single Scene timeline 与 Tiled Scene action 使用不同 envelope、selector、response 字段和状态机。
- Single Scene state 描述一个 World；Tiled Scene state 保留显式 selected-env 行维度。
- Single Scene 按 command ID 跟踪命令状态；Tiled Scene trajectory playback 和异步 planning 各有独立生命周期接口。
- Scene collision coordination 可以把其它机器人冻结为静态规划障碍；Tiled Scene planning 把 selected env
  行作为 request problem，不暴露 Single Scene coordination 字段。
- Tiled Scene articulation 使用 position control；请求 Tiled Scene velocity 或 effort control 的 runtime profile
  会在配置解析时被拒绝。
- Single Scene 与 Tiled Scene 持有不同 Isaac 对象，必须分别通过自己的 lifecycle facade 关闭。

精确消息见 [Single Scene JSON 参考](../reference/single-scene-json.md)和
[Tiled Scene JSON 参考](../reference/tiled-scene-json.md)。

## Workspace 要求

本仓库是 workspace 应用，不是可独立安装的 Python 库。运行时依赖 checkout 内的 `configs/`、
`assets/`、`scripts/` 和 cuRobo task 资源。本地 build backend 会拒绝 wheel、source distribution
和 editable build。

声明的运行环境是 Linux x86-64 和 Python 3.11。从 checkout 根目录创建统一环境：

```bash
uv sync --all-extras
```

项目命令必须在同一根目录以 `PYTHONPATH=src` 运行。启动 Isaac Sim 前，先阅读并接受适用的
NVIDIA/Kit EULA，再在部署环境记录该接受状态：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

应用以大小写不敏感方式接受 `Y`、`YES` 或 `1`。项目不会替用户设置该变量或接受 EULA；缺少
接受状态时会在创建 `SimulationApp` 之前失败。

## 配置所有权

| 位置 | 唯一职责 |
| --- | --- |
| `configs/runtime/` | Runtime 模式、所选 profile、GUI/GPU/render、执行默认值、transport、planner/playback 上限、遥测、相机输出策略、路径和 shutdown |
| `configs/envs/` | World 设置、可视场景、传感器摆放、机器人/对象实例，以及可选 Tiled Scene 拓扑和逐环境 pose override |
| `configs/robots/` | Isaac model、robot kind、joint group、importer/PhysX 设置，以及可选 cuRobo model/TCP binding |
| `configs/controllers/` | Position、velocity、effort 控制方法、关节、gain、limit 和 PhysX drive override |
| `configs/objects/` | Object asset、object kind、运行时 physics/material、state summary 和可选简化 planning collision |
| `configs/curobo/` | Device、task bundle、IK/planner 算法、seed、tolerance、collision cache 和 batch capacity |
| `configs/logging/` | Single Scene 关节跟踪 CSV 开关、路径、采样、flush interval 和列 |
| `tools/object_assets/` | 离线生成资产的几何和写入 USD/PhysX 的属性 |

Runtime 解析顺序是代码默认值、所选 runtime YAML、最后是本次启动显式提供的 CLI 字段。Env profile
只拥有场景事实，不拥有 planner、transport、telemetry 或进程资源策略。所有权案例见
[配置指南](../guides/configuration.md)，校验行为见[配置参考](../reference/configuration.md)。

## 运行边界

### 网络与输入

stdin、TCP JSONL 和 WebSocket 控制路径只接受严格 JSON object。未知字段、重复 YAML key、非有限
JSON 常量、非法 selector 和越界值会被拒绝，不会由运行时猜测。

所有内置控制、状态遥测和相机 live listener 都限制在 loopback。应用不提供认证或 TLS。远程访问
必须使用认证 TLS proxy 或 SSH tunnel，并让 upstream 仍绑定 loopback。Foxglove live 是遥测协议，
不是 JSON 控制 transport。

### 线程所有权

Isaac stage object、articulation/PhysX view、Camera wrapper 和 runtime mutation 只归仿真主线程所有。
Transport 和 planner worker 可以解析消息或消费冻结的 NumPy/Python snapshot，但不能读写 Isaac
对象。文件和遥测 publisher 只接收已经捕获的 immutable data。

### 资源与输出上限

Transport connection、request/event queue、snapshot request、planner work、completed planner
summary、trajectory buffer、telemetry buffer、camera queue 和每相机目录字节数都有显式上限。
溢出行为是拒绝、背压、替换或已声明的 drop policy，不存在无界队列。

CSV、MCAP 和 camera target 会在任何 writer 打开前联合规划和检查。已有数据必须显式选择
`error`、`truncate`、`resume` 或 `timestamped_dir`，具体 sink 可以拒绝自身无法安全实现的策略。
任务流程见[遥测](../guides/telemetry.md)和[相机](../guides/cameras.md)，精确文件与 payload 契约见
[输出参考](../reference/outputs.md)。

### Mutation 与 Fail-Stop

Reset、Tiled Scene `set_state`、snapshot restore 和 env clone 会在第一次 mutation 前完成校验并捕获
rollback state。后续 setter 失败时，已完成写入按逆序补偿。完整回滚后 runtime 可以继续使用。

若回滚失败，或不可逆 queue/cache commit 后发生异常，runtime 会记录第一个 fatal reason、请求
shutdown 并拒绝后续 mutation。此时应重建 runtime，不能在 controller/PhysX 一致性无法证明的
状态上继续运行。

### Shutdown

入口会先停止新的 transport 和 publisher admission，再有界等待后台工作，按依赖关系关闭 planner、
camera、logger 及其 sink，释放 IK/planning 资源，最后关闭 `SimulationApp`。独立 timeout 防止一种
资源消耗另一种资源的 shutdown budget。仍存活的 worker 会保留其 sink 或 runtime dependency，
避免并发关闭；owner 可以在释放 Kit 前重试清理。

详细不变量见[已知约束](../operations/constraints.md)。

## 继续阅读

- [选择 Runtime 与接口](choose-runtime-and-api.md)
- [Single Scene 快速入门](single-scene-quickstart.md)
- [Tiled Scene 快速入门](tiled-scene-quickstart.md)
- [配置指南](../guides/configuration.md)
- [Single Scene CLI 参考](../reference/single-scene-cli.md)
- [Single Scene JSON 参考](../reference/single-scene-json.md)
- [Tiled Scene CLI 参考](../reference/tiled-scene-cli.md)
- [Tiled Scene JSON 参考](../reference/tiled-scene-json.md)
- [控制与轨迹](../guides/control-and-trajectories.md)
- [运动规划](../guides/motion-planning.md)
- [碰撞模型](../guides/collision-models.md)
- [Snapshot 数据与恢复](../reference/snapshots.md)
- [遥测](../guides/telemetry.md)
- [相机](../guides/cameras.md)
- [持久化与 Live 输出](../reference/outputs.md)
- [故障排查](../operations/troubleshooting.md)
- [物体资产](../development/object-assets.md)
