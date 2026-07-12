# 选择 Runtime 与接口

语言：[中文](choose-runtime-and-api.md) | [English](../../en/getting-started/choose-runtime-and-api.md)

先根据仿真的结构选择 runtime，再根据调用方的所有权边界选择 JSON 或 Python。Single Scene 和 Tiled Scene
都支持多机器人；真正的选择条件是一个共享 World，还是具有行维度的克隆环境。

## 按任务选择

| 任务 | Runtime 或工具 | 接口 | 原因 |
| --- | --- | --- | --- |
| 在一个物理 World 中协调一个或多个机器人和对象 | Single Scene | Single Scene JSON | 共享整数 tick timeline 在一次 world step 前应用所有机器人 target |
| 运行一个或多个机器人，但不需要 env batch 维度 | Single Scene | Single Scene JSON | Single Scene 的机器人数量与 runtime 选择无关 |
| 对 selected cloned env 应用相同或逐行不同的命令 | Tiled Scene | Tiled Scene JSON | `env_ids` 保留显式 batch 行维度 |
| 在每个 cloned env 内运行多个机器人 | Tiled Scene | Tiled Scene JSON | Env selector 与 robot selector 处理不同维度 |
| 执行固定时长的 batch 末端运动 | Tiled Scene | Tiled Scene `step` | 同步 `ee_*` action 使用 Tiled batch IK 路径 |
| 提交不阻塞 physics 的规划并随后回放轨迹 | Tiled Scene | `plan`、`planner_status`、`step_trajectory` | Planner work 与 playback 使用独立的有界生命周期 |
| 执行带显式 arm/hand group track 的跨机器人序列 | Single Scene | `plan_timeline` | Single Scene 把所有 track 编译到共同 tick 轴 |
| 捕获或恢复持久逻辑状态 | Single Scene 或 Tiled Scene | Snapshot JSON 或 `linkerbot_sim.snapshots` | 两种 adapter 使用 `linkerbot.snapshot` schema |
| 观察状态、marker 或相机 | Single Scene 或 Tiled Scene | Foxglove live 或 MCAP | Telemetry 只观察 runtime，不是控制 endpoint |
| 不启动 Isaac，检查配置解析结果 | 不选择 runtime | `validate_config.py` | Pure-Python graph validation 解析 profile 并报告 effective runtime |
| 生成内置 rope 或 T block USD asset | 离线工具 | `build_asset.py` | Asset authoring 与仿真 runtime 构建分离 |
| 把完整 runtime 嵌入另一个进程内应用 | Single Scene 或 Tiled Scene | Python runtime facade | 调用方承担主线程、启动和关闭职责 |
| 在没有运行 World 时使用 planning DTO、snapshot、trajectory 或配置解析 | 不选择 runtime | Pure-Python facade | 这些数据/领域层自身不会创建 Isaac |

不要因为 env 只有一个机器人就选择 Single Scene，也不要因为 env 包含多个机器人就选择 Tiled Scene。只有当
cloned environment 行和 `env_ids` 是问题的一部分时才选择 Tiled Scene。

## 选择 JSON 或 Python

| 调用方边界 | 选择 | 契约 |
| --- | --- | --- |
| 另一个进程、shell pipeline、测试工具或大模型生成的客户端 | JSON | 通过 stdin、TCP JSONL 或 WebSocket 使用严格消息 schema |
| 不应持有 Isaac 对象的 service | JSON | Runtime 进程保留仿真线程和关闭所有权 |
| 使用已校验 planning/snapshot DTO 的进程内算法 | Python facade | 调用方从显式 public facade 导入并处理返回的 error/result |
| 创建 Single Scene 或 Tiled Scene runtime object 的进程内应用 | Python runtime facade | 调用方从 workspace 运行，在主线程创建并关闭所有返回资源 |
| 一次性配置检查 | `validate_config.py` | 不创建 Isaac import、GPU context、stage 或 transport |

JSON 是 canonical 外部控制边界。Python 是进程内集成边界，不是另一套可安装 SDK。运行时会从
checkout 解析 profile、asset、script 和 task resource，因此仓库必须保持可用。

## JSON 控制 Transport

Single Scene 与 Tiled Scene 使用各自的消息 dialect，但通过相同 transport family 暴露：

| Transport | Framing | 用途 |
| --- | --- | --- |
| stdin/stdout | 每行一个严格 JSON object | 本地进程控制和 shell automation |
| TCP JSONL | 每行一个严格 JSON object，每个 request 一个直接 response | 具有明确 request/response framing 的简单本地客户端 |
| WebSocket | 每条 text message 一个 JSON object | Browser/async client 和有界 event delivery |
| Foxglove live | Foxglove telemetry protocol | 仅用于状态/相机可视化，不接受控制 JSON |

Binary WebSocket message 会被拒绝。Listener 地址必须是 loopback；应用不提供认证或 TLS。远程访问
应使用认证 proxy 或 SSH tunnel。Transport limit、connection admission、message size 和 queue
overflow 行为来自 runtime profile。

Single Scene 命令见 [Single Scene JSON 参考](../reference/single-scene-json.md)，Tiled Scene 命令见
[Tiled Scene JSON 参考](../reference/tiled-scene-json.md)。不要把 Single Scene 消息发送给 Tiled Scene handler，也不要猜测
缺失的 runtime selector。

## Python Facade 边界

顶层 `linkerbot_sim` package 有意只导出 `REPO_ROOT`。应从有文档的 package facade 导入，不要依赖
传递导入。只有 Python 参考明确列出具体 symbol 及其 lifecycle 契约时，高级 owner-module symbol
才是集成入口。

| 需求 | 已文档化 package facade |
| --- | --- |
| Backend-neutral planning request/result type | `linkerbot_sim.planning` |
| cuRobo config、context、FK/IK、planning 和 adapter | `linkerbot_sim.backends.curobo` |
| Canonical snapshot schema、compatibility 和 runtime dispatch | `linkerbot_sim.snapshots` |
| Controller type 和 `JointController` | `linkerbot_sim.controllers` |
| Command execution step | `linkerbot_sim.execution` |
| Object profile/runtime type | `linkerbot_sim.objects` |
| Robot capability 与 joint-group type | `linkerbot_sim.robots` |
| Sensor 与 camera 配置 type | `linkerbot_sim.sensors` |
| Single Scene 交互入口与循环 | `linkerbot_sim.app.interactive.single_scene` |
| Tiled Scene runtime 创建和消息 dispatch | `linkerbot_sim.app.interactive.tiled_scene` |

导出的 facade 不会免除 lifecycle 要求：

1. 从 checkout 根目录以 `PYTHONPATH=src` 运行。
2. 只有接受 NVIDIA/Kit EULA 后才设置 `OMNI_KIT_ACCEPT_EULA`。
3. 构造 runtime 前完成配置解析。
4. 在仿真主线程创建和修改 Isaac runtime object。
5. 只把冻结的 Python/NumPy data 交给 background worker。
6. 释放 Kit 前关闭 transport/publisher 和 runtime。

以 `_` 开头的名称、test fake、runtime service 子模块，以及没有被文档明确列为集成点的 helper
都是 implementation detail。Package `__all__` 只记录导出项，不能单独形成支持承诺。Pure parser
可以在没有 Isaac 时运行；会修改 Single Scene 或 Tiled Scene state 的 runtime handler 必须在仿真主线程运行。

[Python Facade 参考](../reference/python-api.md)是受支持 import、精确符号、签名和生命周期要求的
唯一完整清单。[运动规划指南](../guides/motion-planning.md)负责 planning 行为；
[Snapshot 参考](../reference/snapshots.md)负责共享 payload、匹配与事务契约，runtime 参考页只拥有
各自消息外层和 selector。

## 校验配置

启动 Isaac 前，或修改任意被引用 profile 后，运行 validator：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_single_scene

PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene
```

Validator 遍历 runtime、env、per-env fragment、robot、controller bundle、object、logging，以及
每个启用 planning 的机器人合并后的 cuRobo binding。它不会启动 `SimulationApp`。以下命令可以检查
每个 effective runtime leaf 及其来源：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene \
  --dump-effective-config
```

详见[配置参考](../reference/configuration.md)。

## 运行交互入口

环境准备并接受 EULA 后，使用 Single Scene runtime profile 启动 Single Scene：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene
```

只使用 Tiled Scene runtime profile 启动 Tiled Scene：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene
```

`--dump-effective-config` 会在 Isaac 启动前退出。显式 CLI 值只覆盖本次启动中对应的 runtime 字段；
省略的 CLI 值保留所选 runtime profile 的设置。

第一次完整运行请使用 [Single Scene 快速入门](single-scene-quickstart.md)或
[Tiled Scene 快速入门](tiled-scene-quickstart.md)。全部启动选项分别由 [Single Scene CLI](../reference/single-scene-cli.md)和
[Tiled Scene CLI](../reference/tiled-scene-cli.md)参考定义。

## 生成或预览资产

Runtime 入口只消费已有 asset，永远不会自动重建。使用离线入口生成内置资产：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python \
  tools/object_assets/flexible/rope/build_asset.py

PYTHONPATH=src .venv/bin/python \
  tools/object_assets/rigid/tblock/build_asset.py
```

两个 builder 都接受 `--config` 和 `--output`，启动 headless Isaac 写入 USD/PhysX schema，保存 asset
后关闭 app。生成几何由 `tools/object_assets` 所有；运行时 placement 和 physics 由 `configs/objects`
与 `configs/envs` 所有。

详见[物体资产](../development/object-assets.md)、[USD 预览](../development/usd-preview.md)和
[碰撞近似](../development/collision-approximation.md)。

## 决策检查表

实现前按顺序回答：

1. 需要一个物理 World，还是 selected cloned env row batch？
2. 调用方需要进程隔离，还是会在进程内持有 Isaac？
3. 每个待修改设置由哪个 profile 所有？
4. 所选消息要求哪些 robot 和 env selector？
5. 请求需要 direct control、IK、joint planning，还是 collision-aware planning？
6. 输出受哪个有界 queue、file policy 和 shutdown timeout 约束？
7. 哪个 terminal response 能证明完成，哪种 response 要求重建 runtime？

完成选择后，应遵循对应 runtime 的精确 reference，不要从另一套 runtime 的案例推导消息。

## 继续阅读

- [项目总览](project-overview.md)
- [Single Scene 快速入门](single-scene-quickstart.md)
- [Tiled Scene 快速入门](tiled-scene-quickstart.md)
- [配置指南](../guides/configuration.md)
- [Single Scene CLI 参考](../reference/single-scene-cli.md)
- [Single Scene JSON 参考](../reference/single-scene-json.md)
- [Tiled Scene CLI 参考](../reference/tiled-scene-cli.md)
- [Tiled Scene JSON 参考](../reference/tiled-scene-json.md)
- [Python Facade 参考](../reference/python-api.md)
- [运动规划](../guides/motion-planning.md)
- [遥测](../guides/telemetry.md)
- [已知约束](../operations/constraints.md)
