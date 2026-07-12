# 文档组织与提交指南

语言：[中文](documentation-guide.md) | [English](../../en/maintenance/documentation-guide.md)

本文面向项目维护者，规定 `docs/` 的内容边界、信息架构、事实所有权和校对要求。
目标是让人类用户与大模型都能从统一入口选择运行形态和调用方式，并能沿链接找到唯一、完整、
与源码一致的说明。

## 1. 范围与读者

项目文档服务三类读者：

- 使用 CLI、YAML 和 JSON 控制仿真的应用用户。
- 直接调用项目明确列出的 Python facade 的算法与工具开发者。
- 修改 runtime、规划、快照、遥测、相机、资产或配置实现的维护者。

仓库根目录的 README 只负责项目定位、最小环境准备、最小启动命令和文档入口。完整字段表、状态机、数据结构、
默认值和错误语义必须由 `docs/` 中对应的唯一参考页拥有。

## 2. 用户可依赖的接口边界

用户可依赖的接口只有文档明确列出的以下表面：

- Single Scene、Tiled Scene 和配置校验入口的 CLI：`scripts/single_scene_interactive.py`、
  `scripts/tiled_scene_interactive.py` 和 `scripts/validate_config.py`。
- 文档列出的离线资产工具 CLI：capsule rope 与 T block 的 `build_asset.py`。
- 配置指南和配置参考列出的项目 YAML profile 结构、所有权、校验规则与 CLI 覆盖优先级。
- Single Scene 与 Tiled Scene JSON 参考列出的请求、响应、selector、状态机和 transport 语义。
- Python 参考中明确命名的 facade、函数、类型、参数、返回值、异常和生命周期约束。

`src/linkerbot_sim/` 的全量模块图用于导航实现，不等于全量公共 Python API。模块存在、符号未以下划线
开头或模块定义了 `__all__`，都不能单独证明它是用户接口。没有被 Python 参考明确列出的模块、类和
函数按内部实现处理；维护文档可以解释其职责和调用关系，但不得把它们描述为用户可依赖的稳定入口。

Python 页面必须为每个列出的入口标明运行前提：

- `pure`：普通项目 Python 环境可导入，不启动 Isaac Sim。
- `Isaac main thread`：必须在 Kit/Isaac 已启动且线程条件满足时调用。
- `cuRobo/CUDA`：依赖项目锁定的 GPU、Torch、Warp 和 cuRobo 运行环境。

本项目是 checkout workspace 应用，不是可安装 Python 库。Python 示例从仓库根目录以
`PYTHONPATH=src` 运行；这项运行方式不会扩大用户接口范围。

## 3. 运行形态与调用链

Single Scene 表示一个 `SingleSceneRuntime`，并不表示场景只能包含一个机器人。Tiled Scene 表示由同构模板
克隆出的并行环境，并提供按 env 选择的控制、状态、轨迹和规划接口。

```text
可选预检：runtime YAML + referenced profiles
  -> validate_config strict parsing and complete profile-graph validation
  -> validation report；不启动 Isaac

runtime CLI：runtime YAML + referenced profiles + 显式 CLI overlay
  -> strict effective-runtime resolution
  -> SingleSceneRuntime or Tiled Scene runtime composition

Single Scene JSON client
  -> stdin / TCP JSONL / WebSocket
  -> Single Scene protocol and timeline compiler
  -> SingleSceneRuntime
  -> controllers, execution and Isaac World

Tiled Scene JSON client
  -> stdin / TCP JSONL / WebSocket
  -> Tiled Scene protocol, selectors and command routing
  -> control, state, trajectory playback or asynchronous planning
  -> Isaac batched views and optional cuRobo services

Documented Python caller
  -> one explicitly documented domain facade
  -> its declared runtime and resource boundary
```

JSON 是进程级控制协议；Python facade 是进程内领域调用。YAML 描述启动配置和场景事实，不是运动
命令。文档入口必须先帮助读者选择 Single Scene 或 Tiled Scene，再选择 JSON 或 Python，不能把这两个维度混为
一类选项。

## 4. 文档类型与目录职责

中英文目录使用相同的 ASCII 相对路径和相同的主题边界：

| 目录 | 职责 |
| --- | --- |
| `getting-started/` | 项目总览、运行形态与接口选择、可完成的最小端到端案例 |
| `guides/` | 面向任务的配置、规划、遥测、相机和其他操作说明 |
| `reference/` | CLI、JSON、配置和数据结构的穷举式契约 |
| `operations/` | 故障定位、运行约束、安全边界和容量边界 |
| `development/` | 源码领域导航、命名、资产生成、碰撞和预览工具 |
| `maintenance/` | 文档所有权、提交边界和校对政策 |

页面开头或第一个相关章节必须明确范围与必要运行前提。教程可以提供最小示例，但不得复制参考页中的
完整字段表。参考页应给出精确名称、类型、单位、shape、坐标系、默认值、终态和拒绝条件。

各类文档采用以下验收边界：

- 快速入门必须形成一条可执行链，依次覆盖 checkout 准备、完整配置校验、EULA 接受、ready、
  discovery、一个最小有效操作、终态检查和进程正常退出。
- CLI 参考必须覆盖 parser 的全部 option，包括成对布尔 flag 的两种形式；同时区分 option 省略时
  的 argparse 值和内置 profile 提供的最终值。
- JSON 参考只拥有 framing、消息、selector、状态转换与响应；完整 CLI 表和 YAML 字段表应链接
  到各自所有者，不在协议页复制。
- Selector 清单来自 parser 与 dispatcher，并包含每个要求显式 env selector 的命令。数值关闭
  哨兵和非法值范围必须分别说明，不能合并成一个模糊范围。

源码模块图按领域列出模块职责、运行前提和对应文档。它应覆盖全部 `src/linkerbot_sim/**/*.py`，
同时把每项标为“已文档化 facade”“明确支持的 owner path”或“内部实现”，避免导航覆盖率被误读为
接口承诺。

## 5. 事实的唯一文档所有者

同一字段表、默认值、状态机或持久化格式只能有一份详细文档所有者。其他页面只保留完成当前任务
所需的最小示例，并链接到所有者。

| 事实 | 源码所有者 | 当前文档所有者 |
| --- | --- | --- |
| Single Scene 启动选项、配置覆盖和进程 marker | `app.interactive.single_scene.cli` | [Single Scene CLI 参考](../reference/single-scene-cli.md) |
| Single Scene transport、命令、timeline 与响应 | `app.interactive`、`app.motion.timeline` | [Single Scene JSON 参考](../reference/single-scene-json.md) |
| Tiled Scene 启动选项、配置覆盖和进程 marker | `app.interactive.tiled_scene.cli` | [Tiled Scene CLI 参考](../reference/tiled-scene-cli.md) |
| Tiled Scene transport、selector、step、state、trajectory 与 planner | `app.interactive.tiled_scene`、`tiled` | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| 控制路径选择、时间、关节顺序和 playback 生命周期 | `app.motion.timeline`、`tiled.control`、`tiled.playback` | [控制与轨迹指南](../guides/control-and-trajectories.md) |
| Profile 分层、引用、CLI 覆盖和常用配置任务 | `configs.runtime`、各领域 config 模块 | [配置指南](../guides/configuration.md) |
| 项目 YAML 字段、per-env fragment、所有权与完整依赖图校验 | `configs`、各领域 config 模块 | [配置参考](../reference/configuration.md) |
| 规划 backend、frame、cuRobo binding、batch 与碰撞能力 | `planning`、`backends.curobo` | [运动规划指南](../guides/motion-planning.md) |
| PhysX、规划和 env 间碰撞层选择 | 资产 importer、planning collision provider、`tiled.scene.collision_filter` | [碰撞模型](../guides/collision-models.md) |
| Snapshot 数据、身份匹配、capture/restore、事务和失败语义 | `snapshots` | [Snapshot 参考](../reference/snapshots.md) |
| Telemetry topic、payload、env 选择、采样与 live publication 语义 | `telemetry` | [遥测指南](../guides/telemetry.md) |
| Sensor camera、frame、modality、capture 与 attachment 语义 | `sensors.camera` | [相机指南](../guides/cameras.md) |
| CSV、MCAP、camera 编码与 metadata、路径、已有数据策略、队列、配额和持久化输出关闭 | logging、telemetry 和 camera output owner | [输出参考](../reference/outputs.md) |
| 运行安全、线程、资源、仿真和配置约束 | 各领域 owner | [运行约束](../operations/constraints.md) |
| 症状到 owner 的诊断与恢复边界 | 各领域 owner | [故障排查](../operations/troubleshooting.md) |
| 资产、关节、link、body 和 profile 命名 | `assets`、`robots`、`objects` | [命名规范](../development/naming.md) |
| 物体资产生成与接入 | `objects`、资产生成脚本 | [物体资产](../development/object-assets.md) |
| Importer 碰撞近似与 USD/PhysX 含义 | 资产 importer | [碰撞近似](../development/collision-approximation.md) |
| USD 资产预览 | Isaac 资产工具 | [USD 预览](../development/usd-preview.md) |

Python 的导入路径、签名、类型、shape、单位、线程、资源和异常由一份 Python 参考拥有；模块图
只负责导航源码领域。对于每个返回资源的调用，Python 参考还必须写明 handle 支持的关闭方法、
启动成功前后的所有权、关闭返回值和可重试条件。领域状态机、匹配规则、时间策略和持久化格式继续
由领域所有者负责，Python 参考只链接，不重复定义。

## 6. 中英文一致性

`docs/en/` 与 `docs/zh-CN/` 使用一一对应的路径、章节职责和接口语义。两种语言必须同步以下事实：

- CLI 名称、选项、默认值和互斥关系。
- YAML 与 JSON 字段、必填条件、类型、枚举和拒绝语义。
- Python 导入路径、签名、返回类型、shape、单位、坐标系和生命周期。
- 每个已文档化 facade 的全部当前导出，以及明确支持的高级 owner symbol；两种语言的符号集合
  必须相同。
- Topic、文件路径、payload、metadata、容量和关闭行为。
- 错误码、终态、超时、背压、事务和 fail-stop 条件。

教程性文字可以按语言习惯组织，但不能删去会改变使用方式的约束。每对页面应提供语言切换链接；
索引中的描述和阅读路径也必须同构。

## 7. 应提交的内容

以下内容属于仓库的长期文档，应与对应代码或配置修改一起校对并提交：

- README、语言索引、入门、指南、参考、运维、开发和维护页面。
- 当前 CLI、YAML、JSON、Python facade 与持久化数据契约。
- 能防止误用或回归的架构边界、资源所有权、线程要求和运行约束。
- 可执行、可严格解析并与项目实际入口一致的示例。
- 源码模块图及其接口分类、运行前提和事实所有者链接。
- 与配置资源同目录、解释该资源用途和边界的 README。

文档应描述仓库当前可观察行为；接口清单与示例只包含当前实现接受的名称和字段，不记录个人
工作日志、本机状态或实现过程。

## 8. 不提交的内容

以下内容不进入 `docs/` 或 Git：

- `docs/_build/`、`docs/tem/`、站点输出和可重新生成的 HTML/API 页面。
- 运行进程产生的 session output，包括 `logs/`、MCAP、关节 CSV、相机帧目录、运行时
  `metadata.jsonl` 以及同一次运行目录中的其它数据。
- `design_plan/` 中未成为当前项目契约的分析、草稿和试验脚本。
- Notebook checkpoint、编辑器文件、工具缓存、本地 agent 状态和临时文件。
- 凭据、token、私有地址、用户数据、本机绝对路径和安全处置记录。
- 瞬时测试数量、磁盘占用、个人执行时间线和一次性审查输出。

自动生成的 API 站点可以作为本地或 CI 产物使用，但仓库保留源码 docstring、手写模块图和领域参考，
不提交可重建站点。

这条规则按来源和所有权判断，不按文件扩展名全局禁止。被测试直接消费且已最小化的数据可以放在
`tests/data` 或明确的 fixture 目录；有来源与用途说明的正式资产可以放在 `assets/`。不能把一次性
运行目录改名后当作 fixture 或正式资产提交。

## 9. 源码变化对应的文档检查

| 源码或配置变化 | 必须检查的文档所有者 |
| --- | --- |
| Single Scene CLI 或 overlay | Single Scene CLI 参考和 Single Scene 快速入门 |
| Single Scene parser、queue、transport 或 timeline | Single Scene JSON 参考、Single Scene 快速入门、运行约束 |
| Tiled Scene CLI 或 overlay | Tiled Scene CLI 参考和 Tiled Scene 快速入门 |
| 配置校验器 parser、输出或退出行为 | 配置参考和 runtime 选择页 |
| Rope 或 T block 资产构建器 parser 或输出 marker | 物体资产开发指南 |
| Tiled Scene selector、action、state、trajectory 或 planner routing | Tiled Scene JSON 参考、Tiled Scene 快速入门、运动规划指南 |
| Runtime 或领域 YAML parser、默认值、校验与 CLI overlay | 配置指南、配置参考、示例 profile |
| Planning request/result、frame、joint order、shape 或单位 | 运动规划指南；若属于已列出的 facade，同时检查 Python 参考 |
| Snapshot capture、restore、identity 或 transaction | Snapshot 唯一参考及 Single Scene/Tiled Scene envelope 章节 |
| Telemetry topic、payload、采样、背压或 MCAP | 遥测指南和输出唯一参考 |
| Camera frame、编码、recorder、sink、路径或容量 | 相机指南和输出唯一参考 |
| CSV 列、采样、刷盘或已有文件策略 | 输出唯一参考和配置指南 |
| 线程、队列、超时、shutdown 或 fail-stop | 对应 runtime 参考和运行约束 |
| Python facade 的导入、签名、类型、异常或资源所有权 | Python 唯一参考和模块图 |
| 内部模块新增、删除、移动或职责变化 | 模块图；只有被列为 facade 时才更新 Python 参考 |
| 资产 importer、命名、碰撞或预览工具 | 对应 development 页面和配置指南 |

代码评审不能只检查被直接修改的文件名，还要沿事实所有者关系检查所有外部可观察行为。

## 10. 提交前验证

从 checkout 根目录执行：

```bash
git status --short --untracked-files=all README*.md docs configs
git diff --check -- README*.md docs configs
PYTHONPATH=src .venv/bin/python scripts/check_markdown_links.py
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_markdown_links.py \
  tests/test_documented_module_map.py \
  tests/test_documentation_contracts.py
.venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
.venv/bin/python scripts/validate_config.py --runtime-profile default_tiled_scene
```

人工校对还必须确认：

- 所有索引和正文链接指向 Git 跟踪的文件，语言切换链接成对存在。
- JSON/JSONL 示例是严格 JSON；YAML 示例无重复 key，并符合当前 parser。
- CLI 表覆盖全部成对 flag，与对应入口的 `--help` 一致，并区分省略时的 parser 值和内置
  profile 最终值。
- 快速入门完成配置校验、启动、发现、操作、终态和关闭整条链；协议 selector 清单与 parser、
  dispatcher 一致。
- Python 示例的导入、签名、shape、单位、坐标系、线程和关闭方式与源码一致。
- Python 参考逐项列出 facade 导出和受支持 owner symbol，并写明启动失败所有权及返回 handle 的
  精确关闭方法与返回结果。
- 每项详细事实只有一个文档所有者，其他页面使用链接而不是复制完整定义。
- 模块图覆盖全部源码模块，并正确区分已文档化 facade、受支持 owner path 与内部实现。
- 中英文的字段、默认值、错误和生命周期语义一致。
- 文档不包含凭据、本机数据、运行产物或不可复现的个人环境结论。
