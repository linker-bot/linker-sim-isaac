# 贡献指南

语言：[中文](CONTRIBUTING_zh.md) | [English](CONTRIBUTING.md)

欢迎在保持产品边界、运行时所有权和设备边界的前提下参与贡献。修改公开行为前，请先阅读
[项目概览](docs/zh-CN/getting-started/project-overview.md)和
[产品与接口选择](docs/zh-CN/getting-started/choose-runtime-and-api.md)。

## 提交变更前

- 先检索现有 Issue 和 Pull Request。
- 如果变更涉及公开 facade、配置 schema、wire protocol、Kit closure、physics owner 或第三方
  素材策略，建议先用一个聚焦的 Issue 明确边界。
- 保持 Mirror 与 Kaleidoscope 能力分离。没有明确架构决策时，不要为 Kaleidoscope 增加
  camera、transport、planner、telemetry worker 或其他由 Mirror 拥有的服务。
- 安全问题按 [SECURITY.md](SECURITY.md) 私下报告，不要创建公开 Issue。

## 开发环境

仿真环境与 CPU 开发环境必须分别安装：

```bash
uv sync --extra simulation --extra visualization --extra training
UV_PROJECT_ENVIRONMENT=.venv-dev \
  uv sync --extra dev --extra visualization
```

不要混用 `dev` 与 `simulation`，也不要使用 `--all-extras`。开发环境中的 PyPI
`usd-core` 不能遮蔽仿真环境里由 Kit 提供的 `pxr`。完整说明见
[安装与环境准备](docs/zh-CN/getting-started/installation.md)。

## 保持变更聚焦

- 优先使用 `linkerbot_sim.configuration`、`linkerbot_sim.isaac`、
  `linkerbot_sim.mirror`、`linkerbot_sim.kaleidoscope` 和
  `linkerbot_sim.training.skrl` 的公开 facade。
- 配置事实只写在对应 owner leaf 中，不重复声明 device、engine、环境数量或 output。
- Isaac、USD、physics、camera 和 planner 资源继续由 owner thread 访问。
- Kaleidoscope 原生 state 与训练数据继续保留在所选 CUDA device。
- 保留现有代码中用于解释所有权、生命周期、设备和物理边界的有效注释；增加新边界时补充简洁说明。
- 不要提交仓库明确排除的 NVIDIA Warehouse 或其他第三方素材。

## 文档

中英文文档使用相同的相对路径。修改公开行为时，需要在同一个 Pull Request 中同步更新两棵
文档树；代码、参数名、operation、字段名和事实表应保持一致。

通过质量门禁运行维护的链接检查。不要为已经删除的产品名称添加重定向页面，也不要把内部 helper
写成稳定公开 API。

## 验证

所有变更至少运行 CPU 门禁：

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev just quality
```

该门禁只统计 `architecture/module_disposition.yaml` 中标记为 `runtime: pure` 的生产模块，
并执行 `tool.linkerbot_sim.coverage.pure_fail_under` 阈值。覆盖范围直接来自架构清单，既不会
因为 CPU 环境缺少 Isaac/CUDA import 而误报，也不依赖任意维护的文件排除列表；因此修改模块
runtime 标签会同步改变覆盖范围，必须接受与模块图其他变更相同的架构审查。独立的
`just test` 使用仿真环境，并继续执行 `tool.coverage.report` 中面向全源码的覆盖率阈值。

移动源码模块后，刷新并检查架构清单：

```bash
just update-architecture
just test-architecture
```

修改 Kit composition、Isaac lifecycle、物理后端、CUDA tensor、cuRobo、渲染、相机或仿真资产时，
还需要在兼容的 NVIDIA 主机上运行：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
just test-simulation
```

如果完整仿真矩阵在一次迭代中成本过高，应先运行最相关的窄门禁，并在 Pull Request 中准确列出
尚未执行的 GPU 检查。
维护者也可以通过受信任的 [Simulation CI 工作流](docs/zh-CN/operations/simulation-ci.md)
运行同一套矩阵。该工作流刻意不响应 Pull Request 事件；手动运行时只能选择已经审查的仓库内
分支，并在 Pull Request 中附上对应的 Actions 链接。

## Pull Request 检查表

- 变更范围单一、明确，并说明所属产品边界。
- 公开 API、配置或 wire 变更包含测试和双语文档。
- `just quality` 通过。
- 相关仿真 smoke 通过；未执行时说明具体原因。
- 新增的仓库内 Markdown 链接可解析。
- 移动模块后架构清单已更新。
- 未提交生成输出、本地环境、凭据、内部路径或被排除的第三方素材。
- 许可与归属变化已同步到 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
