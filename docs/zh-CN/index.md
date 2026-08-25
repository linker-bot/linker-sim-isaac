# 文档索引

语言：[中文](index.md) | [English](../en/index.md)

## 从目标开始

| 目标 | 文档 |
| --- | --- |
| 安装锁定环境并准备可选外部素材 | [安装与环境准备](getting-started/installation.md) |
| 理解两个产品、物理 backend 与资源 owner | [项目概览](getting-started/project-overview.md) |
| 选择交互仿真或 GPU 强化学习接口 | [选择 Mirror 或 Kaleidoscope](getting-started/choose-runtime-and-api.md) |
| 启动现实工作站映像 | [Mirror 快速入门](getting-started/mirror-quickstart.md) |
| 构造 Torch/Gymnasium 并行环境 | [Kaleidoscope 快速入门](getting-started/kaleidoscope-quickstart.md) |
| 运行 Mirror 命令行和 transport | [Mirror CLI](reference/mirror-cli.md) |
| 发送严格 JSON 请求，编写关节、IK、规划和双臂同步运动 | [Mirror JSON 与运动示例](reference/mirror-json.md) |
| 调用 Torch、Gymnasium、state 与 clone API | [Kaleidoscope API](reference/kaleidoscope-api.md) |
| 组合 mode、scene、physics、task 与 output profile | [配置指南](guides/configuration.md) |
| 查询精确配置边界 | [配置参考](reference/configuration.md) |
| 选择关节、IK、直线动作或 Mirror 轨迹 | [控制与轨迹](guides/control-and-trajectories.md) |
| 使用 Mirror cuRobo 规划 | [运动规划](guides/motion-planning.md) |
| 捕获状态、恢复 snapshot 或克隆环境 | [状态与快照](reference/snapshots.md) |
| 配置 Mirror 相机 | [相机](guides/cameras.md) |
| 配置 Mirror 遥测和 Foxglove | [遥测](guides/telemetry.md) |
| 排查启动、GPU residency 或关闭故障 | [故障排查](operations/troubleshooting.md) |
| 定位源码 owner | [源码模块图](development/module-map.md) |

## 入门

- [安装与环境准备](getting-started/installation.md)
- [项目概览](getting-started/project-overview.md)
- [选择 Mirror 或 Kaleidoscope](getting-started/choose-runtime-and-api.md)
- [Mirror 快速入门](getting-started/mirror-quickstart.md)
- [Kaleidoscope 快速入门](getting-started/kaleidoscope-quickstart.md)

## 指南

- [配置](guides/configuration.md)
- [控制与轨迹](guides/control-and-trajectories.md)
- [Mirror 运动规划与 cuRobo](guides/motion-planning.md)
- [碰撞模型](guides/collision-models.md)
- [Mirror 遥测](guides/telemetry.md)
- [Foxglove](guides/foxglove.md)
- [Mirror 相机](guides/cameras.md)

## 参考

- [YAML 配置](reference/configuration.md)
- [Mirror CLI](reference/mirror-cli.md)
- [Mirror JSON](reference/mirror-json.md)
- [Kaleidoscope API](reference/kaleidoscope-api.md)
- [Python facade](reference/python-api.md)
- [状态、快照与克隆](reference/snapshots.md)
- [输出](reference/outputs.md)

## 运维与开发

- [贡献指南](../../CONTRIBUTING_zh.md)
- [约束与安全边界](operations/constraints.md)
- [依赖安全与更新](operations/dependency-security.md)
- [Simulation CI](operations/simulation-ci.md)
- [故障排查](operations/troubleshooting.md)
- [源码模块图](development/module-map.md)
- [Lint 与格式化策略](development/linting.md)
- [静态类型检查](development/type-checking.md)
- [命名规范](development/naming.md)
- [物体资产](development/object-assets.md)
- [碰撞近似](development/collision-approximation.md)
- [USD 预览](development/usd-preview.md)
- [文档维护](maintenance/documentation-guide.md)
