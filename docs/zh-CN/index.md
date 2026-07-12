# 文档索引

语言：[中文](index.md) | [English](../en/index.md)

本文是文档统一入口。先选择 runtime 与接口；任务操作阅读指南，精确字段和状态语义查阅参考页。

## 按目标开始

| 目标 | 阅读 |
| --- | --- |
| 理解系统能力和边界 | [项目概览](getting-started/project-overview.md) |
| 选择 Single Scene 或 Tiled Scene、JSON 或 Python | [Runtime 与 API 选择](getting-started/choose-runtime-and-api.md) |
| 完整运行一次 Single Scene | [Single Scene 快速入门](getting-started/single-scene-quickstart.md) |
| 完整运行一次 Tiled Scene | [Tiled Scene 快速入门](getting-started/tiled-scene-quickstart.md) |
| 配置一次运行 | [配置指南](guides/configuration.md) |
| 使用 Single Scene 指令 | [Single Scene Runtime 与 JSON](reference/single-scene-json.md) |
| 使用批量 Tiled Scene 环境 | [Tiled Scene Runtime 与 JSON](reference/tiled-scene-json.md) |
| 调用受支持的进程内接口 | [Python Facade 参考](reference/python-api.md) |
| 选择控制或轨迹执行路径 | [控制与轨迹](guides/control-and-trajectories.md) |
| 使用 cuRobo 规划 | [运动规划](guides/motion-planning.md) |
| 配置物理、规划或 env 间碰撞 | [碰撞模型](guides/collision-models.md) |
| 捕获、恢复或复制 runtime 状态 | [Snapshot 参考](reference/snapshots.md) |
| 发布状态或检查 MCAP | [实时遥测](guides/telemetry.md) |
| 配置相机与图像输出 | [相机](guides/cameras.md) |
| 理解 CSV、MCAP、图像和 metadata 文件 | [输出参考](reference/outputs.md) |
| 排查启动、协议、规划或关闭故障 | [故障排查](operations/troubleshooting.md) |
| 生成或检查 USD 资产 | [物体资产](development/object-assets.md) |
| 定位源码所有者或内部模块 | [源码模块图](development/module-map.md) |

## 入门

- [项目概览](getting-started/project-overview.md)
- [选择 Runtime 与 API](getting-started/choose-runtime-and-api.md)
- [Single Scene 快速入门](getting-started/single-scene-quickstart.md)
- [Tiled Scene 快速入门](getting-started/tiled-scene-quickstart.md)

## 指南

- [配置](guides/configuration.md)
- [控制与轨迹](guides/control-and-trajectories.md)
- [运动规划与 cuRobo](guides/motion-planning.md)
- [碰撞模型](guides/collision-models.md)
- [实时遥测](guides/telemetry.md)
- [Foxglove 快速参考](guides/foxglove.md)
- [相机](guides/cameras.md)

## 参考

- [YAML 配置参考](reference/configuration.md)
- [Single Scene CLI](reference/single-scene-cli.md)
- [Single Scene Runtime 与 JSON](reference/single-scene-json.md)
- [Tiled Scene CLI](reference/tiled-scene-cli.md)
- [Tiled Scene Runtime 与 JSON](reference/tiled-scene-json.md)
- [Python Facade](reference/python-api.md)
- [Snapshot 数据与恢复](reference/snapshots.md)
- [持久化与 Live 输出](reference/outputs.md)

## 运维

- [已知风险与设计约束](operations/constraints.md)
- [故障排查](operations/troubleshooting.md)

## 开发

- [源码模块图](development/module-map.md)
- [命名规范](development/naming.md)
- [物体资产生成](development/object-assets.md)
- [碰撞近似](development/collision-approximation.md)
- [USD 资产预览](development/usd-preview.md)

## 维护

- [文档组织与维护指南](maintenance/documentation-guide.md)
