# 变更记录

语言：[中文](CHANGELOG_zh.md) | [English](CHANGELOG.md)

此文件记录会影响用户的主要变更。项目用语义版本标识 workspace 契约，开发中的精确修订则由
Git commit 标识。

[Unreleased]: https://github.com/linker-bot/linker-sim-isaac/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/linker-bot/linker-sim-isaac/releases/tag/v0.3.0

## [Unreleased]

### 变更

- 在 self-hosted runner 稳定性问题解决前，GPU/Isaac `Simulation` 工作流暂时只允许手动触发。
- 新增由维护者手动触发的发布工作流；发布前会校验 annotated version tag、CPU quality，以及
  同一 commit 上成功的 Simulation run，随后发布带 SHA-256 校验的源码 workspace 归档。
- 公开协作入口新增 Pull Request 模板和支持问题分流指南。
- 补齐默认分支 ruleset 策略文件，供对应测试和定时 drift 审计读取。

### 修复

- Kaleidoscope PhysX CUDA 的 seed reset 现在会恢复 native mimic follower 的关节位置与
  速度，并刷新 articulation 派生出的 link pose；重复使用同一 seed 时不再继承上一 episode
  的关节历史。

## [0.3.0] - 2026-08-26

### 新增

- Mirror 提供单 World 的现实回放产品，包含严格的版本化 JSON 协议、显式 runtime owner、
  规划、相机、遥测和有界关闭行为。
- Kaleidoscope 提供 PhysX CUDA 与项目自有 Newton multi-world 训练后端，包含 CUDA-resident
  state、snapshot、clone、批量 IK、Gymnasium 与 skrl adapter。
- 仓库维护 CPU quality、静态类型、纯模块覆盖率、依赖审计、架构清单、仓库 ruleset，以及
  独立的 GPU/Isaac 验收契约。
- Runtime 与项目 metadata 暴露相同的 workspace 版本；诊断记录和支持请求同时保留精确的
  Git commit。

[English changelog](CHANGELOG.md)
