# 变更记录

语言：[中文](CHANGELOG_zh.md) | [English](CHANGELOG.md)

此文件记录会影响用户的主要变更。项目用语义版本标识 workspace 契约，开发中的精确修订则由
Git commit 标识。

[Unreleased]: https://github.com/linker-bot/linker-sim-isaac/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/linker-bot/linker-sim-isaac/releases/tag/v0.3.0

## [Unreleased]

- 接入 L6/O6 共用法兰、可选 Gemini 335L 腕部组件及顶部 ZED 模块，提供具名光学挂载与独立相机规格。修正 USD 姿态读回和快照恢复时的根旋转表示切换。

### 变更

- 在 self-hosted runner 稳定性问题解决前，GPU/Isaac `Simulation` 工作流暂时只允许手动触发。
- 新增由维护者手动触发的发布工作流；发布前会校验 annotated version tag、CPU quality，以及
  同一 commit 上成功的 Simulation run，随后发布带 SHA-256 校验的源码 workspace 归档。
- 公开协作入口新增 Pull Request 模板和支持问题分流指南。
- 补齐默认分支 ruleset 策略文件，供对应测试和定时 drift 审计读取。

### 修复

- 将 AR5 臂手默认 `AR5V2_L/R_pinch_tcp` 改名为 `AR5V2_L/R_flange_tcp`，明确其为零偏移机械臂法兰参考。
  显式 TCP selector 需改用新名，不保留旧名别名；不改变坐标变换、物理行为或独立配置的任务 TCP。

- Newton 每个内部子步刷新接触，默认内部步长不超过 2 ms，未显式设置的接触采用明确的 4 ms 响应。保留外部控制/渲染时钟及每 shape 显式接触参数，诊断输出实际积分配置。
  原有 Newton physics YAML leaf 需补充 `max_substep_dt_s` 和 `default_contact_time_constant_s`；增加内部积分会降低吞吐并改变接触轨迹。

- 关节 effort 遥测区分 commanded、applied、projected 来源，按实际物理时钟记录逐关节有效性和缺测原因。
  Newton projected 明确标为不支持；不改变增益、重力和 armature。

- 规划接入当前固定手型、附件、实际工装包围盒与携带物；修正障碍世界坐标到基座坐标转换，增加明确接触对和采样路径覆盖诊断。

- 资产导入期间临时串行化携带的 USD 25.11 物理解析器，规避多 collider 的分配器竞争；之后恢复正常并发。

- Mirror 相机通过原生帧完成事件刷新暂停位姿与 headless 图像，不推进物理；记录保留渲染身份，避免将旧帧标记为新的物理时刻。

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
