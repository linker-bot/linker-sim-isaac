# 变更记录

语言：[中文](CHANGELOG_zh.md) | [English](CHANGELOG.md)

此文件记录会影响用户的主要变更。项目用语义版本标识 workspace 契约，开发中的精确修订则由
Git commit 标识。

[Unreleased]: https://github.com/linker-bot/linker-sim-isaac/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/linker-bot/linker-sim-isaac/releases/tag/v0.3.0

## [Unreleased]

- 接入 L6/O6 共用法兰、可选 Gemini 335L 腕部组件及顶部 ZED 模块，提供具名光学挂载与独立相机规格。修正 USD 姿态读回和快照恢复时的根旋转表示切换。

### 变更

- 支持基线迁移至 Isaac Sim 6.1.0、Newton 1.5、Warp 1.16 和 MuJoCo 3.11。Newton 使用新目标字段并明确要求 DOF 布局，同步适配模型通知与控制快照；仿真环境需按新锁文件重建。

- Mirror 只采集到期且有输出消费者的相机；`step(render=True)` 改为允许按需采集，不再每步强制全部传感器。`render(camera_ids=...)` 获取选中相机新帧，不推进物理或录制；空闲传感器停止渲染，暂停、急停和连续查询独立服务 GUI。

- 在 self-hosted runner 稳定性问题解决前，GPU/Isaac `Simulation` 工作流暂时只允许手动触发。
- 新增由维护者手动触发的发布工作流；发布前会校验 annotated version tag、CPU quality，以及
  同一 commit 上成功的 Simulation run，随后发布带 SHA-256 校验的源码 workspace 归档。
- 公开协作入口新增 Pull Request 模板和支持问题分流指南。
- 补齐默认分支 ruleset 策略文件，供对应测试和定时 drift 审计读取。

### 修复

- 保证配置的 Newton 接触默认值在 Isaac 6.1 MJCF 转换后实际生效：在自有临时源副本中补全缺失默认值，保留显式或继承的 `solref`，不修改源资产或 PhysX 导入。

- 修正 Isaac 6.1 的 PhysX 导入：保留 MJCF 分层后丢失的原生 mimic 目标和系数，将 MJCF/URDF 的 articulation root 元数据放在已有世界固定关节上，使固定相机装配在积分中保持固定；任务空间绑定通过精确 body 关系处理关节型 root。

- 将 AR5 臂手默认 `AR5V2_L/R_pinch_tcp` 改名为 `AR5V2_L/R_flange_tcp`，明确其为零偏移机械臂法兰参考。
  显式 TCP selector 需改用新名，不保留旧名别名；不改变坐标变换、物理行为或独立配置的任务 TCP。

- Newton 每个内部子步刷新接触，默认内部步长不超过 2 ms，未显式设置的接触采用明确的 4 ms 响应。保留外部控制/渲染时钟及每 shape 显式接触参数，诊断输出实际积分配置。
  原有 Newton physics YAML leaf 需补充 `max_substep_dt_s` 和 `default_contact_time_constant_s`；增加内部积分会降低吞吐并改变接触轨迹。

- 关节 effort 遥测区分 commanded、applied、projected 来源，按实际物理时钟记录逐关节有效性和缺测原因。
  Newton projected 明确标为不支持；不改变增益、重力和 armature。

- 规划接入当前固定手型、附件、实际工装包围盒与携带物；修正障碍世界坐标到基座坐标转换，增加明确接触对和采样路径覆盖诊断。

- 配套运行时已覆盖旧问题，删除 USD 解析串行保护和 Warp `func(module=...)` 绕行。Newton 等式审计适配 MuJoCo 命名空间并保留原生 joint-equality 执行者；未显式设置的接触默认值不再修改 resolver 映射。

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
