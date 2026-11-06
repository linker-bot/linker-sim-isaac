# 变更记录

语言：[中文](CHANGELOG_zh.md) | [English](CHANGELOG.md)

此文件记录会影响用户的主要变更。项目用语义版本标识 workspace 契约，开发中的精确修订则由
Git commit 标识。

[Unreleased]: https://github.com/linker-bot/linker-sim-isaac/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/linker-bot/linker-sim-isaac/releases/tag/v0.3.0

## [Unreleased]

- 新增嵌入式 `MirrorRuntime.get_observation()`，读取独立关节/物体根节点观测而不捕获恢复元数据。缓存固定 runtime 指纹，复用 Newton CPU/CUDA 的 TIME/ACT/WARMSTART 打包 kernel，保持每步 canonical state 和现有 snapshot/restore 语义。

- 新增可选 `MirrorVideoRecorder`：自有 CUDA RGB 输入、GPU 布局转换及 NVENC H.264/MP4 编码，校验连续物理时刻/原生帧，提供同步背压、帧数/字节配额与排空后完成标记。`video` extra 固定 PyNvVideoCodec 2.2.4，原仿真依赖不变，不静默采用 host 像素或软件编码。录像用于展示，不替代 depth 数值证据或完整状态恢复。

- Mirror `render()` 默认返回位于 `compute.cuda_device` 的自有 Warp 数组；NumPy 消费者需指定 `device="cpu"`。Newton RGB/depth 使用带布局校验的原生 SyntheticData CUDA buffer 出口，不经 host 图像往返。设备复制在返回前完成，相机关闭后数据仍有效；host 数据或设备不一致时报错。自动文件/live 输出继续显式下载；相机 smoke 默认 Newton CUDA，并验证 GPU 所有权、新帧和录制行为。

- 为 `tblock_push_v1` 增加带版本的 Kaleidoscope reset 模板和 CPU 离线 seed 采样。允许换进程、GPU 编号和环境数量，经正常 reset 事务恢复到指定行并续接逻辑 RNG；任务、后端、控制配置和资产必须一致。先导出一次实际导入的名义输入，再离线采样；已有 snapshot/checkpoint 接口保持独立。

- 接入linker-sim-assets PR #23已合并的AR5/O6组合物理配置：同步原生MJCF动力学/材质，运行与规划资源中的J6/J7统一精确±60°。保留已选柔顺驱动、原始几何及60/480/240 Hz指令/积分/碰撞默认。独立MJCF力矩motor不包含运行时虚拟控制器。

- Newton CPU/CUDA默认采用独立可配置的240 Hz碰撞刷新，保持60 Hz指令和480 Hz积分。接触/三角候选容量1024/16384按world扩展；两端网格工作块倍数8（CPU的64/512块实测耗时接近）。保留容量检查、几何和求解预算；CUDA graph跨tick维护检测相位，reset/restore后先刷新接触。此项改变接触新鲜程度，已知穿透及释放限制见PR #66。

- Newton CPU/CUDA 场景只要包含 AR5/O6 即启用已选接触配置：全场景 elliptic/impratio=100，按部件配置摩擦/阻抗，覆盖材质的 condim 统一为4。其他机器人自身材质保留，但混合场景仍受共享物体/桌面材质及全局参数影响。保留资产显式gap、原始几何、驱动参数和100/50迭代上限；混合与覆盖范围见配置参考。

- AR5/O6及相机组合的Newton默认采用虚拟电机和柔顺输出。所有控制器参数数组按声明名称绑定，L6/O6命令顺序统一为拇指弯曲、拇指侧摆、食指/中指/无名指/小指弯曲。保留真实名称（L6 `thumb_cmc_roll`、O6 `thumb_cmc_yaw`），旧L6数组需重排；L6驱动设置及PhysX动力学不变。
- Newton CPU/CUDA统一为Newton碰撞+MuJoCo Warp，删除原生MuJoCo CPU及contact_pipeline选择，积分状态共用Warp实现。迭代上限100/50、求解器确定性关闭、碰撞确定性开启、接触匹配disabled；未指定gap默认0.0001m，资产显式值（含0）优先。需复验CPU耗时及任务效果，摩擦/阻抗和资产修正另行处理。

- Newton锁定已验证1.8开发提交和MuJoCo 3.15；PhysX与Newton统一加载Warp 1.17 wheel。普通入口默认可自定义的60Hz控制/480Hz物理，专用混合力控独立配置60Hz指令/240Hz反馈/480Hz物理。Mirror场景新增 `control_frequency_hz`；Newton leaf删除 `substeps` / `max_substep_dt_s` 和自动拆步逻辑。混合力控最低频率字段改为 `minimum_feedback_frequency_hz`，保留最低200Hz；hybrid增益、滤波、计数及反馈时序均不变。本步接触、资产默认不变；需重建锁定环境并复验任务行为。

- 接入O6 1014_v4镜像结构、369.98g/手及修正的左惯性、简化碰撞体、两侧20.5mm AR5_08安装和上游从动耦合修复。相机避让路径需重新验证，控制增益保持不变。

- 接入 L6/O6 共用法兰、可选 Gemini 335L 腕部组件及顶部 ZED 模块，提供具名光学挂载与独立相机规格。修正 USD 姿态读回和快照恢复时的根旋转表示切换。

### 变更

- Newton CPU/CUDA profile 和程序规格的默认接触恢复时间设为 10 ms，保留 2 ms 内部步长上限、每子步接触刷新和源模型显式值；改变此数值默认值后需复验任务接触行为。

- 支持基线迁移至 Isaac Sim 6.1.0、Newton 1.5、Warp 1.16 和 MuJoCo 3.11。Newton 使用新目标字段并明确要求 DOF 布局，同步适配模型通知与控制快照；仿真环境需按新锁文件重建。

- Mirror 只采集到期且有输出消费者的相机；`step(render=True)` 改为允许按需采集，不再每步强制全部传感器。`render(camera_ids=...)` 获取选中相机新帧，不推进物理或录制；空闲传感器停止渲染，暂停、急停和连续查询独立服务 GUI。

- 在 self-hosted runner 稳定性问题解决前，GPU/Isaac `Simulation` 工作流暂时只允许手动触发。
- 新增由维护者手动触发的发布工作流；发布前会校验 annotated version tag、CPU quality，以及
  同一 commit 上成功的 Simulation run，随后发布带 SHA-256 校验的源码 workspace 归档。
- 公开协作入口新增 Pull Request 模板和支持问题分流指南。
- 补齐默认分支 ruleset 策略文件，供对应测试和定时 drift 审计读取。

### 修复

- Mirror Newton 的选中相机共享渲染更新，保留逐产品新帧校验、物理冻结、录制频率及闲置/失败清理，避免多相机逐路重复等待 history；预热上限对齐 SyntheticData 的 150 次渲染时钟等待，允许多个产品同时初始化。

- 保证配置的 Newton 接触默认值在 Isaac 6.1 MJCF 转换后实际生效：在自有临时源副本中补全缺失默认值，保留显式或继承的 `solref`，不修改源资产或 PhysX 导入。

- 修正 Isaac 6.1 的 PhysX 导入：保留 MJCF 分层后丢失的原生 mimic 目标和系数，将 MJCF/URDF 的 articulation root 元数据放在已有世界固定关节上，使固定相机装配在积分中保持固定；任务空间绑定通过精确 body 关系处理关节型 root。

- 将 AR5 臂手默认 `AR5V2_L/R_pinch_tcp` 改名为 `AR5V2_L/R_flange_tcp`，明确其为零偏移机械臂法兰参考。
  显式 TCP selector 需改用新名，不保留旧名别名；不改变坐标变换、物理行为或独立配置的任务 TCP。

- Newton 每个内部子步刷新接触，默认内部步长不超过 2 ms，未显式设置的接触使用配置响应，当前默认 10 ms。保留外部控制/渲染时钟及每 shape 显式接触参数，诊断输出实际积分配置。
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
