# Mirror CUDA 图像出口

2026-11-03，关联 Issue #71。Framework 阶段 B3 请求默认 GPU 图像链路；本计划明确纳入
Simulator 相机原生接口维护，在独立分支/worktree 内实现，不改消费者冻结 checkout 或资产。

## 实现边界

- Newton exclusive Kit closure 内使用 SyntheticData device buffer/pointer exports，取代常驻 host
  图像出口；RGB/depth 默认以 CUDA 数据交付，CPU 保存显式下载。不得先读 NumPy 再上传。
- 公开 Mirror render 设备选择，保留 owner、暂停物理、选中相机共享更新和原生 frame/time barrier。
  GPU 结果必须有明确所有权/有效期，校验 dtype、shape、stride、device；不静默 CPU fallback。
- 复用已有 Warp/CUDA 依赖及当前设备，GPU copy 保证返回数据不被后续渲染覆盖。保留必要同步，
  不引入另一套 Kit pump、物理 owner、RPC 或通用图像处理平台。
- 相机 smoke 默认 Newton CUDA；CPU 输出消费者显式选择 CPU。同步中英文相机说明和 changelog。

## 验证

先 CPU 合同测试，再单个受监督真实 Newton CUDA + RTX 小场景；覆盖暂停新帧、RGB/depth 数值、
CUDA buffer、选择器、所有权、显式 CPU 下载和关闭。记录原生帧/设备/时间/耗时与显存，原始日志
放仓库外。运行项目 CPU quality 与针对性 GPU 门禁；未跑完整 GPU 矩阵如实记录。

## 不做

不改资产、实际 Task、Kaleidoscope 视觉、视频编码、策略、USD host 同步设计或依赖版本。
不自动合并；Framework 待已准备新版本后再显式绑定。

## 验证结果（2026-11-03）

- 基于 `b3ca4c1`，Isaac Sim 6.1 / Warp 1.17 / RTX 5090 D v2；资产、依赖及物理参数未修改。
- CPU 定向测试 114 项通过；`just quality` 的架构测试 185 项通过，主测试 1677 项通过、
  62 项因 CPU 环境跳过，pure production coverage 85.49%。格式、lint、CI Pyright 范围和文档链接通过。
- `just smoke-mirror-camera` 通过：PhysX CPU、Newton CPU、Newton CUDA 分别在独立受监督进程
  中运行 3 台 320×240 相机。验证 CUDA 源数据、RGB/depth 类型与形状、自有副本、CPU 显式下载、
  暂停移动/恢复、子集激活、重置、相机关闭后数据保留；所有恢复后的 depth 像素均与初始值一致。
- Newton CUDA 另通过 3 台 640×320、60/30/10 Hz、12 个 control step 的真实录制检查，
  分别产生 6/3/1 帧并成功解码，原生身份和物理时间匹配。
- 三 profile 门禁合计墙钟约 45.1 秒。1 秒间隔采样到的单进程 GPU 显存最大值为 5538 MiB
  （含 Kit/物理/渲染，不是图像独占显存或严格峰值）。Newton CUDA 三相机新帧调用耗时示例：
  首帧 551 ms、移动后 126 ms、恢复后 48 ms、重置后 93 ms；含必要 GPU copy/sync，不含验证下载。
  这些数据不构成优化前后性能对照，也不声明 60 FPS 墙钟吞吐。
- 完整 `just test-simulation`、GUI、1080p 和多 GPU 硬件矩阵未运行；已运行最相关的相机门禁。
  Kit 导入/关闭阶段仍有原有 USD/renderer 警告，所有成功门禁均有完整完成标记并以 0 退出。
- 原始日志、录制数据与显存采样留在仓库外；没有提交素材、私有路径或生成数据。

用户要求先审阅合并本 Simulator PR，然后再继续 Framework 接入。当前消费者冻结 checkout 与生产绑定不变。
