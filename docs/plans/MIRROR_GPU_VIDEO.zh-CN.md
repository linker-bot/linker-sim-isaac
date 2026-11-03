# Mirror CUDA 录像与 NVENC 编码

2026-11-03，关联 [Issue #73](https://github.com/linker-bot/linker-sim-isaac/issues/73)。
Framework B4 请求先完成连续录制/GPU 编码，完整恢复分步。用户明确将本轮 Simulator 维护纳入范围，
要求独立 PR 审批合并后再修改 Framework。基于 `3f3bf47`，不改旧冻结 checkout、当前绑定或资产。

## 实现边界

- 新增 Mirror 的可选视频 writer，接受已采集的自有 CUDA RGB 图像，默认 NVENC H.264/MP4。
  Python facade 延迟加载；编码依赖仅由 Simulator 可选 `video` extra 提供。
- 固定相机、分辨率与整数采样率；每帧必须有递增 native frame/snapshot 与对应物理时刻。
  固定播放时钟仅接收符合采样间隔的真实帧，暂停重复曝光、跳帧或倒退必须显式失败，不补造帧。
- 同步写入形成有界背压；限定总帧数/输出字节和单帧元数据，GPU 上完成 RGB 布局规整，
  编码器消费 CUDA 数据。CPU 只负责压缩码流、MP4 封装和 JSON 元数据，不下载原始像素再上传。
- 初始化检查 GPU/编码能力，失败不静默使用软件编码。记录编解码库、设备和时钟来源。
  排空后才发布完成标记；失败保留未完成产物、释放编码器/文件资源，不覆盖已有证据。
- Writer 不拥有 runtime、不推进物理、不修改自动相机调度、不建线程池/常驻服务。
  RGB 视频不替代 depth/掩码数值证据；原 PNG/NPY/Foxglove 输出保留。
- 新公开 facade 在本 Issue 显式评审；更新架构清单、双语文档、依赖/许可说明及 changelog。

## 验证

1. CPU fixture 验证配置、CUDA 输入合同、身份/时间、配额、失败及关闭，无 GPU 的质量门禁保持可运行。
2. 真实 NVENC 先用已知色块检查通道顺序、分辨率、帧数、时间与解码；再用单个受监督 Mirror/Kit
   小场景覆盖 Newton CUDA、1080p 60 个真实仿真时刻/秒，验证采集到编码不读回原始像素。
3. `just quality`、依赖 delta 审计和相关 GPU 门禁；未执行的完整矩阵明确记录。原始数据留仓库外，
   文档仅记录固定版本、采用理由、验证结论与限制；不宣称墙钟 60 FPS。

## 已核对接口来源

[NVIDIA PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec) 提供 CUDA 输入/NVENC 编码。
公开文档与实际锁定 wheel 可能存在签名差异，实施以固定版本的 API 和真实解码验收共同确认。
本轮候选为 PyNvVideoCodec 2.2.4；不更改 Isaac/Torch/Warp 等已有锁定版本。

## 排除范围

不修改实际 Task、Framework 运行代码、策略/完整 checkpoint、资产/驱动、Kaleidoscope、UI 或物理 owner。
不自动合并，当前 Framework 绑定保持原状直到用户审批后续升级。

## 实施与验收结果（2026-11-03）

- 已提供公开 `linkerbot_sim.mirror.MirrorVideoRecorder`。调用方送入连续物理采样的 CUDA Warp/Torch
  RGB，GPU 规整为 RGBA，NVENC H.264 后直接逐包封装 MP4；不积累整段视频包、不下载原始像素。
- PyNvVideoCodec 2.2.4 实际 `Encode` 返回带 timestamp / picture_type 的 packet 列表。采用无 B 帧、
  明确整数 time base 的直接 mux，并排空编码器后核对帧数。初期发现小 time base 被 MP4 自动提升，
  已通过解码帧率/时长确认正确映射；不以生成文件成功代替播放时间验收。
- 默认同步背压、3600 帧 / 512 MiB 上限，校验帧/时刻/来源、按新目录保存、异常释放及最终完成标记。
  文件配额包含实际 MP4 trailer；缓冲 flush 造成的临时超限作为失败证据保留，不宣称严格磁盘硬配额。
- 47 项 CPU 定向合同测试通过；覆盖 owner、输入设备/格式、非零起点/30 Hz/float32 时钟、身份/时刻
  漂移、字节/帧数配额、编码/索引 I/O/结束/发布失败、包数不匹配及幂等关闭。
- `just quality` 最终通过：185 architecture tests；1724 passed / 62 skipped，pure coverage 85.48%；
  格式、lint、CI Pyright 和 86 个 Markdown 文件链接通过。依赖 delta 为零新增漏洞/不良状态，
  base/head 均有原有 179 项 advisory，不把 delta 通过声明为零漏洞。锁图只新增可选 codec 包。
- 真实 RTX 5090 D v2 / driver 595.84 上：Newton CUDA 使用原生 1920×1080、60 个真实时刻、60 fps，
  产生 60 帧 / 1.000000 s；PhysX CPU 使用 640×320、30 个真实时刻、60 fps，产生 30 帧 / 0.500000 s。
  前者直接输入 Warp，后者输入共享存储 Torch CUDA tensor。两者均禁止采集/写入期间图像 `.numpy()`、
  Tensor `.cpu()` 和 CPU sampler，成功完成后再独立下载样例用于解码验收。
- 两后端 MP4 全部帧成功解码，抽验首/中/末 RGB 的平均绝对误差均小于 1（uint8 0–255），符合有损
  展示用途。色块另确认 RGB 通道顺序。真实中途 set_state 导致 Newton 时间倒退时 writer 正确拒绝，
  该失败无完成标记；成功验收改为录制开始前设定场景，之后保持真实物理时间连续。
- 最终两个受监督 Kit 均退出 0、无残留 GPU compute 进程。1 秒采样的进程显存最大观测值分别
  6670 / 4902 MiB，包含 Kit/物理/渲染/codec，不是录像新增量或严格峰值。Newton 单相机 capture
  中位数约 42 ms、writer.write 约 0.37 ms；PhysX 约 48 ms / 0.36 ms。write 可返回前序编码包，
  最后还有 drain，不将它当作完整编码延迟或墙钟 60 FPS 吞吐。
- 原始验收脚本、日志、输入/解码样例、MP4、时间索引及资源采样留在仓库外的
  `20261103-mirror-gpu-video` 归档。未提交媒体或厂商二进制。
- GUI、多 GPU、长录像压力和完整 `just test-simulation` / trusted Simulation CI 未执行；本轮只运行
  最相关的真实编码/相机门禁。Kit 仍有原有 USD 导入/关闭警告；未改变其物理 owner 或渲染配置。
  可选 codec 进入 CUDA 进程，完整可信 GPU 矩阵仍由维护者审阅后调度，不自动触发未审阅分支。

当前 Framework 运行代码和绑定未修改，等待用户审批合并本 Simulator PR 后再继续 Issue #83。
