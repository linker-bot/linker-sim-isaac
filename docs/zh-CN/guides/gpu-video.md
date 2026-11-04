# Mirror GPU 视频

语言：[中文](gpu-video.md) | [English](../../en/guides/gpu-video.md)

`MirrorVideoRecorder` 是已采集 CUDA RGB 图像的可选同步消费者，使用 NVENC H.264 / MP4 保存展示
录像。不推进物理、不改变相机自动调度、不创建 Kit，也不提供完整状态恢复。

## 安装

在现有 simulation 安装命令中增加 `--extra video`，保留已有其他 extras；不能混装 `dev` 与 `simulation`：

```bash
uv sync --frozen --extra simulation --extra visualization --extra training --extra video
```

video extra 固定 PyNvVideoCodec 2.2.4，原 Isaac/Torch/Warp 版本不变。需要支持 NVENC 的 GPU 与匹配
驱动；缺库、尺寸不支持或编码不可用均明确失败，不静默使用软件编码或 host 像素路径。CPU 合同测试
不安装此 extra。

## 嵌入式调用

从延迟加载的公开 facade 导入。例子假设相机 `top` 已配置为 640×320 RGB、控制周期 60 Hz，且自动
输出关闭；应用应使用实际解析的相机设置及 `config.compute.cuda_device`。

```python
from linkerbot_sim.mirror import MirrorVideoRecorder

camera = next(c for c in runtime.scene_resources.sensor_cameras if c.name == "top")
with MirrorVideoRecorder(
    "/absolute/existing/parent/new-recording",
    camera_name=camera.name,
    resolution=camera.settings.resolution,
    fps=60,
    cuda_device=runtime.config.compute.cuda_device,
    max_frames=60,
    max_bytes=128 * 1024 * 1024,
) as video:
    for frame_index in range(60):
        if frame_index:
            runtime.step(render=False)
        rgb = runtime.render(camera_ids=(camera.name,))[camera.name]["rgb"]
        video.write(
            rgb,
            time_s=float(runtime.physics_runtime.simulation_time),
            capture_metadata=camera.get_capture_metadata(),
        )
# 在关闭 CUDA context 或 Kit 前关闭录像资源。
```

`write` 接收所选设备上的自有 Warp array 或 Torch CUDA tensor：uint8、H×W×3。输入必须已完成生产，
且在 `write` 返回前保持不可修改。RGB→RGBA 布局转换、同步和编码都使用 GPU；CPU I/O 只接收压缩包
和 JSON 元数据。复用输入 surface 与 NVENC 内部 buffer 在工作完成前保持存活。H.264 有损，不能替代
深度/掩码 NPY 或实机标定传感器数据。

## 时间、身份与边界

- 单 writer 固定相机名、正偶数宽高、整数 fps（1–240）、CUDA 设备与码率。默认 60 fps、CUDA 0、
  12 Mbps、最多 3600 帧和每目录 512 MiB；尺寸同时接受真实 NVENC 能力检查。
- 原生 capture 元数据须有匹配的 `physics_time_s`、递增整数 `native_frame_id` / `render_snapshot_index`，
  以及固定的 `native_frame_source` / `render_product_path`。相机关联和可选 recording `metadata` 由调用方
  声明，不冒充本接口独立验证的来源。
- 真实时刻须满足 `first_time + frame_index / fps`。允许 float32 物理时间精度，容差上限小于四分之一个
  采样周期。暂停重复曝光、跳过周期、reset 倒退或来源变化均失败；新分段使用新 writer，不补造帧或压缩缺口。
- 播放从零开始；`frames.jsonl` 保留原始物理时刻和原生身份。H.264 无 B 帧，核验编码包顺序与数量。
  MP4 使用明确整数时间单位，不能在容器调整 time base 后仍把帧号直接当时间戳。
- 每个 JSON 元数据对象/行最多 64 KiB。同步背压避免应用层排队和丢帧。压缩数据加元数据及实际文件
  大小均接受 `max_bytes` 检查，最终包含 MP4 trailer 和完成元数据。缓冲写入/trailer 在检测到失败前
  可能短暂超过配额，失败产物保留。

## 生命周期与失败

输出使用已有父目录下的新绝对目录，拒绝符号链接路径和已有目标。所有调用位于创建 writer 的线程，
调用方仍拥有 runtime。编码可延迟仿真/UI 服务；原生 codec 调用不可抢占，墙钟 deadline 和失控 GPU
进程由已有进程监督负责。

`close()` 排空 NVENC、核验包数、完成 MP4 并释放 codec/文件资源后，才原子发布 `recording.json`。
成功关闭可重复调用并返回同一标记；空录像不能完成。

`write` / `close` 失败后 writer 不可继续使用，资源释放。`abort()` 或 context manager 内部异常直接
释放资源而不发布完成标记。重试使用新目录；可播放的部分 MP4 也不是完成证据。文件为 `video.mp4`、
`frames.jsonl`，成功后增加 schema 为 `linker-sim.mirror-video.v1` 的 `recording.json`。
这是进程完成标记，不承诺断电事务。

## 验证范围

无 CUDA 环境执行 `just test-mirror-video` 和 `just quality`。真实验收还需核对解码画面、帧数、分辨率、
播放帧率/时长与物理时间索引；创建编码器成功或 codec 字符串不足以证明通过。
[实施计划](../../plans/MIRROR_GPU_VIDEO.zh-CN.md)记录固定版本与限制。该功能不承诺墙钟 60 FPS、
完整传感器标定，也不改变 Kaleidoscope 或实际 Task。
