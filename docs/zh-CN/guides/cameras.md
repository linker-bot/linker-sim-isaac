# 相机类型与传感器设置

语言：[中文](cameras.md) | [English](../../en/guides/cameras.md)

本文说明 GUI 观察视角和仿真传感器摄像机的区别，以及 `sensors.cameras` 的配置、
runtime 数据链、输出、容量预算和 Isaac Sim 渲染告警排查方式。

## 术语边界

项目中有两类容易混淆的 camera：

| 类型 | 推荐名称 | 配置位置 | 用途 | 是否产生图像数据 |
| --- | --- | --- | --- | --- |
| GUI 观察视角 | viewport view / GUI viewport | `visuals.viewport` | 只调整 Isaac GUI 打开后的观察角度 | 否 |
| 仿真传感器摄像机 | sensor camera / RGB-D camera | `sensors.cameras` | 场景中的传感器，输出 RGB、depth 等数据 | 是 |

GUI viewport 只服务人工观察，不参与控制、规划、日志或视觉算法输入。传感器摄像机是场景中的 sensor prim，有独立的 prim path、位姿、分辨率、频率和输出配置。

## GUI Viewport

推荐配置：

```yaml
visuals:
  viewport:
    enabled: true
    eye: [1.35, -1.65, 1.05]
    target: [0.0, -0.1, 0.42]
    prim_path: /OmniverseKit_Persp
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `enabled` | 是否在 GUI 启动后设置 viewport 视角。 |
| `eye` | 相机观察点，世界坐标，单位 m。 |
| `target` | 观察目标点，世界坐标，单位 m。 |
| `prim_path` | Isaac viewport 使用的 camera prim path。 |

## Sensor Camera

仿真传感器摄像机使用 env profile 顶层 `sensors.cameras` 配置。示例：

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      parent_prim_path: /World
      prim_path: /World/WorldRGBD
      pose:
        xyz: [0.08, 0.0, 0.08]
        rpy: [0.0, 1.1, 0.0]
      resolution: [640, 480]
      frequency: 30.0
      modalities: [rgb, depth]
      clipping_range: [0.01, 5.0]
      intrinsics:
        fx: 615.0
        fy: 615.0
        cx: 320.0
        cy: 240.0
      output:
        save_dir: logs/cameras/world_rgbd
        foxglove_topic_prefix: /cameras/world_rgbd
        foxglove_live_host: 127.0.0.1
        foxglove_live_port: 8770
        # foxglove_mcap_path: logs/cameras/world_rgbd.mcap
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `enabled` | 是否创建并初始化该摄像机；canonical runtime 自动取帧还需要至少一个有效 output sink。 |
| `parent_prim_path` | 可选父 prim。`/World` 表示世界固定相机；机器人 link path 表示腕部或工具相机。 |
| `prim_path` | 摄像机自身 USD prim path，必须是绝对路径；设置父 prim 时应位于父 prim 下。 |
| `pose.xyz` | 相对于父 prim 的平移；没有父 prim 时为世界坐标。单位 m。 |
| `pose.rpy` | 相对于父 prim 的姿态。单位 rad。 |
| `resolution` | 图像宽高，例如 `[640, 480]`。 |
| `frequency` | 采样频率，单位 Hz。 |
| `env_ids` | 仅 `runtime.mode: tiled_scene` 使用的资源范围；非空、无重复的整数列表，详见“Tiled Scene 相机展开”。Single Scene 会拒绝该字段。 |
| `modalities` | 输出类型，例如 `rgb`、`depth`。 |
| `clipping_range` | 近远裁剪面，单位 m。 |
| `intrinsics.fx` / `intrinsics.fy` | 可选显式 pinhole 焦距，单位像素；配置后会写入 Isaac Camera。 |
| `intrinsics.cx` / `intrinsics.cy` | 可选显式 pinhole 主点，单位像素；需和 `fx/fy` 一起配置。 |
| `output.save_dir` | 可选离线输出目录。 |
| `output.foxglove_topic_prefix` | 可选 Foxglove topic 前缀。 |
| `output.foxglove_live_host` | 可选 Foxglove live server loopback 监听地址；非 loopback 值会被拒绝。 |
| `output.foxglove_live_port` | 可选 Foxglove live server 监听端口。相机端口建议从 `8770` 起分配，避免占用状态流端口 `8765/8766/8767`。 |
| `output.foxglove_mcap_path` | 可选 Foxglove MCAP 输出路径。 |

相机输出端口来自 env profile 的 `output.foxglove_live_port`，不是交互脚本的 `--foxglove-live-port`。后者控制 `SingleSceneRuntime` 或 `TiledSceneRuntime` 的机器人状态流。
与其他内置 listener 相同，相机 live server 只接受 `localhost` 或数值 loopback 地址。远程查看必须
通过认证 TLS 反向代理或 SSH tunnel。

## Runtime 数据链与开关边界

相机配置从 env profile 到输出端的完整数据链如下：

```text
sensors.cameras
  -> SensorCameraSettings
  -> Isaac Camera prim + Replicator render product
  -> RTX render + RGB/depth annotator
  -> world.step(render=True)
  -> 主仿真线程读取 CPU ndarray
  -> bounded queue
  -> 后台文件、Foxglove live 或 MCAP sink
```

创建顺序和线程边界如下：

1. env profile 在普通 Python 层解析，此时不会启动 Isaac 或创建相机。
2. `enabled: true` 的配置会在 runtime scene 构建阶段创建 Isaac Camera wrapper 和 camera prim。
3. `world.reset()` 后初始化 Camera，并按 `modalities` 挂载 annotator。
4. 只要配置了 `save_dir`、`foxglove_live_port` 或 `foxglove_mcap_path` 中任意一个，runtime
   就会创建 camera output observer，并在 headless 下也强制执行 render step。
5. observer 按仿真时间和 `frequency` 在 `world.step()` 后从主仿真线程取帧。后台线程只消费
   已经取出的数组并写文件或网络，不直接访问 USD、Camera wrapper 或 PhysX 对象。
6. 输出 queue 有界；饱和时是阻塞、快速失败还是按声明方向丢帧，由 runtime
   `camera_output.overflow_policy` 决定。

各开关的实际含义：

| 配置 | 实际效果 |
| --- | --- |
| `enabled: false` | 不创建该 Camera prim、render product、annotator 或输出任务。只做控制/规划时优先使用。 |
| `enabled: true` | 创建并初始化相机；并不自动意味着会保存文件或发布网络数据。 |
| `modalities` | 决定挂载哪些 annotator，也决定每次采样会产生哪些 frame。 |
| `output.save_dir` | 启用本地帧 payload 和 `metadata.jsonl`；是否压缩由 runtime 格式字段决定。 |
| `output.foxglove_live_port` | 启用 Foxglove live RawImage 发布。仅填写 host 或 topic prefix 不会启动输出。 |
| `output.foxglove_mcap_path` | 启用相机 MCAP RawImage 记录。 |
| `--gui` | 控制 GUI viewport；没有 `--gui` 仍可因 camera output 执行渲染。 |
| `visuals.viewport.enabled` | 只决定是否设置 GUI 观察视角，不会创建传感器图像。 |

`frequency` 按仿真时间采样，而不是按墙钟时间保证实时帧率。仿真运行速度低于实时速度时，墙钟
输出帧率也会下降；一次 physics step 跨过多个采样周期时，当前实现不会补写中间帧。

## Runtime 输出策略

相机位置和采样事实属于 env profile；进程级队列、编码、文件生命周期、配额与关闭行为属于
runtime profile：

```yaml
runtime:
  camera_output:
    queue_size: 128
    max_bytes_per_camera: 10737418240
    overflow_policy: block
    worker_poll_interval_s: 0.1
    existing_data_policy: error
    shutdown_policy: drain
    rgb_format: png
    depth_format: npz
    metadata_flush_interval_frames: 16
  shutdown:
    camera_publisher_timeout_s: 2.0
```

| 字段 | 运行时语义 |
| --- | --- |
| `queue_size` | 最大排队 modality frame 数；一个 modality frame 是一个 queue item。 |
| `max_bytes_per_camera` | 每个本地相机目录的独立字节配额。 |
| `overflow_policy` | 持久化输出使用 `block` 或 `error`；纯 live 输出还可选择两种丢帧策略。 |
| `worker_poll_interval_s` | worker 轮询以及阻塞 producer 检查失败或关闭的周期。 |
| `existing_data_policy` | 本地相机目录和相机 MCAP 文件的已有数据策略。 |
| `shutdown_policy` | `drain` 写完已接纳帧；`abort` 丢弃排队帧。 |
| `rgb_format` | 本地 RGB payload：`ppm`、`png` 或 `npy`。 |
| `depth_format` | 本地 float32 depth payload：`npy` 或压缩 `npz`。 |
| `metadata_flush_interval_frames` | 两次 flush 之间的 modality metadata row 数。 |
| `shutdown.camera_publisher_timeout_s` | 有界 worker join timeout。 |

只要存在本地目录或相机 MCAP，输出就属于持久化输出，必须使用 `overflow_policy: block` 或
`error`。已有数据检查、配额统计、队列行为、恢复规则、status counter 与关闭语义的精确定义见
[输出与持久化](../reference/outputs.md)。

## Tiled Scene 相机展开

tiled env 中，`sensors.cameras` 保存相机的通用设置，`env_ids` 决定在哪些 env 真正创建资源：

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      env_ids: [0, 1]
      prim_path: /World/WorldRGBD
      modalities: [rgb, depth]
      output:
        save_dir: logs/cameras/world_rgbd

tiled:
  enabled: true
  num_envs: 8
  diagnostics:
    inspect_env_ids: [7]
```

上例只为 env 0 和 1 创建 Camera prim、render product、annotator 和 output channel；env 2 到 7
没有相机渲染开销。`inspect_env_ids: [7]` 只控制诊断关注范围，绝不参与 camera 创建。离线目录和
Foxglove topic 会自动追加 `env_000`、`env_001` 这类后缀。

`env_ids` 必须是非空、无重复的非负整数列表，且每项都小于 `tiled.num_envs`。拼成 `env_id`、使用
空列表、重复值、布尔值或越界值都会带完整字段路径报错。普通 Single Scene 入口也会拒绝这个 tiled-only
字段，避免配置看似生效但实际创建了未限定范围的相机。

启用 `tiled` 的 env profile 中每个 camera 都必须显式给出 `env_ids`，包括 `enabled: false` 的 camera。缺失 selector
不会展开到全部 env，也不会发出告警后继续运行。

每个子环境的相机位置仍可在 `envs/env_XXX.yaml` 中用 `cameras.<name>.pose` 覆盖。per-env 文件只能
覆盖已声明且 scope 包含当前 env 的相机；scope 外 env 的 pose override 会明确报错，不能作为无效配置
静默保留。高并发配置应先选择少量 env 验证吞吐、显存和磁盘预算。

## 输出格式与 Topic

全部受支持 modality 都可以保存到本地。RGB 与 depth 还具有 Foxglove 图像 payload；两种分割
modality 只发布 metadata：

| Modality | 本地 payload | Foxglove live/MCAP | Info topic |
| --- | --- | --- | --- |
| `rgb` | `ppm`、`png` 或 `npy` | `RawImage`，`rgb8` | JSON metadata |
| `depth` | `npy` 或 `npz` | `RawImage`，`32FC1` | JSON metadata |
| `semantic_segmentation` | `npy` | 无图像 channel | JSON metadata |
| `instance_segmentation` | `npy` | 无图像 channel | JSON metadata |

离线保存目录示例：

```text
logs/cameras/world_rgbd/
├── metadata.jsonl
├── rgb/
│   └── 000000.png
├── depth/
│   └── 000000.npz
├── semantic_segmentation/
│   └── 000000.npy
└── instance_segmentation/
    └── 000000.npy
```

RGB 与 depth 扩展名由 `rgb_format/depth_format` 决定。`png` 是 RGB 无损压缩，`npz` 把
depth 存在 `data` key 下；`npy` 直接保留规范化后的 RGB8、float32 depth 或分割数组。

Foxglove/MCAP topic 示例：

| topic | 编码 | 用途 |
| --- | --- | --- |
| `/cameras/world_rgbd/rgb` | Foxglove `RawImage`，`rgb8` | RGB 彩色图像。 |
| `/cameras/world_rgbd/depth` | Foxglove `RawImage`，`32FC1` | float32 深度图。 |
| `/cameras/world_rgbd/info` | JSON | frame index、shape、dtype、内参和相机世界 pose。 |

每个采样 modality 都会在 `/info` 发送一条消息，包含 frame index、仿真 step/time、shape、
dtype，以及可选内参和世界 pose。本地 `metadata.jsonl` 还包含 `relative_path`。精确记录与提交
契约见[输出与持久化](../reference/outputs.md#相机-metadata)。

## GUI 与 Headless

传感器摄像机不依赖 `--gui`。GUI 模式下可以同时查看 viewport 和传感器输出；headless 模式下，只要配置了相机输出，runtime 也会驱动 render 更新，让 Isaac camera annotator 产出图像。

相机刚初始化后的空帧会被跳过，不占用 frame index，也不会中断仿真。

`SINGLE_SCENE_INTERACTIVE_READY` 只表示交互 transport 和命令队列已经就绪。RTX、Fabric、
Replicator 和 SyntheticData 的部分节点会在下一次 `world.step(render=True)` 才执行第一帧惰性
初始化，所以相机告警可能出现在 READY 之后。这种输出顺序本身不表示 runtime 初始化失败。

## 性能与容量预算

bundled 本地格式是未压缩 binary PPM 和 float32 NPY；PNG/NPZ 可以减少磁盘占用，但会增加 worker
CPU 开销。Foxglove/MCAP 使用未压缩 `RawImage`。不计文件头、metadata、协议和队列开销时，单方向
原始 payload 近似为：

```text
bytes_per_second
  = camera_count * width * height * frequency_hz
    * sum(bytes_per_pixel_per_modality)

rgb8    = 3 bytes/pixel
depth   = 4 bytes/pixel
```

单个 640x480、30 Hz、`[rgb, depth]` 相机为：

```text
640 * 480 * 30 * (3 + 4)
  = 64,512,000 bytes/s
  ~= 61.5 MiB/s
  ~= 216 GiB/hour
```

这只是一个输出方向。若同时配置 `save_dir` 和 Foxglove live，磁盘与网络会分别承担相近的
payload；实际开销还包括 metadata、文件系统和协议头。相同设置展开到 64 个 tiled env 时，
理论原始数据达到约 3.84 GiB/s、13.5 TiB/hour，通常会先遇到 GPU readback、queue、磁盘或网络
瓶颈。无损 profile 会通过背压降低仿真推进速度；纯 live 丢帧 profile 会累计 dropped counter。

`max_bytes_per_camera` 是每个本地相机 namespace 的硬配额，包含 payload、metadata 和已有常规
文件。配额补偿与队列失败语义见[输出与持久化](../reference/outputs.md#容量与配额)。

推荐按用途配置：

| 用途 | 推荐配置 |
| --- | --- |
| 控制、规划、回归测试 | `enabled: false`，不创建传感器相机。 |
| 单相机交互调试 | 先使用较低 `frequency`，只开启必要 modality 和一个主要 sink。 |
| 数据集采集 | 使用 headless，预估磁盘容量，保持 `overflow_policy: block`，或用 `error` 快速失败。 |
| 大规模 tiled | 默认关闭相机；先缩小 `num_envs` 做吞吐测试，再逐步扩大。 |
| GPU 视觉算法 | 保持 frame 在 device 侧处理；只有最终确实需要文件/网络编码时才回传 CPU。 |

## Isaac Sim 渲染告警

以下告警来自 Isaac Sim 5.1 的 RTX/Fabric/SyntheticData 路径，与机器人质量、惯量、关节控制或
cuRobo 求解无关。应先确认图像是否正常，再决定是否需要处理。

### `USD->Fabric: Unhandled array type string[]`

通常会同时出现：

```text
[usdrt.population.plugin] Unhandled attribute type VtArray<std::string>
(prim attribute: omni:rtx:material:db:flattener:transmittance_color)
```

`reflection_roughness_constant`、`ior_constant` 也可能出现同类消息。这些属性是 RTX MaterialDB
flattener 生成的内部字符串数组元数据；当前 Fabric/USDRT population 不支持同步该类型，于是
跳过这些属性。它不会改变 PhysX stage，也不表示项目 USD 资产中的质量或碰撞数据损坏。

处理原则：

- 如果 RGB/depth 和 GUI 材质显示正常，可视为预期的一次性告警。
- 不要为了消除日志而从机器人或物体资产中删除材质属性；日志中的属性由 RTX 内部生成。
- 如果后续出现材质缺失或渲染错误，应保留完整 Kit 日志并在升级 Isaac/Kit 后复测。

### `DLSS increasing input dimensions`

默认 GUI runtime profile 设置
`runtime.simulation_app.render.anti_aliasing_gui: 3`，launcher 会把它作为 DLSS 传给 Kit。
DLSS 先用较低内部尺寸渲染，再放大到 render product 的输出尺寸。在当前 Isaac 版本和
640x480 相机上，内部输入为 320x240；240 低于插件要求的最小尺寸 300，所以插件自动提高内部
输入尺寸并打印：

```text
DLSS increasing input dimensions: Render resolution of (320, 240)
is below minimal input resolution of 300.
```

最终相机输出仍是配置的 640x480。该消息表示 DLSS 自动调整，通常不影响数据正确性，但实际
性能和显存占用可能不同于原始估算。需要消除时可选择以下方案之一：

- 保持当前配置并接受一次性提示。
- 使用更高的相机分辨率；按当前 2 倍缩放行为，800x600 会产生不低于 400x300 的内部输入。
- 在 runtime YAML 中，GUI 启动设置 `runtime.simulation_app.render.anti_aliasing_gui`，headless
  启动设置 `runtime.simulation_app.render.anti_aliasing_headless`，并按工作负载选择 Kit
  抗锯齿模式。Env profile 不接受裸 `anti_aliasing` 字段。

### `OgnSdPostRenderVarToHost ... counter-performant`

当前 recorder/Foxglove sink 需要 NumPy CPU 数据，RGB 和 depth 采样会明确请求 CPU frame。
SyntheticData 因此需要把 GPU render texture 读回 host，并提示直接 texture-to-host 路径性能不佳：

```text
OgnSdPostRenderVarToHost : rendervar copy from texture directly to host buffer
is counter-performant. Please use copy from texture to device buffer first.
```

这是性能提示，不代表图像内容错误。低频单相机采集可以接受；高频或 tiled 场景应优先：

1. 关闭不需要的 modality 和 sink。
2. 降低 `resolution` 或 `frequency`。
3. 让算法先消费 GPU device buffer，延后或合并 host copy。
4. 确实需要 CPU 文件/网络输出时，再设计 device staging、批量拷贝和明确的丢帧/背压策略。

### 何时需要升级为故障处理

只有一次初始化告警、后续图像和交互正常时，不建议全局屏蔽日志。出现以下任一情况应继续排查：

- RGB/depth 持续为空、shape 或 dtype 与配置不符。
- 同一告警每帧重复并导致帧率显著下降。
- 输出 `CAMERA_FRAME_PUBLISHER_FAILED`、CUDA OOM、renderer error 或进程退出。
- `metadata.jsonl` 的 frame index 长时间不增长，或增长中出现不可接受的大量间断。
- 相机关闭后仍持续执行对应 render/readback 工作。

## 常见问题

Foxglove 里看不到 RGB
: 先确认本地 `logs/cameras/<name>/rgb/` 是否已有 `.ppm` 文件；如果本地文件正常，多半是 Foxglove Image panel 仍选中了 depth topic，请切换到 `/cameras/<name>/rgb`。

Depth 画面偏黑
: depth 是 `32FC1` 浮点图。需要在 Foxglove Image panel 中调 depth/color scale 的 min/max，例如先试 `0.0` 到 `1.0`，再根据 `.npy` 实际范围调整。

相机 live port 没有数据
: 检查 `sensors.cameras.<name>.enabled` 是否为 `true`，是否设置了 `output.foxglove_live_port`，以及 Foxglove 连接的是否是相机端口，而不是状态流端口。

腕部相机没有跟随机械臂运动
: 检查 `parent_prim_path` 是否指向真实机器人 link，且 `prim_path` 是否位于该 link 路径下。

READY 之后立刻出现渲染告警
: READY 早于第一帧惰性渲染执行。先按上面的告警表判断插件和影响；只要交互、物理和图像正常，不能仅按日志顺序判定启动失败。

只运行控制但 GPU 或磁盘开销仍然很高
: 检查使用的 env profile，而不只是 `--gui`。把 `sensors.cameras.<name>.enabled` 设为 `false`；仅关闭 GUI 不会关闭配置了 output 的 sensor camera。

长时间运行后磁盘迅速增长
: bundled `save_dir` 格式是未压缩 PPM/NPY。按分辨率、频率、modality 和相机数量计算容量；不需要本地文件时删除 `save_dir`，需要保留时降低频率，或在实测 worker 吞吐后选择 PNG/NPZ。

输出目录已经存在
: 选择符合意图的 `runtime.camera_output.existing_data_policy`；验证与数据保留行为见[输出与持久化](../reference/outputs.md#已有数据策略)。
