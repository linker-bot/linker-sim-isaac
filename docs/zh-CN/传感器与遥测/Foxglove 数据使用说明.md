# Foxglove 数据使用说明

本文说明单臂、双臂和 tiled 交互 runtime 如何通过 Foxglove live server 和 MCAP 输出仿真状态，以及仿真传感器相机如何输出 RGB/depth 图像。

这份文档是使用说明。单臂、双臂交互状态流的快速入口见 [实时状态流使用说明](../交互与运行/实时状态流使用说明.md)；tiled telemetry 的完整命令格式见 [Tiled 并行环境使用方式与指令格式](../并行环境/Tiled%20并行环境使用方式与指令格式.md)。

## 能力边界

当前实现包含两类 Foxglove 输出：

- 交互状态流：由 `scripts/single_arm_interactive.py`、`scripts/dual_arm_interactive.py` 或 `scripts/tiled_env_interactive.py` 的 CLI 参数开启，发布关节状态、对象 marker 和完整状态 JSON。
- 传感器相机图像：由 env profile 的 `sensors.cameras.<name>.output` 开启，发布 camera `RawImage` 和 JSON info。

支持：

- Foxglove live server 实时显示。
- 同时写 Foxglove MCAP 文件。
- 发布左右机器人关节位置、速度、差分加速度。
- 可选读取并发布 commanded、measured、applied 三类 effort。
- 可选发布 env runtime object 的 root pose。
- 使用同一份 `StateSnapshot` 同步驱动 live server 和 MCAP。
- 发布 sensor camera RGB `RawImage`。
- 发布 sensor camera depth `RawImage`。
- 保存 RGB、depth 和 metadata 到本地目录。

不支持：

- 通过 Foxglove 发送 motion command。
- 把项目交互 TCP/WebSocket 和 Foxglove live server 放在同一个端口。
- 在后台线程直接读取 Isaac articulation、PhysX view 或 USD stage。
- 非交互脚本的 CLI 级 Foxglove 状态流。底层 `FoxgloveLogger` 可以复用，但当前文档只覆盖单臂、双臂和 tiled 交互入口。
- 通过交互脚本 CLI 覆盖 sensor camera 的输出目录或 Foxglove 端口；相机输出当前由 env profile 配置。
- 把 RGB 和 depth 合并成单个视频 topic；当前它们是两个独立 Image topic。

## 启动 Live Server

日常调试建议把 Foxglove live 端口固定分配，避免多个进程互相抢端口：

| 场景 | 建议端口 | 说明 |
| --- | --- | --- |
| 单臂状态流 | `8765` | 预留给单臂交互/单臂状态观察。 |
| 双臂状态流 | `8766` | `scripts/dual_arm_interactive.py` 的 live server 建议端口。 |
| tiled 状态流 | `8767` | `scripts/tiled_env_interactive.py` 的 telemetry live server 建议端口。 |
| 相机 RawImage | `8770` 起 | 来自 env profile 的 `output.foxglove_live_port`，不要占用上面三个状态流端口。 |

命令 transport 端口不要复用这些 Foxglove live 端口；文档示例中双臂 TCP JSONL 使用 `9001`，双臂 WebSocket JSON 使用 `9002`，tiled TCP JSONL 使用 `9003`。

启动双臂交互 runtime，并开启 Foxglove live server：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --state-rate-hz 60 \
  --state-include-objects
```

如果需要 effort：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

连接方式：

```text
Foxglove Desktop:
  Open connection -> Foxglove WebSocket -> ws://127.0.0.1:8766

Foxglove Web:
  https://app.foxglove.dev/?ds=foxglove-websocket&ds.url=ws://127.0.0.1:8766
```

注意：

- 反斜杠 `\` 后面不要有空格。
- `--foxglove-live-host` 和 `--foxglove-live-port` 是两个 CLI 参数，必须分别给值。
- `8766` 是双臂状态流建议端口；不要和 `--tcp-jsonl-port` 或 `--websocket-port` 共用。
- `ws://localhost:8766` 和 `ws://127.0.0.1:8766` 通常等价，但本机调试推荐先用 `127.0.0.1`，少一层名称解析干扰。

## 同时写 MCAP

只写 MCAP：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-mcap-path logs/interactive_state.mcap \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

live server 和 MCAP 可以同时开启：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --foxglove-mcap-path logs/interactive_state.mcap \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

MCAP 是数据容器，不只能由 Foxglove 读取。任何支持 MCAP 的工具都可以打开文件；不过本项目当前使用了两类编码：

- `/joint_states` 和 `/scene`: Foxglove well-known protobuf schema。
- `/linkerbot/state`: JSON 编码，无自定义 schema。

如果外部程序只想快速拿完整状态，优先读取 `/linkerbot/state`。如果要直接使用 Foxglove 标准可视化能力，优先用 `/joint_states` 和 `/scene`。

## 传感器相机图像

相机输出由 env profile 配置，而不是由 `--foxglove-live-port` 这组状态流 CLI 参数配置。例如 `configs/envs/scene3.yaml` 中的世界固定 RGB-D 相机：

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      parent_prim_path: /World
      prim_path: /World/WorldRGBD
      modalities: [rgb, depth]
      output:
        save_dir: logs/cameras/world_rgbd
        foxglove_topic_prefix: /cameras/world_rgbd
        foxglove_live_host: 127.0.0.1
        foxglove_live_port: 8770
```

相机 live port 是 `output.foxglove_live_port`。如果只看相机图像，Foxglove 连接对应端口，例如：

```text
ws://127.0.0.1:8770
```

相机 topic：

| topic | 编码 | 用途 |
| --- | --- | --- |
| `/cameras/world_rgbd/rgb` | Foxglove `RawImage`，`rgb8` | RGB 彩色图像。 |
| `/cameras/world_rgbd/depth` | Foxglove `RawImage`，`32FC1` | float32 深度图，单位沿用 Isaac camera depth 输出。 |
| `/cameras/world_rgbd/info` | JSON | frame index、shape、dtype、内参和相机世界 pose。 |

在 Foxglove 的 Image panel 里，RGB 和 depth 需要分别选择 topic。topic 列表里出现 `/cameras/world_rgbd/rgb` 只说明数据源收到了 RGB topic；右侧图像面板如果仍选中 `/cameras/world_rgbd/depth`，显示的仍然是深度图。

depth 是 `32FC1` 浮点图，默认显示范围不一定合适。画面看起来偏黑时，先在 Image panel 设置里调 depth/color scale 的 min/max，例如从 `0.0` 到 `1.0` 试起，再根据 `logs/cameras/world_rgbd/depth/*.npy` 的实际数值范围调整。

离线输出目录结构：

```text
logs/cameras/world_rgbd/
├── metadata.jsonl
├── rgb/
│   └── 000000.ppm
└── depth/
    └── 000000.npy
```

如果本地 `rgb/*.ppm` 和 `metadata.jsonl` 中的 `modality: "rgb"` 正常存在，但 Foxglove 面板没有彩色画面，优先检查 Image panel 当前选择的 topic 是否是 `/cameras/world_rgbd/rgb`。

## CLI 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--state-rate-hz` | `60.0` | 状态采样频率。`<=0` 时关闭状态流。只有设置 live port 或 MCAP path 时才会真正启动。 |
| `--state-include-objects` | 关闭 | 采样 env runtime object root pose，并发布到 `/linkerbot/state` 和 `/scene`。 |
| `--state-include-efforts` | 关闭 | 读取 commanded、measured、applied 三类 effort。读取失败或不可用时填 `nan` 或 `null`。 |
| `--foxglove-live-host` | `127.0.0.1` | live server 监听地址。 |
| `--foxglove-live-port` | 无 | 状态流 live server 监听端口。未设置时不启动状态流 live server；单臂建议 `8765`，双臂建议 `8766`，tiled 建议 `8767`；相机 live port 在 env profile 中配置。 |
| `--foxglove-mcap-path` | 无 | 状态流 MCAP 输出路径。未设置时不写状态流 MCAP；相机 MCAP path 在 env profile 中配置。 |
| `--foxglove-joint-effort-field` | `none` | 选择写入 `/joint_states` 标准 `effort` 字段的 effort 语义，可选 `none`、`commanded`、`measured`、`applied`。 |

## Topic 约定

状态流默认发布三个 topic：

| topic | 编码 | 用途 |
| --- | --- | --- |
| `/joint_states` | Foxglove `JointStates` protobuf | 曲线面板和关节状态面板使用。包含实际关节位置、速度，以及可选的一类 effort。 |
| `/scene` | Foxglove `SceneUpdate` protobuf | 3D 面板使用。当前发布 env object root position 的 sphere marker。 |
| `/linkerbot/state` | JSON | 项目完整状态快照。包含 step/time、左右机器人完整关节状态、三类 effort、对象 pose 和 phase。 |

`/joint_states` 的关节名会加侧别前缀，例如：

```text
left/AR5V2_L_arm_joint_1
right/AR5V2_R_arm_joint_1
```

`/scene` 当前只把对象 root position 画成 marker，不表达对象真实几何和姿态。对象完整 pose 请读取 `/linkerbot/state`。

相机 topic 不属于状态流 CLI 参数的一部分；它们由 `sensors.cameras.<name>.output.foxglove_topic_prefix` 决定。当前 `scene3` 的默认前缀是 `/cameras/world_rgbd`。

## `/linkerbot/state` JSON

完整快照结构示例：

```json
{
  "step": 120,
  "time_s": 0.5,
  "phase": "left_tcp_line",
  "robots": {
    "left": {
      "joint_names": ["AR5V2_L_arm_joint_1"],
      "positions_rad": [0.1],
      "velocities_rad_s": [0.0],
      "accelerations_rad_s2": [0.0],
      "commanded_efforts": [0.2],
      "measured_efforts": [0.3],
      "applied_efforts": [0.4]
    },
    "right": {
      "joint_names": ["AR5V2_R_arm_joint_1"],
      "positions_rad": [-0.1],
      "velocities_rad_s": [0.0],
      "accelerations_rad_s2": [0.0],
      "commanded_efforts": [0.2],
      "measured_efforts": [0.3],
      "applied_efforts": [0.4]
    }
  },
  "objects": {
    "Tblock": {
      "prim_path": "/World/TBlock",
      "position_m": [0.15, 0.0, -0.4],
      "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

字段含义：

- `step`: physics step 序号。
- `time_s`: 仿真时间，单位 s。
- `phase`: 当前 motion phase。没有 phase 时省略。
- `positions_rad`: 实际关节位置，单位 rad。
- `velocities_rad_s`: 实际关节速度，单位 rad/s。
- `accelerations_rad_s2`: 由相邻采样点速度差分得到，单位 rad/s^2。第一帧通常是 `nan`。
- `commanded_efforts`: Python controller 缓存的显式 command effort。
- `measured_efforts`: PhysX measured joint effort。
- `applied_efforts`: Isaac articulation runtime applied joint effort。
- `objects`: env `objects[]` 导入对象的 root prim 世界位姿。

如果没有传 `--state-include-efforts`，三类 effort 会是 `null`。如果传了该参数但某个 API 不可用，对应数组会用 `nan` 填充。

## Effort 使用建议

三类 effort 不应混用：

- `commanded_efforts`: 项目控制器在 Python 侧希望下发的 effort。position/velocity implicit drive 下通常没有明确数值。
- `measured_efforts`: PhysX 求解器计算或测得的关节 generalized force，适合观察接触和约束求解后的负载。
- `applied_efforts`: Isaac articulation 当前记录的 actuation effort，适合观察 actuator/drive 侧状态。

Foxglove 标准 `JointStates` 只有一个 `effort` 字段，所以 `/joint_states` 只能选择其中一种语义。需要完整数据时读取 `/linkerbot/state`。

## 读取 MCAP 的建议

建议按用途选择 topic：

- 画关节曲线：读 `/joint_states`。
- 复盘完整状态：读 `/linkerbot/state`。
- 在 Foxglove 3D 面板显示对象位置：读 `/scene`。
- 自己写 Python 分析脚本：优先读 `/linkerbot/state` 的 JSON payload，再按需要解析 `/joint_states`。

MCAP 中的时间戳使用仿真时间换算成纳秒。live server 中如果没有显式时间戳，Foxglove sink 可能使用当前 wall time；当前状态流会把 `time_s` 写入消息和 log time。

## 常见问题

### `handshake failed`

日志示例：

```text
[Error] [foxglove.websocket.server] Dropping client 127.0.0.1:59392: handshake failed
```

这通常不是仿真错误，而是有客户端用错误协议连接了 Foxglove live server。常见原因：

- 把 `ws://127.0.0.1:8766` 直接粘到浏览器地址栏。
- 用普通 WebSocket 调试工具连接。
- 在 Foxglove 里选错数据源类型。

正确方式是在 Foxglove 里选择 `Foxglove WebSocket` 数据源，地址填 `ws://127.0.0.1:8766`。如果后续能看到 `/joint_states`、`/linkerbot/state`、`/scene`，这条日志可以忽略。

### `unrecognized arguments`

多行 shell 命令里，反斜杠后面如果有空格，会导致 shell 把空白参数传给 Python。写成：

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766
```

不要写成 `\  --foxglove-live-port` 这种形式。

### 没有数据

检查：

- 是否已经打印 `SINGLE_ARM_INTERACTIVE_READY`、`DUAL_ARM_INTERACTIVE_READY` 或 `TILED_INTERACTIVE_READY`。
- 是否设置了 `--foxglove-live-port` 或 `--foxglove-mcap-path`。
- `--state-rate-hz` 是否大于 0。
- live port 是否被其它进程占用。
- Foxglove 数据源是否是 `Foxglove WebSocket`，不是普通 WebSocket 或 MCAP 文件源。

如果缺的是相机图像，还要检查：

- env profile 里对应 `sensors.cameras.<name>.enabled` 是否为 `true`。
- 是否设置了 `output.foxglove_live_port` 或 `output.foxglove_mcap_path`。
- Foxglove 连接的是相机 live port，或当前连接中确实能看到 `/cameras/<name>/rgb`、`/cameras/<name>/depth`。
- Image panel 顶部选择的是 RGB topic，而不是 depth topic。
- 本地 `logs/cameras/<name>/rgb/` 是否有 `.ppm` 文件；如果有，说明 RGB 已经采样成功，问题多半在 Foxglove 面板选择或显示设置。

## 和 CSV Logger 的关系

CSV logger 和 Foxglove 状态流是并行能力：

- CSV logger 适合离线数值分析和回归。
- Foxglove live 适合实时曲线、3D marker 和调试面板。
- MCAP 适合把一次交互过程打包成可回放数据。

当前没有把 CSV logger 和 Foxglove sink 合并。后续如果要统一，应共享主线程采样得到的 `StateSnapshot`，不要让多个后台线程分别读取 Isaac/PhysX 状态。
