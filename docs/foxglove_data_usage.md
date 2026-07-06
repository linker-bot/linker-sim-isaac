# Foxglove Data Usage

本文说明当前双臂交互 runtime 如何通过 Foxglove live server 和 MCAP 输出仿真状态，以及外部程序应该如何使用这些数据。

这份文档是使用说明。实时状态流的设计背景和线程边界见 `docs/interactive_realtime_state_streaming.md`。

## 能力边界

当前实现只在 `scripts/dual_arm_interactive.py` 中接入 Foxglove 状态遥测。

支持：

- Foxglove live server 实时显示。
- 同时写 Foxglove MCAP 文件。
- 发布左右机器人关节位置、速度、差分加速度。
- 可选读取并发布 commanded、measured、applied 三类 effort。
- 可选发布 env runtime object 的 root pose。
- 使用同一份 `StateSnapshot` 同步驱动 live server 和 MCAP。

不支持：

- 通过 Foxglove 发送 motion command。
- 把项目交互 TCP/WebSocket 和 Foxglove live server 放在同一个端口。
- 在后台线程直接读取 Isaac articulation、PhysX view 或 USD stage。
- 非交互脚本的 CLI 级 Foxglove 状态流。底层 `FoxgloveLogger` 可以复用，但当前脚本参数只接在双臂交互 runtime 上。

## 启动 Live Server

启动双臂交互 runtime，并开启 Foxglove live server：

```bash
python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8765 \
  --state-rate-hz 60 \
  --state-include-objects
```

如果需要 effort：

```bash
python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8765 \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

连接方式：

```text
Foxglove Desktop:
  Open connection -> Foxglove WebSocket -> ws://127.0.0.1:8765

Foxglove Web:
  https://app.foxglove.dev/?ds=foxglove-websocket&ds.url=ws://127.0.0.1:8765
```

注意：

- 反斜杠 `\` 后面不要有空格。
- `--foxglove-live-host` 和 `--foxglove-live-port` 是两个 CLI 参数，必须分别给值。
- `8765` 只是示例端口；不要和 `--tcp-jsonl-port` 或 `--websocket-port` 共用。
- `ws://localhost:8765` 和 `ws://127.0.0.1:8765` 通常等价，但本机调试推荐先用 `127.0.0.1`，少一层名称解析干扰。

## 同时写 MCAP

只写 MCAP：

```bash
python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-mcap-path logs/interactive_state.mcap \
  --state-rate-hz 60 \
  --state-include-objects \
  --state-include-efforts \
  --foxglove-joint-effort-field measured
```

live server 和 MCAP 可以同时开启：

```bash
python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8765 \
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

## CLI 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--state-rate-hz` | `60.0` | 状态采样频率。`<=0` 时关闭状态流。只有设置 live port 或 MCAP path 时才会真正启动。 |
| `--state-include-objects` | 关闭 | 采样 env runtime object root pose，并发布到 `/linkerbot/state` 和 `/scene`。 |
| `--state-include-efforts` | 关闭 | 读取 commanded、measured、applied 三类 effort。读取失败或不可用时填 `nan` 或 `null`。 |
| `--foxglove-live-host` | `127.0.0.1` | live server 监听地址。 |
| `--foxglove-live-port` | 无 | live server 监听端口。未设置时不启动 live server。 |
| `--foxglove-mcap-path` | 无 | MCAP 输出路径。未设置时不写 MCAP。 |
| `--foxglove-joint-effort-field` | `none` | 选择写入 `/joint_states` 标准 `effort` 字段的 effort 语义，可选 `none`、`commanded`、`measured`、`applied`。 |

## Topic 约定

默认发布三个 topic：

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

- 把 `ws://127.0.0.1:8765` 直接粘到浏览器地址栏。
- 用普通 WebSocket 调试工具连接。
- 在 Foxglove 里选错数据源类型。

正确方式是在 Foxglove 里选择 `Foxglove WebSocket` 数据源，地址填 `ws://127.0.0.1:8765`。如果后续能看到 `/joint_states`、`/linkerbot/state`、`/scene`，这条日志可以忽略。

### `unrecognized arguments`

多行 shell 命令里，反斜杠后面如果有空格，会导致 shell 把空白参数传给 Python。写成：

```bash
python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8765
```

不要写成 `\  --foxglove-live-port` 这种形式。

### 没有数据

检查：

- 是否已经打印 `DUAL_ARM_INTERACTIVE_READY`。
- 是否设置了 `--foxglove-live-port` 或 `--foxglove-mcap-path`。
- `--state-rate-hz` 是否大于 0。
- live port 是否被其它进程占用。
- Foxglove 数据源是否是 `Foxglove WebSocket`，不是普通 WebSocket 或 MCAP 文件源。

## 和 CSV Logger 的关系

CSV logger 和 Foxglove 状态流是并行能力：

- CSV logger 适合离线数值分析和回归。
- Foxglove live 适合实时曲线、3D marker 和调试面板。
- MCAP 适合把一次交互过程打包成可回放数据。

当前没有把 CSV logger 和 Foxglove sink 合并。后续如果要统一，应共享主线程采样得到的 `StateSnapshot`，不要让多个后台线程分别读取 Isaac/PhysX 状态。
