# 输出与持久化

语言：[中文](outputs.md) | [English](../../en/reference/outputs.md)

本文是项目全部内置持久化输出和实时输出的统一参考，定义每类目标由谁配置、写入什么内容、
如何处理已有路径，以及队列受压和关闭时会发生什么。

## 输出所有权与矩阵

输出目的地与进程策略由不同配置负责。数据源所在配置声明目的地，runtime profile 统一声明
writer 行为。

| 输出 | 入口 | 目的地与内容配置 | 已有数据策略 |
| --- | --- | --- | --- |
| 关节跟踪 CSV | 仅 Single Scene | `configs/logging/*.yaml` | `runtime.output.csv_existing_file_policy` |
| 状态 Foxglove live | Single Scene 与 Tiled Scene | `runtime.telemetry` | 不适用，不创建文件 |
| 状态 MCAP | Single Scene 与 Tiled Scene | `runtime.telemetry.mcap.path` | `runtime.output.mcap_existing_file_policy` |
| 相机本地文件 | Single Scene 与 Tiled Scene | `sensors.cameras.<name>.output.save_dir` | `runtime.camera_output.existing_data_policy` |
| 相机 Foxglove live | Single Scene 与 Tiled Scene | env profile 中相机的 topic prefix、live host 与 live port | 不适用，不创建文件 |
| 相机 MCAP | Single Scene 与 Tiled Scene | `sensors.cameras.<name>.output.foxglove_mcap_path` | `runtime.camera_output.existing_data_policy` |

Tiled Scene 入口不创建关节跟踪 CSV。状态 MCAP 与相机 MCAP 是两类独立 sink：路径分别属于
`runtime.telemetry.mcap.path` 和 env profile 中每个 camera 的 output。`runtime.output` 与
`runtime.camera_output` 只拥有对应的已有数据策略和共享 writer 策略，不拥有这些目标路径。

相关 runtime 配置如下：

```yaml
runtime:
  camera_output:
    queue_size: 128
    overflow_policy: block
    worker_poll_interval_s: 0.1
    existing_data_policy: error
    shutdown_policy: drain
    rgb_format: png
    depth_format: npz
    metadata_flush_interval_frames: 16
    max_bytes_per_camera: 10737418240

  telemetry:
    rate_hz: 30.0
    buffer_size: 1
    drop_policy: latest
    on_error: stop
    topics:
      joint_states: /joint_states
      scene: /scene
      state: /linkerbot/state
    mcap:
      path: logs/state.mcap
    foxglove_live:
      enabled: true
      host: 127.0.0.1
      port: 8767

  output:
    csv_existing_file_policy: error
    mcap_existing_file_policy: error

  shutdown:
    state_publisher_timeout_s: 2.0
    camera_publisher_timeout_s: 2.0
```

## Single Scene 关节 CSV

通过 `--logging-profile` 选择 logging profile。只有同时设置
`logging.enabled: true` 和非空 `logging.joint_tracking_path` 时才会输出 CSV：

```yaml
logging:
  enabled: true
  joint_tracking_path: logs/joint_tracking/run.csv
  flush_interval_s: 0.2
  interval_steps: 5
  log_actual_position: true
  log_actual_velocity: true
  log_command_position: true
  log_command_velocity: true
  log_command_effort: true
  log_action_effort: false
  log_measured_effort: false
  log_applied_effort: false
```

配置路径是文件名模板。Single Scene 会加入数值 robot ID 与稳定 label，为每台机器人派生独立文件：

```text
配置路径：logs/joint_tracking/run.csv
robot 0，label 为 left：logs/joint_tracking/run.0.left.csv
```

每个文件首先包含以下公共列：

| 列 | 含义 |
| --- | --- |
| `step` | 采样对应的 physics step |
| `time_s` | 仿真时间，单位 s |
| `phase` | 当前执行阶段 |
| `drive_update` | 本次采样是否刷新了 drive target |

然后按 driven joint 顺序展开已启用的测量列：

| 列名模式 | 配置开关 | 含义 |
| --- | --- | --- |
| `qd_<joint>_rad` | `log_command_position` | 目标位置 |
| `q_<joint>_rad` | `log_actual_position` | 实际位置 |
| `vd_<joint>_rad_s` | `log_command_velocity` | 目标速度 |
| `v_<joint>_rad_s` | `log_actual_velocity` | 实际速度 |
| `pos_err_<joint>_rad` | 两个位置字段同时开启 | 目标位置减实际位置 |
| `vel_err_<joint>_rad_s` | 两个速度字段同时开启 | 目标速度减实际速度 |
| `tau_cmd_<joint>` | `log_command_effort` | 缓存的 effort command |
| `tau_action_<joint>` | `log_action_effort` | 发给 Isaac 的 effort action |
| `tau_measured_<joint>` | `log_measured_effort` | PhysX measured effort |
| `tau_applied_<joint>` | `log_applied_effort` | PhysX applied effort |

不可用的 effort 值写为 `nan`，其物理量纲由 PhysX 关节类型决定。

`interval_steps` 在 `step % interval_steps == 0` 时写入。flush 设置会根据 physics step
一次性换算：

```text
flush_rows = max(1, round(flush_interval_s / physics_dt))
```

writer 每写入 `flush_rows` 行执行 flush，而不是每隔这么多个 physics step。启用采样降频后，
近似仿真时间间隔为 `flush_rows * interval_steps * physics_dt`。

CSV 使用 `resume` 时，已有文件必须具有与当前配置完全一致的表头、以换行结束的最后一条记录、
合法 CSV 语法以及每行正确的列数；全部验证完成后才开始 append。

## 状态 Foxglove 与 MCAP

状态 live 与 MCAP sink 发布同一份选定状态帧。topic 名称和 payload 开关属于
`runtime.telemetry`，完整 payload 契约见[实时状态流](../guides/telemetry.md)。

`runtime.telemetry.rate_hz: 0` 会完全关闭 telemetry。此时即使配置了 live endpoint 或
MCAP path，也不会打开对应 sink。live sink 不创建文件，因此不使用已有数据策略。

状态 MCAP 使用 `runtime.output.mcap_existing_file_policy`。Foxglove SDK 打开解析后的文件时
禁止覆盖。MCAP 无法安全 append，因此会在打开文件和同时配置的 live sink 之前拒绝
`resume`。可使用 `error`、`truncate` 或 `timestamped_dir`。

## 相机文件与 Foxglove

每台启用的相机可以独立选择本地文件、Foxglove live、相机 MCAP，或同时选择多个目标。
没有 active sink 的相机仍会创建，但标准 runtime 不会为其调度自动帧输出。

| Modality | 本地 payload | Foxglove live 与 MCAP payload | `<prefix>/info` |
| --- | --- | --- | --- |
| `rgb` | `ppm`、`png` 或 `npy` | `RawImage`，编码为 `rgb8` | JSON metadata |
| `depth` | `npy` 或 `npz` | `RawImage`，编码为 `32FC1` | JSON metadata |
| `semantic_segmentation` | `npy` | 无图像 payload | JSON metadata |
| `instance_segmentation` | `npy` | 无图像 payload | JSON metadata |

分割帧以原生数组形式保存在本地。Foxglove 会在 `/info` 收到其 metadata，但不会创建分割
`RawImage` channel。

本地目录结构可以是：

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

NPZ depth payload 将数组存储在 `data` key 下。RGB 与 depth 输出前分别规范化为连续的
RGB8 与 float32 数组。即使本地格式启用了压缩，Foxglove `RawImage` 仍是未压缩 payload。

多台相机可以共享同一个 live host/port 或相机 MCAP path；它们复用底层 sink，并保留各自的
topic prefix。每个本地 `save_dir` 始终是独立的相机 namespace。

## 相机 Metadata

本地 `metadata.jsonl` 为每个已提交的 modality frame 保存一条严格 JSON object。只有 payload
创建并校验成功后才追加对应记录：

```json
{
  "camera_name": "world_rgbd",
  "modality": "depth",
  "frame_index": 12,
  "simulation_step": 480,
  "time_s": 2.0,
  "shape": [480, 640],
  "dtype": "float32",
  "relative_path": "depth/000012.npz",
  "intrinsics": [[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
  "camera_position_world": [0.5, -0.6, 0.8],
  "camera_orientation_world": [1.0, 0.0, 0.0, 0.0]
}
```

| 字段 | 是否存在 |
| --- | --- |
| `camera_name`、`modality`、`frame_index` | 始终存在 |
| `simulation_step`、`time_s` | 始终存在，使用仿真时钟 |
| `shape`、`dtype` | 始终存在，描述保存的帧数组 |
| `relative_path` | 仅本地 metadata 包含 |
| `intrinsics` | Camera API 可提供时存在 |
| `camera_position_world`、`camera_orientation_world` | 可读取世界位姿时存在 |

Foxglove `<prefix>/info` 发送不含 `relative_path` 的同类 metadata，因为 live 或 MCAP 消息没有
与其关联的本地 payload。

`metadata_flush_interval_frames` 按 modality row 计数，不按一整组多模态相机采样计数；关闭
sink 时始终 flush 剩余记录。

## 已有数据策略

三个策略字段接受相同的取值，但每个字段只作用于自己负责的目标：

| 策略 | 结果 |
| --- | --- |
| `error` | 最终目标已经存在时失败，包括空目录。 |
| `truncate` | 复检全部计划目标，删除已有目标，然后创建新的空文件 namespace。 |
| `resume` | 由对应 sink 校验内容后，复用类型正确的已有目标。 |
| `timestamped_dir` | 在新命名的 UTC run 目录下解析目标。 |

目录目标使用 `timestamped_dir` 时解析为 `requested/<UTC-run>/`；文件目标解析为
`parent/<UTC-run>/filename`。

同一批 Single Scene robot CSV 共享一个 run name，同一次相机输出预检中的本地目录和相机 MCAP
也共享一个 run name；单独预检的状态 MCAP 可能使用另一个 run name。消费方应读取各 sink
报告的实际路径，不能假定整个进程只有一个时间戳。

`resume` 的内容校验由各 sink 自己定义：

- CSV 校验精确表头、末尾换行、CSV 语法和每行列宽。
- 本地相机输出校验全部 metadata 与引用的 payload，扫描未索引 payload，并计算安全的下一个
  index。
- 状态 MCAP 与相机 MCAP 无法通过 SDK 安全 append，因此拒绝 `resume`。

相机 resume 要求严格 JSON、正确的相机 owner、唯一的 modality/index/path 组合、与配置格式
精确匹配的派生 `relative_path`，以及每条索引实际引用的 payload。metadata 最后一行必须以
换行结束。未索引 payload 会保留且其 index 被占用；每个 modality 的最大 index payload
必须能够完整读取，不完整或不可读的 orphan 会使 resume 失败。

## 联合预检与路径安全

Single Scene 将 CSV 目标、本地相机目录、相机 MCAP 与额外状态 MCAP plan 作为同一个启动集合校验；
Tiled Scene 联合校验相机目标与状态 MCAP。任何 sink 打开或已有目标 truncate 之前，必须先完成
全部校验。

最终输出目标不能是符号链接。本地相机 resume 还会拒绝输出树内部的符号链接。同一次启动中，
canonical target 必须互不相同，也不能存在祖先/后代关系，避免一个目录操作删除另一 sink
的文件。

预检和紧接着的复检可以缩小路径竞态窗口，但不提供跨进程锁。不要让并发 writer 指向同一
namespace。多个 path plan 的 apply 也不是文件系统事务；I/O 失败时可能出现较早的变更已经
生效而较晚的变更尚未执行。

同一帧写入多个 sink 时同样是顺序执行而非原子操作。前一个 sink 可能已接纳该帧，后一个 sink
才发生失败。应使用 status 与 metadata 核对被中断的运行。

## 容量与配额

`runtime.camera_output.max_bytes_per_camera` 独立应用于每个本地相机目录。它包含目录中已有
的全部常规文件、下一个编码 payload，以及为该 payload 建立索引的 metadata row。

recorder 必须先编码 payload 才能得到精确大小。如果 payload 与 metadata 之和会超过配额，
recorder 会删除新建但尚未索引的 payload，且不提交 metadata。删除失败会被明确报告，并可能
留下需要检查的 orphan。metadata append 失败时，recorder 会将 metadata 截断到原 offset 并
删除 payload；补偿未完整执行会作为独立错误报告。

对未压缩 RGB8 与 float32 depth，可以按一个输出方向估算：

```text
bytes_per_second
  = camera_count * width * height * frequency_hz
    * sum(bytes_per_pixel_per_modality)

rgb8  = 3 bytes/pixel
depth = 4 bytes/pixel
```

本地压缩会改变磁盘占用和 worker CPU 成本，但不会降低未压缩 Foxglove `RawImage` payload
或内存中单个 queue item 的大小。

## 队列与错误策略

相机帧与状态 snapshot 使用独立队列，并有意采用不同的受压策略：

| 行为 | 相机输出 | 状态 telemetry |
| --- | --- | --- |
| Queue item | 一个 modality frame | 一个 immutable state snapshot |
| 容量 | 正整数 `camera_output.queue_size` | 正整数 `telemetry.buffer_size` |
| Producer 行为 | 可按策略阻塞或立即失败 | 永不阻塞 physics producer |
| 受压策略 | `block`、`error`、`drop_oldest`、`drop_newest` | `latest`、`drop_oldest`、`drop_newest` |
| Worker 错误 | fail-stop，记录第一次错误并清空队列 | `on_error: stop` 或 `continue` |

相机 `block` 等待空位，同时周期检查 worker 失败或关闭请求；`error` 在队列满时立即抛错；
`drop_newest` 拒绝新帧；`drop_oldest` 淘汰已经接纳的帧。只有全部相机 sink 都是 live-only
时才允许丢帧策略；只要存在本地目录或相机 MCAP，启动时就要求使用 `block` 或 `error`。

Telemetry `latest` 会用最新 snapshot 替换全部待发布 snapshot；`drop_oldest` 只在满队列时
淘汰最早 snapshot；`drop_newest` 在满队列时拒绝新 snapshot。三者都保持仿真 producer
非阻塞。`on_error: stop` 停止发布；`continue` 记录错误后继续接纳后续 snapshot。

## 关闭与状态

相机 `shutdown_policy: drain` 会写完全部已接纳帧再关闭；`abort` 丢弃队列内容并累加
`aborted_frames`。join 等待由 `runtime.shutdown.camera_publisher_timeout_s` 限制。

Telemetry 关闭时始终先停止接纳，再排空已接纳 snapshot，最后关闭 live 与 MCAP sink；join
等待由 `runtime.shutdown.state_publisher_timeout_s` 限制。

任何一类 join 超时后都会保留线程和 sink。worker 仍可能写入时并发关闭 sink 不安全，因此
所属 runtime 会保留 handle，用于 status 报告和之后再次尝试关闭。

相机 status 包含队列深度/容量、所选策略、published/dropped/aborted/overflow-error 计数、
thread/sink/timeout 标志、`last_error`，以及可用时嵌套的 sink 配额信息。Telemetry status
包含 buffer 深度/容量、dropped snapshot、错误数、最后发布序号、最后错误，以及
thread/sink/timeout 状态。不能只根据进程退出就推断全部队列项已经持久化，应先检查这些字段。

## 相关文档

- [相机类型与传感器](../guides/cameras.md)说明相机安装、采样、modalities、tiled scope、渲染与
  容量选择。
- [实时状态流](../guides/telemetry.md)定义状态 topic、payload、Single Scene/Tiled Scene 采样与 effort 选择。
- [Foxglove 快速参考](../guides/foxglove.md)提供最短启动与连接示例。
- [Runtime 配置](configuration.md)说明严格 profile 所有权与校验。
