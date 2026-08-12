# 输出参考

语言：[中文](outputs.md) | [English](../../en/reference/outputs.md)

持久化输出和 live sink 只由 Mirror `outputs` profile 组合。Kaleidoscope 通过 step `info` 返回 CUDA
训练指标；checkpoint 是显式状态冷路径，不属于通用 output worker。

## 类型

| 输出 | Owner | 约束 |
| --- | --- | --- |
| Joint CSV | Mirror logger | 明确列、采样周期、flush、已有文件策略 |
| Hybrid control CSV | Mirror hybrid executor/logger | owner tick 冻结诊断、JSON vector 列、同一采样周期与文件策略 |
| MCAP | Mirror telemetry | 有界 queue、topic/schema 固定、preflight 后打开 |
| Foxglove live | Mirror telemetry | loopback only，无认证/TLS |
| Camera image/depth | `CameraBundle` sink | 每相机 queue、消息/目录 byte 上限 |
| Runtime metadata | Mirror output coordinator | mode/profile/backend/device/fingerprint |
| RL checkpoint | Kaleidoscope cold API | 显式 CUDA↔CPU，schema/versioned |

## 开关与保留配置

`camera.enabled`、`logging.enabled` 或 `telemetry.enabled` 为 `false` 时，可以保留合法路径、端口、
topic 和策略值。严格解析器仍校验这些值的类型、范围与 schema，但 scene assembly 不会解析或
preflight 路径、绑定端口、创建 sink，也不会为关闭项分配 queue。重新设为 `true` 时仍要求必要
consumer：camera 至少有一个输出目标，logging 必须有 `joint_tracking_path`，telemetry 至少有一个
live/MCAP endpoint 和一种消息 modality。

`include_efforts=false` 时也允许保留合法 `joint_effort_field`；运行时会把来源投影为 `none`，不发布
effort 数组。设为 `true` 时，该字段必须选择 `commanded`、`measured` 或 `applied`。

`log_hybrid_control=true` 同时要求 logging 总开关与 `hybrid_control_path`。它复用
`interval_steps`、flush 和 existing-data policy；hybrid CSV 与 joint CSV、camera、MCAP 在任何 writer
打开前联合检查路径冲突，并进入相同 runtime close 顺序。`include_hybrid_control=true` 则把 owner 缓存
诊断发布到 `topics.hybrid_control`；非 hybrid 状态只包含 `active: false`，关闭时不会创建该 channel。

## Path 与已有数据

所有目标在任何 writer 打开前联合规划。Policy 必须显式选择安全行为，例如 `error`、`truncate`、
`resume` 或 `timestamped_dir`；具体 sink 可以拒绝无法证明安全的组合。路径必须留在配置允许的 output
root，不能通过 `..` 逃逸。

严格解析器只生成一个冻结的 `LoggingOutputSettings`。Mirror scene assembly 与
`JointTrackingLogger` 直接消费同一个对象，其中同时包含采样周期、列开关、flush 间隔和已有数据
策略；只有总开关关闭时装配层会直接跳过整个 logger，不会打开保留路径。

## 关闭

停止新 sample admission 后，各 sink 在自己的 timeout 内 drain/close。仍存活 worker 保留 sink
所有权，runtime 不会并发关闭其文件或 camera dependency。关闭报告列出 live resource 和 error；
调用方不能只看进程退出码猜测数据是否完整。
