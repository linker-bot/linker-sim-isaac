# Mirror 遥测

语言：[中文](telemetry.md) | [English](../../en/guides/telemetry.md)

Telemetry、MCAP、Foxglove 和 CSV 是 Mirror 的被动输出。Kaleidoscope step 只返回 dense CUDA
`info` tensor；它不启动 publisher/logger worker，也不把每个 env 序列化为消息。

## 配置开关

`telemetry.enabled=true` 时，必须至少配置一个 live port 或 MCAP path，并启用一种消息 modality。
设为 `false` 时可保留合法 endpoint、topic 和策略；解析器仍检查类型、范围与 schema，但运行时不会
preflight MCAP 路径、绑定端口、分配 publisher buffer 或创建 sink。`include_efforts=false` 同样允许
保留合法 `joint_effort_field`，但运行时会投影为 `none`；开启 effort 时来源不能为 `none`。

`include_hybrid_control=true` 启用独立 JSON modality，topic 由
`topics.hybrid_control` 指定。channel 按需创建；关闭该 modality 时不会创建。启用但当前没有 hybrid
motion 时只发布 `{"active": false}`。运动期间发布最新有限诊断，包括 request/robot、step、tare/参数
generation、`force_axes`、目标/实测 pose 与 wrench、arm effort、contact/饱和标志和 Jacobian 条件指标。

## 数据流

```text
Mirror owner thread capture
        ↓ immutable sample
bounded publisher queue
        ↓
Foxglove live / MCAP / CSV sink
```

采样 stage、articulation、object 和 camera 必须在 Isaac owner thread。后台 worker 只消费已经冻结的
sample，不回调 runtime getter。Topic、modality、decimation、queue capacity、drop/error policy 和 shutdown
timeout 由 Mirror outputs profile 拥有。

Hybrid controller 每个 control tick 只替换一份 owner-owned 缓存；sampler 把它深拷贝进
`StateSnapshot`。后台 publisher 不调用 PhysX wrench/Jacobian/articulation getter。

## 关节 effort 来源

三组数组全部保留，按同一 `joint_names` 排列。旋转关节单位为 N·m，移动关节为 N。

| 数组 | 来源与语义 | 限制 |
| --- | --- | --- |
| `commanded_efforts` | 控制器显式下发的 effort | 隐式位置/速度驱动没有 Python effort 命令，缺测不能当零；显式 PD 在逻辑位置模式也可能有此值。 |
| `applied_efforts` | Isaac actuation effort 读回 | 在所测 direct effort 模式与命令对应，不是隐式驱动的全部输出。 |
| `measured_efforts` | PhysX incoming link wrench 沿关节轴投影 | 包含动力学和约束影响，不是纯外部接触力矩、指尖力或触觉。 |

启用 effort 时，每个机器人的 JSON 状态增加 `effort_metadata`：实际物理采样时间 `sample_time_s`、
`units_by_joint_type`、逐关节 `control_modes`（`mode/method`），以及三来源的 `source`、逐关节
`valid` 和缺失原因 `reason`。mask 和模式与数组共用 joint-name 顺序；未知模式为 `null`。
原因区分禁用、API 缺失、后端不支持、读取失败、形状错误、非有限读回和没有显式命令。
JSON 无效数值仍为 `null`，原 CSV 按来源命名的列继续用 `nan` 表示缺测；需要有效性与原因时同时使用 JSON 状态。

Newton 当前没有 projected effort 读回，measured 数组保持无效，原因标记 `unsupported_backend`，
不将 applied 冒充 measured。直接调用 getter 抛出 `NotImplementedError`，它是 `RuntimeError` 的子类。

不要通过缩放 measured 或清零 armature 强行让三者相等。此前隔离的加速关节实验中，1 N·m
对应约 0.507 N·m projected，armature×角加速度约 0.493 N·m；它解释该次差额，不是通用修正系数
或外力估计公式。新增附件影响惯性，本次遥测不改变重力或增益。

`just smoke-mirror-efforts` 在 PhysX/Newton 中验证正/负/零命令、取消、reset 和来源有效性。
既有笛卡尔混合力位控制继续遵守 physical TCP、tare、有效反馈、后端和 240 Hz 局部控制要求；
开放关节力矩不等于给后端增加 measured wrench 支持。

## 安全与资源

- Live server 只绑定 loopback，无认证/TLS；
- queue 必须有界，drop policy 可观测；
- sink 在写入前统一执行 path/preflight，避免一个输出已覆盖文件后另一个才失败；
- worker 超时仍持有其 sink，runtime 不会提前关闭依赖；
- telemetry failure 按 profile 选择 stop 或记录后继续，不能吞掉异常。

Foxglove 使用见 [Foxglove](foxglove.md)，文件格式与已有文件策略见
[输出参考](../reference/outputs.md)。
