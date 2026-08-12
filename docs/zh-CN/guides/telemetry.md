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

## 安全与资源

- Live server 只绑定 loopback，无认证/TLS；
- queue 必须有界，drop policy 可观测；
- sink 在写入前统一执行 path/preflight，避免一个输出已覆盖文件后另一个才失败；
- worker 超时仍持有其 sink，runtime 不会提前关闭依赖；
- telemetry failure 按 profile 选择 stop 或记录后继续，不能吞掉异常。

Foxglove 使用见 [Foxglove](foxglove.md)，文件格式与已有文件策略见
[输出参考](../reference/outputs.md)。
