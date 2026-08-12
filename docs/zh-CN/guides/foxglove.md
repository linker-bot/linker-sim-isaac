# Mirror Foxglove 快速参考

语言：[中文](foxglove.md) | [English](../../en/guides/foxglove.md)

Foxglove 是 Mirror telemetry sink，不是控制 transport。启用后从 outputs profile 读取 topic、queue、
MCAP/live 和关闭策略；Kaleidoscope 不创建 Foxglove publisher。

Live server 只监听 loopback。用 SSH tunnel 或认证 TLS proxy 提供远程访问，不要直接公网暴露端口。
连接后可查看 joint state、TCP/object marker、camera topic 和 runtime status；具体 modality 取决于 profile。

MCAP 写入与 live publish 消费同一 immutable sample，但各自有独立有界 queue/错误状态。完整资源规则见
[遥测](telemetry.md)和[输出参考](../reference/outputs.md)。
