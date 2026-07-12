"""外部遥测输出适配器。

telemetry 子包负责把仿真运行时状态转换成外部工具可以消费的数据流或文件。
当前输出包括 Foxglove WebSocket、MCAP 和项目内定义的状态 sink。

这些适配器只观察和发送数据，不参与控制闭环，也不修改 robot/world 状态。采样频率、
开启开关和数据来源由 runtime profile 与运行入口决定。
"""
