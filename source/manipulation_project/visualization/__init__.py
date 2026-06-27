"""Isaac 本地可视化辅助工具。

这里放置摄像机、debug draw 和本地 marker 相关的薄封装，避免动作脚本代码直接散落
Isaac viewport 或调试绘制 API 细节。

外部可视化/遥测协议（例如 Foxglove WebSocket、MCAP）放在 ``manipulation_project.telemetry``。
本模块只作为调试显示用途，不应影响控制闭环的数值结果。
"""
