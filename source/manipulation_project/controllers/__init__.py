"""控制器封装和目标生成工具。

controllers 层连接“任务/轨迹产生的命令空间目标”和 Isaac articulation 需要的完整
DOF target。它负责关节名解析、主动关节与 mimic follower 的顺序约定、runtime drive
参数写入以及 YAML profile 解析。入口文件保持轻量，不在导入时触碰实际机器人对象。
"""
