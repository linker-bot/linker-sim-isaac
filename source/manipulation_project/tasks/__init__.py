"""任务级工作流。

tasks 子包把环境、控制器、轨迹、IK 和日志串成可执行流程，例如关节目标、TCP 直线运动和
夹捏抓取。每个任务通常暴露一个配置 dataclass 和一个 ``run`` 方法，调用方只需要提供
Isaac articulation、world、Action 类型以及需要的辅助对象。入口文件不导入底层 Isaac API，
保持普通单元测试可以安全导入。
"""
