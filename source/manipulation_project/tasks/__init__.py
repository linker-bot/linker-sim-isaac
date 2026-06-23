"""Task-level workflows."""
"""任务层入口。

任务层把“环境、控制器、轨迹、IK、日志”串成可执行流程。每个任务通常暴露
一个配置 dataclass 和一个 ``run`` 方法，调用方只需要提供 Isaac articulation、
world、Action 类型以及需要的辅助对象。
"""
