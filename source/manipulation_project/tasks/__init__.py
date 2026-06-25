"""任务级工作流。

tasks 子包把配置、运动学、轨迹采样和执行原语组合成可运行的实验流程，例如关节目标、
TCP 直线运动和夹捏抓取。任务层可以调用 backends/controllers/objects，但不应把底层
Isaac 导入暴露到包入口；因此 ``__init__`` 只描述职责并保持轻量。
"""
"""任务层入口。

任务层把“环境、控制器、轨迹、IK、日志”串成可执行流程。每个任务通常暴露
一个配置 dataclass 和一个 ``run`` 方法，调用方只需要提供 Isaac articulation、
world、Action 类型以及需要的辅助对象。
"""
