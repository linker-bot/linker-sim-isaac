"""组合脚本演示任务占位模块。

这个入口预留给“多个基础任务按阶段串联”的 demo，例如先移动到观察位、
再执行抓取、最后做保持或摆动测试。
"""

from __future__ import annotations


class ScriptedDemoTask:
    """未来的组合脚本任务。

    输入:
        后续会接收多个子任务配置或阶段列表。
    输出:
        当前为空实现，不执行仿真。
    """

    pass
