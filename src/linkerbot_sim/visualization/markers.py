"""Isaac marker 可视化入口预留模块。

未来重复使用的坐标轴、球形目标点和路径 marker 可以放到这里统一创建和更新。当前模块
有意保持无运行时代码：它提供稳定 public module 名称，但不在普通 Python 测试中导入 Isaac
marker 类型或创建 USD/GUI 资源。
"""

from __future__ import annotations
