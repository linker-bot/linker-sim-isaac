"""调试绘制工具占位模块。

后续可以在这里封装 Isaac debug draw API，用于显示 TCP 目标、IK 误差向量、
轨迹采样点等临时可视化元素。
当前保持为空模块是有意的：保留稳定导入路径，同时避免尚未使用的 Isaac debug draw
依赖在无 GUI/headless 测试中产生副作用。
"""

from __future__ import annotations
