"""Marker 可视化工具占位模块。

后续可以把重复使用的坐标轴、球形目标点、路径 marker 放到这里统一创建和更新。
当前不导入任何 Isaac 或 Foxglove 类型，目的是为未来 marker 封装预留 public module，
同时保持现有测试环境轻量。
"""

from __future__ import annotations
