"""Isaac debug draw 可视化入口预留模块。

项目中可能会从脚本或 notebook 统一导入 ``manipulation_project.visualization.debug_draw``。
当前模块有意不导入 Isaac debug draw API，原因是该 API 依赖 GUI/runtime context；保持空实现
可以保留稳定导入路径，同时避免 headless 单元测试在 import 阶段触发 Isaac 依赖。
"""

from __future__ import annotations
