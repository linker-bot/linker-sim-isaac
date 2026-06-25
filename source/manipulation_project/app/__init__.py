"""仿真应用启动相关工具。

该子包只保存命令行解析与 Isaac ``SimulationApp`` 启动的薄封装，避免脚本入口
重复处理 headless/gui、renderer、路径等样板参数。Isaac/Omni 模块启动成本高，
因此这里保持轻量：导入 ``manipulation_project.app`` 本身不会创建窗口或初始化 USD
stage，真正的 Isaac 依赖由 ``launch`` 模块在函数调用时延迟导入。
"""
