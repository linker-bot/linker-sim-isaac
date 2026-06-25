"""机器人状态、关节分组和 mimic 关系工具。

robots 子包不直接持有 Isaac articulation，而是处理与机器人命名和数组顺序有关的纯
Python 数据：关节状态快照、关节组解析、arm/hand 分类以及 MJCF equality/mimic 映射。
这种设计让控制器和任务在进入 Isaac runtime 前就能校验 DOF 名称、配置长度和从动关节
约定。入口文件保持轻量，避免导入时读取大型资产文件。
"""
"""机器人结构与关节关系工具。

该包只描述机器人在控制层需要知道的静态/半静态信息，例如 DOF 名称分组、
MJCF mimic/equality 关系，以及轻量的关节状态容器。
"""
