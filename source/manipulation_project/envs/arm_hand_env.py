"""AR5/L6 组合环境扩展点。"""

from __future__ import annotations

from manipulation_project.envs.base_env import BaseEnv


class ArmHandEnv(BaseEnv):
    """机械臂 + 灵巧手场景的预留基类。

    输入:
        当前没有构造参数。
    输出:
        后续可承载 AR5+L6 的 asset 导入、控制器绑定和场景 reset。
    """

    pass
