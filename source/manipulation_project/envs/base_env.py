"""任务环境基类预留模块。"""

from __future__ import annotations


class BaseEnv:
    """最小环境基类，供后续 Isaac Lab task/env 接口扩展。

    输入:
        当前没有构造参数。
    输出:
        作为类型占位，后续可加入 reset/step/close 等统一接口。
    """

    pass
