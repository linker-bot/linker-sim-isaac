"""任务基类。

当前基类只定义最小执行接口，方便后续把不同脚本任务统一接入 CLI 或实验调度器。
"""

from __future__ import annotations


class BaseTask:
    """最小任务接口。

    输入:
        子类自行在 ``__init__`` 中保存配置、环境或控制器。
    输出:
        ``run`` 的具体返回值由子类定义；未实现时抛出 ``NotImplementedError``。
    """

    def run(self) -> None:
        """执行任务。

        输入:
            无显式参数；子类可以扩展签名。
        返回:
            默认无返回值。基类方法仅作为接口占位。
        """

        raise NotImplementedError
