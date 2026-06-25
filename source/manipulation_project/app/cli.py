"""命令行解析辅助函数。

本模块处在 app 层和脚本入口之间，只保存跨脚本复用的轻量解析逻辑，
例如把命令行中逗号/空格混用的数值序列规整成 Python 对象。具体任务
需要哪些参数仍由各脚本自己的 ``argparse.ArgumentParser`` 定义，避免这里
变成全局配置中心。

职责边界:
        * 不启动 Isaac Sim，也不导入任何 ``omni``/``isaacsim`` 运行时模块。
        * 不读取 YAML 配置文件；配置加载和字段校验属于 ``utils.config``。
        * 不解释任务语义；这里仅把字符串转换为脚本可继续处理的基础类型。

输入输出约定:
        * ``comma_or_space_floats`` 接收 argparse 收集到的字符串列表，支持
            ``--values 1,2 3`` 和 ``--values 1 2 3`` 两种习惯写法。
        * ``optional_path`` 只做 ``str -> Path`` 的包装；是否存在、是否为文件由调用方决定。

这些函数通常作为 ``argparse`` 的 ``type=`` 或后处理回调使用，因此错误边界应保持清晰：
解析失败直接抛出底层 ``ValueError``/``ArgumentTypeError``，由 argparse 负责格式化用户可读
的错误消息，避免脚本入口重复拼接异常文本。
"""

from __future__ import annotations

from pathlib import Path


def comma_or_space_floats(values: list[str]) -> list[float]:
    """解析重复参数或逗号/空格分隔参数中的浮点数。

    参数:
        values: argparse 收到的字符串列表，例如 ``["1,2", "3"]``。
    返回:
        展平后的 float 列表，例如 ``[1.0, 2.0, 3.0]``。
    """

    # argparse 对 ``nargs`` 参数会先给出字符串列表；每个元素仍可能包含逗号。
    # 这里先把逗号统一替换为空格，再使用 ``split`` 去掉多余空白，保证两类输入
    # 最终得到同一种展平顺序，便于后续直接构造 numpy 数组或配置对象。
    parsed: list[float] = []
    for value in values:
        parsed.extend(float(chunk) for chunk in value.replace(",", " ").split())
    return parsed


def optional_path(value: str | None) -> Path | None:
    """把可选字符串路径转成 ``Path``。

    参数:
        value: 路径字符串或 ``None``。
    返回:
        ``Path(value)``；输入为 ``None`` 时返回 ``None``。
    """

    # ``None`` 常用于表示“禁用某个可选输出/输入路径”，不能转换成 ``Path('None')``。
    # 其它路径保持惰性：不在 CLI 层解析相对仓库根目录，也不检查文件系统状态。
    return None if value is None else Path(value)
