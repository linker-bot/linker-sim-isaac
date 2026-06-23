"""命令行解析辅助函数。

这里放跨脚本复用的小型解析器，例如把逗号/空格分隔的数值参数
统一转成 float 列表。脚本入口本身仍各自定义 argparse 参数。
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

    return None if value is None else Path(value)
