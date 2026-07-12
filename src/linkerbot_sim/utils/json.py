"""控制协议与持久化数据共用的严格 JSON 编解码。

标准库 decoder 默认接受 JavaScript 常量 ``NaN``、``Infinity``，并在对象 key 重复时
静默保留最后一个值。这两种行为在仿真控制边界都不安全：非有限数可能进入 PhysX；重复
key 则会让客户端签名/记录的命令与服务端实际执行的命令产生解释歧义。

控制 transport 和项目自有 JSONL 存储统一使用本模块，在协议或领域字段解析开始之前拒绝
歧义输入。这里保持标准 JSON 的其它语义，不负责 schema 字段、类型或业务范围校验。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any


def strict_json_loads(payload: str | bytes | bytearray) -> object:
    """解码一个 JSON 值，并拒绝重复 key、非标准常量和浮点溢出。

    JSON 整数仍由标准库解析为 Python ``int``；带小数点或指数的数字通过有限 float parser，
    因而 ``1e9999`` 这类语法合法但超出 float 范围的输入也会被拒绝。
    """

    return json.loads(
        payload,
        parse_constant=_reject_nonstandard_constant,
        parse_float=_finite_json_float,
        object_pairs_hook=_object_without_duplicate_keys,
    )


def strict_json_dumps(value: object, **kwargs: Any) -> str:
    """编码标准 JSON，并禁止输出 NaN 或正负 Infinity。

    其它 ``json.dumps`` 选项原样透传；``allow_nan=False`` 由本 API 固定，调用方不能通过
    kwargs 放宽控制/持久化边界。
    """

    return json.dumps(value, allow_nan=False, **kwargs)


def _reject_nonstandard_constant(value: str) -> object:
    """拒绝 ``json.loads`` 默认接受的 JavaScript 非标准数值常量。"""

    raise ValueError(f"JSON numeric constant {value!r} is not supported")


def _finite_json_float(value: str) -> float:
    """解析 JSON 浮点文本，同时禁止指数溢出为 infinity。"""

    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number {value!r} is outside the finite float range")
    return parsed


def _object_without_duplicate_keys(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    """按输入顺序构建 JSON object，并在首次重复 key 时拒绝整个 payload。

    使用 ``object_pairs_hook`` 才能在标准 decoder 合并成 dict 之前看到重复项；该检查会
    递归应用于所有层级的 JSON object。
    """

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


__all__ = ["strict_json_dumps", "strict_json_loads"]
