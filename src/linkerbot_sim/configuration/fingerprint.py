"""与加载位置无关的 canonical 配置载荷与 SHA-256 指纹。

该模块只依赖 Python 标准库，可在启动 Kit、导入 Torch 或创建 CUDA 资源之前使用。
验证脚本和运行时必须通过这里识别同一份配置，不能各自依赖 ``repr`` 或 YAML 文件路径。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path


_PROVENANCE_FIELD_NAMES = frozenset({"sources", "provenance"})


def semantic_config_payload(value: object) -> object:
    """把 typed 配置图投影为稳定、JSON-compatible 的语义载荷。

    ``sources``/``provenance`` 只记录配置从哪里加载，不改变仿真行为，因此显式排除。
    dataclass 中同时声明 ``compare=False`` 与 ``repr=False`` 的字段也表示内部 bookkeeping，
    不进入语义身份。其余字段必须完整递归，尤其包括 catalog 已解析的资产和 controller
    profile，保证 controller 增益变化会使 snapshot/clone 兼容性检查失败。
    """

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: semantic_config_payload(getattr(value, item.name))
            for item in fields(value)
            if item.name not in _PROVENANCE_FIELD_NAMES and (item.compare or item.repr)
        }
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "semantic configuration mappings require string keys, "
                    f"got {type(key).__name__}"
                )
            normalized[key] = semantic_config_payload(item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [semantic_config_payload(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [semantic_config_payload(item) for item in value]
        return sorted(normalized_items, key=_canonical_json)
    if isinstance(value, Enum):
        return semantic_config_payload(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "unsupported semantic configuration value type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def semantic_config_fingerprint(value: object) -> str:
    """返回配置语义载荷的 canonical JSON SHA-256。"""

    encoded = _canonical_json(semantic_config_payload(value)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = ["semantic_config_fingerprint", "semantic_config_payload"]
