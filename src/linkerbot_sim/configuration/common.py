"""typed configuration 共用的严格解析原语。

本模块不拥有任何产品、物理引擎或设备语义，也不导入 Isaac、Torch、cuRobo 或资源
I/O。所有配置都能在启动 Kit 和创建 CUDA 资源之前完成纯 Python 校验。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from types import MappingProxyType


class ConfigurationError(ValueError):
    """配置图无法被严格解释时抛出的统一异常。"""


def strict_mapping(value: object, *, label: str) -> dict[str, object]:
    """返回字符串键 mapping，并拒绝 YAML 中难以审计的非字符串键。"""

    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigurationError(f"{label} keys must be non-empty strings, got {key!r}")
        result[key] = item
    return result


def require_keys(
    mapping: Mapping[str, object],
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    label: str,
) -> None:
    """一次性拒绝缺失字段与当前 schema 未声明的字段。"""

    present = set(mapping)
    missing = sorted(set(required) - present)
    unknown = sorted(present - set(required) - set(optional))
    if missing:
        raise ConfigurationError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ConfigurationError(f"{label} contains unknown fields: {', '.join(unknown)}")


def as_string(value: object, *, label: str, choices: set[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConfigurationError(f"{label} must be a non-empty string without leading or trailing whitespace")
    if choices is not None and value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigurationError(f"{label} must be one of {{{allowed}}}, got {value!r}")
    return value


def as_bool(value: object, *, label: str) -> bool:
    # ``bool`` 是 ``int`` 的子类，因此必须使用精确类型判断，避免 0/1 偷渡。
    if type(value) is not bool:
        raise ConfigurationError(f"{label} must be a YAML boolean, got {value!r}")
    return value


def as_int(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ConfigurationError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{label} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{label} must be <= {maximum}, got {value!r}")
    return value


def as_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a finite number, got {value!r}")
    result = float(value)
    if not isfinite(result):
        raise ConfigurationError(f"{label} must be a finite number, got {value!r}")
    if strictly_positive and result <= 0.0:
        raise ConfigurationError(f"{label} must be > 0, got {value!r}")
    if minimum is not None and result < minimum:
        raise ConfigurationError(f"{label} must be >= {minimum}, got {value!r}")
    if maximum is not None and result > maximum:
        raise ConfigurationError(f"{label} must be <= {maximum}, got {value!r}")
    return result


def as_float_tuple(
    value: object,
    *,
    label: str,
    length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{label} must be a numeric sequence of length {length}")
    if len(value) != length:
        raise ConfigurationError(f"{label} must contain exactly {length} items")
    return tuple(
        as_float(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )


def as_string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{label} must be a string sequence")
    result = tuple(
        as_string(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ConfigurationError(f"{label} must not contain duplicate values")
    return result


def reject_forbidden_keys(
    value: object,
    *,
    forbidden_fragments: frozenset[str],
    label: str,
) -> None:
    """递归拒绝某产品闭包不允许出现的配置概念。"""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            fragment = next(
                (item for item in forbidden_fragments if item in key),
                None,
            )
            if fragment is not None:
                raise ConfigurationError(
                    f"{label}.{raw_key} belongs to Kaleidoscope-forbidden configuration concept {fragment!r}"
                )
            reject_forbidden_keys(
                child,
                forbidden_fragments=forbidden_fragments,
                label=f"{label}.{raw_key}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            reject_forbidden_keys(
                child,
                forbidden_fragments=forbidden_fragments,
                label=f"{label}[{index}]",
            )


def deep_freeze_configuration(value: object) -> object:
    """递归复制配置容器，切断 YAML mapping/list 的所有可变别名。"""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: deep_freeze_configuration(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze_configuration(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze_configuration(item) for item in value)
    return value


def deep_freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """深冻结字符串键配置 mapping，并保留只读 ``Mapping`` 接口。"""

    frozen = deep_freeze_configuration(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - 由参数类型保证
        raise TypeError("configuration value must be a mapping")
    return frozen


__all__ = [
    "ConfigurationError",
    "as_bool",
    "as_float",
    "as_float_tuple",
    "as_int",
    "as_string",
    "as_string_tuple",
    "deep_freeze_configuration",
    "deep_freeze_mapping",
    "reject_forbidden_keys",
    "require_keys",
    "strict_mapping",
]
