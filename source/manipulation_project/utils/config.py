"""YAML 配置读取与仓库相对路径解析。

配置文件中的路径默认按仓库根目录解析，避免从不同工作目录启动脚本时路径含义变化。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

from manipulation_project.utils.paths import repo_path


T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件。

    参数:
        path: YAML 路径，可以是绝对路径，也可以是仓库相对路径。
    返回:
        顶层 mapping 转成的 dict；空文件返回空 dict。
    """

    resolved = repo_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML config: {resolved}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并两个 mapping。

    参数:
        base: 基础配置。
        override: 覆盖配置；同名非 mapping 值会替换 ``base``。
    返回:
        新 dict，不会原地修改输入。
    """

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """获取必需的嵌套 mapping。

    参数:
        config: 配置 mapping。
        key: 需要读取的键。
    返回:
        ``config[key]``，类型保证为 ``Mapping``。
    """

    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def get_required(config: Mapping[str, Any], key: str, expected_type: type[T] | tuple[type, ...] | None = None) -> T:
    """读取必需配置项，并可选校验类型。

    参数:
        config: 配置 mapping。
        key: 必需键名。
        expected_type: 可选类型或类型元组。
    返回:
        配置值；缺失或类型不匹配时抛出 ``ValueError``。
    """

    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
    value = config[key]
    if expected_type is not None and not isinstance(value, expected_type):
        raise ValueError(f"Config key {key!r} must be {expected_type}, got {type(value).__name__}")
    return value  # type: ignore[return-value]


def resolve_config_path(config: Mapping[str, Any], key: str) -> Path:
    """读取配置路径并解析为绝对路径。

    参数:
        config: 配置 mapping。
        key: 存放路径字符串的键名。
    返回:
        ``Path``；相对路径会按仓库根目录解析。
    """

    return repo_path(get_required(config, key, (str, Path)))
