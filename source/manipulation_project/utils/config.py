"""YAML 配置读取与仓库相对路径解析。

配置文件中的路径默认按仓库根目录解析，避免从不同工作目录启动脚本时路径含义变化。本模块
只处理纯 Python 数据结构，不导入 Isaac/Omni；因此可以在 CLI 参数解析和测试阶段安全使用。

职责边界:
    * 读取 YAML 并保证顶层是 mapping。
    * 递归合并默认配置和用户覆盖配置。
    * 提供常用的必填字段、嵌套 mapping 和路径读取 helper。

错误统一抛 ``ValueError``/``FileNotFoundError``，让脚本入口决定如何展示。这里不把配置
转换成具体 dataclass；各领域模块负责解释自己的字段含义和单位。
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

    # 统一通过 repo_path 解析，保证 ``configs/...`` 在任意当前工作目录下都指向仓库内文件。
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

    # 只创建新 dict，不修改输入对象；这让默认配置可以在多个测试/脚本中安全复用。
    merged = dict(base)
    for key, value in override.items():
        # 只有两边都是 mapping 时才递归合并；列表和标量按 override 整体替换，避免对轨迹点、
        # 关节目标等有顺序含义的数据做不符合预期的逐项合并。
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
