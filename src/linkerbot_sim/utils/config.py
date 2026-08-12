"""YAML 配置读取与仓库相对路径解析。

配置文件中的路径默认按仓库根目录解析，避免从不同工作目录启动脚本时路径含义变化。本模块
只处理纯 Python 数据结构，不导入 Isaac/Omni；因此可以在 CLI 参数解析和测试阶段安全使用。

职责边界:
    * 读取 YAML 并保证顶层是 mapping。
    * 递归合并默认配置和用户覆盖配置。
    * 严格校验只允许本机监听的网络 host。

错误统一抛 ``ValueError``/``FileNotFoundError``，让脚本入口决定如何展示。这里不把配置
转换成具体 dataclass；各领域模块负责解释自己的字段含义和单位。
"""

from __future__ import annotations

from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from linkerbot_sim.utils.paths import repo_path


class _StrictSafeLoader(yaml.SafeLoader):
    """在每一层 mapping 拒绝重复键的安全 YAML loader。"""

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        """构造一层 mapping，并保留重复键两次出现的位置。

        PyYAML 默认采用后值覆盖前值，这会让配置审计看不到被遮蔽字段。这里在构造 value
        前先检查已解析 key，并把首次行列和重复节点位置一起交给 ``ConstructorError``。
        """

        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        key_marks: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                first_mark = key_marks[key]
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    (
                        f"found duplicate mapping key {key!r}; first occurrence "
                        f"at line {first_mark.line + 1}, "
                        f"column {first_mark.column + 1}"
                    ),
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
            key_marks[key] = key_node.start_mark
        return mapping


def require_loopback_host(value: object, *, label: str) -> str:
    """不执行 DNS 或打开 socket，严格校验 listener host 为 loopback。

    只接受 ``localhost`` 或数值型 IPv4/IPv6 loopback；其他主机名即使当前 DNS 指向本机
    也拒绝，避免部署环境解析变化意外扩大监听面。
    """

    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be a non-empty host without whitespace")
    if value.casefold() == "localhost":
        return value
    try:
        address = ip_address(value)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be localhost or a numeric loopback address"
        ) from exc
    if not address.is_loopback:
        raise ValueError(f"{label} must be a loopback address")
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件。

    参数:
        path: YAML 路径，可以是绝对路径，也可以是仓库相对路径。
    返回:
        顶层 mapping 转成的 dict。
    异常:
        ValueError: YAML 为空、顶层不是 mapping、包含重复键或语法无效。
    """

    # 统一通过 repo_path 解析，保证 ``configs/...`` 在任意当前工作目录下都指向仓库内文件。
    resolved = repo_path(path)
    try:
        with resolved.open("r", encoding="utf-8") as file:
            data = yaml.load(file, Loader=_StrictSafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config {resolved}: {exc}") from exc
    if data is None:
        raise ValueError(f"YAML config is empty; expected a mapping: {resolved}")
    if not isinstance(data, Mapping):
        raise ValueError(
            f"Expected a mapping in YAML config {resolved}, got {type(data).__name__}"
        )
    return dict(data)


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
