"""资产实例 USD prim 路径的跨领域所有权校验。

机器人和对象分别拥有以实例 ``prim_path`` 为根的整棵 USD 子树。两个根路径相同，或其中
一个是另一个的祖先时，后续导入可能覆盖、删除或错误绑定已有 prim。本模块在创建 stage 前
统一拒绝域内及跨域重叠；它只比较规范字符串，不访问 USD stage，也不判断 prim 是否存在。
"""

from __future__ import annotations

from collections.abc import Mapping


def validate_disjoint_instance_prim_paths(
    *,
    robot_paths: Mapping[str, str],
    object_paths: Mapping[str, str],
) -> None:
    """拒绝拥有重叠 USD 子树的机器人和对象实例。

    参数:
        robot_paths: ``机器人实例标识 -> 绝对 prim 路径`` 的 mapping。
        object_paths: ``对象实例标识 -> 绝对 prim 路径`` 的 mapping。
    返回:
        校验通过时返回 ``None``。
    异常:
        ValueError: 同域两个实例或机器人与对象之间存在相同/祖先子孙路径。
    副作用:
        无；不会规范化输入 mapping，也不会访问或修改 USD stage。
    """

    _validate_domain_paths(robot_paths, domain="robot")
    _validate_domain_paths(object_paths, domain="object")
    for robot_name, robot_path in robot_paths.items():
        for object_name, object_path in object_paths.items():
            if not _paths_overlap(robot_path, object_path):
                continue
            raise ValueError(
                "Robot and object instance prim paths overlap: "
                f"robot {robot_name!r}={robot_path!r}, "
                f"object {object_name!r}={object_path!r}"
            )


def _validate_domain_paths(paths: Mapping[str, str], *, domain: str) -> None:
    """在单一实例域内做成对检查，并保留冲突实例名用于错误定位。"""

    items = tuple(paths.items())
    for index, (first_name, first_path) in enumerate(items):
        for second_name, second_path in items[index + 1 :]:
            if not _paths_overlap(first_path, second_path):
                continue
            raise ValueError(
                f"{domain.capitalize()} instance prim paths overlap: "
                f"{domain} {first_name!r}={first_path!r}, "
                f"{domain} {second_name!r}={second_path!r}"
            )


def _paths_overlap(first: str, second: str) -> bool:
    """按 USD 路径段边界判断两棵子树是否相交。"""

    first = first.rstrip("/")
    second = second.rstrip("/")
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


__all__ = ["validate_disjoint_instance_prim_paths"]
