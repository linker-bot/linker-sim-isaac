"""与资产格式无关的 mimic 关节关系发现。

MJCF equality 与 URDF ``<mimic>`` 对从属关节使用不同语法。本模块只按资产扩展名选择对应
严格解析器，并把结果统一成运行时消费的“从属关节、主关节、常数项优先多项式系数”结构。
它不加载 Isaac 资产，也不求值或下发关节命令。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.robots.mimic.mjcf import parse_mjcf_joint_equalities
from linkerbot_sim.robots.mimic.urdf import parse_urdf_joint_mimics


@dataclass(frozen=True)
class AssetMimicRelation:
    """供运行时消费的规范 mimic 关系。

    ``dependent_joint`` 跟随 ``master_joint``；``polycoef`` 按常数项到高次项排序，使 MJCF
    多项式和 URDF 仿射关系使用同一表示。实例冻结，适合在资产解析后缓存和跨控制组件共享。
    """

    dependent_joint: str
    master_joint: str
    polycoef: tuple[float, ...]


def parse_asset_mimic_relations(
    path: str | Path | None,
) -> list[AssetMimicRelation]:
    """按资产扩展名读取并规范化 mimic 关系。

    参数:
        path: MJCF/URDF 文件路径；``None`` 表示没有可解析资产。
    返回:
        按源文件声明顺序排列的新列表；未知扩展名或空路径返回空列表。
    异常:
        OSError: 已识别资产文件无法读取。
        ValueError: mimic/equality 关系重复、引用无效或形成环。
        xml.etree.ElementTree.ParseError: XML 内容格式错误。
    副作用:
        只读取资产 XML，不修改文件或创建仿真实例。
    """

    if path is None:
        return []
    asset_path = Path(path)
    suffix = asset_path.suffix.lower()
    if suffix == ".urdf":
        source = parse_urdf_joint_mimics(asset_path)
    elif suffix in {".xml", ".mjcf"}:
        source = parse_mjcf_joint_equalities(asset_path)
    else:
        return []
    return [
        AssetMimicRelation(
            dependent_joint=relation.dependent_joint,
            master_joint=relation.master_joint,
            polycoef=relation.polycoef,
        )
        for relation in source
    ]


def mimic_follower_joint_names(path: str | Path | None) -> set[str]:
    """返回仿真资产声明的全部从属关节名称。

    该集合用于从普通控制关节中排除由主关节间接驱动的 follower；读取和异常语义与
    :func:`parse_asset_mimic_relations` 相同。
    """

    return {relation.dependent_joint for relation in parse_asset_mimic_relations(path)}


__all__ = [
    "AssetMimicRelation",
    "mimic_follower_joint_names",
    "parse_asset_mimic_relations",
]
