"""关节分组和稀疏目标展开工具。

配置文件里经常只想指定一组关节或少量关节目标，而 Isaac Lab 的 articulation 接口通常需要
完整 DOF 索引或完整目标向量。本模块负责在“名字空间”和“数组空间”之间做转换，并在名字
缺失时尽早报错。

职责边界:
    * 根据 Isaac ``dof_names`` 把有序关节名解析成整数索引。
    * 把稀疏 ``关节名 -> 位置`` 映射展开成完整 DOF 数组。
    * 不猜测关节别名，也不根据字符串相似度自动修正拼写。

顺序约定:
    所有顺序都由传入的名称列表和 Isaac ``dof_names`` 显式决定。配置中的目标顺序不会被
    自动重排；当名称缺失时立即抛错，可以避免后续控制器把目标写到错误 DOF。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from linkerbot_sim.robots.capabilities import RobotKind


@dataclass(frozen=True)
class JointGroup:
    """一个有名字且有顺序的关节组。

    输入字段:
        name: 关节组名称，例如 ``arm`` 或 ``hand``。
        joint_names: 有序关节名元组；数组第 i 个元素默认对应 ``joint_names[i]``。
    输出:
        实例可通过 ``indices_in`` 解析到 Isaac articulation DOF 索引。
    """

    name: str
    joint_names: tuple[str, ...]

    @classmethod
    def from_mapping(cls, name: str, data) -> "JointGroup":
        """从 YAML/字典中的序列创建关节组。

        参数:
            name: 关节组名称。
            data: YAML 读取出的序列，例如 ``["joint1", "joint2"]``。
        返回:
            ``JointGroup`` 实例。
        """

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise ValueError(f"Joint group {name!r} must be a sequence")
        return cls(name=name, joint_names=tuple(str(value) for value in data))

    def indices_in(
        self, dof_names: Sequence[str], *, allow_all: bool = True
    ) -> np.ndarray:
        """把本关节组解析成 articulation DOF 索引。

        参数:
            dof_names: Isaac articulation 返回的完整 DOF 名称序列。
            allow_all: 是否允许 ``["all"]`` 表示全部 DOF。
        返回:
            int ndarray，包含每个关节在 ``dof_names`` 中的索引。
        """

        # ``all`` 是少数允许的语义别名，用于测试或全 DOF 控制；除此之外必须精确匹配名称。
        if (
            allow_all
            and len(self.joint_names) == 1
            and self.joint_names[0].lower() == "all"
        ):
            return np.arange(len(dof_names), dtype=int)
        missing = [name for name in self.joint_names if name not in dof_names]
        if missing:
            raise ValueError(
                f"Joint group {self.name!r} has missing joints: {missing}. Available: {list(dof_names)}"
            )
        return np.asarray(
            [list(dof_names).index(name) for name in self.joint_names], dtype=int
        )


@dataclass(frozen=True)
class JointGroupLayout:
    """一个 articulation 已解析且互不重叠的 command-space groups。"""

    command_joint_names: tuple[str, ...]
    arm: tuple[str, ...] = ()
    hand: tuple[str, ...] = ()
    passive: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        command = _unique_names(self.command_joint_names, "command_joint_names")
        arm = _unique_names(self.arm, "joint_groups.arm")
        hand = _unique_names(self.hand, "joint_groups.hand")
        passive = _unique_names(self.passive, "joint_groups.passive")
        known = set(command)
        missing = sorted((set(arm) | set(hand) | set(passive)) - known)
        if missing:
            raise ValueError(f"joint groups contain unknown command joints: {missing}")
        overlap = sorted(
            (set(arm) & set(hand))
            | (set(arm) & set(passive))
            | (set(hand) & set(passive))
        )
        if overlap:
            raise ValueError(f"joint groups contain multiple writers: {overlap}")
        unassigned = sorted(known - set(arm) - set(hand) - set(passive))
        if unassigned:
            raise ValueError(
                "command joints must belong to arm, hand, or explicitly passive: "
                f"{unassigned}"
            )
        object.__setattr__(self, "command_joint_names", command)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "hand", hand)
        object.__setattr__(self, "passive", passive)

    @classmethod
    def resolve(
        cls,
        *,
        kind: RobotKind | str,
        command_joint_names: Sequence[str],
        joint_groups: object,
        planning_joint_names: Sequence[str] = (),
    ) -> "JointGroupLayout":
        """解析显式 groups，并校验 robot kind 与 planning-joint coverage。"""

        robot_kind = kind if isinstance(kind, RobotKind) else RobotKind.parse(kind)
        command = _unique_names(command_joint_names, "command_joint_names")
        explicit = _joint_group_mapping(joint_groups)
        arm = _group_names(explicit, "arm")
        hand = _group_names(explicit, "hand")
        passive = _group_names(explicit, "passive")

        layout = cls(command, arm=arm, hand=hand, passive=passive)
        layout.validate_kind(robot_kind)
        if planning_joint_names:
            layout.validate_planning_joints(planning_joint_names)
        return layout

    def validate_kind(self, kind: RobotKind | str) -> None:
        """校验 arm/hand group 是否与 ``RobotKind`` 声明完全一致。"""

        robot_kind = kind if isinstance(kind, RobotKind) else RobotKind.parse(kind)
        if robot_kind.has_arm != bool(self.arm):
            raise ValueError(
                f"robot.kind {robot_kind.value!r} requires "
                f"arm group present={robot_kind.has_arm}, got {list(self.arm)}"
            )
        if robot_kind.has_hand != bool(self.hand):
            raise ValueError(
                f"robot.kind {robot_kind.value!r} requires "
                f"hand group present={robot_kind.has_hand}, got {list(self.hand)}"
            )

    def validate_planning_joints(self, planning_joint_names: Sequence[str]) -> None:
        """要求 cuRobo active joints 与 arm group 集合完全一致。"""

        planning = tuple(str(name) for name in planning_joint_names)
        if len(set(planning)) != len(planning):
            raise ValueError("planning joint names contain duplicates")
        if set(planning) != set(self.arm):
            raise ValueError(
                "cuRobo active joints must exactly match joint_groups.arm: "
                f"planning={list(planning)}, arm={list(self.arm)}"
            )

    def names(self, group: str) -> tuple[str, ...]:
        """按 canonical group 名称返回冻结 joint order。"""

        normalized = str(group).lower()
        if normalized not in {"arm", "hand", "passive"}:
            raise ValueError(f"unknown joint group: {group!r}")
        return getattr(self, normalized)

    def indices(self, group: str) -> np.ndarray:
        """返回 group joints 在完整 command-space 中的列索引。"""

        selected = set(self.names(group))
        return np.asarray(
            [
                index
                for index, name in enumerate(self.command_joint_names)
                if name in selected
            ],
            dtype=int,
        )


def _unique_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    """冻结 joint name sequence，并拒绝空名称或重复项。"""

    names = tuple(str(value) for value in values)
    if any(not name for name in names):
        raise ValueError(f"{label} contains an empty joint name")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicates")
    return names


def _joint_group_mapping(value: object) -> Mapping[str, object]:
    """读取必需 joint_groups mapping，并拒绝未声明 group。"""

    if value is None:
        raise ValueError("joint_groups is required")
    if not isinstance(value, Mapping):
        raise ValueError("joint_groups must be a mapping")
    unsupported = set(value) - {"arm", "hand", "passive"}
    if unsupported:
        paths = ", ".join(f"joint_groups.{key}" for key in sorted(unsupported))
        raise ValueError(
            f"joint_groups contains unsupported keys: {sorted(unsupported)} "
            f"(full paths: {paths})"
        )
    return value


def _group_names(mapping, key: str) -> tuple[str, ...]:
    """读取单个 group 的 joint name sequence。"""

    value = mapping.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"joint_groups.{key} must be a sequence")
    return tuple(str(name) for name in value)


def resolve_joint_indices(
    dof_names: Sequence[str], requested_names: Sequence[str] | None
) -> np.ndarray:
    """把可选的关节名列表解析成 DOF 索引。

    参数:
        dof_names: Isaac articulation 的完整 DOF 名称序列。
        requested_names: 请求的关节名列表；为空或为 ``["all"]`` 时选择全部 DOF。
    返回:
        int ndarray，包含被选中 DOF 的索引。
    """

    if not requested_names or (
        len(requested_names) == 1 and requested_names[0].lower() == "all"
    ):
        return np.arange(len(dof_names), dtype=int)
    missing = [name for name in requested_names if name not in dof_names]
    if missing:
        raise ValueError(
            f"Requested joints were not found: {missing}. Available DOFs: {list(dof_names)}"
        )
    return np.asarray(
        [list(dof_names).index(name) for name in requested_names], dtype=int
    )


def target_vector_from_mapping(
    dof_names: Sequence[str],
    targets: dict[str, float],
    *,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """把稀疏的 ``关节名 -> 位置`` 映射展开成有序目标向量。

    参数:
        dof_names: 有序关节名序列，定义返回向量顺序；可以是完整 DOF，也可以是
            controller command space。
        targets: 稀疏目标映射，值单位通常为 rad。
        base: 可选基准向量；未在 ``targets`` 中出现的关节沿用它。
    返回:
        shape ``(len(dof_names),)`` 的目标向量；没有 ``base`` 时未指定关节为 0。
    """

    # base 表示“未指定关节沿用当前/上一阶段目标”。没有 base 时只能填 0，适合构造简单
    # 全新目标；抓取动作通常会传 base 以避免未涉及关节突然归零。
    if base is None:
        vector = np.zeros(len(dof_names), dtype=float)
    else:
        vector = np.asarray(base, dtype=float).reshape(-1).copy()
        if vector.size != len(dof_names):
            raise ValueError(
                f"base target expected {len(dof_names)} values, got {vector.size}"
            )

    index_by_name = {name: index for index, name in enumerate(dof_names)}
    missing = [name for name in targets if name not in index_by_name]
    if missing:
        raise ValueError(
            f"Target joints were not found: {missing}. Available DOFs: {list(dof_names)}"
        )
    # Python dict 保留插入顺序，但这里按名称定位写入完整数组，因此稀疏映射顺序不影响结果。
    for name, value in targets.items():
        vector[index_by_name[name]] = float(value)
    return vector
