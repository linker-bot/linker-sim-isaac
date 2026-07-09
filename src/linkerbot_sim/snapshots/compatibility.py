"""恢复逻辑仿真快照前的兼容性检查。

snapshot schema 只保证“数据形状正确”；真正写回 runtime 前，还必须确认目标 runtime
有对应机器人、对象、关节和 body。本模块负责把 snapshot 名字空间解析到目标 runtime
名字空间，并给 adapter 提供稳定的 index mapping。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from linkerbot_sim.snapshots.schema import ObjectSnapshot, SimulationSnapshot


@dataclass(frozen=True)
class RobotTargetDescriptor:
    """目标 runtime 中一个机器人角色的元信息。

    adapter 会从真实 runtime 生成该描述符；兼容性检查只依赖名字和 profile/fingerprint，
    不直接触碰 Isaac/控制器对象。
    """

    role: str
    joint_names: tuple[str, ...]
    robot_profile: str | None = None
    asset_fingerprint: str | None = None
    command_joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """标准化目标机器人名字字段，并拒绝空 role/重复关节名。"""

        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(
            self,
            "joint_names",
            tuple(str(name) for name in self.joint_names),
        )
        object.__setattr__(
            self,
            "command_joint_names",
            tuple(str(name) for name in self.command_joint_names),
        )
        if not self.role:
            raise ValueError("RobotTargetDescriptor.role cannot be empty")
        if not self.joint_names:
            raise ValueError("RobotTargetDescriptor.joint_names cannot be empty")
        _reject_duplicates(self.joint_names, "RobotTargetDescriptor.joint_names")
        _reject_duplicates(
            self.command_joint_names,
            "RobotTargetDescriptor.command_joint_names",
        )


@dataclass(frozen=True)
class ObjectTargetDescriptor:
    """目标 runtime 中一个对象的元信息。

    ``body_names`` 只在 dynamic-chain 等多刚体对象上使用；普通刚体对象为空即可。
    """

    name: str
    object_profile: str | None = None
    body_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """标准化目标对象名字字段，并拒绝重复 body 名。"""

        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "body_names",
            tuple(str(name) for name in self.body_names),
        )
        if not self.name:
            raise ValueError("ObjectTargetDescriptor.name cannot be empty")
        _reject_duplicates(
            self.body_names,
            "ObjectTargetDescriptor.body_names",
        )


@dataclass(frozen=True)
class SnapshotTargetDescriptor:
    """目标 runtime 的完整快照恢复能力描述。

    tiled、single、dual 都会被统一表达成这个结构，所以后续检查逻辑不需要知道目标
    runtime 的具体类型。
    """

    runtime_kind: str
    robots: Mapping[str, RobotTargetDescriptor]
    objects: Mapping[str, ObjectTargetDescriptor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结并校验目标 runtime 描述符的顶层结构。"""

        object.__setattr__(self, "runtime_kind", str(self.runtime_kind))
        object.__setattr__(self, "robots", _robot_targets(self.robots))
        object.__setattr__(self, "objects", _object_targets(self.objects))
        if not self.runtime_kind:
            raise ValueError("SnapshotTargetDescriptor.runtime_kind cannot be empty")


@dataclass(frozen=True)
class JointMapping:
    """从 snapshot 顺序到目标 runtime 顺序的 index mapping。

    关节/body 名字可能相同但排列不同，因此恢复时不能按数组位置盲写，必须按名字解析出
    source/target index 后再拷贝。
    """

    source_indices: np.ndarray
    target_indices: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """把索引数组规范化为一维 int，并校验与名字数量一致。"""

        source = np.asarray(self.source_indices, dtype=int).reshape(-1)
        target = np.asarray(self.target_indices, dtype=int).reshape(-1)
        names = tuple(str(name) for name in self.names)
        if source.shape != target.shape or source.size != len(names):
            raise ValueError("JointMapping indices and names must have the same length")
        object.__setattr__(self, "source_indices", source)
        object.__setattr__(self, "target_indices", target)
        object.__setattr__(self, "names", names)


@dataclass(frozen=True)
class RobotCompatibilityMapping:
    """一个 snapshot 机器人角色到一个目标机器人角色的解析结果。"""

    source_role: str
    target_role: str
    joints: JointMapping
    command_joints: JointMapping | None = None


@dataclass(frozen=True)
class ObjectCompatibilityMapping:
    """一个 snapshot 对象到目标对象的解析结果。"""

    source_name: str
    target_name: str
    bodies: JointMapping | None = None


@dataclass(frozen=True)
class SnapshotCompatibilityResult:
    """兼容性检查结果。

    ``robot_mappings`` 和 ``object_mappings`` 只包含可以恢复的 entry；
    ``issues`` 收集所有不兼容原因。adapter 在 strict 模式下通常通过
    ``require_snapshot_compatibility`` 直接失败。
    """

    compatible: bool
    issues: tuple[str, ...]
    robot_mappings: Mapping[str, RobotCompatibilityMapping] = field(default_factory=dict)
    object_mappings: Mapping[str, ObjectCompatibilityMapping] = field(default_factory=dict)
    partial: bool = False


class SnapshotCompatibilityError(ValueError):
    """当 snapshot 不能安全写入目标 runtime 时抛出。"""


def check_snapshot_compatibility(
    snapshot: SimulationSnapshot,
    target: SnapshotTargetDescriptor,
    *,
    robot_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotCompatibilityResult:
    """检查 ``snapshot`` 是否可以恢复到 ``target``。

    ``robot_map`` 表示 snapshot robot role 到目标 robot role 的映射。strict 模式要求
    名字集合完全一致，但允许顺序不同；非 strict 模式只恢复两侧共有的名字。
    """

    issues: list[str] = []
    robot_mappings: dict[str, RobotCompatibilityMapping] = {}
    object_mappings: dict[str, ObjectCompatibilityMapping] = {}
    resolved_robot_map = _resolve_robot_map(snapshot, target, robot_map, issues)
    for source_role, target_role in resolved_robot_map.items():
        # 先检查 profile/fingerprint 这类“同名但不同资产”的风险，再解析关节名字。
        # profile/fingerprint 任一侧缺失时不阻断，便于兼容旧 runtime 和测试 fake。
        source_robot = snapshot.robots[source_role]
        target_robot = target.robots[target_role]
        _check_optional_match(
            source_robot.robot_profile,
            target_robot.robot_profile,
            f"robots.{source_role}.robot_profile",
            issues,
        )
        _check_optional_match(
            source_robot.asset_fingerprint,
            target_robot.asset_fingerprint,
            f"robots.{source_role}.asset_fingerprint",
            issues,
        )
        joint_mapping = _name_mapping(
            source_robot.joint_names,
            target_robot.joint_names,
            f"robots.{source_role}.joint_names",
            strict=strict,
            issues=issues,
        )
        command_mapping = None
        if source_robot.command_joint_names and target_robot.command_joint_names:
            # command targets 与 joint state 分开映射，方便以后支持“保存全 DOF、
            # 只恢复 command DOF”的更宽 schema。
            command_mapping = _name_mapping(
                source_robot.command_joint_names,
                target_robot.command_joint_names,
                f"robots.{source_role}.command_joint_names",
                strict=strict,
                issues=issues,
            )
        if joint_mapping is not None:
            robot_mappings[target_role] = RobotCompatibilityMapping(
                source_role=source_role,
                target_role=target_role,
                joints=joint_mapping,
                command_joints=command_mapping,
            )
    for object_name, source_object in snapshot.objects.items():
        # 对象当前按名字匹配。机器人可以通过 robot_map 改名，是因为 single/dual/tiled
        # 常用不同 role；对象名通常来自 runtime handle，自动改名反而更危险。
        target_object = target.objects.get(object_name)
        if target_object is None:
            issues.append(f"objects.{object_name} is not present in target")
            continue
        _check_optional_match(
            source_object.object_profile,
            target_object.object_profile,
            f"objects.{object_name}.object_profile",
            issues,
        )
        body_mapping = _object_body_mapping(
            source_object,
            target_object,
            strict=strict,
            issues=issues,
        )
        if source_object.body_names and body_mapping is None:
            continue
        object_mappings[object_name] = ObjectCompatibilityMapping(
            source_name=object_name,
            target_name=object_name,
            bodies=body_mapping,
        )
    partial = len(robot_mappings) < len(snapshot.robots) or len(object_mappings) < len(
        snapshot.objects
    )
    return SnapshotCompatibilityResult(
        compatible=not issues,
        issues=tuple(issues),
        robot_mappings=robot_mappings,
        object_mappings=object_mappings,
        partial=partial,
    )


def require_snapshot_compatibility(
    snapshot: SimulationSnapshot,
    target: SnapshotTargetDescriptor,
    *,
    robot_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotCompatibilityResult:
    """返回兼容性结果；如不兼容则抛出 ``SnapshotCompatibilityError``。"""

    result = check_snapshot_compatibility(
        snapshot,
        target,
        robot_map=robot_map,
        strict=strict,
    )
    if not result.compatible:
        raise SnapshotCompatibilityError("; ".join(result.issues))
    return result


def _resolve_robot_map(
    snapshot: SimulationSnapshot,
    target: SnapshotTargetDescriptor,
    robot_map: Mapping[str, str] | None,
    issues: list[str],
) -> dict[str, str]:
    """解析 snapshot robot role 到目标 robot role 的映射。

    映射优先级：显式 ``robot_map``、同名 role、一对一自动映射。除此之外都要求用户显式
    指定，避免 dual/single/tiled 多机器人场景误配。
    """

    if robot_map is not None:
        # 显式 robot_map 最高优先级，主要用于 dual -> single、single -> tiled named robot
        # 这类 role 名字不一致但机械臂配置兼容的场景。
        resolved: dict[str, str] = {}
        for source_role, target_role in robot_map.items():
            source = str(source_role)
            target_role_str = str(target_role)
            if source not in snapshot.robots:
                issues.append(f"robot_map source role {source!r} is not in snapshot")
                continue
            if target_role_str not in target.robots:
                issues.append(f"robot_map target role {target_role_str!r} is not in target")
                continue
            resolved[source] = target_role_str
        return resolved
    # 常规情况优先按同名 role 匹配，例如 tiled 的同名机器人或 dual 的 left/right。
    exact = {
        role: role
        for role in snapshot.robots
        if role in target.robots
    }
    if exact:
        return exact
    # 单机器人 snapshot 和单机器人目标之间允许自动映射，避免用户为 single <-> tiled
    # 这种一对一场景额外写 robot_map。
    if len(snapshot.robots) == 1 and len(target.robots) == 1:
        return {
            next(iter(snapshot.robots.keys())): next(iter(target.robots.keys())),
        }
    issues.append("robot_map is required to map snapshot robots to target robots")
    return {}


def _name_mapping(
    source_names: tuple[str, ...],
    target_names: tuple[str, ...],
    label: str,
    *,
    strict: bool,
    issues: list[str],
) -> JointMapping | None:
    """按名字生成 source -> target 的关节/body 索引映射。

    strict 模式要求两侧名字集合完全一致；非 strict 模式允许只恢复交集，但交集为空仍然
    视为不可恢复。
    """

    # 以名字建立映射，而不是假设关节/body 数组顺序一致。很多 USD/控制器配置会调整
    # command joint 顺序，按名字恢复可以避免“看起来维度一致但写错关节”的问题。
    source_index = {name: index for index, name in enumerate(source_names)}
    target_index = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in target_names if name not in source_index]
    extra = [name for name in source_names if name not in target_index]
    if strict and (missing or extra):
        # strict 模式要求完整互通；缺目标关节或快照多出关节都视为不兼容。
        if missing:
            issues.append(f"{label} missing in snapshot: {missing}")
        if extra:
            issues.append(f"{label} missing in target: {extra}")
        return None
    # 非 strict 模式只恢复交集，用于调试或资产小幅变更后的手动迁移。
    common_names = tuple(name for name in target_names if name in source_index)
    if not common_names:
        issues.append(f"{label} has no common names")
        return None
    return JointMapping(
        source_indices=np.asarray([source_index[name] for name in common_names], dtype=int),
        target_indices=np.asarray([target_index[name] for name in common_names], dtype=int),
        names=common_names,
    )


def _object_body_mapping(
    source_object: ObjectSnapshot,
    target_object: ObjectTargetDescriptor,
    *,
    strict: bool,
    issues: list[str],
) -> JointMapping | None:
    """解析 dynamic object 的 child body 名字映射。

    普通刚体对象不需要 body mapping；只有 source/target 都声明 body_names 时才生成
    映射。
    """

    # 普通对象没有 body_names，只需要恢复 root pose；dynamic-chain 必须两侧都有 body
    # 名字，否则无法安全恢复每个 child rigid body。
    if not source_object.body_names and not target_object.body_names:
        return None
    if source_object.body_names and not target_object.body_names:
        issues.append(f"objects.{source_object.name}.body_names missing in target")
        return None
    if target_object.body_names and not source_object.body_names:
        issues.append(f"objects.{source_object.name}.body_names missing in snapshot")
        return None
    return _name_mapping(
        source_object.body_names,
        target_object.body_names,
        f"objects.{source_object.name}.body_names",
        strict=strict,
        issues=issues,
    )


def _check_optional_match(
    source: str | None,
    target: str | None,
    label: str,
    issues: list[str],
) -> None:
    """在两侧都提供 profile/fingerprint 时要求值相同。"""

    if source is not None and target is not None and source != target:
        issues.append(f"{label} mismatch: snapshot={source!r} target={target!r}")


def _robot_targets(
    values: Mapping[str, RobotTargetDescriptor],
) -> dict[str, RobotTargetDescriptor]:
    """校验目标 robot descriptor mapping，并返回普通 dict 副本。"""

    if not isinstance(values, Mapping):
        raise ValueError("SnapshotTargetDescriptor.robots must be a mapping")
    result = {}
    for role, descriptor in values.items():
        if not isinstance(descriptor, RobotTargetDescriptor):
            raise ValueError("target robot values must be RobotTargetDescriptor")
        key = str(role)
        if key != descriptor.role:
            raise ValueError(
                f"target robot key {key!r} does not match role {descriptor.role!r}"
            )
        result[key] = descriptor
    return result


def _object_targets(
    values: Mapping[str, ObjectTargetDescriptor],
) -> dict[str, ObjectTargetDescriptor]:
    """校验目标 object descriptor mapping，并返回普通 dict 副本。"""

    if not isinstance(values, Mapping):
        raise ValueError("SnapshotTargetDescriptor.objects must be a mapping")
    result = {}
    for name, descriptor in values.items():
        if not isinstance(descriptor, ObjectTargetDescriptor):
            raise ValueError("target object values must be ObjectTargetDescriptor")
        key = str(name)
        if key != descriptor.name:
            raise ValueError(
                f"target object key {key!r} does not match name {descriptor.name!r}"
            )
        result[key] = descriptor
    return result


def _reject_duplicates(values: tuple[str, ...], label: str) -> None:
    """如果名字 tuple 中存在重复项则抛出字段化错误。"""

    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicates")
