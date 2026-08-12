# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""把一个 Newton prototype 复制成彼此独立的 homogeneous worlds。

label 重写和逐 world builder 流程借鉴 Isaac Lab ``release/3.0.0-beta2.patch1`` 中
BSD-3-Clause 的 ``isaaclab_newton.cloner.newton_replicate``。本项目只需要同构场景，因此
公开合同是“一个已配置 USD prototype + 每个 Newton world 的目标 root/transform”。

本模块刻意停在 ``ModelBuilder.finalize`` 之前：replication 只拥有拓扑构建，不拥有 CUDA
model/state/solver。Newton runtime 随后在自己的 stream 上 finalize，并分别对单份 prototype
与最终多 world model 做 native JOINT equality 审计。这里任何时候都不会删除、改写或用
drive 替代 prototype 解析出的任一主从关系。
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
import math
from typing import Any


# 项目在 USD 中按 SI/弧度语义写入 revolute drive 的 stiffness 和 damping。Newton
# ``add_usd`` 内部会先除以 ``DegreesToRadian / joint_drive_gains_scaling``；传入一度对应
# 的弧度值可令这个换算的净倍率为 1，避免把已经是每弧度的增益再次放大 180/pi 倍。
_USD_RADIAN_DRIVE_GAIN_SCALING = math.pi / 180.0


# 这些 Newton 内建字符串列都有对应 ``*_world`` 列。label 必须按 world 重写，否则
# env_1 的 view 仍会绑定到 env_0 路径。列表同时覆盖两种原生耦合表示，目的只是保证复制
# 完整；生产 ``mujoco`` variant 随后会审计为仅有 equality_constraint(EqType.JOINT)，不会
# 将 constraint_mimic 当作可互换实现。
_BUILTIN_LABEL_TYPES: tuple[str, ...] = (
    "body",
    "joint",
    "shape",
    "articulation",
    "constraint_mimic",
    "equality_constraint",
)


@dataclass(frozen=True)
class NewtonReplicationResult:
    """单次解析 prototype 并复制 worlds 后的未 finalize 结果。

    ``builder`` 包含全局静态实体和全部 world；``prototype_builder`` 保留未经复制的一份
    拓扑证据。二者同时返回是为了让 manager 能区分“资产解析就错了”与“复制后映射错了”。
    """

    builder: object
    """包含全局静态 shape 与全部 Newton world 的最终 builder。"""

    prototype_builder: object
    """只从 ``prototype_root`` 解析一次、尚未复制的证据 builder。"""

    global_stage_info: Mapping[str, Any]
    """解析 environment scope 之外 stage 内容时得到的全局元数据。"""

    prototype_stage_info: Mapping[str, Any]
    """唯一一次 prototype ``add_usd`` 返回的环境内元数据。"""

    prototype_root: str
    destination_roots: tuple[str, ...]
    world_transforms: tuple[object, ...]
    source_world_transform: object
    environment_root: str

    @property
    def num_worlds(self) -> int:
        """返回 ``builder`` 中彼此独立的 Newton world 数。"""

        return len(self.destination_roots)


@dataclass(frozen=True)
class _NewtonDependencies:
    model_builder_type: type
    solver_mujoco_type: type
    schema_resolver_newton_type: type
    schema_resolver_mjc_type: type
    schema_resolver_physx_type: type
    warp: object


@dataclass(frozen=True)
class _BuilderCounts:
    particle: int
    body: int
    shape: int
    joint: int
    articulation: int
    equality_constraint: int
    constraint_mimic: int

    @classmethod
    def from_builder(cls, builder: object) -> _BuilderCounts:
        return cls(
            particle=int(getattr(builder, "particle_count", 0)),
            body=int(getattr(builder, "body_count", 0)),
            shape=int(getattr(builder, "shape_count", 0)),
            joint=int(getattr(builder, "joint_count", 0)),
            articulation=int(getattr(builder, "articulation_count", 0)),
            equality_constraint=len(getattr(builder, "equality_constraint_type", ())),
            constraint_mimic=len(getattr(builder, "constraint_mimic_joint0", ())),
        )

    def replicated_over(
        self, prototype: _BuilderCounts, world_count: int
    ) -> _BuilderCounts:
        return _BuilderCounts(
            particle=self.particle + prototype.particle * world_count,
            body=self.body + prototype.body * world_count,
            shape=self.shape + prototype.shape * world_count,
            joint=self.joint + prototype.joint * world_count,
            articulation=self.articulation + prototype.articulation * world_count,
            equality_constraint=(
                self.equality_constraint + prototype.equality_constraint * world_count
            ),
            constraint_mimic=(
                self.constraint_mimic + prototype.constraint_mimic * world_count
            ),
        )


def _load_newton_dependencies() -> _NewtonDependencies:
    """延迟导入 Newton/Warp，使 PhysX-only 进程不注册任何 Newton 原生状态。"""

    import newton
    import warp as wp
    from newton._src.usd.schemas import (
        SchemaResolverMjc,
        SchemaResolverNewton,
        SchemaResolverPhysx,
    )

    return _NewtonDependencies(
        model_builder_type=newton.ModelBuilder,
        solver_mujoco_type=newton.solvers.SolverMuJoCo,
        schema_resolver_newton_type=SchemaResolverNewton,
        schema_resolver_mjc_type=SchemaResolverMjc,
        schema_resolver_physx_type=SchemaResolverPhysx,
        warp=wp,
    )


def _new_registered_builder(dependencies: _NewtonDependencies, *, up_axis: str):
    """创建 builder，并在 USD 解析前注册 MuJoCo 自定义列。

    注册顺序不可后移：equality/contact/actuator 等 solver metadata 在 ``add_usd`` 时写入；
    解析完成后再注册只会得到空列，最终 solver 看似创建成功却丢失原生 equality 映射。
    """

    builder = dependencies.model_builder_type(up_axis=up_axis)
    dependencies.solver_mujoco_type.register_custom_attributes(builder)
    return builder


def _new_schema_resolvers(dependencies: _NewtonDependencies) -> list[object]:
    """返回与 Isaac Sim Newton extension 一致的 schema resolver 优先级。

    Newton/MJC 特有 schema 先消费，PhysX resolver 作为兼容兜底。这里保持与 Isaac Newton
    extension 相同的顺序，避免因调用方自行拼装 resolver 而产生解析语义漂移。
    """

    return [
        dependencies.schema_resolver_newton_type(),
        dependencies.schema_resolver_mjc_type(),
        dependencies.schema_resolver_physx_type(),
    ]


def _normalized_prim_path(value: object, *, label: str) -> str:
    path = str(value).strip()
    if not path.startswith("/"):
        raise ValueError(f"{label} must be an absolute prim path: {value!r}")
    normalized = path.rstrip("/") or "/"
    if "//" in normalized:
        raise ValueError(f"{label} contains an empty path component: {value!r}")
    return normalized


def _path_is_at_or_below(path: str, root: str) -> bool:
    return root == "/" or path == root or path.startswith(f"{root}/")


def _parent_prim_path(path: str) -> str:
    parent, _, _ = path.rpartition("/")
    return parent or "/"


def _default_environment_root(
    prototype_root: str, destination_roots: tuple[str, ...]
) -> str:
    """选择不会吞掉共享场景实体的 global-parse ignore root。

    replicated scene 的 sibling env 通常都在 ``/World/envs`` 下，global parse 必须整体跳过它们，
    否则 env 会先作为 global dynamic body 解析一次、再复制一次。Mirror 的机器人却常
    与 ground 同处 ``/World``；此时若忽略整个 parent，会连共享地面也删掉，所以只忽略
    prototype 自身。
    """

    parents = {_parent_prim_path(path) for path in destination_roots}
    if len(destination_roots) > 1 and len(parents) == 1:
        parent = next(iter(parents))
        if _path_is_at_or_below(prototype_root, parent):
            return parent
    return prototype_root


def _validate_stage_prototype(stage: object, prototype_root: str) -> None:
    getter = getattr(stage, "GetPrimAtPath", None)
    if not callable(getter):
        return
    prim = getter(prototype_root)
    if prim is None:
        raise ValueError(f"prototype prim does not exist: {prototype_root}")
    is_valid = getattr(prim, "IsValid", None)
    if callable(is_valid) and not bool(is_valid()):
        raise ValueError(f"prototype prim does not exist: {prototype_root}")


def _unique_paths(paths: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def _rewrite_path_prefix(value: str, source_root: str, destination_root: str) -> str:
    if not value.startswith(source_root):
        return value
    suffix = value[len(source_root) :]
    # 只在 USD path component 边界重写；``/env_01`` 虽有 ``/env_0`` 文本前缀，却是
    # sibling 而非其 descendant，误改会让 topology label 与 stage path 永久错位。
    if suffix and not suffix.startswith("/"):
        return value
    return destination_root + suffix


def _rename_sequence(
    values: MutableSequence[object],
    worlds: Sequence[object],
    *,
    source_root: str,
    destination_by_world: Mapping[int, str],
) -> None:
    if len(values) != len(worlds):
        raise ValueError(
            f"label/world column length mismatch: {len(values)} vs {len(worlds)}"
        )
    for index, value in enumerate(values):
        if not isinstance(value, str):
            continue
        destination = destination_by_world.get(int(worlds[index]))
        if destination is None:
            continue
        values[index] = _rewrite_path_prefix(value, source_root, destination)


def _rename_mapping(
    values: MutableMapping[object, object],
    worlds: Mapping[object, object],
    *,
    source_root: str,
    destination_by_world: Mapping[int, str],
) -> None:
    if set(values) != set(worlds):
        raise ValueError("label/world mapping keys do not match")
    for key, value in tuple(values.items()):
        if not isinstance(value, str):
            continue
        destination = destination_by_world.get(int(worlds[key]))
        if destination is None:
            continue
        values[key] = _rewrite_path_prefix(value, source_root, destination)


def _rename_column(
    values: object,
    worlds: object,
    *,
    source_root: str,
    destination_by_world: Mapping[int, str],
) -> None:
    if isinstance(values, MutableMapping) and isinstance(worlds, Mapping):
        _rename_mapping(
            values,
            worlds,
            source_root=source_root,
            destination_by_world=destination_by_world,
        )
        return
    if isinstance(values, MutableSequence) and isinstance(worlds, Sequence):
        _rename_sequence(
            values,
            worlds,
            source_root=source_root,
            destination_by_world=destination_by_world,
        )
        return
    raise TypeError(
        "label/world columns must both be mutable mappings or mutable sequences"
    )


def rename_replicated_builder_labels(
    builder: object,
    *,
    source_root: str,
    destination_roots: Sequence[str],
) -> None:
    """按 Newton world 把 source-root label 改写到对应 destination root。

    除内建列外，也处理“同 frequency 且有 ``references='world'`` sibling”的未来 solver
    字符串列。不能只改 body/joint：equality 或 actuator label 若仍指向 prototype，会造成
    view 可绑定但 solver provenance 指向错误 world 的半一致拓扑。
    """

    normalized_source = _normalized_prim_path(source_root, label="source_root")
    destinations = tuple(
        _normalized_prim_path(path, label=f"destination_roots[{index}]")
        for index, path in enumerate(destination_roots)
    )
    destination_by_world = dict(enumerate(destinations))

    for entity_type in _BUILTIN_LABEL_TYPES:
        labels = getattr(builder, f"{entity_type}_label", None)
        if labels is None:
            labels = getattr(builder, f"{entity_type}_key", None)
        worlds = getattr(builder, f"{entity_type}_world", None)
        if labels is None or worlds is None:
            continue
        _rename_column(
            labels,
            worlds,
            source_root=normalized_source,
            destination_by_world=destination_by_world,
        )

    custom_attributes = getattr(builder, "custom_attributes", None)
    if not isinstance(custom_attributes, Mapping):
        return
    world_by_frequency: dict[object, object] = {}
    for attribute in custom_attributes.values():
        if getattr(attribute, "references", None) == "world":
            world_by_frequency[getattr(attribute, "frequency", None)] = attribute
    for attribute in custom_attributes.values():
        if getattr(attribute, "dtype", None) is not str:
            continue
        world_attribute = world_by_frequency.get(getattr(attribute, "frequency", None))
        if world_attribute is None:
            continue
        values = getattr(attribute, "values", None)
        worlds = getattr(world_attribute, "values", None)
        if values is None or worlds is None or len(values) == 0:
            continue
        _rename_column(
            values,
            worlds,
            source_root=normalized_source,
            destination_by_world=destination_by_world,
        )


def _world_counts(worlds: Sequence[object], world_count: int) -> tuple[int, ...]:
    counts = [0] * world_count
    for value in worlds:
        world = int(value)
        if world >= 0:
            if world >= world_count:
                raise RuntimeError(
                    f"entity references invalid Newton world {world}; count={world_count}"
                )
            counts[world] += 1
    return tuple(counts)


def _validate_replication_contract(
    builder: object,
    *,
    global_counts: _BuilderCounts,
    prototype_counts: _BuilderCounts,
    world_count: int,
) -> None:
    """在 finalize 前验证同构计数、world 归属和 global-scope 边界。

    builder 阶段的错误信息仍能指出具体列；等 finalize 后才发现混入 global dynamic entity，
    SolverMuJoCo 往往只会报告容量或 world layout 错误，定位代价高得多。
    """

    current_world = int(getattr(builder, "current_world", -1))
    if current_world != -1:
        raise RuntimeError(
            f"Newton builder returned with an open world context: {current_world}"
        )
    actual_world_count = int(getattr(builder, "world_count", 0))
    if actual_world_count != world_count:
        raise RuntimeError(
            f"Newton world count mismatch: expected {world_count}, "
            f"got {actual_world_count}"
        )

    actual_counts = _BuilderCounts.from_builder(builder)
    expected_counts = global_counts.replicated_over(prototype_counts, world_count)
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Newton replicated entity counts do not match the single prototype: "
            f"expected={expected_counts}, actual={actual_counts}"
        )

    per_world_columns = (
        ("particle", "particle_world", prototype_counts.particle),
        ("body", "body_world", prototype_counts.body),
        ("shape", "shape_world", prototype_counts.shape),
        ("joint", "joint_world", prototype_counts.joint),
        (
            "articulation",
            "articulation_world",
            prototype_counts.articulation,
        ),
        (
            "equality constraint",
            "equality_constraint_world",
            prototype_counts.equality_constraint,
        ),
        (
            "mimic constraint",
            "constraint_mimic_world",
            prototype_counts.constraint_mimic,
        ),
    )
    for label, attribute_name, expected_per_world in per_world_columns:
        worlds = getattr(builder, attribute_name, ())
        counts = _world_counts(worlds, world_count)
        if counts != (expected_per_world,) * world_count:
            raise RuntimeError(
                f"Newton {label} counts are not homogeneous: "
                f"expected {expected_per_world} per world, got {counts}"
            )

    # SolverMuJoCo separate-world 允许 world=-1 的共享静态 shape，却不允许 global dynamic
    # entity 或 constraint。前者是地面等跨 world collider，后者会把独立环境重新耦合起来。
    for label, attribute_name in (
        ("particle", "particle_world"),
        ("body", "body_world"),
        ("joint", "joint_world"),
        ("articulation", "articulation_world"),
        ("equality constraint", "equality_constraint_world"),
        ("mimic constraint", "constraint_mimic_world"),
    ):
        if any(int(world) == -1 for world in getattr(builder, attribute_name, ())):
            raise RuntimeError(
                f"global Newton scope contains a {label}; only static shapes are allowed"
            )


def build_replicated_newton_builder(
    stage: object,
    *,
    prototype_root: str,
    destination_roots: Sequence[str],
    world_transforms: Sequence[object] | None = None,
    source_world_transform: object | None = None,
    environment_root: str | None = None,
    global_ignore_paths: Sequence[str] = (),
    up_axis: str = "Z",
    load_visual_shapes: bool = True,
    skip_mesh_approximation: bool = True,
    joint_drive_gains_scaling: float = _USD_RADIAN_DRIVE_GAIN_SCALING,
) -> NewtonReplicationResult:
    """只解析一次已配置 USD prototype，并复制进多个 Newton world。

    Args:
        stage: Composed USD stage containing the configured source environment.
        prototype_root: Source root parsed exactly once, normally
            ``/World/envs/env_0`` for replicated environments.
        destination_roots: Final USD root corresponding to each sequential
            Newton world. A one-item sequence supports Mirror use.
        world_transforms: Absolute transform for each environment origin.  If
            omitted, all worlds use identity transforms.
        source_world_transform: Origin transform already represented by the
            prototype.  Defaults to the first world transform.
        environment_root: Stage subtree excluded from the global parse.  For
            multiple sibling destinations this defaults to their common parent;
            for one world it defaults to ``prototype_root`` so shared ``/World``
            siblings remain visible.
        global_ignore_paths: Additional paths excluded from the global parse.
        up_axis: Newton builder up axis.
        load_visual_shapes: Forwarded to the prototype/global USD parsers.
        skip_mesh_approximation: Preserve importer collision topology by
            default instead of rebuilding per-copy approximations.
        joint_drive_gains_scaling: Newton USD parser 的 drive 增益单位换算因子。默认使用
            ``pi / 180``，使项目已按弧度定义的 stiffness/damping 保持原值；显式传值仍会
            原样转发，供导入非项目资产时覆盖。

    Returns:
        未 finalize 的 :class:`NewtonReplicationResult`。任何 native equality/mimic 行都
        不会在这里被删除、禁用、转换或替换为 drive。
    """

    source_root = _normalized_prim_path(prototype_root, label="prototype_root")
    if source_root == "/":
        raise ValueError("prototype_root must identify one scene/environment")
    destinations = tuple(
        _normalized_prim_path(path, label=f"destination_roots[{index}]")
        for index, path in enumerate(destination_roots)
    )
    if not destinations:
        raise ValueError("destination_roots must contain at least one world")
    if len(set(destinations)) != len(destinations):
        raise ValueError("destination_roots must be unique")

    ignored_environment_root = (
        _default_environment_root(source_root, destinations)
        if environment_root is None
        else _normalized_prim_path(environment_root, label="environment_root")
    )
    if not _path_is_at_or_below(source_root, ignored_environment_root):
        raise ValueError(
            f"prototype_root {source_root!r} is outside environment_root "
            f"{ignored_environment_root!r}"
        )
    for destination in destinations:
        if not _path_is_at_or_below(destination, ignored_environment_root):
            raise ValueError(
                f"destination root {destination!r} is outside environment_root "
                f"{ignored_environment_root!r}"
            )
    _validate_stage_prototype(stage, source_root)

    dependencies = _load_newton_dependencies()
    wp = dependencies.warp
    if world_transforms is None:
        transforms = tuple(wp.transform_identity() for _ in destinations)
    else:
        transforms = tuple(world_transforms)
    if len(transforms) != len(destinations):
        raise ValueError(
            "world_transforms and destination_roots must have equal length: "
            f"{len(transforms)} vs {len(destinations)}"
        )
    source_transform = (
        transforms[0] if source_world_transform is None else source_world_transform
    )

    builder = _new_registered_builder(dependencies, up_axis=up_axis)
    prototype_builder = _new_registered_builder(dependencies, up_axis=up_axis)
    # fixed joint 不折叠，才能让 prototype/final model 的 joint label、equality endpoint
    # 与资产审计保持可追踪。视觉 shape 可关闭，但 collision topology 默认原样保留。
    common_parse_kwargs = {
        "only_load_enabled_rigid_bodies": True,
        "joint_drive_gains_scaling": float(joint_drive_gains_scaling),
        "verbose": False,
        "collapse_fixed_joints": False,
        "load_visual_shapes": bool(load_visual_shapes),
        "skip_mesh_approximation": bool(skip_mesh_approximation),
        "force_position_velocity_actuation": True,
    }
    ignored_paths = _unique_paths(
        [
            ignored_environment_root,
            *(
                _normalized_prim_path(path, label="global_ignore_paths item")
                for path in global_ignore_paths
            ),
        ]
    )
    global_stage_info = builder.add_usd(
        stage,
        ignore_paths=ignored_paths,
        schema_resolvers=_new_schema_resolvers(dependencies),
        **common_parse_kwargs,
    )
    global_counts = _BuilderCounts.from_builder(builder)

    # 这是唯一一次 prototype parse。逐 destination 重复 add_usd 不仅有 CPU 开销，还可能
    # 让 importer 每次生成不同的匿名 label/index。parse 与 add_builder 之间也不得编辑任一
    # native coupling table，确保 prototype 中解析出的全部 equality 逐 world 原样复制。
    prototype_stage_info = prototype_builder.add_usd(
        stage,
        root_path=source_root,
        schema_resolvers=_new_schema_resolvers(dependencies),
        **common_parse_kwargs,
    )
    prototype_counts = _BuilderCounts.from_builder(prototype_builder)

    inverse_source_transform = wp.transform_inverse(source_transform)
    for world_transform in transforms:
        # add_builder 接受相对 prototype 的变换，因此使用
        # T_destination * inverse(T_source)。每次复制必须落在 begin/end_world 配对中，
        # 否则实体会进入 world=-1 全局域并破坏环境隔离。
        builder.begin_world()
        try:
            relative_transform = wp.transform_multiply(
                world_transform, inverse_source_transform
            )
            builder.add_builder(prototype_builder, xform=relative_transform)
        except BaseException:
            # 即使复制异常也先闭合 world context；这样上层在记录失败 topology 时不会看到
            # 一个“仍处于 begin_world 内”的二次错误，并保留原始异常作为真正原因。
            builder.end_world()
            raise
        builder.end_world()

    rename_replicated_builder_labels(
        builder,
        source_root=source_root,
        destination_roots=destinations,
    )
    _validate_replication_contract(
        builder,
        global_counts=global_counts,
        prototype_counts=prototype_counts,
        world_count=len(destinations),
    )

    return NewtonReplicationResult(
        builder=builder,
        prototype_builder=prototype_builder,
        global_stage_info=global_stage_info,
        prototype_stage_info=prototype_stage_info,
        prototype_root=source_root,
        destination_roots=destinations,
        world_transforms=transforms,
        source_world_transform=source_transform,
        environment_root=ignored_environment_root,
    )


__all__ = [
    "NewtonReplicationResult",
    "build_replicated_newton_builder",
    "rename_replicated_builder_labels",
]
