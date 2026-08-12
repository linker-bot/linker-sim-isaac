"""Newton 原生主从关节约束审计和冷状态投影。

当前 Newton MuJoCo variant 把 LinkerHand 的 MJCF ``equality/joint`` 保留为
``EqType.JOINT``。求解阶段由 MuJoCo-Warp 的 ``mjEQ_JOINT`` 唯一执行主从关系；本模块
不创建 follower drive、actuator 或逐步 target writer。

只写受控 master 的 reset/command 冷路径会直接改写 generalized state，却没有提供 follower。
为了避免下一次 FK/碰撞读取到旧 follower，:class:`NewtonColdStateProjector` 可在这些显式
边界按同一多项式补齐 ``joint_q/joint_qd``。完整 snapshot/clone 已保存 solver 产出的 follower，
必须只做 FK 而不能再次冷投影；本 projector 也不会自行注册 callback 或进入常规 step loop。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np


NATIVE_JOINT_EQUALITY_EXECUTOR = "newton.eq_joint"
"""当前 Newton 主从关系的唯一动态执行者。"""

COLD_STATE_PROJECTION_SCOPE = "reset_restore_only"
"""状态投影只允许出现在 reset/restore 等冷路径。"""

_EQ_TYPE_JOINT = 2
_JOINT_TYPE_PRISMATIC = 0
_JOINT_TYPE_REVOLUTE = 1
_POLYCOEF_WIDTH = 5


class NewtonConstraintAuditError(RuntimeError):
    """Newton 主从约束结构或单一执行者合同不成立。"""


@dataclass(frozen=True)
class ExpectedMasterFollowerConstraint:
    """一条应出现在某个 Newton world 的精确主从关节关系。

    ``follower_joint_label`` 与 ``master_joint_label`` 必须是 builder/model 中的完整、唯一
    joint label，不做 basename、正则或后缀模糊匹配。``polycoef`` 使用 MuJoCo 常数项优先
    顺序，短于五项时在审计中补零。
    """

    world: int
    follower_joint_label: str
    master_joint_label: str
    polycoef: tuple[float, ...]
    constraint_label: str | None = None


@dataclass(frozen=True)
class MasterFollowerExecutorMetadata:
    """由 stage/replication 层提交的主从关系单一执行者证据。

    约束数组本身无法证明 USD follower drive 或 importer actuator 已停用，因此调用方必须
    把完成 stage 审计后的具体残留路径/label 传入。任何非空残留都会 fail closed。
    """

    dynamic_executor: str = NATIVE_JOINT_EQUALITY_EXECUTOR
    state_projection_scope: str = COLD_STATE_PROJECTION_SCOPE
    runtime_target_writer: str | None = None
    follower_drive_prim_paths: tuple[str, ...] = ()
    follower_actuator_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeMasterFollowerBinding:
    """已解析到 Newton equality、joint 和 generalized-state index 的关系。

    这里保存的是 finalized 全局数组下标，不是 articulation 内局部 DOF 序号。``q`` 与
    ``qd`` 在 FREE/BALL joint 上宽度不同，因而两套 index 必须分别审计，不能由其中一套
    假设偏移关系推导另一套。
    """

    equality_index: int
    world: int
    follower_joint_index: int
    master_joint_index: int
    follower_joint_label: str
    master_joint_label: str
    follower_q_index: int
    master_q_index: int
    follower_qd_index: int
    master_qd_index: int
    polycoef: tuple[float, float, float, float, float]
    constraint_label: str


@dataclass(frozen=True)
class NativeMasterFollowerAudit:
    """通过审计后可供 reset projector 和运行时 provenance 使用的不可变结果。"""

    representation: Literal["builder", "model"]
    world_count: int
    relations_per_world: int
    bindings: tuple[NativeMasterFollowerBinding, ...]
    executor_metadata: MasterFollowerExecutorMetadata

    @property
    def relation_count(self) -> int:
        """返回全部 world 的主从关系数。"""

        return len(self.bindings)

    def bindings_for_world(self, world: int) -> tuple[NativeMasterFollowerBinding, ...]:
        """按稳定 equality 顺序返回指定 world 的关系。"""

        return tuple(
            binding for binding in self.bindings if binding.world == int(world)
        )


def audit_native_master_follower_constraints(
    source: object,
    expectations: Sequence[ExpectedMasterFollowerConstraint],
    *,
    expected_world_count: int,
    executor_metadata: MasterFollowerExecutorMetadata,
    expected_relations_per_world: int = 10,
    representation: Literal["auto", "builder", "model"] = "auto",
    coefficient_atol: float = 1.0e-6,
) -> NativeMasterFollowerAudit:
    """严格审计 Newton builder 或 finalized model 的原生主从约束。

    当前 MuJoCo variant 的生产合同是：

    * 所有 Newton ``constraint_mimic`` 列均为空；
    * 每个 world 恰有调用方声明数量的 ``EqType.JOINT``（当前单臂为 5 条、双臂为 10 条）；
    * equality ``joint1`` 是 follower、``joint2`` 是 master；
    * 两个关节属于 equality 所在 world，且都是标量 revolute/prismatic joint；
    * label、index、五项 polycoef 与调用方的精确期望一一对应；
    * follower 没有第二条 equality，也没有 drive、actuator 或 runtime target writer。

    这是初始化/切换 gate，会把 Warp model 列同步到 host 做结构审计；不得在热 step 中调用。
    """

    world_count = _positive_int(expected_world_count, "expected_world_count")
    relations_per_world = _positive_int(
        expected_relations_per_world, "expected_relations_per_world"
    )
    if not isfinite(float(coefficient_atol)) or float(coefficient_atol) < 0.0:
        raise ValueError("coefficient_atol must be finite and non-negative")
    _validate_executor_metadata(executor_metadata)

    resolved_representation = _resolve_representation(source, representation)
    actual_world_count = getattr(source, "world_count", None)
    if actual_world_count is not None and int(actual_world_count) != world_count:
        raise NewtonConstraintAuditError(
            "Newton world count differs from the requested topology: "
            f"actual={int(actual_world_count)}, expected={world_count}"
        )

    expected = tuple(expectations)
    expected_total = world_count * relations_per_world
    if len(expected) != expected_total:
        raise NewtonConstraintAuditError(
            "Master/follower expectation count must equal world_count * relations_per_world: "
            f"actual={len(expected)}, expected={expected_total}"
        )
    expected_world_counts = Counter(item.world for item in expected)
    if expected_world_counts != Counter(
        {world: relations_per_world for world in range(world_count)}
    ):
        raise NewtonConstraintAuditError(
            "Master/follower expectations must contain the exact per-world relation count: "
            f"actual={dict(sorted(expected_world_counts.items()))}, "
            f"expected_per_world={relations_per_world}"
        )

    _require_zero_mimic_representation(source, resolved_representation)

    equality_types = _read_1d(source, "equality_constraint_type", dtype=np.int32)
    equality_joint1 = _read_1d(source, "equality_constraint_joint1", dtype=np.int32)
    equality_joint2 = _read_1d(source, "equality_constraint_joint2", dtype=np.int32)
    equality_enabled = _read_1d(source, "equality_constraint_enabled", dtype=np.bool_)
    equality_world = _read_1d(source, "equality_constraint_world", dtype=np.int32)
    # CONNECT/WELD 行可以合法地没有名字；必需的 JOINT 行会在下方按 EqType 过滤后，
    # 再严格检查其 label 非空且能精确映射。
    equality_labels = _read_labels(
        source, "equality_constraint_label", allow_empty=True
    )
    equality_polycoef = _read_polycoef(source, "equality_constraint_polycoef")
    equality_count = int(equality_types.size)
    _require_equal_lengths(
        "Newton equality columns",
        equality_count,
        {
            "joint1": equality_joint1.size,
            "joint2": equality_joint2.size,
            "enabled": equality_enabled.size,
            "world": equality_world.size,
            "label": len(equality_labels),
            "polycoef": equality_polycoef.shape[0],
        },
    )
    if resolved_representation == "model":
        declared_count = getattr(source, "equality_constraint_count", None)
        if declared_count is None or int(declared_count) != equality_count:
            raise NewtonConstraintAuditError(
                "Finalized Newton equality_constraint_count does not match its columns: "
                f"declared={declared_count!r}, columns={equality_count}"
            )

    joint_labels = _read_labels(source, "joint_label")
    joint_world = _read_1d(source, "joint_world", dtype=np.int32)
    joint_type = _read_1d(source, "joint_type", dtype=np.int32)
    joint_q_start = _read_1d(source, "joint_q_start", dtype=np.int32)
    joint_qd_start = _read_1d(source, "joint_qd_start", dtype=np.int32)
    joint_count = len(joint_labels)
    _require_equal_lengths(
        "Newton joint columns",
        joint_count,
        {
            "world": joint_world.size,
            "type": joint_type.size,
        },
    )
    if joint_q_start.size not in {joint_count, joint_count + 1}:
        raise NewtonConstraintAuditError(
            "Newton joint_q_start must have builder or model length: "
            f"actual={joint_q_start.size}, joint_count={joint_count}"
        )
    if joint_qd_start.size not in {joint_count, joint_count + 1}:
        raise NewtonConstraintAuditError(
            "Newton joint_qd_start must have builder or model length: "
            f"actual={joint_qd_start.size}, joint_count={joint_count}"
        )
    joint_indices_by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(joint_labels):
        joint_indices_by_label[label].append(index)

    expected_by_key: dict[tuple[int, int, int], ExpectedMasterFollowerConstraint] = {}
    follower_keys: set[tuple[int, int]] = set()
    # 期望关系用 (world, follower, master) 建精确键。只按 joint basename 匹配会把 replicated
    # replica 误认为同一关节；只按 equality 顺序匹配又会把 importer 的重排误当成语义变化。
    for item in expected:
        world = int(item.world)
        follower_label = _nonempty_label(
            item.follower_joint_label, "follower_joint_label"
        )
        master_label = _nonempty_label(item.master_joint_label, "master_joint_label")
        if follower_label == master_label:
            raise NewtonConstraintAuditError(
                f"Master/follower relation cannot reference one joint twice: {follower_label!r}"
            )
        resolved_joints: dict[str, int] = {}
        for role, label in (
            ("follower", follower_label),
            ("master", master_label),
        ):
            matches = joint_indices_by_label.get(label, ())
            if not matches:
                raise NewtonConstraintAuditError(
                    "Expected master/follower joint label is absent from Newton "
                    f"topology: {label!r}"
                )
            if len(matches) != 1:
                raise NewtonConstraintAuditError(
                    "Expected master/follower joint label must resolve exactly once: "
                    f"role={role}, label={label!r}, matches={list(matches)!r}"
                )
            resolved_joints[role] = int(matches[0])
        follower_joint = resolved_joints["follower"]
        master_joint = resolved_joints["master"]
        for role, joint_index in (
            ("follower", follower_joint),
            ("master", master_joint),
        ):
            if int(joint_world[joint_index]) != world:
                raise NewtonConstraintAuditError(
                    f"Expected {role} joint belongs to the wrong Newton world: "
                    f"label={joint_labels[joint_index]!r}, joint_world={int(joint_world[joint_index])}, "
                    f"expected_world={world}"
                )
        follower_key = (world, follower_joint)
        if follower_key in follower_keys:
            raise NewtonConstraintAuditError(
                "A follower joint may have only one native equality executor: "
                f"world={world}, joint={follower_label!r}"
            )
        follower_keys.add(follower_key)
        key = (world, follower_joint, master_joint)
        if key in expected_by_key:
            raise NewtonConstraintAuditError(
                "Duplicate master/follower expectation: "
                f"world={world}, follower={follower_label!r}, master={master_label!r}"
            )
        _normalize_polycoef(item.polycoef)
        expected_by_key[key] = item

    joint_equality_indices = np.flatnonzero(equality_types == _EQ_TYPE_JOINT)
    if joint_equality_indices.size != expected_total:
        raise NewtonConstraintAuditError(
            "Newton native joint-equality count differs from the required topology: "
            f"actual={joint_equality_indices.size}, expected={expected_total}"
        )
    actual_world_counts = Counter(
        int(equality_world[index]) for index in joint_equality_indices
    )
    if actual_world_counts != Counter(
        {world: relations_per_world for world in range(world_count)}
    ):
        raise NewtonConstraintAuditError(
            "Newton EqType.JOINT rows must be distributed evenly across worlds: "
            f"actual={dict(sorted(actual_world_counts.items()))}, "
            f"expected_per_world={relations_per_world}"
        )

    total_q = _state_width(source, "joint_coord_count", "joint_q", joint_q_start)
    total_qd = _state_width(source, "joint_dof_count", "joint_qd", joint_qd_start)
    bindings: list[NativeMasterFollowerBinding] = []
    matched_keys: set[tuple[int, int, int]] = set()
    for equality_index_value in joint_equality_indices:
        equality_index = int(equality_index_value)
        world = int(equality_world[equality_index])
        follower_joint = int(equality_joint1[equality_index])
        master_joint = int(equality_joint2[equality_index])
        for role, joint_index in (
            ("follower", follower_joint),
            ("master", master_joint),
        ):
            if joint_index < 0 or joint_index >= joint_count:
                raise NewtonConstraintAuditError(
                    f"Newton EqType.JOINT {role} index is out of range: "
                    f"equality={equality_index}, joint={joint_index}, joint_count={joint_count}"
                )
            if int(joint_world[joint_index]) != world:
                raise NewtonConstraintAuditError(
                    f"Newton EqType.JOINT {role} is assigned to a different world: "
                    f"equality={equality_index}, equality_world={world}, "
                    f"joint={joint_labels[joint_index]!r}, joint_world={int(joint_world[joint_index])}"
                )
            if int(joint_type[joint_index]) not in {
                _JOINT_TYPE_PRISMATIC,
                _JOINT_TYPE_REVOLUTE,
            }:
                raise NewtonConstraintAuditError(
                    f"Newton EqType.JOINT requires scalar prismatic/revolute joints: "
                    f"equality={equality_index}, {role}={joint_labels[joint_index]!r}, "
                    f"joint_type={int(joint_type[joint_index])}"
                )
        if follower_joint == master_joint:
            raise NewtonConstraintAuditError(
                f"Newton EqType.JOINT {equality_index} constrains a joint to itself"
            )
        if not bool(equality_enabled[equality_index]):
            raise NewtonConstraintAuditError(
                f"Required Newton EqType.JOINT row is disabled: equality={equality_index}"
            )

        key = (world, follower_joint, master_joint)
        item = expected_by_key.get(key)
        if item is None:
            raise NewtonConstraintAuditError(
                "Unexpected Newton EqType.JOINT row: "
                f"equality={equality_index}, world={world}, "
                f"follower={joint_labels[follower_joint]!r}, master={joint_labels[master_joint]!r}"
            )
        if key in matched_keys:
            raise NewtonConstraintAuditError(
                "One master/follower relation has more than one EqType.JOINT executor: "
                f"world={world}, follower={joint_labels[follower_joint]!r}, "
                f"master={joint_labels[master_joint]!r}"
            )
        matched_keys.add(key)
        expected_polycoef = _normalize_polycoef(item.polycoef)
        actual_polycoef = tuple(
            float(value) for value in equality_polycoef[equality_index]
        )
        if not np.allclose(
            actual_polycoef,
            expected_polycoef,
            rtol=0.0,
            atol=float(coefficient_atol),
        ):
            raise NewtonConstraintAuditError(
                "Newton EqType.JOINT polycoef differs from the asset relation: "
                f"equality={equality_index}, actual={actual_polycoef}, "
                f"expected={expected_polycoef}"
            )
        constraint_label = equality_labels[equality_index]
        if not constraint_label:
            raise NewtonConstraintAuditError(
                "Required Newton EqType.JOINT row must have a non-empty exact label: "
                f"equality={equality_index}, world={world}, "
                f"follower={joint_labels[follower_joint]!r}, "
                f"master={joint_labels[master_joint]!r}"
            )
        if (
            item.constraint_label is not None
            and constraint_label != item.constraint_label
        ):
            raise NewtonConstraintAuditError(
                "Newton EqType.JOINT label differs from the exact expectation: "
                f"equality={equality_index}, actual={constraint_label!r}, "
                f"expected={item.constraint_label!r}"
            )

        # JOINT equality 的 joint1/joint2 语义分别是 follower/master。两者在当前资产中
        # 必须各自拥有一个标量 q 与 qd；若 importer 改成多坐标 joint，继续套用这个
        # 多项式会静默写错 generalized state，所以初始化阶段直接拒绝。
        follower_q = _scalar_state_index(
            joint_q_start,
            joint_index=follower_joint,
            total_width=total_q,
            label="joint_q",
        )
        master_q = _scalar_state_index(
            joint_q_start,
            joint_index=master_joint,
            total_width=total_q,
            label="joint_q",
        )
        follower_qd = _scalar_state_index(
            joint_qd_start,
            joint_index=follower_joint,
            total_width=total_qd,
            label="joint_qd",
        )
        master_qd = _scalar_state_index(
            joint_qd_start,
            joint_index=master_joint,
            total_width=total_qd,
            label="joint_qd",
        )
        bindings.append(
            NativeMasterFollowerBinding(
                equality_index=equality_index,
                world=world,
                follower_joint_index=follower_joint,
                master_joint_index=master_joint,
                follower_joint_label=joint_labels[follower_joint],
                master_joint_label=joint_labels[master_joint],
                follower_q_index=follower_q,
                master_q_index=master_q,
                follower_qd_index=follower_qd,
                master_qd_index=master_qd,
                polycoef=expected_polycoef,
                constraint_label=constraint_label,
            )
        )

    missing = set(expected_by_key).difference(matched_keys)
    if missing:
        descriptions = [
            (
                world,
                joint_labels[follower],
                joint_labels[master],
            )
            for world, follower, master in sorted(missing)
        ]
        raise NewtonConstraintAuditError(
            f"Expected Newton master/follower relations are missing: {descriptions!r}"
        )

    ordered = tuple(
        sorted(bindings, key=lambda item: (item.world, item.equality_index))
    )
    _topological_binding_levels(ordered)
    return NativeMasterFollowerAudit(
        representation=resolved_representation,
        world_count=world_count,
        relations_per_world=relations_per_world,
        bindings=ordered,
        executor_metadata=executor_metadata,
    )


class NewtonColdStateProjector:
    """把审计通过的主从关系投影到 selected-world Warp state。

    构造时把 relation index/polycoef 固定在目标 device；每次调用只读取调用方提供的固定长度
    ``wp.int32`` world mask，并在相同 stream 上原地更新 ``joint_q/joint_qd``。依赖链按拓扑层
    分多次 launch，确保上一层 follower 成为下一层 master 时读到的是新状态。

    本类不持有 solver、不注册 callback、不写 control target。调用方只能在缺少 follower 的
    command/reset 冷路径使用，并在同一 stream 上随后执行 masked FK/solver reset；完整
    q/qd restore 不得调用本类，否则会破坏 snapshot 的逐位状态语义。
    """

    def __init__(
        self,
        audit: NativeMasterFollowerAudit,
        *,
        device: object,
        stream: object | None = None,
    ) -> None:
        _validate_executor_metadata(audit.executor_metadata)
        if (
            audit.executor_metadata.state_projection_scope
            != COLD_STATE_PROJECTION_SCOPE
        ):
            raise NewtonConstraintAuditError(
                "Newton state projector requires reset/restore-only metadata"
            )
        import warp as wp

        self.audit = audit
        self.world_count = audit.world_count
        self.device = wp.get_device(device)
        self.stream = stream
        with _warp_stream_scope(stream):
            self._levels: tuple[_WarpProjectionLevel, ...] = tuple(
                _make_warp_projection_level(level, device=self.device)
                for level in _topological_binding_levels(audit.bindings)
            )

    @property
    def relation_count(self) -> int:
        """返回本 projector 固定绑定的关系总数。"""

        return self.audit.relation_count

    def project(
        self,
        *,
        joint_q: object,
        joint_qd: object,
        selected_world_mask: object,
        stream: object | None = None,
    ) -> None:
        """在当前 Warp stream 上投影 selected world 的 follower ``q/qd``。

        ``selected_world_mask`` 必须是 shape ``(world_count,)``、dtype ``wp.int32`` 的 Warp
        数组，非零表示选中。固定 shape 使 kernel launch 拓扑不随 selected world 数量变化，
        也避免把 env row、articulation row 与 Newton world id 混为一谈。方法异步返回，不做
        device synchronize；后续 FK/solver reset 必须复用同一 stream 才能继承写后读顺序。
        """

        import warp as wp

        _require_warp_vector(
            selected_world_mask,
            name="selected_world_mask",
            dtype=wp.int32,
            minimum_size=self.world_count,
            exact_size=self.world_count,
            device=self.device,
        )
        max_q = max(binding.follower_q_index for binding in self.audit.bindings)
        max_q = max(max_q, *(binding.master_q_index for binding in self.audit.bindings))
        max_qd = max(binding.follower_qd_index for binding in self.audit.bindings)
        max_qd = max(
            max_qd, *(binding.master_qd_index for binding in self.audit.bindings)
        )
        _require_warp_vector(
            joint_q,
            name="joint_q",
            dtype=wp.float32,
            minimum_size=max_q + 1,
            device=self.device,
        )
        _require_warp_vector(
            joint_qd,
            name="joint_qd",
            dtype=wp.float32,
            minimum_size=max_qd + 1,
            device=self.device,
        )
        launch_stream = self.stream if stream is None else stream
        with _warp_stream_scope(launch_stream):
            kernel = _cold_projection_kernel()
            # 同一层关系之间没有 follower→master 依赖，可并行 launch；层与层在同一 stream
            # 上顺序入队，因此后一层读取到前一层刚计算出的 follower 值，无需 CPU barrier。
            for level in self._levels:
                wp.launch(
                    kernel,
                    dim=level.count,
                    inputs=[
                        level.world,
                        level.follower_q,
                        level.master_q,
                        level.follower_qd,
                        level.master_qd,
                        level.polycoef,
                        selected_world_mask,
                    ],
                    outputs=[joint_q, joint_qd],
                    device=self.device,
                    stream=launch_stream,
                )


class NewtonDeviceWorldMasks:
    """在 owner device 上复用 world 与 articulation 选择掩码。

    ``selected_worlds`` 来自已经完成冷路径拓扑绑定的 host 元数据，但掩码本身始终驻留在
    Newton device。局部选择通过只接收标量 world id 的 Warp kernel 原地写入预分配数组，
    再由第二个 kernel 将 ``articulation_world`` 映射为 FK 所需的 bool mask；运行时不会把
    topology 下载为 NumPy，也不会为每次 reset/restore 动态上传 host 数组。

    同一个实例只能由一条有序 owner stream 使用。manager 正是按此合同依次排入 mask 写入、
    equality projection 与 FK，因此即使下一次选择复用同一块内存，也不会覆盖前一批 kernel
    尚未消费的值。
    """

    def __init__(
        self,
        *,
        world_count: int,
        articulation_world: object,
        device: object,
        stream: object | None = None,
    ) -> None:
        import warp as wp

        self.world_count = _positive_int(world_count, "world_count")
        self.device = wp.get_device(device)
        self.stream = stream
        _require_warp_vector(
            articulation_world,
            name="articulation_world",
            dtype=wp.int32,
            minimum_size=0,
            device=self.device,
        )
        self._articulation_world = articulation_world
        self.articulation_count = int(articulation_world.shape[0])
        with _warp_stream_scope(stream):
            self._all_worlds = wp.ones(
                self.world_count,
                dtype=wp.int32,
                device=self.device,
            )
            self._selected_worlds = wp.zeros(
                self.world_count,
                dtype=wp.int32,
                device=self.device,
            )
            self._selected_articulations = wp.empty(
                self.articulation_count,
                dtype=wp.bool,
                device=self.device,
            )

    def world_mask(
        self,
        selected_worlds: Sequence[int],
        *,
        masked_rows: Sequence[tuple[object, object]] = (),
    ) -> object:
        """合并 host world id 与 device row mask，返回固定 shape 的 device mask。

        ``masked_rows`` 的每项是 ``(row_world, row_enabled)``，两者均为同长度 Warp array。
        前者在 view 注册冷路径一次性上传，后者通常是 SAME_STEP 的 CUDA bool reset mask。
        """

        import warp as wp

        worlds = _validated_selected_worlds(
            selected_worlds,
            world_count=self.world_count,
        )
        if len(worlds) == self.world_count:
            return self._all_worlds
        _write_selected_world_mask(
            self._selected_worlds,
            worlds,
            world_count=self.world_count,
            device=self.device,
            stream=self.stream,
        )
        with _warp_stream_scope(self.stream):
            kernel = _masked_rows_to_worlds_kernel()
            for row_world, row_enabled in masked_rows:
                _require_warp_vector(
                    row_world,
                    name="row_world",
                    dtype=wp.int32,
                    minimum_size=0,
                    device=self.device,
                )
                _require_warp_vector(
                    row_enabled,
                    name="row_enabled",
                    dtype=wp.bool,
                    minimum_size=int(row_world.shape[0]),
                    exact_size=int(row_world.shape[0]),
                    device=self.device,
                )
                wp.launch(
                    kernel,
                    dim=int(row_world.shape[0]),
                    inputs=[row_world, row_enabled],
                    outputs=[self._selected_worlds],
                    device=self.device,
                    stream=self.stream,
                )
        return self._selected_worlds

    def articulation_mask(self, selected_world_mask: object) -> object:
        """把 device world mask 映射成 ``newton.eval_fk`` 的 articulation mask。"""

        import warp as wp

        _require_warp_vector(
            selected_world_mask,
            name="selected_world_mask",
            dtype=wp.int32,
            minimum_size=self.world_count,
            exact_size=self.world_count,
            device=self.device,
        )
        with _warp_stream_scope(self.stream):
            wp.launch(
                _articulation_world_mask_kernel(),
                dim=self.articulation_count,
                inputs=[self._articulation_world, selected_world_mask],
                outputs=[self._selected_articulations],
                device=self.device,
                stream=self.stream,
            )
        return self._selected_articulations


def make_selected_world_mask(
    selected_worlds: Sequence[int],
    *,
    world_count: int,
    device: object,
    stream: object | None = None,
) -> object:
    """为 reset/restore 冷路径创建固定长度 ``wp.int32`` world mask。

    mask 的下标域是 Newton ``world``，不是 view 的 batch row。即使只恢复一个环境也保留
    ``world_count`` 长度，从而让每条 equality 通过自身 ``relation_world`` 做 O(1) 判定。
    """

    import warp as wp

    count = _positive_int(world_count, "world_count")
    selected = _validated_selected_worlds(selected_worlds, world_count=count)
    with _warp_stream_scope(stream):
        mask = wp.zeros(count, dtype=wp.int32, device=device)
    _write_selected_world_mask(
        mask,
        selected,
        world_count=count,
        device=device,
        stream=stream,
    )
    return mask


def _warp_stream_scope(stream: object | None) -> object:
    """把 owner stream 设为 current，但不隐式建立跨流 fence。

    projector 的数组创建、投影与后续 FK 都由 manager 排在同一流；这里若启用 sync_enter
    反而会把 default stream 不相关工作带入关键路径，破坏 selected reset 的异步语义。
    """

    if stream is None:
        return nullcontext()
    import warp as wp

    return wp.ScopedStream(stream, sync_enter=False, sync_exit=False)


@dataclass(frozen=True)
class _WarpProjectionLevel:
    count: int
    world: object
    follower_q: object
    master_q: object
    follower_qd: object
    master_qd: object
    polycoef: object


_projection_kernel: Any | None = None
_mark_selected_world_kernel_instance: Any | None = None
_articulation_world_mask_kernel_instance: Any | None = None
_masked_rows_to_worlds_kernel_instance: Any | None = None


def _write_selected_world_mask(
    mask: object,
    selected_worlds: Sequence[int],
    *,
    world_count: int,
    device: object,
    stream: object | None,
) -> None:
    """只用 device memset 与标量 kernel 参数更新一块 persistent world mask。"""

    import warp as wp

    _require_warp_vector(
        mask,
        name="selected_world_mask",
        dtype=wp.int32,
        minimum_size=world_count,
        exact_size=world_count,
        device=wp.get_device(device),
    )
    worlds = _validated_selected_worlds(selected_worlds, world_count=world_count)
    with _warp_stream_scope(stream):
        mask.zero_()
        kernel = _mark_selected_world_kernel()
        for world in worlds:
            wp.launch(
                kernel,
                dim=1,
                inputs=[world],
                outputs=[mask],
                device=device,
                stream=stream,
            )


def _mark_selected_world_kernel() -> object:
    global _mark_selected_world_kernel_instance
    if _mark_selected_world_kernel_instance is not None:
        return _mark_selected_world_kernel_instance
    import warp as wp

    @wp.kernel
    def mark_selected_world(
        world: wp.int32,
        selected_world_mask: wp.array(dtype=wp.int32),
    ):
        # ``wp`` 必须出现在函数体内，Warp 在 ``from __future__ import annotations`` 下才会
        # 把局部 import 捕获进 nested-kernel 的注解求值环境。
        if wp.tid() == 0:
            selected_world_mask[world] = 1

    _mark_selected_world_kernel_instance = mark_selected_world
    return _mark_selected_world_kernel_instance


def _articulation_world_mask_kernel() -> object:
    global _articulation_world_mask_kernel_instance
    if _articulation_world_mask_kernel_instance is not None:
        return _articulation_world_mask_kernel_instance
    import warp as wp

    @wp.kernel
    def map_articulation_worlds(
        articulation_world: wp.array(dtype=wp.int32),
        selected_world_mask: wp.array(dtype=wp.int32),
        selected_articulation_mask: wp.array(dtype=wp.bool),
    ):
        articulation = wp.tid()
        world = articulation_world[articulation]
        selected_articulation_mask[articulation] = selected_world_mask[world] != 0

    _articulation_world_mask_kernel_instance = map_articulation_worlds
    return _articulation_world_mask_kernel_instance


def _masked_rows_to_worlds_kernel() -> object:
    global _masked_rows_to_worlds_kernel_instance
    if _masked_rows_to_worlds_kernel_instance is not None:
        return _masked_rows_to_worlds_kernel_instance
    import warp as wp

    @wp.kernel
    def include_enabled_row_worlds(
        row_world: wp.array(dtype=wp.int32),
        row_enabled: wp.array(dtype=wp.bool),
        selected_world_mask: wp.array(dtype=wp.int32),
    ):
        row = wp.tid()
        if row_enabled[row]:
            selected_world_mask[row_world[row]] = 1

    _masked_rows_to_worlds_kernel_instance = include_enabled_row_worlds
    return _masked_rows_to_worlds_kernel_instance


def _cold_projection_kernel() -> object:
    global _projection_kernel
    if _projection_kernel is not None:
        return _projection_kernel
    import warp as wp

    @wp.kernel
    def project_native_joint_equalities(
        relation_world: wp.array(dtype=wp.int32),
        follower_q_index: wp.array(dtype=wp.int32),
        master_q_index: wp.array(dtype=wp.int32),
        follower_qd_index: wp.array(dtype=wp.int32),
        master_qd_index: wp.array(dtype=wp.int32),
        polycoef: wp.array(dtype=wp.float32),
        selected_world_mask: wp.array(dtype=wp.int32),
        joint_q: wp.array(dtype=wp.float32),
        joint_qd: wp.array(dtype=wp.float32),
    ):
        relation = wp.tid()
        world = relation_world[relation]
        if selected_world_mask[world] != 0:
            # MuJoCo joint equality 使用常数项优先的四次多项式：
            # q_f = c0 + c1*q_m + ... + c4*q_m^4。
            # 速度不是再次套同一多项式，而是按链式法则
            # qd_f = (dq_f/dq_m) * qd_m；Horner 形式减少乘法并与 solver 语义一致。
            base = relation * _POLYCOEF_WIDTH
            master_position = joint_q[master_q_index[relation]]
            c0 = polycoef[base]
            c1 = polycoef[base + 1]
            c2 = polycoef[base + 2]
            c3 = polycoef[base + 3]
            c4 = polycoef[base + 4]
            follower_position = (
                ((c4 * master_position + c3) * master_position + c2) * master_position
                + c1
            ) * master_position + c0
            derivative = (
                (wp.float32(4.0) * c4 * master_position + wp.float32(3.0) * c3)
                * master_position
                + wp.float32(2.0) * c2
            ) * master_position + c1
            joint_q[follower_q_index[relation]] = follower_position
            joint_qd[follower_qd_index[relation]] = (
                derivative * joint_qd[master_qd_index[relation]]
            )

    _projection_kernel = project_native_joint_equalities
    return _projection_kernel


def _make_warp_projection_level(
    bindings: Sequence[NativeMasterFollowerBinding], *, device: object
) -> _WarpProjectionLevel:
    import warp as wp

    items = tuple(bindings)
    polycoef = np.asarray(
        [coefficient for item in items for coefficient in item.polycoef],
        dtype=np.float32,
    )
    return _WarpProjectionLevel(
        count=len(items),
        world=wp.array([item.world for item in items], dtype=wp.int32, device=device),
        follower_q=wp.array(
            [item.follower_q_index for item in items], dtype=wp.int32, device=device
        ),
        master_q=wp.array(
            [item.master_q_index for item in items], dtype=wp.int32, device=device
        ),
        follower_qd=wp.array(
            [item.follower_qd_index for item in items], dtype=wp.int32, device=device
        ),
        master_qd=wp.array(
            [item.master_qd_index for item in items], dtype=wp.int32, device=device
        ),
        polycoef=wp.array(polycoef, dtype=wp.float32, device=device),
    )


def _topological_binding_levels(
    bindings: Sequence[NativeMasterFollowerBinding],
) -> tuple[tuple[NativeMasterFollowerBinding, ...], ...]:
    """把主从依赖 DAG 分层，供同层并行、跨层顺序投影。

    例如 A→B、B→C 时，B 既是第一条关系的 follower 又是第二条的 master；若两条关系在
    一个 kernel 中并行，C 可能读到旧 B。按深度拆 launch 可利用同一 stream 的顺序保证，
    同时显式拒绝会令投影无定义的环。
    """

    by_follower = {binding.follower_joint_index: binding for binding in bindings}
    if len(by_follower) != len(bindings):
        raise NewtonConstraintAuditError(
            "A Newton follower joint has more than one equality executor"
        )
    depths: dict[int, int] = {}
    visiting: set[int] = set()

    def depth(binding: NativeMasterFollowerBinding) -> int:
        follower = binding.follower_joint_index
        if follower in depths:
            return depths[follower]
        if follower in visiting:
            raise NewtonConstraintAuditError(
                "Newton master/follower equality graph contains a cycle: "
                f"joint={binding.follower_joint_label!r}"
            )
        visiting.add(follower)
        parent = by_follower.get(binding.master_joint_index)
        result = 0 if parent is None else depth(parent) + 1
        visiting.remove(follower)
        depths[follower] = result
        return result

    grouped: dict[int, list[NativeMasterFollowerBinding]] = defaultdict(list)
    for item in bindings:
        grouped[depth(item)].append(item)
    return tuple(
        tuple(
            sorted(grouped[level], key=lambda item: (item.world, item.equality_index))
        )
        for level in sorted(grouped)
    )


def _validate_executor_metadata(metadata: MasterFollowerExecutorMetadata) -> None:
    # “期望数量的 equality 存在”并不足以证明语义正确；如果 follower drive、actuator 或逐步
    # target writer 仍存在，它们会与 EqType.JOINT 同时争夺同一 DOF。单一执行者合同因此
    # 将这些 provenance 也纳入初始化 gate，而不是寄希望于运行时调参抵消。
    if not isinstance(metadata, MasterFollowerExecutorMetadata):
        raise TypeError("executor_metadata must be MasterFollowerExecutorMetadata")
    if metadata.dynamic_executor != NATIVE_JOINT_EQUALITY_EXECUTOR:
        raise NewtonConstraintAuditError(
            "Newton master/follower dynamic executor must be native EqType.JOINT: "
            f"actual={metadata.dynamic_executor!r}"
        )
    if metadata.state_projection_scope != COLD_STATE_PROJECTION_SCOPE:
        raise NewtonConstraintAuditError(
            "Newton master/follower state projection must be reset/restore-only: "
            f"actual={metadata.state_projection_scope!r}"
        )
    if metadata.runtime_target_writer is not None:
        raise NewtonConstraintAuditError(
            "Newton native master/follower constraints cannot have a runtime target writer: "
            f"writer={metadata.runtime_target_writer!r}"
        )
    if metadata.follower_drive_prim_paths:
        raise NewtonConstraintAuditError(
            "Newton native master/follower constraints cannot retain follower drives: "
            f"paths={list(metadata.follower_drive_prim_paths)!r}"
        )
    if metadata.follower_actuator_labels:
        raise NewtonConstraintAuditError(
            "Newton native master/follower constraints cannot retain follower actuators: "
            f"labels={list(metadata.follower_actuator_labels)!r}"
        )


def _resolve_representation(
    source: object, representation: Literal["auto", "builder", "model"]
) -> Literal["builder", "model"]:
    if representation not in {"auto", "builder", "model"}:
        raise ValueError(
            f"unsupported Newton constraint representation: {representation!r}"
        )
    if representation != "auto":
        return representation
    equality_types = getattr(source, "equality_constraint_type", None)
    return "builder" if isinstance(equality_types, list) else "model"


def _require_zero_mimic_representation(
    source: object, representation: Literal["builder", "model"]
) -> None:
    columns = (
        "constraint_mimic_joint0",
        "constraint_mimic_joint1",
        "constraint_mimic_coef0",
        "constraint_mimic_coef1",
        "constraint_mimic_enabled",
        "constraint_mimic_label",
        "constraint_mimic_world",
    )
    missing = [name for name in columns if not hasattr(source, name)]
    if missing:
        raise NewtonConstraintAuditError(
            "Newton source does not expose the constraint_mimic columns needed "
            f"to prove a single EqType.JOINT representation: missing={missing!r}"
        )
    nonempty = {
        name: _column_length(getattr(source, name, None))
        for name in columns
        if _column_length(getattr(source, name, None)) != 0
    }
    declared = getattr(source, "constraint_mimic_count", None)
    if representation == "model" and declared is None:
        raise NewtonConstraintAuditError(
            "Finalized Newton model does not expose constraint_mimic_count"
        )
    if declared is not None and int(declared) != 0:
        nonempty["constraint_mimic_count"] = int(declared)
    if nonempty:
        raise NewtonConstraintAuditError(
            "Current Newton MuJoCo variant requires zero constraint_mimic rows; "
            "EqType.JOINT must be the only native executor: "
            f"representation={representation}, nonempty={nonempty}"
        )


def _read_1d(source: object, name: str, *, dtype: object) -> np.ndarray:
    value = getattr(source, name, None)
    if value is None:
        raise NewtonConstraintAuditError(
            f"Newton source does not expose required column {name!r}"
        )
    array = _to_host_array(value, dtype=dtype)
    if array.ndim != 1:
        raise NewtonConstraintAuditError(
            f"Newton column {name!r} must be one-dimensional, got {array.shape}"
        )
    return array


def _read_polycoef(source: object, name: str) -> np.ndarray:
    value = getattr(source, name, None)
    if value is None:
        raise NewtonConstraintAuditError(
            f"Newton source does not expose required column {name!r}"
        )
    array = _to_host_array(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, _POLYCOEF_WIDTH), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != _POLYCOEF_WIDTH:
        raise NewtonConstraintAuditError(
            "Newton equality_constraint_polycoef must have shape (count, 5), "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise NewtonConstraintAuditError(
            "Newton equality polycoef must contain finite values"
        )
    return array


def _read_labels(
    source: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = getattr(source, name, None)
    if value is None:
        raise NewtonConstraintAuditError(
            f"Newton source does not expose required labels {name!r}"
        )
    try:
        result = tuple(str(item) for item in value)
    except TypeError as exc:
        raise NewtonConstraintAuditError(
            f"Newton labels {name!r} must be iterable"
        ) from exc
    if not allow_empty and any(not item for item in result):
        raise NewtonConstraintAuditError(
            f"Newton labels {name!r} cannot contain empty values"
        )
    return result


def _to_host_array(value: object, *, dtype: object) -> np.ndarray:
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value, dtype=dtype)


def _column_length(value: object | None) -> int:
    if value is None:
        return 0
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _require_equal_lengths(
    label: str, expected: int, actual_by_name: Mapping[str, int]
) -> None:
    mismatched = {
        name: int(actual)
        for name, actual in actual_by_name.items()
        if int(actual) != expected
    }
    if mismatched:
        raise NewtonConstraintAuditError(
            f"{label} have inconsistent lengths: expected={expected}, actual={mismatched}"
        )


def _normalize_polycoef(
    values: Sequence[float],
) -> tuple[float, float, float, float, float]:
    coefficients = tuple(float(value) for value in values)
    if len(coefficients) > _POLYCOEF_WIDTH:
        raise NewtonConstraintAuditError(
            f"Newton joint equality accepts at most five coefficients, got {len(coefficients)}"
        )
    if not all(isfinite(value) for value in coefficients):
        raise NewtonConstraintAuditError(
            f"Newton joint equality coefficients must be finite: {coefficients!r}"
        )
    padded = coefficients + (0.0,) * (_POLYCOEF_WIDTH - len(coefficients))
    return padded  # type: ignore[return-value]


def _state_width(
    source: object,
    count_name: str,
    values_name: str,
    starts: np.ndarray,
) -> int:
    declared = getattr(source, count_name, None)
    if declared is not None:
        return int(declared)
    values = getattr(source, values_name, None)
    if values is not None:
        return _column_length(values)
    if starts.size:
        return int(starts[-1])
    return 0


def _scalar_state_index(
    starts: np.ndarray,
    *,
    joint_index: int,
    total_width: int,
    label: str,
) -> int:
    start = int(starts[joint_index])
    end = int(starts[joint_index + 1]) if joint_index + 1 < starts.size else total_width
    if start < 0 or end - start != 1 or end > total_width:
        raise NewtonConstraintAuditError(
            f"Newton master/follower joint must own exactly one {label} coordinate: "
            f"joint={joint_index}, start={start}, end={end}, total={total_width}"
        )
    return start


def _positive_int(value: object, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return result


def _validated_selected_worlds(
    selected_worlds: Sequence[int],
    *,
    world_count: int,
) -> tuple[int, ...]:
    """规范化静态 world id，同时拒绝隐式截断 float/bool 等模糊 selector。"""

    from operator import index

    worlds: list[int] = []
    try:
        for value in selected_worlds:
            if isinstance(value, bool):
                raise TypeError
            worlds.append(index(value))
    except TypeError as exc:
        raise ValueError(
            "selected_worlds must be unique indices in [0, world_count): "
            f"selected={list(selected_worlds)!r}, world_count={world_count}"
        ) from exc
    if any(world < 0 or world >= world_count for world in worlds) or len(
        set(worlds)
    ) != len(worlds):
        raise ValueError(
            "selected_worlds must be unique indices in [0, world_count): "
            f"selected={worlds!r}, world_count={world_count}"
        )
    return tuple(worlds)


def _nonempty_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise NewtonConstraintAuditError(
            f"{name} must be a non-empty exact Newton label"
        )
    return value


def _require_warp_vector(
    value: object,
    *,
    name: str,
    dtype: object,
    minimum_size: int,
    device: object,
    exact_size: int | None = None,
) -> None:
    actual_dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    actual_device = getattr(value, "device", None)
    if actual_dtype != dtype or shape is None or len(shape) != 1:
        raise TypeError(
            f"{name} must be a one-dimensional Warp array with dtype {dtype}: "
            f"dtype={actual_dtype!r}, shape={shape!r}"
        )
    size = int(shape[0])
    if size < int(minimum_size) or (exact_size is not None and size != int(exact_size)):
        requirement = (
            f"exactly {exact_size}"
            if exact_size is not None
            else f"at least {minimum_size}"
        )
        raise ValueError(f"{name} must contain {requirement} entries, got {size}")
    if str(actual_device) != str(device):
        raise ValueError(
            f"{name} device differs from the Newton state projector: "
            f"actual={actual_device}, expected={device}"
        )


__all__ = [
    "COLD_STATE_PROJECTION_SCOPE",
    "NATIVE_JOINT_EQUALITY_EXECUTOR",
    "ExpectedMasterFollowerConstraint",
    "MasterFollowerExecutorMetadata",
    "NativeMasterFollowerAudit",
    "NativeMasterFollowerBinding",
    "NewtonColdStateProjector",
    "NewtonConstraintAuditError",
    "NewtonDeviceWorldMasks",
    "audit_native_master_follower_constraints",
    "make_selected_world_mask",
]
