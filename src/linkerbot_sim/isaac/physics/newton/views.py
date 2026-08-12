"""基于 persistent Warp gather/scatter buffer 的 Newton device views。

Isaac Core 兼容层仍需要 articulation/rigid view 形状，但 Newton 不创建 extension-owned
physics tensor entity；这些 view 直接把 finalized-model 的精确 label 绑定到 manager-owned
``Model``、``State`` 和 ``Control`` 数组。view 只是索引与 buffer facade，不拥有第二份状态。

绑定只在初始化冷路径发生，并有意采用严格合同：每个请求 path 必须精确命中一个 label；
每行只属于一个明确 Newton world；replica 必须具有相同有序 topology；普通 rigid setter 只
允许写 world-root FREE body。运行时 indexed 读写复用缓存的 Warp index/output/upload buffer，
不会在热 target setter 中把整批数据读回 CPU。

状态语义尤其重要：articulation 的 ``joint_q/joint_qd`` 是 generalized owner state；
``body_q/body_qd`` 是 maximal 派生状态。dynamic-chain 为兼容旧快照 API 可以读取后者，但
精确 clone/restore 必须保存前者，并只对 selected articulation 执行 IK/FK。Warp selector 的
host 解析只保留给 cold reset/restore，不能进入常规 step loop。
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np


_JOINT_TYPE_FIXED = 3
_JOINT_TYPE_FREE = 4
_JOINT_TYPE_BALL = 2


class NewtonViewBindingError(RuntimeError):
    """请求的 Newton view 无法唯一绑定到 finalized Newton model。"""


def _host_array(value: object, *, dtype: object, name: str) -> np.ndarray:
    """仅在 view 构造冷路径读取不可变拓扑元数据。

    该转换不能用于 step/state 热路径：运行时数值状态必须继续由 Warp device owner 持有，
        不能借这个 helper 隐式下载到主机。
    """

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    try:
        return np.asarray(candidate, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise NewtonViewBindingError(
            f"Newton model column {name!r} cannot be read as {dtype!r}"
        ) from exc


def _one_dimensional_int_column(
    source: object,
    name: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    value = getattr(source, name, None)
    if value is None:
        raise NewtonViewBindingError(f"Newton model is missing {name!r}")
    result = _host_array(value, dtype=np.int64, name=name).reshape(-1)
    if expected_size is not None and result.size != expected_size:
        raise NewtonViewBindingError(
            f"Newton model column {name!r} has size {result.size}; "
            f"expected {expected_size}"
        )
    return result


def _labels(source: object, name: str) -> tuple[str, ...]:
    value = getattr(source, name, None)
    if value is None:
        raise NewtonViewBindingError(f"Newton model is missing {name!r}")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise NewtonViewBindingError(f"Newton model column {name!r} has an empty label")
    return result


def _exact_paths(paths: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(str(path).rstrip("/") or "/" for path in paths)
    if not result:
        raise ValueError(f"{label} must contain at least one exact path")
    if any(not path.startswith("/") or "//" in path for path in result):
        raise ValueError(f"{label} must contain normalized absolute prim paths")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicate paths")
    return result


def _exact_label_indices(
    labels: Sequence[str], paths: Sequence[str], *, entity: str
) -> tuple[int, ...]:
    indices_by_label: dict[str, list[int]] = {}
    for index, item in enumerate(labels):
        indices_by_label.setdefault(str(item), []).append(index)
    result: list[int] = []
    for path in paths:
        matches = indices_by_label.get(path, ())
        if len(matches) != 1:
            raise NewtonViewBindingError(
                f"Newton {entity} path must match exactly one finalized-model label: "
                f"path={path!r}, matches={list(matches)}"
            )
        result.append(int(matches[0]))
    return tuple(result)


def _validated_world_indices(
    actual: Sequence[int],
    expected: Sequence[int] | None,
    *,
    entity: str,
) -> tuple[int, ...]:
    worlds = tuple(int(value) for value in actual)
    if any(world < 0 for world in worlds):
        raise NewtonViewBindingError(
            f"Newton {entity} rows must belong to non-global worlds: {worlds}"
        )
    if len(set(worlds)) != len(worlds):
        raise NewtonViewBindingError(
            f"Newton {entity} view must contain at most one row per world: {worlds}"
        )
    if expected is not None:
        requested = tuple(int(value) for value in expected)
        if requested != worlds:
            raise NewtonViewBindingError(
                f"Newton {entity} paths are not in the requested world order: "
                f"actual={worlds}, expected={requested}"
            )
    return worlds


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _relative_topology_path(
    path: str,
    *,
    articulation_path: str,
    fallback: str | None = None,
) -> str:
    """返回不含 env 前缀、可跨 replicated world 比较的拓扑标签。

    Newton 为 FREE root joint 生成的标签可能是 ``joint_1``，而不是 USD path；但其
    child body 仍有精确的 replicated path，因此只允许该场景使用由 child 推导的 fallback。
    """

    normalized = str(path).rstrip("/")
    root = str(articulation_path).rstrip("/")
    if normalized == root:
        return "."
    prefix = f"{root}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    if fallback is not None:
        return str(fallback)
    raise NewtonViewBindingError(
        "dynamic-chain topology label is outside its articulation root: "
        f"label={normalized!r}, articulation={root!r}"
    )


def _generalized_component_names(
    *,
    joint_type: int,
    width: int,
    velocity: bool,
) -> tuple[str, ...]:
    """用稳定分量名描述 Newton generalized-coordinate ABI，不泄漏全局索引。"""

    if joint_type == _JOINT_TYPE_FREE:
        expected = 6 if velocity else 7
        if width != expected:
            raise NewtonViewBindingError(
                "Newton FREE joint has an unsupported generalized-coordinate "
                f"width: expected={expected}, actual={width}"
            )
        return (
            ("linear.x", "linear.y", "linear.z", "angular.x", "angular.y", "angular.z")
            if velocity
            else (
                "translation.x",
                "translation.y",
                "translation.z",
                "quaternion.x",
                "quaternion.y",
                "quaternion.z",
                "quaternion.w",
            )
        )
    if joint_type == _JOINT_TYPE_BALL:
        expected = 3 if velocity else 4
        if width != expected:
            raise NewtonViewBindingError(
                "Newton BALL joint has an unsupported generalized-coordinate "
                f"width: expected={expected}, actual={width}"
            )
        return (
            ("angular.x", "angular.y", "angular.z")
            if velocity
            else (
                "quaternion.x",
                "quaternion.y",
                "quaternion.z",
                "quaternion.w",
            )
        )
    axis = "qd" if velocity else "q"
    return tuple(f"{axis}[{index}]" for index in range(width))


@dataclass(frozen=True)
class NewtonArticulationBinding:
    """每个 world 的精确 articulation 与标量 DOF 全局下标映射。

    ``q_indices`` 和 ``qd_indices`` 分开保存：Newton generalized position 与 velocity 的
    存储宽度并非对所有 joint 相同，不能假定二者共享 index 或固定偏移。
    """

    paths: tuple[str, ...]
    articulation_indices: tuple[int, ...]
    world_indices: tuple[int, ...]
    dof_names: tuple[str, ...]
    dof_paths: tuple[tuple[str, ...], ...]
    q_indices: tuple[tuple[int, ...], ...]
    qd_indices: tuple[tuple[int, ...], ...]

    @property
    def count(self) -> int:
        return len(self.paths)

    @property
    def num_dofs(self) -> int:
        return len(self.dof_names)

    @classmethod
    def from_model(
        cls,
        model: object,
        paths: Sequence[str],
        *,
        world_indices: Sequence[int] | None = None,
    ) -> NewtonArticulationBinding:
        exact = _exact_paths(paths, label="articulation paths")
        articulation_labels = _labels(model, "articulation_label")
        articulation_ids = _exact_label_indices(
            articulation_labels, exact, entity="articulation"
        )
        articulation_world = _one_dimensional_int_column(
            model,
            "articulation_world",
            expected_size=len(articulation_labels),
        )
        worlds = _validated_world_indices(
            [articulation_world[index] for index in articulation_ids],
            world_indices,
            entity="articulation",
        )

        joint_labels = _labels(model, "joint_label")
        joint_count = len(joint_labels)
        joint_world = _one_dimensional_int_column(
            model, "joint_world", expected_size=joint_count
        )
        joint_q_start = _one_dimensional_int_column(model, "joint_q_start")
        joint_qd_start = _one_dimensional_int_column(model, "joint_qd_start")
        articulation_start = _one_dimensional_int_column(model, "articulation_start")
        if joint_q_start.size != joint_count + 1:
            raise NewtonViewBindingError(
                "finalized Newton joint_q_start must include its sentinel"
            )
        if joint_qd_start.size != joint_count + 1:
            raise NewtonViewBindingError(
                "finalized Newton joint_qd_start must include its sentinel"
            )
        if articulation_start.size != len(articulation_labels) + 1:
            raise NewtonViewBindingError(
                "finalized Newton articulation_start must include its sentinel"
            )

        paths_by_row: list[tuple[str, ...]] = []
        names_by_row: list[tuple[str, ...]] = []
        q_by_row: list[tuple[int, ...]] = []
        qd_by_row: list[tuple[int, ...]] = []
        for articulation_id, world in zip(articulation_ids, worlds, strict=True):
            start = int(articulation_start[articulation_id])
            end = int(articulation_start[articulation_id + 1])
            if start < 0 or end < start or end > joint_count:
                raise NewtonViewBindingError(
                    f"Newton articulation {articulation_id} has an invalid joint range "
                    f"[{start}, {end})"
                )
            row_paths: list[str] = []
            row_names: list[str] = []
            row_q: list[int] = []
            row_qd: list[int] = []
            for joint_index in range(start, end):
                if int(joint_world[joint_index]) != world:
                    raise NewtonViewBindingError(
                        f"Newton articulation {articulation_id} contains joint "
                        f"{joint_index} from world {int(joint_world[joint_index])}"
                    )
                q_start = int(joint_q_start[joint_index])
                q_end = int(joint_q_start[joint_index + 1])
                qd_start = int(joint_qd_start[joint_index])
                qd_end = int(joint_qd_start[joint_index + 1])
                q_size = q_end - q_start
                qd_size = qd_end - qd_start
                if q_size == 0 and qd_size == 0:
                    continue
                if q_size != 1 or qd_size != 1:
                    raise NewtonViewBindingError(
                        "Newton articulation view currently supports scalar DOFs only: "
                        f"joint={joint_labels[joint_index]!r}, q_size={q_size}, "
                        f"qd_size={qd_size}"
                    )
                joint_path = joint_labels[joint_index]
                row_paths.append(joint_path)
                row_names.append(_basename(joint_path))
                row_q.append(q_start)
                row_qd.append(qd_start)
            if not row_names:
                raise NewtonViewBindingError(
                    f"Newton articulation {exact[len(names_by_row)]!r} has no scalar DOFs"
                )
            if len(set(row_names)) != len(row_names):
                raise NewtonViewBindingError(
                    "Newton articulation has duplicate scalar DOF basenames: "
                    f"path={exact[len(names_by_row)]!r}, names={row_names}"
                )
            paths_by_row.append(tuple(row_paths))
            names_by_row.append(tuple(row_names))
            q_by_row.append(tuple(row_q))
            qd_by_row.append(tuple(row_qd))

        expected_names = names_by_row[0]
        for row, names in enumerate(names_by_row[1:], start=1):
            if names != expected_names:
                raise NewtonViewBindingError(
                    "Newton articulation rows must have identical ordered DOFs: "
                    f"row_0={expected_names}, row_{row}={names}"
                )
        return cls(
            paths=exact,
            articulation_indices=articulation_ids,
            world_indices=worlds,
            dof_names=expected_names,
            dof_paths=tuple(paths_by_row),
            q_indices=tuple(q_by_row),
            qd_indices=tuple(qd_by_row),
        )


@dataclass(frozen=True)
class NewtonRigidBodyBinding:
    """每个 world 的精确 body 映射，并记录可写 root FREE joint。

    body 若没有 world-root FREE joint，其 maximal pose 虽可读取，却不能独立写回 generalized
    state；把普通 articulated link 当 rigid root 写入会制造两套互相矛盾的状态。
    """

    paths: tuple[str, ...]
    body_indices: tuple[int, ...]
    world_indices: tuple[int, ...]
    free_q_starts: tuple[int, ...]
    free_qd_starts: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.paths)

    @classmethod
    def from_model(
        cls,
        model: object,
        paths: Sequence[str],
        *,
        world_indices: Sequence[int] | None = None,
    ) -> NewtonRigidBodyBinding:
        exact = _exact_paths(paths, label="rigid body paths")
        body_labels = _labels(model, "body_label")
        body_ids = _exact_label_indices(body_labels, exact, entity="body")
        body_world = _one_dimensional_int_column(
            model, "body_world", expected_size=len(body_labels)
        )
        worlds = _validated_world_indices(
            [body_world[index] for index in body_ids],
            world_indices,
            entity="rigid body",
        )

        joint_labels = _labels(model, "joint_label")
        joint_count = len(joint_labels)
        joint_child = _one_dimensional_int_column(
            model, "joint_child", expected_size=joint_count
        )
        joint_parent = _one_dimensional_int_column(
            model, "joint_parent", expected_size=joint_count
        )
        joint_type = _one_dimensional_int_column(
            model, "joint_type", expected_size=joint_count
        )
        joint_world = _one_dimensional_int_column(
            model, "joint_world", expected_size=joint_count
        )
        q_start = _one_dimensional_int_column(model, "joint_q_start")
        qd_start = _one_dimensional_int_column(model, "joint_qd_start")
        if q_start.size != joint_count + 1 or qd_start.size != joint_count + 1:
            raise NewtonViewBindingError(
                "finalized Newton joint start arrays must include sentinels"
            )

        free_q: list[int] = []
        free_qd: list[int] = []
        for body_id, world in zip(body_ids, worlds, strict=True):
            incoming = np.flatnonzero(joint_child == body_id)
            free = [
                int(index)
                for index in incoming
                if int(joint_type[index]) == _JOINT_TYPE_FREE
            ]
            if not free:
                free_q.append(-1)
                free_qd.append(-1)
                continue
            if len(free) != 1:
                raise NewtonViewBindingError(
                    f"Newton body {body_labels[body_id]!r} has multiple FREE joints: {free}"
                )
            joint_index = free[0]
            q_size = int(q_start[joint_index + 1] - q_start[joint_index])
            qd_size = int(qd_start[joint_index + 1] - qd_start[joint_index])
            if (
                int(joint_parent[joint_index]) != -1
                or int(joint_world[joint_index]) != world
                or q_size != 7
                or qd_size != 6
            ):
                raise NewtonViewBindingError(
                    "Newton rigid writes require a world-root FREE joint with 7/6 "
                    f"coordinates: body={body_labels[body_id]!r}, "
                    f"joint={joint_labels[joint_index]!r}"
                )
            free_q.append(int(q_start[joint_index]))
            free_qd.append(int(qd_start[joint_index]))
        return cls(
            paths=exact,
            body_indices=body_ids,
            world_indices=worlds,
            free_q_starts=tuple(free_q),
            free_qd_starts=tuple(free_qd),
        )


@dataclass(frozen=True)
class NewtonDynamicChainBinding:
    """replicated dynamic chain 的精确同构 articulation/body 映射。

    body row 使用 replicated view 约定的 env-major 顺序；
    ``q_indices``/``qd_indices`` 保存每条链完整 generalized owner state。body transform 只是
    portable/maximal 交换格式，不足以唯一表示带关节约束的链状态。

    signature 使用相对 articulation root 的 topology 名称，不含 env_0/env_1 前缀和全局数组
    index；这样 snapshot 可跨相同 ABI 的 replica 恢复，同时仍会拒绝 joint 类型、parent-child
    或坐标宽度发生变化的场景。
    """

    articulation_paths: tuple[str, ...]
    articulation_indices: tuple[int, ...]
    world_indices: tuple[int, ...]
    body_paths_by_env: tuple[tuple[str, ...], ...]
    body_indices_by_env: tuple[tuple[int, ...], ...]
    q_indices: tuple[tuple[int, ...], ...]
    qd_indices: tuple[tuple[int, ...], ...]
    q_coordinate_names: tuple[str, ...]
    qd_coordinate_names: tuple[str, ...]
    coordinate_signature: tuple[str, ...]
    world_translation_q_indices: tuple[int, ...]

    @property
    def env_count(self) -> int:
        return len(self.articulation_paths)

    @property
    def body_count(self) -> int:
        return len(self.body_paths_by_env[0])

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(path for row in self.body_paths_by_env for path in row)

    @property
    def body_indices(self) -> tuple[int, ...]:
        return tuple(index for row in self.body_indices_by_env for index in row)

    @property
    def count(self) -> int:
        return self.env_count * self.body_count

    @property
    def free_q_starts(self) -> tuple[int, ...]:
        # chain restore 不能只更新 FREE root；所有 maximal body row 必须由 selected eval_ik
        # 原子转换回 generalized state。这里返回 -1，避免专用完整-body scatter 把任一 body
        # 误当作可独立写入的 FREE root；对外 fail-closed 由本 view 的 _require_writable 保证。
        return (-1,) * self.count

    @property
    def free_qd_starts(self) -> tuple[int, ...]:
        return (-1,) * self.count

    @property
    def row_world_indices(self) -> tuple[int, ...]:
        return tuple(
            world for world in self.world_indices for _ in range(self.body_count)
        )

    @property
    def generalized_coordinate_signature(self) -> tuple[str, ...]:
        """返回所有 replicated env 必须共享的稳定拓扑/ABI 签名。"""

        return self.coordinate_signature

    @classmethod
    def from_model(
        cls,
        model: object,
        *,
        articulation_paths: Sequence[str],
        body_paths_by_env: Sequence[Sequence[str]],
        world_indices: Sequence[int] | None = None,
    ) -> NewtonDynamicChainBinding:
        articulations = _exact_paths(
            articulation_paths, label="dynamic-chain articulation paths"
        )
        body_rows = tuple(
            _exact_paths(row, label=f"dynamic-chain body paths row {row_index}")
            for row_index, row in enumerate(body_paths_by_env)
        )
        if len(body_rows) != len(articulations):
            raise NewtonViewBindingError(
                "dynamic-chain articulation/body row counts differ: "
                f"{len(articulations)} vs {len(body_rows)}"
            )
        body_count = len(body_rows[0])
        if any(len(row) != body_count for row in body_rows):
            raise NewtonViewBindingError(
                "Newton dynamic-chain rows must have identical body counts"
            )
        suffixes = tuple(
            path[len(articulations[0]) :]
            if path.startswith(f"{articulations[0]}/")
            else ""
            for path in body_rows[0]
        )
        if any(not suffix for suffix in suffixes):
            raise NewtonViewBindingError(
                "dynamic-chain body paths must be below their articulation root"
            )
        for articulation, row in zip(articulations, body_rows, strict=True):
            actual = tuple(
                path[len(articulation) :] if path.startswith(f"{articulation}/") else ""
                for path in row
            )
            if actual != suffixes:
                raise NewtonViewBindingError(
                    "Newton dynamic-chain rows must have identical ordered "
                    f"body suffixes: expected={suffixes}, actual={actual}"
                )

        articulation_labels = _labels(model, "articulation_label")
        articulation_ids = _exact_label_indices(
            articulation_labels, articulations, entity="dynamic-chain articulation"
        )
        articulation_world = _one_dimensional_int_column(
            model,
            "articulation_world",
            expected_size=len(articulation_labels),
        )
        worlds = _validated_world_indices(
            [articulation_world[index] for index in articulation_ids],
            world_indices,
            entity="dynamic-chain articulation",
        )

        body_labels = _labels(model, "body_label")
        body_world = _one_dimensional_int_column(
            model, "body_world", expected_size=len(body_labels)
        )
        body_ids_by_env: list[tuple[int, ...]] = []
        for row, world in zip(body_rows, worlds, strict=True):
            body_ids = _exact_label_indices(
                body_labels, row, entity="dynamic-chain body"
            )
            wrong_world = [
                body_labels[index]
                for index in body_ids
                if int(body_world[index]) != world
            ]
            if wrong_world:
                raise NewtonViewBindingError(
                    "dynamic-chain bodies do not belong to their articulation world: "
                    f"{wrong_world}"
                )
            body_ids_by_env.append(body_ids)

        joint_labels = _labels(model, "joint_label")
        joint_count = len(joint_labels)
        articulation_start = _one_dimensional_int_column(model, "articulation_start")
        joint_child = _one_dimensional_int_column(
            model, "joint_child", expected_size=joint_count
        )
        joint_parent = _one_dimensional_int_column(
            model, "joint_parent", expected_size=joint_count
        )
        joint_type = _one_dimensional_int_column(
            model, "joint_type", expected_size=joint_count
        )
        joint_world = _one_dimensional_int_column(
            model, "joint_world", expected_size=joint_count
        )
        q_start = _one_dimensional_int_column(model, "joint_q_start")
        qd_start = _one_dimensional_int_column(model, "joint_qd_start")
        if articulation_start.size != len(articulation_labels) + 1:
            raise NewtonViewBindingError(
                "finalized Newton articulation_start must include its sentinel"
            )
        if q_start.size != joint_count + 1 or qd_start.size != joint_count + 1:
            raise NewtonViewBindingError(
                "finalized Newton joint start arrays must include sentinels"
            )

        q_rows: list[tuple[int, ...]] = []
        qd_rows: list[tuple[int, ...]] = []
        q_name_rows: list[tuple[str, ...]] = []
        qd_name_rows: list[tuple[str, ...]] = []
        signature_rows: list[tuple[str, ...]] = []
        translation_rows: list[tuple[int, ...]] = []
        dimensions: tuple[int, int] | None = None
        for articulation_path, articulation_id, world, requested_bodies in zip(
            articulations,
            articulation_ids,
            worlds,
            body_ids_by_env,
            strict=True,
        ):
            start = int(articulation_start[articulation_id])
            end = int(articulation_start[articulation_id + 1])
            if start < 0 or end <= start or end > joint_count:
                raise NewtonViewBindingError(
                    f"dynamic-chain articulation has invalid joint range [{start}, {end})"
                )
            if any(int(joint_world[index]) != world for index in range(start, end)):
                raise NewtonViewBindingError(
                    "dynamic-chain articulation contains a joint from another world"
                )
            articulation_bodies = tuple(
                int(joint_child[index])
                for index in range(start, end)
                if int(joint_child[index]) >= 0
            )
            if len(set(articulation_bodies)) != len(articulation_bodies):
                raise NewtonViewBindingError(
                    "dynamic-chain articulation has duplicate child-body ownership"
                )
            if set(articulation_bodies) != set(requested_bodies):
                missing = sorted(set(articulation_bodies) - set(requested_bodies))
                extra = sorted(set(requested_bodies) - set(articulation_bodies))
                raise NewtonViewBindingError(
                    "dynamic-chain view must contain every articulation body exactly "
                    f"once: missing={missing}, extra={extra}"
                )
            q_row = tuple(range(int(q_start[start]), int(q_start[end])))
            qd_row = tuple(range(int(qd_start[start]), int(qd_start[end])))
            # 保存 articulation 的完整连续坐标区间，而不只保存可控 DOF。FREE root、被动
            # joint 和 equality follower 都属于精确恢复所需的 owner state。
            if not q_row or not qd_row:
                raise NewtonViewBindingError(
                    "dynamic-chain articulation has no generalized coordinates"
                )
            current_dimensions = (len(q_row), len(qd_row))
            if dimensions is None:
                dimensions = current_dimensions
            elif current_dimensions != dimensions:
                raise NewtonViewBindingError(
                    "Newton dynamic-chain rows must have identical generalized "
                    f"dimensions: expected={dimensions}, actual={current_dimensions}"
                )
            q_rows.append(q_row)
            qd_rows.append(qd_row)

            q_names: list[str] = []
            qd_names: list[str] = []
            signature = [
                "newton-generalized-state-v1",
                "quaternion-abi=xyzw;twist-abi=linear.xyz,angular.xyz",
                "body-topology="
                + ",".join(
                    sorted(
                        _relative_topology_path(
                            body_labels[body_id],
                            articulation_path=articulation_path,
                        )
                        for body_id in requested_bodies
                    )
                ),
            ]
            translation_indices: list[int] = []
            for joint_index in range(start, end):
                child_id = int(joint_child[joint_index])
                if child_id < 0 or child_id >= len(body_labels):
                    raise NewtonViewBindingError(
                        "dynamic-chain joint has an invalid child body index: "
                        f"joint={joint_labels[joint_index]!r}, child={child_id}"
                    )
                child = _relative_topology_path(
                    body_labels[child_id],
                    articulation_path=articulation_path,
                )
                parent_id = int(joint_parent[joint_index])
                parent = (
                    "<world>"
                    if parent_id < 0
                    else _relative_topology_path(
                        body_labels[parent_id],
                        articulation_path=articulation_path,
                    )
                )
                joint = _relative_topology_path(
                    joint_labels[joint_index],
                    articulation_path=articulation_path,
                    fallback=(f"@root/{child}" if parent_id < 0 else None),
                )
                q_width = int(q_start[joint_index + 1] - q_start[joint_index])
                qd_width = int(qd_start[joint_index + 1] - qd_start[joint_index])
                kind = int(joint_type[joint_index])
                q_components = _generalized_component_names(
                    joint_type=kind,
                    width=q_width,
                    velocity=False,
                )
                qd_components = _generalized_component_names(
                    joint_type=kind,
                    width=qd_width,
                    velocity=True,
                )
                q_offset = int(q_start[joint_index] - q_start[start])
                if kind == _JOINT_TYPE_FREE and parent_id < 0:
                    translation_indices.extend((q_offset, q_offset + 1, q_offset + 2))
                q_names.extend(f"{joint}|{component}" for component in q_components)
                qd_names.extend(f"{joint}|{component}" for component in qd_components)
                signature.append(
                    f"joint={joint};parent={parent};child={child};type={kind};"
                    f"q_width={q_width};qd_width={qd_width}"
                )
            q_name_row = tuple(q_names)
            qd_name_row = tuple(qd_names)
            if len(q_name_row) != len(q_row) or len(qd_name_row) != len(qd_row):
                raise NewtonViewBindingError(
                    "dynamic-chain generalized coordinate identity width mismatch"
                )
            if len(set(q_name_row)) != len(q_name_row) or len(set(qd_name_row)) != len(
                qd_name_row
            ):
                raise NewtonViewBindingError(
                    "dynamic-chain generalized coordinate names are not unique"
                )
            q_name_rows.append(q_name_row)
            qd_name_rows.append(qd_name_row)
            signature_rows.append(tuple(signature))
            translation_rows.append(tuple(translation_indices))
        for label, rows in (
            ("q coordinate names", q_name_rows),
            ("qd coordinate names", qd_name_rows),
            ("coordinate signature", signature_rows),
            ("world translation indices", translation_rows),
        ):
            if any(row != rows[0] for row in rows[1:]):
                raise NewtonViewBindingError(
                    f"Newton dynamic-chain replicas have different {label}"
                )
        return cls(
            articulation_paths=articulations,
            articulation_indices=articulation_ids,
            world_indices=worlds,
            body_paths_by_env=body_rows,
            body_indices_by_env=tuple(body_ids_by_env),
            q_indices=tuple(q_rows),
            qd_indices=tuple(qd_rows),
            q_coordinate_names=q_name_rows[0],
            qd_coordinate_names=qd_name_rows[0],
            coordinate_signature=signature_rows[0],
            world_translation_q_indices=translation_rows[0],
        )


@dataclass
class _WarpSelection:
    """一个稳定 selector 及其按用途缓存的 GPU output/staging buffer。"""

    rows_host: tuple[int, ...]
    columns_host: tuple[int, ...]
    rows: object
    columns: object
    outputs: dict[str, object] = field(default_factory=dict)
    staging: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _WarpKernels:
    gather_scalar: object
    scatter_scalar: object
    broadcast_vector: object
    broadcast_matrix: object
    gather_pose: object
    scatter_pose: object
    gather_velocity: object
    scatter_velocity: object


_warp_kernels: _WarpKernels | None = None


def _load_warp_kernels() -> _WarpKernels:
    global _warp_kernels
    if _warp_kernels is not None:
        return _warp_kernels
    import warp as wp

    @wp.kernel(enable_backward=False)
    def gather_scalar(
        source: wp.array(dtype=wp.float32),
        global_indices: wp.array(dtype=wp.int32),
        rows: wp.array(dtype=wp.int32),
        columns: wp.array(dtype=wp.int32),
        binding_column_count: int,
        output: wp.array2d(dtype=wp.float32),
    ):
        row, column = wp.tid()
        mapping_index = rows[row] * binding_column_count + columns[column]
        output[row, column] = source[global_indices[mapping_index]]

    @wp.kernel(enable_backward=False)
    def scatter_scalar(
        values: wp.array2d(dtype=wp.float32),
        global_indices: wp.array(dtype=wp.int32),
        rows: wp.array(dtype=wp.int32),
        columns: wp.array(dtype=wp.int32),
        binding_column_count: int,
        destination: wp.array(dtype=wp.float32),
    ):
        row, column = wp.tid()
        mapping_index = rows[row] * binding_column_count + columns[column]
        destination[global_indices[mapping_index]] = values[row, column]

    @wp.kernel(enable_backward=False)
    def broadcast_vector(
        source: wp.array(dtype=wp.float32),
        source_columns: int,
        output: wp.array2d(dtype=wp.float32),
    ):
        row, column = wp.tid()
        source_column = wp.where(source_columns == 1, 0, column)
        output[row, column] = source[source_column]

    @wp.kernel(enable_backward=False)
    def broadcast_matrix(
        source: wp.array2d(dtype=wp.float32),
        source_rows: int,
        source_columns: int,
        output: wp.array2d(dtype=wp.float32),
    ):
        row, column = wp.tid()
        source_row = wp.where(source_rows == 1, 0, row)
        source_column = wp.where(source_columns == 1, 0, column)
        output[row, column] = source[source_row, source_column]

    @wp.kernel(enable_backward=False)
    def gather_pose(
        body_q: wp.array(dtype=wp.transform),
        body_indices: wp.array(dtype=wp.int32),
        rows: wp.array(dtype=wp.int32),
        positions: wp.array2d(dtype=wp.float32),
        orientations_wxyz: wp.array2d(dtype=wp.float32),
    ):
        row = wp.tid()
        transform = body_q[body_indices[rows[row]]]
        position = wp.transform_get_translation(transform)
        orientation = wp.transform_get_rotation(transform)
        positions[row, 0] = position[0]
        positions[row, 1] = position[1]
        positions[row, 2] = position[2]
        orientations_wxyz[row, 0] = orientation[3]
        orientations_wxyz[row, 1] = orientation[0]
        orientations_wxyz[row, 2] = orientation[1]
        orientations_wxyz[row, 3] = orientation[2]

    @wp.kernel(enable_backward=False)
    def scatter_pose(
        positions: wp.array2d(dtype=wp.float32),
        orientations_wxyz: wp.array2d(dtype=wp.float32),
        write_positions: int,
        write_orientations: int,
        body_indices: wp.array(dtype=wp.int32),
        free_q_starts: wp.array(dtype=wp.int32),
        rows: wp.array(dtype=wp.int32),
        body_q: wp.array(dtype=wp.transform),
        joint_q: wp.array(dtype=wp.float32),
    ):
        output_row = wp.tid()
        binding_row = rows[output_row]
        body_id = body_indices[binding_row]
        current = body_q[body_id]
        position = wp.transform_get_translation(current)
        orientation = wp.transform_get_rotation(current)
        if write_positions != 0:
            position = wp.vec3(
                positions[output_row, 0],
                positions[output_row, 1],
                positions[output_row, 2],
            )
        if write_orientations != 0:
            orientation = wp.normalize(
                wp.quat(
                    orientations_wxyz[output_row, 1],
                    orientations_wxyz[output_row, 2],
                    orientations_wxyz[output_row, 3],
                    orientations_wxyz[output_row, 0],
                )
            )
        updated = wp.transform(position, orientation)
        body_q[body_id] = updated
        q_start = free_q_starts[binding_row]
        if q_start >= 0:
            joint_q[q_start + 0] = position[0]
            joint_q[q_start + 1] = position[1]
            joint_q[q_start + 2] = position[2]
            joint_q[q_start + 3] = orientation[0]
            joint_q[q_start + 4] = orientation[1]
            joint_q[q_start + 5] = orientation[2]
            joint_q[q_start + 6] = orientation[3]

    @wp.kernel(enable_backward=False)
    def gather_velocity(
        body_qd: wp.array(dtype=wp.spatial_vector),
        body_indices: wp.array(dtype=wp.int32),
        rows: wp.array(dtype=wp.int32),
        linear: wp.array2d(dtype=wp.float32),
        angular: wp.array2d(dtype=wp.float32),
    ):
        row = wp.tid()
        velocity = body_qd[body_indices[rows[row]]]
        # Newton 1.2.1 面向公共状态消费者的 body twist 顺序是
        # [linear, angular]；这里显式拆分，避免受旧 kernel 注释中的相反顺序影响。
        linear[row, 0] = velocity[0]
        linear[row, 1] = velocity[1]
        linear[row, 2] = velocity[2]
        angular[row, 0] = velocity[3]
        angular[row, 1] = velocity[4]
        angular[row, 2] = velocity[5]

    @wp.kernel(enable_backward=False)
    def scatter_velocity(
        linear: wp.array2d(dtype=wp.float32),
        angular: wp.array2d(dtype=wp.float32),
        write_linear: int,
        write_angular: int,
        body_indices: wp.array(dtype=wp.int32),
        free_qd_starts: wp.array(dtype=wp.int32),
        rows: wp.array(dtype=wp.int32),
        body_qd: wp.array(dtype=wp.spatial_vector),
        joint_qd: wp.array(dtype=wp.float32),
    ):
        output_row = wp.tid()
        binding_row = rows[output_row]
        body_id = body_indices[binding_row]
        current = body_qd[body_id]
        v0 = current[0]
        v1 = current[1]
        v2 = current[2]
        v3 = current[3]
        v4 = current[4]
        v5 = current[5]
        if write_linear != 0:
            v0 = linear[output_row, 0]
            v1 = linear[output_row, 1]
            v2 = linear[output_row, 2]
        if write_angular != 0:
            v3 = angular[output_row, 0]
            v4 = angular[output_row, 1]
            v5 = angular[output_row, 2]
        body_qd[body_id] = wp.spatial_vector(v0, v1, v2, v3, v4, v5)
        qd_start = free_qd_starts[binding_row]
        if qd_start >= 0:
            joint_qd[qd_start + 0] = v0
            joint_qd[qd_start + 1] = v1
            joint_qd[qd_start + 2] = v2
            joint_qd[qd_start + 3] = v3
            joint_qd[qd_start + 4] = v4
            joint_qd[qd_start + 5] = v5

    _warp_kernels = _WarpKernels(
        gather_scalar=gather_scalar,
        scatter_scalar=scatter_scalar,
        broadcast_vector=broadcast_vector,
        broadcast_matrix=broadcast_matrix,
        gather_pose=gather_pose,
        scatter_pose=scatter_pose,
        gather_velocity=gather_velocity,
        scatter_velocity=scatter_velocity,
    )
    return _warp_kernels


def _selector_to_host(
    value: object | None,
    *,
    count: int,
    label: str,
    stream: object | None = None,
) -> tuple[int, ...]:
    if value is None:
        return tuple(range(count))
    candidate = value
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method) and not isinstance(candidate, np.ndarray):
        # 这是为既有 Core wrapper 的 cold full-batch setter 保留的兼容分支。热 target
        # selector 应是预先准备好的 host index，不能让 ``.numpy()`` 混入正常 step。
        # 若 selector 刚由 caller stream 产生，sync_enter 用事件把 manager stream 排在其后；
        # Warp CUDA ``.numpy()`` 再经 null stream 等待当前 manager stream，全程不做全设备同步。
        with _warp_stream_scope(stream, sync_enter=True):
            candidate = numpy_method()
    raw = np.asarray(candidate)
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{label} must contain integer indices")
    selected = raw.astype(np.int64, copy=False).reshape(-1)
    if np.any(selected < 0) or np.any(selected >= count):
        raise IndexError(f"{label} contains an out-of-range index")
    if np.unique(selected).size != selected.size:
        raise ValueError(f"{label} must not contain duplicate indices")
    return tuple(int(index) for index in selected)


def _strict_finite_matrix(
    value: object,
    *,
    shape: tuple[int, int],
    label: str,
    stream: object | None,
) -> np.ndarray:
    """把冷路径 snapshot tensor 严格复制为指定形状的有限 host matrix。

    该函数服务 Mirror/持久化恢复等显式 CPU 边界；Kaleidoscope GPU episode snapshot
    不经过这里。
    """

    candidate = value
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method) and not isinstance(candidate, np.ndarray):
        with _warp_stream_scope(stream, sync_enter=True):
            candidate = numpy_method()
    try:
        result = np.asarray(candidate, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric matrix") from exc
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def _strict_identity(
    value: Sequence[str],
    *,
    expected: tuple[str, ...],
    label: str,
    unique: bool,
) -> tuple[str, ...]:
    try:
        actual = tuple(str(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence") from exc
    if any(not item for item in actual):
        raise ValueError(f"{label} cannot contain empty entries")
    if unique and len(set(actual)) != len(actual):
        raise ValueError(f"{label} cannot contain duplicate entries")
    if actual != expected:
        raise ValueError(
            f"Newton dynamic-chain {label} mismatch: "
            f"snapshot={actual!r}, target={expected!r}"
        )
    return actual


def _warp_stream_scope(
    stream: object | None,
    *,
    sync_enter: bool = False,
    sync_exit: bool = False,
) -> object:
    """返回基于事件的 Warp stream hand-off context，而非全局同步。

    ``sync_enter`` 只在消费 caller-owned GPU 输入前使用；``sync_exit`` 只在返回值可能马上
    被 null stream/host 消费时使用。manager 内部连续 kernel 默认二者均为 False，依靠同流
    FIFO 排序。
    """

    if stream is None:
        return nullcontext()
    import warp as wp

    return wp.ScopedStream(
        stream,
        sync_enter=sync_enter,
        sync_exit=sync_exit,
    )


class _WarpMatrixBinding:
    """把一维 owner array 映射为二维逻辑 view，并持有 persistent Warp buffers。

    selector、输出和上传 staging 按 (rows, columns, slot) 缓存，既避免每 tick 分配，也使
    CUDA Graph 捕获期间地址稳定。CUDA ``release`` 只能在 manager 确认 owner stream
    空闲后调用；CPU execution 同步执行且没有 owner stream。
    """

    def __init__(
        self,
        global_indices: Sequence[Sequence[int]],
        *,
        device: object,
        stream_provider: object,
    ) -> None:
        import warp as wp

        matrix = np.asarray(global_indices, dtype=np.int32)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("global_indices must be a non-empty matrix")
        self.row_count = int(matrix.shape[0])
        self.column_count = int(matrix.shape[1])
        self.device = wp.get_device(device)
        self._stream_provider = stream_provider
        with _warp_stream_scope(self._stream()):
            self._global_indices = wp.array(
                np.ascontiguousarray(matrix.reshape(-1)),
                dtype=wp.int32,
                device=self.device,
            )
        self._selections: dict[
            tuple[tuple[int, ...], tuple[int, ...]], _WarpSelection
        ] = {}
        self.selection(None, None)

    def _stream(self) -> object | None:
        getter = self._stream_provider
        return getter() if callable(getter) else None

    def release(self) -> None:
        """释放该 mapping 独占的全部持久 Warp allocation 与 selection cache。"""

        for selection in self._selections.values():
            selection.outputs.clear()
            selection.staging.clear()
            selection.rows = None
            selection.columns = None
        self._selections.clear()
        self._global_indices = None
        self._stream_provider = None
        self.device = None

    def selection(self, rows: object | None, columns: object | None) -> _WarpSelection:
        import warp as wp

        row_values = _selector_to_host(
            rows,
            count=self.row_count,
            label="articulation indices",
            stream=self._stream(),
        )
        column_values = _selector_to_host(
            columns,
            count=self.column_count,
            label="DOF indices",
            stream=self._stream(),
        )
        key = (row_values, column_values)
        cached = self._selections.get(key)
        if cached is not None:
            return cached
        with _warp_stream_scope(self._stream()):
            result = _WarpSelection(
                rows_host=row_values,
                columns_host=column_values,
                rows=wp.array(row_values, dtype=wp.int32, device=self.device),
                columns=wp.array(
                    column_values,
                    dtype=wp.int32,
                    device=self.device,
                ),
            )
        self._selections[key] = result
        return result

    def output(self, selection: _WarpSelection, slot: str) -> object:
        import warp as wp

        value = selection.outputs.get(slot)
        if value is None:
            with _warp_stream_scope(self._stream()):
                value = wp.empty(
                    (len(selection.rows_host), len(selection.columns_host)),
                    dtype=wp.float32,
                    device=self.device,
                )
            selection.outputs[slot] = value
        return value

    def staging(self, selection: _WarpSelection, slot: str) -> object:
        import warp as wp

        value = selection.staging.get(slot)
        if value is None:
            with _warp_stream_scope(self._stream()):
                value = wp.empty(
                    (len(selection.rows_host), len(selection.columns_host)),
                    dtype=wp.float32,
                    device=self.device,
                )
            selection.staging[slot] = value
        return value

    def gather(
        self,
        source: object,
        *,
        rows: object | None,
        columns: object | None,
        slot: str,
    ) -> object:
        import warp as wp

        selection = self.selection(rows, columns)
        output = self.output(selection, slot)
        if not selection.rows_host or not selection.columns_host:
            return output
        kernels = _load_warp_kernels()
        stream = self._stream()
        # 返回值常被调用方立即通过 Warp null stream 的 ``.numpy()`` 消费。sync_exit 以事件
        # 交接可见性，使 consumer 等待本次 gather，而不阻塞 CPU 或整个 device。
        with _warp_stream_scope(stream, sync_exit=True):
            wp.launch(
                kernels.gather_scalar,
                dim=(len(selection.rows_host), len(selection.columns_host)),
                inputs=[
                    source,
                    self._global_indices,
                    selection.rows,
                    selection.columns,
                    self.column_count,
                ],
                outputs=[output],
                device=self.device,
                stream=stream,
            )
        return output

    def _fill_staging(
        self,
        values: object,
        *,
        selection: _WarpSelection,
        slot: str,
    ) -> object:
        import warp as wp

        shape = (len(selection.rows_host), len(selection.columns_host))
        staging = self.staging(selection, slot)
        if isinstance(values, wp.array):
            if values.dtype != wp.float32:
                raise TypeError(f"{slot} Warp input must have dtype wp.float32")
            kernels = _load_warp_kernels()
            stream = self._stream()
            # caller-owned Warp tensor 可能刚写在 caller current stream；消费前用事件交给
            # manager stream，避免数据竞争。这不是 global/device sync。
            with _warp_stream_scope(stream, sync_enter=True):
                if values.ndim == 1:
                    source_columns = int(values.shape[0])
                    if source_columns not in {1, shape[1]}:
                        raise ValueError(
                            f"{slot} cannot broadcast Warp shape {values.shape} to {shape}"
                        )
                    wp.launch(
                        kernels.broadcast_vector,
                        dim=shape,
                        inputs=[values, source_columns],
                        outputs=[staging],
                        device=self.device,
                        stream=stream,
                    )
                    return staging
                if values.ndim == 2:
                    source_shape = (int(values.shape[0]), int(values.shape[1]))
                    if source_shape[0] not in {1, shape[0]} or source_shape[1] not in {
                        1,
                        shape[1],
                    }:
                        raise ValueError(
                            f"{slot} cannot broadcast Warp shape {values.shape} to {shape}"
                        )
                    wp.launch(
                        kernels.broadcast_matrix,
                        dim=shape,
                        inputs=[values, source_shape[0], source_shape[1]],
                        outputs=[staging],
                        device=self.device,
                        stream=stream,
                    )
                    return staging
            raise ValueError(f"{slot} Warp input must be one- or two-dimensional")

        try:
            broadcast = np.broadcast_to(np.asarray(values, dtype=np.float32), shape)
        except ValueError as exc:
            raise ValueError(f"{slot} cannot broadcast input to {shape}") from exc
        contiguous = np.ascontiguousarray(broadcast, dtype=np.float32)
        with _warp_stream_scope(self._stream()):
            staging.assign(contiguous)
        return staging

    def scatter(
        self,
        destination: object,
        values: object,
        *,
        rows: object | None,
        columns: object | None,
        slot: str,
    ) -> _WarpSelection:
        import warp as wp

        selection = self.selection(rows, columns)
        if not selection.rows_host or not selection.columns_host:
            return selection
        staging = self._fill_staging(values, selection=selection, slot=slot)
        kernels = _load_warp_kernels()
        wp.launch(
            kernels.scatter_scalar,
            dim=(len(selection.rows_host), len(selection.columns_host)),
            inputs=[
                staging,
                self._global_indices,
                selection.rows,
                selection.columns,
                self.column_count,
            ],
            outputs=[destination],
            device=self.device,
            stream=self._stream(),
        )
        return selection


class _NewtonViewBase:
    """所有 Newton view 的生命周期基类；manager 是状态 owner，view 只拥有映射资源。"""

    def __init__(
        self,
        manager: object | None,
        *,
        model: object | None,
        state: object | None,
        control: object | None,
    ) -> None:
        self._manager = manager
        self._model = model if model is not None else getattr(manager, "model", None)
        if self._model is None:
            raise ValueError("Newton view requires a finalized model")
        self._state_fallback = state
        self._control_fallback = control
        self._closed = False
        self._registered = False

    @property
    def valid(self) -> bool:
        return not self._closed and not bool(getattr(self._manager, "closed", False))

    def is_physics_tensor_entity_valid(self) -> bool:
        return self.valid

    def initialize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self._require_valid()

    def close(self) -> None:
        if self._closed:
            return
        release = getattr(self._manager, "release_newton_view", None)
        if callable(release):
            release(self)
            return
        self._release_from_manager()

    def _register_with_manager(self) -> None:
        register = getattr(self._manager, "register_newton_view", None)
        if callable(register):
            register(self)
            self._registered = True

    def _release_from_manager(self) -> None:
        """失效化 view，并断开全部 native/GPU owner 引用。

        manager.close 会在 CUDA owner stream 同步后调用这里；CPU execution 没有 stream。
        先释放 subclass persistent buffer，再清 model/state/manager 引用，可阻止迟到的业务
        调用访问半销毁 Newton state。
        """

        if self._closed:
            return
        self._closed = True
        try:
            self._release_view_resources()
        finally:
            self._model = None
            self._state_fallback = None
            self._control_fallback = None
            self._manager = None
            self._registered = False

    def _release_view_resources(self) -> None:
        """供子类释放持久 Warp array 和 mapping cache 的 teardown 钩子。"""

    def _require_valid(self) -> None:
        if not self.valid:
            raise RuntimeError("Newton view is closed")

    def _runtime_value(self, name: str, fallback: object | None) -> object:
        self._require_valid()
        value = getattr(self._manager, name, None)
        if value is None and name == "state":
            value = getattr(self._manager, "current_state", None)
        if value is None:
            value = fallback
        if value is None:
            raise RuntimeError(f"Newton manager has no live {name}")
        return value

    def _state(self) -> object:
        return self._runtime_value("state", self._state_fallback)

    def _control(self) -> object:
        return self._runtime_value("control", self._control_fallback)

    @property
    def owner_stream(self) -> object:
        """返回 manager 的 live CUDA stream，供零拷贝设备适配层做事件交接。

        该属性不转移 stream 所有权；调用方不得同步、销毁或替换它。view 关闭后继续访问会
        fail closed，避免 Torch external-stream wrapper 指向已销毁的 native stream。
        """

        self._require_valid()
        stream = self._stream()
        if stream is None:
            raise RuntimeError("Newton view has no live owner stream")
        return stream

    def borrow_runtime_array(self, *, category: str, field: str) -> object:
        """借用当前 state/control Warp array，不复制也不缓存 owner。

        manager 将来可以在 step 边界替换 state/control，因此设备适配层必须在每次操作时重新
        借用。返回 array 的生命周期不超过 view/manager，调用方只能建立零拷贝临时别名。
        """

        if category == "state":
            owner = self._state()
        elif category == "control":
            owner = self._control()
        else:
            raise ValueError("runtime array category must be state or control")
        return _required_array(owner, str(field), category=category)

    def notify_device_write(
        self,
        *,
        category: str,
        field: str,
        device_row_mask: object | None = None,
    ) -> None:
        """通知 manager 一个 CUDA-selector 写入，保守标记本 view 的全部 world。

        普通 CUDA selector 的值不能下载到 host 来精确构造 ``world_indices``，因此默认保守
        覆盖整个 view。SAME_STEP 可额外传固定 shape 的 CUDA bool ``device_row_mask``；manager
        会用注册期缓存的 row→world 映射在设备端生成选择 mask，不调用 nonzero/any/item。
        """

        binding = getattr(self, "binding", None)
        worlds = getattr(binding, "world_indices", None)
        if worlds is None:
            raise RuntimeError("Newton view has no world binding")
        self._notify(
            category=str(category),
            field=str(field),
            world_indices=tuple(int(world) for world in worlds),
            device_row_mask=device_row_mask,
        )

    def _stream(self) -> object | None:
        stream = getattr(self._manager, "stream", None)
        if stream is None:
            stream = getattr(self._manager, "physics_stream", None)
        return stream

    def _notify(
        self,
        *,
        category: str,
        field: str,
        world_indices: tuple[int, ...],
        device_row_mask: object | None = None,
    ) -> None:
        self._require_valid()
        callback = getattr(self._manager, "on_newton_view_write", None)
        if callable(callback):
            kwargs = dict(
                view=self,
                category=category,
                field=field,
                world_indices=world_indices,
            )
            if device_row_mask is not None:
                kwargs["device_row_mask"] = device_row_mask
            callback(**kwargs)


def _required_array(owner: object, name: str, *, category: str) -> object:
    value = getattr(owner, name, None)
    if value is None:
        raise RuntimeError(f"Newton {category} is missing array {name!r}")
    return value


class NewtonArticulationView(_NewtonViewBase):
    """面向兼容层的 articulation view，直接读写 generalized/control owner arrays。"""

    def __init__(
        self,
        manager: object | None,
        *,
        paths: Sequence[str],
        name: str = "newton_runtime_articulation",
        model: object | None = None,
        state: object | None = None,
        control: object | None = None,
        world_indices: Sequence[int] | None = None,
        controllable_dof_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            manager,
            model=model,
            state=state,
            control=control,
        )
        self.name = str(name)
        # finalize 与 model-column upload 在 manager stream 上异步完成。binding 的冷路径
        # ``numpy`` 拓扑读取前先切到该 stream，让 Warp null-stream host copy 正确排在 upload
        # 之后；此同步只发生在 view 构造期。
        with _warp_stream_scope(self._stream()):
            self.binding = NewtonArticulationBinding.from_model(
                self._model,
                paths,
                world_indices=world_indices,
            )
        self.paths = list(self.binding.paths)
        self.dof_names = list(self.binding.dof_names)
        self.num_dofs = self.binding.num_dofs
        self.num_dof = self.num_dofs
        self.count = self.binding.count
        self.max_dofs = self.num_dofs
        self._backend = "warp"
        self._device = str(getattr(self._model, "device", "cpu"))
        self._q = _WarpMatrixBinding(
            self.binding.q_indices,
            device=self._device,
            stream_provider=self._stream,
        )
        self._qd = _WarpMatrixBinding(
            self.binding.qd_indices,
            device=self._device,
            stream_provider=self._stream,
        )
        # 既有 Core 冷状态路径会查找这个 raw tensor-view 属性。回指当前 facade 只满足接口
        # 形状，不创建 extension-owned physics entity，也不复制 model/state ownership。
        self._physics_articulation_view = self
        self._controllable_columns: frozenset[int] | None = None
        self._native_controllable_columns = (
            self._infer_controllable_columns_from_audit()
        )
        if controllable_dof_names is not None:
            self.bind_controllable_dofs(controllable_dof_names)
        elif self._native_controllable_columns is not None:
            self._controllable_columns = self._native_controllable_columns

        import warp as wp

        stiffness = _required_array(self._model, "joint_target_ke", category="model")
        damping = _required_array(self._model, "joint_target_kd", category="model")
        with _warp_stream_scope(self._stream()):
            self._default_stiffness = wp.clone(stiffness)
            self._default_damping = wp.clone(damping)
        self._register_with_manager()

    def __len__(self) -> int:
        self._require_valid()
        return self.count

    def _release_view_resources(self) -> None:
        for name in ("_q", "_qd"):
            mapping = getattr(self, name, None)
            release = getattr(mapping, "release", None)
            if callable(release):
                release()
            setattr(self, name, None)
        self._default_stiffness = None
        self._default_damping = None
        self._physics_articulation_view = None

    def _infer_controllable_columns_from_audit(self) -> frozenset[int] | None:
        # equality follower 由 SolverMuJoCo 唯一执行，不得同时接收 target/effort/gain writer。
        # 用全局 qd index 反查每个 replica 的同一逻辑列，同时验证 replicated rows 映射一致。
        audit = getattr(self._manager, "native_master_follower_audit", None)
        if audit is None:
            audit = getattr(self._manager, "constraint_audit", None)
        bindings = getattr(audit, "bindings", None)
        if bindings is None:
            return None
        follower_indices = {
            int(getattr(item, "follower_qd_index")) for item in bindings
        }
        columns: set[int] = set()
        for column in range(self.num_dofs):
            global_indices = {int(row[column]) for row in self.binding.qd_indices}
            if global_indices.isdisjoint(follower_indices):
                columns.add(column)
            elif not global_indices.issubset(follower_indices):
                raise NewtonViewBindingError(
                    "native follower mapping differs between articulation rows"
                )
        return frozenset(columns)

    @property
    def controllable_dof_names(self) -> tuple[str, ...]:
        self._require_valid()
        if self._controllable_columns is None:
            return ()
        return tuple(
            self.dof_names[index] for index in sorted(self._controllable_columns)
        )

    def bind_controllable_dofs(self, names: Sequence[str]) -> None:
        """一次性冻结 command DOF；后续 follower target 写入必须 fail closed。

        equality follower 由 solver 唯一驱动，若它同时进入 command writer，就会出现两套
        动态执行者。绑定结果因此既是列映射，也是单一执行者能力门禁。
        """

        self._require_valid()

        requested = tuple(str(name) for name in names)
        if len(set(requested)) != len(requested):
            raise ValueError("controllable_dof_names contains duplicates")
        by_name = {name: index for index, name in enumerate(self.dof_names)}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise NewtonViewBindingError(
                f"controllable DOFs are absent from articulation: {missing}"
            )
        columns = frozenset(by_name[name] for name in requested)
        if self._native_controllable_columns is not None:
            forbidden = [
                self.dof_names[index]
                for index in sorted(
                    columns.difference(self._native_controllable_columns)
                )
            ]
            if forbidden:
                raise RuntimeError(
                    "Newton controllable DOFs cannot include "
                    f"native-equality followers: {forbidden}"
                )
        if (
            self._controllable_columns is not None
            and columns != self._controllable_columns
        ):
            raise RuntimeError("Newton controllable DOFs cannot be rebound")
        self._controllable_columns = columns

    def prepare_dof_selection(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        """在 CUDA Graph capture 前预分配 selector、output 与 upload buffers。

        捕获后若某个新 selection 首次触发分配，会令 graph 地址合同失效；生产热路径应先用
        本方法把会出现的 selection 全部固化。
        """

        self._require_valid()

        selection = self._qd.selection(indices, dof_indices)
        for slot in (
            "position_targets",
            "velocity_targets",
            "efforts",
            "joint_qd",
        ):
            self._qd.output(selection, slot)
            self._qd.staging(selection, slot)
        q_selection = self._q.selection(indices, dof_indices)
        self._q.output(q_selection, "joint_q")
        self._q.staging(q_selection, "joint_q")

    def _selected_worlds(self, selection: _WarpSelection) -> tuple[int, ...]:
        return tuple(self.binding.world_indices[row] for row in selection.rows_host)

    def _require_controllable(self, indices: object | None, *, field: str) -> None:
        self._require_valid()
        if self._controllable_columns is None:
            raise RuntimeError(
                "Newton command DOFs must be bound before writing " + field
            )
        selected = _selector_to_host(
            indices,
            count=self.num_dofs,
            label="DOF indices",
            stream=self._stream(),
        )
        forbidden = [
            self.dof_names[index]
            for index in selected
            if index not in self._controllable_columns
        ]
        if forbidden:
            raise RuntimeError(
                f"Newton {field} cannot write native-equality follower DOFs: "
                f"{forbidden}"
            )

    def get_dof_positions(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> object:
        state = self._state()
        return self._q.gather(
            _required_array(state, "joint_q", category="state"),
            rows=indices,
            columns=dof_indices,
            slot="joint_q",
        )

    def get_dof_velocities(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> object:
        state = self._state()
        return self._qd.gather(
            _required_array(state, "joint_qd", category="state"),
            rows=indices,
            columns=dof_indices,
            slot="joint_qd",
        )

    def set_dof_positions(
        self,
        positions: object,
        indices: object | None = None,
        *,
        dof_indices: object | None = None,
    ) -> None:
        state = self._state()
        selection = self._q.scatter(
            _required_array(state, "joint_q", category="state"),
            positions,
            rows=indices,
            columns=dof_indices,
            slot="joint_q",
        )
        self._notify(
            category="state",
            field="joint_q",
            world_indices=self._selected_worlds(selection),
        )

    def set_dof_velocities(
        self,
        velocities: object,
        indices: object | None = None,
        *,
        dof_indices: object | None = None,
    ) -> None:
        state = self._state()
        selection = self._qd.scatter(
            _required_array(state, "joint_qd", category="state"),
            velocities,
            rows=indices,
            columns=dof_indices,
            slot="joint_qd",
        )
        self._notify(
            category="state",
            field="joint_qd",
            world_indices=self._selected_worlds(selection),
        )

    def get_dof_position_targets(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> object:
        control = self._control()
        return self._qd.gather(
            _required_array(control, "joint_target_pos", category="control"),
            rows=indices,
            columns=dof_indices,
            slot="position_targets",
        )

    def set_dof_position_targets(
        self,
        positions: object,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        self._require_controllable(dof_indices, field="position target")
        control = self._control()
        selection = self._qd.scatter(
            _required_array(control, "joint_target_pos", category="control"),
            positions,
            rows=indices,
            columns=dof_indices,
            slot="position_targets",
        )
        self._notify(
            category="control",
            field="joint_target_pos",
            world_indices=self._selected_worlds(selection),
        )

    def get_dof_velocity_targets(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> object:
        control = self._control()
        return self._qd.gather(
            _required_array(control, "joint_target_vel", category="control"),
            rows=indices,
            columns=dof_indices,
            slot="velocity_targets",
        )

    def set_dof_velocity_targets(
        self,
        velocities: object,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        self._require_controllable(dof_indices, field="velocity target")
        control = self._control()
        selection = self._qd.scatter(
            _required_array(control, "joint_target_vel", category="control"),
            velocities,
            rows=indices,
            columns=dof_indices,
            slot="velocity_targets",
        )
        self._notify(
            category="control",
            field="joint_target_vel",
            world_indices=self._selected_worlds(selection),
        )

    def get_dof_efforts(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> object:
        control = self._control()
        return self._qd.gather(
            _required_array(control, "joint_f", category="control"),
            rows=indices,
            columns=dof_indices,
            slot="efforts",
        )

    def set_dof_efforts(
        self,
        efforts: object,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        self._require_controllable(dof_indices, field="effort")
        control = self._control()
        selection = self._qd.scatter(
            _required_array(control, "joint_f", category="control"),
            efforts,
            rows=indices,
            columns=dof_indices,
            slot="efforts",
        )
        self._notify(
            category="control",
            field="joint_f",
            world_indices=self._selected_worlds(selection),
        )

    def get_dof_gains(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> tuple[object, object]:
        self._require_valid()
        return (
            self._qd.gather(
                _required_array(self._model, "joint_target_ke", category="model"),
                rows=indices,
                columns=dof_indices,
                slot="stiffness",
            ),
            self._qd.gather(
                _required_array(self._model, "joint_target_kd", category="model"),
                rows=indices,
                columns=dof_indices,
                slot="damping",
            ),
        )

    def set_dof_gains(
        self,
        stiffnesses: object | None = None,
        dampings: object | None = None,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
        update_default_gains: bool = True,
    ) -> None:
        self._require_valid()
        if stiffnesses is None and dampings is None:
            raise ValueError("at least one Newton gain must be provided")
        self._require_controllable(dof_indices, field="gains")
        selection: _WarpSelection | None = None
        if stiffnesses is not None:
            selection = self._qd.scatter(
                _required_array(self._model, "joint_target_ke", category="model"),
                stiffnesses,
                rows=indices,
                columns=dof_indices,
                slot="stiffness",
            )
            if update_default_gains:
                self._qd.scatter(
                    self._default_stiffness,
                    stiffnesses,
                    rows=indices,
                    columns=dof_indices,
                    slot="default_stiffness",
                )
        if dampings is not None:
            selection = self._qd.scatter(
                _required_array(self._model, "joint_target_kd", category="model"),
                dampings,
                rows=indices,
                columns=dof_indices,
                slot="damping",
            )
            if update_default_gains:
                self._qd.scatter(
                    self._default_damping,
                    dampings,
                    rows=indices,
                    columns=dof_indices,
                    slot="default_damping",
                )
        assert selection is not None
        self._notify(
            category="model",
            field="joint_gains",
            world_indices=self._selected_worlds(selection),
        )

    def get_dof_max_efforts(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> object:
        self._require_valid()
        return self._qd.gather(
            _required_array(self._model, "joint_effort_limit", category="model"),
            rows=indices,
            columns=dof_indices,
            slot="max_efforts",
        )

    def prepare_dof_control_runtime(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        """Preallocate control-mutation buffers without changing model values."""

        self._require_valid()
        self._require_controllable(dof_indices, field="control runtime preparation")
        selection = self._qd.selection(indices, dof_indices)
        for slot in (
            "mode_position_zero",
            "mode_velocity_zero",
            "mode_effort_zero",
            "stiffness",
            "damping",
            "max_efforts",
        ):
            self._qd.staging(selection, slot)
        for slot in (
            "mode_position_stiffness",
            "mode_position_damping",
            "mode_velocity_damping",
        ):
            self._qd.output(selection, slot)

    def set_dof_max_efforts(
        self,
        max_efforts: object,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        self._require_valid()
        selection = self._qd.scatter(
            _required_array(self._model, "joint_effort_limit", category="model"),
            max_efforts,
            rows=indices,
            columns=dof_indices,
            slot="max_efforts",
        )
        self._notify(
            category="model",
            field="joint_effort_limit",
            world_indices=self._selected_worlds(selection),
        )

    def switch_dof_control_mode(
        self,
        mode: str,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        self._require_valid()
        normalized = str(mode).strip().lower()
        if normalized not in {"position", "velocity", "effort"}:
            raise ValueError(
                f"Newton control mode must be position, velocity or effort; got {mode!r}"
            )
        self._require_controllable(dof_indices, field="control mode")
        selection = self._qd.selection(indices, dof_indices)
        zero = self._qd.staging(selection, f"mode_{normalized}_zero")
        stream = self._stream()
        if stream is None:
            zero.zero_()
        else:
            with _warp_stream_scope(stream):
                zero.zero_()
        if normalized == "position":
            stiffness = self._qd.gather(
                self._default_stiffness,
                rows=indices,
                columns=dof_indices,
                slot="mode_position_stiffness",
            )
            damping = self._qd.gather(
                self._default_damping,
                rows=indices,
                columns=dof_indices,
                slot="mode_position_damping",
            )
        elif normalized == "velocity":
            stiffness = zero
            damping = self._qd.gather(
                self._default_damping,
                rows=indices,
                columns=dof_indices,
                slot="mode_velocity_damping",
            )
        else:
            stiffness = zero
            damping = zero
        self.set_dof_gains(
            stiffnesses=stiffness,
            dampings=damping,
            indices=indices,
            dof_indices=dof_indices,
            update_default_gains=False,
        )

    def set_dof_drive_types(
        self,
        mode: str,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> None:
        self._require_valid()
        del indices
        normalized = str(mode).strip().lower()
        if normalized != "force":
            raise RuntimeError("Newton currently supports force effort mode only")
        self._require_controllable(dof_indices, field="effort mode")

    def get_dof_drive_types(
        self,
        *,
        indices: object | None = None,
        dof_indices: object | None = None,
    ) -> list[list[str]]:
        self._require_valid()
        rows = _selector_to_host(
            indices,
            count=self.count,
            label="articulation indices",
            stream=self._stream(),
        )
        columns = _selector_to_host(
            dof_indices,
            count=self.num_dofs,
            label="DOF indices",
            stream=self._stream(),
        )
        return [["force"] * len(columns) for _ in rows]


class NewtonRigidBodyView(_NewtonViewBase):
    """面向兼容层的 maximal rigid-body view。

    读取来自 ``body_q/body_qd``。写入仅限 world-root FREE body，并同时更新对应 generalized
    root ``joint_q/joint_qd``；否则下一次 FK 会用旧 generalized state 覆盖刚写的 body 值。
    """

    def __init__(
        self,
        manager: object | None,
        *,
        paths: Sequence[str],
        name: str = "newton_runtime_rigid_body",
        model: object | None = None,
        state: object | None = None,
        world_indices: Sequence[int] | None = None,
        _binding: NewtonRigidBodyBinding | NewtonDynamicChainBinding | None = None,
    ) -> None:
        super().__init__(
            manager,
            model=model,
            state=state,
            control=None,
        )
        import warp as wp

        self.name = str(name)
        with _warp_stream_scope(self._stream()):
            self.binding = (
                NewtonRigidBodyBinding.from_model(
                    self._model,
                    paths,
                    world_indices=world_indices,
                )
                if _binding is None
                else _binding
            )
        self.paths = list(self.binding.paths)
        self.count = self.binding.count
        self._backend = "warp"
        self._device = str(getattr(self._model, "device", "cpu"))
        self._device_object = wp.get_device(self._device)
        with _warp_stream_scope(self._stream()):
            self._body_indices = wp.array(
                self.binding.body_indices,
                dtype=wp.int32,
                device=self._device_object,
            )
            self._free_q_starts = wp.array(
                self.binding.free_q_starts,
                dtype=wp.int32,
                device=self._device_object,
            )
            self._free_qd_starts = wp.array(
                self.binding.free_qd_starts,
                dtype=wp.int32,
                device=self._device_object,
            )
        self._selections: dict[tuple[int, ...], _WarpSelection] = {}
        self._selection(None)
        self._register_with_manager()

    def __len__(self) -> int:
        self._require_valid()
        return self.count

    def _release_view_resources(self) -> None:
        selections = getattr(self, "_selections", None)
        if selections is not None:
            for selection in selections.values():
                selection.outputs.clear()
                selection.staging.clear()
                selection.rows = None
                selection.columns = None
            selections.clear()
        self._body_indices = None
        self._free_q_starts = None
        self._free_qd_starts = None
        self._device_object = None

    def _selection(self, indices: object | None) -> _WarpSelection:
        import warp as wp

        rows = _selector_to_host(
            indices,
            count=self.count,
            label="rigid body indices",
            stream=self._stream(),
        )
        cached = self._selections.get(rows)
        if cached is not None:
            return cached
        with _warp_stream_scope(self._stream()):
            selection = _WarpSelection(
                rows_host=rows,
                columns_host=(),
                rows=wp.array(rows, dtype=wp.int32, device=self._device_object),
                columns=wp.empty(0, dtype=wp.int32, device=self._device_object),
            )
        self._selections[rows] = selection
        return selection

    def _matrix(
        self,
        selection: _WarpSelection,
        slot: str,
        columns: int,
        *,
        staging: bool,
    ) -> object:
        import warp as wp

        values = selection.staging if staging else selection.outputs
        result = values.get(slot)
        if result is None:
            with _warp_stream_scope(self._stream()):
                result = wp.empty(
                    (len(selection.rows_host), columns),
                    dtype=wp.float32,
                    device=self._device_object,
                )
            values[slot] = result
        return result

    def _fill_matrix(
        self,
        values: object,
        *,
        selection: _WarpSelection,
        slot: str,
        columns: int,
    ) -> object:
        import warp as wp

        rows = len(selection.rows_host)
        staging = self._matrix(selection, slot, columns, staging=True)
        shape = (rows, columns)
        if isinstance(values, wp.array):
            kernels = _load_warp_kernels()
            if values.dtype != wp.float32:
                raise TypeError(f"{slot} Warp input must have dtype wp.float32")
            stream = self._stream()
            with _warp_stream_scope(stream, sync_enter=True):
                if values.ndim == 1:
                    source_columns = int(values.shape[0])
                    if source_columns not in {1, columns}:
                        raise ValueError(
                            f"{slot} cannot broadcast Warp shape {values.shape} to {shape}"
                        )
                    wp.launch(
                        kernels.broadcast_vector,
                        dim=shape,
                        inputs=[values, source_columns],
                        outputs=[staging],
                        device=self._device_object,
                        stream=stream,
                    )
                    return staging
                if values.ndim == 2:
                    source_shape = (int(values.shape[0]), int(values.shape[1]))
                    if source_shape[0] not in {1, rows} or source_shape[1] not in {
                        1,
                        columns,
                    }:
                        raise ValueError(
                            f"{slot} cannot broadcast Warp shape {values.shape} to {shape}"
                        )
                    wp.launch(
                        kernels.broadcast_matrix,
                        dim=shape,
                        inputs=[values, source_shape[0], source_shape[1]],
                        outputs=[staging],
                        device=self._device_object,
                        stream=stream,
                    )
                    return staging
            raise ValueError(f"{slot} Warp input must be one- or two-dimensional")
        try:
            broadcast = np.broadcast_to(np.asarray(values, dtype=np.float32), shape)
        except ValueError as exc:
            raise ValueError(f"{slot} cannot broadcast input to {shape}") from exc
        contiguous = np.ascontiguousarray(broadcast, dtype=np.float32)
        with _warp_stream_scope(self._stream()):
            staging.assign(contiguous)
        return staging

    def _selected_worlds(self, selection: _WarpSelection) -> tuple[int, ...]:
        return tuple(self.binding.world_indices[row] for row in selection.rows_host)

    def _require_writable(self, selection: _WarpSelection) -> None:
        # 非 FREE articulated link 的 maximal pose 不是独立自由度，无法逐 body 无损反演为
        # q/qd。dynamic-chain 另走“完整 env rows → selected IK”路径，普通 view 在此拒绝。
        missing = [
            self.binding.paths[row]
            for row in selection.rows_host
            if self.binding.free_q_starts[row] < 0
            or self.binding.free_qd_starts[row] < 0
        ]
        if missing:
            raise RuntimeError(
                "Newton rigid state writes require world-root FREE bodies; "
                f"non-writable={missing}"
            )

    def get_world_poses(
        self, *, indices: object | None = None
    ) -> tuple[object, object]:
        self._require_valid()
        import warp as wp

        state = self._state()
        selection = self._selection(indices)
        positions = self._matrix(selection, "positions", 3, staging=False)
        orientations = self._matrix(selection, "orientations", 4, staging=False)
        if selection.rows_host:
            kernels = _load_warp_kernels()
            stream = self._stream()
            with _warp_stream_scope(stream, sync_exit=True):
                wp.launch(
                    kernels.gather_pose,
                    dim=len(selection.rows_host),
                    inputs=[
                        _required_array(state, "body_q", category="state"),
                        self._body_indices,
                        selection.rows,
                    ],
                    outputs=[positions, orientations],
                    device=self._device_object,
                    stream=stream,
                )
        return positions, orientations

    def set_world_poses(
        self,
        positions: object | None = None,
        orientations: object | None = None,
        *,
        indices: object | None = None,
    ) -> None:
        self._require_valid()
        import warp as wp

        if positions is None and orientations is None:
            raise ValueError("at least one rigid pose component must be provided")
        state = self._state()
        selection = self._selection(indices)
        self._require_writable(selection)
        position_values = (
            self._matrix(selection, "pose_position_unused", 3, staging=True)
            if positions is None
            else self._fill_matrix(
                positions,
                selection=selection,
                slot="pose_positions",
                columns=3,
            )
        )
        orientation_values = (
            self._matrix(selection, "pose_orientation_unused", 4, staging=True)
            if orientations is None
            else self._fill_matrix(
                orientations,
                selection=selection,
                slot="pose_orientations",
                columns=4,
            )
        )
        if orientations is not None and not isinstance(orientations, wp.array):
            host = np.broadcast_to(
                np.asarray(orientations, dtype=np.float32),
                (len(selection.rows_host), 4),
            )
            norms = np.linalg.norm(host, axis=1)
            if not np.all(np.isfinite(host)) or np.any(norms <= 0.0):
                raise ValueError(
                    "rigid orientations must contain finite nonzero quaternions"
                )
        if positions is not None and not isinstance(positions, wp.array):
            host = np.broadcast_to(
                np.asarray(positions, dtype=np.float32),
                (len(selection.rows_host), 3),
            )
            if not np.all(np.isfinite(host)):
                raise ValueError("rigid positions must be finite")
        if selection.rows_host:
            kernels = _load_warp_kernels()
            wp.launch(
                kernels.scatter_pose,
                dim=len(selection.rows_host),
                inputs=[
                    position_values,
                    orientation_values,
                    int(positions is not None),
                    int(orientations is not None),
                    self._body_indices,
                    self._free_q_starts,
                    selection.rows,
                ],
                outputs=[
                    _required_array(state, "body_q", category="state"),
                    _required_array(state, "joint_q", category="state"),
                ],
                device=self._device_object,
                stream=self._stream(),
            )
        self._notify(
            category="state",
            field="body_q",
            world_indices=self._selected_worlds(selection),
        )

    def get_velocities(self, *, indices: object | None = None) -> tuple[object, object]:
        self._require_valid()
        import warp as wp

        state = self._state()
        selection = self._selection(indices)
        linear = self._matrix(selection, "linear_velocity", 3, staging=False)
        angular = self._matrix(selection, "angular_velocity", 3, staging=False)
        if selection.rows_host:
            kernels = _load_warp_kernels()
            stream = self._stream()
            with _warp_stream_scope(stream, sync_exit=True):
                wp.launch(
                    kernels.gather_velocity,
                    dim=len(selection.rows_host),
                    inputs=[
                        _required_array(state, "body_qd", category="state"),
                        self._body_indices,
                        selection.rows,
                    ],
                    outputs=[linear, angular],
                    device=self._device_object,
                    stream=stream,
                )
        return linear, angular

    def set_velocities(
        self,
        linear: object | None = None,
        angular: object | None = None,
        *,
        indices: object | None = None,
    ) -> None:
        self._require_valid()
        import warp as wp

        if linear is None and angular is None:
            raise ValueError("at least one rigid velocity component must be provided")
        state = self._state()
        selection = self._selection(indices)
        self._require_writable(selection)
        linear_values = (
            self._matrix(selection, "linear_velocity_unused", 3, staging=True)
            if linear is None
            else self._fill_matrix(
                linear,
                selection=selection,
                slot="linear_velocity",
                columns=3,
            )
        )
        angular_values = (
            self._matrix(selection, "angular_velocity_unused", 3, staging=True)
            if angular is None
            else self._fill_matrix(
                angular,
                selection=selection,
                slot="angular_velocity",
                columns=3,
            )
        )
        for label, values in (("linear", linear), ("angular", angular)):
            if values is None or isinstance(values, wp.array):
                continue
            host = np.asarray(values, dtype=np.float32)
            if not np.all(np.isfinite(host)):
                raise ValueError(f"rigid {label} velocities must be finite")
        if selection.rows_host:
            kernels = _load_warp_kernels()
            wp.launch(
                kernels.scatter_velocity,
                dim=len(selection.rows_host),
                inputs=[
                    linear_values,
                    angular_values,
                    int(linear is not None),
                    int(angular is not None),
                    self._body_indices,
                    self._free_qd_starts,
                    selection.rows,
                ],
                outputs=[
                    _required_array(state, "body_qd", category="state"),
                    _required_array(state, "joint_qd", category="state"),
                ],
                device=self._device_object,
                stream=self._stream(),
            )
        self._notify(
            category="state",
            field="body_qd",
            world_indices=self._selected_worlds(selection),
        )

    def get_linear_velocities(self, *, indices: object | None = None) -> object:
        return self.get_velocities(indices=indices)[0]

    def get_angular_velocities(self, *, indices: object | None = None) -> object:
        return self.get_velocities(indices=indices)[1]

    def set_linear_velocities(
        self, values: object, *, indices: object | None = None
    ) -> None:
        self.set_velocities(linear=values, indices=indices)

    def set_angular_velocities(
        self, values: object, *, indices: object | None = None
    ) -> None:
        self.set_velocities(angular=values, indices=indices)


class NewtonDynamicChainView(NewtonRigidBodyView):
    """外观为 rigid view、权威状态为 generalized ``q/qd`` 的 dynamic-chain view。

    为兼容 portable snapshot schema，读取仍可 gather maximal body state；精确 Newton
    snapshot 则同时携带稳定 topology signature 和完整 q/qd。restore 必须以环境为原子单位：
    generalized 路径直接 scatter q/qd 后做 selected FK；旧 maximal 路径要求每个 selected env
    的全部 body row，再用 selected IK 重建 q/qd、用 FK 回写约束一致的 body state。

    这种区分很关键：接触求解后 maximal twist 可能因约束流形投影出现小差异，但 q/qd 才是
    clone/rollback 必须精确一致的 owner state。
    """

    def __init__(
        self,
        manager: object | None,
        *,
        articulation_paths: Sequence[str],
        body_paths_by_env: Sequence[Sequence[str]],
        name: str = "newton_runtime_dynamic_chain",
        model: object | None = None,
        state: object | None = None,
        world_indices: Sequence[int] | None = None,
    ) -> None:
        resolved_model = model if model is not None else getattr(manager, "model", None)
        if resolved_model is None:
            raise ValueError("Newton dynamic-chain view requires a finalized model")
        stream = getattr(manager, "stream", None)
        with _warp_stream_scope(stream):
            binding = NewtonDynamicChainBinding.from_model(
                resolved_model,
                articulation_paths=articulation_paths,
                body_paths_by_env=body_paths_by_env,
                world_indices=world_indices,
            )
        super().__init__(
            manager,
            paths=binding.paths,
            name=name,
            model=resolved_model,
            state=state,
            world_indices=None,
            _binding=binding,
        )
        self.binding = binding
        self.articulation_paths = list(binding.articulation_paths)
        self.body_paths_by_env = binding.body_paths_by_env
        self.env_count = binding.env_count
        self.body_count = binding.body_count
        self._ik_indices: dict[tuple[int, ...], object] = {}
        self._generalized_q_mapping = _WarpMatrixBinding(
            binding.q_indices,
            device=self._device_object,
            stream_provider=self._stream,
        )
        self._generalized_qd_mapping = _WarpMatrixBinding(
            binding.qd_indices,
            device=self._device_object,
            stream_provider=self._stream,
        )

    @property
    def q_coordinate_names(self) -> tuple[str, ...]:
        return self.binding.q_coordinate_names

    @property
    def qd_coordinate_names(self) -> tuple[str, ...]:
        return self.binding.qd_coordinate_names

    @property
    def generalized_coordinate_signature(self) -> tuple[str, ...]:
        return self.binding.coordinate_signature

    @property
    def generalized_world_translation_q_indices(self) -> tuple[int, ...]:
        """返回以 world 坐标表达 FREE-root translation 的 q 列。"""

        return self.binding.world_translation_q_indices

    def _release_view_resources(self) -> None:
        q_mapping = getattr(self, "_generalized_q_mapping", None)
        if q_mapping is not None:
            q_mapping.release()
        qd_mapping = getattr(self, "_generalized_qd_mapping", None)
        if qd_mapping is not None:
            qd_mapping.release()
        self._generalized_q_mapping = None
        self._generalized_qd_mapping = None
        super()._release_view_resources()
        indices = getattr(self, "_ik_indices", None)
        if indices is not None:
            indices.clear()

    def _selected_worlds(self, selection: _WarpSelection) -> tuple[int, ...]:
        env_rows = self._complete_env_rows(selection)
        return tuple(self.binding.world_indices[row] for row in env_rows)

    def _require_writable(self, selection: _WarpSelection) -> None:
        del selection
        raise RuntimeError(
            "Newton dynamic-chain body state must be restored atomically "
            "through generalized coordinates"
        )

    def _complete_env_rows(self, selection: _WarpSelection) -> tuple[int, ...]:
        """验证 selection 是若干完整、env-major 且不交错的 chain body rows。

        partial body selection 无法唯一反演整条 articulated chain；即使 eval_ik 接受这些
        maximal 值，也会让未提供 body 的旧状态参与求解，导致 restore 依赖恢复前状态。
        """

        rows = selection.rows_host
        if not rows:
            return ()
        selected_envs: list[int] = []
        for row in rows:
            env_row = int(row) // self.body_count
            if env_row not in selected_envs:
                selected_envs.append(env_row)
        expected = tuple(
            env_row * self.body_count + body_row
            for env_row in selected_envs
            for body_row in range(self.body_count)
        )
        if rows != expected:
            raise RuntimeError(
                "Newton dynamic-chain restore requires complete env-major "
                f"body rows: actual={rows}, expected={expected}"
            )
        return tuple(selected_envs)

    def _articulation_indices_for(self, env_rows: tuple[int, ...]) -> object:
        # selected IK/FK 接受 articulation index，不接受 world mask；这里把 env row 映射到
        # finalized model index 并缓存 GPU 数组，保证局部 restore 不重算未选 world。
        import warp as wp

        cached = self._ik_indices.get(env_rows)
        if cached is not None:
            return cached
        values = tuple(self.binding.articulation_indices[row] for row in env_rows)
        with _warp_stream_scope(self._stream()):
            cached = wp.array(
                values,
                dtype=wp.int32,
                device=self._device_object,
            )
        self._ik_indices[env_rows] = cached
        return cached

    def get_generalized_state(
        self,
        *,
        indices: object | None = None,
    ) -> tuple[object, object]:
        """按环境 gather 权威 q/qd；返回矩阵列由稳定 coordinate identity 描述。"""

        self._require_valid()
        state = self._state()
        return (
            self._generalized_q_mapping.gather(
                _required_array(state, "joint_q", category="state"),
                rows=indices,
                columns=None,
                slot="dynamic_chain_generalized_q",
            ),
            self._generalized_qd_mapping.gather(
                _required_array(state, "joint_qd", category="state"),
                rows=indices,
                columns=None,
                slot="dynamic_chain_generalized_qd",
            ),
        )

    def _validated_generalized_state(
        self,
        *,
        signature: Sequence[str],
        q_names: Sequence[str],
        qd_names: Sequence[str],
        q: object,
        qd: object,
        indices: object | None,
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        """在写 owner state 前完成全部 ABI identity、shape 与有限值检查。

        先完整 preflight 再 scatter，保证 signature 或 qd 失败时 q 也尚未被部分写入，便于
        clone/restore/rollback 保持事务式语义。
        """

        self._require_valid()
        env_rows = _selector_to_host(
            indices,
            count=self.env_count,
            label="dynamic-chain environment indices",
            stream=self._stream(),
        )
        _strict_identity(
            signature,
            expected=self.generalized_coordinate_signature,
            label="generalized coordinate signature",
            unique=False,
        )
        _strict_identity(
            q_names,
            expected=self.q_coordinate_names,
            label="generalized q coordinate names",
            unique=True,
        )
        _strict_identity(
            qd_names,
            expected=self.qd_coordinate_names,
            label="generalized qd coordinate names",
            unique=True,
        )
        q_values = _strict_finite_matrix(
            q,
            shape=(len(env_rows), len(self.q_coordinate_names)),
            label="generalized q",
            stream=self._stream(),
        )
        qd_values = _strict_finite_matrix(
            qd,
            shape=(len(env_rows), len(self.qd_coordinate_names)),
            label="generalized qd",
            stream=self._stream(),
        )
        return env_rows, q_values, qd_values

    def validate_generalized_state(
        self,
        *,
        signature: Sequence[str],
        q_names: Sequence[str],
        qd_names: Sequence[str],
        q: object,
        qd: object,
        indices: object | None = None,
    ) -> None:
        """只预检批量 owner-state restore，不写入任何 Newton buffer。

        snapshot 恢复先验证 topology signature、字段名、shape 与有限性，随后才允许统一
        提交 q/qd，避免已知不兼容数据造成部分 world 被改写。
        """

        self._validated_generalized_state(
            signature=signature,
            q_names=q_names,
            qd_names=qd_names,
            q=q,
            qd=qd,
            indices=indices,
        )

    def set_generalized_state(
        self,
        *,
        signature: Sequence[str],
        q_names: Sequence[str],
        qd_names: Sequence[str],
        q: object,
        qd: object,
        indices: object | None = None,
    ) -> None:
        """精确恢复 owner q/qd，并仅通过 selected FK 派生相应 body rows。

        这里不调用 IK：snapshot 已经保存了 Newton 自己的 generalized coordinates，再从
        maximal state 反演会引入不必要的投影误差。写通知随后让 manager 在统一冷边界投影
        JOINT equality，并使 follower 状态与资产多项式一致。
        """

        import newton

        env_rows, q_values, qd_values = self._validated_generalized_state(
            signature=signature,
            q_names=q_names,
            qd_names=qd_names,
            q=q,
            qd=qd,
            indices=indices,
        )
        if not env_rows:
            return
        state = self._state()
        stream = self._stream()
        with _warp_stream_scope(stream):
            self._generalized_q_mapping.scatter(
                _required_array(state, "joint_q", category="state"),
                q_values,
                rows=env_rows,
                columns=None,
                slot="dynamic_chain_generalized_q_restore",
            )
            self._generalized_qd_mapping.scatter(
                _required_array(state, "joint_qd", category="state"),
                qd_values,
                rows=env_rows,
                columns=None,
                slot="dynamic_chain_generalized_qd_restore",
            )
            newton.eval_fk(
                self._model,
                _required_array(state, "joint_q", category="state"),
                _required_array(state, "joint_qd", category="state"),
                state,
                indices=self._articulation_indices_for(env_rows),
            )
        self._notify(
            category="state",
            field="joint_q/joint_qd",
            world_indices=tuple(self.binding.world_indices[row] for row in env_rows),
        )

    def set_articulated_body_states(
        self,
        *,
        positions: object,
        orientations: object,
        velocities: object,
        indices: object,
    ) -> None:
        """兼容旧 maximal snapshot：恢复完整 selected chains，并经 selected IK 提交。

        scatter_pose/velocity 先写完整 body rows，IK 把它们转换为 generalized owner state，
        FK 再生成一份自洽 maximal 表示。两者在同一 owner stream 顺序执行，未选 articulation
        的 q/qd 与 body state 均不会被触碰。
        """

        self._require_valid()
        import newton
        import warp as wp

        selection = self._selection(indices)
        env_rows = self._complete_env_rows(selection)
        row_count = len(selection.rows_host)
        if row_count == 0:
            return
        position_host = np.asarray(positions, dtype=np.float32).reshape(row_count, 3)
        orientation_host = np.asarray(orientations, dtype=np.float32).reshape(
            row_count, 4
        )
        velocity_host = np.asarray(velocities, dtype=np.float32).reshape(row_count, 6)
        norms = np.linalg.norm(orientation_host, axis=1)
        if (
            not np.all(np.isfinite(position_host))
            or not np.all(np.isfinite(orientation_host))
            or not np.all(np.isfinite(velocity_host))
            or np.any(norms <= 0.0)
        ):
            raise ValueError(
                "Newton dynamic-chain state must be finite with nonzero quaternions"
            )
        orientation_host = orientation_host / norms[:, None]
        position_values = self._fill_matrix(
            position_host,
            selection=selection,
            slot="chain_positions",
            columns=3,
        )
        orientation_values = self._fill_matrix(
            orientation_host,
            selection=selection,
            slot="chain_orientations",
            columns=4,
        )
        linear_values = self._fill_matrix(
            velocity_host[:, :3],
            selection=selection,
            slot="chain_linear_velocities",
            columns=3,
        )
        angular_values = self._fill_matrix(
            velocity_host[:, 3:],
            selection=selection,
            slot="chain_angular_velocities",
            columns=3,
        )
        state = self._state()
        kernels = _load_warp_kernels()
        articulation_indices = self._articulation_indices_for(env_rows)
        stream = self._stream()
        with _warp_stream_scope(stream):
            wp.launch(
                kernels.scatter_pose,
                dim=row_count,
                inputs=[
                    position_values,
                    orientation_values,
                    1,
                    1,
                    self._body_indices,
                    self._free_q_starts,
                    selection.rows,
                ],
                outputs=[
                    _required_array(state, "body_q", category="state"),
                    _required_array(state, "joint_q", category="state"),
                ],
                device=self._device_object,
                stream=stream,
            )
            wp.launch(
                kernels.scatter_velocity,
                dim=row_count,
                inputs=[
                    linear_values,
                    angular_values,
                    1,
                    1,
                    self._body_indices,
                    self._free_qd_starts,
                    selection.rows,
                ],
                outputs=[
                    _required_array(state, "body_qd", category="state"),
                    _required_array(state, "joint_qd", category="state"),
                ],
                device=self._device_object,
                stream=stream,
            )
            newton.eval_ik(
                self._model,
                state,
                _required_array(state, "joint_q", category="state"),
                _required_array(state, "joint_qd", category="state"),
                indices=articulation_indices,
            )
            newton.eval_fk(
                self._model,
                _required_array(state, "joint_q", category="state"),
                _required_array(state, "joint_qd", category="state"),
                state,
                indices=articulation_indices,
            )
        self._notify(
            category="state",
            field="joint_q/joint_qd",
            world_indices=tuple(self.binding.world_indices[row] for row in env_rows),
        )


__all__ = [
    "NewtonArticulationBinding",
    "NewtonArticulationView",
    "NewtonDynamicChainBinding",
    "NewtonDynamicChainView",
    "NewtonRigidBodyBinding",
    "NewtonRigidBodyView",
    "NewtonViewBindingError",
]
