"""把 Newton runtime 的状态显式同步到 renderer-facing USD。

Newton state 是唯一权威状态；USD 在这里仅是渲染镜像。``env_0`` 保留导入后的
prototype，其余 world 用 internal reference 物化为 render clone；每次 render 前，将 Newton
``body_q`` 的 world transform 写入对应 prim。此路径没有 Isaac physics extension，也没有
逐帧 stage-update callback，因此不会出现 renderer 反向改写物理状态或隐式推进时间。

``body_q`` 是 generalized ``joint_q/joint_qd`` 经 FK/solver 得到的 maximal body 表示，适合
直接发布给 renderer，但不能作为 dynamic-chain snapshot 的权威恢复格式；后者仍由 view
保存和恢复 generalized state。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_WORLD_TRANSFORM_OP_SUFFIX = "newtonRenderWorld"
_WORLD_TRANSFORM_OP_NAME = f"xformOp:transform:{_WORLD_TRANSFORM_OP_SUFFIX}"


@dataclass(frozen=True)
class _BodyBinding:
    """固定 model body index 到 USD transform op，避免热渲染路径重新遍历 stage。"""

    body_index: int
    world_index: int
    prim_path: str
    transform_op: object


class NewtonRenderSync:
    """把 finalized Newton model 精确映射到 renderer 可见 USD prim。

    prototype 的 transform 拓扑必须在 Newton 解析/finalize 前冻结。初始化时只创建尚
    不存在的 clone root，并复用、审计 prototype 与 clone body 已有的 matrix op；不会对
    renderer 已知 prim 清空或改写 ``xformOpOrder``。``sync`` 热路径只做一次
    device→host 快照读取和缓存 op 写入，不再解析 label 或创建 USD property。
    """

    def __init__(
        self,
        *,
        stage: object,
        model: object,
        prototype_root: str,
        destination_roots: tuple[str, ...],
        world_transforms: tuple[object, ...],
        visible_world_indices: tuple[int, ...] | None = None,
    ) -> None:
        self._stage = stage
        self._prototype_root = str(prototype_root).rstrip("/")
        self._destination_roots = tuple(
            str(path).rstrip("/") for path in destination_roots
        )
        if not self._prototype_root or not self._destination_roots:
            raise ValueError("Newton render roots must be non-empty")
        if self._destination_roots[0] != self._prototype_root:
            raise ValueError("Newton render prototype must be destination world zero")
        self._world_transforms = _host_world_transforms(world_transforms)
        if self._world_transforms.shape[0] != len(self._destination_roots):
            raise ValueError(
                "Newton render world transforms and destination roots "
                "must have equal length"
            )
        if visible_world_indices is None:
            visible_world_indices = tuple(range(len(self._destination_roots)))
        if not isinstance(visible_world_indices, tuple) or not visible_world_indices:
            raise TypeError("visible_world_indices must be a non-empty tuple or None")
        if any(
            type(index) is not int or index < 0 or index >= len(self._destination_roots)
            for index in visible_world_indices
        ):
            raise ValueError(
                "visible_world_indices must reference existing render worlds"
            )
        if len(set(visible_world_indices)) != len(visible_world_indices):
            raise ValueError("visible_world_indices must be unique")
        self._visible_world_indices = visible_world_indices
        self._visible_world_set = frozenset(visible_world_indices)
        self._materialize_render_clones()
        self._author_world_visibility()
        self._validate_world_roots()
        self._bindings = self._bind_model_bodies(model)
        self._closed = False
        self._sync_count = 0

    @property
    def body_count(self) -> int:
        return len(self._bindings)

    @property
    def sync_count(self) -> int:
        return self._sync_count

    def sync(self, body_q: object) -> None:
        """把一份已同步的 host snapshot 发布到 renderer-facing USD。

        CUDA 调用者必须先同步 Newton owner stream；CPU execution 没有 stream。本类不自行
        触碰 physics stream，以免每个 camera/resource 各自制造一次同步屏障。多个 render
        product 因而共享同一快照。
        """

        if self._closed:
            raise RuntimeError("Newton render sync is closed")
        values = _host_transforms(body_q)
        if values.shape[0] <= max(item.body_index for item in self._bindings):
            raise RuntimeError(
                "Newton body transform array is shorter than the render binding"
            )
        for binding in self._bindings:
            _set_world_transform(binding.transform_op, values[binding.body_index])
        self._sync_count += 1

    def close(self) -> None:
        """断开 cached op 与 stage 强引用；不删除 session 拥有的 USD prim。"""

        self._closed = True
        self._bindings = ()
        self._stage = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "body_count": self.body_count,
            "world_count": len(self._destination_roots),
            "visible_world_indices": list(self._visible_world_indices),
            "sync_count": self._sync_count,
            "representation": "usd_internal_reference_world_xforms",
        }

    def _materialize_render_clones(self) -> None:
        from pxr import Sdf, UsdGeom

        source = self._stage.GetPrimAtPath(Sdf.Path(self._prototype_root))
        if source is None or not bool(source.IsValid()):
            raise RuntimeError(
                f"Newton render prototype is missing: {self._prototype_root}"
            )
        for world in self._visible_world_indices:
            destination = self._destination_roots[world]
            if destination == self._prototype_root:
                continue
            existing = self._stage.GetPrimAtPath(Sdf.Path(destination))
            if existing is not None and bool(existing.IsValid()):
                raise RuntimeError(
                    "Newton render destination already exists; refusing to "
                    f"overlay an ambiguous clone: {destination}"
                )
            xform = UsdGeom.Xform.Define(self._stage, destination)
            # clone root 尚未引用 prototype，此刻写入最终 op 拓扑不会让 Hydra 对已
            # population 的引用子树做结构性重同步。之后只允许更新矩阵值。
            _author_canonical_world_transform(
                UsdGeom.Xformable(xform.GetPrim()),
                self._world_transforms[world],
            )
            prim = xform.GetPrim()
            # internal reference 只复用 prototype 的渲染/层级描述，不创建 physics owner。
            # destination 已存在时拒绝 overlay，避免同一路径同时来自用户 clone 和本同步器。
            prim.GetReferences().AddInternalReference(self._prototype_root)
            if not bool(self._stage.GetPrimAtPath(Sdf.Path(destination)).IsValid()):
                raise RuntimeError(
                    f"Newton render clone did not compose: {destination}"
                )

    def _author_world_visibility(self) -> None:
        """隐藏未选中的 source，并用 destination 本地意见保证所选 clone 可见。"""

        from pxr import Sdf, UsdGeom

        source = self._stage.GetPrimAtPath(Sdf.Path(self._prototype_root))
        source_imageable = UsdGeom.Imageable(source)
        if 0 in self._visible_world_set:
            source_imageable.MakeVisible()
        else:
            source_imageable.MakeInvisible()
        for world in self._visible_world_indices:
            prim = self._stage.GetPrimAtPath(Sdf.Path(self._destination_roots[world]))
            if prim is None or not bool(prim.IsValid()):
                raise RuntimeError(
                    "Newton selected render world is missing: "
                    f"{self._destination_roots[world]}"
                )
            # clone 的本地 inherited 意见强于 internal reference 中 prototype 的
            # invisible 意见，因此选择 env_1+ 时不会连同 source 一起被隐藏。
            UsdGeom.Imageable(prim).MakeVisible()

    def _validate_world_roots(self) -> None:
        from pxr import Sdf, UsdGeom

        for world in self._visible_world_indices:
            path = self._destination_roots[world]
            prim = self._stage.GetPrimAtPath(Sdf.Path(path))
            if prim is None or not bool(prim.IsValid()):
                raise RuntimeError(f"Newton render world root is missing: {path}")
            op = _require_canonical_world_transform_op(
                UsdGeom.Xformable(prim),
                prim_path=path,
            )
            # prototype 在 prepare 阶段已取得最终值，clone root 则在添加 reference 前
            # author。这里只原地更新属性值，不触碰 property/op-order 拓扑。
            _set_world_transform(op, self._world_transforms[world])

    def _bind_model_bodies(self, model: object) -> tuple[_BodyBinding, ...]:
        from pxr import Sdf, UsdGeom

        labels = tuple(str(value) for value in getattr(model, "body_label", ()))
        worlds = _host_ints(getattr(model, "body_world", ()))
        if len(labels) != len(worlds):
            raise RuntimeError(
                "Newton body label/world columns have different lengths for render sync"
            )
        bindings: list[_BodyBinding] = []
        seen_paths: set[str] = set()
        missing: list[str] = []
        duplicate: list[str] = []
        for body_index, (path, world) in enumerate(zip(labels, worlds, strict=True)):
            # world=-1 是共享/global body，不属于任何 env 的 render clone；world-local body
            # 则必须位于该 world destination root 下，不能靠 basename 猜测对应 prim。
            if int(world) < 0:
                continue
            if int(world) >= len(self._destination_roots):
                raise RuntimeError(
                    "Newton body references an invalid render world: "
                    f"body={body_index}, world={int(world)}"
                )
            if int(world) not in self._visible_world_set:
                continue
            expected_root = self._destination_roots[int(world)]
            if path != expected_root and not path.startswith(expected_root + "/"):
                raise RuntimeError(
                    "Newton body label is outside its render world root: "
                    f"body={body_index}, label={path!r}, root={expected_root!r}"
                )
            if path in seen_paths:
                duplicate.append(path)
                continue
            seen_paths.add(path)
            prim = self._stage.GetPrimAtPath(Sdf.Path(path))
            if (
                prim is None
                or not bool(prim.IsValid())
                or not bool(prim.IsA(UsdGeom.Xformable))
            ):
                missing.append(path)
                continue
            bindings.append(
                _BodyBinding(
                    body_index=body_index,
                    world_index=int(world),
                    prim_path=path,
                    transform_op=_require_canonical_world_transform_op(
                        UsdGeom.Xformable(prim),
                        prim_path=path,
                    ),
                )
            )
        if duplicate or missing:
            raise RuntimeError(
                "Newton render body mapping must be exact and unique: "
                f"duplicate={sorted(set(duplicate))}, missing={sorted(set(missing))}"
            )
        if not bindings:
            raise RuntimeError("Newton render sync found no world-local bodies")
        return tuple(bindings)


def _host_transforms(value: object) -> np.ndarray:
    candidate = value
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    # 后续会原地归一化 quaternion。必须先取得 owned copy；否则 CPU 调用方直接传入
    # float64 NumPy view 时，renderer 会反向改写 Newton/快照源状态。
    result = np.array(candidate, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 7:
        raise RuntimeError(
            f"Newton body transforms must have shape (N, 7), got {result.shape}"
        )
    if not bool(np.isfinite(result).all()):
        raise RuntimeError("Newton body transforms contain non-finite values")
    norms = np.linalg.norm(result[:, 3:7], axis=1)
    if bool(np.any(norms <= 0.0)):
        raise RuntimeError("Newton body transform contains a zero quaternion")
    result[:, 3:7] /= norms[:, None]
    return result


def _host_world_transforms(values: tuple[object, ...]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for value in values:
        candidate = value
        numpy_method = getattr(candidate, "numpy", None)
        if callable(numpy_method):
            candidate = numpy_method()
        rows.append(np.asarray(candidate, dtype=np.float64).reshape(7))
    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    return _host_transforms(np.stack(rows, axis=0))


def _host_ints(value: object) -> np.ndarray:
    candidate = value
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return np.asarray(candidate, dtype=np.int32).reshape(-1)


def prepare_newton_render_stage(
    *,
    stage: object,
    prototype_root: str,
    world_transform: object,
) -> tuple[str, ...]:
    """冻结 prototype root，并审计资产 author 阶段已经准备好的刚体。

    manager 调用本函数时 Hydra 可能已经追踪资产 prim，因此这里只为 composition root
    写最终 world transform；刚体必须更早由 prepare_newton_render_subtree 完成。
    任何遗漏都直接失败，不能在这个晚边界静默更换 body op order。
    """

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    root_path = str(prototype_root).rstrip("/")
    if not root_path:
        raise ValueError("Newton render prototype root must be non-empty")
    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if (
        root is None
        or not bool(root.IsValid())
        or not bool(root.IsA(UsdGeom.Xformable))
    ):
        raise RuntimeError(f"Newton render prototype is not Xformable: {root_path}")

    body_prims = tuple(
        prim
        for prim in Usd.PrimRange(root)
        if prim != root
        and bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        and bool(prim.IsA(UsdGeom.Xformable))
    )
    world_value = _host_world_transforms((world_transform,))[0]
    _author_canonical_world_transform(UsdGeom.Xformable(root), world_value)
    unprepared = tuple(
        str(prim.GetPath())
        for prim in body_prims
        if _canonical_world_transform_op(UsdGeom.Xformable(prim)) is None
    )
    if unprepared:
        raise RuntimeError(
            "Newton render body topology must be prepared during asset authoring: "
            f"unprepared={list(unprepared)!r}"
        )
    return tuple(str(prim.GetPath()) for prim in body_prims)


def author_newton_render_root_pose(
    *,
    stage: object,
    root_path: str,
    pose: object,
) -> object:
    """在 reference/import 暴露资产前直接写入最终 Newton render root op。

    render-enabled Newton 资产不能先发布 ``translate/rotateXYZ``，再把它们替换为
    renderer 使用的 matrix op；Hydra 可能异步消费前一份 ``xformOpOrder``，最终形成
    stale-op 警告。本函数让调用方在添加 reference 前就建立唯一 canonical topology，
    后续 root pose 和逐帧同步都只更新 matrix value。
    """

    from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz
    from pxr import Sdf, UsdGeom

    path = str(root_path).rstrip("/")
    if not path:
        raise ValueError("Newton render root path must be non-empty")
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if (
        prim is None
        or not bool(prim.IsValid())
        or not bool(prim.IsA(UsdGeom.Xformable))
    ):
        raise RuntimeError(f"Newton render root is not Xformable: {path}")
    xyz = np.asarray(getattr(pose, "xyz"), dtype=np.float64).reshape(-1)
    rpy = np.asarray(getattr(pose, "rpy"), dtype=np.float64).reshape(-1)
    if xyz.size != 3 or not bool(np.isfinite(xyz).all()):
        raise ValueError("Newton render root pose xyz must contain 3 finite values")
    if rpy.size != 3 or not bool(np.isfinite(rpy).all()):
        raise ValueError("Newton render root pose rpy must contain 3 finite values")
    quat_wxyz = rpy_xyz_to_quat_wxyz(rpy)
    value = np.asarray(
        [*xyz, quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    return _author_canonical_world_transform(UsdGeom.Xformable(prim), value)


def prepare_newton_render_subtree(
    *,
    stage: object,
    subtree_root: str,
) -> tuple[str, ...]:
    """在单个资产完成 reference/import 与 root pose 后冻结全部刚体拓扑。

    Newton body_q 提供世界坐标变换，所以 body 使用 reset stack 的单一 matrix op。
    全部 world matrix 必须先由同一个 cache 采样，再开始 author；这样嵌套机器人 link、
    rigid object 与 dynamic-chain segment 不会因 parent 先被 reset 而改变 child 位姿。
    本函数幂等：重复调用只更新 matrix 值，不重复写 xformOpOrder。
    """

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    root_path = str(subtree_root).rstrip("/")
    if not root_path:
        raise ValueError("Newton render subtree root must be non-empty")
    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if root is None or not bool(root.IsValid()):
        raise RuntimeError(f"Newton render asset subtree is missing: {root_path}")
    body_prims = tuple(
        prim
        for prim in Usd.PrimRange(root)
        if bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        and bool(prim.IsA(UsdGeom.Xformable))
    )
    cache = UsdGeom.XformCache()
    body_world_matrices = tuple(
        (prim, cache.GetLocalToWorldTransform(prim)) for prim in body_prims
    )
    for prim, matrix in body_world_matrices:
        _author_canonical_world_matrix(UsdGeom.Xformable(prim), matrix)
    return tuple(str(prim.GetPath()) for prim in body_prims)


def _author_canonical_world_transform(xform: object, value: np.ndarray) -> None:
    _author_canonical_world_matrix(xform, _matrix_from_transform(value))


def _author_canonical_world_matrix(xform: object, matrix: object) -> object:
    """在冷 author 阶段写入唯一 matrix op，且不清空原 op order。"""

    from pxr import Sdf, UsdGeom

    attr = xform.GetPrim().GetAttribute(_WORLD_TRANSFORM_OP_NAME)
    if bool(attr.IsValid()):
        op = UsdGeom.XformOp(attr)
        if (
            op.GetOpType() != UsdGeom.XformOp.TypeTransform
            or op.GetPrecision() != UsdGeom.XformOp.PrecisionDouble
        ):
            raise RuntimeError(
                "Newton render transform attribute has incompatible type or "
                f"precision: {xform.GetPrim().GetPath()}"
            )
    else:
        # 直接创建 matrix attr，最后由 SetXformOpOrder 一次发布完整 topology；
        # AddTransformOp 会先把未 reset 的中间顺序写入 stage，造成额外 Hydra notice。
        attr = xform.GetPrim().CreateAttribute(
            _WORLD_TRANSFORM_OP_NAME,
            Sdf.ValueTypeNames.Matrix4d,
            custom=False,
        )
        op = UsdGeom.XformOp(attr)
    op.Set(matrix)
    # body_q 是 world transform。reset stack 阻止嵌套 MJCF body 再乘 parent。只有首次
    # prepare 才写 op order；重复 prepare 只更新值，避免制造无意义的 Hydra topology notice。
    if _canonical_world_transform_op(xform) is None and not bool(
        xform.SetXformOpOrder([op], resetXformStack=True)
    ):
        raise RuntimeError(
            f"Cannot freeze Newton render xform topology: {xform.GetPrim().GetPath()}"
        )
    return op


def _require_canonical_world_transform_op(
    xform: object,
    *,
    prim_path: str,
) -> object:
    """只验证并返回 prepare 阶段创建的 op，运行期绝不修补拓扑。"""

    op = _canonical_world_transform_op(xform)
    if op is None:
        ops = tuple(xform.GetOrderedXformOps())
        names = [str(op.GetOpName()) for op in ops]
        raise RuntimeError(
            "Newton render xform topology was not prepared before binding: "
            f"prim={prim_path!r}, ops={names}, reset={xform.GetResetXformStack()}"
        )
    return op


def _canonical_world_transform_op(xform: object) -> object | None:
    """拓扑完全符合 render 合同时返回唯一 op，否则返回 None。"""

    from pxr import UsdGeom

    ops = tuple(xform.GetOrderedXformOps())
    if (
        len(ops) == 1
        and str(ops[0].GetOpName()) == _WORLD_TRANSFORM_OP_NAME
        and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform
        and ops[0].GetPrecision() == UsdGeom.XformOp.PrecisionDouble
        and bool(xform.GetResetXformStack())
    ):
        return ops[0]
    return None


def _matrix_from_transform(value: np.ndarray) -> object:
    """把 xyzw pose 转成 USD double matrix。"""

    from pxr import Gf

    xyz = value[:3]
    quat_xyzw = value[3:7]
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotateOnly(
        Gf.Quatd(
            float(quat_xyzw[3]),
            Gf.Vec3d(
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ),
        )
    )
    matrix.SetTranslateOnly(Gf.Vec3d(*(float(item) for item in xyz)))
    return matrix


def _set_world_transform(op: object, value: np.ndarray) -> None:
    """更新 cached matrix op，不在热路径重建 USD xform property。"""

    op.Set(_matrix_from_transform(value))


__all__ = [
    "NewtonRenderSync",
    "author_newton_render_root_pose",
    "prepare_newton_render_stage",
    "prepare_newton_render_subtree",
]
