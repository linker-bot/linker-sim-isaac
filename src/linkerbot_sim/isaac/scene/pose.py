"""Isaac/USD prim 位姿读写与刚体速度清零；只能在仿真主线程调用。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol

import numpy as np

from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


@dataclass(frozen=True, slots=True)
class PrimWorldTransform:
    """One exact USD prim identity and its complete Gf world transform."""

    prim_path: str
    matrix_row_major: tuple[float, ...]

    def __post_init__(self) -> None:
        _absolute_prim_path(self.prim_path)
        matrix = self.matrix_row_major
        if not isinstance(matrix, tuple) or len(matrix) != 16:
            raise ValueError("matrix_row_major must be a tuple of 16 values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in matrix
        ):
            raise ValueError("matrix_row_major must contain finite numbers")
        array = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        has_zero_last_column = np.allclose(
            array[:3, 3],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        has_unit_bottom_right = math.isclose(
            float(array[3, 3]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if not has_zero_last_column or not has_unit_bottom_right:
            raise ValueError("matrix_row_major must be an affine Gf matrix")
        if np.linalg.det(array[:3, :3]) == 0.0:
            raise ValueError("matrix_row_major linear transform must be invertible")


class _PrimWorldTransformInspector(Protocol):
    def inspect(self, prim_paths: tuple[str, ...]) -> Sequence[PrimWorldTransform]: ...


def read_prim_world_transforms(
    stage: object,
    prim_paths: Sequence[str],
    *,
    inspector: _PrimWorldTransformInspector | None = None,
) -> tuple[PrimWorldTransform, ...]:
    """Read complete world matrices in requested path order on the Isaac thread."""

    if isinstance(prim_paths, (str, bytes)):
        raise TypeError("prim_paths must be a sequence of strings")
    paths = tuple(prim_paths)
    if not paths:
        raise ValueError("prim_paths cannot be empty")
    for path in paths:
        _absolute_prim_path(path)
    if len(set(paths)) != len(paths):
        raise ValueError("prim_paths must not contain duplicates")

    resolved_inspector = inspector or _UsdPrimWorldTransformInspector(stage)
    values = tuple(resolved_inspector.inspect(paths))
    by_path: dict[str, PrimWorldTransform] = {}
    for value in values:
        if not isinstance(value, PrimWorldTransform):
            raise TypeError("world-transform inspector returned an invalid value")
        if value.prim_path not in paths:
            raise RuntimeError(
                "world-transform inspector returned an unrequested path: "
                f"{value.prim_path}"
            )
        if value.prim_path in by_path:
            raise RuntimeError(
                "world-transform inspector returned a duplicate path: "
                f"{value.prim_path}"
            )
        by_path[value.prim_path] = value
    missing = tuple(path for path in paths if path not in by_path)
    if missing:
        raise RuntimeError(f"world-transform inspector omitted paths: {missing!r}")
    return tuple(by_path[path] for path in paths)


class _UsdPrimWorldTransformInspector:
    """Gf matrix reader loaded only inside an active Isaac USD runtime."""

    def __init__(self, stage: object) -> None:
        try:
            from pxr import Sdf, UsdGeom
        except ImportError as exc:
            raise RuntimeError(
                "world-transform inspection requires an active Isaac USD runtime"
            ) from exc
        if stage is None:
            raise ValueError("stage is required")
        self._stage = stage
        self._Sdf = Sdf
        self._UsdGeom = UsdGeom

    def inspect(self, prim_paths: tuple[str, ...]) -> tuple[PrimWorldTransform, ...]:
        cache = self._UsdGeom.XformCache()
        values: list[PrimWorldTransform] = []
        for path in prim_paths:
            prim = self._stage.GetPrimAtPath(self._Sdf.Path(path))
            if not prim.IsValid():
                raise RuntimeError(f"world-transform prim does not exist: {path}")
            if not bool(prim.IsA(self._UsdGeom.Xformable)):
                raise RuntimeError(f"world-transform prim is not Xformable: {path}")
            matrix = cache.GetLocalToWorldTransform(prim)
            values.append(
                PrimWorldTransform(
                    prim_path=path,
                    matrix_row_major=tuple(
                        float(matrix[row][column])
                        for row in range(4)
                        for column in range(4)
                    ),
                )
            )
        return tuple(values)


def read_prim_world_pose(
    stage: object,
    prim_path: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取 USD prim 的世界位置与 ``wxyz`` 四元数；无效路径返回 ``None``。"""

    from pxr import Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid():
        return None
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    position = np.asarray(
        [translation[0], translation[1], translation[2]],
        dtype=float,
    )
    # Gf stores row-vector transforms; SciPy consumes column-vector rotations.
    # Without the transpose, snapshots and camera extrinsics use the inverse pose.
    return position, matrix_to_quat_wxyz(
        _matrix3_to_numpy(matrix.ExtractRotationMatrix()).T
    )


def apply_prim_local_pose_and_zero_velocity(
    stage: object,
    prim_path: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
) -> bool:
    """写回 prim local pose，并对 prim 树中已有 RigidBodyAPI 清零速度。"""

    from pxr import Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid():
        return False
    quat = np.asarray(orientation_wxyz, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("object orientation quaternion must be non-zero")
    quat = quat / norm
    xyz = np.asarray(position, dtype=float).reshape(3)
    xform = UsdGeom.Xformable(prim)
    translate_op = _get_or_add_translate_op(xform)
    rotate_op = next(
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:rotateXYZ"
        ),
        None,
    )
    if rotate_op is not None and all(
        op.GetOpName() in {"xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"}
        for op in xform.GetOrderedXformOps()
    ):
        # Preserve the imported root's op topology while PhysX is live. Replacing
        # rotateXYZ with a newly added orient op can rebase fixed child bodies on
        # the next physics-to-USD publication, even for an identical pose.
        from pxr import Gf
        from scipy.spatial.transform import Rotation

        angles = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        vector = (
            Gf.Vec3f
            if rotate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
            else Gf.Vec3d
        )
        _set_translate_op(translate_op, xyz)
        rotate_op.Set(vector(*angles))
        _zero_rigid_body_velocities(prim)
        return True
    orient_op = _get_or_add_orient_op(xform)
    _set_translate_op(translate_op, xyz)
    _set_orient_op(orient_op, quat)
    # scale 等非平移/旋转 op 必须保留；否则对象 reset 后会丢失原有缩放。
    managed = {"xformOp:translate", "xformOp:orient"}
    preserved = [
        op
        for op in xform.GetOrderedXformOps()
        if op.GetOpName() not in managed
        and not op.GetOpName().startswith("xformOp:rotate")
        and op.GetOpName() != "xformOp:transform"
    ]
    xform.SetXformOpOrder([translate_op, orient_op, *preserved])
    _zero_rigid_body_velocities(prim)
    return True


def _get_or_add_translate_op(xform: object) -> object:
    """复用已有 translate op；没有时创建 double precision op。"""

    from pxr import UsdGeom

    existing = _xform_op_by_name(xform, "xformOp:translate")
    if existing is not None:
        return existing
    return xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)


def _get_or_add_orient_op(xform: object) -> object:
    """复用已有 orient op；没有时创建 double precision op。"""

    from pxr import UsdGeom

    existing = _xform_op_by_name(xform, "xformOp:orient")
    if existing is not None:
        return existing
    return xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)


def _xform_op_by_name(xform: object, op_name: str) -> object | None:
    """按 attr 名称查找已有 xform op，包括不在当前 op order 中的属性。"""

    from pxr import UsdGeom

    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == op_name:
            return op
    attr = xform.GetPrim().GetAttribute(op_name)
    if attr is not None and attr.IsValid():
        return UsdGeom.XformOp(attr)
    return None


def _set_translate_op(op: object, xyz: np.ndarray) -> None:
    """按已有 translate op precision 写入位置。"""

    from pxr import Gf, UsdGeom

    value = (
        Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
        else Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    )
    op.Set(value)


def _set_orient_op(op: object, quat_wxyz: np.ndarray) -> None:
    """按已有 orient op precision 写入四元数。"""

    from pxr import Gf, UsdGeom

    if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        value = Gf.Quatf(
            float(quat_wxyz[0]),
            Gf.Vec3f(
                float(quat_wxyz[1]),
                float(quat_wxyz[2]),
                float(quat_wxyz[3]),
            ),
        )
    else:
        value = Gf.Quatd(
            float(quat_wxyz[0]),
            Gf.Vec3d(
                float(quat_wxyz[1]),
                float(quat_wxyz[2]),
                float(quat_wxyz[3]),
            ),
        )
    op.Set(value)


def _zero_rigid_body_velocities(root_prim: object) -> None:
    """对 prim 子树中已经应用 RigidBodyAPI 的 prim 清零线/角速度。"""

    from pxr import Gf, UsdPhysics

    stack = [root_prim]
    while stack:
        prim = stack.pop()
        stack.extend(list(prim.GetChildren()))
        try:
            has_api = bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        except Exception:
            has_api = False
        if not has_api:
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        for attr_name in ("GetVelocityAttr", "GetAngularVelocityAttr"):
            attr_getter = getattr(api, attr_name, None)
            if not callable(attr_getter):
                continue
            attr = attr_getter()
            if attr is not None and attr.IsValid():
                attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _absolute_prim_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
    ):
        raise ValueError("prim path must be a canonical absolute USD prim path")
    return value


def _matrix3_to_numpy(matrix: object) -> np.ndarray:
    """把 USD/Gf 3x3 matrix 转成 numpy。"""

    return np.asarray(
        [[float(matrix[row][column]) for column in range(3)] for row in range(3)],
        dtype=float,
    )


__all__ = [
    "PrimWorldTransform",
    "apply_prim_local_pose_and_zero_velocity",
    "read_prim_world_pose",
    "read_prim_world_transforms",
]
