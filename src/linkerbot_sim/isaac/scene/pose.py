"""Isaac/USD prim 位姿读写与刚体速度清零；只能在仿真主线程调用。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


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
    return position, matrix_to_quat_wxyz(
        _matrix3_to_numpy(matrix.ExtractRotationMatrix())
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


def _matrix3_to_numpy(matrix: object) -> np.ndarray:
    """把 USD/Gf 3x3 matrix 转成 numpy。"""

    return np.asarray(
        [[float(matrix[row][column]) for column in range(3)] for row in range(3)],
        dtype=float,
    )


__all__ = [
    "apply_prim_local_pose_and_zero_velocity",
    "read_prim_world_pose",
]
