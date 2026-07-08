"""Object state readers for Isaac tiled interactive runtime."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


def _read_tiled_object_states(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    env_origins: np.ndarray,
    env_ids: np.ndarray,
) -> dict[str, object]:
    """读取 selected env 的 object world/local pose，返回 JSON-compatible dict。"""

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
    objects: dict[str, object] = {}
    for object_name, prim_paths in object_prim_paths.items():
        positions: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        valid_env_ids: list[int] = []
        for env_id in selected:
            if int(env_id) >= len(prim_paths):
                continue
            pose = _read_prim_world_pose(stage, str(prim_paths[int(env_id)]))
            if pose is None:
                continue
            position, orientation = pose
            positions.append(position)
            orientations.append(orientation)
            valid_env_ids.append(int(env_id))
        if not positions:
            continue
        position_array = np.vstack(positions)
        orientation_array = np.vstack(orientations)
        local_positions = position_array - origins[np.asarray(valid_env_ids, dtype=int)]
        objects[str(object_name)] = {
            "env_ids": valid_env_ids,
            "positions_world": position_array.tolist(),
            "positions_local": local_positions.tolist(),
            "orientations_wxyz": orientation_array.tolist(),
        }
    return objects


def _capture_tiled_object_pose_snapshot(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    env_origins: np.ndarray,
    env_ids: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """缓存 tiled objects 的初始 env-local pose。"""

    state = _read_tiled_object_states(
        stage=stage,
        object_prim_paths=object_prim_paths,
        env_origins=env_origins,
        env_ids=env_ids,
    )
    snapshot: dict[str, dict[str, np.ndarray]] = {}
    for object_name, object_state in state.items():
        if not isinstance(object_state, Mapping):
            continue
        snapshot[str(object_name)] = {
            "env_ids": np.asarray(object_state.get("env_ids", ()), dtype=int).reshape(-1),
            "positions_local": np.asarray(
                object_state.get("positions_local", ()), dtype=float
            ).reshape(-1, 3),
            "orientations_wxyz": np.asarray(
                object_state.get("orientations_wxyz", ()), dtype=float
            ).reshape(-1, 4),
        }
    return snapshot


def _restore_tiled_object_pose_snapshot(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    snapshot: Mapping[str, Mapping[str, np.ndarray]],
    env_ids: np.ndarray,
) -> int:
    """按 selected env 恢复 tiled objects 初始 pose，并尽量清零刚体速度。"""

    selected = {int(env_id) for env_id in np.asarray(env_ids, dtype=int).reshape(-1)}
    restored = 0
    for object_name, object_state in snapshot.items():
        paths = object_prim_paths.get(str(object_name))
        if paths is None:
            continue
        snapshot_env_ids = np.asarray(object_state.get("env_ids", ()), dtype=int).reshape(-1)
        positions = np.asarray(object_state.get("positions_local", ()), dtype=float).reshape(-1, 3)
        orientations = np.asarray(object_state.get("orientations_wxyz", ()), dtype=float).reshape(-1, 4)
        row_count = min(snapshot_env_ids.size, positions.shape[0], orientations.shape[0])
        for row in range(row_count):
            env_id = int(snapshot_env_ids[row])
            if env_id not in selected or env_id >= len(paths):
                continue
            if _apply_prim_local_pose_and_zero_velocity(
                stage,
                str(paths[env_id]),
                positions[row],
                orientations[row],
            ):
                restored += 1
    return restored


def _read_prim_world_pose(stage: object, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """读取 USD prim 的世界位姿；只能在仿真主线程调用。"""

    from pxr import Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid():
        return None
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    position = np.asarray([translation[0], translation[1], translation[2]], dtype=float)
    return position, matrix_to_quat_wxyz(_matrix3_to_numpy(matrix.ExtractRotationMatrix()))


def _apply_prim_local_pose_and_zero_velocity(
    stage: object,
    prim_path: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
) -> bool:
    """写回 prim local pose，并对 prim 树中已有 RigidBodyAPI 尽量清零速度。"""

    from pxr import Gf, Sdf, UsdGeom

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
    xform.SetXformOpOrder([translate_op, orient_op])
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
    """按 op attr 名称查找已有 xform op，包括不在当前 op order 中的属性。"""

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
            Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
        )
    else:
        value = Gf.Quatd(
            float(quat_wxyz[0]),
            Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
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
        [[float(matrix[row][col]) for col in range(3)] for row in range(3)],
        dtype=float,
    )
