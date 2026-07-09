"""Object state readers for Isaac tiled interactive runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


@dataclass(frozen=True)
class TiledDynamicChainObjectPoseView:
    """动态链式对象的 batched rigid-body view。

    ``RigidPrim`` 的 rows 按 env-major 展开：
    ``env_0/body_0..N, env_1/body_0..N, ...``。对外仍按 env 读写 object
    state，但 snapshot/restore 会保存和恢复每个 child rigid body 的 PhysX pose。
    """

    view: object
    body_names: tuple[str, ...]
    body_paths_by_env: tuple[tuple[str, ...], ...]

    @property
    def body_count(self) -> int:
        """返回每个 env 中的 rigid body 数。"""

        return len(self.body_names)

    @property
    def env_count(self) -> int:
        """返回 view 覆盖的 tiled env 数。"""

        return len(self.body_paths_by_env)

    def read_object_state(
        self,
        *,
        object_name: str,
        env_ids: np.ndarray,
        env_origins: np.ndarray,
    ) -> dict[str, object]:
        """读取 selected env 中所有 child rigid body 的 world/local pose。

        world pose 用于直接调试 PhysX 状态；local pose 用于 runtime-neutral snapshot，
        因为不同 tiled env 的 world origin 不同，跨 env 复制时必须保持 env 内局部布局。
        """

        selected = np.asarray(env_ids, dtype=int).reshape(-1)
        indices = self._flat_indices(selected)
        get_world_poses = getattr(self.view, "get_world_poses", None)
        if not callable(get_world_poses):
            raise RuntimeError(
                f"tiled dynamic-chain object {object_name!r} view does not provide get_world_poses"
            )
        try:
            positions, orientations = get_world_poses(indices=indices)
        except Exception as exc:
            raise RuntimeError(
                f"failed to read tiled dynamic-chain object {object_name!r} poses from rigid view"
            ) from exc
        try:
            body_positions = _to_numpy_array(positions).reshape(
                selected.size, self.body_count, 3
            )
            body_orientations = _to_numpy_array(orientations).reshape(
                selected.size, self.body_count, 4
            )
        except Exception as exc:
            raise RuntimeError(
                f"tiled dynamic-chain object {object_name!r} view returned invalid pose shape"
            ) from exc
        object_positions = body_positions.mean(axis=1)
        object_orientations = body_orientations[:, 0, :]
        origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
        return {
            "env_ids": selected.astype(int).tolist(),
            "positions_world": object_positions.tolist(),
            "positions_local": (object_positions - origins[selected]).tolist(),
            "orientations_wxyz": object_orientations.tolist(),
            "body_names": list(self.body_names),
            "body_positions_world": body_positions.tolist(),
            # body local position 按目标 env origin 归一化，后续 set_snapshot 到其它 env 时
            # 再加回目标 env origin，就能得到正确的 world pose。
            "body_positions_local": (body_positions - origins[selected, None, :]).tolist(),
            "body_orientations_wxyz": body_orientations.tolist(),
        }

    def restore_object_state(
        self,
        *,
        object_name: str,
        object_state: Mapping[str, object],
        selected_env_ids: set[int],
        env_origins: np.ndarray | None = None,
    ) -> int:
        """按 snapshot 恢复 selected env 的所有 child rigid body pose。

        优先接受 ``body_positions_world`` 以兼容已有 get_state/cache；如果只有
        ``body_positions_local``，则必须提供目标 ``env_origins``，用目标 env origin
        把局部位姿还原成 PhysX 需要的 world pose。
        """

        snapshot_env_ids = np.asarray(object_state.get("env_ids", ()), dtype=int).reshape(-1)
        rows = [
            row
            for row, env_id in enumerate(snapshot_env_ids)
            if int(env_id) in selected_env_ids
        ]
        if not rows:
            return 0
        # 新的 runtime-neutral snapshot 只保证 local body pose；旧的 tiled state/cache 可能
        # 同时带 world pose。这里两种都接受，保持向后兼容。
        has_world_positions = "body_positions_world" in object_state
        has_local_positions = "body_positions_local" in object_state
        missing = []
        if not has_world_positions and not has_local_positions:
            missing.append("body_positions_world or body_positions_local")
        if "body_orientations_wxyz" not in object_state:
            missing.append("body_orientations_wxyz")
        if missing:
            raise RuntimeError(
                f"tiled dynamic-chain object {object_name!r} snapshot missing {missing}"
            )
        try:
            body_positions_world = None
            if has_world_positions:
                body_positions_world = np.asarray(
                    object_state.get("body_positions_world", ()), dtype=float
                ).reshape(snapshot_env_ids.size, self.body_count, 3)
            body_positions_local = None
            if has_local_positions:
                body_positions_local = np.asarray(
                    object_state.get("body_positions_local", ()), dtype=float
                ).reshape(snapshot_env_ids.size, self.body_count, 3)
            body_orientations = np.asarray(
                object_state.get("body_orientations_wxyz", ()), dtype=float
            ).reshape(snapshot_env_ids.size, self.body_count, 4)
        except Exception as exc:
            raise RuntimeError(
                f"tiled dynamic-chain object {object_name!r} snapshot has invalid body pose shape"
            ) from exc
        env_index = np.asarray([int(snapshot_env_ids[row]) for row in rows], dtype=int)
        indices = self._flat_indices(env_index)
        if body_positions_world is not None:
            positions = np.asarray(body_positions_world[rows], dtype=float).reshape(-1, 3)
        else:
            if env_origins is None:
                raise RuntimeError(
                    f"cannot restore tiled dynamic-chain object {object_name!r} "
                    "without env origins"
                )
            origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
            assert body_positions_local is not None
            # 注意这里使用“目标 env”的 origin，而不是 snapshot source env 的 origin；
            # 这样 env0 的 local 快照才能正确复现在 env1/env2/... 的 world 坐标中。
            positions = (
                np.asarray(body_positions_local[rows], dtype=float)
                + origins[env_index, None, :]
            ).reshape(-1, 3)
        orientations = np.asarray(body_orientations[rows], dtype=float).reshape(-1, 4)
        set_world_poses = getattr(self.view, "set_world_poses", None)
        if not callable(set_world_poses):
            raise RuntimeError(
                f"tiled dynamic-chain object {object_name!r} view does not provide set_world_poses"
            )
        try:
            set_world_poses(
                positions=positions,
                orientations=orientations,
                indices=indices,
            )
            _zero_object_view_velocities(
                self.view,
                row_count=positions.shape[0],
                indices=indices,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to restore tiled dynamic-chain object {object_name!r} poses through rigid view"
            ) from exc
        return len(rows)

    def _flat_indices(self, env_ids: np.ndarray) -> np.ndarray:
        """把 env ids 转成 env-major rigid body row indices。"""

        selected = np.asarray(env_ids, dtype=int).reshape(-1)
        if self.body_count < 1:
            raise RuntimeError("dynamic-chain object view has no rigid bodies")
        if np.any(selected < 0) or np.any(selected >= self.env_count):
            raise ValueError("env_ids contains out-of-range dynamic-chain object env id")
        offsets = selected[:, None] * self.body_count
        body_offsets = np.arange(self.body_count, dtype=int)[None, :]
        return (offsets + body_offsets).reshape(-1)


def _read_tiled_object_states(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    env_origins: np.ndarray,
    env_ids: np.ndarray,
    object_pose_views: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """读取 selected env 的 object world/local pose，返回 JSON-compatible dict。

    返回中同时保留 world/local：world 方便 set_state 原样写回，local 则用于 snapshot
    跨 env/跨 runtime 传递。
    """

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
    objects: dict[str, object] = {}
    for object_name, prim_paths in object_prim_paths.items():
        view_state = _read_object_view_state(
            object_pose_views.get(str(object_name)) if object_pose_views is not None else None,
            object_name=str(object_name),
            env_ids=selected,
            env_origins=origins,
        )
        if view_state is not None:
            objects[str(object_name)] = view_state
            continue
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
    object_pose_views: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """缓存 tiled objects 的初始 env-local pose。

    reset 使用这份缓存恢复对象初始状态；它与新 snapshot restore 共用同一条恢复路径。
    """

    state = _read_tiled_object_states(
        stage=stage,
        object_prim_paths=object_prim_paths,
        env_origins=env_origins,
        env_ids=env_ids,
        object_pose_views=object_pose_views,
    )
    snapshot: dict[str, dict[str, object]] = {}
    for object_name, object_state in state.items():
        if not isinstance(object_state, Mapping):
            continue
        entry: dict[str, object] = {
            "env_ids": np.asarray(object_state.get("env_ids", ()), dtype=int).reshape(-1),
            "positions_world": np.asarray(
                object_state.get("positions_world", ()), dtype=float
            ).reshape(-1, 3),
            "positions_local": np.asarray(
                object_state.get("positions_local", ()), dtype=float
            ).reshape(-1, 3),
            "orientations_wxyz": np.asarray(
                object_state.get("orientations_wxyz", ()), dtype=float
            ).reshape(-1, 4),
        }
        if "body_names" in object_state:
            entry["body_names"] = tuple(str(name) for name in object_state["body_names"])
        if "body_positions_world" in object_state:
            entry["body_positions_world"] = np.asarray(
                object_state.get("body_positions_world", ()), dtype=float
            )
        if "body_positions_local" in object_state:
            entry["body_positions_local"] = np.asarray(
                object_state.get("body_positions_local", ()), dtype=float
            )
        if "body_orientations_wxyz" in object_state:
            entry["body_orientations_wxyz"] = np.asarray(
                object_state.get("body_orientations_wxyz", ()), dtype=float
            )
        snapshot[str(object_name)] = entry
    return snapshot


def _restore_tiled_object_pose_snapshot(
    *,
    stage: object,
    object_prim_paths: Mapping[str, tuple[str, ...]],
    snapshot: Mapping[str, Mapping[str, object]],
    env_ids: np.ndarray,
    env_origins: np.ndarray | None = None,
    object_pose_views: Mapping[str, object] | None = None,
) -> int:
    """按 selected env 恢复 tiled objects pose，并尽量清零刚体速度。

    ``snapshot`` 可以来自初始化缓存、get_state，也可以由 runtime-neutral
    ``ObjectSnapshot`` 转换而来。恢复时只处理 ``env_ids`` 选中的 env。
    """

    selected = {int(env_id) for env_id in np.asarray(env_ids, dtype=int).reshape(-1)}
    restored = 0
    for object_name, object_state in snapshot.items():
        paths = object_prim_paths.get(str(object_name))
        if paths is None:
            continue
        snapshot_env_ids = np.asarray(object_state.get("env_ids", ()), dtype=int).reshape(-1)
        world_positions = np.asarray(
            object_state.get("positions_world", ()), dtype=float
        ).reshape(-1, 3)
        positions = np.asarray(object_state.get("positions_local", ()), dtype=float).reshape(-1, 3)
        orientations = np.asarray(object_state.get("orientations_wxyz", ()), dtype=float).reshape(-1, 4)
        row_count = min(snapshot_env_ids.size, positions.shape[0], orientations.shape[0])
        object_view = (
            object_pose_views.get(str(object_name)) if object_pose_views is not None else None
        )
        view_restore = getattr(object_view, "restore_object_state", None)
        if callable(view_restore):
            # dynamic-chain wrapper 自己知道 env-major body row 展开规则，优先交给它恢复。
            restored += int(
                view_restore(
                    object_name=str(object_name),
                    object_state=object_state,
                    selected_env_ids=selected,
                    env_origins=env_origins,
                )
            )
            continue
        view_restored = _restore_object_view_state(
            object_view,
            object_name=str(object_name),
            snapshot_env_ids=snapshot_env_ids[:row_count],
            positions_world=world_positions[:row_count],
            positions_local=positions[:row_count],
            orientations_wxyz=orientations[:row_count],
            env_origins=env_origins,
            selected_env_ids=selected,
        )
        if view_restored is not None:
            restored += view_restored
            continue
        for row in range(row_count):
            env_id = int(snapshot_env_ids[row])
            if env_id not in selected or env_id >= len(paths):
                continue
            # 没有 rigid view 时退回到直接写 prim pose；这是静态测试和部分简单对象的兜底路径。
            if _apply_prim_local_pose_and_zero_velocity(
                stage,
                str(paths[env_id]),
                positions[row],
                orientations[row],
            ):
                restored += 1
    return restored


def _read_object_view_state(
    view: object | None,
    *,
    object_name: str,
    env_ids: np.ndarray,
    env_origins: np.ndarray,
) -> dict[str, object] | None:
    """优先从 Isaac rigid view 读取 dynamic object pose。

    rigid view 读到的是 world pose；这里同步生成 env-local pose，供 snapshot/clone 使用。
    """

    if view is None:
        return None
    read_state = getattr(view, "read_object_state", None)
    if callable(read_state):
        return read_state(
            object_name=object_name,
            env_ids=env_ids,
            env_origins=env_origins,
        )
    get_world_poses = getattr(view, "get_world_poses", None)
    if not callable(get_world_poses):
        raise RuntimeError(
            f"tiled object {object_name!r} rigid view does not provide get_world_poses"
        )
    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    try:
        positions, orientations = get_world_poses(indices=selected)
    except Exception as exc:
        raise RuntimeError(
            f"failed to read tiled object {object_name!r} pose from rigid view"
        ) from exc
    position_array = _to_numpy_array(positions).reshape(-1, 3)
    orientation_array = _to_numpy_array(orientations).reshape(-1, 4)
    row_count = min(selected.size, position_array.shape[0], orientation_array.shape[0])
    selected = selected[:row_count]
    position_array = position_array[:row_count]
    orientation_array = orientation_array[:row_count]
    local_positions = position_array - env_origins[selected]
    return {
        "env_ids": selected.astype(int).tolist(),
        "positions_world": position_array.tolist(),
        "positions_local": local_positions.tolist(),
        "orientations_wxyz": orientation_array.tolist(),
    }


def _restore_object_view_state(
    view: object | None,
    *,
    object_name: str,
    snapshot_env_ids: np.ndarray,
    positions_world: np.ndarray,
    positions_local: np.ndarray,
    orientations_wxyz: np.ndarray,
    env_origins: np.ndarray | None,
    selected_env_ids: set[int],
) -> int | None:
    """用 Isaac rigid view 恢复 dynamic object pose；view 存在时失败即报错。

    对普通 rigid object，indices 就是 env id；对 dynamic-chain，则由上面的 wrapper 处理
    env-major body indices。
    """

    if view is None:
        return None
    rows = [
        row
        for row, env_id in enumerate(np.asarray(snapshot_env_ids, dtype=int).reshape(-1))
        if int(env_id) in selected_env_ids
    ]
    if not rows:
        return 0
    env_index = np.asarray([int(snapshot_env_ids[row]) for row in rows], dtype=int)
    if positions_world.shape[0] >= snapshot_env_ids.shape[0]:
        world_positions = np.asarray(positions_world[rows], dtype=float).reshape(-1, 3)
    else:
        if env_origins is None:
            raise RuntimeError(
                f"cannot restore tiled object {object_name!r} rigid view without env origins"
            )
        origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
        # local -> world 使用目标 env origin，保证 env 间复制不会保留 source env 的偏移。
        world_positions = np.asarray(positions_local[rows], dtype=float).reshape(-1, 3) + origins[
            env_index
        ]
    orientations = np.asarray(orientations_wxyz[rows], dtype=float).reshape(-1, 4)
    set_world_poses = getattr(view, "set_world_poses", None)
    if not callable(set_world_poses):
        raise RuntimeError(
            f"tiled object {object_name!r} rigid view does not provide set_world_poses"
        )
    try:
        set_world_poses(
            positions=world_positions,
            orientations=orientations,
            indices=env_index,
        )
        _zero_object_view_velocities(view, row_count=len(rows), indices=env_index)
    except Exception as exc:
        raise RuntimeError(
            f"failed to restore tiled object {object_name!r} pose through rigid view"
        ) from exc
    return len(rows)


def _zero_object_view_velocities(view: object, *, row_count: int, indices: np.ndarray) -> None:
    """清零 rigid view 线速度和角速度，兼容 Isaac 新旧 API。"""

    set_velocities = getattr(view, "set_velocities", None)
    if callable(set_velocities):
        set_velocities(np.zeros((int(row_count), 6), dtype=float), indices=indices)
        return
    set_linear = getattr(view, "set_linear_velocities", None)
    set_angular = getattr(view, "set_angular_velocities", None)
    if callable(set_linear) and callable(set_angular):
        set_linear(np.zeros((int(row_count), 3), dtype=float), indices=indices)
        set_angular(np.zeros((int(row_count), 3), dtype=float), indices=indices)
        return
    raise RuntimeError("rigid view does not provide a velocity reset API")


def _to_numpy_array(value: object) -> np.ndarray:
    """把 Isaac/torch/warp 返回值尽量转成 numpy 数组。"""

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    return np.asarray(candidate, dtype=float)


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
