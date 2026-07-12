"""tiled rigid object 与动态链对象的 batched PhysX view 读写。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


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
    reference_body: str

    def __post_init__(self) -> None:
        """尽早拒绝不存在或不唯一的 reference body。"""

        match_count = self.body_names.count(self.reference_body)
        if match_count != 1:
            raise ValueError(
                "dynamic-chain state_summary.reference_body must identify exactly "
                f"one body; got {self.reference_body!r} in {self.body_names!r}"
            )

    @property
    def body_count(self) -> int:
        """返回每个 env 中的 rigid body 数。"""

        return len(self.body_names)

    @property
    def env_count(self) -> int:
        """返回 view 覆盖的 tiled env 数。"""

        return len(self.body_paths_by_env)

    @property
    def reference_body_index(self) -> int:
        """返回显式 reference body 的稳定列索引。"""

        return self.body_names.index(self.reference_body)

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
        try:
            body_velocities = _read_object_view_velocities(
                self.view,
                row_count=selected.size * self.body_count,
                indices=indices,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to read tiled dynamic-chain object {object_name!r} "
                "velocities from rigid view"
            ) from exc
        reference_index = self.reference_body_index
        object_positions = body_positions[:, reference_index, :]
        object_orientations = body_orientations[:, reference_index, :]
        origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
        result: dict[str, object] = {
            "env_ids": selected.astype(int).tolist(),
            "positions_world": object_positions.tolist(),
            "positions_local": (object_positions - origins[selected]).tolist(),
            "orientations_wxyz": object_orientations.tolist(),
            "body_names": list(self.body_names),
            "body_positions_world": body_positions.tolist(),
            # body local position 按目标 env origin 归一化，后续 set_snapshot 到其它 env 时
            # 再加回目标 env origin，就能得到正确的 world pose。
            "body_positions_local": (
                body_positions - origins[selected, None, :]
            ).tolist(),
            "body_orientations_wxyz": body_orientations.tolist(),
        }
        if body_velocities is not None:
            linear, angular = body_velocities
            body_linear = linear.reshape(selected.size, self.body_count, 3)
            body_angular = angular.reshape(selected.size, self.body_count, 3)
            if reference_index is None:
                object_linear = body_linear.mean(axis=1)
                object_angular = body_angular[:, 0, :]
            else:
                object_linear = body_linear[:, reference_index, :]
                object_angular = body_angular[:, reference_index, :]
            result.update(
                {
                    "linear_velocities": object_linear.tolist(),
                    "angular_velocities": object_angular.tolist(),
                    "body_linear_velocities": body_linear.tolist(),
                    "body_angular_velocities": body_angular.tolist(),
                }
            )
        return result

    def restore_object_state(
        self,
        *,
        object_name: str,
        object_state: Mapping[str, object],
        selected_env_ids: set[int],
        env_origins: np.ndarray | None = None,
    ) -> int:
        """按 snapshot 恢复 selected env 的所有 child rigid body pose。

        ``body_positions_world`` 可直接写回 PhysX；只有 ``body_positions_local`` 时必须提供目标
        ``env_origins``，用目标 env origin 还原 world pose。
        """

        snapshot_env_ids = np.asarray(
            object_state.get("env_ids", ()), dtype=int
        ).reshape(-1)
        rows = [
            row
            for row, env_id in enumerate(snapshot_env_ids)
            if int(env_id) in selected_env_ids
        ]
        if not rows:
            return 0
        # local pose 支持跨 env 恢复，world pose 支持同一 tiled scene 的直接状态写回。
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
            body_linear_velocities = None
            if "body_linear_velocities" in object_state:
                body_linear_velocities = np.asarray(
                    object_state.get("body_linear_velocities", ()), dtype=float
                ).reshape(snapshot_env_ids.size, self.body_count, 3)
            body_angular_velocities = None
            if "body_angular_velocities" in object_state:
                body_angular_velocities = np.asarray(
                    object_state.get("body_angular_velocities", ()), dtype=float
                ).reshape(snapshot_env_ids.size, self.body_count, 3)
        except Exception as exc:
            raise RuntimeError(
                f"tiled dynamic-chain object {object_name!r} snapshot has invalid body state shape"
            ) from exc
        env_index = np.asarray([int(snapshot_env_ids[row]) for row in rows], dtype=int)
        indices = self._flat_indices(env_index)
        if body_positions_world is not None:
            positions = np.asarray(body_positions_world[rows], dtype=float).reshape(
                -1, 3
            )
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
            _set_object_view_velocities(
                self.view,
                row_count=positions.shape[0],
                indices=indices,
                linear_velocities=(
                    None
                    if body_linear_velocities is None
                    else body_linear_velocities[rows].reshape(-1, 3)
                ),
                angular_velocities=(
                    None
                    if body_angular_velocities is None
                    else body_angular_velocities[rows].reshape(-1, 3)
                ),
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
            raise ValueError(
                "env_ids contains out-of-range dynamic-chain object env id"
            )
        offsets = selected[:, None] * self.body_count
        body_offsets = np.arange(self.body_count, dtype=int)[None, :]
        return (offsets + body_offsets).reshape(-1)


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
    try:
        velocities = _read_object_view_velocities(
            view,
            row_count=row_count,
            indices=selected,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to read tiled object {object_name!r} velocity from rigid view"
        ) from exc
    local_positions = position_array - env_origins[selected]
    result: dict[str, object] = {
        "env_ids": selected.astype(int).tolist(),
        "positions_world": position_array.tolist(),
        "positions_local": local_positions.tolist(),
        "orientations_wxyz": orientation_array.tolist(),
    }
    if velocities is not None:
        linear, angular = velocities
        result["linear_velocities"] = linear.tolist()
        result["angular_velocities"] = angular.tolist()
    return result


def _restore_object_view_state(
    view: object | None,
    *,
    object_name: str,
    snapshot_env_ids: np.ndarray,
    positions_world: np.ndarray,
    positions_local: np.ndarray,
    orientations_wxyz: np.ndarray,
    linear_velocities: np.ndarray | None,
    angular_velocities: np.ndarray | None,
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
        for row, env_id in enumerate(
            np.asarray(snapshot_env_ids, dtype=int).reshape(-1)
        )
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
        world_positions = (
            np.asarray(positions_local[rows], dtype=float).reshape(-1, 3)
            + origins[env_index]
        )
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
        _set_object_view_velocities(
            view,
            row_count=len(rows),
            indices=env_index,
            linear_velocities=(
                None
                if linear_velocities is None
                else np.asarray(linear_velocities[rows], dtype=float).reshape(-1, 3)
            ),
            angular_velocities=(
                None
                if angular_velocities is None
                else np.asarray(angular_velocities[rows], dtype=float).reshape(-1, 3)
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to restore tiled object {object_name!r} pose through rigid view"
        ) from exc
    return len(rows)


def _zero_object_view_velocities(
    view: object, *, row_count: int, indices: np.ndarray
) -> None:
    """清零 rigid view 线速度和角速度，兼容 Isaac 新旧 API。"""

    _set_object_view_velocities(
        view,
        row_count=row_count,
        indices=indices,
        linear_velocities=None,
        angular_velocities=None,
    )


def _read_object_view_velocities(
    view: object,
    *,
    row_count: int,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取 rigid view 的线/角速度；旧 view 无读取 API 时返回 ``None``。"""

    get_velocities = getattr(view, "get_velocities", None)
    if callable(get_velocities):
        velocities = _to_numpy_array(get_velocities(indices=indices)).reshape(
            int(row_count), 6
        )
        return velocities[:, :3], velocities[:, 3:]
    get_linear = getattr(view, "get_linear_velocities", None)
    get_angular = getattr(view, "get_angular_velocities", None)
    if not callable(get_linear) and not callable(get_angular):
        return None
    if not callable(get_linear) or not callable(get_angular):
        raise RuntimeError("rigid view provides an incomplete velocity read API")
    linear = _to_numpy_array(get_linear(indices=indices)).reshape(int(row_count), 3)
    angular = _to_numpy_array(get_angular(indices=indices)).reshape(int(row_count), 3)
    return linear, angular


def _set_object_view_velocities(
    view: object,
    *,
    row_count: int,
    indices: np.ndarray,
    linear_velocities: np.ndarray | None,
    angular_velocities: np.ndarray | None,
) -> None:
    """写回 rigid view 速度；缺失分量按旧快照语义清零。"""

    linear = (
        np.zeros((int(row_count), 3), dtype=float)
        if linear_velocities is None
        else np.asarray(linear_velocities, dtype=float).reshape(int(row_count), 3)
    )
    angular = (
        np.zeros((int(row_count), 3), dtype=float)
        if angular_velocities is None
        else np.asarray(angular_velocities, dtype=float).reshape(int(row_count), 3)
    )

    set_velocities = getattr(view, "set_velocities", None)
    if callable(set_velocities):
        set_velocities(np.concatenate((linear, angular), axis=1), indices=indices)
        return
    set_linear = getattr(view, "set_linear_velocities", None)
    set_angular = getattr(view, "set_angular_velocities", None)
    if callable(set_linear) and callable(set_angular):
        set_linear(linear, indices=indices)
        set_angular(angular, indices=indices)
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


__all__ = ["TiledDynamicChainObjectPoseView"]
