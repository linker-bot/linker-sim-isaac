"""runtime object 到后端无关规划碰撞对象的转换。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.utils.math_utils import make_rpy_transform


def collision_objects_from_runtime_objects(
    object_handles: Sequence[object], *, stage: object | None = None
) -> tuple[CollisionObject, ...]:
    """读取显式 ``planning_collision``，优先使用 stage 中的当前 world pose。"""

    result: list[CollisionObject] = []
    for handle in tuple(object_handles):
        collision = _collision_config(handle)
        if collision is None:
            continue
        root_pose = _runtime_object_root_pose_matrix(handle, stage=stage)
        local_pose = _local_collision_pose_matrix(collision)
        result.append(
            CollisionObject(
                name=str(getattr(handle, "name", "object")),
                shape=str(collision.shape),
                pose=root_pose @ local_pose,
                size=tuple(float(value) for value in collision.size),
                enabled=bool(collision.enabled),
                padding=float(collision.padding),
            )
        )
    return tuple(result)


def _collision_config(handle: object):
    """读取 runtime object 的可选 planning collision 配置。"""

    config = getattr(handle, "config", None)
    return None if config is None else getattr(config, "planning_collision", None)


def _runtime_object_root_pose_matrix(
    handle: object, *, stage: object | None
) -> np.ndarray:
    """stage 存在时必须读取实时 world pose；仅无 stage 时使用配置 pose。"""

    if stage is not None:
        prim_path = _runtime_object_prim_path(handle)
        if prim_path is None:
            raise RuntimeError(
                f"runtime object {getattr(handle, 'name', 'object')!r} "
                "has no prim path for collision pose lookup"
            )
        return _read_stage_world_matrix(stage, prim_path)
    config = getattr(handle, "config", None)
    root_pose = getattr(config, "root_pose", None)
    if root_pose is None:
        return np.eye(4, dtype=float)
    return make_rpy_transform(
        getattr(root_pose, "xyz", (0.0, 0.0, 0.0)),
        getattr(root_pose, "rpy", (0.0, 0.0, 0.0)),
    )


def _local_collision_pose_matrix(collision: object) -> np.ndarray:
    """把 collision shape 相对 object root 的 xyz/rpy 转成 transform。"""

    return make_rpy_transform(
        getattr(collision, "xyz", (0.0, 0.0, 0.0)),
        getattr(collision, "rpy", (0.0, 0.0, 0.0)),
    )


def _runtime_object_prim_path(handle: object) -> str | None:
    """从 imported model 或 config 中解析 object 当前 prim path。"""

    for source in (getattr(handle, "model", None), getattr(handle, "config", None)):
        if source is None:
            continue
        for attr in ("imported_path", "prim_path"):
            value = getattr(source, attr, None)
            if value is not None:
                return str(value)
    return None


def _read_stage_world_matrix(stage: object, prim_path: str) -> np.ndarray:
    """读取 USD world matrix；任何读取失败都向 provider 调用方传播。"""

    from pxr import Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"collision pose prim does not exist: {prim_path}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    result = np.eye(4, dtype=float)
    rotation = matrix.ExtractRotationMatrix()
    result[:3, :3] = np.asarray(
        [[float(rotation[row][col]) for col in range(3)] for row in range(3)],
        dtype=float,
    )
    translation = matrix.ExtractTranslation()
    result[:3, 3] = np.asarray(
        [translation[0], translation[1], translation[2]], dtype=float
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"collision pose for {prim_path} contains non-finite values")
    return result


__all__ = ["collision_objects_from_runtime_objects"]
