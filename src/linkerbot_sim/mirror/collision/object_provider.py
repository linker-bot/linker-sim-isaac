"""Mirror 运行时对象到后端无关规划碰撞对象的转换。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from linkerbot_sim.objects.runtime import runtime_object_prim_path
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.utils.math_utils import make_rpy_transform
from linkerbot_sim.utils.rotations import quat_wxyz_to_matrix


def collision_objects_from_runtime_objects(
    object_handles: Sequence[object],
    *,
    stage: object | None = None,
    state_views: Mapping[str, object] | None = None,
) -> tuple[CollisionObject, ...]:
    """读取显式碰撞体；动态对象优先使用 live physics world pose。"""

    result: list[CollisionObject] = []
    views = {} if state_views is None else state_views
    for handle in tuple(object_handles):
        collision = _collision_config(handle)
        if collision is None:
            continue
        state_view = _state_view_for_handle(handle, views)
        root_pose = _runtime_object_root_pose_matrix(
            handle,
            stage=stage,
            state_view=state_view,
        )
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
    handle: object,
    *,
    stage: object | None,
    state_view: object | None = None,
) -> np.ndarray:
    """优先读取 live world pose；静态对象从 stage 或配置读取。"""

    if state_view is not None:
        require_support = getattr(state_view, "require_velocity_support", None)
        if callable(require_support):
            require_support(object_name=_runtime_object_state_name(handle))
        live_pose = getattr(state_view, "root_world_pose", None)
        pose = live_pose() if callable(live_pose) else None
        if pose is not None:
            position, orientation_wxyz = pose
            result = np.eye(4, dtype=float)
            result[:3, :3] = quat_wxyz_to_matrix(orientation_wxyz)
            result[:3, 3] = np.asarray(position, dtype=float).reshape(3)
            if not np.all(np.isfinite(result)):
                raise RuntimeError("live collision pose contains non-finite values")
            return result

    if stage is not None:
        prim_path = runtime_object_prim_path(handle)
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


def _state_view_for_handle(
    handle: object,
    state_views: Mapping[str, object],
) -> object | None:
    """按 runtime handle 优先、scene name 兜底解析 live state view。"""

    runtime_handle = getattr(handle, "runtime_handle", None)
    if runtime_handle is not None and str(runtime_handle) in state_views:
        return state_views[str(runtime_handle)]
    name = str(getattr(handle, "name", ""))
    return state_views.get(name)


def _runtime_object_state_name(handle: object) -> str:
    runtime_handle = getattr(handle, "runtime_handle", None)
    return (
        str(runtime_handle)
        if runtime_handle is not None
        else str(getattr(handle, "name", "object"))
    )


def _local_collision_pose_matrix(collision: object) -> np.ndarray:
    """把 collision shape 相对 object root 的 xyz/rpy 转成 transform。"""

    return make_rpy_transform(
        getattr(collision, "xyz", (0.0, 0.0, 0.0)),
        getattr(collision, "rpy", (0.0, 0.0, 0.0)),
    )


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
