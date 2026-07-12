"""Scene 对象的 live PhysX rigid views，供运行时状态读写复用。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True)
class SceneObjectStateView:
    """一个 Scene 对象可选的 root/body rigid views。"""

    root_view: object | None = None
    body_view: object | None = None
    body_names: tuple[str, ...] = ()
    velocity_capability: str = "complete"
    velocity_error: str | None = None

    def require_velocity_support(self, *, object_name: str) -> None:
        """对 unsupported view 给出稳定、可诊断的 snapshot 错误。"""

        if self.velocity_capability != "complete":
            detail = self.velocity_error or "live rigid view is unavailable"
            raise RuntimeError(
                f"Scene object {object_name!r} velocity snapshot is unsupported: {detail}"
            )

    def root_velocities(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回 root live PhysX 线/角速度，单位 m/s、rad/s。"""

        if self.root_view is None:
            return None
        velocities = _view_velocities(self.root_view, row_count=1)
        return velocities[0, :3], velocities[0, 3:]

    def body_velocities(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回全部 child body live PhysX 线/角速度。"""

        if self.body_view is None:
            return None
        velocities = _view_velocities(
            self.body_view,
            row_count=len(self.body_names),
        )
        return velocities[:, :3], velocities[:, 3:]

    def set_root_velocities(
        self,
        linear: np.ndarray,
        angular: np.ndarray,
    ) -> None:
        """写回完整的 root live PhysX 线速度与角速度。"""

        if self.root_view is None:
            raise RuntimeError("object root does not have a live rigid view")
        _set_view_velocity_row(
            self.root_view,
            index=0,
            linear=linear,
            angular=angular,
        )

    def set_body_velocities(
        self,
        *,
        body_index: int,
        linear: np.ndarray,
        angular: np.ndarray,
    ) -> None:
        """按 body index 写回 live PhysX 速度。"""

        if self.body_view is None:
            raise RuntimeError("object does not have a live body rigid view")
        _set_view_velocity_row(
            self.body_view,
            index=int(body_index),
            linear=linear,
            angular=angular,
        )


def create_scene_object_state_views(
    handles: Sequence[object],
) -> dict[str, SceneObjectStateView]:
    """在 ``world.reset`` 后为动态 Scene 对象创建并初始化 rigid views。"""

    specs = [_object_view_spec(handle) for handle in handles]
    if not any(root_path or body_paths for _, root_path, _, body_paths in specs):
        return {}
    active_specs = [spec for spec in specs if spec[1] is not None or spec[3]]
    try:
        from isaacsim.core.prims import RigidPrim
    except Exception as exc:
        return {
            name: SceneObjectStateView(
                body_names=body_names,
                velocity_capability="unsupported",
                velocity_error=f"RigidPrim import failed: {exc}",
            )
            for name, _root_path, body_names, _body_paths in active_specs
        }

    result: dict[str, SceneObjectStateView] = {}
    for name, root_path, body_names, body_paths in active_specs:
        try:
            root_view = (
                None
                if root_path is None
                else _create_rigid_view(
                    RigidPrim,
                    paths=(root_path,),
                    name=f"scene_object_{_identifier_suffix(name)}",
                )
            )
            body_view = (
                None
                if not body_paths
                else _create_rigid_view(
                    RigidPrim,
                    paths=body_paths,
                    name=f"scene_object_{_identifier_suffix(name)}_bodies",
                )
            )
            result[name] = SceneObjectStateView(
                root_view=root_view,
                body_view=body_view,
                body_names=body_names,
            )
        except Exception as exc:
            result[name] = SceneObjectStateView(
                body_names=body_names,
                velocity_capability="unsupported",
                velocity_error=str(exc),
            )
    return result


def _object_view_spec(
    handle: object,
) -> tuple[str, str | None, tuple[str, ...], tuple[str, ...]]:
    """从 RuntimeObjectHandle 提取 dynamic rigid view 路径。"""

    name = str(getattr(handle, "runtime_handle", None) or getattr(handle, "name", ""))
    kind = str(getattr(handle, "kind", ""))
    model = getattr(handle, "model", None)
    if kind == "rigid":
        if bool(getattr(model, "static", False)):
            return name, None, (), ()
        path = getattr(model, "prim_path", None)
        return name, (None if path is None else str(path)), (), ()
    if kind != "dynamic_chain":
        return name, None, (), ()
    bodies = tuple(model.get("bodies", ()) or ()) if isinstance(model, Mapping) else ()
    body_names = tuple(_prim_name(body) for body in bodies)
    body_paths = tuple(_prim_path(body) for body in bodies)
    return name, None, body_names, body_paths


def _create_rigid_view(rigid_prim_type, *, paths: Sequence[str], name: str) -> object:
    """创建 exact-path RigidPrim 并绑定 live physics handle。"""

    try:
        view = rigid_prim_type(
            prim_paths_expr=list(paths),
            name=name,
            reset_xform_properties=False,
        )
        initialize = getattr(view, "initialize", None)
        if callable(initialize):
            initialize()
        handle_valid = getattr(view, "is_physics_handle_valid", None)
        if callable(handle_valid) and not bool(handle_valid()):
            raise RuntimeError("RigidPrim did not acquire a live physics handle")
        return view
    except Exception as exc:
        raise RuntimeError(
            f"failed to create Scene object rigid view for paths {list(paths)!r}"
        ) from exc


def _view_velocities(view: object, *, row_count: int) -> np.ndarray:
    """读取 RigidPrim combined velocities，并标准化到 numpy。"""

    getter = getattr(view, "get_velocities", None)
    if not callable(getter):
        raise RuntimeError("RigidPrim view does not provide get_velocities")
    indices = np.arange(int(row_count), dtype=int)
    return _to_numpy(getter(indices=indices)).reshape(int(row_count), 6)


def _set_view_velocity_row(
    view: object,
    *,
    index: int,
    linear: np.ndarray,
    angular: np.ndarray,
) -> None:
    """用 combined API 写一行速度，避免 GPU pipeline 的 split API 限制。"""

    setter = getattr(view, "set_velocities", None)
    if not callable(setter):
        raise RuntimeError("RigidPrim view does not provide set_velocities")
    value = np.empty((1, 6), dtype=float)
    value[0, :3] = np.asarray(linear, dtype=float).reshape(3)
    value[0, 3:] = np.asarray(angular, dtype=float).reshape(3)
    setter(value, indices=np.asarray([int(index)], dtype=int))


def _to_numpy(value: object) -> np.ndarray:
    """兼容 numpy/torch/warp tensor-like 返回值。"""

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.asarray(current, dtype=float)


def _prim_path(prim: object) -> str:
    getter = getattr(prim, "GetPath", None)
    return str(getter() if callable(getter) else prim)


def _prim_name(prim: object) -> str:
    getter = getattr(prim, "GetName", None)
    return str(getter() if callable(getter) else _prim_path(prim).rsplit("/", 1)[-1])


def _identifier_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value)) or "object"


__all__ = ["SceneObjectStateView", "create_scene_object_state_views"]
