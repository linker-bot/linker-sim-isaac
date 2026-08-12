"""Scene 对象的 live physics rigid views，供运行时状态读写复用。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

import numpy as np

from linkerbot_sim.isaac.physics.backend import normalize_physics_backend
from linkerbot_sim.isaac.physics.core_api import create_rigid_prim_core_view
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


@dataclass(frozen=True)
class SceneObjectStateView:
    """一个 Scene 对象可选的 root/body rigid views。"""

    root_view: object | None = None
    body_view: object | None = None
    body_names: tuple[str, ...] = ()
    reference_body: str | None = None
    velocity_capability: str = "complete"
    velocity_error: str | None = None
    immutable_position: tuple[float, float, float] | None = None
    immutable_orientation_wxyz: tuple[float, float, float, float] | None = None
    position_atol: float = 1.0e-6
    orientation_atol: float = 1.0e-6

    def __post_init__(self) -> None:
        if (self.immutable_position is None) != (
            self.immutable_orientation_wxyz is None
        ):
            raise ValueError(
                "immutable object position and orientation must be provided together"
            )
        if self.reference_body is None:
            return
        if self.body_names.count(self.reference_body) != 1:
            raise ValueError(
                "Scene object reference_body must identify exactly one body; "
                f"got {self.reference_body!r} in {self.body_names!r}"
            )

    @property
    def has_live_root(self) -> bool:
        """返回对象级 root 是否有 live rigid-body row。"""

        return (
            self._root_view_and_index() is not None
            or self.immutable_position is not None
        )

    def require_velocity_support(self, *, object_name: str) -> None:
        """对 unsupported view 给出稳定、可诊断的 snapshot 错误。"""

        if self.velocity_capability != "complete":
            detail = self.velocity_error or "live rigid view is unavailable"
            raise RuntimeError(
                f"Scene object {object_name!r} velocity snapshot is unsupported: {detail}"
            )

    def root_velocities(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回 root live 线/角速度，单位 m/s、rad/s。"""

        target = self._root_view_and_index()
        if target is None:
            if self.immutable_position is None:
                return None
            zeros = np.zeros(3, dtype=float)
            return zeros.copy(), zeros
        view, index = target
        velocities = _view_velocities(view, indices=np.asarray([index], dtype=int))
        return velocities[0, :3], velocities[0, 3:]

    def body_velocities(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回全部 child body live 线/角速度。"""

        if self.body_view is None:
            return None
        velocities = _view_velocities(
            self.body_view,
            indices=np.arange(len(self.body_names), dtype=int),
        )
        return velocities[:, :3], velocities[:, 3:]

    @property
    def has_generalized_state(self) -> bool:
        """Whether this is a Newton owner-state dynamic-chain view."""

        if self.body_view is None:
            return False
        return all(
            callable(getattr(self.body_view, method, None))
            for method in (
                "get_generalized_state",
                "validate_generalized_state",
                "set_generalized_state",
            )
        )

    def generalized_state(self) -> dict[str, object] | None:
        """Read the single articulation's exact generalized owner q/qd."""

        if not self.has_generalized_state:
            return None
        assert self.body_view is not None
        q_names = tuple(str(item) for item in self.body_view.q_coordinate_names)
        qd_names = tuple(str(item) for item in self.body_view.qd_coordinate_names)
        signature = tuple(
            str(item) for item in self.body_view.generalized_coordinate_signature
        )
        if not signature or not q_names or not qd_names:
            raise RuntimeError("Newton dynamic-chain generalized identity is empty")
        q, qd = self.body_view.get_generalized_state(indices=np.asarray([0], dtype=int))
        q_values = tensor_like_to_numpy(q, dtype=float)
        qd_values = tensor_like_to_numpy(qd, dtype=float)
        if q_values.shape != (1, len(q_names)) or qd_values.shape != (
            1,
            len(qd_names),
        ):
            raise RuntimeError(
                "Newton dynamic-chain generalized owner state has an invalid shape"
            )
        if not np.all(np.isfinite(q_values)) or not np.all(np.isfinite(qd_values)):
            raise RuntimeError(
                "Newton dynamic-chain generalized owner state must be finite"
            )
        return {
            "generalized_signature": signature,
            "generalized_q_names": q_names,
            "generalized_qd_names": qd_names,
            "generalized_q": q_values[0].copy(),
            "generalized_qd": qd_values[0].copy(),
        }

    def preflight_generalized_state(
        self,
        *,
        signature: Sequence[str],
        q_names: Sequence[str],
        qd_names: Sequence[str],
        q: object,
        qd: object,
    ) -> None:
        """Fail closed on owner identity/value mismatch without mutation."""

        if not self.has_generalized_state:
            raise RuntimeError(
                "object does not expose a Newton generalized-state target"
            )
        assert self.body_view is not None
        self.body_view.validate_generalized_state(
            signature=signature,
            q_names=q_names,
            qd_names=qd_names,
            q=np.asarray(q, dtype=float).reshape(1, -1),
            qd=np.asarray(qd, dtype=float).reshape(1, -1),
            indices=np.asarray([0], dtype=int),
        )

    def set_generalized_state(
        self,
        *,
        signature: Sequence[str],
        q_names: Sequence[str],
        qd_names: Sequence[str],
        q: object,
        qd: object,
    ) -> None:
        """Restore exact owner state and let Newton derive bodies by FK."""

        if not self.has_generalized_state:
            raise RuntimeError(
                "object does not expose a Newton generalized-state target"
            )
        assert self.body_view is not None
        self.body_view.set_generalized_state(
            signature=signature,
            q_names=q_names,
            qd_names=qd_names,
            q=np.asarray(q, dtype=float).reshape(1, -1),
            qd=np.asarray(qd, dtype=float).reshape(1, -1),
            indices=np.asarray([0], dtype=int),
        )

    def root_world_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回对象级 live world pose；动态链使用显式 reference body。"""

        target = self._root_view_and_index()
        if target is None:
            if self.immutable_position is None:
                return None
            return (
                np.asarray(self.immutable_position, dtype=float).copy(),
                np.asarray(self.immutable_orientation_wxyz, dtype=float).copy(),
            )
        view, index = target
        positions, orientations = _view_world_poses(
            view,
            indices=np.asarray([index], dtype=int),
        )
        return positions[0], orientations[0]

    def body_world_poses(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回全部 child body 的 live world pose，四元数顺序为 wxyz。"""

        if self.body_view is None:
            return None
        return _view_world_poses(
            self.body_view,
            indices=np.arange(len(self.body_names), dtype=int),
        )

    def set_root_world_pose(
        self,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> None:
        """写回对象级 live world pose。"""

        target = self._root_view_and_index()
        if target is None:
            if self.immutable_position is not None:
                self._require_immutable_pose(position, orientation_wxyz)
                return
            raise RuntimeError("object root does not have a live rigid view")
        view, index = target
        _set_view_world_pose_row(
            view,
            index=index,
            position=position,
            orientation_wxyz=orientation_wxyz,
        )

    def set_body_world_pose(
        self,
        *,
        body_index: int,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> None:
        """按 body index 写回 live world pose。"""

        if self.body_view is None:
            raise RuntimeError("object does not have a live body rigid view")
        _set_view_world_pose_row(
            self.body_view,
            index=int(body_index),
            position=position,
            orientation_wxyz=orientation_wxyz,
        )

    def set_root_velocities(
        self,
        linear: np.ndarray,
        angular: np.ndarray,
    ) -> None:
        """写回完整的 root live 线速度与角速度。"""

        target = self._root_view_and_index()
        if target is None:
            if self.immutable_position is not None:
                self._require_zero_immutable_velocities(linear, angular)
                return
            raise RuntimeError("object root does not have a live rigid view")
        view, index = target
        _set_view_velocity_row(
            view,
            index=index,
            linear=linear,
            angular=angular,
        )

    def set_root_state(
        self,
        *,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        """原子恢复一个 live root rigid body。"""

        target = self._root_view_and_index()
        if target is None:
            if self.immutable_position is not None:
                self._require_immutable_pose(position, orientation_wxyz)
                self._require_zero_immutable_velocities(
                    linear_velocity,
                    angular_velocity,
                )
                return
            raise RuntimeError("object root does not have a live rigid view")
        view, index = target
        articulated_setter = getattr(view, "set_articulated_body_states", None)
        if callable(articulated_setter):
            articulated_setter(
                positions=np.asarray(position, dtype=float).reshape(1, 3),
                orientations=np.asarray(orientation_wxyz, dtype=float).reshape(1, 4),
                velocities=np.concatenate(
                    (
                        np.asarray(linear_velocity, dtype=float).reshape(1, 3),
                        np.asarray(angular_velocity, dtype=float).reshape(1, 3),
                    ),
                    axis=1,
                ),
                indices=np.asarray([index], dtype=int),
            )
            return
        self.set_root_world_pose(position, orientation_wxyz)
        self.set_root_velocities(linear_velocity, angular_velocity)

    def set_body_velocities(
        self,
        *,
        body_index: int,
        linear: np.ndarray,
        angular: np.ndarray,
    ) -> None:
        """按 body index 写回 live 速度。"""

        if self.body_view is None:
            raise RuntimeError("object does not have a live body rigid view")
        _set_view_velocity_row(
            self.body_view,
            index=int(body_index),
            linear=linear,
            angular=angular,
        )

    def set_body_states(
        self,
        *,
        body_indices: np.ndarray,
        positions: np.ndarray,
        orientations_wxyz: np.ndarray,
        linear_velocities: np.ndarray,
        angular_velocities: np.ndarray,
    ) -> None:
        """批量恢复 child bodies；Newton 动态链会同步其 generalized state。"""

        if self.body_view is None:
            raise RuntimeError("object does not have a live body rigid view")
        selected = np.asarray(body_indices, dtype=int).reshape(-1)
        row_count = int(selected.size)
        positions = np.asarray(positions, dtype=float).reshape(row_count, 3)
        orientations = np.asarray(orientations_wxyz, dtype=float).reshape(row_count, 4)
        linear = np.asarray(linear_velocities, dtype=float).reshape(row_count, 3)
        angular = np.asarray(angular_velocities, dtype=float).reshape(row_count, 3)
        velocities = np.concatenate((linear, angular), axis=1)
        articulated_setter = getattr(
            self.body_view, "set_articulated_body_states", None
        )
        if callable(articulated_setter):
            full_indices = np.arange(len(self.body_names), dtype=int)
            if np.any(selected < 0) or np.any(selected >= full_indices.size):
                raise IndexError("dynamic-chain body index is out of range")
            if np.unique(selected).size != selected.size:
                raise ValueError("dynamic-chain body indices must be unique")
            if selected.size != full_indices.size:
                try:
                    current_poses = self.body_world_poses()
                    current_velocities = self.body_velocities()
                except Exception as exc:
                    raise RuntimeError(
                        "Newton partial dynamic-chain restore cannot read complete current body state"
                    ) from exc
                if current_poses is None or current_velocities is None:
                    raise RuntimeError(
                        "Newton partial dynamic-chain restore requires complete current body state"
                    )
                merged_positions = (
                    np.asarray(current_poses[0], dtype=float)
                    .reshape(full_indices.size, 3)
                    .copy()
                )
                merged_orientations = (
                    np.asarray(current_poses[1], dtype=float)
                    .reshape(full_indices.size, 4)
                    .copy()
                )
                merged_velocities = np.concatenate(
                    (
                        np.asarray(current_velocities[0], dtype=float).reshape(
                            full_indices.size, 3
                        ),
                        np.asarray(current_velocities[1], dtype=float).reshape(
                            full_indices.size, 3
                        ),
                    ),
                    axis=1,
                )
                merged_positions[selected] = positions
                merged_orientations[selected] = orientations
                merged_velocities[selected] = velocities
                positions = merged_positions
                orientations = merged_orientations
                velocities = merged_velocities
                selected = full_indices
            articulated_setter(
                positions=positions,
                orientations=orientations,
                velocities=velocities,
                indices=selected,
            )
            return
        pose_setter = getattr(self.body_view, "set_world_poses", None)
        velocity_setter = getattr(self.body_view, "set_velocities", None)
        if not callable(pose_setter) or not callable(velocity_setter):
            raise RuntimeError("object body rigid view does not provide state setters")
        pose_setter(
            positions=positions,
            orientations=orientations,
            indices=selected,
        )
        velocity_setter(velocities, indices=selected)

    def _root_view_and_index(self) -> tuple[object, int] | None:
        """解析 rigid root row，动态链的 reference body 复用 body view。"""

        if self.root_view is not None:
            return self.root_view, 0
        if self.body_view is None or self.reference_body is None:
            return None
        return self.body_view, self.body_names.index(self.reference_body)

    def _require_immutable_pose(
        self,
        position: object,
        orientation_wxyz: object,
    ) -> None:
        """Accept an exact static no-op and reject a fake live relocation."""

        assert self.immutable_position is not None
        assert self.immutable_orientation_wxyz is not None
        requested_position = np.asarray(position, dtype=float).reshape(3)
        actual_position = np.asarray(self.immutable_position, dtype=float).reshape(3)
        requested_orientation = _normalized_quaternion(orientation_wxyz)
        actual_orientation = _normalized_quaternion(self.immutable_orientation_wxyz)
        orientation_error = min(
            float(np.linalg.norm(actual_orientation - requested_orientation)),
            float(np.linalg.norm(actual_orientation + requested_orientation)),
        )
        if not np.allclose(
            actual_position,
            requested_position,
            rtol=0.0,
            atol=float(self.position_atol),
        ) or orientation_error > float(self.orientation_atol):
            raise RuntimeError(
                "Newton cannot relocate an immutable Mirror object: "
                f"current_position={actual_position.tolist()}, "
                f"requested_position={requested_position.tolist()}, "
                f"orientation_error={orientation_error}"
            )

    @staticmethod
    def _require_zero_immutable_velocities(linear: object, angular: object) -> None:
        """Static Newton objects cannot acquire linear or angular velocity."""

        for name, value in (("linear", linear), ("angular", angular)):
            array = np.asarray(value, dtype=float).reshape(3)
            if not np.allclose(array, 0.0, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    "Newton cannot restore non-zero velocity on an "
                    f"immutable Mirror object: component={name}, "
                    f"value={array.tolist()}"
                )


def create_scene_object_state_views(
    handles: Sequence[object],
    *,
    physics_backend: object,
    stage: object | None = None,
    immutable_static: bool = False,
) -> dict[str, SceneObjectStateView]:
    """在 ``world.reset`` 后为动态 Scene 对象创建并初始化 rigid views。"""

    backend = normalize_physics_backend(physics_backend)
    specs = [_object_view_spec(handle) for handle in handles]
    static_specs = (
        [_immutable_static_view_spec(handle) for handle in handles]
        if immutable_static
        else []
    )
    static_specs = [spec for spec in static_specs if spec is not None]
    if not static_specs and not any(
        root_path or body_paths for _, root_path, _, body_paths, _, _ in specs
    ):
        return {}
    active_specs = [spec for spec in specs if spec[1] is not None or spec[3]]
    result: dict[str, SceneObjectStateView] = {}
    if static_specs:
        if stage is None:
            raise RuntimeError(
                "immutable Mirror object views require the active USD stage"
            )
        from linkerbot_sim.isaac.scene.pose import read_prim_world_pose

        for name, prim_path in static_specs:
            pose = read_prim_world_pose(stage, prim_path)
            if pose is None:
                raise RuntimeError(
                    f"immutable Mirror object prim is unavailable: {prim_path}"
                )
            result[name] = SceneObjectStateView(
                immutable_position=tuple(
                    float(value)
                    for value in np.asarray(pose[0], dtype=float).reshape(3)
                ),
                immutable_orientation_wxyz=tuple(
                    float(value) for value in _normalized_quaternion(pose[1])
                ),
            )
    for (
        name,
        root_path,
        body_names,
        body_paths,
        reference_body,
        articulation_path,
    ) in active_specs:
        try:
            root_view = (
                None
                if root_path is None
                else _create_rigid_view(
                    paths=(root_path,),
                    name=f"scene_object_{_identifier_suffix(name)}",
                    physics_backend=backend,
                )
            )
            body_view = (
                None
                if not body_paths
                else _create_dynamic_chain_or_rigid_view(
                    paths=body_paths,
                    articulation_path=articulation_path,
                    name=f"scene_object_{_identifier_suffix(name)}_bodies",
                    physics_backend=backend,
                )
            )
            result[name] = SceneObjectStateView(
                root_view=root_view,
                body_view=body_view,
                body_names=body_names,
                reference_body=reference_body,
            )
        except Exception as exc:
            if body_paths and (immutable_static or backend == "newton"):
                raise RuntimeError(
                    "failed to create required Newton dynamic-chain state view: "
                    f"object={name!r}, articulation={articulation_path!r}"
                ) from exc
            result[name] = SceneObjectStateView(
                body_names=body_names,
                reference_body=reference_body,
                velocity_capability="unsupported",
                velocity_error=str(exc),
            )
    return result


def _immutable_static_view_spec(handle: object) -> tuple[str, str] | None:
    """Return the authored pose source for a Newton static rigid object."""

    if str(getattr(handle, "kind", "")) != "rigid":
        return None
    model = getattr(handle, "model", None)
    if not bool(getattr(model, "static", False)):
        return None
    path = getattr(model, "prim_path", None)
    if path is None:
        raise RuntimeError(
            f"static object {getattr(handle, 'name', '<unnamed>')!r} has no prim path"
        )
    name = str(getattr(handle, "runtime_handle", None) or getattr(handle, "name", ""))
    if not name:
        raise RuntimeError("static object has no runtime name")
    return name, str(path)


def _object_view_spec(
    handle: object,
) -> tuple[
    str,
    str | None,
    tuple[str, ...],
    tuple[str, ...],
    str | None,
    str | None,
]:
    """从 RuntimeObjectHandle 提取 dynamic rigid view 路径。"""

    name = str(getattr(handle, "runtime_handle", None) or getattr(handle, "name", ""))
    kind = str(getattr(handle, "kind", ""))
    model = getattr(handle, "model", None)
    if kind == "rigid":
        if bool(getattr(model, "static", False)):
            return name, None, (), (), None, None
        path = getattr(model, "prim_path", None)
        return name, (None if path is None else str(path)), (), (), None, None
    if kind != "dynamic_chain":
        return name, None, (), (), None, None
    bodies = tuple(model.get("bodies", ()) or ()) if isinstance(model, Mapping) else ()
    body_names = tuple(_prim_name(body) for body in bodies)
    body_paths = tuple(_prim_path(body) for body in bodies)
    root = model.get("root") if isinstance(model, Mapping) else None
    articulation_path = None if root is None else _prim_path(root)
    if articulation_path is None:
        config_path = getattr(getattr(handle, "config", None), "prim_path", None)
        articulation_path = None if config_path is None else str(config_path)
    if articulation_path is None and body_paths:
        marker = "/Bodies/"
        if all(marker in path for path in body_paths):
            candidates = {path.split(marker, 1)[0] for path in body_paths}
            if len(candidates) == 1:
                articulation_path = candidates.pop()
    state_summary = getattr(handle, "state_summary", None)
    reference_body = getattr(state_summary, "reference_body", None)
    if reference_body is not None:
        reference_body = str(reference_body)
    return name, None, body_names, body_paths, reference_body, articulation_path


def _create_dynamic_chain_or_rigid_view(
    *,
    paths: Sequence[str],
    articulation_path: str | None,
    name: str,
    physics_backend: str,
) -> object:
    """Use a generalized-coordinate owner for Newton chain bodies."""

    if physics_backend != "newton":
        return _create_rigid_view(
            paths=paths,
            name=name,
            physics_backend=physics_backend,
        )
    if articulation_path is None:
        raise RuntimeError(
            "Newton dynamic-chain view requires its articulation root path"
        )
    from linkerbot_sim.isaac.physics.core_api import RigidPrimCoreView
    from linkerbot_sim.isaac.physics.manager import active_physics_manager
    from linkerbot_sim.isaac.physics.newton.views import NewtonDynamicChainView

    manager = active_physics_manager()
    assert manager is not None
    view = RigidPrimCoreView(
        NewtonDynamicChainView(
            manager,
            articulation_paths=(articulation_path,),
            body_paths_by_env=(tuple(str(path) for path in paths),),
            world_indices=(0,),
            name=name,
        ),
        physics_backend="newton",
    )
    handle_valid = getattr(view, "is_physics_handle_valid", None)
    if callable(handle_valid) and not bool(handle_valid()):
        raise RuntimeError("Newton dynamic-chain view is not live")
    return view


def _create_rigid_view(
    *, paths: Sequence[str], name: str, physics_backend: str
) -> object:
    """创建 exact-path RigidPrim 并绑定 live physics handle。"""

    try:
        view = create_rigid_prim_core_view(
            paths=paths,
            name=name,
            physics_backend=physics_backend,
        )
        handle_valid = getattr(view, "is_physics_handle_valid", None)
        if callable(handle_valid) and not bool(handle_valid()):
            raise RuntimeError("RigidPrim did not acquire a live physics handle")
        return view
    except Exception as exc:
        raise RuntimeError(
            f"failed to create Scene object rigid view for paths {list(paths)!r}"
        ) from exc


def _view_world_poses(
    view: object,
    *,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 live world pose，并标准化为 Nx3 与 Nx4(wxyz)。"""

    getter = getattr(view, "get_world_poses", None)
    if not callable(getter):
        raise RuntimeError("RigidPrim view does not provide get_world_poses")
    row_count = int(indices.size)
    positions, orientations = getter(indices=indices)
    return (
        tensor_like_to_numpy(positions, dtype=float).reshape(row_count, 3),
        tensor_like_to_numpy(orientations, dtype=float).reshape(row_count, 4),
    )


def _view_velocities(view: object, *, indices: np.ndarray) -> np.ndarray:
    """读取 RigidPrim combined velocities，并标准化到 numpy。"""

    getter = getattr(view, "get_velocities", None)
    if not callable(getter):
        raise RuntimeError("RigidPrim view does not provide get_velocities")
    return tensor_like_to_numpy(getter(indices=indices), dtype=float).reshape(
        int(indices.size), 6
    )


def _set_view_world_pose_row(
    view: object,
    *,
    index: int,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
) -> None:
    """通过 legacy/Experimental 共用签名写一行 world pose。"""

    setter = getattr(view, "set_world_poses", None)
    if not callable(setter):
        raise RuntimeError("RigidPrim view does not provide set_world_poses")
    setter(
        positions=np.asarray(position, dtype=float).reshape(1, 3),
        orientations=np.asarray(orientation_wxyz, dtype=float).reshape(1, 4),
        indices=np.asarray([int(index)], dtype=int),
    )


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


def _prim_path(prim: object) -> str:
    getter = getattr(prim, "GetPath", None)
    return str(getter() if callable(getter) else prim)


def _prim_name(prim: object) -> str:
    getter = getattr(prim, "GetName", None)
    return str(getter() if callable(getter) else _prim_path(prim).rsplit("/", 1)[-1])


def _identifier_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value)) or "object"


def _normalized_quaternion(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("object orientation quaternion must be finite and non-zero")
    return quaternion / norm


__all__ = ["SceneObjectStateView", "create_scene_object_state_views"]
