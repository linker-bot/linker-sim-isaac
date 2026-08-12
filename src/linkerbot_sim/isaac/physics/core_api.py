"""Isaac legacy Core 与 Experimental Core 的内部迁移桥。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from linkerbot_sim.isaac.physics.backend import (
    active_physics_backend,
    normalize_physics_backend,
)
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


_EXPERIMENTAL_CORE_ENV = "LINKERBOT_EXPERIMENTAL_CORE"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


@dataclass
class ExperimentalArticulationAction:
    """不依赖 legacy Core 的单 articulation 控制 action。"""

    joint_positions: object | None = None
    joint_velocities: object | None = None
    joint_efforts: object | None = None
    joint_indices: object | None = None


def use_experimental_core(*, physics_backend: object | None = None) -> bool:
    """判断是否必须使用 Isaac 6 Experimental Core prim views。"""

    configured = os.getenv(_EXPERIMENTAL_CORE_ENV)
    requested = False
    if configured is not None:
        normalized = configured.strip().lower()
        if normalized not in _TRUE_VALUES | _FALSE_VALUES:
            raise ValueError(
                f"{_EXPERIMENTAL_CORE_ENV} must be a boolean value, got {configured!r}"
            )
        requested = normalized in _TRUE_VALUES
    backend = (
        active_physics_backend()
        if physics_backend is None
        else normalize_physics_backend(physics_backend)
    )
    return requested or backend == "newton"


class ArticulationCoreView:
    """以 legacy joint API 暴露 Experimental ``Articulation``。"""

    def __init__(self, view: object, *, physics_backend: object | None = None) -> None:
        self._view = view
        self.physics_backend = (
            None
            if physics_backend is None
            else normalize_physics_backend(physics_backend)
        )

    @property
    def raw_view(self) -> object:
        return self._view

    @property
    def dof_names(self) -> list[str]:
        return list(getattr(self._view, "dof_names"))

    @property
    def num_dof(self) -> int:
        return int(getattr(self._view, "num_dofs"))

    @property
    def num_dofs(self) -> int:
        return self.num_dof

    @property
    def count(self) -> int:
        try:
            return len(self._view)  # type: ignore[arg-type]
        except TypeError:
            return int(getattr(self._view, "count"))

    def initialize(self, *args: object, **kwargs: object) -> None:
        """Experimental view 由 SimulationManager 生命周期自动绑定 tensor entity。"""

    def is_physics_handle_valid(self) -> bool:
        checker = getattr(self._view, "is_physics_tensor_entity_valid", None)
        return (
            bool(checker())
            if callable(checker)
            else bool(getattr(self._view, "valid", True))
        )

    def get_joint_positions(
        self,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> np.ndarray:
        if _is_newton_runtime_view(self._view):
            return _to_numpy(
                self._view.get_dof_positions(
                    indices=_indices(indices), dof_indices=_indices(joint_indices)
                )
            )
        if self.physics_backend == "newton" and (
            indices is not None or joint_indices is not None
        ):
            return _selected_matrix(
                _to_numpy(self._view.get_dof_positions()),
                indices=indices,
                columns=joint_indices,
                label="Newton articulation positions",
            )
        return _to_numpy(
            self._view.get_dof_positions(
                indices=_indices(indices), dof_indices=_indices(joint_indices)
            )
        )

    def get_joint_velocities(
        self,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> np.ndarray:
        if _is_newton_runtime_view(self._view):
            return _to_numpy(
                self._view.get_dof_velocities(
                    indices=_indices(indices), dof_indices=_indices(joint_indices)
                )
            )
        if self.physics_backend == "newton" and (
            indices is not None or joint_indices is not None
        ):
            return _selected_matrix(
                _to_numpy(self._view.get_dof_velocities()),
                indices=indices,
                columns=joint_indices,
                label="Newton articulation velocities",
            )
        return _to_numpy(
            self._view.get_dof_velocities(
                indices=_indices(indices), dof_indices=_indices(joint_indices)
            )
        )

    def set_joint_positions(
        self,
        positions: object,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> None:
        if _is_newton_runtime_view(self._view):
            self._view.set_dof_positions(
                _values(positions),
                indices=_indices(indices),
                dof_indices=_indices(joint_indices),
            )
            return
        if self.physics_backend == "newton":
            rows = np.arange(self.count, dtype=np.int32) if indices is None else indices
            positions = _merge_selected_matrix(
                _to_numpy(self._view.get_dof_positions()),
                positions,
                indices=rows,
                columns=joint_indices,
                label="Newton articulation positions",
            )
            # Isaac 6.0.1 Experimental Core 的非零 indexed teleport 在 Newton 下不能
            # 可靠读回，而且高级 setter 会把 state 同时写进 drive target。冷路径在 CPU
            # 合并完整 batch 后只调用 Newton tensor state setter，既保留未选 env，也不
            # 污染控制目标。
            self._set_newton_tensor_dof_state(positions=positions)
            return
        self._view.set_dof_positions(
            _values(positions),
            indices=_indices(indices),
            dof_indices=_indices(joint_indices),
        )

    def set_joint_velocities(
        self,
        velocities: object,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> None:
        if _is_newton_runtime_view(self._view):
            self._view.set_dof_velocities(
                _values(velocities),
                indices=_indices(indices),
                dof_indices=_indices(joint_indices),
            )
            return
        if self.physics_backend == "newton":
            rows = np.arange(self.count, dtype=np.int32) if indices is None else indices
            velocities = _merge_selected_matrix(
                _to_numpy(self._view.get_dof_velocities()),
                velocities,
                indices=rows,
                columns=joint_indices,
                label="Newton articulation velocities",
            )
            self._set_newton_tensor_dof_state(velocities=velocities)
            return
        self._view.set_dof_velocities(
            _values(velocities),
            indices=_indices(indices),
            dof_indices=_indices(joint_indices),
        )

    def set_joint_position_targets(
        self,
        positions: object,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> None:
        self._view.set_dof_position_targets(
            _values(positions),
            indices=_indices(indices),
            dof_indices=_indices(joint_indices),
        )

    def get_joint_position_targets(
        self,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> np.ndarray:
        if _is_newton_runtime_view(self._view):
            return _to_numpy(
                self._view.get_dof_position_targets(
                    indices=_indices(indices), dof_indices=_indices(joint_indices)
                )
            )
        if self.physics_backend == "newton" and (
            indices is not None or joint_indices is not None
        ):
            return _selected_matrix(
                _to_numpy(self._view.get_dof_position_targets()),
                indices=indices,
                columns=joint_indices,
                label="Newton articulation position targets",
            )
        return _to_numpy(
            self._view.get_dof_position_targets(
                indices=_indices(indices), dof_indices=_indices(joint_indices)
            )
        )

    def set_joint_velocity_targets(
        self,
        velocities: object,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> None:
        self._view.set_dof_velocity_targets(
            _values(velocities),
            indices=_indices(indices),
            dof_indices=_indices(joint_indices),
        )

    def set_joint_efforts(
        self,
        efforts: object,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> None:
        """Write indexed articulation effort targets through the public core facade."""

        self._view.set_dof_efforts(
            _values(efforts),
            indices=_indices(indices),
            dof_indices=_indices(joint_indices),
        )

    def get_joint_velocity_targets(
        self,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> np.ndarray:
        if _is_newton_runtime_view(self._view):
            return _to_numpy(
                self._view.get_dof_velocity_targets(
                    indices=_indices(indices), dof_indices=_indices(joint_indices)
                )
            )
        if self.physics_backend == "newton" and (
            indices is not None or joint_indices is not None
        ):
            return _selected_matrix(
                _to_numpy(self._view.get_dof_velocity_targets()),
                indices=indices,
                columns=joint_indices,
                label="Newton articulation velocity targets",
            )
        return _to_numpy(
            self._view.get_dof_velocity_targets(
                indices=_indices(indices), dof_indices=_indices(joint_indices)
            )
        )

    def _set_newton_tensor_dof_state(
        self,
        *,
        positions: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> None:
        """写 Newton generalized state，绕开 Experimental setter 的 target 副作用。"""

        tensor_view = getattr(self._view, "_physics_articulation_view", None)
        if tensor_view is None:
            raise RuntimeError(
                "Newton articulation state restore requires a live articulation tensor view"
            )
        if (positions is None) == (velocities is None):
            raise ValueError(
                "exactly one Newton articulation state tensor must be provided"
            )
        values = positions if positions is not None else velocities
        assert values is not None
        matrix = np.ascontiguousarray(values, dtype=np.float32)
        expected_shape = (int(tensor_view.count), int(tensor_view.max_dofs))
        if matrix.shape != expected_shape:
            raise ValueError(
                "Newton articulation state tensor must match the raw tensor view: "
                f"expected={expected_shape}, actual={matrix.shape}"
            )
        if positions is not None:
            current = tensor_view.get_dof_positions()
            tensor_values, all_indices = _newton_warp_full_batch(
                matrix,
                device=getattr(current, "device", None),
            )
            tensor_view.set_dof_positions(tensor_values, all_indices)
        else:
            current = tensor_view.get_dof_velocities()
            tensor_values, all_indices = _newton_warp_full_batch(
                matrix,
                device=getattr(current, "device", None),
            )
            tensor_view.set_dof_velocities(tensor_values, all_indices)

    def get_applied_joint_efforts(
        self,
        *,
        indices: object | None = None,
        joint_indices: object | None = None,
    ) -> np.ndarray:
        return _to_numpy(
            self._view.get_dof_efforts(
                indices=_indices(indices), dof_indices=_indices(joint_indices)
            )
        )

    def apply_action(self, action: object) -> None:
        """把 legacy ``ArticulationActions`` 字段映射到 Experimental targets。"""

        joint_indices = getattr(action, "joint_indices", None)
        positions = getattr(action, "joint_positions", None)
        velocities = getattr(action, "joint_velocities", None)
        efforts = getattr(action, "joint_efforts", None)
        if positions is not None:
            self.set_joint_position_targets(positions, joint_indices=joint_indices)
        if velocities is not None:
            self.set_joint_velocity_targets(velocities, joint_indices=joint_indices)
        if efforts is not None:
            self._view.set_dof_efforts(
                _values(efforts), dof_indices=_indices(joint_indices)
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._view, name)


class ExperimentalArticulationController:
    """把 Experimental articulation 的 DOF API 还原为 legacy controller 契约。"""

    def __init__(self, view: object) -> None:
        self._view = view

    def get_gains(self) -> tuple[np.ndarray, np.ndarray]:
        stiffnesses, dampings = self._view.get_dof_gains()
        return _single_values(stiffnesses), _single_values(dampings)

    def set_gains(
        self,
        kps: object | None = None,
        kds: object | None = None,
        save_to_usd: bool = False,
    ) -> None:
        if save_to_usd:
            raise ValueError(
                "Experimental articulation controller does not support save_to_usd=True"
            )
        dof_indices = self._direct_controllable_dof_indices()
        self._view.set_dof_gains(
            stiffnesses=(
                None
                if kps is None
                else self._selected_direct_values(kps, dof_indices=dof_indices)
            ),
            dampings=(
                None
                if kds is None
                else self._selected_direct_values(kds, dof_indices=dof_indices)
            ),
            dof_indices=dof_indices,
        )

    def set_max_efforts(
        self,
        values: object,
        joint_indices: object | None = None,
    ) -> None:
        dof_indices = self._direct_model_write_dof_indices(joint_indices)
        if joint_indices is None:
            selected_values = self._selected_direct_values(
                values, dof_indices=dof_indices
            )
        else:
            selected_values = _values(values)
        self._view.set_dof_max_efforts(
            selected_values,
            dof_indices=dof_indices,
        )

    def _direct_controllable_dof_indices(self) -> np.ndarray | None:
        """返回 Newton model mutation 已审计的 command 列。"""

        if not _is_newton_runtime_view(self._view):
            return None
        names = tuple(str(name) for name in self._view.controllable_dof_names)
        if not names:
            raise RuntimeError(
                "Newton controller requires an audited controllable DOF set"
            )
        by_name = {str(name): index for index, name in enumerate(self._view.dof_names)}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise RuntimeError(
                f"Newton controllable DOFs are absent from the view: {missing}"
            )
        return np.asarray([by_name[name] for name in names], dtype=np.int32)

    def _direct_model_write_dof_indices(
        self,
        joint_indices: object | None,
    ) -> np.ndarray | None:
        """把 Newton model/drive 写入限制在已审计 command DOF 内。

        equality follower 不属于 controller writer；允许 legacy 全 DOF API 穿透会让 follower
        同时受 DriveAPI 与 native equality 驱动，因此这里必须 fail closed。
        """

        requested = _indices(joint_indices)
        audited = self._direct_controllable_dof_indices()
        if audited is None:
            return requested
        if requested is None:
            return audited
        audited_set = {int(index) for index in audited}
        forbidden = [int(index) for index in requested if int(index) not in audited_set]
        if forbidden:
            names = tuple(str(name) for name in self._view.dof_names)
            labels = [
                names[index] if 0 <= index < len(names) else f"index:{index}"
                for index in forbidden
            ]
            raise RuntimeError(
                "Newton model writes cannot target DOFs outside the "
                f"audited controllable set: {labels}"
            )
        return requested

    def _selected_direct_values(
        self,
        values: object,
        *,
        dof_indices: np.ndarray | None,
    ) -> np.ndarray:
        """写 Newton model 前，从 legacy 全 DOF 向量切出 command 列。"""

        array = _values(values)
        if dof_indices is None:
            return array
        num_dofs = len(self._view.dof_names)
        if array.ndim == 0 or array.shape[-1] != num_dofs:
            raise ValueError(
                "Newton controller values must contain the complete legacy "
                f"DOF axis before audited slicing: expected={num_dofs}, "
                f"actual={array.shape}"
            )
        return np.take(array, dof_indices, axis=-1)

    def get_max_efforts(self) -> np.ndarray:
        return _single_values(self._view.get_dof_max_efforts())

    def switch_control_mode(self, mode: str) -> None:
        self._view.switch_dof_control_mode(
            str(mode),
            dof_indices=self._direct_model_write_dof_indices(None),
        )

    def switch_dof_control_mode(self, dof_index: int, mode: str) -> None:
        self._view.switch_dof_control_mode(
            str(mode),
            dof_indices=self._direct_model_write_dof_indices([dof_index]),
        )

    def set_effort_modes(
        self,
        mode: str,
        joint_indices: object | None = None,
    ) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"force", "acceleration"}:
            raise ValueError(
                "Experimental articulation effort mode must be 'force' or "
                f"'acceleration', got {mode!r}"
            )
        self._view.set_dof_drive_types(
            normalized,
            dof_indices=self._direct_model_write_dof_indices(joint_indices),
        )

    def get_effort_modes(self) -> list[str]:
        modes = self._view.get_dof_drive_types(indices=[0])
        return list(modes[0])


class SingleArticulationCoreView(ArticulationCoreView):
    """以 legacy ``SingleArticulation`` 形状暴露一个 Experimental articulation。"""

    requires_scene_registration = False

    def __init__(
        self,
        view: object,
        *,
        prim_path: str,
        name: str,
        physics_backend: object,
    ) -> None:
        super().__init__(view, physics_backend=physics_backend)
        self.prim_path = str(prim_path)
        self.name = str(name)
        self.physics_backend = normalize_physics_backend(physics_backend)
        self.supports_per_link_gravity = self.physics_backend != "newton"
        self._controller = ExperimentalArticulationController(view)
        # 少量日志/控制兼容代码会读取 legacy 私有 view；指回 facade 可保留一维返回契约。
        self._articulation_view = self

    @property
    def handles_initialized(self) -> bool:
        return self.is_physics_handle_valid()

    def get_articulation_controller(self) -> ExperimentalArticulationController:
        return self._controller

    def get_joint_positions(self, joint_indices: object | None = None) -> np.ndarray:
        return _single_values(super().get_joint_positions(joint_indices=joint_indices))

    def get_joint_velocities(self, joint_indices: object | None = None) -> np.ndarray:
        return _single_values(super().get_joint_velocities(joint_indices=joint_indices))

    def set_joint_positions(
        self,
        positions: object,
        joint_indices: object | None = None,
    ) -> None:
        super().set_joint_positions(positions, joint_indices=joint_indices)

    def set_joint_velocities(
        self,
        velocities: object,
        joint_indices: object | None = None,
    ) -> None:
        super().set_joint_velocities(velocities, joint_indices=joint_indices)

    def get_joint_position_targets(
        self, joint_indices: object | None = None
    ) -> np.ndarray:
        return _single_values(
            super().get_joint_position_targets(joint_indices=joint_indices)
        )

    def get_joint_velocity_targets(
        self, joint_indices: object | None = None
    ) -> np.ndarray:
        return _single_values(
            super().get_joint_velocity_targets(joint_indices=joint_indices)
        )

    def set_joint_efforts(
        self,
        efforts: object,
        joint_indices: object | None = None,
    ) -> None:
        self._view.set_dof_efforts(
            _values(efforts), dof_indices=_indices(joint_indices)
        )

    def get_applied_joint_efforts(
        self, joint_indices: object | None = None
    ) -> np.ndarray:
        return _single_values(
            super().get_applied_joint_efforts(joint_indices=joint_indices)
        )

    def get_measured_joint_efforts(
        self, joint_indices: object | None = None
    ) -> np.ndarray:
        if self.physics_backend == "newton":
            raise RuntimeError(
                "Newton 1.2.1 does not implement projected/measured joint efforts"
            )
        return _single_values(
            self._view.get_dof_projected_joint_forces(
                dof_indices=_indices(joint_indices)
            )
        )

    def get_max_efforts(self) -> np.ndarray:
        return self._controller.get_max_efforts()

    def disable_gravity(self) -> None:
        if not self.supports_per_link_gravity:
            raise RuntimeError(
                "Newton 1.2.1 does not implement runtime per-link gravity "
                "disabling; robot gravity policy must be projected to MuJoCo "
                "gravcomp before Newton model finalization"
            )
        self._view.set_link_enabled_gravities(False)

    def enable_gravity(self) -> None:
        if not self.supports_per_link_gravity:
            raise RuntimeError(
                "Newton 1.2.1 does not implement runtime per-link gravity "
                "enabling; robot gravity policy must be projected before direct "
                "model finalization"
            )
        self._view.set_link_enabled_gravities(True)


class RigidPrimCoreView:
    """把 Experimental ``RigidPrim`` 的 split velocity API 还原为 legacy Nx6。"""

    def __init__(self, view: object, *, physics_backend: object | None = None) -> None:
        self._view = view
        self.physics_backend = (
            None
            if physics_backend is None
            else normalize_physics_backend(physics_backend)
        )

    @property
    def raw_view(self) -> object:
        return self._view

    @property
    def count(self) -> int:
        try:
            return len(self._view)  # type: ignore[arg-type]
        except TypeError:
            return int(getattr(self._view, "count"))

    def initialize(self, *args: object, **kwargs: object) -> None:
        """Experimental view 不需要 legacy 显式初始化。"""

    def is_physics_handle_valid(self) -> bool:
        checker = getattr(self._view, "is_physics_tensor_entity_valid", None)
        return (
            bool(checker())
            if callable(checker)
            else bool(getattr(self._view, "valid", True))
        )

    def get_world_poses(
        self, *, indices: object | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if _is_newton_runtime_view(self._view):
            positions, orientations = self._view.get_world_poses(
                indices=_indices(indices)
            )
            return _to_numpy(positions), _to_numpy(orientations)
        if self.physics_backend == "newton":
            tensor_view = getattr(self._view, "_physics_rigid_body_view", None)
            if tensor_view is not None:
                transforms = _to_numpy(tensor_view.get_transforms())
                positions = transforms[:, :3]
                orientations = transforms[:, [6, 3, 4, 5]]
            else:
                positions, orientations = self._view.get_world_poses()
                positions = _to_numpy(positions)
                orientations = _to_numpy(orientations)
            return (
                _selected_matrix(
                    positions,
                    indices=indices,
                    columns=None,
                    label="Newton rigid positions",
                ),
                _selected_matrix(
                    orientations,
                    indices=indices,
                    columns=None,
                    label="Newton rigid orientations",
                ),
            )
        positions, orientations = self._view.get_world_poses(indices=_indices(indices))
        return _to_numpy(positions), _to_numpy(orientations)

    def set_world_poses(
        self,
        *,
        positions: object | None = None,
        orientations: object | None = None,
        indices: object | None = None,
    ) -> None:
        if _is_newton_runtime_view(self._view):
            self._view.set_world_poses(
                positions=None if positions is None else _values(positions),
                orientations=(None if orientations is None else _values(orientations)),
                indices=_indices(indices),
            )
            return
        if self.physics_backend == "newton" and indices is not None:
            current_positions, current_orientations = self.get_world_poses()
            merged_positions = (
                None
                if positions is None
                else _merge_selected_matrix(
                    current_positions,
                    positions,
                    indices=indices,
                    columns=None,
                    label="Newton rigid positions",
                )
            )
            merged_orientations = (
                None
                if orientations is None
                else _merge_selected_matrix(
                    current_orientations,
                    orientations,
                    indices=indices,
                    columns=None,
                    label="Newton rigid orientations",
                )
            )
            if not self._set_newton_tensor_world_poses(
                positions=merged_positions,
                orientations=merged_orientations,
            ):
                self._view.set_world_poses(
                    positions=(
                        None if merged_positions is None else _values(merged_positions)
                    ),
                    orientations=(
                        None
                        if merged_orientations is None
                        else _values(merged_orientations)
                    ),
                )
            return
        self._view.set_world_poses(
            positions=None if positions is None else _values(positions),
            orientations=None if orientations is None else _values(orientations),
            indices=_indices(indices),
        )

    def get_velocities(self, *, indices: object | None = None) -> np.ndarray:
        if _is_newton_runtime_view(self._view):
            linear, angular = self._view.get_velocities(indices=_indices(indices))
            return np.concatenate((_to_numpy(linear), _to_numpy(angular)), axis=-1)
        if self.physics_backend == "newton":
            tensor_view = getattr(self._view, "_physics_rigid_body_view", None)
            if tensor_view is not None:
                # Newton 1.2.1 body twist 与 Isaac tensor 合同都采用 [linear, angular]；
                # bundled tensor kernel 中遗留的 angular-first 注释不能作为 ABI 依据。
                combined = _to_numpy(tensor_view.get_velocities())
            else:
                linear, angular = self._view.get_velocities()
                combined = np.concatenate(
                    (_to_numpy(linear), _to_numpy(angular)), axis=-1
                )
            return _selected_matrix(
                combined,
                indices=indices,
                columns=None,
                label="Newton rigid velocities",
            )
        linear, angular = self._view.get_velocities(indices=_indices(indices))
        return np.concatenate((_to_numpy(linear), _to_numpy(angular)), axis=-1)

    def set_velocities(
        self,
        velocities: object,
        *,
        indices: object | None = None,
    ) -> None:
        combined = _values(velocities)
        if combined.ndim == 1:
            combined = combined.reshape(1, -1)
        if combined.ndim != 2 or combined.shape[1] != 6:
            raise ValueError(
                f"rigid velocities must have shape (N, 6), got {combined.shape}"
            )
        if _is_newton_runtime_view(self._view):
            self._view.set_velocities(
                combined[:, :3],
                combined[:, 3:],
                indices=_indices(indices),
            )
            return
        if self.physics_backend == "newton":
            if indices is not None:
                combined = _merge_selected_matrix(
                    self.get_velocities(),
                    combined,
                    indices=indices,
                    columns=None,
                    label="Newton rigid velocities",
                )
            if not self._set_newton_tensor_velocities(combined):
                self._view.set_velocities(combined[:, :3], combined[:, 3:])
            return
        self._view.set_velocities(
            combined[:, :3], combined[:, 3:], indices=_indices(indices)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._view, name)

    def set_articulated_body_states(
        self,
        *,
        positions: object,
        orientations: object,
        velocities: object,
        indices: object,
    ) -> None:
        """原子恢复 articulated rigid bodies，并保持其它 Newton articulation 不变。

        Newton 的 rigid-body pose setter只同步 free-joint 坐标，不能从一组 child body
        poses 重建链内 generalized coordinates。这里在写入 body_q/body_qd 后只对本次
        view 命中的 articulation 执行 IK；IK 输出以写入前的全局 q/qd 为底，因此机器人
        和未选 env 的 articulation 不会被对象恢复污染。
        """

        selected = _indices(indices)
        assert selected is not None
        if _is_newton_runtime_view(self._view):
            generalized_setter = getattr(
                self._view, "set_articulated_body_states", None
            )
            if callable(generalized_setter):
                generalized_setter(
                    positions=positions,
                    orientations=orientations,
                    velocities=velocities,
                    indices=selected,
                )
                return
            # 普通 Newton rigid view 只绑定 world-root FREE body，并同时更新 maximal 与
            # generalized state；只有 dynamic chain 才需要上面的 selected IK 恢复路径。
            self.set_world_poses(
                positions=positions,
                orientations=orientations,
                indices=selected,
            )
            self.set_velocities(velocities, indices=selected)
            return
        if self.physics_backend != "newton":
            self.set_world_poses(
                positions=positions,
                orientations=orientations,
                indices=selected,
            )
            self.set_velocities(velocities, indices=selected)
            return

        tensor_view = getattr(self._view, "_physics_rigid_body_view", None)
        if tensor_view is None:
            raise RuntimeError(
                "Newton articulated body restore requires a live rigid-body tensor view"
            )
        self._set_newton_articulated_body_states(
            tensor_view=tensor_view,
            positions=positions,
            orientations=orientations,
            velocities=velocities,
            indices=selected,
        )

    def _set_newton_tensor_world_poses(
        self,
        *,
        positions: np.ndarray | None,
        orientations: np.ndarray | None,
    ) -> bool:
        """直接写 Newton tensor transforms，绕开 Experimental Warp 高级索引。"""

        tensor_view = getattr(self._view, "_physics_rigid_body_view", None)
        if tensor_view is None:
            return False
        current = tensor_view.get_transforms()
        transforms = np.asarray(_to_numpy(current), dtype=np.float32).copy()
        if positions is not None:
            transforms[:, :3] = np.asarray(positions, dtype=np.float32)
        if orientations is not None:
            transforms[:, [6, 3, 4, 5]] = np.asarray(orientations, dtype=np.float32)
        values, all_indices = _newton_warp_full_batch(
            transforms,
            device=getattr(current, "device", None),
        )
        tensor_view.set_transforms(values, all_indices)
        return True

    def _set_newton_tensor_velocities(self, velocities: np.ndarray) -> bool:
        """直接写 Newton tensor velocity matrix，并保持 env/body 行顺序。"""

        tensor_view = getattr(self._view, "_physics_rigid_body_view", None)
        if tensor_view is None:
            return False
        current = tensor_view.get_velocities()
        combined = np.asarray(velocities, dtype=np.float32)
        values, all_indices = _newton_warp_full_batch(
            combined,
            device=getattr(current, "device", None),
        )
        tensor_view.set_velocities(values, all_indices)
        return True

    def _set_newton_articulated_body_states(
        self,
        *,
        tensor_view: object,
        positions: object,
        orientations: object,
        velocities: object,
        indices: np.ndarray,
    ) -> None:
        """直接更新 Newton maximal state，再受限重建对应 generalized state。"""

        import newton
        import warp as wp

        current_transforms_tensor = tensor_view.get_transforms()
        current_velocities_tensor = tensor_view.get_velocities()
        current_transforms = np.asarray(
            _to_numpy(current_transforms_tensor), dtype=np.float32
        ).copy()
        current_velocities = np.asarray(
            _to_numpy(current_velocities_tensor), dtype=np.float32
        ).copy()
        if current_transforms.ndim != 2 or current_transforms.shape[1] != 7:
            raise RuntimeError("Newton rigid transform tensor must have shape (N, 7)")
        if current_velocities.shape != (current_transforms.shape[0], 6):
            raise RuntimeError("Newton rigid velocity tensor must have shape (N, 6)")
        if np.any(indices < 0) or np.any(indices >= current_transforms.shape[0]):
            raise IndexError("Newton articulated body indices are out of range")
        if np.unique(indices).size != indices.size:
            raise ValueError("Newton articulated body indices must be unique")

        row_count = int(indices.size)
        requested_positions = np.asarray(positions, dtype=np.float32).reshape(
            row_count, 3
        )
        requested_orientations = np.asarray(orientations, dtype=np.float32).reshape(
            row_count, 4
        )
        requested_velocities = np.asarray(velocities, dtype=np.float32).reshape(
            row_count, 6
        )
        quaternion_norms = np.linalg.norm(requested_orientations, axis=1)
        if (
            not np.all(np.isfinite(requested_positions))
            or not np.all(np.isfinite(requested_orientations))
            or not np.all(np.isfinite(requested_velocities))
            or np.any(quaternion_norms <= 0.0)
        ):
            raise ValueError(
                "Newton articulated body state must be finite and normalized"
            )
        requested_orientations = requested_orientations / quaternion_norms[:, None]

        context = _newton_rigid_tensor_context(tensor_view)
        model = context.model
        state = context.state
        selection = _newton_selection_for_rigid_rows(
            raw_view=self._view,
            tensor_view=tensor_view,
            model=model,
            row_indices=indices,
        )
        _require_complete_newton_articulation_bodies(
            model=model,
            selection=selection,
        )
        original_body_q = wp.clone(state.body_q)
        original_body_qd = wp.clone(state.body_qd)
        updated_body_q = np.asarray(_to_numpy(original_body_q), dtype=np.float32).copy()
        updated_body_qd = np.asarray(
            _to_numpy(original_body_qd), dtype=np.float32
        ).copy()
        if updated_body_q.ndim != 2 or updated_body_q.shape[1] != 7:
            raise RuntimeError("Newton body_q tensor must have shape (N, 7)")
        if updated_body_qd.shape != (updated_body_q.shape[0], 6):
            raise RuntimeError("Newton body_qd tensor must have shape (N, 6)")
        updated_body_q[selection.body_ids, :3] = requested_positions
        updated_body_q[selection.body_ids[:, None], np.asarray([6, 3, 4, 5])] = (
            requested_orientations
        )
        updated_body_qd[selection.body_ids] = requested_velocities
        ik_indices = wp.from_numpy(
            np.ascontiguousarray(selection.articulation_ids, dtype=np.int32),
            dtype=wp.int32,
            device=str(model.device),
        )
        original_q = wp.clone(state.joint_q)
        original_qd = wp.clone(state.joint_qd)
        ik_q = wp.clone(original_q)
        ik_qd = wp.clone(original_qd)
        try:
            state.body_q.assign(updated_body_q)
            state.body_qd.assign(updated_body_qd)
            newton.eval_ik(
                model,
                state,
                ik_q,
                ik_qd,
                indices=ik_indices,
            )
            wp.copy(state.joint_q, ik_q)
            wp.copy(state.joint_qd, ik_qd)

            actual_transforms = np.asarray(
                _to_numpy(tensor_view.get_transforms()), dtype=np.float32
            )
            actual_velocities = np.asarray(
                _to_numpy(tensor_view.get_velocities()), dtype=np.float32
            )
            for label, expected, actual in (
                (
                    "positions",
                    requested_positions,
                    actual_transforms[indices, :3],
                ),
                (
                    "orientations",
                    requested_orientations,
                    actual_transforms[indices][:, [6, 3, 4, 5]],
                ),
                (
                    "velocities",
                    requested_velocities,
                    actual_velocities[indices],
                ),
            ):
                if not np.allclose(expected, actual, rtol=1.0e-6, atol=1.0e-6):
                    difference = np.abs(expected - actual)
                    flat_index = int(np.argmax(difference))
                    index = np.unravel_index(flat_index, difference.shape)
                    raise RuntimeError(
                        f"Newton articulated body {label} failed immediate readback: "
                        f"index={index}, expected={float(expected[index])}, "
                        f"actual={float(actual[index])}, "
                        f"max_abs_diff={float(difference[index])}"
                    )
        except BaseException:
            try:
                wp.copy(state.body_q, original_body_q)
                wp.copy(state.body_qd, original_body_qd)
            finally:
                wp.copy(state.joint_q, original_q)
                wp.copy(state.joint_qd, original_qd)
            raise


def create_articulation_core_view(
    *,
    paths: Sequence[str],
    name: str,
    world_scene: object | None = None,
    physics_backend: object | None = None,
    controllable_dof_names: Sequence[str] | None = None,
) -> object:
    """按 active backend 创建 legacy 或 Experimental articulation view。"""

    backend = (
        active_physics_backend()
        if physics_backend is None
        else normalize_physics_backend(physics_backend)
    )
    if backend == "newton":
        from linkerbot_sim.isaac.physics.manager import active_physics_manager
        from linkerbot_sim.isaac.physics.newton.views import (
            NewtonArticulationView,
        )

        manager = active_physics_manager()
        assert manager is not None
        return ArticulationCoreView(
            NewtonArticulationView(
                manager,
                paths=paths,
                name=name,
                controllable_dof_names=controllable_dof_names,
            ),
            physics_backend=backend,
        )
    if use_experimental_core(physics_backend=backend):
        from isaacsim.core.experimental.prims import Articulation

        return ArticulationCoreView(
            Articulation(
                paths=list(paths),
                reset_xform_op_properties=False,
            ),
            physics_backend=backend,
        )

    from isaacsim.core.prims import Articulation

    view = Articulation(
        prim_paths_expr=list(paths),
        name=name,
        reset_xform_properties=False,
    )
    add = getattr(world_scene, "add", None)
    return add(view) if callable(add) else view


def create_single_articulation_core_view(
    *,
    prim_path: str,
    name: str = "articulation",
    physics_backend: object | None = None,
) -> object:
    """按 active backend 创建 legacy 或 Experimental 单 articulation facade。"""

    backend = (
        active_physics_backend()
        if physics_backend is None
        else normalize_physics_backend(physics_backend)
    )
    if backend == "newton":
        from linkerbot_sim.isaac.physics.manager import active_physics_manager
        from linkerbot_sim.isaac.physics.newton.views import (
            NewtonArticulationView,
        )

        manager = active_physics_manager()
        assert manager is not None
        return SingleArticulationCoreView(
            NewtonArticulationView(
                manager,
                paths=(str(prim_path),),
                name=name,
            ),
            prim_path=prim_path,
            name=name,
            physics_backend=backend,
        )
    if use_experimental_core(physics_backend=backend):
        from isaacsim.core.experimental.prims import Articulation

        return SingleArticulationCoreView(
            Articulation(
                paths=str(prim_path),
                reset_xform_op_properties=False,
            ),
            prim_path=prim_path,
            name=name,
            physics_backend=backend,
        )

    from isaacsim.core.prims import SingleArticulation

    return SingleArticulation(prim_path=prim_path, name=name)


def create_rigid_prim_core_view(
    *,
    paths: Sequence[str],
    name: str,
    physics_backend: object | None = None,
) -> object:
    """按 active backend 创建 legacy 或 Experimental rigid-body view。"""

    backend = (
        active_physics_backend()
        if physics_backend is None
        else normalize_physics_backend(physics_backend)
    )
    if backend == "newton":
        from linkerbot_sim.isaac.physics.manager import active_physics_manager
        from linkerbot_sim.isaac.physics.newton.views import NewtonRigidBodyView

        manager = active_physics_manager()
        assert manager is not None
        return RigidPrimCoreView(
            NewtonRigidBodyView(manager, paths=paths, name=name),
            physics_backend=backend,
        )
    if use_experimental_core(physics_backend=backend):
        from isaacsim.core.experimental.prims import RigidPrim

        return RigidPrimCoreView(
            RigidPrim(
                paths=list(paths),
                reset_xform_op_properties=False,
            ),
            physics_backend=backend,
        )

    from isaacsim.core.prims import RigidPrim

    view = RigidPrim(
        prim_paths_expr=list(paths),
        name=name,
        reset_xform_properties=False,
    )
    initialize = getattr(view, "initialize", None)
    if callable(initialize):
        initialize()
    return view


def _indices(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.int32).reshape(-1)


def _is_newton_runtime_view(value: object) -> bool:
    """不在 PhysX 进程导入 Newton，仅凭 owner module 识别 Newton raw view。"""

    return value.__class__.__module__ == ("linkerbot_sim.isaac.physics.newton.views")


def _values(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _merge_selected_matrix(
    current: object,
    updates: object,
    *,
    indices: object,
    columns: object | None,
    label: str,
) -> np.ndarray:
    """把 selected rows/columns 广播合并进完整二维 tensor 的 numpy 副本。"""

    matrix = np.asarray(current, dtype=np.float32).copy()
    if matrix.ndim != 2:
        raise ValueError(f"{label} current value must be a 2D matrix")
    rows = _indices(indices)
    assert rows is not None
    cols = (
        np.arange(matrix.shape[1], dtype=np.int32)
        if columns is None
        else _indices(columns)
    )
    assert cols is not None
    if np.any(rows < 0) or np.any(rows >= matrix.shape[0]):
        raise IndexError(f"{label} row indices are out of range")
    if np.any(cols < 0) or np.any(cols >= matrix.shape[1]):
        raise IndexError(f"{label} column indices are out of range")
    try:
        values = np.broadcast_to(
            np.asarray(updates, dtype=np.float32),
            (rows.size, cols.size),
        )
    except ValueError as exc:
        raise ValueError(
            f"{label} cannot broadcast updates to {(rows.size, cols.size)}"
        ) from exc
    matrix[np.ix_(rows, cols)] = values
    return matrix


def _selected_matrix(
    current: object,
    *,
    indices: object | None,
    columns: object | None,
    label: str,
) -> np.ndarray:
    """在 CPU 侧选择 Newton full tensor 行列，避开 Experimental indexed gather。"""

    matrix = np.asarray(current).copy()
    if matrix.ndim != 2:
        raise ValueError(f"{label} current value must be a 2D matrix")
    rows = (
        np.arange(matrix.shape[0], dtype=np.int32)
        if indices is None
        else _indices(indices)
    )
    cols = (
        np.arange(matrix.shape[1], dtype=np.int32)
        if columns is None
        else _indices(columns)
    )
    assert rows is not None
    assert cols is not None
    if np.any(rows < 0) or np.any(rows >= matrix.shape[0]):
        raise IndexError(f"{label} row indices are out of range")
    if np.any(cols < 0) or np.any(cols >= matrix.shape[1]):
        raise IndexError(f"{label} column indices are out of range")
    return matrix[np.ix_(rows, cols)].copy()


def _to_numpy(value: object) -> np.ndarray:
    return tensor_like_to_numpy(value)


def _newton_warp_full_batch(
    values: np.ndarray,
    *,
    device: object,
) -> tuple[object, object]:
    """把 contiguous numpy matrix 转成 Newton tensor API 需要的 Warp full batch。"""

    import warp as wp

    matrix = np.ascontiguousarray(values, dtype=np.float32)
    target_device = None if device is None else str(device)
    tensor = wp.from_numpy(matrix, dtype=wp.float32, device=target_device)
    indices = wp.from_numpy(
        np.arange(matrix.shape[0], dtype=np.int32),
        dtype=wp.int32,
        device=target_device,
    )
    return tensor, indices


@dataclass(frozen=True)
class _NewtonRigidTensorContext:
    """Isaac tensor API wrapper 对应的 Newton stage/model/state。"""

    stage: object
    model: object
    state: object


@dataclass(frozen=True)
class _NewtonRigidSelection:
    """rigid view selected rows 对应的 Newton body/articulation IDs。"""

    body_ids: np.ndarray
    articulation_ids: np.ndarray


def _newton_rigid_tensor_context(tensor_view: object) -> _NewtonRigidTensorContext:
    """解析 Isaac 6.0.1 当前 Newton stage，并校验 tensor view 归属。"""

    api_backend = getattr(tensor_view, "_backend", None)
    providers = tuple(
        provider for provider in (tensor_view, api_backend) if provider is not None
    )
    private_stage = next(
        (
            value
            for provider in providers
            for name in ("_newton_stage", "newton_stage")
            if (value := getattr(provider, name, None)) is not None
        ),
        None,
    )
    acquired_stage = None
    if private_stage is None:
        try:
            from isaacsim.physics.newton import acquire_stage
        except ImportError:
            pass
        else:
            acquired_stage = acquire_stage()
    stage = acquired_stage if acquired_stage is not None else private_stage
    private_model = next(
        (
            value
            for provider in providers
            for name in ("_model", "model")
            if (value := getattr(provider, name, None)) is not None
        ),
        None,
    )
    model = getattr(stage, "model", None)
    if model is None:
        model = private_model
    state = getattr(stage, "state_0", None)
    if (
        stage is None
        or model is None
        or state is None
        or (private_stage is not None and private_stage is not stage)
        or (private_model is not None and private_model is not model)
    ):
        raise RuntimeError(
            "Newton rigid-body tensor internals are unavailable or inconsistent"
        )
    return _NewtonRigidTensorContext(
        stage=stage,
        model=model,
        state=state,
    )


def _newton_articulation_ids_for_rigid_rows(
    *,
    raw_view: object,
    tensor_view: object,
    model: object,
    row_indices: np.ndarray,
) -> np.ndarray:
    """把 rigid-view rows 严格映射到 Newton articulation ids。"""

    return _newton_selection_for_rigid_rows(
        raw_view=raw_view,
        tensor_view=tensor_view,
        model=model,
        row_indices=row_indices,
    ).articulation_ids


def _require_complete_newton_articulation_bodies(
    *,
    model: object,
    selection: _NewtonRigidSelection,
) -> None:
    """拒绝通过 maximal body state 部分重建 Newton articulation。"""

    joint_children = np.asarray(
        _to_numpy(getattr(model, "joint_child")), dtype=np.int64
    ).reshape(-1)
    joint_articulations = np.asarray(
        _to_numpy(getattr(model, "joint_articulation")), dtype=np.int64
    ).reshape(-1)
    expected_body_ids = np.unique(
        joint_children[
            np.isin(
                joint_articulations, selection.articulation_ids, assume_unique=False
            )
        ]
    )
    selected_body_ids = np.unique(selection.body_ids.astype(np.int64, copy=False))
    if np.array_equal(expected_body_ids, selected_body_ids):
        return
    missing = np.setdiff1d(expected_body_ids, selected_body_ids, assume_unique=True)
    extra = np.setdiff1d(selected_body_ids, expected_body_ids, assume_unique=True)
    raise RuntimeError(
        "Newton articulated body restore requires complete articulation body coverage: "
        f"articulation_ids={selection.articulation_ids.tolist()}, "
        f"missing_body_ids={missing.tolist()}, extra_body_ids={extra.tolist()}"
    )


def _newton_selection_for_rigid_rows(
    *,
    raw_view: object,
    tensor_view: object,
    model: object,
    row_indices: np.ndarray,
) -> _NewtonRigidSelection:
    """把 rigid-view rows 严格映射到 Newton body 与 articulation IDs。"""

    body_labels_value = getattr(model, "body_label", None)
    joint_child_value = getattr(model, "joint_child", None)
    joint_articulation_value = getattr(model, "joint_articulation", None)
    if (
        body_labels_value is None
        or joint_child_value is None
        or joint_articulation_value is None
    ):
        raise RuntimeError("Newton model does not expose rigid-to-articulation mapping")

    body_labels = tuple(str(path) for path in body_labels_value)
    joint_children = np.asarray(_to_numpy(joint_child_value), dtype=np.int64).reshape(
        -1
    )
    joint_articulations = np.asarray(
        _to_numpy(joint_articulation_value), dtype=np.int64
    ).reshape(-1)
    if joint_children.shape != joint_articulations.shape:
        raise RuntimeError(
            "Newton joint child/articulation arrays have different shapes"
        )

    authored_paths = tuple(str(path) for path in getattr(raw_view, "paths", ()))
    api_backend = getattr(tensor_view, "_backend", None)
    tensor_paths = next(
        (
            tuple(str(path) for path in value)
            for owner in (tensor_view, api_backend)
            if owner is not None
            for name in ("body_paths", "prim_paths")
            if (value := getattr(owner, name, None))
        ),
        (),
    )
    if not authored_paths or not tensor_paths:
        raise RuntimeError("Newton rigid-body tensor path order cannot be verified")
    if authored_paths != tensor_paths:
        raise RuntimeError(
            "Newton rigid-body tensor path order differs from the authored exact-path view"
        )

    if len(authored_paths) != int(getattr(tensor_view, "count")):
        raise RuntimeError(
            "Newton rigid body path count does not match its tensor view"
        )
    body_ids_by_path: dict[str, list[int]] = {}
    for body_id, path in enumerate(body_labels):
        body_ids_by_path.setdefault(path, []).append(body_id)

    body_ids: list[int] = []
    articulation_ids: list[int] = []
    for row in np.asarray(row_indices, dtype=np.int64).reshape(-1):
        if row < 0 or row >= len(authored_paths):
            raise IndexError("Newton articulated rigid-body row is out of range")
        path = authored_paths[int(row)]
        matched_body_ids = body_ids_by_path.get(path, ())
        if len(matched_body_ids) != 1:
            raise RuntimeError(
                "Newton rigid-body path must identify exactly one model body: "
                f"row={int(row)}, path={path!r}, "
                f"body_ids={list(matched_body_ids)}"
            )
        body_id = int(matched_body_ids[0])
        body_ids.append(body_id)
        incoming = np.flatnonzero(joint_children == body_id)
        if incoming.size != 1:
            raise RuntimeError(
                "Newton articulated rigid body must have exactly one incoming joint: "
                f"row={int(row)}, body_id={body_id}, incoming={incoming.tolist()}"
            )
        articulation_id = int(joint_articulations[int(incoming[0])])
        if articulation_id < 0:
            raise RuntimeError(
                "Newton articulated rigid body is not assigned to an articulation: "
                f"row={int(row)}, body_id={body_id}"
            )
        articulation_ids.append(articulation_id)
    return _NewtonRigidSelection(
        body_ids=np.asarray(body_ids, dtype=np.int32),
        articulation_ids=np.asarray(sorted(set(articulation_ids)), dtype=np.int32),
    )


def _single_values(value: object) -> np.ndarray:
    """把 Experimental 单 prim 的 ``(1, D)`` 返回值还原为 legacy ``(D,)``。"""

    array = _to_numpy(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        return array[0]
    return array


__all__ = [
    "ArticulationCoreView",
    "ExperimentalArticulationAction",
    "ExperimentalArticulationController",
    "RigidPrimCoreView",
    "SingleArticulationCoreView",
    "create_articulation_core_view",
    "create_rigid_prim_core_view",
    "create_single_articulation_core_view",
    "use_experimental_core",
]
