"""Isaac/PhysX raw view 的 DLPack/Torch CUDA port 实现。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from linkerbot_sim.controllers.runtime_projection import CommandRuntimeProjection
from linkerbot_sim.kaleidoscope.control_runtime import (
    PreparedKaleidoscopeControlRuntime,
    prepare_device_control_runtime,
)
from linkerbot_sim.kaleidoscope.geometry import (
    quaternion_multiply_wxyz,
    quaternion_rotate_wxyz,
)
from linkerbot_sim.kaleidoscope.tensors import assert_finite_async, require_cuda_tensor
from linkerbot_sim.robots.mimic.runtime import MimicFollowerControl
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


@dataclass(frozen=True, slots=True)
class PreparedPhysxControlRuntime:
    common: PreparedKaleidoscopeControlRuntime
    drive_stiffness: torch.Tensor
    drive_damping: torch.Tensor
    effort_limits: torch.Tensor
    physical_mode_indices: tuple[tuple[str, torch.Tensor, tuple[int, ...]], ...]


@dataclass(slots=True)
class IsaacArticulationTensorPort:
    """只暴露一个机器人的 command joints 与 TCP rigid-link tensor。"""

    label: str
    view: object
    tcp_view: object
    command_joint_indices: torch.Tensor
    device: torch.device
    command_joint_names: tuple[str, ...] | None = None
    command_joint_indices_host: tuple[int, ...] | None = None
    orientation_order: str = "wxyz"
    tcp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mimic_follower_controls: tuple[MimicFollowerControl, ...] = ()
    command_dim: int = field(init=False)
    state_dim: int = field(init=False)
    num_envs: int = field(init=False)
    command_state_indices: torch.Tensor = field(init=False, repr=False)
    _all_joint_indices: torch.Tensor = field(init=False, repr=False)
    _tcp_offset_position: torch.Tensor = field(init=False, repr=False)
    _tcp_offset_orientation: torch.Tensor = field(init=False, repr=False)
    _all_env_ids: torch.Tensor = field(init=False, repr=False)
    _position_feedforward: torch.Tensor = field(init=False, repr=False)
    _mimic_dependent_indices: torch.Tensor = field(init=False, repr=False)
    _mimic_master_indices: torch.Tensor = field(init=False, repr=False)
    _mimic_polycoef: torch.Tensor = field(init=False, repr=False)
    _reset_nominal_joint_positions: torch.Tensor | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _device_reset_mask: torch.Tensor | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _active_control_runtime: PreparedPhysxControlRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise ValueError("articulation tensor port requires CUDA")
        indices = require_cuda_tensor(
            self.command_joint_indices,
            name=f"{self.label} command joint indices",
            ndim=1,
            dtype=torch.int64,
        )
        if indices.device != self.device:
            raise ValueError("command joint indices must live on port.device")
        self.command_joint_indices = indices
        self.command_dim = indices.numel()
        dof_names = tuple(str(name) for name in getattr(self.view, "dof_names", ()))
        if not dof_names:
            raise ValueError("articulation tensor port requires stable DOF names")
        self.state_dim = len(dof_names)
        names = self.command_joint_names
        host_indices = self.command_joint_indices_host
        if names is not None:
            names = tuple(str(name) for name in names)
            if len(names) != self.command_dim or len(set(names)) != len(names):
                raise ValueError("command_joint_names must match the command width")
        if host_indices is not None:
            host_indices = tuple(int(index) for index in host_indices)
            if (
                len(host_indices) != self.command_dim
                or len(set(host_indices)) != len(host_indices)
                or any(index < 0 or index >= self.state_dim for index in host_indices)
            ):
                raise ValueError(
                    "command_joint_indices_host must identify command DOFs"
                )
        if names is None and host_indices is not None:
            names = tuple(dof_names[index] for index in host_indices)
        elif names is not None and host_indices is None:
            try:
                host_indices = tuple(dof_names.index(name) for name in names)
            except ValueError as exc:
                raise ValueError(
                    "command_joint_names must exist in articulation DOFs"
                ) from exc
        elif names is not None and host_indices is not None:
            if tuple(dof_names[index] for index in host_indices) != names:
                raise ValueError("command joint names and host indices disagree")
        self.command_joint_names = names
        self.command_joint_indices_host = host_indices
        self._prepare_mimic_metadata(dof_names, host_indices)
        count = getattr(self.view, "count", None)
        if count is None:
            probe = getattr(self.view, "q", None)
            count = 0 if probe is None else int(probe.shape[0])
        self.num_envs = int(count)
        if self.num_envs < 1:
            raise ValueError("articulation tensor port requires a non-empty view")
        self.command_state_indices = indices.clone()
        self._all_joint_indices = torch.arange(
            self.state_dim,
            device=self.device,
            dtype=torch.int64,
        )
        self._all_env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.int64,
        )
        self._position_feedforward = torch.zeros(
            (self.num_envs, self.command_dim),
            device=self.device,
            dtype=torch.float32,
        )
        if self.orientation_order not in {"wxyz", "xyzw"}:
            raise ValueError("orientation_order must be wxyz or xyzw")
        if len(self.tcp_offset_xyz) != 3 or len(self.tcp_offset_rpy) != 3:
            raise ValueError("TCP fixed offsets must be length-3 tuples")
        self._tcp_offset_position = torch.tensor(
            self.tcp_offset_xyz,
            device=self.device,
            dtype=torch.float32,
        )
        self._tcp_offset_orientation = torch.tensor(
            rpy_xyz_to_quat_wxyz(self.tcp_offset_rpy),
            device=self.device,
            dtype=torch.float32,
        )

    def _prepare_mimic_metadata(
        self,
        dof_names: tuple[str, ...],
        command_indices_host: tuple[int, ...] | None,
    ) -> None:
        controls = tuple(self.mimic_follower_controls)
        self.mimic_follower_controls = controls
        command_indices = set(command_indices_host or ())
        dependent_indices: set[int] = set()
        for control in controls:
            if (
                control.dependent_index < 0
                or control.dependent_index >= self.state_dim
                or dof_names[control.dependent_index] != control.dependent_joint
            ):
                raise ValueError(
                    "mimic follower metadata does not match articulation DOFs"
                )
            if (
                control.master_index < 0
                or control.master_index >= self.state_dim
                or dof_names[control.master_index] != control.master_joint
            ):
                raise ValueError(
                    "mimic master metadata does not match articulation DOFs"
                )
            if control.dependent_index in dependent_indices:
                raise ValueError("mimic follower indices must be unique")
            if control.dependent_index in command_indices:
                raise ValueError("mimic followers cannot remain in command joints")
            if not control.polycoef:
                raise ValueError("mimic follower polycoef cannot be empty")
            dependent_indices.add(control.dependent_index)

        self._mimic_dependent_indices = torch.tensor(
            [control.dependent_index for control in controls],
            device=self.device,
            dtype=torch.int64,
        )
        self._mimic_master_indices = torch.tensor(
            [control.master_index for control in controls],
            device=self.device,
            dtype=torch.int64,
        )
        coefficient_width = max(
            (len(control.polycoef) for control in controls), default=0
        )
        coefficients = [
            (
                *control.polycoef,
                *(0.0 for _ in range(coefficient_width - len(control.polycoef))),
            )
            for control in controls
        ]
        self._mimic_polycoef = torch.tensor(
            coefficients,
            device=self.device,
            dtype=torch.float32,
        ).reshape(len(controls), coefficient_width)

    def read_joint_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._read_joint("get_joint_positions", env_ids)

    def read_joint_velocities(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._read_joint("get_joint_velocities", env_ids)

    def read_all_joint_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._read_joint(
            "get_joint_positions",
            env_ids,
            joint_indices=self._all_joint_indices,
        )

    def read_all_joint_velocities(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._read_joint(
            "get_joint_velocities",
            env_ids,
            joint_indices=self._all_joint_indices,
        )

    def read_tcp_pose_wxyz(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, orientations = _read_pose(self.tcp_view, env_ids)
        positions = as_torch_cuda(positions, device=self.device)
        orientations = as_torch_cuda(orientations, device=self.device)
        body_orientation = _to_wxyz(orientations, order=self.orientation_order)
        # raw view 读取的是承载 TCP 的刚体。custom TCP 是该刚体下的固定 frame；
        # 位姿复合留在同一 CUDA device，不为 joint-only task 创建 cuRobo FK context。
        offset = self._tcp_offset_position[None, :].expand(positions.shape[0], -1)
        offset_orientation = self._tcp_offset_orientation[None, :].expand(
            positions.shape[0], -1
        )
        return (
            positions + quaternion_rotate_wxyz(body_orientation, offset),
            quaternion_multiply_wxyz(
                body_orientation,
                offset_orientation,
                normalize_result=True,
            ),
        )

    def write_joint_positions(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        nominal = self._reset_nominal_joint_positions
        if nominal is None:
            self._write_joint_state("set_joint_positions", env_ids, values)
            return
        command = self._validated_command_values(
            env_ids,
            values,
            name="reset joint positions",
        )
        full = nominal[None, :].expand(env_ids.numel(), -1).clone()
        full.index_copy_(1, self.command_joint_indices, command)
        self._project_mimic_positions(full)
        self.write_all_joint_positions(
            env_ids,
            self._preserve_unmasked_joint_state(
                env_ids,
                fresh=full,
                velocity=False,
            ),
        )

    def write_joint_velocities(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        if self._reset_nominal_joint_positions is None:
            self._write_joint_state("set_joint_velocities", env_ids, values)
            return
        command = self._validated_command_values(
            env_ids,
            values,
            name="reset joint velocities",
        )
        full = torch.zeros(
            (env_ids.numel(), self.state_dim),
            device=self.device,
            dtype=torch.float32,
        )
        full.index_copy_(1, self.command_joint_indices, command)
        self._project_mimic_velocities(
            self.read_all_joint_positions(env_ids),
            full,
        )
        self.write_all_joint_velocities(
            env_ids,
            self._preserve_unmasked_joint_state(
                env_ids,
                fresh=full,
                velocity=True,
            ),
        )

    def prepare_full_dof_reset(self, nominal_joint_positions: torch.Tensor) -> None:
        """Freeze the complete articulation reset pose after backend startup."""

        if self._reset_nominal_joint_positions is not None:
            raise RuntimeError("full-DOF reset state is already prepared")
        nominal = require_cuda_tensor(
            nominal_joint_positions,
            name=f"{self.label} nominal full joint positions",
            ndim=1,
            dtype=torch.float32,
        )
        if nominal.device != self.device or nominal.shape != (self.state_dim,):
            raise ValueError(
                f"{self.label} nominal full joint positions must have shape "
                f"({self.state_dim},) on {self.device}"
            )
        assert_finite_async(
            nominal,
            name=f"{self.label} nominal full joint positions",
        )
        self._reset_nominal_joint_positions = nominal.clone()

    def set_device_reset_mask(self, reset_mask: torch.Tensor | None) -> None:
        """Select SAME_STEP rows while keeping reset projection CUDA-resident."""

        if reset_mask is None:
            self._device_reset_mask = None
            return
        mask = require_cuda_tensor(
            reset_mask,
            name=f"{self.label} device reset mask",
            ndim=1,
            leading_dim=self.num_envs,
            dtype=torch.bool,
        )
        if mask.device != self.device:
            raise ValueError(
                f"{self.label} device reset mask must live on {self.device}"
            )
        self._device_reset_mask = mask

    def _project_mimic_positions(self, full_positions: torch.Tensor) -> None:
        if not self.mimic_follower_controls:
            return
        master = full_positions.index_select(1, self._mimic_master_indices)
        projected = torch.zeros_like(master)
        for coefficient in reversed(self._mimic_polycoef.unbind(dim=1)):
            projected.mul_(master).add_(coefficient)
        full_positions.index_copy_(1, self._mimic_dependent_indices, projected)

    def _project_mimic_velocities(
        self,
        full_positions: torch.Tensor,
        full_velocities: torch.Tensor,
    ) -> None:
        if not self.mimic_follower_controls:
            return
        master_position = full_positions.index_select(1, self._mimic_master_indices)
        derivative = torch.zeros_like(master_position)
        for degree in range(self._mimic_polycoef.shape[1] - 1, 0, -1):
            derivative.mul_(master_position).add_(
                self._mimic_polycoef[:, degree],
                alpha=degree,
            )
        master_velocity = full_velocities.index_select(1, self._mimic_master_indices)
        full_velocities.index_copy_(
            1,
            self._mimic_dependent_indices,
            derivative * master_velocity,
        )

    def _preserve_unmasked_joint_state(
        self,
        env_ids: torch.Tensor,
        *,
        fresh: torch.Tensor,
        velocity: bool,
    ) -> torch.Tensor:
        mask = self._device_reset_mask
        if mask is None:
            return fresh
        selected_mask = mask.index_select(0, env_ids)[:, None]
        current = (
            self.read_all_joint_velocities(env_ids)
            if velocity
            else self.read_all_joint_positions(env_ids)
        )
        return torch.where(selected_mask, fresh, current)

    def write_joint_position_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        prepared = self._require_active_control_runtime("position")
        data = self._validated_command_values(env_ids, values, name="position targets")
        implicit = prepared.common.implicit_indices
        if implicit.numel():
            self._write_joint(
                "set_joint_position_targets",
                env_ids,
                data.index_select(1, implicit),
                joint_indices=self.command_joint_indices.index_select(0, implicit),
                width=implicit.numel(),
            )
        explicit = prepared.common.explicit_indices
        if explicit.numel():
            q = self.read_joint_positions(env_ids).index_select(1, explicit)
            qd = self.read_joint_velocities(env_ids).index_select(1, explicit)
            desired_qd = self._position_feedforward.index_select(
                0, env_ids
            ).index_select(1, explicit)
            effort = prepared.common.stiffness.index_select(0, explicit) * (
                data.index_select(1, explicit) - q
            ) + prepared.common.damping.index_select(0, explicit) * (desired_qd - qd)
            limit = prepared.common.effort_limits.index_select(0, explicit)
            self._write_joint(
                "set_joint_efforts",
                env_ids,
                torch.clamp(effort, min=-limit, max=limit),
                joint_indices=self.command_joint_indices.index_select(0, explicit),
                width=explicit.numel(),
            )

    def write_joint_velocity_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        prepared = self._require_active_control_runtime(("position", "velocity"))
        data = self._validated_command_values(env_ids, values, name="velocity targets")
        if prepared.common.mode == "position":
            self._position_feedforward.index_copy_(0, env_ids, data)
        implicit = prepared.common.implicit_indices
        if implicit.numel():
            self._write_joint(
                "set_joint_velocity_targets",
                env_ids,
                data.index_select(1, implicit),
                joint_indices=self.command_joint_indices.index_select(0, implicit),
                width=implicit.numel(),
            )
        explicit = prepared.common.explicit_indices
        if explicit.numel() and prepared.common.mode == "velocity":
            qd = self.read_joint_velocities(env_ids).index_select(1, explicit)
            effort = prepared.common.damping.index_select(0, explicit) * (
                data.index_select(1, explicit) - qd
            )
            limit = prepared.common.effort_limits.index_select(0, explicit)
            self._write_joint(
                "set_joint_efforts",
                env_ids,
                torch.clamp(effort, min=-limit, max=limit),
                joint_indices=self.command_joint_indices.index_select(0, explicit),
                width=explicit.numel(),
            )

    def write_joint_effort_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        prepared = self._require_active_control_runtime("effort")
        data = self._validated_command_values(env_ids, values, name="effort targets")
        limit = prepared.common.effort_limits
        self._write_joint(
            "set_joint_efforts",
            env_ids,
            torch.clamp(data, min=-limit, max=limit),
        )

    def write_joint_targets(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Compatibility wrapper for position-only callers."""

        if self._active_control_runtime is None:
            self._write_joint("set_joint_position_targets", env_ids, values)
            return
        self.write_joint_position_targets(env_ids, values)

    def prepare_control_runtime(
        self,
        projection: CommandRuntimeProjection,
    ) -> PreparedPhysxControlRuntime:
        names = self.command_joint_names
        host_indices = self.command_joint_indices_host
        if names is None or host_indices is None:
            raise RuntimeError(
                "PhysX control runtime requires construction-time command joint metadata"
            )
        if projection.joint_names != names:
            raise ValueError("PhysX control projection joint order does not match port")
        common = prepare_device_control_runtime(
            projection,
            mode=projection.modes[0],
            device=self.device,
        )

        def rows(values: torch.Tensor) -> torch.Tensor:
            return values[None, :].expand(self.num_envs, -1).contiguous()

        physical: list[tuple[str, torch.Tensor, tuple[int, ...]]] = []
        for mode in dict.fromkeys(projection.physical_modes):
            relative = torch.tensor(
                [
                    index
                    for index, value in enumerate(projection.physical_modes)
                    if value == mode
                ],
                device=self.device,
                dtype=torch.int64,
            )
            global_indices = self.command_joint_indices.index_select(
                0, relative
            ).contiguous()
            physical.append(
                (
                    mode,
                    global_indices,
                    tuple(
                        host_indices[index]
                        for index in range(len(projection.physical_modes))
                        if projection.physical_modes[index] == mode
                    ),
                )
            )
        return PreparedPhysxControlRuntime(
            common=common,
            drive_stiffness=rows(common.drive_stiffness),
            drive_damping=rows(common.drive_damping),
            effort_limits=rows(common.effort_limits),
            physical_mode_indices=tuple(physical),
        )

    def apply_prepared_control_runtime(
        self,
        prepared: PreparedPhysxControlRuntime,
    ) -> None:
        self.validate_prepared_control_runtime(prepared)
        switch = getattr(self.view, "switch_dof_control_mode")
        grouped_api = hasattr(self.view, "raw_view")
        for mode, indices, indices_host in prepared.physical_mode_indices:
            if grouped_api:
                switch(mode, indices=self._all_env_ids, dof_indices=indices)
            else:
                for index in indices_host:
                    switch(mode, dof_index=index, indices=self._all_env_ids)
        set_gains = getattr(self.view, "set_gains", None)
        if callable(set_gains):
            set_gains(
                kps=prepared.drive_stiffness,
                kds=prepared.drive_damping,
                indices=self._all_env_ids,
                joint_indices=self.command_joint_indices,
            )
        else:
            set_gains = getattr(self.view, "set_dof_gains", None)
            if not callable(set_gains):
                raise RuntimeError("PhysX articulation does not expose gain mutation")
            set_gains(
                stiffnesses=prepared.drive_stiffness,
                dampings=prepared.drive_damping,
                indices=self._all_env_ids,
                dof_indices=self.command_joint_indices,
            )
        set_limits = getattr(self.view, "set_max_efforts", None)
        if callable(set_limits):
            set_limits(
                prepared.effort_limits,
                indices=self._all_env_ids,
                joint_indices=self.command_joint_indices,
            )
        else:
            set_limits = getattr(self.view, "set_dof_max_efforts", None)
            if not callable(set_limits):
                raise RuntimeError("PhysX articulation does not expose effort limits")
            set_limits(
                prepared.effort_limits,
                indices=self._all_env_ids,
                dof_indices=self.command_joint_indices,
            )
        set_effort_modes = getattr(self.view, "set_effort_modes", None)
        if callable(set_effort_modes):
            set_effort_modes(
                "force",
                indices=self._all_env_ids,
                joint_indices=self.command_joint_indices,
            )
        else:
            set_drive_types = getattr(self.view, "set_dof_drive_types", None)
            if not callable(set_drive_types):
                raise RuntimeError("PhysX articulation does not expose effort modes")
            set_drive_types("force", dof_indices=self.command_joint_indices)
        self._active_control_runtime = prepared

    def validate_prepared_control_runtime(
        self,
        prepared: PreparedPhysxControlRuntime,
    ) -> None:
        if not isinstance(prepared, PreparedPhysxControlRuntime):
            raise TypeError("prepared must be PreparedPhysxControlRuntime")
        if prepared.common.device != self.device:
            raise ValueError("prepared PhysX control runtime has the wrong device")
        alternatives = (
            ("switch_dof_control_mode",),
            ("set_gains", "set_dof_gains"),
            ("set_max_efforts", "set_dof_max_efforts"),
            ("set_effort_modes", "set_dof_drive_types"),
        )
        for names in alternatives:
            if not any(callable(getattr(self.view, name, None)) for name in names):
                raise RuntimeError(
                    "PhysX articulation does not expose required control mutation: "
                    + " or ".join(f"{name}()" for name in names)
                )

    def synchronize_control_writes(self) -> None:
        torch.cuda.synchronize(self.device)

    def _require_active_control_runtime(
        self,
        expected: str | tuple[str, ...],
    ) -> PreparedPhysxControlRuntime:
        prepared = self._active_control_runtime
        if prepared is None:
            raise RuntimeError("PhysX control runtime has not been configured")
        modes = (expected,) if isinstance(expected, str) else expected
        if prepared.common.mode not in modes:
            raise RuntimeError(
                f"PhysX {self.label} is in {prepared.common.mode!r} control mode; "
                f"expected one of {modes}"
            )
        return prepared

    def _validated_command_values(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        result = require_cuda_tensor(
            values,
            name=f"{self.label} {name}",
            ndim=2,
            leading_dim=env_ids.numel(),
            dtype=torch.float32,
        )
        if result.device != self.device or result.shape[1:] != (self.command_dim,):
            raise ValueError(
                f"{self.label} {name} must have shape "
                f"({env_ids.numel()},{self.command_dim}) on {self.device}"
            )
        assert_finite_async(result, name=f"{self.label} {name}")
        return result.contiguous()

    def write_all_joint_positions(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self._write_joint_state(
            "set_joint_positions",
            env_ids,
            values,
            joint_indices=self._all_joint_indices,
            width=self.state_dim,
        )

    def write_all_joint_velocities(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self._write_joint_state(
            "set_joint_velocities",
            env_ids,
            values,
            joint_indices=self._all_joint_indices,
            width=self.state_dim,
        )

    def _write_joint_state(
        self,
        method_name: str,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        joint_indices: torch.Tensor | None = None,
        width: int | None = None,
    ) -> None:
        # Isaac 的高层 state setter 会把整行 q/qd 同步写进对应 drive target。
        # state/snapshot/clone 的合同要求 teleport 不改变后继控制输入，因此先借用
        # 公开 action buffer，复制所选 CUDA 行，并在 state setter 后恢复两类 target。
        position_targets, velocity_targets, effort_targets = (
            self._selected_drive_targets(env_ids)
        )
        self._write_joint(
            method_name,
            env_ids,
            values,
            joint_indices=joint_indices,
            width=width,
        )
        self._write_joint(
            "set_joint_position_targets",
            env_ids,
            position_targets,
            joint_indices=self._all_joint_indices,
            width=self.state_dim,
        )
        self._write_joint(
            "set_joint_velocity_targets",
            env_ids,
            velocity_targets,
            joint_indices=self._all_joint_indices,
            width=self.state_dim,
        )
        if effort_targets is not None:
            self._write_joint(
                "set_joint_efforts",
                env_ids,
                effort_targets,
                joint_indices=self._all_joint_indices,
                width=self.state_dim,
            )

    def _selected_drive_targets(
        self,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        getter = getattr(self.view, "get_applied_actions", None)
        if not callable(getter):
            raise RuntimeError(
                "articulation view must expose public get_applied_actions() so state "
                "writes can preserve PhysX drive targets"
            )
        actions = _borrow_view(getter)

        def selected(field_name: str, *, optional: bool = False) -> torch.Tensor | None:
            raw = getattr(actions, field_name, None)
            if raw is None:
                if optional:
                    return None
                raise RuntimeError(
                    f"articulation applied actions do not expose {field_name}"
                )
            tensor = as_torch_cuda(raw, device=self.device)
            if (
                tensor.dtype != torch.float32
                or tensor.ndim != 2
                or tensor.shape[1] != self.state_dim
            ):
                raise ValueError(
                    f"articulation {field_name} must be float32 with shape "
                    f"(N,{self.state_dim})"
                )
            # index_select 生成 owned CUDA tensor；后续 Isaac setter 可安全修改 raw buffer。
            return tensor.index_select(0, env_ids)

        return (
            selected("joint_positions"),
            selected("joint_velocities"),
            selected("joint_efforts", optional=True),
        )

    def close(self) -> None:
        # Raw views 的 native owner 是 PhysxRuntime/World；这里只使 mode adapter 失效。
        for value in (self.tcp_view, self.view):
            invalidate = getattr(value, "invalidate", None)
            if callable(invalidate):
                invalidate()

    def _read_joint(
        self,
        method_name: str,
        env_ids: torch.Tensor,
        *,
        joint_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        method = getattr(self.view, method_name)
        value = _borrow_view(
            method,
            indices=env_ids,
            joint_indices=(
                self.command_joint_indices if joint_indices is None else joint_indices
            ),
        )
        return as_torch_cuda(value, device=self.device)

    def _write_joint(
        self,
        method_name: str,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        joint_indices: torch.Tensor | None = None,
        width: int | None = None,
    ) -> None:
        expected_width = self.command_dim if width is None else int(width)
        value = require_cuda_tensor(
            values,
            name=f"{self.label} {method_name} values",
            ndim=2,
            leading_dim=env_ids.numel(),
            dtype=torch.float32,
        )
        if value.device != self.device or value.shape[1:] != (expected_width,):
            raise ValueError(
                f"{self.label} {method_name} values must have shape "
                f"({env_ids.numel()},{expected_width}) on {self.device}"
            )
        method = getattr(self.view, method_name)
        method(
            value,
            indices=env_ids,
            joint_indices=(
                self.command_joint_indices if joint_indices is None else joint_indices
            ),
        )


@dataclass(slots=True)
class IsaacRigidObjectTensorPort:
    """一个动态刚体 root view 的 world-pose/COM-velocity port。"""

    label: str
    view: object
    device: torch.device
    orientation_order: str = "wxyz"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise ValueError("rigid object tensor port requires CUDA")
        if self.orientation_order not in {"wxyz", "xyzw"}:
            raise ValueError("orientation_order must be wxyz or xyzw")

    def read_pose_wxyz(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, orientations = _read_pose(self.view, env_ids)
        return (
            as_torch_cuda(positions, device=self.device),
            _to_wxyz(
                as_torch_cuda(orientations, device=self.device),
                order=self.orientation_order,
            ),
        )

    def read_com_velocity(self, env_ids: torch.Tensor) -> torch.Tensor:
        value = _borrow_view(getattr(self.view, "get_velocities"), indices=env_ids)
        return as_torch_cuda(value, device=self.device)

    def write_pose_wxyz(
        self,
        env_ids: torch.Tensor,
        positions_world: torch.Tensor,
        orientations_wxyz: torch.Tensor,
    ) -> None:
        positions = require_cuda_tensor(
            positions_world,
            name=f"{self.label} positions",
            ndim=2,
            leading_dim=env_ids.numel(),
            dtype=torch.float32,
        )
        orientations = require_cuda_tensor(
            orientations_wxyz,
            name=f"{self.label} orientations",
            ndim=2,
            leading_dim=env_ids.numel(),
            dtype=torch.float32,
        )
        raw_orientation = _from_wxyz(orientations, order=self.orientation_order)
        setter = getattr(self.view, "set_world_poses", None)
        if callable(setter):
            setter(
                positions=positions,
                orientations=raw_orientation,
                indices=env_ids,
            )
            return
        setter = getattr(self.view, "set_transforms", None)
        if not callable(setter):
            raise RuntimeError("rigid view does not expose a CUDA pose setter")
        setter(
            torch.cat((positions, raw_orientation), dim=1),
            indices=env_ids,
        )

    def write_velocity(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        velocity = require_cuda_tensor(
            values,
            name=f"{self.label} COM velocity",
            ndim=2,
            leading_dim=env_ids.numel(),
            dtype=torch.float32,
        )
        getattr(self.view, "set_velocities")(velocity, indices=env_ids)

    def close(self) -> None:
        invalidate = getattr(self.view, "invalidate", None)
        if callable(invalidate):
            invalidate()


def as_torch_cuda(value: object, *, device: torch.device) -> torch.Tensor:
    """通过原生 Torch 或 DLPack 取得零拷贝 CUDA view；绝不回退 NumPy。"""

    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "__dlpack__"):
        tensor = torch.from_dlpack(value)
    else:
        to_dlpack = getattr(value, "to_dlpack", None)
        if not callable(to_dlpack):
            raise TypeError(
                f"Isaac view returned non-Torch/non-DLPack value {type(value).__name__}"
            )
        tensor = torch.from_dlpack(to_dlpack())
    if tensor.device != device:
        raise ValueError(
            f"Isaac view tensor must live on {device}, got {tensor.device}"
        )
    return tensor


def _read_pose(view: object, env_ids: torch.Tensor) -> tuple[object, object]:
    getter = getattr(view, "get_world_poses", None)
    if callable(getter):
        return _borrow_view(getter, indices=env_ids)
    getter = getattr(view, "get_transforms", None)
    if not callable(getter):
        raise RuntimeError("rigid view does not expose a CUDA pose getter")
    transforms = as_torch_cuda(
        _borrow_view(getter, indices=env_ids), device=env_ids.device
    )
    if transforms.ndim != 2 or transforms.shape[1] != 7:
        raise ValueError("rigid transforms must have shape (N,7)")
    return transforms[:, :3], transforms[:, 3:7]


def _borrow_view(method, *args: object, **kwargs: object):
    """优先请求 borrowed tensor；旧签名不接受 clone 时仅去掉该关键字重试。"""

    try:
        return method(*args, clone=False, **kwargs)
    except TypeError as exc:
        if "clone" not in str(exc):
            raise
        return method(*args, **kwargs)


def _to_wxyz(value: torch.Tensor, *, order: str) -> torch.Tensor:
    if order == "wxyz":
        return value
    return value[:, (3, 0, 1, 2)]


def _from_wxyz(value: torch.Tensor, *, order: str) -> torch.Tensor:
    if order == "wxyz":
        return value
    return value[:, (1, 2, 3, 0)]


__all__ = [
    "IsaacArticulationTensorPort",
    "IsaacRigidObjectTensorPort",
    "as_torch_cuda",
]
