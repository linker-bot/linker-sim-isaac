"""物理后端 tensor port 到 Kaleidoscope canonical GPU buffer 的窄适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch

from linkerbot_sim.controllers.control_mode import require_control_mode
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.kaleidoscope.observations import TBlockState
from linkerbot_sim.kaleidoscope.resets import TBlockResetCommand
from linkerbot_sim.kaleidoscope.state_api import StateBinding
from linkerbot_sim.kaleidoscope.tensors import (
    normalize_env_ids,
    require_cuda_tensor,
)


@runtime_checkable
class RobotTensorPort(Protocol):
    """一个机器人 raw backend view 的 command/full-state CUDA 操作。"""

    label: str
    command_dim: int
    state_dim: int
    command_state_indices: torch.Tensor
    device: torch.device

    def read_joint_positions(self, env_ids: torch.Tensor) -> torch.Tensor: ...

    def read_joint_velocities(self, env_ids: torch.Tensor) -> torch.Tensor: ...

    def read_all_joint_positions(self, env_ids: torch.Tensor) -> torch.Tensor: ...

    def read_all_joint_velocities(self, env_ids: torch.Tensor) -> torch.Tensor: ...

    def read_tcp_pose_wxyz(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def write_joint_positions(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None: ...

    def write_joint_velocities(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None: ...

    def write_joint_position_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None: ...

    def write_joint_velocity_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None: ...

    def write_joint_effort_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None: ...

    def write_all_joint_positions(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None: ...

    def write_all_joint_velocities(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class RigidObjectTensorPort(Protocol):
    """T-block root rigid view 的 CUDA-only 操作；pose 使用 world frame/wxyz。"""

    label: str
    device: torch.device

    def read_pose_wxyz(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def read_com_velocity(self, env_ids: torch.Tensor) -> torch.Tensor: ...

    def write_pose_wxyz(
        self,
        env_ids: torch.Tensor,
        positions_world: torch.Tensor,
        orientations_wxyz: torch.Tensor,
    ) -> None: ...

    def write_velocity(self, env_ids: torch.Tensor, values: torch.Tensor) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class PhysicsStateTensorPort(Protocol):
    """后端私有、但决定后继物理状态的 per-env CUDA buffer。"""

    field_name: str
    num_envs: int
    device: torch.device
    tensor: torch.Tensor

    def write(self, env_ids: torch.Tensor, values: torch.Tensor) -> None: ...

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RobotColumns:
    label: str
    columns: slice


class KaleidoscopeTensorViews:
    """立即复制 backend borrowed tensor，并维护 snapshot/clone 的 canonical state。"""

    def __init__(
        self,
        *,
        robot_ports: tuple[RobotTensorPort, ...],
        object_port: RigidObjectTensorPort,
        env_origins: torch.Tensor,
        physics_state_port: PhysicsStateTensorPort | None = None,
    ) -> None:
        if not robot_ports:
            raise ValueError("Kaleidoscope requires at least one controlled robot")
        origins = require_cuda_tensor(
            env_origins, name="environment origins", ndim=2, dtype=torch.float32
        )
        if origins.shape[1:] != (3,):
            raise ValueError("env_origins must have shape (N,3)")
        self.num_envs = origins.shape[0]
        self.device = origins.device
        ports: tuple[object, ...] = (
            *robot_ports,
            object_port,
            *(() if physics_state_port is None else (physics_state_port,)),
        )
        port_devices = tuple(torch.device(getattr(port, "device")) for port in ports)
        if any(device != self.device for device in port_devices):
            raise ValueError(
                "Kaleidoscope view ports must share environment origins device "
                f"{self.device}; got {port_devices}"
            )
        if any(port.label == object_port.label for port in robot_ports):
            raise ValueError("robot and object labels must be distinct")
        labels = tuple(port.label for port in robot_ports)
        if len(set(labels)) != len(labels):
            raise ValueError("robot port labels cannot contain duplicates")
        self.robot_ports = robot_ports
        self.object_port = object_port
        self.physics_state_port = physics_state_port
        if physics_state_port is not None:
            state_tensor = require_cuda_tensor(
                physics_state_port.tensor,
                name=f"physics state {physics_state_port.field_name!r}",
                ndim=2,
                leading_dim=self.num_envs,
                dtype=torch.float32,
            )
            if (
                physics_state_port.num_envs != self.num_envs
                or physics_state_port.device != self.device
                or state_tensor.device != self.device
            ):
                raise ValueError(
                    "physics state port must match view num_envs and CUDA device"
                )
        self.env_origins = origins.clone()
        command_offset = 0
        state_offset = 0
        columns: list[RobotColumns] = []
        state_columns: list[RobotColumns] = []
        for port in robot_ports:
            if port.command_dim < 1 or port.state_dim < port.command_dim:
                raise ValueError(
                    "robot command_dim/state_dim must be positive and ordered"
                )
            command_indices = require_cuda_tensor(
                port.command_state_indices,
                name=f"{port.label} command-state indices",
                ndim=1,
                dtype=torch.int64,
            )
            if command_indices.device != self.device or command_indices.shape != (
                port.command_dim,
            ):
                raise ValueError(
                    f"{port.label} command-state indices must have shape "
                    f"({port.command_dim},) on {self.device}"
                )
            torch._assert_async(
                torch.all((command_indices >= 0) & (command_indices < port.state_dim)),
                f"{port.label} command-state indices are out of range",
            )
            columns.append(
                RobotColumns(
                    port.label,
                    slice(command_offset, command_offset + port.command_dim),
                )
            )
            state_columns.append(
                RobotColumns(
                    port.label,
                    slice(state_offset, state_offset + port.state_dim),
                )
            )
            command_offset += port.command_dim
            state_offset += port.state_dim
        self.robot_columns = tuple(columns)
        self.robot_state_columns = tuple(state_columns)
        self.command_dim = command_offset
        self.state_dim = state_offset
        count = self.num_envs
        self.joint_positions = torch.zeros(
            (count, command_offset), device=self.device, dtype=torch.float32
        )
        self.joint_velocities = torch.zeros_like(self.joint_positions)
        self.position_references = torch.zeros_like(self.joint_positions)
        self.control_targets = torch.zeros_like(self.joint_positions)
        self.all_joint_positions = torch.zeros(
            (count, state_offset), device=self.device, dtype=torch.float32
        )
        self.all_joint_velocities = torch.zeros_like(self.all_joint_positions)
        self.tcp_positions_local = torch.zeros(
            (count, len(robot_ports), 3), device=self.device, dtype=torch.float32
        )
        self.tcp_orientations_wxyz = torch.zeros(
            (count, len(robot_ports), 4), device=self.device, dtype=torch.float32
        )
        # Identity wxyz: a valid quaternion fallback for the hold-last-finite guard below,
        # in case an env reads non-finite before any finite TCP has been recorded (Gitea #67).
        self.tcp_orientations_wxyz[..., 0] = 1.0
        # Device-resident count of non-finite TCP link-pose rows held to their last-finite
        # value; incremented sync-free on the hot path, inspectable on a cold boundary.
        self._nonfinite_tcp_holds = torch.zeros((), device=self.device, dtype=torch.int64)
        self.block_pose_local_wxyz = torch.zeros(
            (count, 7), device=self.device, dtype=torch.float32
        )
        self.block_position_local = self.block_pose_local_wxyz[:, :3]
        self.block_orientation_wxyz = self.block_pose_local_wxyz[:, 3:7]
        self.block_com_velocity = torch.zeros(
            (count, 6), device=self.device, dtype=torch.float32
        )
        self.external_safety_stop = torch.zeros(
            count, device=self.device, dtype=torch.bool
        )
        # Views 的环境拓扑与 device 不会热变更；refresh/write target 是每个 physics
        # decision 的热路径，必须复用固定 selector，不能逐拍 torch.arange。
        self._all_env_ids = torch.arange(count, device=self.device, dtype=torch.int64)
        self._closed = False
        self._closing_started = False
        self._close_completed: set[str] = set()
        self._control_mode_provider: Callable[[], ControlMode] = lambda: "position"
        self._control_mode_provider_bound = False

    @property
    def command_targets(self) -> torch.Tensor:
        """Deprecated storage alias for the position-reference accumulator."""

        return self.position_references

    def bind_control_mode_provider(self, provider: Callable[[], ControlMode]) -> None:
        self._require_open()
        if not callable(provider):
            raise TypeError("control mode provider must be callable")
        if self._control_mode_provider_bound:
            raise RuntimeError("control mode provider is already bound")
        require_control_mode(provider(), label="active control mode")
        self._control_mode_provider = provider
        self._control_mode_provider_bound = True

    def refresh(self, env_ids: torch.Tensor | None = None) -> TBlockState:
        """从 raw view 读取 K 行并立刻复制到 runtime-owned canonical buffer。"""

        self._require_open()
        ids = (
            self._all_env_ids
            if env_ids is None
            else normalize_env_ids(
                env_ids,
                num_envs=self.num_envs,
                device=self.device,
                allow_empty=True,
            )
        )
        origins = self.env_origins.index_select(0, ids)
        for robot_index, (port, columns, full_columns) in enumerate(
            zip(
                self.robot_ports,
                self.robot_columns,
                self.robot_state_columns,
                strict=True,
            )
        ):
            full_q = _owned_rows(
                port.read_all_joint_positions(ids),
                name=f"{port.label} full joint positions",
                rows=ids.numel(),
                width=port.state_dim,
                device=self.device,
            )
            full_qd = _owned_rows(
                port.read_all_joint_velocities(ids),
                name=f"{port.label} full joint velocities",
                rows=ids.numel(),
                width=port.state_dim,
                device=self.device,
            )
            q = full_q.index_select(1, port.command_state_indices)
            qd = full_qd.index_select(1, port.command_state_indices)
            tcp_world, tcp_q = port.read_tcp_pose_wxyz(ids)
            tcp_world = _owned_rows(
                tcp_world,
                name=f"{port.label} TCP positions",
                rows=ids.numel(),
                width=3,
                device=self.device,
            )
            tcp_q = _owned_rows(
                tcp_q,
                name=f"{port.label} TCP orientations",
                rows=ids.numel(),
                width=4,
                device=self.device,
            )
            self.joint_positions[:, columns.columns].index_copy_(0, ids, q)
            self.joint_velocities[:, columns.columns].index_copy_(0, ids, qd)
            self.all_joint_positions[:, full_columns.columns].index_copy_(
                0,
                ids,
                full_q,
            )
            self.all_joint_velocities[:, full_columns.columns].index_copy_(
                0,
                ids,
                full_qd,
            )
            new_tcp_local = tcp_world - origins
            # Workaround (Gitea #67): at scale PhysX/Fabric intermittently returns a non-finite
            # (NaN) link transform for an env, and it is sticky -- re-stepping cannot clear it and
            # the articulation port exposes no link-pose overwrite. Hold each such row's
            # last-finite TCP (the persistent buffers already carry the previous good value) so no
            # consumer (observation / cuRobo IK target) ever sees NaN. Unconditional where() keeps
            # the hot path host-sync-free.
            tcp_finite = torch.isfinite(new_tcp_local).all(dim=1) & torch.isfinite(
                tcp_q
            ).all(dim=1)
            prev_local = self.tcp_positions_local[:, robot_index].index_select(0, ids)
            prev_quat = self.tcp_orientations_wxyz[:, robot_index].index_select(0, ids)
            self._nonfinite_tcp_holds += (~tcp_finite).sum()
            self.tcp_positions_local[:, robot_index].index_copy_(
                0, ids, torch.where(tcp_finite[:, None], new_tcp_local, prev_local)
            )
            self.tcp_orientations_wxyz[:, robot_index].index_copy_(
                0, ids, torch.where(tcp_finite[:, None], tcp_q, prev_quat)
            )
        block_world, block_q = self.object_port.read_pose_wxyz(ids)
        block_world = _owned_rows(
            block_world,
            name="T-block positions",
            rows=ids.numel(),
            width=3,
            device=self.device,
        )
        block_q = _owned_rows(
            block_q,
            name="T-block orientations",
            rows=ids.numel(),
            width=4,
            device=self.device,
        )
        block_velocity = _owned_rows(
            self.object_port.read_com_velocity(ids),
            name="T-block COM velocity",
            rows=ids.numel(),
            width=6,
            device=self.device,
        )
        self.block_position_local.index_copy_(0, ids, block_world - origins)
        self.block_orientation_wxyz.index_copy_(0, ids, block_q)
        self.block_com_velocity.index_copy_(0, ids, block_velocity)
        return self._selected_state(ids)

    def write_position_targets(
        self,
        targets: torch.Tensor,
        velocities: torch.Tensor | None = None,
    ) -> None:
        self._require_open()
        values = _owned_rows(
            targets,
            name="joint position targets",
            rows=self.num_envs,
            width=self.command_dim,
            device=self.device,
        )
        velocity_values = (
            None
            if velocities is None
            else _owned_rows(
                velocities,
                name="joint velocity feed-forward targets",
                rows=self.num_envs,
                width=self.command_dim,
                device=self.device,
            )
        )
        ids = self._all_env_ids
        for port, columns in zip(self.robot_ports, self.robot_columns, strict=True):
            if velocity_values is not None:
                writer = getattr(port, "write_joint_velocity_targets", None)
                if callable(writer):
                    writer(ids, velocity_values[:, columns.columns])
            # Explicit position control evaluates PD inside the position writer. Its
            # feed-forward cache must therefore contain this tick's desired velocity.
            _write_position_target(port, ids, values[:, columns.columns])
        self.control_targets.copy_(values)

    def write_velocity_targets(self, targets: torch.Tensor) -> None:
        self._write_control_targets(targets, mode="velocity")

    def write_effort_targets(self, targets: torch.Tensor) -> None:
        self._write_control_targets(targets, mode="effort")

    def write_joint_targets(self, targets: torch.Tensor) -> None:
        """Compatibility wrapper for the original position-only contract."""

        self.write_position_targets(targets)

    def commit_position_reference(self, values: torch.Tensor) -> None:
        self.position_references.copy_(
            _owned_rows(
                values,
                name="joint position references",
                rows=self.num_envs,
                width=self.command_dim,
                device=self.device,
            )
        )

    def _write_control_targets(
        self, targets: torch.Tensor, *, mode: ControlMode
    ) -> None:
        self._require_open()
        values = _owned_rows(
            targets,
            name=f"joint {mode} targets",
            rows=self.num_envs,
            width=self.command_dim,
            device=self.device,
        )
        method_name = {
            "velocity": "write_joint_velocity_targets",
            "effort": "write_joint_effort_targets",
        }.get(mode)
        if method_name is None:
            raise ValueError("control target mode must be velocity or effort")
        ids = self._all_env_ids
        for port, columns in zip(self.robot_ports, self.robot_columns, strict=True):
            getattr(port, method_name)(ids, values[:, columns.columns])
        self.control_targets.copy_(values)

    def write_reset(
        self,
        command: TBlockResetCommand,
        *,
        control_mode: ControlMode = "position",
    ) -> None:
        self._require_open()
        mode = require_control_mode(control_mode, label="reset control mode")
        ids = command.env_ids
        masked_ports: list[object] = []
        try:
            if command.device_reset_mask is not None:
                # Newton 通过该可选窄接口把 SAME_STEP bool mask 一路带到 manager；
                # PhysX port 无此方法，仍消费 task 已经 where-blend 的固定 N 行值。
                for port in (*self.robot_ports, self.object_port):
                    setter = getattr(port, "set_device_reset_mask", None)
                    if callable(setter):
                        setter(command.device_reset_mask)
                        masked_ports.append(port)
            for port, columns in zip(self.robot_ports, self.robot_columns, strict=True):
                port.write_joint_positions(
                    ids, command.joint_positions[:, columns.columns]
                )
                port.write_joint_velocities(
                    ids, command.joint_velocities[:, columns.columns]
                )
                neutral = command.joint_targets[:, columns.columns]
                if mode == "position":
                    velocity_writer = getattr(
                        port,
                        "write_joint_velocity_targets",
                        None,
                    )
                    if callable(velocity_writer):
                        velocity_writer(ids, torch.zeros_like(neutral))
                    _write_position_target(port, ids, neutral)
                elif mode == "velocity":
                    port.write_joint_velocity_targets(ids, torch.zeros_like(neutral))
                else:
                    port.write_joint_effort_targets(ids, torch.zeros_like(neutral))
            self.object_port.write_pose_wxyz(
                ids,
                command.block_position + self.env_origins.index_select(0, ids),
                command.block_orientation_wxyz,
            )
            self.object_port.write_velocity(ids, command.block_velocity)
            if self.physics_state_port is not None:
                self.physics_state_port.reset(
                    ids,
                    reset_mask=command.device_reset_mask,
                )
        finally:
            for port in reversed(masked_ports):
                port.set_device_reset_mask(None)
        self.joint_positions.index_copy_(0, ids, command.joint_positions)
        self.joint_velocities.index_copy_(0, ids, command.joint_velocities)
        self.position_references.index_copy_(0, ids, command.joint_targets)
        neutral_targets = (
            command.joint_targets
            if mode == "position"
            else torch.zeros_like(command.joint_targets)
        )
        self.control_targets.index_copy_(0, ids, neutral_targets)
        self.block_position_local.index_copy_(0, ids, command.block_position)
        self.block_orientation_wxyz.index_copy_(0, ids, command.block_orientation_wxyz)
        self.block_com_velocity.index_copy_(0, ids, command.block_velocity)

    def state_bindings(
        self, task_fields: dict[str, torch.Tensor]
    ) -> dict[str, StateBinding]:
        """建立 engine owner state + task/RNG state 的完整 snapshot 字段表。"""

        bindings: dict[str, StateBinding] = {
            "robot.q": StateBinding(self.all_joint_positions, self._write_all_q),
            "robot.qd": StateBinding(self.all_joint_velocities, self._write_all_qd),
            "robot.target": StateBinding(self.control_targets, self._write_target),
            "robot.position_reference": StateBinding(self.position_references),
            "object.pose_local_wxyz": StateBinding(
                self.block_pose_local_wxyz, self._write_block_pose
            ),
            "object.com_velocity": StateBinding(
                self.block_com_velocity, self.object_port.write_velocity
            ),
        }
        if self.physics_state_port is not None:
            name = self.physics_state_port.field_name
            if name in bindings:
                raise ValueError(f"duplicate physics state field {name!r}")
            bindings[name] = StateBinding(
                self.physics_state_port.tensor,
                self.physics_state_port.write,
            )
        for name, tensor in task_fields.items():
            if name in bindings:
                raise ValueError(f"duplicate state field {name!r}")
            bindings[name] = StateBinding(tensor, finite=tensor.is_floating_point())
        return bindings

    def close(self) -> None:
        if self._closed:
            return
        self._closing_started = True
        first_error: BaseException | None = None
        ports: tuple[object, ...] = (
            *reversed(self.robot_ports),
            self.object_port,
            *(() if self.physics_state_port is None else (self.physics_state_port,)),
        )
        for port in ports:
            key = str(getattr(port, "label", getattr(port, "field_name", "<port>")))
            if key in self._close_completed:
                continue
            try:
                port.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        "additional tensor-port close failure: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                self._close_completed.add(key)
        expected = {
            str(getattr(port, "label", getattr(port, "field_name", "<port>")))
            for port in ports
        }
        self._closed = self._close_completed == expected
        if first_error is not None:
            raise first_error

    def _selected_state(self, ids: torch.Tensor) -> TBlockState:
        def select(value: torch.Tensor) -> torch.Tensor:
            # index_select 本身创建 owned storage，task 不会借用 canonical buffer。
            return value.index_select(0, ids)

        return TBlockState(
            joint_positions=select(self.joint_positions),
            joint_velocities=select(self.joint_velocities),
            position_references=select(self.position_references),
            tcp_positions_local=select(self.tcp_positions_local),
            tcp_orientations_wxyz=select(self.tcp_orientations_wxyz),
            block_position_local=select(self.block_position_local),
            block_orientation_wxyz=select(self.block_orientation_wxyz),
            block_com_velocity=select(self.block_com_velocity),
            external_safety_stop=select(self.external_safety_stop),
        )

    def _write_all_q(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        for port, command_columns, state_columns in zip(
            self.robot_ports,
            self.robot_columns,
            self.robot_state_columns,
            strict=True,
        ):
            state = values[:, state_columns.columns]
            port.write_all_joint_positions(ids, state)
            self.joint_positions[:, command_columns.columns].index_copy_(
                0,
                ids,
                state.index_select(1, port.command_state_indices),
            )

    def _write_all_qd(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        for port, command_columns, state_columns in zip(
            self.robot_ports,
            self.robot_columns,
            self.robot_state_columns,
            strict=True,
        ):
            state = values[:, state_columns.columns]
            port.write_all_joint_velocities(ids, state)
            self.joint_velocities[:, command_columns.columns].index_copy_(
                0,
                ids,
                state.index_select(1, port.command_state_indices),
            )

    def _write_target(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        mode = require_control_mode(
            self._control_mode_provider(),
            label="active control mode",
        )
        method_name = {
            "position": "write_joint_position_targets",
            "velocity": "write_joint_velocity_targets",
            "effort": "write_joint_effort_targets",
        }[mode]
        for port, columns in zip(self.robot_ports, self.robot_columns, strict=True):
            if mode == "position":
                velocity_writer = getattr(
                    port,
                    "write_joint_velocity_targets",
                    None,
                )
                if callable(velocity_writer):
                    velocity_writer(ids, torch.zeros_like(values[:, columns.columns]))
                _write_position_target(port, ids, values[:, columns.columns])
            else:
                getattr(port, method_name)(ids, values[:, columns.columns])

    def _write_block_pose(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        self.object_port.write_pose_wxyz(
            ids,
            values[:, :3] + self.env_origins.index_select(0, ids),
            values[:, 3:7],
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("KaleidoscopeTensorViews is closed")
        if self._closing_started:
            raise RuntimeError("KaleidoscopeTensorViews teardown has started")


def _owned_rows(
    value: object,
    *,
    name: str,
    rows: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = require_cuda_tensor(
        value, name=name, ndim=2, leading_dim=rows, dtype=torch.float32
    )
    if tensor.device != device or tensor.shape[1:] != (width,):
        raise ValueError(f"{name} must have shape ({rows},{width}) on {device}")
    # Isaac raw tensor 可能在下一次 physics step 被后端复用覆盖；canonical view 必须立即 clone，
    # 不能把 borrowed storage 暴露给 task、snapshot 或训练 adapter。
    return tensor.clone(memory_format=torch.contiguous_format)


def _write_position_target(
    port: object,
    env_ids: torch.Tensor,
    values: torch.Tensor,
) -> None:
    writer = getattr(port, "write_joint_position_targets", None)
    if callable(writer):
        writer(env_ids, values)
        return
    legacy = getattr(port, "write_joint_targets", None)
    if not callable(legacy):
        raise TypeError(
            f"robot port {getattr(port, 'label', '<unknown>')!r} has no position writer"
        )
    legacy(env_ids, values)


__all__ = [
    "KaleidoscopeTensorViews",
    "PhysicsStateTensorPort",
    "RigidObjectTensorPort",
    "RobotColumns",
    "RobotTensorPort",
]
