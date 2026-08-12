"""project-owned Newton 到 Kaleidoscope CUDA tensor port 的零拷贝适配层。

Newton 的唯一状态 owner 是 Warp ``State/Control``；本模块只通过 DLPack 建立临时 Torch
别名，并用构造期上传的 topology index 在 GPU 上 gather/scatter。所有返回 tensor 都是
persistent borrowed buffer：调用方必须在下一次同字段读取前消费或复制它们。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import math

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
from linkerbot_sim.kaleidoscope.tensors import (
    assert_finite_async,
    normalize_env_ids,
    require_cuda_tensor,
)
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


@dataclass(frozen=True, slots=True)
class PreparedNewtonControlRuntime:
    common: PreparedKaleidoscopeControlRuntime
    drive_stiffness: torch.Tensor
    drive_damping: torch.Tensor
    effort_limits: torch.Tensor
    drive_stiffness_warp: object
    drive_damping_warp: object
    effort_limits_warp: object
    physical_mode_indices: tuple[tuple[str, tuple[int, ...]], ...]


def _warp_alias(value: object, *, device: torch.device, name: str) -> torch.Tensor:
    """用 Warp DLPack capsule 建立零拷贝 Torch alias，禁止任何 host fallback。"""

    import warp as wp

    try:
        tensor = torch.from_dlpack(wp.to_dlpack(value))
    except Exception as exc:
        raise TypeError(f"{name} is not a DLPack-compatible Warp array") from exc
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must expose float32 scalars, got {tensor.dtype}")
    return tensor


@contextmanager
def _owner_stream_scope(view: object, *, device: torch.device) -> Iterator[None]:
    """用 CUDA event 在 caller Torch stream 与 Newton owner stream 间双向交接。"""

    owner = getattr(view, "owner_stream", None)
    if owner is None:
        # Raw Newton view 通过 owner_stream 暴露 manager stream；solver-state port 直接
        # 持有 NewtonRuntime，因此从 runtime.stream 取得同一 CUDA stream。
        owner = getattr(view, "stream")
    pointer = int(getattr(owner, "cuda_stream"))
    external = torch.cuda.ExternalStream(pointer, device=device)
    caller = torch.cuda.current_stream(device)
    if caller.cuda_stream != external.cuda_stream:
        external.wait_stream(caller)
    try:
        with torch.cuda.stream(external):
            yield
    finally:
        if caller.cuda_stream != external.cuda_stream:
            caller.wait_stream(external)


def _device_index_matrix(
    rows: Sequence[Sequence[int]],
    *,
    columns: Sequence[int],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """把 finalized topology 的 host metadata 一次性上传为 GPU index matrix。"""

    selected = tuple(tuple(int(row[column]) for column in columns) for row in rows)
    if not selected or not selected[0]:
        raise ValueError(f"{name} cannot be empty")
    width = len(selected[0])
    if any(len(row) != width for row in selected):
        raise ValueError(f"{name} rows must have identical widths")
    return torch.tensor(selected, device=device, dtype=torch.int64)


class _MappedScalarIO:
    """一维 Warp owner array 的 persistent 二维 GPU gather/scatter。"""

    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices
        self._selected: dict[tuple[str, int], torch.Tensor] = {}
        self._outputs: dict[tuple[str, int], torch.Tensor] = {}
        self._subsets: dict[tuple[int, int], _MappedScalarIO] = {}

    @property
    def row_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])

    def gather(
        self,
        owner: torch.Tensor,
        env_ids: torch.Tensor,
        *,
        slot: str,
    ) -> torch.Tensor:
        selected = self._selection(env_ids, slot=slot)
        output = self._output(env_ids.numel(), slot=slot)
        torch.index_select(
            owner.reshape(-1),
            0,
            selected.reshape(-1),
            out=output.reshape(-1),
        )
        return output

    def scatter(
        self,
        owner: torch.Tensor,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        slot: str,
    ) -> None:
        selected = self._selection(env_ids, slot=slot)
        owner.reshape(-1).index_copy_(
            0,
            selected.reshape(-1),
            values.reshape(-1),
        )

    def subset(self, columns: torch.Tensor) -> "_MappedScalarIO":
        """Return a construction-time cached column projection of this mapping."""

        selector = require_cuda_tensor(
            columns,
            name="Newton mapped scalar columns",
            ndim=1,
            dtype=torch.int64,
        )
        if selector.device != self.indices.device:
            raise ValueError(
                "Newton mapped scalar columns must share the mapping device"
            )
        if selector.numel() < 1:
            raise ValueError("Newton mapped scalar columns cannot be empty")
        torch._assert_async(
            torch.all((selector >= 0) & (selector < self.width)),
            "Newton mapped scalar columns are out of range",
        )
        key = (selector.data_ptr(), selector.numel())
        result = self._subsets.get(key)
        if result is None:
            result = _MappedScalarIO(
                self.indices.index_select(1, selector).contiguous()
            )
            self._subsets[key] = result
        return result

    def prewarm(
        self,
        env_ids: torch.Tensor,
        *,
        gather_slots: Sequence[str] = (),
        scatter_slots: Sequence[str] = (),
    ) -> None:
        """Allocate stable selector/output buffers before a switch or graph capture."""

        for slot in gather_slots:
            self._selection(env_ids, slot=slot)
            self._output(env_ids.numel(), slot=slot)
        for slot in scatter_slots:
            self._selection(env_ids, slot=slot)

    def _selection(self, env_ids: torch.Tensor, *, slot: str) -> torch.Tensor:
        key = (slot, env_ids.numel())
        selected = self._selected.get(key)
        if selected is None:
            selected = torch.empty(
                (env_ids.numel(), self.width),
                device=self.indices.device,
                dtype=torch.int64,
            )
            self._selected[key] = selected
        torch.index_select(self.indices, 0, env_ids, out=selected)
        return selected

    def _output(self, rows: int, *, slot: str) -> torch.Tensor:
        key = (slot, rows)
        output = self._outputs.get(key)
        if output is None:
            output = torch.empty(
                (rows, self.width),
                device=self.indices.device,
                dtype=torch.float32,
            )
            self._outputs[key] = output
        return output

    def clear(self) -> None:
        for subset in self._subsets.values():
            subset.clear()
        self._subsets.clear()
        self._selected.clear()
        self._outputs.clear()
        self.indices = torch.empty(0, device=self.indices.device, dtype=torch.int64)


class _BodyIO:
    """Newton transform/spatial-vector array 的 env-row GPU 映射。"""

    def __init__(
        self,
        view: object,
        *,
        device: torch.device,
        expected_worlds: tuple[int, ...],
        require_writable: bool,
    ) -> None:
        binding = getattr(view, "binding", None)
        body_indices = tuple(
            int(value) for value in getattr(binding, "body_indices", ())
        )
        worlds = tuple(int(value) for value in getattr(binding, "world_indices", ()))
        if not body_indices or len(body_indices) != len(expected_worlds):
            raise ValueError(
                "Newton body view must bind exactly one body per environment"
            )
        if worlds != expected_worlds:
            raise ValueError("Newton body/articulation world order must match")
        self.view = view
        self.device = device
        self.row_count = len(body_indices)
        self.body_indices = torch.tensor(body_indices, device=device, dtype=torch.int64)
        self._selected_body: dict[tuple[str, int], torch.Tensor] = {}
        self._buffers: dict[tuple[str, int, int], torch.Tensor] = {}
        self.free_q: _MappedScalarIO | None = None
        self.free_qd: _MappedScalarIO | None = None
        if require_writable:
            q_starts = tuple(
                int(value) for value in getattr(binding, "free_q_starts", ())
            )
            qd_starts = tuple(
                int(value) for value in getattr(binding, "free_qd_starts", ())
            )
            if (
                len(q_starts) != self.row_count
                or len(qd_starts) != self.row_count
                or any(value < 0 for value in (*q_starts, *qd_starts))
            ):
                raise ValueError(
                    "Newton rigid object port requires world-root FREE bodies"
                )
            self.free_q = _MappedScalarIO(
                torch.tensor(
                    [[start + column for column in range(7)] for start in q_starts],
                    device=device,
                    dtype=torch.int64,
                )
            )
            self.free_qd = _MappedScalarIO(
                torch.tensor(
                    [[start + column for column in range(6)] for start in qd_starts],
                    device=device,
                    dtype=torch.int64,
                )
            )

    def read_pose(
        self,
        env_ids: torch.Tensor,
        *,
        offset_position: torch.Tensor | None = None,
        offset_orientation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _owner_stream_scope(self.view, device=self.device):
            owner = _warp_alias(
                self.view.borrow_runtime_array(category="state", field="body_q"),
                device=self.device,
                name="Newton body_q",
            )
            raw = self._gather_body(owner, env_ids, slot="pose", width=7)
            orientation = self._buffer("orientation_wxyz", env_ids.numel(), 4)
            orientation[:, 0].copy_(raw[:, 6])
            orientation[:, 1:4].copy_(raw[:, 3:6])
            if offset_position is None or offset_orientation is None:
                return raw[:, :3], orientation
            position = self._buffer("offset_position", env_ids.numel(), 3)
            position.copy_(
                raw[:, :3]
                + quaternion_rotate_wxyz(
                    orientation,
                    offset_position[None, :].expand(env_ids.numel(), -1),
                )
            )
            composed = self._buffer("offset_orientation", env_ids.numel(), 4)
            composed.copy_(
                quaternion_multiply_wxyz(
                    orientation,
                    offset_orientation[None, :].expand(env_ids.numel(), -1),
                    normalize_result=True,
                )
            )
            return position, composed

    def read_velocity(self, env_ids: torch.Tensor) -> torch.Tensor:
        with _owner_stream_scope(self.view, device=self.device):
            owner = _warp_alias(
                self.view.borrow_runtime_array(category="state", field="body_qd"),
                device=self.device,
                name="Newton body_qd",
            )
            return self._gather_body(owner, env_ids, slot="velocity", width=6)

    def write_pose(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        orientations_wxyz: torch.Tensor,
        *,
        preserve_row_mask: torch.Tensor | None = None,
        device_row_mask: torch.Tensor | None = None,
    ) -> None:
        if self.free_q is None:
            raise RuntimeError("this Newton body mapping is read-only")
        norm = torch.linalg.vector_norm(orientations_wxyz, dim=1, keepdim=True)
        torch._assert_async(torch.all(norm > 1.0e-8), "rigid quaternion norm is zero")
        with _owner_stream_scope(self.view, device=self.device):
            raw = self._buffer("pose_write", env_ids.numel(), 7)
            raw[:, :3].copy_(positions)
            normalized = orientations_wxyz / norm
            raw[:, 3:6].copy_(normalized[:, 1:4])
            raw[:, 6].copy_(normalized[:, 0])
            body_q = _warp_alias(
                self.view.borrow_runtime_array(category="state", field="body_q"),
                device=self.device,
                name="Newton body_q",
            )
            body_values = raw
            if preserve_row_mask is not None:
                current = self._gather_body(
                    body_q,
                    env_ids,
                    slot="pose_write_current",
                    width=7,
                )
                body_values = torch.where(preserve_row_mask[:, None], raw, current)
            body_q.index_copy_(
                0,
                self._body_selection(env_ids, slot="pose_write"),
                body_values,
            )
            joint_q = _warp_alias(
                self.view.borrow_runtime_array(category="state", field="joint_q"),
                device=self.device,
                name="Newton joint_q",
            )
            joint_values = raw
            if preserve_row_mask is not None:
                current_joint = self.free_q.gather(
                    joint_q,
                    env_ids,
                    slot="pose_write_current_joint",
                )
                joint_values = torch.where(
                    preserve_row_mask[:, None], raw, current_joint
                )
            self.free_q.scatter(
                joint_q,
                env_ids,
                joint_values,
                slot="pose_write",
            )
        self.view.notify_device_write(
            category="state",
            field="body_q",
            device_row_mask=device_row_mask,
        )

    def write_velocity(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        preserve_row_mask: torch.Tensor | None = None,
        device_row_mask: torch.Tensor | None = None,
    ) -> None:
        if self.free_qd is None:
            raise RuntimeError("this Newton body mapping is read-only")
        with _owner_stream_scope(self.view, device=self.device):
            body_qd = _warp_alias(
                self.view.borrow_runtime_array(category="state", field="body_qd"),
                device=self.device,
                name="Newton body_qd",
            )
            body_values = values
            if preserve_row_mask is not None:
                current = self._gather_body(
                    body_qd,
                    env_ids,
                    slot="velocity_write_current",
                    width=6,
                )
                body_values = torch.where(preserve_row_mask[:, None], values, current)
            body_qd.index_copy_(
                0,
                self._body_selection(env_ids, slot="velocity_write"),
                body_values,
            )
            joint_qd = _warp_alias(
                self.view.borrow_runtime_array(category="state", field="joint_qd"),
                device=self.device,
                name="Newton joint_qd",
            )
            joint_values = values
            if preserve_row_mask is not None:
                current_joint = self.free_qd.gather(
                    joint_qd,
                    env_ids,
                    slot="velocity_write_current_joint",
                )
                joint_values = torch.where(
                    preserve_row_mask[:, None], values, current_joint
                )
            self.free_qd.scatter(
                joint_qd,
                env_ids,
                joint_values,
                slot="velocity_write",
            )
        self.view.notify_device_write(
            category="state",
            field="body_qd",
            device_row_mask=device_row_mask,
        )

    def _gather_body(
        self,
        owner: torch.Tensor,
        env_ids: torch.Tensor,
        *,
        slot: str,
        width: int,
    ) -> torch.Tensor:
        output = self._buffer(slot, env_ids.numel(), width)
        torch.index_select(
            owner,
            0,
            self._body_selection(env_ids, slot=slot),
            out=output,
        )
        return output

    def _body_selection(self, env_ids: torch.Tensor, *, slot: str) -> torch.Tensor:
        key = (slot, env_ids.numel())
        selected = self._selected_body.get(key)
        if selected is None:
            selected = torch.empty(
                env_ids.numel(), device=self.device, dtype=torch.int64
            )
            self._selected_body[key] = selected
        torch.index_select(self.body_indices, 0, env_ids, out=selected)
        return selected

    def _buffer(self, slot: str, rows: int, width: int) -> torch.Tensor:
        key = (slot, rows, width)
        result = self._buffers.get(key)
        if result is None:
            result = torch.empty((rows, width), device=self.device, dtype=torch.float32)
            self._buffers[key] = result
        return result

    def clear(self) -> None:
        self._selected_body.clear()
        self._buffers.clear()
        if self.free_q is not None:
            self.free_q.clear()
        if self.free_qd is not None:
            self.free_qd.clear()
        self.body_indices = torch.empty(0, device=self.device, dtype=torch.int64)


@dataclass(slots=True)
class NewtonArticulationTensorPort:
    """一个 replicated robot 的 controlled joint 与 TCP CUDA port。"""

    label: str
    view: object
    tcp_view: object
    command_dof_names: tuple[str, ...]
    device: torch.device
    tcp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    command_dim: int = field(init=False)
    state_dim: int = field(init=False)
    num_envs: int = field(init=False)
    command_state_indices: torch.Tensor = field(init=False, repr=False)
    _q: _MappedScalarIO = field(init=False, repr=False)
    _qd: _MappedScalarIO = field(init=False, repr=False)
    _all_q: _MappedScalarIO = field(init=False, repr=False)
    _all_qd: _MappedScalarIO = field(init=False, repr=False)
    _position_target: _MappedScalarIO = field(init=False, repr=False)
    _velocity_target: _MappedScalarIO = field(init=False, repr=False)
    _effort_target: _MappedScalarIO = field(init=False, repr=False)
    _tcp: _BodyIO = field(init=False, repr=False)
    _tcp_offset_position: torch.Tensor = field(init=False, repr=False)
    _tcp_offset_orientation: torch.Tensor = field(init=False, repr=False)
    _all_env_ids: torch.Tensor = field(init=False, repr=False)
    _command_state_indices_host: tuple[int, ...] = field(init=False, repr=False)
    _position_feedforward: torch.Tensor = field(init=False, repr=False)
    _active_control_runtime: PreparedNewtonControlRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _device_reset_mask: torch.Tensor | None = field(
        init=False, default=None, repr=False
    )
    _state_row_masks: dict[str, torch.Tensor] = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _close_completed: set[str] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise ValueError("Newton articulation tensor port requires CUDA")
        if len(self.tcp_offset_xyz) != 3 or not all(
            math.isfinite(float(value)) for value in self.tcp_offset_xyz
        ):
            raise ValueError("TCP offset XYZ must contain three finite values")
        names = tuple(str(name) for name in self.command_dof_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("command_dof_names must be non-empty and unique")
        self.view.bind_controllable_dofs(names)
        binding = getattr(self.view, "binding", None)
        dof_names = tuple(str(name) for name in getattr(binding, "dof_names", ()))
        by_name = {name: index for index, name in enumerate(dof_names)}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Newton articulation is missing command DOFs: {missing}")
        columns = tuple(by_name[name] for name in names)
        worlds = tuple(int(value) for value in getattr(binding, "world_indices", ()))
        if not worlds:
            raise ValueError("Newton articulation binding has no worlds")
        self.command_dof_names = names
        self.command_dim = len(columns)
        self.state_dim = len(dof_names)
        self.num_envs = len(worlds)
        self.command_state_indices = torch.tensor(
            columns,
            device=self.device,
            dtype=torch.int64,
        )
        self._command_state_indices_host = columns
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
        self._state_row_masks = {
            name: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for name in ("joint_q", "joint_qd", "joint_q_full", "joint_qd_full")
        }
        all_columns = tuple(range(self.state_dim))
        self._q = _MappedScalarIO(
            _device_index_matrix(
                binding.q_indices,
                columns=columns,
                device=self.device,
                name="Newton command q indices",
            )
        )
        qd_indices = _device_index_matrix(
            binding.qd_indices,
            columns=columns,
            device=self.device,
            name="Newton command qd indices",
        )
        self._qd = _MappedScalarIO(qd_indices)
        self._position_target = _MappedScalarIO(qd_indices.clone())
        self._velocity_target = _MappedScalarIO(qd_indices.clone())
        self._effort_target = _MappedScalarIO(qd_indices.clone())
        self._all_q = _MappedScalarIO(
            _device_index_matrix(
                binding.q_indices,
                columns=all_columns,
                device=self.device,
                name="Newton full q indices",
            )
        )
        self._all_qd = _MappedScalarIO(
            _device_index_matrix(
                binding.qd_indices,
                columns=all_columns,
                device=self.device,
                name="Newton full qd indices",
            )
        )
        self._tcp = _BodyIO(
            self.tcp_view,
            device=self.device,
            expected_worlds=worlds,
            require_writable=False,
        )
        if int(self.view.owner_stream.cuda_stream) != int(
            self.tcp_view.owner_stream.cuda_stream
        ):
            raise ValueError(
                "Newton articulation and TCP views must share one owner stream"
            )
        self._tcp_offset_position = torch.tensor(
            self.tcp_offset_xyz, device=self.device, dtype=torch.float32
        )
        self._tcp_offset_orientation = torch.tensor(
            rpy_xyz_to_quat_wxyz(self.tcp_offset_rpy),
            device=self.device,
            dtype=torch.float32,
        )
        self._validate_owner_devices()

    def read_joint_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        ids = self._env_ids(env_ids)
        with _owner_stream_scope(self.view, device=self.device):
            owner = self._owner_alias("state", "joint_q")
            return self._q.gather(owner, ids, slot="joint_q")

    def read_joint_velocities(self, env_ids: torch.Tensor) -> torch.Tensor:
        ids = self._env_ids(env_ids)
        with _owner_stream_scope(self.view, device=self.device):
            owner = self._owner_alias("state", "joint_qd")
            return self._qd.gather(owner, ids, slot="joint_qd")

    def read_all_joint_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        ids = self._env_ids(env_ids)
        with _owner_stream_scope(self.view, device=self.device):
            owner = self._owner_alias("state", "joint_q")
            return self._all_q.gather(owner, ids, slot="all_joint_q")

    def read_all_joint_velocities(self, env_ids: torch.Tensor) -> torch.Tensor:
        ids = self._env_ids(env_ids)
        with _owner_stream_scope(self.view, device=self.device):
            owner = self._owner_alias("state", "joint_qd")
            return self._all_qd.gather(owner, ids, slot="all_joint_qd")

    def read_tcp_pose_wxyz(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._env_ids(env_ids)
        return self._tcp.read_pose(
            ids,
            offset_position=self._tcp_offset_position,
            offset_orientation=self._tcp_offset_orientation,
        )

    def write_joint_positions(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        self._write_joint("state", "joint_q", self._q, env_ids, values)

    def write_joint_velocities(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        self._write_joint("state", "joint_qd", self._qd, env_ids, values)

    def write_joint_position_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        prepared = self._require_active_control_runtime("position")
        ids = self._env_ids(env_ids)
        data = self._values(
            values,
            rows=ids.numel(),
            width=self.command_dim,
            name="position targets",
        )
        implicit = prepared.common.implicit_indices
        if implicit.numel():
            self._write_joint(
                "control",
                "joint_target_pos",
                self._position_target.subset(implicit),
                ids,
                data.index_select(1, implicit),
                width=implicit.numel(),
            )
        explicit = prepared.common.explicit_indices
        if explicit.numel():
            q = self.read_joint_positions(ids).index_select(1, explicit)
            qd = self.read_joint_velocities(ids).index_select(1, explicit)
            desired_qd = self._position_feedforward.index_select(0, ids).index_select(
                1, explicit
            )
            effort = prepared.common.stiffness.index_select(0, explicit) * (
                data.index_select(1, explicit) - q
            ) + prepared.common.damping.index_select(0, explicit) * (desired_qd - qd)
            limit = prepared.common.effort_limits.index_select(0, explicit)
            self._write_joint(
                "control",
                "joint_f",
                self._effort_target.subset(explicit),
                ids,
                torch.clamp(effort, min=-limit, max=limit),
                width=explicit.numel(),
            )

    def write_joint_velocity_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        prepared = self._require_active_control_runtime(("position", "velocity"))
        ids = self._env_ids(env_ids)
        data = self._values(
            values,
            rows=ids.numel(),
            width=self.command_dim,
            name="velocity targets",
        )
        if prepared.common.mode == "position":
            self._position_feedforward.index_copy_(0, ids, data)
        implicit = prepared.common.implicit_indices
        if implicit.numel():
            self._write_joint(
                "control",
                "joint_target_vel",
                self._velocity_target.subset(implicit),
                ids,
                data.index_select(1, implicit),
                width=implicit.numel(),
            )
        explicit = prepared.common.explicit_indices
        if explicit.numel() and prepared.common.mode == "velocity":
            qd = self.read_joint_velocities(ids).index_select(1, explicit)
            effort = prepared.common.damping.index_select(0, explicit) * (
                data.index_select(1, explicit) - qd
            )
            limit = prepared.common.effort_limits.index_select(0, explicit)
            self._write_joint(
                "control",
                "joint_f",
                self._effort_target.subset(explicit),
                ids,
                torch.clamp(effort, min=-limit, max=limit),
                width=explicit.numel(),
            )

    def write_joint_effort_targets(
        self, env_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        prepared = self._require_active_control_runtime("effort")
        ids = self._env_ids(env_ids)
        data = self._values(
            values,
            rows=ids.numel(),
            width=self.command_dim,
            name="effort targets",
        )
        limit = prepared.common.effort_limits
        self._write_joint(
            "control",
            "joint_f",
            self._effort_target,
            ids,
            torch.clamp(data, min=-limit, max=limit),
        )

    def write_joint_targets(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Compatibility wrapper for position-only callers."""

        if self._active_control_runtime is None:
            self._write_joint(
                "control",
                "joint_target_pos",
                self._position_target,
                env_ids,
                values,
            )
            return
        self.write_joint_position_targets(env_ids, values)

    def prepare_control_runtime(
        self,
        projection: CommandRuntimeProjection,
    ) -> PreparedNewtonControlRuntime:
        if projection.joint_names != self.command_dof_names:
            raise ValueError(
                "Newton control projection joint order does not match port"
            )
        self._require_control_mutation_api()
        common = prepare_device_control_runtime(
            projection,
            mode=projection.modes[0],
            device=self.device,
        )

        def rows(values: torch.Tensor) -> torch.Tensor:
            return values[None, :].expand(self.num_envs, -1).contiguous()

        physical: list[tuple[str, tuple[int, ...]]] = []
        for mode in dict.fromkeys(projection.physical_modes):
            relative = tuple(
                index
                for index, value in enumerate(projection.physical_modes)
                if value == mode
            )
            indices = tuple(
                self._command_state_indices_host[index] for index in relative
            )
            self.view.prepare_dof_control_runtime(
                indices=None,
                dof_indices=indices,
            )
            physical.append(
                (
                    mode,
                    indices,
                )
            )
        if common.implicit_indices.numel():
            self._position_target.subset(common.implicit_indices).prewarm(
                self._all_env_ids,
                scatter_slots=("write_joint_target_pos",),
            )
            self._velocity_target.subset(common.implicit_indices).prewarm(
                self._all_env_ids,
                scatter_slots=("write_joint_target_vel",),
            )
        if common.explicit_indices.numel():
            self._q.prewarm(self._all_env_ids, gather_slots=("joint_q",))
            self._qd.prewarm(self._all_env_ids, gather_slots=("joint_qd",))
            self._effort_target.subset(common.explicit_indices).prewarm(
                self._all_env_ids,
                scatter_slots=("write_joint_f",),
            )
        if common.mode == "effort":
            self._effort_target.prewarm(
                self._all_env_ids,
                scatter_slots=("write_joint_f",),
            )
        drive_stiffness = rows(common.drive_stiffness)
        drive_damping = rows(common.drive_damping)
        effort_limits = rows(common.effort_limits)
        import warp as wp

        return PreparedNewtonControlRuntime(
            common=common,
            drive_stiffness=drive_stiffness,
            drive_damping=drive_damping,
            effort_limits=effort_limits,
            drive_stiffness_warp=wp.from_torch(
                drive_stiffness,
                dtype=wp.float32,
            ),
            drive_damping_warp=wp.from_torch(
                drive_damping,
                dtype=wp.float32,
            ),
            effort_limits_warp=wp.from_torch(
                effort_limits,
                dtype=wp.float32,
            ),
            physical_mode_indices=tuple(physical),
        )

    def validate_prepared_control_runtime(
        self,
        prepared: PreparedNewtonControlRuntime,
    ) -> None:
        if not isinstance(prepared, PreparedNewtonControlRuntime):
            raise TypeError("prepared must be PreparedNewtonControlRuntime")
        if prepared.common.device != self.device:
            raise ValueError("prepared Newton control runtime has the wrong device")
        self._require_control_mutation_api()

    def apply_prepared_control_runtime(
        self,
        prepared: PreparedNewtonControlRuntime,
    ) -> None:
        self.validate_prepared_control_runtime(prepared)
        for mode, indices in prepared.physical_mode_indices:
            self.view.switch_dof_control_mode(
                mode,
                indices=None,
                dof_indices=indices,
            )
        self.view.set_dof_gains(
            stiffnesses=prepared.drive_stiffness_warp,
            dampings=prepared.drive_damping_warp,
            indices=None,
            dof_indices=self._command_state_indices_host,
            update_default_gains=False,
        )
        self.view.set_dof_max_efforts(
            prepared.effort_limits_warp,
            indices=None,
            dof_indices=self._command_state_indices_host,
        )
        self.view.set_dof_drive_types(
            "force",
            dof_indices=self._command_state_indices_host,
        )
        self._active_control_runtime = prepared

    def synchronize_control_writes(self) -> None:
        owner = getattr(self.view, "owner_stream", None)
        synchronizer = getattr(owner, "synchronize", None)
        if callable(synchronizer):
            synchronizer()
        else:
            torch.cuda.synchronize(self.device)

    def _require_active_control_runtime(
        self,
        expected: str | tuple[str, ...],
    ) -> PreparedNewtonControlRuntime:
        prepared = self._active_control_runtime
        if prepared is None:
            raise RuntimeError("Newton control runtime has not been configured")
        modes = (expected,) if isinstance(expected, str) else expected
        if prepared.common.mode not in modes:
            raise RuntimeError(
                f"Newton {self.label} is in {prepared.common.mode!r} control mode; "
                f"expected one of {modes}"
            )
        return prepared

    def _require_control_mutation_api(self) -> None:
        for name in (
            "switch_dof_control_mode",
            "set_dof_gains",
            "set_dof_max_efforts",
            "set_dof_drive_types",
            "prepare_dof_control_runtime",
        ):
            if not callable(getattr(self.view, name, None)):
                raise RuntimeError(
                    f"Newton articulation does not expose required {name}() mutation"
                )

    def write_all_joint_positions(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self._write_joint(
            "state",
            "joint_q",
            self._all_q,
            env_ids,
            values,
            width=self.state_dim,
            notification_field="joint_q_full",
        )

    def write_all_joint_velocities(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self._write_joint(
            "state",
            "joint_qd",
            self._all_qd,
            env_ids,
            values,
            width=self.state_dim,
            notification_field="joint_qd_full",
        )

    def set_device_reset_mask(self, value: torch.Tensor | None) -> None:
        """设置一次固定 N 行 SAME_STEP 写入使用的 CUDA bool mask。"""

        if value is None:
            self._device_reset_mask = None
            return
        mask = require_cuda_tensor(
            value,
            name="Newton articulation reset mask",
            ndim=1,
            leading_dim=self.num_envs,
            dtype=torch.bool,
        )
        if mask.device != self.device:
            raise ValueError(
                f"Newton articulation reset mask must live on {self.device}"
            )
        self._device_reset_mask = mask

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        children = (("tcp", self.tcp_view), ("articulation", self.view))
        for name, raw_view in children:
            if name in self._close_completed:
                continue
            try:
                raw_view.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._close_completed.add(name)
        if len(self._close_completed) == len(children):
            self._q.clear()
            self._qd.clear()
            self._all_q.clear()
            self._all_qd.clear()
            self._position_target.clear()
            self._velocity_target.clear()
            self._effort_target.clear()
            self._tcp.clear()
            self._closed = True
        if first_error is not None:
            raise first_error

    def _write_joint(
        self,
        category: str,
        field_name: str,
        mapping: _MappedScalarIO,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        width: int | None = None,
        notification_field: str | None = None,
    ) -> None:
        ids = self._env_ids(env_ids)
        data = self._values(
            values,
            rows=ids.numel(),
            width=self.command_dim if width is None else width,
            name=field_name,
        )
        with _owner_stream_scope(self.view, device=self.device):
            owner = self._owner_alias(category, field_name)
            notify_field = (
                field_name if notification_field is None else notification_field
            )
            preserve_row_mask, device_row_mask = self._state_masks_for(
                ids,
                category=category,
                field=notify_field,
            )
            if preserve_row_mask is not None:
                current = mapping.gather(
                    owner,
                    ids,
                    slot=f"preserve_{field_name}",
                )
                data = torch.where(preserve_row_mask[:, None], data, current)
            mapping.scatter(owner, ids, data, slot=f"write_{field_name}")
        self.view.notify_device_write(
            category=category,
            field=notify_field,
            device_row_mask=device_row_mask,
        )

    def _state_masks_for(
        self,
        ids: torch.Tensor,
        *,
        category: str,
        field: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        mask = self._device_reset_mask
        if mask is not None:
            if ids.shape != (self.num_envs,):
                raise ValueError("device-masked reset must use one row per environment")
            torch._assert_async(
                torch.all(ids == self._all_env_ids),
                "device-masked reset env_ids must use canonical environment order",
            )
            return mask, mask
        if category != "state":
            return None, None
        selected = self._state_row_masks[field]
        selected.zero_()
        selected.index_fill_(0, ids, True)
        return None, selected

    def _owner_alias(self, category: str, field_name: str) -> torch.Tensor:
        return _warp_alias(
            self.view.borrow_runtime_array(category=category, field=field_name),
            device=self.device,
            name=f"Newton {field_name}",
        )

    def _env_ids(self, value: torch.Tensor) -> torch.Tensor:
        self._require_open()
        return normalize_env_ids(
            value,
            num_envs=self.num_envs,
            device=self.device,
            allow_empty=True,
        )

    def _values(
        self,
        value: torch.Tensor,
        *,
        rows: int,
        width: int,
        name: str,
    ) -> torch.Tensor:
        result = require_cuda_tensor(
            value,
            name=f"{self.label} {name}",
            ndim=2,
            leading_dim=rows,
            dtype=torch.float32,
        )
        if result.device != self.device or result.shape[1:] != (width,):
            raise ValueError(
                f"{self.label} {name} must have shape ({rows},{width}) on {self.device}"
            )
        assert_finite_async(result, name=f"{self.label} {name}")
        return result.contiguous()

    def _validate_owner_devices(self) -> None:
        for view, category, name in (
            (self.view, "state", "joint_q"),
            (self.view, "state", "joint_qd"),
            (self.view, "control", "joint_target_pos"),
            (self.view, "control", "joint_target_vel"),
            (self.view, "control", "joint_f"),
            (self.tcp_view, "state", "body_q"),
        ):
            _warp_alias(
                view.borrow_runtime_array(category=category, field=name),
                device=self.device,
                name=f"Newton {name}",
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Newton articulation tensor port is closed")


@dataclass(slots=True)
class NewtonRigidObjectTensorPort:
    """一个 replicated world-root FREE object 的 pose/velocity CUDA port。"""

    label: str
    view: object
    device: torch.device
    num_envs: int = field(init=False)
    _body: _BodyIO = field(init=False, repr=False)
    _all_env_ids: torch.Tensor = field(init=False, repr=False)
    _device_reset_mask: torch.Tensor | None = field(
        init=False, default=None, repr=False
    )
    _state_row_masks: dict[str, torch.Tensor] = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise ValueError("Newton rigid object tensor port requires CUDA")
        binding = getattr(self.view, "binding", None)
        worlds = tuple(int(value) for value in getattr(binding, "world_indices", ()))
        if not worlds:
            raise ValueError("Newton rigid object binding has no worlds")
        self.num_envs = len(worlds)
        self._all_env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.int64,
        )
        self._state_row_masks = {
            name: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for name in ("body_q", "body_qd")
        }
        self._body = _BodyIO(
            self.view,
            device=self.device,
            expected_worlds=worlds,
            require_writable=True,
        )
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            _warp_alias(
                self.view.borrow_runtime_array(category="state", field=name),
                device=self.device,
                name=f"Newton {name}",
            )

    def read_pose_wxyz(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._body.read_pose(self._env_ids(env_ids))

    def read_com_velocity(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._body.read_velocity(self._env_ids(env_ids))

    def write_pose_wxyz(
        self,
        env_ids: torch.Tensor,
        positions_world: torch.Tensor,
        orientations_wxyz: torch.Tensor,
    ) -> None:
        ids = self._env_ids(env_ids)
        positions = _rigid_values(
            positions_world,
            rows=ids.numel(),
            width=3,
            device=self.device,
            name=f"{self.label} positions",
        )
        orientations = _rigid_values(
            orientations_wxyz,
            rows=ids.numel(),
            width=4,
            device=self.device,
            name=f"{self.label} orientations",
        )
        preserve_mask, device_mask = self._state_masks_for(ids, field="body_q")
        self._body.write_pose(
            ids,
            positions,
            orientations,
            preserve_row_mask=preserve_mask,
            device_row_mask=device_mask,
        )

    def write_velocity(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        ids = self._env_ids(env_ids)
        velocity = _rigid_values(
            values,
            rows=ids.numel(),
            width=6,
            device=self.device,
            name=f"{self.label} COM velocity",
        )
        preserve_mask, device_mask = self._state_masks_for(ids, field="body_qd")
        self._body.write_velocity(
            ids,
            velocity,
            preserve_row_mask=preserve_mask,
            device_row_mask=device_mask,
        )

    def set_device_reset_mask(self, value: torch.Tensor | None) -> None:
        """设置一次固定 N 行 SAME_STEP 写入使用的 CUDA bool mask。"""

        if value is None:
            self._device_reset_mask = None
            return
        mask = require_cuda_tensor(
            value,
            name="Newton rigid reset mask",
            ndim=1,
            leading_dim=self.num_envs,
            dtype=torch.bool,
        )
        if mask.device != self.device:
            raise ValueError(f"Newton rigid reset mask must live on {self.device}")
        self._device_reset_mask = mask

    def close(self) -> None:
        if self._closed:
            return
        self.view.close()
        self._body.clear()
        self._closed = True

    def _env_ids(self, value: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Newton rigid object tensor port is closed")
        return normalize_env_ids(
            value,
            num_envs=self.num_envs,
            device=self.device,
            allow_empty=True,
        )

    def _state_masks_for(
        self, ids: torch.Tensor, *, field: str
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        mask = self._device_reset_mask
        if mask is not None:
            if ids.shape != (self.num_envs,):
                raise ValueError("device-masked reset must use one row per environment")
            torch._assert_async(
                torch.all(ids == self._all_env_ids),
                "device-masked reset env_ids must use canonical environment order",
            )
            return mask, mask
        selected = self._state_row_masks[field]
        selected.zero_()
        selected.index_fill_(0, ids, True)
        return None, selected


@dataclass(slots=True)
class NewtonSolverIntegrationTensorPort:
    """SolverMuJoCo ``TIME|ACT|WARMSTART`` 的 Kaleidoscope GPU state port。

    manager 持有实际 Warp buffer，本 port 只建立 DLPack Torch alias，并预分配 full-N staged
    payload 与 bool active mask。snapshot/restore/clone 和 episode reset 都只提交 GPU kernel；
    不读取 selector 内容，也不构造变长 host world-id 列表。
    """

    runtime: object
    device: torch.device
    field_name: str = field(init=False, default="solver.persistent")
    num_envs: int = field(init=False)
    tensor: torch.Tensor = field(init=False, repr=False)
    _staged: torch.Tensor = field(init=False, repr=False)
    _active: torch.Tensor = field(init=False, repr=False)
    _staged_warp: object = field(init=False, repr=False)
    _active_warp: object = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        import warp as wp

        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise ValueError("Newton solver integration port requires CUDA")
        if str(getattr(self.runtime, "kind", "")) != "newton_cuda":
            raise TypeError("solver integration port requires NewtonRuntime")
        self.num_envs = int(getattr(self.runtime, "world_count", 0))
        if self.num_envs < 1:
            raise ValueError(
                "Newton solver integration port requires at least one world"
            )
        self.tensor = _warp_alias(
            self.runtime.borrow_solver_integration_state(),
            device=self.device,
            name="Newton solver integration state",
        )
        width = int(getattr(self.runtime, "solver_integration_state_width", 0))
        if self.tensor.ndim != 2 or self.tensor.shape != (self.num_envs, width):
            raise RuntimeError(
                "Newton solver integration buffer shape differs from manager metadata"
            )
        if width < 1 or not self.tensor.is_contiguous():
            raise RuntimeError(
                "Newton solver integration buffer must be a non-empty contiguous matrix"
            )
        self._staged = torch.empty_like(self.tensor)
        self._active = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        # ``wp.from_torch`` 建立零拷贝 device alias；两份 storage 都在构造期固定，热路径
        # 不会为了 partial RPC/reset 重分配 Warp array 或做 H2D。
        self._staged_warp = wp.from_torch(self._staged, dtype=wp.float32)
        self._active_warp = wp.from_torch(self._active, dtype=wp.bool)

    def write(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """把 K 行 state payload 恢复到 selected solver worlds。"""

        ids = self._env_ids(env_ids)
        data = require_cuda_tensor(
            values,
            name="Newton solver integration payload",
            ndim=2,
            leading_dim=ids.numel(),
            dtype=torch.float32,
        )
        if data.device != self.device or data.shape[1:] != self.tensor.shape[1:]:
            raise ValueError(
                "Newton solver integration payload has the wrong device or width"
            )
        assert_finite_async(data, name="Newton solver integration payload")
        with _owner_stream_scope(self.runtime, device=self.device):
            # set_state 采用 full-N input + active mask。先复制 live engine state，确保
            # 未选 rows 的 staging 始终有效，再只覆盖 RPC payload 对应的 rows。
            self._staged.copy_(self.tensor)
            self._staged.index_copy_(0, ids, data.contiguous())
            self._active.zero_()
            self._active.index_fill_(0, ids, True)
            self.runtime.set_solver_integration_state(
                self._staged_warp,
                active_world_mask=self._active_warp,
            )

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> None:
        """恢复显式 K rows；SAME_STEP 可用等宽 device mask 只选择 done rows。"""

        ids = self._env_ids(env_ids)
        mask: torch.Tensor | None = None
        if reset_mask is not None:
            mask = require_cuda_tensor(
                reset_mask,
                name="Newton solver reset mask",
                ndim=1,
                leading_dim=ids.numel(),
                dtype=torch.bool,
            )
            if mask.device != self.device:
                raise ValueError("Newton solver reset mask must share the port device")
        with _owner_stream_scope(self.runtime, device=self.device):
            self._active.zero_()
            if mask is None:
                self._active.index_fill_(0, ids, True)
            else:
                self._active.index_copy_(0, ids, mask)
            self.runtime.reset_solver_integration_state(self._active_warp)

    def close(self) -> None:
        """断开 staging；manager canonical buffer 仍由 Session 负责最终销毁。"""

        if self._closed:
            return
        self._staged_warp = None
        self._active_warp = None
        self._staged = torch.empty((0, 0), device=self.device)
        self._active = torch.empty(0, device=self.device, dtype=torch.bool)
        self.runtime = None
        self._closed = True

    def _env_ids(self, value: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Newton solver integration port is closed")
        return normalize_env_ids(
            value,
            num_envs=self.num_envs,
            device=self.device,
            allow_empty=True,
        )


def _rigid_values(
    value: torch.Tensor,
    *,
    rows: int,
    width: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    result = require_cuda_tensor(
        value,
        name=name,
        ndim=2,
        leading_dim=rows,
        dtype=torch.float32,
    )
    if result.device != device or result.shape[1:] != (width,):
        raise ValueError(f"{name} must have shape ({rows},{width}) on {device}")
    assert_finite_async(result, name=name)
    return result.contiguous()


__all__ = [
    "NewtonArticulationTensorPort",
    "NewtonRigidObjectTensorPort",
    "NewtonSolverIntegrationTensorPort",
]
