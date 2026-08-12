from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from linkerbot_sim.controllers.runtime_projection import CommandRuntimeProjection
from linkerbot_sim.kaleidoscope.physx_ports import (
    IsaacArticulationTensorPort,
    IsaacRigidObjectTensorPort,
    as_torch_cuda,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for PhysX ports"
)


class _ArticulationView:
    def __init__(self) -> None:
        self.dof_names = ["j0", "j1", "j2", "j3"]
        self.q = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
        self.qd = torch.zeros_like(self.q)
        self.target = self.q + 20.0
        self.velocity_target = -self.q - 3.0
        self.effort_target = self.q + 40.0
        self.stiffness = torch.zeros_like(self.q)
        self.damping = torch.zeros_like(self.q)
        self.max_effort = torch.zeros_like(self.q)
        self.events: list[tuple[str, object]] = []

    def get_applied_actions(self, *, clone):
        assert clone is False
        return SimpleNamespace(
            joint_positions=self.target,
            joint_velocities=self.velocity_target,
            joint_efforts=self.effort_target,
        )

    def get_joint_positions(self, *, indices, joint_indices, clone):
        assert clone is False
        return self.q.index_select(0, indices).index_select(1, joint_indices)

    def get_joint_velocities(self, *, indices, joint_indices, clone):
        assert clone is False
        return self.qd.index_select(0, indices).index_select(1, joint_indices)

    def set_joint_positions(self, values, *, indices, joint_indices):
        selected = self.q.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.q.index_copy_(0, indices, selected)
        # Isaac 的高层 state setter 会用完整 state 行覆盖完整 drive target 行。
        self.target.index_copy_(0, indices, self.q.index_select(0, indices))
        self.effort_target.index_fill_(0, indices, 0.0)

    def set_joint_velocities(self, values, *, indices, joint_indices):
        selected = self.qd.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.qd.index_copy_(0, indices, selected)
        self.velocity_target.index_copy_(0, indices, self.qd.index_select(0, indices))
        self.effort_target.index_fill_(0, indices, 0.0)

    def set_joint_position_targets(self, values, *, indices, joint_indices):
        selected = self.target.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.target.index_copy_(0, indices, selected)

    def set_joint_velocity_targets(self, values, *, indices, joint_indices):
        selected = self.velocity_target.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.velocity_target.index_copy_(0, indices, selected)

    def set_joint_efforts(self, values, *, indices, joint_indices):
        selected = self.effort_target.index_select(0, indices)
        selected.index_copy_(1, joint_indices, values)
        self.effort_target.index_copy_(0, indices, selected)

    def switch_dof_control_mode(self, mode, *, dof_index, indices):
        self.events.append(("switch", (mode, dof_index, indices.clone())))

    def set_dof_gains(self, *, stiffnesses, dampings, indices, dof_indices):
        self.events.append(("gains", dof_indices.clone()))
        selected_kp = self.stiffness.index_select(0, indices)
        selected_kd = self.damping.index_select(0, indices)
        selected_kp.index_copy_(1, dof_indices, stiffnesses)
        selected_kd.index_copy_(1, dof_indices, dampings)
        self.stiffness.index_copy_(0, indices, selected_kp)
        self.damping.index_copy_(0, indices, selected_kd)

    def set_dof_max_efforts(self, values, *, indices, dof_indices):
        self.events.append(("limits", dof_indices.clone()))
        selected = self.max_effort.index_select(0, indices)
        selected.index_copy_(1, dof_indices, values)
        self.max_effort.index_copy_(0, indices, selected)

    def set_dof_drive_types(self, drive_type, *, dof_indices):
        self.events.append(("drive", (drive_type, dof_indices.clone())))


class _RigidView:
    def __init__(self) -> None:
        self.position = torch.zeros((3, 3), device="cuda")
        self.orientation_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, device="cuda")
        self.velocity = torch.zeros((3, 6), device="cuda")

    def get_world_poses(self, *, indices, clone):
        assert clone is False
        return self.position.index_select(
            0, indices
        ), self.orientation_xyzw.index_select(0, indices)

    def get_velocities(self, *, indices, clone):
        assert clone is False
        return self.velocity.index_select(0, indices)

    def set_world_poses(self, *, positions, orientations, indices):
        self.position.index_copy_(0, indices, positions)
        self.orientation_xyzw.index_copy_(0, indices, orientations)

    def set_velocities(self, values, *, indices):
        self.velocity.index_copy_(0, indices, values)


def test_articulation_port_uses_gpu_indices_and_command_columns() -> None:
    articulation = _ArticulationView()
    tcp = _RigidView()
    port = IsaacArticulationTensorPort(
        label="arm",
        view=articulation,
        tcp_view=tcp,
        command_joint_indices=torch.tensor([1, 3], device="cuda"),
        device=torch.device("cuda:0"),
        orientation_order="xyzw",
    )
    ids = torch.tensor([0, 2], device="cuda")
    q = port.read_joint_positions(ids)
    torch.testing.assert_close(
        q, torch.tensor([[1.0, 3.0], [9.0, 11.0]], device="cuda")
    )
    _position, orientation = port.read_tcp_pose_wxyz(ids)
    torch.testing.assert_close(
        orientation, torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device="cuda")
    )
    port.write_joint_targets(ids, torch.ones((2, 2), device="cuda"))
    torch.testing.assert_close(
        articulation.target[ids][:, [1, 3]], torch.ones((2, 2), device="cuda")
    )
    full = port.read_all_joint_positions(ids)
    torch.testing.assert_close(full, articulation.q.index_select(0, ids))
    original_position_targets = articulation.target.clone()
    original_velocity_targets = articulation.velocity_target.clone()
    original_effort_targets = articulation.effort_target.clone()

    port.write_joint_positions(ids, torch.full((2, 2), 6.0, device="cuda"))
    torch.testing.assert_close(
        articulation.q.index_select(0, ids).index_select(1, port.command_joint_indices),
        torch.full((2, 2), 6.0, device="cuda"),
    )
    torch.testing.assert_close(articulation.target, original_position_targets)
    torch.testing.assert_close(articulation.velocity_target, original_velocity_targets)
    torch.testing.assert_close(articulation.effort_target, original_effort_targets)

    replacement = torch.full((2, 4), 7.0, device="cuda")
    port.write_all_joint_positions(ids, replacement)
    torch.testing.assert_close(articulation.q.index_select(0, ids), replacement)
    torch.testing.assert_close(articulation.target, original_position_targets)
    torch.testing.assert_close(articulation.velocity_target, original_velocity_targets)
    torch.testing.assert_close(articulation.effort_target, original_effort_targets)

    port.write_joint_velocities(ids, torch.full((2, 2), 4.0, device="cuda"))
    torch.testing.assert_close(
        articulation.qd.index_select(0, ids).index_select(
            1, port.command_joint_indices
        ),
        torch.full((2, 2), 4.0, device="cuda"),
    )
    torch.testing.assert_close(articulation.target, original_position_targets)
    torch.testing.assert_close(articulation.velocity_target, original_velocity_targets)
    torch.testing.assert_close(articulation.effort_target, original_effort_targets)

    velocity_replacement = torch.full((2, 4), 5.0, device="cuda")
    port.write_all_joint_velocities(ids, velocity_replacement)
    torch.testing.assert_close(
        articulation.qd.index_select(0, ids), velocity_replacement
    )
    torch.testing.assert_close(articulation.target, original_position_targets)
    torch.testing.assert_close(articulation.velocity_target, original_velocity_targets)
    torch.testing.assert_close(articulation.effort_target, original_effort_targets)


def _projection(
    mode: str,
    method: str,
    *,
    stiffness: tuple[float, float],
    damping: tuple[float, float],
    limits: tuple[float, float],
) -> CommandRuntimeProjection:
    physical = mode if method == "implicit" else "effort"
    zeros = np.zeros(2, dtype=float)
    return CommandRuntimeProjection(
        joint_names=("j1", "j3"),
        components=("arm", "arm"),
        modes=(mode, mode),
        methods=(method, method),
        physical_modes=(physical, physical),
        stiffness=np.asarray(stiffness, dtype=float),
        damping=np.asarray(damping, dtype=float),
        effort_limits=np.asarray(limits, dtype=float),
        drive_stiffness=(
            np.asarray(stiffness, dtype=float) if method == "implicit" else zeros
        ),
        drive_damping=(
            np.asarray(damping, dtype=float) if method == "implicit" else zeros
        ),
    )


def _control_port() -> tuple[IsaacArticulationTensorPort, _ArticulationView]:
    articulation = _ArticulationView()
    return (
        IsaacArticulationTensorPort(
            label="arm",
            view=articulation,
            tcp_view=_RigidView(),
            command_joint_indices=torch.tensor([1, 3], device="cuda"),
            command_joint_names=("j1", "j3"),
            command_joint_indices_host=(1, 3),
            device=torch.device("cuda:0"),
            orientation_order="xyzw",
        ),
        articulation,
    )


def test_physx_explicit_pd_d_and_direct_effort_are_clipped_to_command_dofs() -> None:
    port, articulation = _control_port()
    ids = torch.tensor([0, 2], device="cuda")
    articulation.qd[0] = torch.tensor([9.0, 0.5, 8.0, -0.5], device="cuda")
    articulation.qd[2] = torch.tensor([7.0, 1.0, 6.0, -1.0], device="cuda")
    untouched = articulation.effort_target[:, [0, 2]].clone()

    position = port.prepare_control_runtime(
        _projection(
            "position",
            "explicit",
            stiffness=(2.0, 3.0),
            damping=(4.0, 5.0),
            limits=(6.0, 7.0),
        )
    )
    articulation.events.clear()
    port.apply_prepared_control_runtime(position)
    assert [event[0] for event in articulation.events] == [
        "switch",
        "switch",
        "gains",
        "limits",
        "drive",
    ]
    desired_qd = torch.tensor([[1.5, -1.5], [2.0, -2.0]], device="cuda")
    desired_q = torch.tensor([[5.0, -2.0], [8.0, 3.0]], device="cuda")
    port.write_joint_velocity_targets(ids, desired_qd)
    port.write_joint_position_targets(ids, desired_q)
    actual_q = articulation.q.index_select(0, ids)[:, [1, 3]]
    actual_qd = articulation.qd.index_select(0, ids)[:, [1, 3]]
    expected_pd = torch.clamp(
        torch.tensor([2.0, 3.0], device="cuda") * (desired_q - actual_q)
        + torch.tensor([4.0, 5.0], device="cuda") * (desired_qd - actual_qd),
        min=-torch.tensor([6.0, 7.0], device="cuda"),
        max=torch.tensor([6.0, 7.0], device="cuda"),
    )
    torch.testing.assert_close(
        articulation.effort_target.index_select(0, ids)[:, [1, 3]], expected_pd
    )

    velocity = port.prepare_control_runtime(
        _projection(
            "velocity",
            "explicit",
            stiffness=(0.0, 0.0),
            damping=(2.0, 4.0),
            limits=(3.0, 5.0),
        )
    )
    port.apply_prepared_control_runtime(velocity)
    desired_velocity = torch.tensor([[4.0, -4.0], [5.0, -5.0]], device="cuda")
    port.write_joint_velocity_targets(ids, desired_velocity)
    expected_d = torch.clamp(
        torch.tensor([2.0, 4.0], device="cuda") * (desired_velocity - actual_qd),
        min=-torch.tensor([3.0, 5.0], device="cuda"),
        max=torch.tensor([3.0, 5.0], device="cuda"),
    )
    torch.testing.assert_close(
        articulation.effort_target.index_select(0, ids)[:, [1, 3]], expected_d
    )

    effort = port.prepare_control_runtime(
        _projection(
            "effort",
            "direct",
            stiffness=(0.0, 0.0),
            damping=(0.0, 0.0),
            limits=(1.5, 2.5),
        )
    )
    port.apply_prepared_control_runtime(effort)
    port.write_joint_effort_targets(
        ids,
        torch.tensor([[8.0, -8.0], [-9.0, 9.0]], device="cuda"),
    )
    torch.testing.assert_close(
        articulation.effort_target.index_select(0, ids)[:, [1, 3]],
        torch.tensor([[1.5, -2.5], [-1.5, 2.5]], device="cuda"),
    )
    torch.testing.assert_close(articulation.effort_target[:, [0, 2]], untouched)


def test_rigid_port_converts_wxyz_and_keeps_pose_velocity_on_cuda() -> None:
    view = _RigidView()
    port = IsaacRigidObjectTensorPort(
        label="block",
        view=view,
        device=torch.device("cuda:0"),
        orientation_order="xyzw",
    )
    ids = torch.tensor([1], device="cuda")
    port.write_pose_wxyz(
        ids,
        torch.tensor([[2.0, 3.0, 4.0]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
    )
    port.write_velocity(ids, torch.ones((1, 6), device="cuda"))
    position, orientation = port.read_pose_wxyz(ids)
    torch.testing.assert_close(position, torch.tensor([[2.0, 3.0, 4.0]], device="cuda"))
    torch.testing.assert_close(
        orientation, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    )
    assert as_torch_cuda(view.velocity, device=torch.device("cuda:0")) is view.velocity


def test_articulation_port_applies_custom_tcp_fixed_offset_on_gpu() -> None:
    articulation = _ArticulationView()
    tcp = _RigidView()
    half = 2.0**-0.5
    tcp.orientation_xyzw[:] = torch.tensor(
        [0.0, 0.0, half, half],
        device="cuda",
    )
    port = IsaacArticulationTensorPort(
        label="arm",
        view=articulation,
        tcp_view=tcp,
        command_joint_indices=torch.tensor([0, 1], device="cuda"),
        device=torch.device("cuda:0"),
        orientation_order="xyzw",
        tcp_offset_xyz=(1.0, 0.0, 0.0),
        tcp_offset_rpy=(0.0, 0.0, math.pi / 2.0),
    )
    position, orientation = port.read_tcp_pose_wxyz(torch.tensor([0], device="cuda"))
    torch.testing.assert_close(
        position,
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda"),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        orientation,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda"),
        atol=1.0e-6,
        rtol=0.0,
    )
