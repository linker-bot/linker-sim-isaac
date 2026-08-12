from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.physx_task_space import (
    PhysxTaskSpaceMetadataError,
    PhysxTaskSpacePort,
    PhysxTaskSpaceSensorError,
    quaternion_matrix_wxyz,
    rpy_matrix,
    skew,
)
from linkerbot_sim.robots.tcp_binding import PhysicalTcpBinding


ARM_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
DOF_NAMES = ("finger", "j2", "j1", "j3", "j4", "j5", "j6")
ARM_COLUMNS = (2, 1, 3, 4, 5, 6)


def _stage(*, duplicate_incoming: bool = False):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = "/World/Robot"
    UsdGeom.Xform.Define(stage, root)
    base = UsdGeom.Xform.Define(stage, f"{root}/base").GetPrim()
    flange = UsdGeom.Xform.Define(stage, f"{root}/flange").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.RigidBodyAPI.Apply(flange)
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joint6")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(base.GetPath()))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(flange.GetPath()))])
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.2, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if duplicate_incoming:
        duplicate = UsdPhysics.FixedJoint.Define(stage, f"{root}/duplicate")
        duplicate.CreateBody0Rel().SetTargets([Sdf.Path(str(base.GetPath()))])
        duplicate.CreateBody1Rel().SetTargets([Sdf.Path(str(flange.GetPath()))])
    return stage, root


@dataclass
class _FakeArticulation:
    prim_path: str
    device: str = "cpu"
    body_names = ("base", "flange")
    dof_names = DOF_NAMES
    fixed_base = True
    link_indices = {"base": 0, "flange": 1}

    def __post_init__(self) -> None:
        half = math.sqrt(0.5)
        self.positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        self.orientations = np.asarray([[1.0, 0.0, 0.0, 0.0], [half, 0.0, 0.0, half]])
        self.velocities = np.asarray([[0.0] * 6, [1.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        jacobian = np.zeros((1, 6, len(DOF_NAMES)), dtype=float)
        jacobian[0, :, ARM_COLUMNS] = np.eye(6)
        self.jacobians = jacobian[None, ...]
        self.reactions = np.asarray([[[0.0] * 6, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]])
        self.q = np.arange(len(DOF_NAMES), dtype=float) + 10.0
        self.qd = np.arange(len(DOF_NAMES), dtype=float) + 20.0

    def is_fixed_base(self) -> bool:
        return True

    def get_link_world_poses(self):
        return self.positions, self.orientations

    def get_link_velocities(self):
        return self.velocities

    def get_jacobians(self):
        return self.jacobians

    def get_measured_joint_forces(self):
        return self.reactions

    def get_joint_positions(self):
        return self.q

    def get_joint_velocities(self):
        return self.qd


def _binding(root: str) -> PhysicalTcpBinding:
    return PhysicalTcpBinding(
        tcp_frame_name="tcp",
        parent_frame_name="flange",
        parent_body_path=f"{root}/flange",
        offset_xyz=(1.0, 0.0, 0.0),
        offset_rpy=(0.0, 0.0, math.pi / 2.0),
    )


def test_metadata_binding_and_nonzero_tcp_transform_are_consistent() -> None:
    stage, root = _stage()
    articulation = _FakeArticulation(root)
    port = PhysxTaskSpacePort(
        articulation,
        stage,
        ARM_NAMES,
        _binding(root),
    )

    observation = port.observe()

    assert port.binding.parent_body_name == "flange"
    assert port.binding.incoming_joint_name == "joint6"
    assert port.binding.jacobian_body_row == 0
    assert port.binding.reaction_row == 1
    assert port.binding.arm_column_indices == ARM_COLUMNS
    np.testing.assert_allclose(observation.position, [1.0, 3.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(
        quaternion_matrix_wxyz(observation.orientation_wxyz),
        rpy_matrix((0.0, 0.0, math.pi)),
        atol=1e-12,
    )
    np.testing.assert_allclose(observation.twist, [-1, 0, 0, 0, 0, 2], atol=1e-12)
    offset_world = np.asarray([0.0, 1.0, 0.0])
    expected_jacobian = np.eye(6)
    expected_jacobian[:3] -= skew(offset_world) @ expected_jacobian[3:]
    np.testing.assert_allclose(observation.jacobian, expected_jacobian, atol=1e-12)
    np.testing.assert_allclose(
        observation.joint_positions, articulation.q[list(ARM_COLUMNS)]
    )
    np.testing.assert_allclose(
        observation.joint_velocities, articulation.qd[list(ARM_COLUMNS)]
    )
    # Raw PhysX contract is environment-on-tool. Moving the moment from the
    # incoming joint at y=2.2 to TCP y=3.0 subtracts r x F.
    np.testing.assert_allclose(
        observation.external_wrench_environment_on_tool,
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        atol=1e-12,
    )
    assert observation.sequence == 0
    assert port.observe().sequence == 1


def test_port_rejects_cuda_ambiguous_joint_and_shape_changes() -> None:
    stage, root = _stage()
    with pytest.raises(PhysxTaskSpaceMetadataError, match="CPU"):
        PhysxTaskSpacePort(
            _FakeArticulation(root, device="cuda:0"),
            stage,
            ARM_NAMES,
            _binding(root),
        )

    duplicate_stage, duplicate_root = _stage(duplicate_incoming=True)
    with pytest.raises(PhysxTaskSpaceMetadataError, match="one incoming"):
        PhysxTaskSpacePort(
            _FakeArticulation(duplicate_root),
            duplicate_stage,
            ARM_NAMES,
            _binding(duplicate_root),
        )

    articulation = _FakeArticulation(root)
    port = PhysxTaskSpacePort(
        articulation,
        stage,
        ARM_NAMES,
        _binding(root),
    )
    articulation.jacobians = np.zeros((1, 2, 6, len(DOF_NAMES)))
    with pytest.raises(PhysxTaskSpaceSensorError, match="shape changed"):
        port.observe()


def test_port_rejects_unbound_arm_name_without_array_offset_guessing() -> None:
    stage, root = _stage()

    with pytest.raises(PhysxTaskSpaceMetadataError, match="must match exactly"):
        PhysxTaskSpacePort(
            _FakeArticulation(root),
            stage,
            (*ARM_NAMES[:-1], "missing"),
            _binding(root),
        )


def test_port_rejects_link_index_metadata_that_disagrees_with_body_rows() -> None:
    stage, root = _stage()
    articulation = _FakeArticulation(root)
    articulation.link_indices = {"base": 1, "flange": 0}

    with pytest.raises(PhysxTaskSpaceMetadataError, match="body-name rows"):
        PhysxTaskSpacePort(
            articulation,
            stage,
            ARM_NAMES,
            _binding(root),
        )
