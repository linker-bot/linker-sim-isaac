import numpy as np
from pxr import Gf, Usd, UsdGeom

from linkerbot_sim.isaac.scene.pose import (
    apply_prim_local_pose_and_zero_velocity,
    read_prim_world_pose,
)
from linkerbot_sim.utils.rotations import quat_wxyz_to_matrix, rpy_xyz_to_quat_wxyz


def test_rotated_static_camera_pose_roundtrip_preserves_child_world_transform():
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/CameraStand")
    child = UsdGeom.Xform.Define(stage, "/World/CameraStand/Optical")
    child.AddTranslateOp().Set(Gf.Vec3d(0.04, -0.08, 0.02))
    parent.AddTranslateOp().Set(Gf.Vec3d(0.3, -0.2, 0.1))
    parent.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, -90))
    original_ops = [op.GetOpName() for op in parent.GetOrderedXformOps()]
    q = rpy_xyz_to_quat_wxyz([-np.pi / 2, 0, -np.pi / 2])
    assert apply_prim_local_pose_and_zero_velocity(
        stage, str(parent.GetPath()), np.array([0.3, -0.2, 0.1]), q
    )
    first = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(child.GetPrim()))
    position, actual_q = read_prim_world_pose(stage, str(parent.GetPath()))
    np.testing.assert_allclose(
        quat_wxyz_to_matrix(actual_q), quat_wxyz_to_matrix(q), atol=1e-12
    )
    apply_prim_local_pose_and_zero_velocity(
        stage, str(parent.GetPath()), position, actual_q
    )
    np.testing.assert_allclose(
        np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(child.GetPrim())),
        first,
        atol=1e-12,
    )

    assert [op.GetOpName() for op in parent.GetOrderedXformOps()] == original_ops
