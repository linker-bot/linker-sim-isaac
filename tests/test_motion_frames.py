from __future__ import annotations

import numpy as np

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.planning.frames import FrameTransformer
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


def test_world_pose_position_and_orientation_use_same_base_transform() -> None:
    transformer = FrameTransformer.from_root_pose(
        RootPoseConfig(
            xyz=(1.0, 2.0, 0.0),
            rpy=(0.0, 0.0, np.pi / 2.0),
        )
    )
    target = transformer.pose_to_robot_base(
        position=np.asarray([1.0, 3.0, 0.0]),
        orientation_wxyz=rpy_xyz_to_quat_wxyz((0.0, 0.0, np.pi / 2.0)),
        reference_frame="world",
    )
    np.testing.assert_allclose(target.position, [1.0, 0.0, 0.0], atol=1.0e-8)
    np.testing.assert_allclose(
        target.orientation_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1.0e-8
    )


def test_tcp_and_world_offsets_rotate_into_robot_base() -> None:
    transformer = FrameTransformer.from_root_pose(
        RootPoseConfig(rpy=(0.0, 0.0, np.pi / 2.0)),
        tcp_position_in_base=np.asarray([0.0, 0.0, 1.0]),
        tcp_orientation_wxyz_in_base=rpy_xyz_to_quat_wxyz((0.0, 0.0, np.pi / 2.0)),
    )
    np.testing.assert_allclose(
        transformer.offset_to_robot_base(
            np.asarray([1.0, 0.0, 0.0]), offset_frame="world"
        ),
        [0.0, -1.0, 0.0],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        transformer.offset_to_robot_base(
            np.asarray([1.0, 0.0, 0.0]), offset_frame="tcp"
        ),
        [0.0, 1.0, 0.0],
        atol=1.0e-8,
    )
