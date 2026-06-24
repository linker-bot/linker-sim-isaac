from __future__ import annotations

import numpy as np

from manipulation_project.assets.asset_paths import DEFAULT_AR5_L6_MJCF, DEFAULT_AR5_URDF
from manipulation_project.backends.cumotion.tcp_urdf_builder import write_tcp_urdf
from manipulation_project.tcp.pinch_tcp import infer_hand_body_names, fingertip_pinch_local_offset, make_pinch_tcp
from manipulation_project.tcp.tcp_frame import TcpFrame


def test_pinch_center_is_between_thumb_and_index() -> None:
    targets = {
        "L6V1_L_hand_thumb_cmc_roll": 0.95,
        "L6V1_L_hand_thumb_cmc_pitch": 0.7,
        "L6V1_L_hand_index_mcp_pitch": 0.85,
    }
    center, thumb, index = fingertip_pinch_local_offset(DEFAULT_AR5_L6_MJCF, targets)
    np.testing.assert_allclose(center, 0.5 * (thumb + index))
    assert center.shape == (3,)


def test_hand_body_names_are_inferred_from_joint_prefix() -> None:
    assert infer_hand_body_names({"DexHandV2_R_hand_index_mcp_pitch": 0.1}) == (
        "DexHandV2_R_hand_base_link",
        "DexHandV2_R_hand_thumb_tip",
        "DexHandV2_R_hand_index_tip",
    )


def test_make_pinch_tcp() -> None:
    tcp = make_pinch_tcp(
        DEFAULT_AR5_L6_MJCF,
        {"L6V1_L_hand_thumb_cmc_roll": 0.95, "L6V1_L_hand_thumb_cmc_pitch": 0.7, "L6V1_L_hand_index_mcp_pitch": 0.85},
        parent_frame="AR5V2_L_arm_flan_link",
    )
    assert tcp.frame_name == "pinch_tcp"
    assert tcp.xyz.shape == (3,)


def test_write_tcp_urdf(tmp_path) -> None:
    output = tmp_path / "with_tcp.urdf"
    tcp = TcpFrame.from_xyz_rpy(
        "unit_test_tcp",
        "AR5V2_L_arm_flan_link",
        xyz=[0.0, 0.0, 0.13],
    )
    write_tcp_urdf(DEFAULT_AR5_URDF, output, tcp)
    text = output.read_text(encoding="utf-8")
    assert 'name="unit_test_tcp"' in text
    assert 'link="AR5V2_L_arm_flan_link"' in text
