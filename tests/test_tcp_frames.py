from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from linkerbot_sim.assets.asset_paths import (
    DEFAULT_AR5_L6_MJCF,
    DEFAULT_AR5_URDF,
)
from linkerbot_sim.backends.cumotion.tcp_frame import TcpFrame
from linkerbot_sim.backends.cumotion.tcp_urdf_builder import write_tcp_urdf
from linkerbot_sim.utils.math_utils import make_transform, quat_wxyz_to_matrix
from scripts.pinch_grasp import (
    find_mjcf_body,
    infer_hand_body_names,
    fingertip_pinch_local_offset,
    make_pinch_tcp_transform,
    parse_mjcf_quat_wxyz,
    parse_mjcf_vec3,
)


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


def test_make_pinch_tcp_transform() -> None:
    targets = {
        "L6V1_L_hand_thumb_cmc_roll": 0.95,
        "L6V1_L_hand_thumb_cmc_pitch": 0.7,
        "L6V1_L_hand_index_mcp_pitch": 0.85,
    }
    tcp = make_pinch_tcp_transform(
        DEFAULT_AR5_L6_MJCF,
        targets,
    )
    hand_center, _thumb, _index = fingertip_pinch_local_offset(
        DEFAULT_AR5_L6_MJCF,
        targets,
    )
    # The L6 hand base is mounted to the AR5 flange with a fixed rotation; the
    # public TCP transform must be expressed relative to the flange, not the
    # hand base.
    root = ET.parse(DEFAULT_AR5_L6_MJCF).getroot()
    hand_base = find_mjcf_body(root, "L6V1_L_hand_base_link")
    flange_from_hand = make_transform(
        parse_mjcf_vec3(hand_base.get("pos")),
        quat_wxyz_to_matrix(parse_mjcf_quat_wxyz(hand_base.get("quat"))),
    )
    expected = (
        flange_from_hand
        @ np.asarray([*hand_center, 1.0], dtype=float).reshape(4, 1)
    )[:3, 0]
    assert tcp.frame_name == "pinch_tcp"
    assert tcp.xyz.shape == (3,)
    np.testing.assert_allclose(tcp.xyz, expected)


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
