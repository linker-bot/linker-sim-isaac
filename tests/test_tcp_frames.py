from __future__ import annotations

from linkerbot_sim.assets.asset_paths import DEFAULT_AR5_URDF
from linkerbot_sim.backends.cumotion.tcp_frame import TcpFrame
from linkerbot_sim.backends.cumotion.tcp_urdf_builder import write_tcp_urdf


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
