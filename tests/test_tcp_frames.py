from __future__ import annotations

import numpy as np

from linkerbot_sim.backends.curobo.config import CuroboTcpFrame
from linkerbot_sim.backends.curobo.profile_merge import robot_curobo_config
from linkerbot_sim.backends.curobo.robot_model import (
    write_curobo_tcp_urdf_with_frames,
)
from linkerbot_sim.configs.profiles import load_profile_yaml


def test_write_tcp_urdf(tmp_path) -> None:
    urdf_path = robot_curobo_config(
        load_profile_yaml("robot", "ar5v2_l")
    ).robot.urdf_path
    assert urdf_path is not None
    output = tmp_path / "with_tcp.urdf"
    tcp = CuroboTcpFrame(
        frame_name="unit_test_tcp",
        parent_frame="AR5V2_L_arm_flan_link",
        xyz=np.asarray([0.0, 0.0, 0.13], dtype=float),
        rpy=np.zeros(3, dtype=float),
    )
    write_curobo_tcp_urdf_with_frames(urdf_path, output, (tcp,))
    text = output.read_text(encoding="utf-8")
    assert 'name="unit_test_tcp"' in text
    assert 'link="AR5V2_L_arm_flan_link"' in text
