"""The O6 runtime and planning assets must share the 1014 installation contract."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from linkerbot_sim.mirror.collision.urdf_kinematics import _UrdfKinematics

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "packages/linker-robot-assets/src/linker_robot_assets/assets"


@pytest.mark.parametrize("side", ("l", "r"))
@pytest.mark.parametrize("camera", ("", "_gemini335l"))
def test_o6_profiles_use_matching_runtime_and_planning_units(side, camera):
    name = f"ar5_08_o6{camera}_{side}"
    profile = yaml.safe_load((ROOT / f"configs/robots/{name}.yaml").read_text())
    robot = profile["robot"]
    assert robot["asset_type"] == "mjcf"
    mjcf = ROOT / robot["asset_path"]
    urdf = ROOT / robot["planning_collision"]["mounted_urdf"]
    assert mjcf.parent == urdf.parent == ASSETS / "units" / name
    tree = ET.parse(urdf).getroot()
    prefix = f"hand_{side}h_"
    masses = [
        float(link.find("inertial/mass").get("value"))
        for link in tree.findall("link")
        if link.get("name", "").startswith(prefix)
    ]
    assert len(masses) == 12
    assert sum(masses) == pytest.approx(0.36998, abs=1e-12)
    fk = _UrdfKinematics(urdf).link_transforms({})
    palm = fk[f"{prefix}hand_base_link"]
    seat_in_palm = np.linalg.inv(palm) @ fk["flange_hand_mount"]
    np.testing.assert_allclose(seat_in_palm[:3, 3], [0, 0, 0.002], atol=1e-10)
    side_tag = side.upper()
    tool = fk[f"arm_AR5_5_08{side_tag}_W4C4A6_tcp"]
    np.testing.assert_allclose(
        (np.linalg.inv(tool) @ palm)[:3, 3], [0, 0, 0.0205], atol=1e-10
    )
    xml = ET.parse(mjcf).getroot()
    assert len(xml.findall("equality/joint")) == 5
    for equality in xml.findall("equality/joint"):
        name = equality.get("joint1")
        mimic = tree.find(f"joint[@name='{name}']/mimic")
        assert equality.get("joint2") == mimic.get("joint")
        ratio = float(equality.get("polycoef").split()[1])
        assert ratio == float(mimic.get("multiplier"))
        assert equality.get("solref") == "-10000000 -63000"
    assert profile["joint_groups"]["hand"] == [
        prefix + joint
        for joint in (
            "thumb_cmc_pitch",
            "thumb_cmc_yaw",
            "index_mcp_pitch",
            "middle_mcp_pitch",
            "ring_mcp_pitch",
            "pinky_mcp_pitch",
        )
    ]
    # Composed models keep the existing external torque-control interface.
    assert not xml.findall("actuator/position")
    assert len(xml.findall("actuator/motor")) == 13
