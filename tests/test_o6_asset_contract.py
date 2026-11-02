"""Pinned assets must agree with runtime drive and planning contracts."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from linkerbot_sim.mirror.collision.urdf_kinematics import _UrdfKinematics

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "packages/linker-robot-assets/src/linker_robot_assets/assets"
CONTROLLERS = ROOT / "configs/controllers/newton_compliant"
DRIVE = yaml.safe_load((CONTROLLERS / "newton_drive.yaml").read_text())


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
        np.testing.assert_allclose(
            np.fromstring(equality.get("solref"), sep=" "), DRIVE["equality_solref"]
        )
        np.testing.assert_allclose(
            np.fromstring(equality.get("solimp"), sep=" "), DRIVE["equality_solimp"]
        )
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

    # Asset torque caps and passive dynamics must agree with the selected drive;
    # virtual advancing limits are intentionally a different controller layer.
    for group in ("arm", "hand"):
        controller = yaml.safe_load(
            (CONTROLLERS / f"{group}_controller.yaml").read_text()
        )
        names = profile["joint_groups"][group]
        caps = np.broadcast_to(
            controller["position_control"]["active_joints"]["max_force"], len(names)
        )
        for name, cap in zip(names, caps, strict=True):
            motor = xml.find(f"actuator/motor[@joint='{name}']")
            np.testing.assert_allclose(
                np.fromstring(motor.get("forcerange"), sep=" "), [-cap, cap]
            )
            joint = xml.find(f".//worldbody//joint[@name='{name}']")
            assert float(joint.get("armature")) == DRIVE[f"{group}_armature"]
            if group == "arm":
                damping = DRIVE["arm_passive_damping"]
            else:
                kind = (
                    "thumb_yaw"
                    if name.endswith("thumb_cmc_yaw")
                    else "thumb_flexion"
                    if name.endswith("thumb_cmc_pitch")
                    else "finger_flexion"
                )
                damping = DRIVE[kind]["passive_damping"]
            assert float(joint.get("damping")) == damping
    for equality in xml.findall("equality/joint"):
        follower = xml.find(f".//worldbody//joint[@name='{equality.get('joint1')}']")
        assert float(follower.get("damping")) == DRIVE["follower_damping"]


@pytest.mark.parametrize(
    "profile_path",
    sorted((ROOT / "configs/robots").glob("ar5_08_*.yaml")),
    ids=lambda p: p.stem,
)
def test_ar5_runtime_and_planner_wrist_limits_agree(profile_path):
    profile = yaml.safe_load(profile_path.read_text())
    robot = profile["robot"]
    paths = {robot["asset_path"], profile["curobo"]["robot"]["urdf_path"]}
    mounted = robot.get("planning_collision", {}).get("mounted_urdf")
    if mounted:
        paths.add(mounted)
    for path in paths:
        root = ET.parse(ROOT / path).getroot()
        mjcf = root.tag == "mujoco"
        joints = root.findall(".//worldbody//joint" if mjcf else "joint")
        wrists = [
            j for j in joints if j.get("name", "").endswith(("_joint_6", "_joint_7"))
        ]
        assert len(wrists) == 2, path
        for joint in wrists:
            if mjcf:
                limits = np.fromstring(joint.get("range"), sep=" ")
            else:
                limit = joint.find("limit")
                limits = [float(limit.get("lower")), float(limit.get("upper"))]
            np.testing.assert_allclose(limits, [-np.pi / 3, np.pi / 3], atol=1e-12)
