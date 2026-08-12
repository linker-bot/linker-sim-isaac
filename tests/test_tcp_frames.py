from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from linkerbot_sim.backends.curobo.config import CuroboTcpFrame
from linkerbot_sim.backends.curobo.profile_merge import curobo_config_from_profiles
from linkerbot_sim.backends.curobo.robot_model import (
    write_curobo_tcp_urdf_with_frames,
)
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.robots.tcp_binding import (
    PhysicalTcpBinding,
    resolve_physical_tcp_binding,
)
from linkerbot_sim.utils.config import load_yaml


def load_robot_profile_by_name(name: str) -> RobotProfileSettings:
    path = Path("configs/robots") / f"{name}.yaml"
    return RobotProfileSettings.from_mapping(load_yaml(path), source=str(path))


def test_write_tcp_urdf(tmp_path) -> None:
    urdf_path = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l"),
        cuda_device=0,
    ).robot.urdf_path
    assert urdf_path is not None
    output = tmp_path / "with_tcp.urdf"
    tcp = CuroboTcpFrame(
        frame_name="unit_test_tcp",
        parent_frame="AR5V2_L_arm_flan_link",
        xyz=(0.0, 0.0, 0.13),
        rpy=(0.0, 0.0, 0.0),
    )
    write_curobo_tcp_urdf_with_frames(urdf_path, output, (tcp,))
    text = output.read_text(encoding="utf-8")
    assert 'name="unit_test_tcp"' in text
    assert 'link="AR5V2_L_arm_flan_link"' in text


def _stage_with_parent(*, duplicate: bool = False):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = "/World/Robots/left"
    UsdGeom.Xform.Define(stage, root)
    parent_name = "AR5V2_L_arm_flan_link"
    parent = UsdGeom.Xform.Define(stage, f"{root}/{parent_name}").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(parent)
    if duplicate:
        second = UsdGeom.Xform.Define(stage, f"{root}/nested/{parent_name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(second)
    return stage, root


def test_physical_tcp_binding_preserves_nonzero_fixed_transform() -> None:
    profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    robot = profile.curobo.robot
    assert robot is not None
    frame = replace(
        robot.custom_tcp_frames[0],
        xyz=(0.01, -0.02, 0.13),
        rpy=(0.2, -0.1, 0.3),
    )
    profile = replace(
        profile,
        curobo=replace(
            profile.curobo,
            robot=replace(robot, custom_tcp_frames=(frame,)),
        ),
    )
    stage, root = _stage_with_parent()

    binding = resolve_physical_tcp_binding(
        stage=stage,
        imported_root_path=root,
        profile=profile,
    )

    assert binding == PhysicalTcpBinding(
        tcp_frame_name="AR5V2_L_pinch_tcp",
        parent_frame_name="AR5V2_L_arm_flan_link",
        parent_body_path=f"{root}/AR5V2_L_arm_flan_link",
        offset_xyz=(0.01, -0.02, 0.13),
        offset_rpy=(0.2, -0.1, 0.3),
    )


@pytest.mark.parametrize("duplicate", [False, True])
def test_physical_tcp_binding_rejects_missing_or_duplicate_parent(
    duplicate: bool,
) -> None:
    profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    if duplicate:
        stage, root = _stage_with_parent(duplicate=True)
    else:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        root = "/World/Robots/left"
        UsdGeom.Xform.Define(stage, root)

    with pytest.raises(RuntimeError, match="exactly one rigid body"):
        resolve_physical_tcp_binding(
            stage=stage,
            imported_root_path=root,
            profile=profile,
        )


def test_physical_tcp_binding_rejects_disabled_model() -> None:
    profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    profile = replace(
        profile,
        curobo=replace(
            profile.curobo,
            binding=replace(profile.curobo.binding, enabled=False),
        ),
    )
    stage, root = _stage_with_parent()

    with pytest.raises(ValueError, match="does not enable"):
        resolve_physical_tcp_binding(
            stage=stage,
            imported_root_path=root,
            profile=profile,
        )
