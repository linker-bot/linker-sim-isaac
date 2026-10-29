from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from linkerbot_sim.configuration.controllers import (
    compliant_drive_from_mapping,
    controller_profiles_from_mappings,
    validate_compliant_drive_selection,
)
from linkerbot_sim.controllers.projection import joint_control_settings

ROOT = Path("configs/controllers/newton_compliant")


def document():
    return yaml.safe_load((ROOT / "newton_drive.yaml").read_text())


def profiles():
    return controller_profiles_from_mappings(
        {
            name: yaml.safe_load((ROOT / filename).read_text())
            for name, filename in [
                ("arm", "arm_controller.yaml"),
                ("hand", "hand_controller.yaml"),
                ("newton_drive", "newton_drive.yaml"),
            ]
        }
    )


def test_selected_native_and_virtual_limits_remain_distinct():
    bundle = profiles()
    settings = bundle.compliant_drive
    assert settings is not None
    assert settings.thumb_yaw.advancing_torque == 3
    assert (
        settings.thumb_flexion.advancing_torque
        == settings.finger_flexion.advancing_torque
        == 5
    )
    assert bundle.hand.position_control.max_force == (60, 10, 60, 60, 60, 60)
    assert settings.arm_force_guard < bundle.arm.position_control.max_force
    assert settings.follower_damping == 0.4
    assert bundle.hand.position_control.follower_stiffness == (0.0,)
    validate_compliant_drive_selection(
        bundle,
        engine="newton",
        asset_path="units/ar5_08_o6_gemini335l_l/workstation.xml",
    )


@pytest.mark.parametrize(
    "engine,unit,mode",
    [
        ("physx", "ar5_08_o6_r", "position"),
        ("newton", "ar5_08_l6_r", "position"),
        ("newton", "ar5_08_o6_r", "effort"),
        ("newton", "ar5_08_o6_r", "velocity"),
    ],
)
def test_unsupported_combinations_fail_before_import(engine, unit, mode):
    with pytest.raises(ValueError):
        validate_compliant_drive_selection(
            profiles(),
            engine=engine,
            asset_path=f"units/{unit}/workstation.xml",
            mode=mode,
        )


@pytest.mark.parametrize("mode", ["velocity", "effort"])
def test_mode_transition_cannot_silently_bypass_virtual_motor(mode):
    with pytest.raises(ValueError, match="position"):
        joint_control_settings(profiles(), mode=mode)


@pytest.mark.parametrize(
    "key,value",
    [
        ("arm_force_guard", True),
        ("hand_armature", -1),
        ("motor_stiffness", 0),
        ("motor_speed_limit", float("nan")),
        ("arm_armature", "0.1"),
        ("equality_solref", [0, 1]),
        ("equality_solref", [1]),
        ("equality_solimp", [0.99, 0.5, 0.001, 0.5, 2]),
        ("equality_solimp", [0.9, 0.99]),
        ("thumb_yaw", {}),
    ],
)
def test_invalid_drive_parameters_rejected(key, value):
    data = document()
    data[key] = value
    with pytest.raises(ValueError):
        compliant_drive_from_mapping(data, label="drive")


def test_missing_unknown_and_reversed_motor_limits_rejected():
    for mutate in (
        lambda d: d.pop("motor_stiffness"),
        lambda d: d.update({"unknown": 1}),
        lambda d: d["thumb_yaw"].update(lower=2),
    ):
        data = deepcopy(document())
        mutate(data)
        with pytest.raises(ValueError):
            compliant_drive_from_mapping(data, label="drive")


@pytest.mark.parametrize("side", ["lh", "rh"])
def test_named_binding_survives_reordered_import_and_runtime(side):
    from types import SimpleNamespace
    from linkerbot_sim.configuration.controllers import bind_controller_joints
    from linkerbot_sim.controllers.types import resolve_joint_parameter

    source = profiles()
    arm = tuple(f"arm_joint_{i}" for i in range(1, 8))
    hand = tuple(
        f"hand_{side}_{suffix}"
        for suffix in (
            "thumb_cmc_pitch",
            "thumb_cmc_yaw",
            "index_mcp_pitch",
            "middle_mcp_pitch",
            "ring_mcp_pitch",
            "pinky_mcp_pitch",
        )
    )
    bound = bind_controller_joints(source, SimpleNamespace(arm=arm, hand=hand))
    settings = bound.hand.position_control
    order = tuple(sorted(hand))
    gains = dict(
        zip(
            order,
            resolve_joint_parameter(settings.stiffness, order, label="kp"),
            strict=True,
        )
    )
    caps = dict(
        zip(
            order,
            resolve_joint_parameter(settings.max_force, order, label="cap"),
            strict=True,
        )
    )
    assert gains[f"hand_{side}_thumb_cmc_yaw"] == 600
    assert gains[f"hand_{side}_thumb_cmc_pitch"] == 2500
    assert gains[f"hand_{side}_index_mcp_pitch"] == 1250
    assert caps[f"hand_{side}_thumb_cmc_yaw"] == 10
    assert caps[f"hand_{side}_index_mcp_pitch"] == 60
    assert source.hand.position_control.max_force == (60, 10, 60, 60, 60, 60)
    assert bind_controller_joints(bound, SimpleNamespace(arm=arm, hand=hand)) == bound
