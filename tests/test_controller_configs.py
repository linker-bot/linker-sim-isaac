from __future__ import annotations

from pathlib import Path

import pytest

from linkerbot_sim.mirror.scene_assembly import (
    _load_controller_profiles_cached,
)
from linkerbot_sim.configuration.controllers import (
    ControllerProfiles,
    controller_profile_from_mapping,
    controller_profiles_from_mappings,
    normalize_controller_bundle_name,
)
from linkerbot_sim.controllers.projection import (
    hybrid_force_position_settings,
    joint_control_settings,
    robot_usd_override_configs,
)
from linkerbot_sim.robots.classification import (
    RobotComponentMapping,
    component_for_name,
)
from linkerbot_sim.utils.config import load_yaml


def _controller_bundle(
    name: str, *, controllers_root: Path = Path("configs/controllers")
) -> ControllerProfiles:
    bundle = controllers_root / name
    documents = {
        component: load_yaml(path)
        for component, path in {
            "arm": bundle / "arm_controller.yaml",
            "hand": bundle / "hand_controller.yaml",
            "default": bundle / "default_controller.yaml",
        }.items()
        if path.is_file()
    }
    return controller_profiles_from_mappings(documents, source=str(bundle))


def _controller_profile_document(
    target: str = "arm", *, position_stiffness: float = 1000.0
) -> dict[str, object]:
    """构造完整 strict profile；负向测试只删除或替换其目标字段。"""

    return {
        "target": target,
        "position_control": {
            "method": "implicit",
            "active_joints": {
                "stiffness": position_stiffness,
                "damping": 50.0,
                "max_force": 100.0,
            },
            "follower_joints": {
                "stiffness": 50000.0,
                "damping": 50.0,
                "max_force": 100.0,
            },
        },
        "velocity_control": {
            "method": "explicit",
            "active_joints": {
                "damping": 20.0,
                "max_force": 100.0,
            },
            "follower_joints": {
                "stiffness": 50000.0,
                "damping": 40.0,
                "max_force": 100.0,
            },
        },
        "effort_control": {
            "method": "direct",
            "active_joints": {"effort_limit": 100.0},
            "follower_joints": {
                "stiffness": 50000.0,
                "damping": 40.0,
                "max_force": 100.0,
            },
        },
    }


def _delete_profile_path(profile: dict[str, object], path: str) -> None:
    """按测试用点分路径删除字段。"""

    parts = path.split(".")
    current = profile
    for part in parts[:-1]:
        value = current[part]
        assert isinstance(value, dict)
        current = value
    del current[parts[-1]]


def test_controller_profiles_split_arm_and_hand() -> None:
    profiles = _controller_bundle("physx")
    assert profiles.arm.name == "arm"
    assert profiles.hand.name == "hand"
    assert profiles.arm.position_control.method == "implicit"
    assert profiles.arm.velocity_control
    assert profiles.hand.effort_control

    position = joint_control_settings(profiles, mode="position")
    position_again = joint_control_settings(profiles, mode="position")
    assert position.default is position.arm
    assert position.arm is position_again.arm
    assert position.hand is position_again.hand
    assert position.component("AR5V2_L_arm_joint_1").mode == "position"
    assert position.component("AR5V2_L_arm_joint_1").method == "implicit"
    assert position.component("AR5V2_L_arm_joint_1").stiffness == (1000.0,)
    assert position.component("L6V1_L_hand_index_mcp_pitch").follower_stiffness == (
        50000.0,
    )

    velocity = joint_control_settings(profiles, mode="velocity")
    assert velocity.component("AR5V2_L_arm_joint_1").mode == "velocity"
    assert velocity.component("AR5V2_L_arm_joint_1").method == "explicit"
    assert velocity.component("AR5V2_L_arm_joint_1").stiffness == (0.0,)
    assert velocity.component("L6V1_L_hand_index_mcp_pitch").follower_stiffness == (
        50000.0,
    )
    assert velocity.component("L6V1_L_hand_index_mcp_pitch").follower_damping == (40.0,)

    effort = joint_control_settings(profiles, mode="effort")
    assert effort.component("L6V1_L_hand_index_mcp_pitch").mode == "effort"
    assert effort.component("L6V1_L_hand_index_mcp_pitch").method == "direct"
    assert effort.component("L6V1_L_hand_index_mcp_pitch").effort_limit == 100.0
    assert effort.component("L6V1_L_hand_index_mcp_pitch").stiffness == (0.0,)
    assert effort.component("L6V1_L_hand_index_mcp_pitch").damping == (0.0,)
    assert effort.component("L6V1_L_hand_index_mcp_pitch").max_force == 100.0
    assert effort.component("L6V1_L_hand_index_mcp_pitch").follower_stiffness == (
        50000.0,
    )
    assert effort.component("L6V1_L_hand_index_mcp_pitch").follower_damping == (40.0,)

    usd = robot_usd_override_configs(profiles)
    assert set(usd) == {"default", "arm", "hand"}
    assert usd["default"] is usd["arm"]
    assert usd["hand"].follower_drive_stiffness_seed == 50000.0


def test_controller_runtime_projections_consume_fully_parsed_profiles() -> None:
    profiles = _controller_bundle("physx")

    for mode in ("position", "velocity", "effort"):
        joint_control_settings(profiles, mode=mode)
    hybrid = hybrid_force_position_settings(profiles)
    assert hybrid.arm is not None
    assert (hybrid.arm.mode, hybrid.arm.method) == ("effort", "direct")
    assert hybrid.hand is not None
    assert (hybrid.hand.mode, hybrid.hand.method) == ("position", "implicit")
    robot_usd_override_configs(profiles)


def _write_controller_profile(
    path: Path, *, target: str, stiffness: float = 1000.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"target: {target}",
                "position_control:",
                "  method: implicit",
                "  active_joints:",
                f"    stiffness: {stiffness}",
                "    damping: 50.0",
                "    max_force: 100.0",
                "  follower_joints:",
                "    stiffness: 50000.0",
                "    damping: 50.0",
                "    max_force: 100.0",
                "velocity_control:",
                "  method: explicit",
                "  active_joints:",
                "    damping: 20.0",
                "    max_force: 100.0",
                "  follower_joints:",
                "    stiffness: 50000.0",
                "    damping: 40.0",
                "    max_force: 100.0",
                "effort_control:",
                "  method: direct",
                "  active_joints:",
                "    effort_limit: 100.0",
                "  follower_joints:",
                "    stiffness: 50000.0",
                "    damping: 40.0",
                "    max_force: 100.0",
            )
        ),
        encoding="utf-8",
    )


def test_controller_bundle_loads_canonical_physx_directory() -> None:
    profiles = _controller_bundle("physx")

    assert profiles.arm.name == "arm"
    assert profiles.hand.name == "hand"


def test_newton_bundle_is_backend_specific_and_conservative() -> None:
    physx = joint_control_settings(_controller_bundle("physx"))
    newton_profiles = _controller_bundle("newton")
    newton = joint_control_settings(newton_profiles)

    assert newton.arm.stiffness == (200.0,)
    assert newton.arm.damping == (20.0,)
    assert newton.arm.max_force == 100.0
    assert newton.hand.stiffness == (5.0,)
    assert newton.hand.damping == (0.2,)
    assert newton.hand.max_force == 0.5
    assert newton.hand.follower_stiffness == (0.0,)
    assert newton.hand.follower_damping == (0.0,)
    assert newton.hand.follower_max_force == 0.0
    assert newton.arm.stiffness != physx.arm.stiffness
    assert newton.hand.stiffness != physx.hand.stiffness

    usd_seeds = robot_usd_override_configs(newton_profiles)
    for seed in usd_seeds.values():
        assert seed.contact_material_override is False
        assert seed.friction_combine_mode is None
        assert seed.joint_friction is None
        assert seed.follower_joint_friction is None
        assert seed.rigid_body_linear_damping is None
        assert seed.rigid_body_angular_damping is None


def test_controller_bundle_supports_optional_default_component(tmp_path: Path) -> None:
    bundle = tmp_path / "lab"
    _write_controller_profile(bundle / "arm_controller.yaml", target="arm")
    _write_controller_profile(bundle / "hand_controller.yaml", target="hand")
    _write_controller_profile(
        bundle / "default_controller.yaml",
        target="default",
        stiffness=321.0,
    )

    profiles = _controller_bundle("lab", controllers_root=tmp_path)

    assert profiles.default is not None
    settings = joint_control_settings(profiles)
    settings_again = joint_control_settings(profiles)
    assert settings.default is not settings.arm
    assert settings.default is settings_again.default
    assert settings.default.stiffness == (321.0,)
    overrides = robot_usd_override_configs(profiles)
    assert overrides["default"] is not overrides["arm"]
    assert overrides["default"].drive_stiffness_seed == 321.0


@pytest.mark.parametrize("name", ("../default", "a/b", r"a\b", "", "."))
def test_controller_bundle_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="controller bundle name"):
        normalize_controller_bundle_name(name, label="controller bundle name")


@pytest.mark.parametrize("field", ("typo", "physx", "implicit_position_drive"))
def test_controller_profile_rejects_unknown_top_level_fields(field: str) -> None:
    profile = _controller_profile_document()
    profile[field] = {}
    with pytest.raises(ValueError, match=rf"controller\[arm\]\.{field}"):
        controller_profile_from_mapping("arm", profile)


def test_controller_profile_rejects_physx_joint_friction_field() -> None:
    profile = _controller_profile_document()
    position = profile["position_control"]
    assert isinstance(position, dict)
    active = position["active_joints"]
    assert isinstance(active, dict)
    active["joint_friction"] = 0.5
    with pytest.raises(
        ValueError,
        match=r"controller\[arm\]\.position_control\.active_joints\.joint_friction",
    ):
        controller_profile_from_mapping("arm", profile)


def test_scene_controller_bundle_cache_preserves_robot_order() -> None:
    calls: list[str] = []
    sentinels = {"shared": object(), "other": object()}

    def loader(name: str):
        calls.append(name)
        return sentinels[name]

    resolved = _load_controller_profiles_cached(
        ("shared", "other", "shared"), loader=loader
    )

    assert calls == ["shared", "other"]
    assert resolved == (
        sentinels["shared"],
        sentinels["other"],
        sentinels["shared"],
    )


def test_controller_profiles_require_arm_and_hand_documents() -> None:
    with pytest.raises(ValueError, match="missing controller profiles"):
        controller_profiles_from_mappings({"arm": {}}, source="unit bundle")


def test_controller_profile_requires_target_to_match_document_role() -> None:
    profile = _controller_profile_document(target="hand")

    with pytest.raises(
        ValueError,
        match=r"controller\[arm\]\.target must equal 'arm', got 'hand'",
    ):
        controller_profile_from_mapping("arm", profile)


def test_controller_profiles_reject_unknown_control_fields_and_methods() -> None:
    profile = _controller_profile_document()
    position = profile["position_control"]
    assert isinstance(position, dict)
    position["type"] = "implicit"
    with pytest.raises(ValueError, match=r"controller\[arm\]\.position_control\.type"):
        controller_profile_from_mapping("arm", profile)

    profile = _controller_profile_document()
    position = profile["position_control"]
    assert isinstance(position, dict)
    position["method"] = "implicit_drive"
    with pytest.raises(ValueError, match="method must be one of"):
        controller_profile_from_mapping("arm", profile)


@pytest.mark.parametrize(
    "path",
    (
        "target",
        "position_control",
        "velocity_control",
        "effort_control",
        "position_control.method",
        "position_control.active_joints",
        "position_control.follower_joints",
        "velocity_control.method",
        "velocity_control.active_joints",
        "velocity_control.follower_joints",
        "effort_control.method",
        "effort_control.active_joints",
        "effort_control.follower_joints",
        "position_control.active_joints.stiffness",
        "position_control.active_joints.damping",
        "position_control.active_joints.max_force",
        "velocity_control.active_joints.damping",
        "velocity_control.active_joints.max_force",
        "effort_control.active_joints.effort_limit",
        "position_control.follower_joints.stiffness",
        "position_control.follower_joints.damping",
        "position_control.follower_joints.max_force",
        "velocity_control.follower_joints.stiffness",
        "velocity_control.follower_joints.damping",
        "velocity_control.follower_joints.max_force",
        "effort_control.follower_joints.stiffness",
        "effort_control.follower_joints.damping",
        "effort_control.follower_joints.max_force",
    ),
)
def test_controller_profile_rejects_every_missing_required_field(path: str) -> None:
    profile = _controller_profile_document()
    _delete_profile_path(profile, path)

    expected_path = rf"controller\[arm\]\.{path.replace('.', r'\.')} is required"
    with pytest.raises(ValueError, match=expected_path):
        controller_profile_from_mapping("arm", profile)


@pytest.mark.parametrize(
    ("mode", "field"),
    (
        ("position", "effort_limit"),
        ("velocity", "stiffness"),
        ("velocity", "effort_limit"),
        ("effort", "stiffness"),
        ("effort", "damping"),
        ("effort", "max_force"),
    ),
)
def test_controller_profile_rejects_parameters_unused_by_mode(
    mode: str, field: str
) -> None:
    profile = _controller_profile_document()
    control = profile[f"{mode}_control"]
    assert isinstance(control, dict)
    active = control["active_joints"]
    assert isinstance(active, dict)
    active[field] = 1.0

    with pytest.raises(
        ValueError,
        match=rf"controller\[arm\]\.{mode}_control\.active_joints\.{field}",
    ):
        controller_profile_from_mapping("arm", profile)


def test_robot_name_classification_supports_left_and_right() -> None:
    assert component_for_name("AR5V2_L_arm_link1") == "arm"
    assert component_for_name("AR5V2_R_arm_link1") == "arm"
    assert component_for_name("L6V1_L_hand_index_mcp_pitch") == "hand"
    assert component_for_name("L6V1_R_hand_index_mcp_pitch") == "hand"
    assert component_for_name("world") == "default"


def test_robot_name_classification_uses_category_token_not_device_prefix() -> None:
    assert component_for_name("UR10V3_R_arm_joint_2") == "arm"
    assert component_for_name("DexHandV2_L_hand_index_mcp") == "hand"
    assert component_for_name("mobilebaseV1_base_link") == "default"


def test_exact_component_groups_override_nonstandard_names_and_validate_ownership() -> (
    None
):
    mapping = RobotComponentMapping.from_profile(
        {
            "joint_groups": {
                "arm": ["axis_a"],
                "hand": ["axis_b"],
                "passive": ["axis_free"],
            },
            "rigid_body_groups": {
                "arm": ["body_a"],
                "hand": ["body_b"],
                "default": ["mount"],
            },
        }
    )

    assert mapping.joint_component("axis_a") == "arm"
    assert mapping.joint_component("axis_b") == "hand"
    assert mapping.joint_component("axis_free") == "default"
    assert mapping.rigid_body_component("body_b") == "hand"
    assert mapping.joint_component("fixture_arm_joint") == "arm"

    with pytest.raises(ValueError, match="multiple components"):
        RobotComponentMapping.from_profile(
            {"joint_groups": {"arm": ["same"], "hand": ["same"]}}
        )
    with pytest.raises(ValueError, match="unsupported keys: typo"):
        RobotComponentMapping.from_profile(
            {"rigid_body_groups": {"arm": [], "typo": []}}
        )


def test_controller_parameters_accept_scalar_sequence_and_exact_name_map() -> None:
    document = _controller_profile_document()
    position = document["position_control"]
    assert isinstance(position, dict)
    active = position["active_joints"]
    assert isinstance(active, dict)
    active.update(
        {
            "stiffness": [11.0, 12.0],
            "damping": {"axis_a": 1.0, "axis_b": 2.0},
            "max_force": 9.0,
        }
    )
    profile = controller_profile_from_mapping(
        "arm",
        document,
    )
    profiles = _controller_bundle("physx")
    settings = joint_control_settings(
        profiles.__class__(arm=profile, hand=profiles.hand), mode="position"
    ).arm

    assert settings is not None
    assert settings.stiffness == (11.0, 12.0)
    assert settings.damping == {"axis_a": 1.0, "axis_b": 2.0}
    with pytest.raises(TypeError):
        settings.damping["axis_a"] = 3.0  # type: ignore[index]
    assert settings.max_force == 9.0


def test_controller_parameters_reject_unknown_fields_and_illegal_values() -> None:
    for field, value, message in (
        ("stifness", 1.0, "unsupported keys: stifness"),
        ("stiffness", -1.0, "finite and non-negative"),
        ("stiffness", [], "cannot be empty"),
    ):
        profile = _controller_profile_document()
        position = profile["position_control"]
        assert isinstance(position, dict)
        active = position["active_joints"]
        assert isinstance(active, dict)
        active[field] = value
        with pytest.raises(ValueError, match=message):
            controller_profile_from_mapping("arm", profile)


def test_all_bundled_controller_yaml_loads_strictly() -> None:
    paths = sorted(Path("configs/controllers").glob("**/*.yaml"))

    assert paths
    for path in paths:
        data = load_yaml(path)
        target = data.get("target")
        assert isinstance(target, str)
        profile = controller_profile_from_mapping(target, data, source=str(path))
        for mode in ("position", "velocity", "effort"):
            settings = joint_control_settings(
                ControllerProfiles(arm=profile, hand=profile),
                mode=mode,
            )
            assert settings.arm is not None
            assert settings.arm.mode == mode


def test_controller_nested_typo_reports_complete_path_eagerly() -> None:
    profile = _controller_profile_document()
    position = profile["position_control"]
    assert isinstance(position, dict)
    active = position["active_joints"]
    assert isinstance(active, dict)
    del active["stiffness"]
    active["stifness"] = 100.0
    with pytest.raises(
        ValueError,
        match=r"controller\[arm\]\.position_control\.active_joints\.stifness",
    ):
        controller_profile_from_mapping("arm", profile)


@pytest.mark.parametrize(
    ("field", "value", "path"),
    (
        ("target", 1, "controller[arm].target"),
        ("method", False, "position_control.method"),
        ("stiffness", True, "active_joints.stiffness"),
        ("stiffness", -1.0, "active_joints.stiffness"),
    ),
)
def test_controller_rejects_coerced_types_and_invalid_ranges(
    field: str,
    value: object,
    path: str,
) -> None:
    profile = _controller_profile_document()
    if field == "target":
        profile[field] = value
    elif field == "method":
        position = profile["position_control"]
        assert isinstance(position, dict)
        position[field] = value
    else:
        position = profile["position_control"]
        assert isinstance(position, dict)
        active = position["active_joints"]
        assert isinstance(active, dict)
        active[field] = value

    with pytest.raises(ValueError, match=path.replace("[", r"\[").replace("]", r"\]")):
        controller_profile_from_mapping("arm", profile)


def test_controller_unknown_field_reports_source_context() -> None:
    source = "/tmp/controller.yaml"
    profile = _controller_profile_document()
    profile["typo"] = True
    with pytest.raises(ValueError, match=r"controller\[arm\]\.typo") as exc_info:
        controller_profile_from_mapping("arm", profile, source=source)

    assert source in str(exc_info.value)


def test_controller_nested_missing_field_reports_source_context() -> None:
    source = "/tmp/controller.yaml"
    profile = _controller_profile_document()
    _delete_profile_path(profile, "velocity_control.active_joints.max_force")

    with pytest.raises(
        ValueError,
        match=r"velocity_control\.active_joints\.max_force is required",
    ) as exc_info:
        controller_profile_from_mapping("arm", profile, source=source)

    assert source in str(exc_info.value)
