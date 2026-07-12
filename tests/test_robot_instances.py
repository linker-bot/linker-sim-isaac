from __future__ import annotations

import pytest

from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    robot_instances_from_env_config,
    resolve_controller_profile,
)
from linkerbot_sim.configs.instance_paths import validate_disjoint_instance_prim_paths
from linkerbot_sim.configs.profiles import load_profile_yaml


def _instance(
    profile: str,
    *,
    label: str | None = None,
    prim_path: str | None = None,
    controller_profile: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "robot_profile": profile,
        "root_pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    }
    if label is not None:
        result["label"] = label
    if prim_path is not None:
        result["prim_path"] = prim_path
    if controller_profile is not None:
        result["controller_profile"] = controller_profile
    return result


def test_robot_instances_parse_list_and_generate_dense_ids() -> None:
    instances = robot_instances_from_env_config(
        {
            "robots": [
                _instance("ar5v2_l6v1_l", label="work_arm"),
                _instance("ar5v2_l6v1_r"),
            ]
        }
    )

    assert [item.robot_id for item in instances] == [0, 1]
    assert [item.label for item in instances] == [
        "work_arm",
        "ar5v2_l6v1_r_1",
    ]
    assert not hasattr(instances[0], "name")
    assert instances[0].effective_prim_path == "/World/Robots/work_arm"


def test_robot_instances_reordering_changes_ids_not_explicit_labels() -> None:
    first = _instance("profile_a", label="robot_a")
    second = _instance("profile_b", label="robot_b")

    original = robot_instances_from_env_config({"robots": [first, second]})
    reordered = robot_instances_from_env_config({"robots": [second, first]})

    assert [(item.robot_id, item.label) for item in original] == [
        (0, "robot_a"),
        (1, "robot_b"),
    ]
    assert [(item.robot_id, item.label) for item in reordered] == [
        (0, "robot_b"),
        (1, "robot_a"),
    ]


def test_robot_instances_reject_configured_id_duplicate_identity_and_path() -> None:
    configured_id = _instance("profile_a")
    configured_id["robot_id"] = 7
    with pytest.raises(ValueError, match=r"robots\[0\]\.robot_id.*generated"):
        robot_instances_from_env_config({"robots": [configured_id]})

    with pytest.raises(ValueError, match="Duplicate robot label"):
        robot_instances_from_env_config(
            {
                "robots": [
                    _instance("profile_a", label="duplicate"),
                    _instance("profile_b", label="duplicate"),
                ]
            }
        )

    with pytest.raises(ValueError, match="Duplicate robot prim path"):
        robot_instances_from_env_config(
            {
                "robots": [
                    _instance("profile_a", label="a", prim_path="/World/Same"),
                    _instance("profile_b", label="b", prim_path="/World/Same"),
                ]
            }
        )


def test_robot_and_object_instance_paths_cannot_share_or_nest_prim_trees() -> None:
    with pytest.raises(ValueError, match="prim paths overlap"):
        validate_disjoint_instance_prim_paths(
            robot_paths={"robot": "/World/Robots/robot"},
            object_paths={"fixture": "/World/Robots/robot/fixture"},
        )

    validate_disjoint_instance_prim_paths(
        robot_paths={"robot": "/World/Robots/robot"},
        object_paths={"fixture": "/World/Objects/fixture"},
    )


@pytest.mark.parametrize(
    ("robot_paths", "object_paths", "message"),
    (
        (
            {"parent": "/World/Robots", "child": "/World/Robots/child"},
            {},
            "Robot instance prim paths overlap",
        ),
        (
            {},
            {"parent": "/World/Objects", "child": "/World/Objects/child"},
            "Object instance prim paths overlap",
        ),
    ),
)
def test_same_domain_instance_paths_cannot_nest_prim_trees(
    robot_paths: dict[str, str],
    object_paths: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_disjoint_instance_prim_paths(
            robot_paths=robot_paths,
            object_paths=object_paths,
        )


@pytest.mark.parametrize(
    "prim_path", ["World/Robot", "", "/", "/World/Robot/", "/World//Robot"]
)
def test_robot_instances_reject_invalid_prim_path(prim_path: str) -> None:
    with pytest.raises(ValueError, match=r"robots\[0\]\.prim_path"):
        robot_instances_from_env_config(
            {"robots": [_instance("profile_a", label="robot_a", prim_path=prim_path)]}
        )


def test_robot_instances_empty_requires_explicit_allow_empty() -> None:
    with pytest.raises(ValueError, match="robots cannot be empty"):
        robot_instances_from_env_config({"robots": []})
    assert robot_instances_from_env_config({"robots": []}, allow_empty=True) == ()


def test_robot_execution_applies_instance_label_and_prim_path() -> None:
    profile = load_profile_yaml("robot", "ar5v2_l6v1_l")
    instance = robot_instances_from_env_config(
        {"robots": [_instance("ar5v2_l6v1_l", label="robot_a")]}
    )[0]

    execution = RobotExecutionConfig.from_mapping(profile, scene_instance=instance)

    assert execution.robot.name == "robot_a"
    assert execution.robot.prim_path == "/World/Robots/robot_a"
    assert execution.root_pose == instance.root_pose


def test_robot_execution_requires_scene_instance() -> None:
    profile = load_profile_yaml("robot", "ar5v2_l6v1_l")

    with pytest.raises(TypeError, match="scene_instance"):
        RobotExecutionConfig.from_mapping(profile)  # type: ignore[call-arg]


def test_controller_profile_resolution_uses_instance_then_robot_then_runtime() -> None:
    profile = load_profile_yaml("robot", "ar5v2_l6v1_l")
    profile["robot"]["controller_profile"] = "robot_bundle"

    instance_override = robot_instances_from_env_config(
        {
            "robots": [
                _instance(
                    "ar5v2_l6v1_l",
                    label="instance_override",
                    controller_profile="instance_bundle",
                )
            ]
        }
    )[0]
    execution = RobotExecutionConfig.from_mapping(
        profile, scene_instance=instance_override
    )
    assert (
        resolve_controller_profile(instance_override, execution.robot, "runtime_bundle")
        == "instance_bundle"
    )

    robot_override = robot_instances_from_env_config(
        {"robots": [_instance("ar5v2_l6v1_l", label="robot_override")]}
    )[0]
    execution = RobotExecutionConfig.from_mapping(
        profile, scene_instance=robot_override
    )
    assert (
        resolve_controller_profile(robot_override, execution.robot, "runtime_bundle")
        == "robot_bundle"
    )

    profile["robot"].pop("controller_profile")
    execution = RobotExecutionConfig.from_mapping(
        profile, scene_instance=robot_override
    )
    assert (
        resolve_controller_profile(robot_override, execution.robot, "runtime_bundle")
        == "runtime_bundle"
    )


@pytest.mark.parametrize("value", ("../escape", "nested/bundle", ""))
def test_robot_instance_rejects_unsafe_controller_profile(value: str) -> None:
    with pytest.raises(ValueError, match=r"robots\[0\]\.controller_profile"):
        robot_instances_from_env_config(
            {
                "robots": [
                    _instance(
                        "ar5v2_l6v1_l",
                        label="robot_a",
                        controller_profile=value,
                    )
                ]
            }
        )
