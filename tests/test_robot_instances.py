from __future__ import annotations

from dataclasses import replace

import pytest

from linkerbot_sim.assets.instance_paths import validate_disjoint_instance_prim_paths
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    RobotSceneInstanceConfig,
    resolve_controller_profile,
    robot_scene_instances_from_settings,
)
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.configuration.scenes import PoseSettings, RobotInstanceSettings
from linkerbot_sim.utils.config import load_yaml


def load_robot_profile_by_name(name: str) -> RobotProfileSettings:
    path = f"configs/robots/{name}.yaml"
    return RobotProfileSettings.from_mapping(load_yaml(path), source=path)


def _instance(
    profile: str,
    *,
    label: str,
    controller_profile: str | None = None,
) -> RobotInstanceSettings:
    return RobotInstanceSettings(
        label=label,
        robot_profile=profile,
        root_pose=PoseSettings(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)),
        controller_profile=controller_profile,
    )


def test_typed_robot_instances_preserve_scene_order_and_generate_dense_ids() -> None:
    instances = robot_scene_instances_from_settings(
        (
            _instance("ar5v2_l6v1_l", label="left_arm"),
            _instance("ar5v2_l6v1_r", label="right_arm"),
        )
    )

    assert [(item.robot_id, item.label) for item in instances] == [
        (0, "left_arm"),
        (1, "right_arm"),
    ]
    assert instances[0].effective_prim_path == "/World/Robots/left_arm"


def test_typed_robot_instance_reordering_changes_ids_not_labels() -> None:
    first = _instance("profile_a", label="robot_a")
    second = _instance("profile_b", label="robot_b")

    original = robot_scene_instances_from_settings((first, second))
    reordered = robot_scene_instances_from_settings((second, first))

    assert [(item.robot_id, item.label) for item in original] == [
        (0, "robot_a"),
        (1, "robot_b"),
    ]
    assert [(item.robot_id, item.label) for item in reordered] == [
        (0, "robot_b"),
        (1, "robot_a"),
    ]


def test_typed_robot_instance_factory_rejects_unparsed_values() -> None:
    with pytest.raises(TypeError, match="RobotInstanceSettings"):
        robot_scene_instances_from_settings(({"label": "robot"},))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"robot_id": True},
        {"robot_id": -1},
        {"label": "unsafe/name"},
        {"prim_path": "World/Robot"},
        {"prim_path": "/World/Robot/"},
    ),
)
def test_robot_scene_instance_rejects_invalid_typed_identity(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "robot_profile": "profile_a",
        "root_pose": RootPoseConfig(),
        "robot_id": 0,
        "label": "robot_a",
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        RobotSceneInstanceConfig(**values)  # type: ignore[arg-type]


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


def test_robot_execution_applies_instance_label_and_prim_path() -> None:
    profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    instance = robot_scene_instances_from_settings(
        (_instance("ar5v2_l6v1_l", label="robot_a"),)
    )[0]

    execution = RobotExecutionConfig.from_profile(profile, scene_instance=instance)

    assert execution.robot.name == "robot_a"
    assert execution.robot.prim_path == "/World/Robots/robot_a"
    assert execution.root_pose == instance.root_pose


def test_robot_execution_requires_scene_instance() -> None:
    profile = load_robot_profile_by_name("ar5v2_l6v1_l")

    with pytest.raises(TypeError, match="scene_instance"):
        RobotExecutionConfig.from_profile(profile)  # type: ignore[call-arg]


def test_controller_profile_resolution_uses_instance_then_robot_then_runtime() -> None:
    profile = replace(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        controller_profile="robot_bundle",
    )
    instance_override = robot_scene_instances_from_settings(
        (
            _instance(
                "ar5v2_l6v1_l",
                label="instance_override",
                controller_profile="instance_bundle",
            ),
        )
    )[0]
    execution = RobotExecutionConfig.from_profile(
        profile,
        scene_instance=instance_override,
    )
    assert (
        resolve_controller_profile(instance_override, execution.robot, "runtime_bundle")
        == "instance_bundle"
    )

    robot_override = robot_scene_instances_from_settings(
        (_instance("ar5v2_l6v1_l", label="robot_override"),)
    )[0]
    execution = RobotExecutionConfig.from_profile(
        profile, scene_instance=robot_override
    )
    assert (
        resolve_controller_profile(robot_override, execution.robot, "runtime_bundle")
        == "robot_bundle"
    )

    default_profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    execution = RobotExecutionConfig.from_profile(
        default_profile,
        scene_instance=robot_override,
    )
    assert (
        resolve_controller_profile(robot_override, execution.robot, "runtime_bundle")
        == "runtime_bundle"
    )


@pytest.mark.parametrize("value", ("../escape", "nested/bundle", ""))
def test_robot_instance_rejects_unsafe_controller_profile(value: str) -> None:
    with pytest.raises(ValueError, match="controller_profile"):
        RobotSceneInstanceConfig(
            robot_profile="ar5v2_l6v1_l",
            root_pose=RootPoseConfig(),
            label="robot_a",
            controller_profile=value,
        )
