from __future__ import annotations

import xml.etree.ElementTree as ET

from linkerbot_sim.assets.robot_loader import (
    DualRobotExecutionConfig,
    dual_robot_root_poses_from_env_config,
)
from linkerbot_sim.backends.cumotion.dual_urdf import (
    build_dual_arm_urdf_from_root_poses,
    build_dual_arm_xrdf,
    dual_cumotion_config_from_sides,
    dual_urdf_generation_config_from_robot_config,
    prepare_cumotion_config_from_robot_config,
)
from linkerbot_sim.app.motion.dual_arm_semantics import (
    dual_arm_semantics_from_robot_configs,
)
from linkerbot_sim.utils.config import load_yaml


def _dual_robot_config() -> dict[str, object]:
    return dual_cumotion_config_from_sides(
        left=load_yaml("configs/robots/ar5v2_l6v1_l.yaml"),
        right=load_yaml("configs/robots/ar5v2_l6v1_r.yaml"),
    )


def _dual_execution_config(root_poses):
    return DualRobotExecutionConfig.from_robot_configs(
        left=load_yaml("configs/robots/ar5v2_l6v1_l.yaml"),
        right=load_yaml("configs/robots/ar5v2_l6v1_r.yaml"),
        root_poses=root_poses,
    )


def test_dual_urdf_generation_uses_robot_root_poses(tmp_path) -> None:
    robot_config = _dual_robot_config()
    env_config = load_yaml("configs/envs/scene2.yaml")
    root_poses = dual_robot_root_poses_from_env_config(env_config)
    generation_config = dual_urdf_generation_config_from_robot_config(robot_config)
    assert generation_config is not None
    generation_config = generation_config.__class__(
        left_xrdf_path=generation_config.left_xrdf_path,
        right_xrdf_path=generation_config.right_xrdf_path,
        left_urdf_path=generation_config.left_urdf_path,
        right_urdf_path=generation_config.right_urdf_path,
        output_dir=tmp_path,
        robot_name=generation_config.robot_name,
        parent_link=generation_config.parent_link,
        left_base_link=generation_config.left_base_link,
        right_base_link=generation_config.right_base_link,
        left_mount_joint=generation_config.left_mount_joint,
        right_mount_joint=generation_config.right_mount_joint,
    )
    dual_execution = _dual_execution_config(root_poses)

    output = build_dual_arm_urdf_from_root_poses(
        generation_config,
        left_pose=dual_execution.left.root_pose,
        right_pose=dual_execution.right.root_pose,
    )

    root = ET.parse(output).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    left_origin = joints[generation_config.left_mount_joint].find("origin")
    right_origin = joints[generation_config.right_mount_joint].find("origin")
    assert left_origin is not None
    assert right_origin is not None
    assert left_origin.get("xyz") == _expected_vec(dual_execution.left.root_pose.xyz)
    assert left_origin.get("rpy") == _expected_vec(dual_execution.left.root_pose.rpy)
    assert right_origin.get("xyz") == _expected_vec(dual_execution.right.root_pose.xyz)
    assert right_origin.get("rpy") == _expected_vec(dual_execution.right.root_pose.rpy)
    assert root.find("link[@name='AR5V2_L_arm_base']") is not None
    assert root.find("link[@name='AR5V2_R_arm_base']") is not None


def test_dual_xrdf_generation_merges_left_and_right_cspace(tmp_path) -> None:
    left_config = load_yaml("configs/robots/ar5v2_l6v1_l.yaml")
    right_config = load_yaml("configs/robots/ar5v2_l6v1_r.yaml")
    robot_config = dual_cumotion_config_from_sides(left=left_config, right=right_config)
    semantics = dual_arm_semantics_from_robot_configs(
        {"left": left_config, "right": right_config}
    )
    generation_config = dual_urdf_generation_config_from_robot_config(robot_config)
    assert generation_config is not None
    generation_config = generation_config.__class__(
        left_xrdf_path=generation_config.left_xrdf_path,
        right_xrdf_path=generation_config.right_xrdf_path,
        left_urdf_path=generation_config.left_urdf_path,
        right_urdf_path=generation_config.right_urdf_path,
        output_dir=tmp_path,
        robot_name=generation_config.robot_name,
        parent_link=generation_config.parent_link,
        left_base_link=generation_config.left_base_link,
        right_base_link=generation_config.right_base_link,
        left_mount_joint=generation_config.left_mount_joint,
        right_mount_joint=generation_config.right_mount_joint,
    )

    output = build_dual_arm_xrdf(generation_config)

    import yaml

    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["cspace"]["joint_names"] == [
        *semantics.left_arm_joints,
        *semantics.right_arm_joints,
    ]
    assert data["tool_frames"] == [
        robot_config["cumotion"]["left"]["flange_frame"],
        robot_config["cumotion"]["right"]["flange_frame"],
    ]
    assert len(data["cspace"]["acceleration_limits"]) == 14
    assert len(data["cspace"]["jerk_limits"]) == 14


def test_prepare_cumotion_config_from_robot_config_generates_dual_assets(tmp_path) -> None:
    robot_config = _dual_robot_config()
    env_config = load_yaml("configs/envs/scene2.yaml")
    robot_config["cumotion"]["output_dir"] = str(tmp_path)

    prepared = prepare_cumotion_config_from_robot_config(
        robot_config,
        dual_root_poses=dual_robot_root_poses_from_env_config(env_config),
    )

    assert prepared.generated_assets is True
    assert prepared.urdf_path.is_file()
    assert prepared.xrdf_path.is_file()
    assert prepared.backend_config.urdf_path == prepared.urdf_path
    assert prepared.backend_config.xrdf_path == prepared.xrdf_path
    assert prepared.backend_config.flange_frame is None
    assert prepared.flange_frames == {
        "left": robot_config["cumotion"]["left"]["flange_frame"],
        "right": robot_config["cumotion"]["right"]["flange_frame"],
    }


def _expected_vec(values: tuple[float, float, float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)
