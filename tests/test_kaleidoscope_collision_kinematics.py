from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.configuration.catalog import load_yaml_mapping
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.isaac.physics.collision_shapes import (
    CONVEX_DECOMPOSITION,
    PhysxCollisionShapeMetadata,
)
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    RobotCollisionShape,
)
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    FullArticulatedForwardKinematics,
    RobotRootWorldTransformBatch,
    build_full_articulated_kinematics_spec,
    compose_candidate_collider_world_transforms,
    full_articulated_urdf_path,
)
from linkerbot_sim.mirror.collision.urdf_kinematics import _UrdfKinematics


ROOT = Path(__file__).resolve().parents[1]

COLLIDER_LOCAL = (
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
)


def _write_robot_asset(tmp_path: Path, joints: str) -> object:
    mjcf_path = tmp_path / "workstation.xml"
    urdf_path = tmp_path / "workstation.urdf"
    mjcf_path.write_text("<mujoco/>", encoding="utf-8")
    urdf_path.write_text(
        f"""<robot name="test">
  <link name="world"/>
  <link name="base"/>
  <link name="finger"/>
  <link name="tip"/>
  {joints}
</robot>
""",
        encoding="utf-8",
    )
    return SimpleNamespace(asset_type="mjcf", asset_path=str(mjcf_path))


def _valid_joints() -> str:
    return """
  <joint name="mount" type="fixed">
    <parent link="world"/><child link="base"/>
  </joint>
  <joint name="finger_joint" type="revolute">
    <parent link="base"/><child link="finger"/>
    <origin xyz="1 2 3" rpy="0.1 0.2 0.3"/>
    <axis xyz="0 0 2"/>
  </joint>
  <joint name="tip_joint" type="revolute">
    <parent link="finger"/><child link="tip"/>
    <axis xyz="0 1 0"/>
    <mimic joint="finger_joint" multiplier="1.5" offset="0.1"/>
  </joint>
"""


def _robot_catalog(*, rigid_body_name: str = "finger") -> RobotCollisionCatalog:
    shapes: list[RobotCollisionShape] = []
    for env_id in range(2):
        env_root = f"/World/envs/env_{env_id}"
        imported_root = f"{env_root}/left"
        articulation = f"{imported_root}/Robot"
        rigid_body = f"{articulation}/{rigid_body_name}"
        collision = f"{rigid_body}/collision"
        metadata = PhysxCollisionShapeMetadata(
            collision_path=collision,
            rigid_body_path=rigid_body,
            geometry_type="Mesh",
            mesh_approximation=CONVEX_DECOMPOSITION,
            shape_to_rigid_body_matrix_row_major=COLLIDER_LOCAL,
        )
        shapes.append(
            RobotCollisionShape(
                env_id=env_id,
                robot_id=0,
                robot_label="left",
                env_root_path=env_root,
                imported_root_path=imported_root,
                articulation_path=articulation,
                collision_suffix=f"/{rigid_body_name}/collision",
                rigid_body_suffix=f"/{rigid_body_name}",
                metadata=metadata,
            )
        )
    return RobotCollisionCatalog(
        num_envs=2,
        robot_labels=("left",),
        shapes=tuple(shapes),
    )


def _robot_root_transforms(
    *,
    env_ids: tuple[int, ...] = (0, 1),
    robot_label: str = "left",
    imported_root_paths: tuple[str, ...] | None = None,
    dtype: np.dtype = np.dtype(np.float64),
) -> RobotRootWorldTransformBatch:
    roots = np.broadcast_to(np.eye(4, dtype=dtype), (len(env_ids), 4, 4)).copy()
    if imported_root_paths is None:
        imported_root_paths = tuple(
            f"/World/envs/env_{env_id}/{robot_label}" for env_id in env_ids
        )
    return RobotRootWorldTransformBatch(
        env_ids=env_ids,
        robot_label=robot_label,
        imported_root_paths=imported_root_paths,
        matrices=roots,
    )


def test_full_spec_preserves_command_order_and_mimic_topology(tmp_path: Path) -> None:
    profile = _write_robot_asset(tmp_path, _valid_joints())

    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("tip", "finger"),
    )

    assert spec.urdf_path == tmp_path / "workstation.urdf"
    assert spec.root_link == "world"
    assert [joint.name for joint in spec.joints] == [
        "mount",
        "finger_joint",
        "tip_joint",
    ]
    assert spec.command_joint_names == ("finger_joint",)
    assert spec.required_link_names == ("tip", "finger")
    assert spec.mimic_joint_names == ("tip_joint",)
    finger = spec.joints[1]
    assert finger.origin_xyz == (1.0, 2.0, 3.0)
    assert finger.origin_rpy == (0.1, 0.2, 0.3)
    assert finger.axis_xyz == (0.0, 0.0, 1.0)
    mimic = spec.joints[2].mimic
    assert mimic is not None
    assert (mimic.source_joint, mimic.multiplier, mimic.offset) == (
        "finger_joint",
        1.5,
        0.1,
    )


@pytest.mark.parametrize("joint_type", ["revolute", "continuous"])
def test_batched_fk_outputs_gf_row_matrices_and_applies_mimic(
    tmp_path: Path,
    joint_type: str,
) -> None:
    joints = _valid_joints().replace('rpy="0.1 0.2 0.3"', 'rpy="0 0 0"')
    if joint_type == "continuous":
        joints = joints.replace('type="revolute"', 'type="continuous"', 1)
    profile = _write_robot_asset(tmp_path, joints)
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("finger", "tip"),
    )
    fk = FullArticulatedForwardKinematics(spec)
    positions = np.asarray(
        [
            [[0.0], [np.pi / 2.0]],
            [[-np.pi / 2.0], [np.pi]],
        ],
        dtype=np.float64,
    )

    result = fk.compute(positions)

    assert result.matrices.shape == (2, 2, 4, 4, 4)
    np.testing.assert_allclose(
        result.matrices_for_link("world"),
        np.broadcast_to(np.eye(4), (2, 2, 4, 4)),
        atol=1e-12,
    )
    finger = result.matrices_for_link("finger")
    np.testing.assert_allclose(
        finger[0, 1],
        np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 2.0, 3.0, 1.0],
            ]
        ),
        atol=1e-12,
    )
    tip = result.matrices_for_link("tip")
    assert not np.allclose(tip[0, 0], finger[0, 0])
    assert not np.allclose(tip[0, 1], finger[0, 1])
    assert result.matrices.flags.writeable is False
    with pytest.raises(KeyError, match="unknown FK link"):
        result.matrices_for_link("missing")


def test_batched_fk_supports_prismatic_motion(tmp_path: Path) -> None:
    joints = _valid_joints().replace('rpy="0.1 0.2 0.3"', 'rpy="0 0 0"')
    joints = joints.replace('type="revolute"', 'type="prismatic"', 1)
    profile = _write_robot_asset(tmp_path, joints)
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("finger",),
    )

    result = FullArticulatedForwardKinematics(spec).compute(
        np.asarray([[[2.0]]], dtype=np.float32)
    )

    assert result.matrices.dtype == np.float32
    np.testing.assert_allclose(
        result.matrices_for_link("finger")[0, 0, 3, :3],
        (1.0, 2.0, 5.0),
    )


def test_collider_composition_preserves_scale_and_selected_env_identity(
    tmp_path: Path,
) -> None:
    joints = _valid_joints().replace('rpy="0.1 0.2 0.3"', 'rpy="0 0 0"')
    profile = _write_robot_asset(tmp_path, joints)
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("finger",),
    )
    link_transforms = FullArticulatedForwardKinematics(spec).compute(
        np.asarray(
            [
                [[0.0], [np.pi / 2.0]],
                [[0.0], [-np.pi / 2.0]],
            ],
            dtype=np.float64,
        )
    )
    roots = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    roots[0, 3, :3] = (10.0, 0.0, 0.0)
    roots[1, 3, :3] = (20.0, 0.0, 0.0)
    root_transforms = _robot_root_transforms()
    root_transforms = RobotRootWorldTransformBatch(
        env_ids=root_transforms.env_ids,
        robot_label=root_transforms.robot_label,
        imported_root_paths=root_transforms.imported_root_paths,
        matrices=roots,
    )

    result = compose_candidate_collider_world_transforms(
        robot_catalog=_robot_catalog(),
        robot_label="left",
        kinematics_spec=spec,
        link_transforms=link_transforms,
        robot_root_transforms=root_transforms,
    )

    assert result.env_ids == (0, 1)
    assert result.collision_suffixes == ("/finger/collision",)
    assert result.collision_paths == (
        ("/World/envs/env_0/left/Robot/finger/collision",),
        ("/World/envs/env_1/left/Robot/finger/collision",),
    )
    assert result.matrices.shape == (2, 2, 1, 4, 4)
    np.testing.assert_allclose(
        result.matrices[0, 0, 0, 3, :3],
        (11.0, 2.0, 4.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.linalg.det(result.matrices[0, 0, 0, :3, :3]),
        0.125,
        atol=1e-12,
    )
    expected = (
        np.asarray(COLLIDER_LOCAL).reshape(4, 4)
        @ link_transforms.matrices_for_link("finger")[1, 1]
        @ roots[1]
    )
    np.testing.assert_allclose(result.matrices[1, 1, 0], expected, atol=1e-12)
    assert result.matrices.flags.writeable is False


def test_collider_composition_requires_exact_urdf_rigid_body_link(
    tmp_path: Path,
) -> None:
    profile = _write_robot_asset(tmp_path, _valid_joints())
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("finger",),
    )
    transforms = FullArticulatedForwardKinematics(spec).compute(
        np.zeros((1, 1, 1), dtype=np.float64)
    )

    with pytest.raises(ValueError, match="absent from full URDF links"):
        compose_candidate_collider_world_transforms(
            robot_catalog=_robot_catalog(rigid_body_name="finger_similar"),
            robot_label="left",
            kinematics_spec=spec,
            link_transforms=transforms,
            robot_root_transforms=_robot_root_transforms(env_ids=(0,)),
        )


@pytest.mark.parametrize(
    ("root_transforms", "error", "message"),
    [
        (
            _robot_root_transforms(env_ids=(2,)),
            ValueError,
            "outside robot catalog range",
        ),
        (
            _robot_root_transforms(env_ids=(0,), dtype=np.dtype(np.float32)),
            TypeError,
            "same dtype",
        ),
    ],
)
def test_collider_composition_rejects_invalid_env_or_root_matrix_contract(
    tmp_path: Path,
    root_transforms: RobotRootWorldTransformBatch,
    error: type[Exception],
    message: str,
) -> None:
    profile = _write_robot_asset(tmp_path, _valid_joints())
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("finger",),
    )
    transforms = FullArticulatedForwardKinematics(spec).compute(
        np.zeros((len(root_transforms.env_ids), 1, 1), dtype=np.float64)
    )

    with pytest.raises(error, match=message):
        compose_candidate_collider_world_transforms(
            robot_catalog=_robot_catalog(),
            robot_label="left",
            kinematics_spec=spec,
            link_transforms=transforms,
            robot_root_transforms=root_transforms,
        )


def test_collider_composition_rejects_cross_robot_or_env_root_identity(
    tmp_path: Path,
) -> None:
    profile = _write_robot_asset(tmp_path, _valid_joints())
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("finger",),
    )
    transforms = FullArticulatedForwardKinematics(spec).compute(
        np.zeros((2, 1, 1), dtype=np.float64)
    )

    with pytest.raises(ValueError, match="label does not match"):
        compose_candidate_collider_world_transforms(
            robot_catalog=_robot_catalog(),
            robot_label="left",
            kinematics_spec=spec,
            link_transforms=transforms,
            robot_root_transforms=_robot_root_transforms(robot_label="right"),
        )
    with pytest.raises(ValueError, match="FK environment rows"):
        compose_candidate_collider_world_transforms(
            robot_catalog=_robot_catalog(),
            robot_label="left",
            kinematics_spec=spec,
            link_transforms=transforms,
            robot_root_transforms=_robot_root_transforms(env_ids=(0,)),
        )
    with pytest.raises(ValueError, match="root path mismatch for env 0"):
        compose_candidate_collider_world_transforms(
            robot_catalog=_robot_catalog(),
            robot_label="left",
            kinematics_spec=spec,
            link_transforms=transforms,
            robot_root_transforms=_robot_root_transforms(
                imported_root_paths=(
                    "/World/envs/env_1/left",
                    "/World/envs/env_0/left",
                )
            ),
        )


def test_robot_root_transform_batch_rejects_invalid_matrix_or_env_contract() -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        RobotRootWorldTransformBatch(
            env_ids=(0, 0),
            robot_label="left",
            imported_root_paths=(
                "/World/envs/env_0/left-a",
                "/World/envs/env_0/left-b",
            ),
            matrices=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
        )
    with pytest.raises(ValueError, match="shape"):
        RobotRootWorldTransformBatch(
            env_ids=(0,),
            robot_label="left",
            imported_root_paths=("/World/envs/env_0/left",),
            matrices=np.zeros((1, 3, 3), dtype=np.float64),
        )
    non_affine = np.eye(4, dtype=np.float64)[None]
    non_affine[0, 0, 3] = 1.0
    with pytest.raises(ValueError, match="affine Gf matrices"):
        RobotRootWorldTransformBatch(
            env_ids=(0,),
            robot_label="left",
            imported_root_paths=("/World/envs/env_0/left",),
            matrices=non_affine,
        )
    singular = np.eye(4, dtype=np.float64)[None]
    singular[0, 2, 2] = 0.0
    with pytest.raises(ValueError, match="must be invertible"):
        RobotRootWorldTransformBatch(
            env_ids=(0,),
            robot_label="left",
            imported_root_paths=("/World/envs/env_0/left",),
            matrices=singular,
        )


@pytest.mark.parametrize(
    ("positions", "error", "message"),
    [
        (np.zeros((1, 1, 1), dtype=np.int64), TypeError, "float32 or float64"),
        (np.zeros((1, 1), dtype=np.float32), ValueError, "must have shape"),
        (np.zeros((1, 1, 2), dtype=np.float32), ValueError, "last dimension"),
        (np.zeros((0, 1, 1), dtype=np.float32), ValueError, "cannot be empty"),
        (
            np.asarray([[[float("nan")]]], dtype=np.float32),
            ValueError,
            "finite values",
        ),
    ],
)
def test_batched_fk_rejects_invalid_joint_batches(
    tmp_path: Path,
    positions: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    profile = _write_robot_asset(tmp_path, _valid_joints())
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=(),
    )

    with pytest.raises(error, match=message):
        FullArticulatedForwardKinematics(spec).compute(positions)


def test_full_spec_rejects_command_or_collision_link_drift(tmp_path: Path) -> None:
    profile = _write_robot_asset(tmp_path, _valid_joints())

    with pytest.raises(ValueError, match="do not exactly match"):
        build_full_articulated_kinematics_spec(
            robot_profile=profile,
            command_joint_names=("tip_joint",),
            required_link_names=("tip",),
        )
    with pytest.raises(ValueError, match="absent from full URDF"):
        build_full_articulated_kinematics_spec(
            robot_profile=profile,
            command_joint_names=("finger_joint",),
            required_link_names=("missing_link",),
        )


def test_full_spec_rejects_unresolved_mimic_source(tmp_path: Path) -> None:
    joints = _valid_joints().replace(
        'mimic joint="finger_joint"',
        'mimic joint="missing_joint"',
    )
    profile = _write_robot_asset(tmp_path, joints)

    with pytest.raises(ValueError, match="unknown moving joint"):
        build_full_articulated_kinematics_spec(
            robot_profile=profile,
            command_joint_names=("finger_joint",),
            required_link_names=(),
        )


def test_full_spec_rejects_disconnected_or_unsupported_topology(
    tmp_path: Path,
) -> None:
    disconnected = _valid_joints().replace(
        '<parent link="finger"/><child link="tip"/>',
        '<parent link="tip"/><child link="finger"/>',
    )
    profile = _write_robot_asset(tmp_path, disconnected)
    with pytest.raises(
        ValueError,
        match="multiple parent|root link|cannot be resolved",
    ):
        build_full_articulated_kinematics_spec(
            robot_profile=profile,
            command_joint_names=("finger_joint",),
            required_link_names=(),
        )

    profile = _write_robot_asset(
        tmp_path,
        _valid_joints().replace('type="revolute"', 'type="planar"', 1),
    )
    with pytest.raises(ValueError, match="unsupported URDF joint type"):
        build_full_articulated_kinematics_spec(
            robot_profile=profile,
            command_joint_names=("finger_joint",),
            required_link_names=(),
        )


def test_full_urdf_resolver_refuses_missing_companion(tmp_path: Path) -> None:
    mjcf_path = tmp_path / "robot.xml"
    mjcf_path.write_text("<mujoco/>", encoding="utf-8")
    profile = SimpleNamespace(asset_type="mjcf", asset_path=str(mjcf_path))

    with pytest.raises(FileNotFoundError, match="full articulated URDF companion"):
        full_articulated_urdf_path(profile)


@pytest.mark.parametrize(
    ("profile_name", "side", "arm_link"),
    [
        ("ar5_08_l6_l", "lh", "arm_AR5_5_08L_W4C4A6_link7"),
        ("ar5_08_l6_r", "rh", "arm_AR5_5_08R_W4C4A6_link7"),
    ],
)
def test_locked_l6_profiles_resolve_complete_arm_hand_urdf(
    profile_name: str,
    side: str,
    arm_link: str,
) -> None:
    profile_path = ROOT / "configs" / "robots" / f"{profile_name}.yaml"
    profile = RobotProfileSettings.from_mapping(
        load_yaml_mapping(profile_path),
        source=str(profile_path),
    )
    commands = profile.joint_groups.arm + profile.joint_groups.hand
    hand_link = f"hand_{side}_index_distal"

    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=commands,
        required_link_names=(arm_link, hand_link),
    )

    assert spec.urdf_path.name == "workstation.urdf"
    assert spec.urdf_path.parent == Path(profile.asset_path).resolve().parent
    assert spec.root_link == "world"
    assert len(spec.command_joint_names) == 13
    assert len(spec.mimic_joint_names) == 5
    assert len(spec.moving_joint_names) == 18
    assert arm_link in spec.link_names
    assert hand_link in spec.link_names
    assert "flange_hand_mount" in spec.link_names

    result = FullArticulatedForwardKinematics(spec).compute(
        np.zeros((1, 1, len(commands)), dtype=np.float64)
    )
    hand_base = result.matrices_for_link(f"hand_{side}_hand_base_link")[0, 0]
    np.testing.assert_allclose(
        hand_base[3, :3],
        (0.0, 0.0, 0.897),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.det(hand_base[:3, :3]),
        1.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("profile_name", ["ar5_08_l6_l", "ar5_08_l6_r"])
def test_full_fk_matches_existing_urdf_reference_for_every_link(
    profile_name: str,
) -> None:
    profile_path = ROOT / "configs" / "robots" / f"{profile_name}.yaml"
    profile = RobotProfileSettings.from_mapping(
        load_yaml_mapping(profile_path),
        source=str(profile_path),
    )
    commands = profile.joint_groups.arm + profile.joint_groups.hand
    spec = build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=commands,
        required_link_names=(),
    )
    positions = np.concatenate(
        (
            np.linspace(-0.2, 0.2, len(profile.joint_groups.arm)),
            np.full(len(profile.joint_groups.hand), 0.1),
        )
    )
    values = dict(zip(commands, positions, strict=True))
    for joint in spec.joints:
        if joint.mimic is not None:
            values[joint.name] = (
                values[joint.mimic.source_joint] * joint.mimic.multiplier
                + joint.mimic.offset
            )

    expected = _UrdfKinematics(spec.urdf_path).link_transforms(values)
    actual = FullArticulatedForwardKinematics(spec).compute(positions.reshape(1, 1, -1))

    for link_name in spec.link_names:
        np.testing.assert_allclose(
            actual.matrices_for_link(link_name)[0, 0],
            expected[link_name].T,
            rtol=0.0,
            atol=1e-12,
        )
