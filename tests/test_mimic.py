from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.assets.robot_config import RobotAssetConfig
from linkerbot_sim.app.interactive.tiled_scene import command_utils
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.robots.mimic.assets import (
    mimic_follower_joint_names,
    parse_asset_mimic_relations,
)
from linkerbot_sim.robots.mimic.mjcf import (
    MjcfJointEquality,
    expand_targets_with_mjcf_equalities,
    mjcf_equality_follower_joint_names,
    parse_mjcf_joint_equalities,
)
from linkerbot_sim.robots.mimic.runtime import (
    MimicFollowerControl,
    MimicFollowerTargetMapper,
    resolve_mimic_follower_controls,
)
from linkerbot_sim.robots.mimic.urdf import parse_urdf_joint_mimics
from linkerbot_sim.tiled.scene.views import _command_joint_indices


AR5_L6_MJCF = RobotAssetConfig.from_mapping(
    load_profile_yaml("robot", "ar5v2_l6v1_l"),
    prim_path="/World/Robots/test_robot",
).asset_path


def test_parse_ar5_l6_mjcf_equalities() -> None:
    equalities = parse_mjcf_joint_equalities(AR5_L6_MJCF)
    names = {equality.dependent_joint for equality in equalities}
    assert "L6V1_L_hand_index_dip" in names
    assert "L6V1_L_hand_thumb_dip" in names
    assert len(equalities) == 5


def test_expand_hand_targets_with_followers() -> None:
    expanded = expand_targets_with_mjcf_equalities(
        {"L6V1_L_hand_index_mcp_pitch": 0.4, "L6V1_L_hand_thumb_cmc_pitch": 0.5},
        AR5_L6_MJCF,
    )
    assert np.isclose(expanded["L6V1_L_hand_index_dip"], 0.4 * 1.125676)
    assert np.isclose(expanded["L6V1_L_hand_thumb_dip"], 0.5 * 1.226495)


def test_resolve_follower_controls() -> None:
    dof_names = [
        "L6V1_L_hand_index_mcp_pitch",
        "L6V1_L_hand_index_dip",
        "L6V1_L_hand_thumb_cmc_pitch",
        "L6V1_L_hand_thumb_dip",
    ]
    controls = resolve_mimic_follower_controls(dof_names, AR5_L6_MJCF)
    follower_names = {control.dependent_joint for control in controls}
    assert follower_names == {"L6V1_L_hand_index_dip", "L6V1_L_hand_thumb_dip"}
    assert {control.master_index for control in controls} == {0, 2}


def test_polycoef_mimic_position_and_velocity() -> None:
    equality = MjcfJointEquality(
        name="nonlinear",
        dependent_joint="follower",
        master_joint="master",
        polycoef=(0.2, 1.5, -0.25, 0.1),
    )
    master_position = 0.4
    master_velocity = 0.2
    expected_position = (
        0.2
        + 1.5 * master_position
        - 0.25 * master_position**2
        + 0.1 * master_position**3
    )
    expected_velocity = (
        1.5 - 0.5 * master_position + 0.3 * master_position**2
    ) * master_velocity
    assert np.isclose(equality.evaluate_position(master_position), expected_position)
    assert np.isclose(
        equality.evaluate_velocity(master_position, master_velocity), expected_velocity
    )


def test_follower_mapper_uses_actual_master_state() -> None:
    mapper = MimicFollowerTargetMapper(["master", "follower"], None)
    mapper.controls = [
        MimicFollowerControl(
            dependent_joint="follower",
            master_joint="master",
            dependent_index=1,
            master_index=0,
            polycoef=(0.1, 2.0, 0.5),
        )
    ]
    target_positions = np.asarray([0.9, 0.9])
    target_velocities = np.asarray([0.8, 0.8])
    actual_positions = np.asarray([0.3, 0.0])
    actual_velocities = np.asarray([0.4, 0.0])
    mapper.apply_from_actual(
        target_positions, target_velocities, actual_positions, actual_velocities
    )
    np.testing.assert_allclose(target_positions, [0.9, 0.1 + 2.0 * 0.3 + 0.5 * 0.3**2])
    np.testing.assert_allclose(target_velocities, [0.8, (2.0 + 2.0 * 0.5 * 0.3) * 0.4])


def test_follower_joint_name_set() -> None:
    names = mjcf_equality_follower_joint_names(AR5_L6_MJCF)
    assert {
        "L6V1_L_hand_index_dip",
        "L6V1_L_hand_middle_dip",
        "L6V1_L_hand_ring_dip",
        "L6V1_L_hand_pinky_dip",
        "L6V1_L_hand_thumb_dip",
    } <= names


def _write_mimic_mjcf(
    path: Path,
    *,
    joint_names: tuple[str, ...] = ("master", "follower"),
    equality_xml: str,
) -> Path:
    joints = "\n".join(
        f'      <joint name="{joint_name}" type="hinge"/>' for joint_name in joint_names
    )
    path.write_text(
        f"""\
<mujoco model="mimic_test">
  <worldbody>
    <body name="root">
{joints}
    </body>
  </worldbody>
  <equality>
{equality_xml}
  </equality>
</mujoco>
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("joint1", "joint2", "role"),
    (
        ("missing", "master", "joint1"),
        ("follower", "missing", "joint2"),
    ),
)
def test_mjcf_mimic_rejects_dangling_joint_reference(
    tmp_path: Path,
    joint1: str,
    joint2: str,
    role: str,
) -> None:
    mjcf_path = _write_mimic_mjcf(
        tmp_path / "dangling.xml",
        equality_xml=(
            f'    <joint name="dangling" joint1="{joint1}" joint2="{joint2}"/>'
        ),
    )

    with pytest.raises(
        ValueError, match=rf"'dangling' {role} references unknown joint 'missing'"
    ):
        parse_mjcf_joint_equalities(mjcf_path)


def test_mjcf_mimic_rejects_self_reference(tmp_path: Path) -> None:
    mjcf_path = _write_mimic_mjcf(
        tmp_path / "self.xml",
        equality_xml='    <joint name="self" joint1="master" joint2="master"/>',
    )

    with pytest.raises(ValueError, match="joint 'master' cannot mimic itself"):
        parse_mjcf_joint_equalities(mjcf_path)


def test_mjcf_mimic_rejects_duplicate_follower(tmp_path: Path) -> None:
    mjcf_path = _write_mimic_mjcf(
        tmp_path / "duplicate.xml",
        equality_xml="""\
    <joint name="first" joint1="follower" joint2="master"/>
    <joint name="second" joint1="follower" joint2="master"/>""",
    )

    with pytest.raises(ValueError, match="follower in more than one equality"):
        parse_mjcf_joint_equalities(mjcf_path)


def test_mjcf_mimic_rejects_cycle(tmp_path: Path) -> None:
    mjcf_path = _write_mimic_mjcf(
        tmp_path / "cycle.xml",
        joint_names=("first", "second"),
        equality_xml="""\
    <joint name="first_follows_second" joint1="first" joint2="second"/>
    <joint name="second_follows_first" joint1="second" joint2="first"/>""",
    )

    with pytest.raises(ValueError, match=r"mimic cycle.*first -> second -> first"):
        parse_mjcf_joint_equalities(mjcf_path)


@pytest.mark.parametrize("non_finite", ("nan", "inf", "-inf"))
def test_mjcf_mimic_rejects_non_finite_polycoef(
    tmp_path: Path, non_finite: str
) -> None:
    mjcf_path = _write_mimic_mjcf(
        tmp_path / "non_finite.xml",
        equality_xml=(
            '    <joint name="bad_polycoef" joint1="follower" joint2="master" '
            f'polycoef="0 {non_finite}"/>'
        ),
    )

    with pytest.raises(ValueError, match="polycoef must contain only finite values"):
        parse_mjcf_joint_equalities(mjcf_path)


def test_mjcf_single_joint_equality_is_not_a_mimic_relation(tmp_path: Path) -> None:
    mjcf_path = _write_mimic_mjcf(
        tmp_path / "fixed.xml",
        equality_xml='    <joint name="fixed" joint1="follower" polycoef="0.2"/>',
    )

    assert parse_mjcf_joint_equalities(mjcf_path) == []


def _write_mimic_urdf(path: Path) -> Path:
    path.write_text(
        """\
<robot name="mimic_test">
  <link name="base"/>
  <link name="master_link"/>
  <link name="follower_link"/>
  <joint name="master" type="revolute">
    <parent link="base"/>
    <child link="master_link"/>
  </joint>
  <joint name="follower" type="revolute">
    <parent link="master_link"/>
    <child link="follower_link"/>
    <mimic joint="master" multiplier="-1.5" offset="0.2"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_parse_urdf_mimic_preserves_multiplier_and_offset(tmp_path: Path) -> None:
    urdf_path = _write_mimic_urdf(tmp_path / "robot.urdf")

    relations = parse_urdf_joint_mimics(urdf_path)
    generic_relations = parse_asset_mimic_relations(urdf_path)

    assert len(relations) == 1
    assert relations[0].dependent_joint == "follower"
    assert relations[0].master_joint == "master"
    assert relations[0].polycoef == (0.2, -1.5)
    assert generic_relations[0].polycoef == (0.2, -1.5)
    assert mimic_follower_joint_names(urdf_path) == {"follower"}


@pytest.mark.parametrize(
    "urdf_path",
    (
        Path("assets/single_system/hand/L6V1_L/L6V1_L.urdf"),
        Path("assets/single_system/hand/L6V1_R/L6V1_R.urdf"),
    ),
)
def test_parse_bundled_hand_urdf_ignores_transmission_joint_references(
    urdf_path: Path,
) -> None:
    relations = parse_urdf_joint_mimics(urdf_path)

    assert len(relations) == 5
    assert len({relation.dependent_joint for relation in relations}) == 5


def test_urdf_mimic_runtime_uses_affine_relation(tmp_path: Path) -> None:
    urdf_path = _write_mimic_urdf(tmp_path / "robot.urdf")
    mapper = MimicFollowerTargetMapper(["master", "follower"], urdf_path)
    target_positions = np.zeros(2, dtype=float)
    target_velocities = np.zeros(2, dtype=float)

    mapper.apply_from_actual(
        target_positions,
        target_velocities,
        actual_positions=np.asarray([0.4, 0.0]),
        actual_velocities=np.asarray([0.3, 0.0]),
    )

    np.testing.assert_allclose(target_positions, [0.0, -0.4])
    np.testing.assert_allclose(target_velocities, [0.0, -0.45])


def test_tiled_command_space_excludes_urdf_mimic_follower(tmp_path: Path) -> None:
    urdf_path = _write_mimic_urdf(tmp_path / "robot.urdf")

    indices = _command_joint_indices(
        dof_names=("master", "follower"),
        controlled_joints=("all",),
        mimic_path=urdf_path,
    )

    np.testing.assert_array_equal(indices, [0])


def test_urdf_mimic_rejects_dangling_master(tmp_path: Path) -> None:
    urdf_path = tmp_path / "dangling.urdf"
    urdf_path.write_text(
        """\
<robot name="dangling">
  <joint name="follower" type="revolute">
    <mimic joint="missing"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mimics unknown joint 'missing'"):
        parse_urdf_joint_mimics(urdf_path)


def test_urdf_mimic_rejects_cycles(tmp_path: Path) -> None:
    urdf_path = tmp_path / "cycle.urdf"
    urdf_path.write_text(
        """\
<robot name="cycle">
  <joint name="first" type="revolute"><mimic joint="second"/></joint>
  <joint name="second" type="revolute"><mimic joint="first"/></joint>
</robot>
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mimic cycle"):
        parse_urdf_joint_mimics(urdf_path)


def test_mimic_binding_rejects_relation_with_only_one_present_joint(
    tmp_path: Path,
) -> None:
    urdf_path = _write_mimic_urdf(tmp_path / "robot.urdf")

    with pytest.raises(ValueError, match="missing 'master'"):
        resolve_mimic_follower_controls(["follower"], urdf_path)


def test_tiled_runtime_applies_mjcf_followers_from_actual_master_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    class _View:
        def get_joint_positions(self):
            return np.asarray([[0.4, 0.0], [-0.2, 0.0]], dtype=float)

        def get_joint_velocities(self):
            return np.asarray([[0.3, 0.0], [0.5, 0.0]], dtype=float)

    monkeypatch.setattr(
        command_utils,
        "_apply_joint_targets",
        lambda _view, targets, *, velocities, joint_indices: applied.append(
            (
                np.asarray(targets),
                np.asarray(velocities),
                np.asarray(joint_indices),
            )
        ),
    )
    articulation = SimpleNamespace(
        view=_View(),
        runtime_mimic_controls=(
            MimicFollowerControl(
                dependent_joint="follower",
                master_joint="master",
                dependent_index=1,
                master_index=0,
                polycoef=(0.2, -1.5),
            ),
        ),
    )

    command_utils._apply_runtime_mimic_targets(articulation)

    assert len(applied) == 1
    positions, velocities, indices = applied[0]
    np.testing.assert_allclose(positions, [[-0.4], [0.5]])
    np.testing.assert_allclose(velocities, [[-0.45], [-0.75]])
    np.testing.assert_array_equal(indices, [1])
