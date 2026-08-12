from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.snapshots.compatibility import (
    ObjectTargetDescriptor,
    RobotTargetDescriptor,
    SnapshotCompatibilityError,
    SnapshotTargetDescriptor,
    check_snapshot_compatibility,
    require_snapshot_compatibility,
)
from linkerbot_sim.snapshots.schema import (
    ObjectSnapshot,
    RobotSnapshot,
    SceneSnapshot,
    SnapshotMetadata,
)


def _robot(
    label: str,
    robot_id: int,
    *,
    joint_names: tuple[str, ...] = ("j0",),
    positions: tuple[float, ...] = (0.0,),
    robot_profile: str | None = None,
    asset_fingerprint: str | None = None,
) -> RobotSnapshot:
    return RobotSnapshot(
        label=label,
        robot_id=robot_id,
        robot_profile=robot_profile,
        asset_fingerprint=asset_fingerprint,
        joint_names=joint_names,
        joint_positions=np.asarray(positions, dtype=float),
        joint_velocities=np.zeros(len(joint_names), dtype=float),
    )


def test_snapshot_round_trips_json_mapping() -> None:
    snapshot = SceneSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="mirror",
            source_env_id=0,
            step=12,
            time_s=0.5,
        ),
        robots={
            "workcell_robot": RobotSnapshot(
                label="workcell_robot",
                robot_id=3,
                robot_profile="arm_hand",
                asset_fingerprint="asset-a",
                joint_names=("j0", "j1"),
                joint_positions=np.asarray([0.1, 0.2]),
                joint_velocities=np.asarray([0.0, 0.3]),
                command_joint_names=("j0", "j1"),
                command_targets=np.asarray([0.1, 0.2]),
            )
        },
        objects={
            "block": ObjectSnapshot(
                name="block",
                object_profile="block-v1",
                positions_local=np.asarray([0.3, 0.0, -0.4]),
                orientations_wxyz=np.asarray([2.0, 0.0, 0.0, 0.0]),
                body_names=("body0", "body1"),
                body_positions_local=np.asarray(
                    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float
                ),
                body_orientations_wxyz=np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    dtype=float,
                ),
                generalized_signature=(
                    "newton-generalized-state-v1",
                    "joint=@root/body0;type=4;q_width=7;qd_width=6",
                ),
                generalized_q_names=tuple(f"root.q[{index}]" for index in range(7)),
                generalized_qd_names=tuple(f"root.qd[{index}]" for index in range(6)),
                generalized_q=np.asarray([0.3, 0.0, -0.4, 0.0, 0.0, 0.0, 1.0]),
                generalized_qd=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                generalized_world_origin=np.asarray([2.0, 0.0, 0.0]),
            )
        },
    )

    payload = snapshot.as_dict()
    restored = SceneSnapshot.from_mapping(payload)

    assert payload["schema"] == "linkerbot.scene-snapshot.v1"
    assert isinstance(payload["robots"], list)
    assert payload["robots"][0]["robot_id"] == 3
    assert "role" not in payload["robots"][0]
    assert restored.metadata.source_runtime == "mirror"
    assert restored.robots["workcell_robot"].robot_profile == "arm_hand"
    np.testing.assert_allclose(
        restored.robots["workcell_robot"].joint_positions, [0.1, 0.2]
    )
    np.testing.assert_allclose(
        restored.objects["block"].orientations_wxyz,
        [1.0, 0.0, 0.0, 0.0],
    )
    assert restored.objects["block"].generalized_signature == (
        "newton-generalized-state-v1",
        "joint=@root/body0;type=4;q_width=7;qd_width=6",
    )
    np.testing.assert_array_equal(
        restored.objects["block"].generalized_q,
        [0.3, 0.0, -0.4, 0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        restored.objects["block"].generalized_world_origin,
        [2.0, 0.0, 0.0],
    )


def test_snapshot_reader_rejects_invalid_schema_shape_and_fields() -> None:
    with pytest.raises(ValueError, match="unsupported snapshot schema"):
        SceneSnapshot.from_mapping({"schema": "another.snapshot", "robots": []})
    with pytest.raises(ValueError, match="snapshot.schema is required"):
        SceneSnapshot.from_mapping({"robots": []})
    with pytest.raises(ValueError, match="robots must be an array"):
        SceneSnapshot.from_mapping(
            {"schema": "linkerbot.scene-snapshot.v1", "robots": {}}
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        SceneSnapshot.from_mapping(
            {
                "schema": "linkerbot.scene-snapshot.v1",
                "robots": [
                    {
                        "unexpected_identity": "robot",
                        "label": "robot",
                        "robot_id": 0,
                        "joint_names": ["j0"],
                        "joint_positions": [0.0],
                        "joint_velocities": [0.0],
                    }
                ],
            }
        )


def test_robot_snapshot_requires_canonical_identity_and_matching_shapes() -> None:
    with pytest.raises(ValueError, match="robot_id"):
        RobotSnapshot.from_mapping(
            {
                "label": "robot",
                "joint_names": ["j0"],
                "joint_positions": [0.0],
                "joint_velocities": [0.0],
            }
        )
    with pytest.raises(ValueError, match="joint_velocities"):
        RobotSnapshot(
            label="robot",
            robot_id=0,
            joint_names=("j0", "j1"),
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(1),
        )


def test_object_snapshot_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        ObjectSnapshot(
            name="block",
            positions_local=np.zeros(3),
            orientations_wxyz=np.zeros(4),
        )


def test_object_snapshot_requires_body_pose_when_body_names_present() -> None:
    with pytest.raises(ValueError, match="body_positions_local"):
        ObjectSnapshot(
            name="rope",
            positions_local=np.zeros(3),
            orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            body_names=("body0",),
            body_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        )


def test_object_snapshot_generalized_fields_are_grouped_and_old_payload_stays_valid() -> (
    None
):
    with pytest.raises(ValueError, match="must be provided together"):
        ObjectSnapshot(
            name="rope",
            positions_local=np.zeros(3),
            orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            generalized_signature=("abi",),
            generalized_q_names=("q",),
            generalized_qd_names=("qd",),
            generalized_q=np.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="generalized_q.*finite"):
        ObjectSnapshot(
            name="rope",
            positions_local=np.zeros(3),
            orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            generalized_signature=("abi",),
            generalized_q_names=("q",),
            generalized_qd_names=("qd",),
            generalized_q=np.asarray([np.nan]),
            generalized_qd=np.asarray([0.0]),
        )

    old = ObjectSnapshot.from_mapping(
        {
            "name": "legacy_rope",
            "positions_local": [0.0, 0.0, 0.0],
            "orientations_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    )
    assert old.generalized_q is None
    assert "generalized_q" not in old.as_dict()


def test_compatibility_matches_label_and_allows_joint_reordering() -> None:
    snapshot = SceneSnapshot(
        robots={
            "robot_a": _robot(
                "robot_a",
                7,
                joint_names=("j0", "j1"),
                positions=(0.1, 0.2),
                robot_profile="arm",
                asset_fingerprint="asset",
            )
        }
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="mirror",
        robots={
            "robot_a": RobotTargetDescriptor(
                label="robot_a",
                robot_profile="arm",
                asset_fingerprint="asset",
                joint_names=("j1", "j0"),
            )
        },
    )

    result = require_snapshot_compatibility(snapshot, target)

    mapping = result.robot_mappings["robot_a"]
    assert mapping.source_label == "robot_a"
    assert mapping.target_label == "robot_a"
    assert mapping.joints.names == ("j1", "j0")
    np.testing.assert_array_equal(mapping.joints.source_indices, [1, 0])
    np.testing.assert_array_equal(mapping.joints.target_indices, [0, 1])


def test_non_strict_joint_subset_is_reported_as_partial() -> None:
    snapshot = SceneSnapshot(
        robots={
            "robot": RobotSnapshot(
                label="robot",
                robot_id=0,
                joint_names=("j0",),
                joint_positions=np.asarray([0.1]),
                joint_velocities=np.asarray([0.2]),
                command_joint_names=("j0",),
                command_targets=np.asarray([0.3]),
            )
        }
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="mirror",
        robots={
            "robot": RobotTargetDescriptor(
                label="robot",
                joint_names=("j0", "j1"),
                command_joint_names=("j0", "j1"),
            )
        },
    )

    result = require_snapshot_compatibility(snapshot, target, strict=False)

    assert result.partial is True
    assert result.robot_mappings["robot"].joints.names == ("j0",)
    assert result.robot_mappings["robot"].command_joints is not None


def test_compatibility_rejects_profile_mismatch() -> None:
    snapshot = SceneSnapshot(
        robots={"robot": _robot("robot", 0, robot_profile="arm-a")}
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="mirror",
        robots={
            "robot": RobotTargetDescriptor(
                label="robot", robot_profile="arm-b", joint_names=("j0",)
            )
        },
    )

    result = check_snapshot_compatibility(snapshot, target)

    assert not result.compatible
    assert "robot_profile mismatch" in result.issues[0]


def test_compatibility_requires_exact_label_or_explicit_label_map() -> None:
    snapshot = SceneSnapshot(robots={"source": _robot("source", 0)})
    target = SnapshotTargetDescriptor(
        runtime_kind="mirror",
        robots={"target": RobotTargetDescriptor(label="target", joint_names=("j0",))},
    )

    with pytest.raises(SnapshotCompatibilityError, match="provide label_map"):
        require_snapshot_compatibility(snapshot, target)

    result = require_snapshot_compatibility(
        snapshot,
        target,
        label_map={"source": "target"},
    )
    assert result.robot_mappings["target"].source_label == "source"


def test_compatibility_rejects_duplicate_label_map_targets() -> None:
    snapshot = SceneSnapshot(robots={"a": _robot("a", 0), "b": _robot("b", 1)})
    target = SnapshotTargetDescriptor(
        runtime_kind="mirror",
        robots={"target": RobotTargetDescriptor(label="target", joint_names=("j0",))},
    )

    with pytest.raises(SnapshotCompatibilityError, match="duplicated"):
        require_snapshot_compatibility(
            snapshot,
            target,
            label_map={"a": "target", "b": "target"},
        )


def test_compatibility_checks_dynamic_body_names() -> None:
    snapshot = SceneSnapshot(
        robots={"robot": _robot("robot", 0)},
        objects={
            "rope": ObjectSnapshot(
                name="rope",
                positions_local=np.zeros(3),
                orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
                body_names=("body0", "body1"),
                body_positions_local=np.zeros((2, 3)),
                body_orientations_wxyz=np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            )
        },
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="mirror",
        robots={"robot": RobotTargetDescriptor(label="robot", joint_names=("j0",))},
        objects={
            "rope": ObjectTargetDescriptor(name="rope", body_names=("body1", "body0")),
        },
    )

    result = require_snapshot_compatibility(snapshot, target)

    body_mapping = result.object_mappings["rope"].bodies
    assert body_mapping is not None
    assert body_mapping.names == ("body1", "body0")
    np.testing.assert_array_equal(body_mapping.source_indices, [1, 0])


def test_non_strict_body_subset_is_reported_as_partial() -> None:
    snapshot = SceneSnapshot(
        robots={},
        objects={
            "rope": ObjectSnapshot(
                name="rope",
                positions_local=np.zeros(3),
                orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
                body_names=("body1",),
                body_positions_local=np.asarray([[0.2, 0.0, 0.0]]),
                body_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            )
        },
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="mirror",
        robots={},
        objects={
            "rope": ObjectTargetDescriptor(
                name="rope",
                body_names=("body0", "body1"),
            )
        },
    )

    result = require_snapshot_compatibility(snapshot, target, strict=False)

    assert result.partial is True
    assert result.object_mappings["rope"].bodies is not None
    assert result.object_mappings["rope"].bodies.names == ("body1",)
