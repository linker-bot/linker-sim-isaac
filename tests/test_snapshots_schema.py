from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.snapshots import (
    ObjectSnapshot,
    ObjectTargetDescriptor,
    RobotSnapshot,
    RobotTargetDescriptor,
    SimulationSnapshot,
    SnapshotCompatibilityError,
    SnapshotMetadata,
    SnapshotTargetDescriptor,
    check_snapshot_compatibility,
    require_snapshot_compatibility,
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
    snapshot = SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="tiled_scene",
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
            )
        },
    )

    payload = snapshot.as_dict()
    restored = SimulationSnapshot.from_mapping(payload)

    assert payload["schema"] == "linkerbot.snapshot"
    assert isinstance(payload["robots"], list)
    assert payload["robots"][0]["robot_id"] == 3
    assert "role" not in payload["robots"][0]
    assert restored.metadata.source_runtime == "tiled_scene"
    assert restored.robots["workcell_robot"].robot_profile == "arm_hand"
    np.testing.assert_allclose(
        restored.robots["workcell_robot"].joint_positions, [0.1, 0.2]
    )
    np.testing.assert_allclose(
        restored.objects["block"].orientations_wxyz,
        [1.0, 0.0, 0.0, 0.0],
    )


def test_snapshot_reader_rejects_invalid_schema_shape_and_fields() -> None:
    with pytest.raises(ValueError, match="unsupported snapshot schema"):
        SimulationSnapshot.from_mapping({"schema": "another.snapshot", "robots": []})
    with pytest.raises(ValueError, match="snapshot.schema is required"):
        SimulationSnapshot.from_mapping({"robots": []})
    with pytest.raises(ValueError, match="robots must be an array"):
        SimulationSnapshot.from_mapping({"schema": "linkerbot.snapshot", "robots": {}})
    with pytest.raises(ValueError, match="unsupported fields"):
        SimulationSnapshot.from_mapping(
            {
                "schema": "linkerbot.snapshot",
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


def test_compatibility_matches_label_and_allows_joint_reordering() -> None:
    snapshot = SimulationSnapshot(
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
        runtime_kind="single_scene",
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


def test_compatibility_rejects_profile_mismatch() -> None:
    snapshot = SimulationSnapshot(
        robots={"robot": _robot("robot", 0, robot_profile="arm-a")}
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="single_scene",
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
    snapshot = SimulationSnapshot(robots={"source": _robot("source", 0)})
    target = SnapshotTargetDescriptor(
        runtime_kind="single_scene",
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
    snapshot = SimulationSnapshot(robots={"a": _robot("a", 0), "b": _robot("b", 1)})
    target = SnapshotTargetDescriptor(
        runtime_kind="single_scene",
        robots={"target": RobotTargetDescriptor(label="target", joint_names=("j0",))},
    )

    with pytest.raises(SnapshotCompatibilityError, match="duplicated"):
        require_snapshot_compatibility(
            snapshot,
            target,
            label_map={"a": "target", "b": "target"},
        )


def test_compatibility_checks_dynamic_body_names() -> None:
    snapshot = SimulationSnapshot(
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
        runtime_kind="single_scene",
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
