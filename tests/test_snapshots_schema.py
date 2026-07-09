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


def test_snapshot_round_trips_json_mapping() -> None:
    snapshot = SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="tiled",
            source_env_id=0,
            step=12,
            time_s=0.5,
        ),
        robots={
            "left": RobotSnapshot(
                role="left",
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
                    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                    dtype=float,
                ),
                body_orientations_wxyz=np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            )
        },
    )

    restored = SimulationSnapshot.from_mapping(snapshot.as_dict())

    assert restored.metadata.source_runtime == "tiled"
    assert restored.metadata.source_env_id == 0
    assert restored.robots["left"].robot_profile == "arm_hand"
    np.testing.assert_allclose(restored.robots["left"].joint_positions, [0.1, 0.2])
    np.testing.assert_allclose(
        restored.objects["block"].orientations_wxyz,
        [1.0, 0.0, 0.0, 0.0],
    )


def test_robot_snapshot_rejects_mismatched_velocity_shape() -> None:
    with pytest.raises(ValueError, match="joint_velocities"):
        RobotSnapshot(
            role="single",
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


def test_compatibility_allows_joint_reordering() -> None:
    snapshot = SimulationSnapshot(
        robots={
            "left": RobotSnapshot(
                role="left",
                robot_profile="arm",
                asset_fingerprint="asset",
                joint_names=("j0", "j1"),
                joint_positions=np.asarray([0.1, 0.2]),
                joint_velocities=np.zeros(2),
            )
        }
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="dual",
        robots={
            "left": RobotTargetDescriptor(
                role="left",
                robot_profile="arm",
                asset_fingerprint="asset",
                joint_names=("j1", "j0"),
            )
        },
    )

    result = require_snapshot_compatibility(snapshot, target)

    mapping = result.robot_mappings["left"].joints
    assert mapping.names == ("j1", "j0")
    np.testing.assert_array_equal(mapping.source_indices, [1, 0])
    np.testing.assert_array_equal(mapping.target_indices, [0, 1])


def test_compatibility_rejects_profile_mismatch() -> None:
    snapshot = SimulationSnapshot(
        robots={
            "single": RobotSnapshot(
                role="single",
                robot_profile="arm-a",
                joint_names=("j0",),
                joint_positions=np.asarray([0.1]),
                joint_velocities=np.zeros(1),
            )
        }
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="single",
        robots={
            "single": RobotTargetDescriptor(
                role="single",
                robot_profile="arm-b",
                joint_names=("j0",),
            )
        },
    )

    result = check_snapshot_compatibility(snapshot, target)

    assert not result.compatible
    assert "robot_profile mismatch" in result.issues[0]


def test_compatibility_requires_robot_map_for_ambiguous_roles() -> None:
    snapshot = SimulationSnapshot(
        robots={
            "left": RobotSnapshot(
                role="left",
                joint_names=("j0",),
                joint_positions=np.asarray([0.1]),
                joint_velocities=np.zeros(1),
            ),
            "right": RobotSnapshot(
                role="right",
                joint_names=("j0",),
                joint_positions=np.asarray([0.2]),
                joint_velocities=np.zeros(1),
            ),
        }
    )
    target = SnapshotTargetDescriptor(
        runtime_kind="single",
        robots={
            "single": RobotTargetDescriptor(role="single", joint_names=("j0",)),
        },
    )

    with pytest.raises(SnapshotCompatibilityError, match="robot_map is required"):
        require_snapshot_compatibility(snapshot, target)


def test_compatibility_checks_dynamic_body_names() -> None:
    snapshot = SimulationSnapshot(
        robots={
            "single": RobotSnapshot(
                role="single",
                joint_names=("j0",),
                joint_positions=np.asarray([0.0]),
                joint_velocities=np.zeros(1),
            )
        },
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
        runtime_kind="single",
        robots={
            "single": RobotTargetDescriptor(role="single", joint_names=("j0",)),
        },
        objects={
            "rope": ObjectTargetDescriptor(name="rope", body_names=("body1", "body0")),
        },
    )

    result = require_snapshot_compatibility(snapshot, target)

    body_mapping = result.object_mappings["rope"].bodies
    assert body_mapping is not None
    assert body_mapping.names == ("body1", "body0")
    np.testing.assert_array_equal(body_mapping.source_indices, [1, 0])
