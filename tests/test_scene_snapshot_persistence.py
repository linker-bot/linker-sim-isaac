from __future__ import annotations

import json

import numpy as np
import pytest

from linkerbot_sim.snapshots import (
    SceneSnapshot,
    load_scene_snapshot,
    save_scene_snapshot,
    validate_scene_snapshot,
)
from linkerbot_sim.snapshots.schema import RobotSnapshot, SnapshotMetadata


def _snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="mirror",
            coordinate_frame="scene-local",
        ),
        robots={
            "arm": RobotSnapshot(
                label="arm",
                robot_id=0,
                joint_names=("j0", "j1"),
                joint_positions=np.asarray([0.1, 0.2]),
                joint_velocities=np.asarray([0.0, 0.0]),
            )
        },
    )


def test_scene_snapshot_facade_has_exact_stable_exports() -> None:
    import linkerbot_sim.snapshots as facade

    assert facade.__all__ == [
        "SceneSnapshot",
        "load_scene_snapshot",
        "save_scene_snapshot",
        "validate_scene_snapshot",
    ]


def test_scene_snapshot_atomic_save_load_and_replace_policy(tmp_path) -> None:
    path = tmp_path / "state" / "snapshot.json"
    saved = save_scene_snapshot(_snapshot(), path)
    loaded = load_scene_snapshot(path)

    assert saved == path
    assert loaded.metadata.source_runtime == "mirror"
    np.testing.assert_allclose(loaded.robots["arm"].joint_positions, [0.1, 0.2])
    with pytest.raises(FileExistsError):
        save_scene_snapshot(_snapshot(), path)
    save_scene_snapshot(_snapshot(), path, replace=True)


def test_validate_returns_owned_copy_and_rejects_bad_schema(tmp_path) -> None:
    original = _snapshot()
    validated = validate_scene_snapshot(original)
    original.robots["arm"].joint_positions[0] = 9.0

    assert validated.robots["arm"].joint_positions[0] == pytest.approx(0.1)
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"schema": "obsolete", "robots": {}, "objects": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported snapshot schema"):
        load_scene_snapshot(path)
