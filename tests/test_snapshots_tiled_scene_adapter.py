from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.tiled.state.object_io import (
    restore_tiled_object_pose_snapshot,
)
from linkerbot_sim.tiled.state.object_views import (
    TiledDynamicChainObjectPoseView,
)
from linkerbot_sim.snapshots import (
    get_single_scene_snapshot,
    get_snapshot,
    get_tiled_scene_snapshot,
    set_tiled_scene_snapshot,
)
from linkerbot_sim.snapshots.schema import SimulationSnapshot
from linkerbot_sim.snapshots.tiled_scene_adapter import (
    _object_snapshots_from_tiled_state,
    _tiled_restore_payload_from_snapshot,
)
from linkerbot_sim.snapshots.runtime_objects import (
    _asset_fingerprint_from_path,
    _imported_asset_fingerprint,
)
from tests.fakes.tiled_scene_runtime_fake import (
    DebugBatchIKBackend,
    DebugTiledSceneRuntime,
)


def make_debug_runtime() -> DebugTiledSceneRuntime:
    return DebugTiledSceneRuntime.create(
        env_name="unit",
        env_config={
            "env": {"physics_frequency": 100.0},
            "tiled": {
                "enabled": True,
                "num_envs": 3,
                "spacing": 2.0,
            },
        },
        command_dim=3,
        default_decimation=2,
        tcp_frame_name="tcp",
        ik_solver=DebugBatchIKBackend(),
    )


class _TiledMatrixView:
    def __init__(
        self,
        positions: np.ndarray,
        *,
        fail_position_calls: set[int] = frozenset(),
    ) -> None:
        self.positions = np.asarray(positions, dtype=float).copy()
        self.velocities = np.zeros_like(self.positions)
        self.position_calls = 0
        self.fail_position_calls = set(fail_position_calls)

    def get_joint_positions(self, *, indices=None, joint_indices=None):
        rows = np.asarray(indices, dtype=int)
        columns = (
            np.arange(self.positions.shape[1], dtype=int)
            if joint_indices is None
            else np.asarray(joint_indices, dtype=int)
        )
        return self.positions[np.ix_(rows, columns)].copy()

    def get_joint_velocities(self, *, indices=None, joint_indices=None):
        rows = np.asarray(indices, dtype=int)
        columns = (
            np.arange(self.velocities.shape[1], dtype=int)
            if joint_indices is None
            else np.asarray(joint_indices, dtype=int)
        )
        return self.velocities[np.ix_(rows, columns)].copy()

    def set_joint_positions(self, values, *, indices=None, joint_indices=None):
        self.position_calls += 1
        if self.position_calls in self.fail_position_calls:
            raise RuntimeError(f"position setter {self.position_calls} failed")
        rows = np.asarray(indices, dtype=int)
        columns = (
            np.arange(self.positions.shape[1], dtype=int)
            if joint_indices is None
            else np.asarray(joint_indices, dtype=int)
        )
        self.positions[np.ix_(rows, columns)] = np.asarray(values, dtype=float)

    def set_joint_velocities(self, values, *, indices=None, joint_indices=None):
        rows = np.asarray(indices, dtype=int)
        columns = (
            np.arange(self.velocities.shape[1], dtype=int)
            if joint_indices is None
            else np.asarray(joint_indices, dtype=int)
        )
        self.velocities[np.ix_(rows, columns)] = np.asarray(values, dtype=float)


class _TiledAdapterCache:
    def __init__(self, target: np.ndarray) -> None:
        self.last_target = np.asarray(target, dtype=float).copy()

    def reset(self) -> None:
        self.last_target = None


def _isaac_snapshot_runtime(
    views: dict[str, _TiledMatrixView],
) -> SimpleNamespace:
    robot_names = tuple(views)
    articulation_views = {
        name: SimpleNamespace(
            view=view,
            command_joint_indices=np.asarray([0, 1], dtype=int),
            command_joint_names=("j0", "j1"),
        )
        for name, view in views.items()
    }
    targets = {name: view.positions.copy() + 100.0 for name, view in views.items()}
    adapters = {
        name: _TiledAdapterCache(view.positions + 200.0) for name, view in views.items()
    }
    config = SimpleNamespace(
        num_envs=2,
        metadata_for_env=lambda env_id: {"env_id": int(env_id)},
    )
    runtime = SimpleNamespace(
        scene=SimpleNamespace(
            config=config,
            articulation_views=articulation_views,
            robots={},
            object_prim_paths={},
            object_handles=(),
            env_origins=np.zeros((2, 3), dtype=float),
        ),
        session=SimpleNamespace(stage=object()),
        robot_names=robot_names,
        target_positions=targets,
        object_pose_views={},
        tcp_positions_world={
            name: np.full((2, 3), index + 1.0, dtype=float)
            for index, name in enumerate(robot_names)
        },
        tcp_orientations_wxyz={
            name: np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)) for name in robot_names
        },
        trajectory_buffer=SimpleNamespace(clear=lambda **_kwargs: None),
        planner_manager=SimpleNamespace(cancel_matching=lambda **_kwargs: None),
        quit_event=threading.Event(),
        step=0,
        time_s=0.0,
        _command_adapter=adapters.__getitem__,
        _refresh_tcp_state=lambda *_args, **_kwargs: None,
    )
    runtime.adapters = adapters
    return runtime


def test_asset_fingerprint_is_runtime_independent(tmp_path) -> None:
    asset = tmp_path / "robot.urdf"
    asset.write_text("<robot name='test'/>", encoding="utf-8")

    from_scene_path = _imported_asset_fingerprint(SimpleNamespace(asset_path=asset))
    from_tiled_path = _asset_fingerprint_from_path(str(asset))

    assert from_scene_path == from_tiled_path


def test_debug_tiled_scene_snapshot_restores_selected_envs() -> None:
    runtime = make_debug_runtime()
    runtime.current_positions[:] = [
        [0.1, 0.2, 0.3],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    runtime.adapter.last_target = runtime.current_positions.copy()

    snapshot = get_tiled_scene_snapshot(runtime, env_id=0)
    result = set_tiled_scene_snapshot(runtime, snapshot, env_ids=[1, 2])

    assert snapshot.metadata.source_runtime == "tiled_scene_debug"
    assert result.accepted
    assert result.env_ids == (1, 2)
    assert result.robots == ("debug",)
    np.testing.assert_allclose(runtime.current_positions[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(runtime.current_positions[1], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(runtime.current_positions[2], [0.1, 0.2, 0.3])
    assert runtime.adapter.last_target is None


def test_debug_tiled_scene_clone_state_uses_source_env_snapshot() -> None:
    runtime = make_debug_runtime()
    runtime.current_positions[:] = [
        [0.0, 0.0, 0.0],
        [0.4, 0.5, 0.6],
        [9.0, 9.0, 9.0],
    ]

    response = runtime.clone_state(
        source_env_id=1,
        target_env_ids=np.asarray([2], dtype=int),
    )

    assert response["event"] == "state_cloned"
    assert response["accepted"] is True
    assert response["source_env_id"] == 1
    assert response["target_env_ids"] == [2]
    np.testing.assert_allclose(runtime.current_positions[2], [0.4, 0.5, 0.6])


def test_isaac_tiled_snapshot_rolls_back_distinct_env_state_and_caches() -> None:
    views = {
        "left": _TiledMatrixView([[0.1, 0.2], [1.1, 1.2]]),
        "right": _TiledMatrixView(
            [[2.1, 2.2], [3.1, 3.2]],
            fail_position_calls={1},
        ),
    }
    runtime = _isaac_snapshot_runtime(views)
    snapshot = get_tiled_scene_snapshot(runtime, env_id=0)
    original_positions = {name: view.positions.copy() for name, view in views.items()}
    original_targets = {
        name: values.copy() for name, values in runtime.target_positions.items()
    }
    original_adapter_targets = {
        name: adapter.last_target.copy() for name, adapter in runtime.adapters.items()
    }

    with pytest.raises(RuntimeError, match="position setter 1 failed"):
        set_tiled_scene_snapshot(runtime, snapshot, env_ids=[1])

    for name, view in views.items():
        np.testing.assert_allclose(view.positions, original_positions[name])
        np.testing.assert_allclose(
            runtime.target_positions[name], original_targets[name]
        )
        np.testing.assert_allclose(
            runtime.adapters[name].last_target,
            original_adapter_targets[name],
        )
    assert not hasattr(runtime, "fatal_error")
    assert not runtime.quit_event.is_set()

    result = set_tiled_scene_snapshot(runtime, snapshot, env_ids=[1])
    assert result.accepted
    for name, view in views.items():
        np.testing.assert_allclose(view.positions[1], original_positions[name][0])
        # command_targets, not physical q, are the canonical post-restore target.
        np.testing.assert_allclose(
            runtime.target_positions[name][1],
            original_targets[name][0],
        )


def test_dynamic_chain_restore_accepts_local_body_snapshot() -> None:
    calls = []

    class FakeRigidBodyView:
        def set_world_poses(self, *, positions, orientations, indices):
            calls.append(
                (
                    "poses",
                    np.asarray(positions, dtype=float).tolist(),
                    np.asarray(orientations, dtype=float).tolist(),
                    np.asarray(indices, dtype=int).tolist(),
                )
            )

        def set_velocities(self, velocities, *, indices):
            calls.append(
                (
                    "velocities",
                    np.asarray(velocities, dtype=float).tolist(),
                    np.asarray(indices, dtype=int).tolist(),
                )
            )

    object_view = TiledDynamicChainObjectPoseView(
        view=FakeRigidBodyView(),
        body_names=("body0", "body1"),
        body_paths_by_env=(
            ("/World/envs/env_0/Rope/body0", "/World/envs/env_0/Rope/body1"),
            ("/World/envs/env_1/Rope/body0", "/World/envs/env_1/Rope/body1"),
        ),
        reference_body="body0",
    )

    restored = restore_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths={
            "rope": ("/World/envs/env_0/Rope", "/World/envs/env_1/Rope")
        },
        snapshot={
            "rope": {
                "env_ids": np.asarray([1], dtype=int),
                "positions_local": np.asarray([[0.1, 0.0, -0.1]], dtype=float),
                "orientations_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
                "body_names": ("body0", "body1"),
                "body_positions_local": np.asarray(
                    [[[0.0, 0.0, -0.1], [0.2, 0.1, -0.1]]],
                    dtype=float,
                ),
                "body_orientations_wxyz": np.asarray(
                    [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
                    dtype=float,
                ),
            }
        },
        env_ids=np.asarray([1], dtype=int),
        env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        object_pose_views={"rope": object_view},
    )

    assert restored == 1
    assert calls == [
        (
            "poses",
            [[3.0, 0.0, -0.1], [3.2, 0.1, -0.1]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [2, 3],
        ),
        (
            "velocities",
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [2, 3],
        ),
    ]


def test_tiled_object_velocity_fields_round_trip_through_snapshot_adapter() -> None:
    objects = _object_snapshots_from_tiled_state(
        {
            "rope": {
                "positions_local": [[0.1, 0.2, 0.3]],
                "orientations_wxyz": [[1.0, 0.0, 0.0, 0.0]],
                "linear_velocities": [[1.0, 2.0, 3.0]],
                "angular_velocities": [[4.0, 5.0, 6.0]],
                "body_names": ["body0", "body1"],
                "body_positions_local": [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]],
                "body_orientations_wxyz": [
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                ],
                "body_linear_velocities": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                "body_angular_velocities": [[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
            }
        },
        object_profiles={"rope": "test_rope"},
    )
    rope = objects["rope"]

    np.testing.assert_allclose(rope.linear_velocities, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rope.angular_velocities, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(
        rope.body_linear_velocities,
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    np.testing.assert_allclose(
        rope.body_angular_velocities,
        [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
    )

    payload = _tiled_restore_payload_from_snapshot(
        SimulationSnapshot(robots={}, objects=objects),
        env_ids=np.asarray([1, 2], dtype=int),
    )["rope"]

    np.testing.assert_allclose(
        payload["linear_velocities"],
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
    )
    np.testing.assert_allclose(
        payload["angular_velocities"],
        [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
    )
    np.testing.assert_allclose(
        payload["body_linear_velocities"],
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ],
    )
    np.testing.assert_allclose(
        payload["body_angular_velocities"],
        [
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ],
    )


def test_runtime_get_snapshot_response_is_json_compatible() -> None:
    runtime = make_debug_runtime()
    runtime.current_positions[0, :] = [0.7, 0.8, 0.9]

    response = runtime.get_snapshot(env_id=0)

    assert response["event"] == "snapshot"
    assert response["accepted"] is True
    assert response["snapshot"]["robots"][0]["label"] == "debug"
    assert response["snapshot"]["robots"][0]["joint_positions"] == [0.7, 0.8, 0.9]


def test_tiled_snapshot_exposes_per_env_metadata() -> None:
    runtime = DebugTiledSceneRuntime.create(
        env_name="unit",
        env_config={
            "env": {"physics_frequency": 100.0},
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "per_env": [{"env_id": 1, "metadata": {"replay_id": "case_001"}}],
            },
        },
        command_dim=3,
        default_decimation=2,
        tcp_frame_name="tcp",
        ik_solver=DebugBatchIKBackend(),
    )

    snapshot = get_tiled_scene_snapshot(runtime, env_id=1)

    assert snapshot.metadata.info == {"per_env": {"replay_id": "case_001"}}


def test_snapshot_adapters_use_domain_owned_module_paths() -> None:
    assert get_snapshot.__module__ == "linkerbot_sim.snapshots.dispatch"
    assert (
        get_single_scene_snapshot.__module__
        == "linkerbot_sim.snapshots.single_scene_adapter"
    )
    assert (
        get_tiled_scene_snapshot.__module__
        == "linkerbot_sim.snapshots.tiled_scene_adapter"
    )


def test_snapshots_domain_does_not_depend_on_application_layer() -> None:
    package = Path(__file__).parents[1] / "src" / "linkerbot_sim" / "snapshots"
    offenders = {
        path.name
        for path in package.glob("*.py")
        if "linkerbot_sim.app" in path.read_text(encoding="utf-8")
    }

    assert offenders == set()
