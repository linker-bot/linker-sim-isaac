from __future__ import annotations

import numpy as np

from linkerbot_sim.app.interactive.tiled.object_states import (
    TiledDynamicChainObjectPoseView,
    _restore_tiled_object_pose_snapshot,
)
from linkerbot_sim.snapshots import (
    get_tiled_snapshot,
    set_tiled_snapshot,
)
from tests.fakes.tiled_runtime_fake import (
    DebugBatchedIKSolver,
    DebugTiledInteractiveRuntime,
)


def make_debug_runtime() -> DebugTiledInteractiveRuntime:
    return DebugTiledInteractiveRuntime.create(
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
        ik_solver=DebugBatchedIKSolver(),
    )


def test_debug_tiled_snapshot_restores_selected_envs() -> None:
    runtime = make_debug_runtime()
    runtime.current_positions[:] = [
        [0.1, 0.2, 0.3],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    runtime.adapter.last_target = runtime.current_positions.copy()

    snapshot = get_tiled_snapshot(runtime, env_id=0)
    result = set_tiled_snapshot(runtime, snapshot, env_ids=[1, 2])

    assert result.accepted
    assert result.env_ids == (1, 2)
    assert result.robots == ("debug",)
    np.testing.assert_allclose(runtime.current_positions[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(runtime.current_positions[1], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(runtime.current_positions[2], [0.1, 0.2, 0.3])
    assert runtime.adapter.last_target is None


def test_debug_tiled_clone_state_uses_source_env_snapshot() -> None:
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
    )

    restored = _restore_tiled_object_pose_snapshot(
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


def test_runtime_get_snapshot_response_is_json_compatible() -> None:
    runtime = make_debug_runtime()
    runtime.current_positions[0, :] = [0.7, 0.8, 0.9]

    response = runtime.get_snapshot(env_id=0)

    assert response["event"] == "snapshot"
    assert response["accepted"] is True
    assert response["snapshot"]["robots"]["debug"]["joint_positions"] == [0.7, 0.8, 0.9]
