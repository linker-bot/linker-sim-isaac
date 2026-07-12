from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.objects.state_views import SceneObjectStateView
from linkerbot_sim.snapshots import runtime_objects
from linkerbot_sim.tiled.state.object_io import (
    capture_tiled_object_pose_snapshot,
    restore_tiled_object_pose_snapshot,
)
from linkerbot_sim.tiled.state.object_views import TiledDynamicChainObjectPoseView


class _FakeRigidView:
    def __init__(self, positions, orientations, velocities) -> None:
        self.positions = np.asarray(positions, dtype=float)
        self.orientations = np.asarray(orientations, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.writes: list[tuple[str, np.ndarray, np.ndarray]] = []

    def get_world_poses(self, *, indices):
        return self.positions[indices], self.orientations[indices]

    def get_velocities(self, *, indices):
        return self.velocities[indices]

    def set_world_poses(self, *, positions, orientations, indices):
        self.writes.append(("pose", np.asarray(indices), np.asarray(positions)))

    def set_velocities(self, velocities, *, indices):
        self.writes.append(("velocity", np.asarray(indices), np.asarray(velocities)))


def test_scene_live_rigid_view_uses_canonical_rad_per_second(monkeypatch) -> None:
    view = _FakeRigidView(
        positions=[[0.0, 0.0, 0.0]],
        orientations=[[1.0, 0.0, 0.0, 0.0]],
        velocities=[[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
    )
    state_view = SceneObjectStateView(root_view=view)
    handle = SimpleNamespace(
        runtime_handle="block",
        kind="rigid",
        config=SimpleNamespace(object_profile="block"),
        model=SimpleNamespace(prim_path="/block"),
    )
    monkeypatch.setattr(
        runtime_objects,
        "read_prim_world_pose",
        lambda _stage, _path: (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(
        runtime_objects,
        "apply_prim_local_pose_and_zero_velocity",
        lambda *_args: True,
    )

    objects = runtime_objects._runtime_object_snapshots(
        stage=object(),
        handles=(handle,),
        state_views={"block": state_view},
    )
    np.testing.assert_allclose(objects["block"].angular_velocities, [0.1, 0.2, 0.3])

    mapping = SimpleNamespace(source_name="block", bodies=None)
    runtime_objects._restore_runtime_objects(
        SimpleNamespace(
            session=SimpleNamespace(stage=object()),
            object_handles=(handle,),
            object_state_views={"block": state_view},
        ),
        SimpleNamespace(objects=objects),
        compatibility=SimpleNamespace(object_mappings={"block": mapping}),
    )
    assert view.writes[-1][0] == "velocity"
    np.testing.assert_allclose(view.writes[-1][2], [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])


def test_scene_live_rigid_view_rejects_missing_velocity(monkeypatch) -> None:
    view = _FakeRigidView(
        positions=[[0.0, 0.0, 0.0]],
        orientations=[[1.0, 0.0, 0.0, 0.0]],
        velocities=[[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]],
    )
    monkeypatch.setattr(
        runtime_objects,
        "apply_prim_local_pose_and_zero_velocity",
        lambda *_args: True,
    )

    with pytest.raises(ValueError, match="missing required velocity state"):
        runtime_objects._apply_prim_local_pose_and_velocity(
            object(),
            "/block",
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            None,
            None,
            state_view=SceneObjectStateView(root_view=view),
        )

    assert view.writes == []


def test_scene_usd_fallback_converts_angular_velocity_units(monkeypatch) -> None:
    writes: dict[str, np.ndarray] = {}

    class _Attr:
        def __init__(self, name: str, value) -> None:
            self.name = name
            self.value = value

        def Get(self):
            return self.value

        def Set(self, value) -> None:
            writes[self.name] = np.asarray(value, dtype=float)

        def IsValid(self) -> bool:
            return True

    linear_attr = _Attr("linear", [1.0, 2.0, 3.0])
    angular_attr = _Attr("angular", [180.0, 90.0, 45.0])
    api = SimpleNamespace(
        GetVelocityAttr=lambda: linear_attr,
        GetAngularVelocityAttr=lambda: angular_attr,
    )
    prim = SimpleNamespace(IsValid=lambda: True, HasAPI=lambda _api: True)
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: prim)

    def rigid_body_api(_prim):
        return api

    pxr = ModuleType("pxr")
    pxr.Sdf = SimpleNamespace(Path=lambda value: value)
    pxr.UsdPhysics = SimpleNamespace(RigidBodyAPI=rigid_body_api)
    pxr.Gf = SimpleNamespace(Vec3f=lambda x, y, z: (x, y, z))
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    linear, angular = runtime_objects._read_prim_rigid_body_velocities(stage, "/x")
    np.testing.assert_allclose(linear, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(angular, [np.pi, np.pi / 2.0, np.pi / 4.0])

    monkeypatch.setattr(
        runtime_objects,
        "apply_prim_local_pose_and_zero_velocity",
        lambda *_args: True,
    )
    runtime_objects._apply_prim_local_pose_and_velocity(
        stage,
        "/x",
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([4.0, 5.0, 6.0]),
        np.asarray([np.pi, np.pi / 2.0, np.pi / 4.0]),
    )

    np.testing.assert_allclose(writes["linear"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(writes["angular"], [180.0, 90.0, 45.0])


def test_scene_unsupported_velocity_view_is_explicit() -> None:
    view = SceneObjectStateView(
        velocity_capability="unsupported",
        velocity_error="physics handle unavailable",
    )

    with np.testing.assert_raises_regex(RuntimeError, "physics handle unavailable"):
        view.require_velocity_support(object_name="rope")


def test_tiled_rigid_object_velocity_capture_restore_round_trip() -> None:
    view = _FakeRigidView(
        positions=[[0.1, 0.0, 0.0], [2.2, 0.0, 0.0]],
        orientations=[[1.0, 0.0, 0.0, 0.0]] * 2,
        velocities=[
            [0.1, 0.2, 0.3, 1.0, 1.1, 1.2],
            [0.4, 0.5, 0.6, 1.3, 1.4, 1.5],
        ],
    )
    paths = {"block": ("/env0/block", "/env1/block")}
    origins = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    snapshot = capture_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths=paths,
        env_origins=origins,
        env_ids=np.asarray([0, 1]),
        object_pose_views={"block": view},
    )
    np.testing.assert_allclose(
        snapshot["block"]["linear_velocities"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    np.testing.assert_allclose(
        snapshot["block"]["angular_velocities"], [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]
    )

    restore_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths=paths,
        snapshot=snapshot,
        env_ids=np.asarray([1]),
        env_origins=origins,
        object_pose_views={"block": view},
    )
    assert view.writes[-1][0] == "velocity"
    np.testing.assert_array_equal(view.writes[-1][1], [1])
    np.testing.assert_allclose(view.writes[-1][2], [[0.4, 0.5, 0.6, 1.3, 1.4, 1.5]])


def test_tiled_dynamic_chain_body_velocity_capture_restore_round_trip() -> None:
    view = _FakeRigidView(
        positions=[[0, 0, 0], [0.2, 0, 0], [2, 0, 0], [2.2, 0, 0]],
        orientations=[[1.0, 0.0, 0.0, 0.0]] * 4,
        velocities=[
            [0.1, 0, 0, 1.1, 0, 0],
            [0.3, 0, 0, 1.3, 0, 0],
            [0.5, 0, 0, 1.5, 0, 0],
            [0.7, 0, 0, 1.7, 0, 0],
        ],
    )
    wrapper = TiledDynamicChainObjectPoseView(
        view=view,
        body_names=("a", "b"),
        body_paths_by_env=(("/env0/a", "/env0/b"), ("/env1/a", "/env1/b")),
        reference_body="a",
    )
    paths = {"rope": ("/env0/rope", "/env1/rope")}
    origins = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    snapshot = capture_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths=paths,
        env_origins=origins,
        env_ids=np.asarray([0, 1]),
        object_pose_views={"rope": wrapper},
    )
    np.testing.assert_allclose(
        snapshot["rope"]["body_linear_velocities"][:, :, 0],
        [[0.1, 0.3], [0.5, 0.7]],
    )
    np.testing.assert_allclose(snapshot["rope"]["linear_velocities"][:, 0], [0.1, 0.5])

    restore_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths=paths,
        snapshot=snapshot,
        env_ids=np.asarray([1]),
        env_origins=origins,
        object_pose_views={"rope": wrapper},
    )
    np.testing.assert_array_equal(view.writes[-1][1], [2, 3])
    np.testing.assert_allclose(view.writes[-1][2], view.velocities[2:4])


def test_tiled_dynamic_chain_reference_body_drives_coherent_state_summary() -> None:
    view = _FakeRigidView(
        positions=[[0, 0, 0], [0.2, 0.1, 0], [2, 0, 0], [2.3, 0.2, 0]],
        orientations=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        velocities=[
            [0.1, 0, 0, 1.1, 0, 0],
            [0.3, 0.1, 0, 1.3, 0.1, 0],
            [0.5, 0, 0, 1.5, 0, 0],
            [0.7, 0.2, 0, 1.7, 0.2, 0],
        ],
    )
    wrapper = TiledDynamicChainObjectPoseView(
        view=view,
        body_names=("a", "b"),
        body_paths_by_env=(("/env0/a", "/env0/b"), ("/env1/a", "/env1/b")),
        reference_body="b",
    )

    state = wrapper.read_object_state(
        object_name="rope",
        env_ids=np.asarray([0, 1]),
        env_origins=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(state["positions_world"], [[0.2, 0.1, 0], [2.3, 0.2, 0]])
    np.testing.assert_allclose(state["positions_local"], [[0.2, 0.1, 0], [0.3, 0.2, 0]])
    np.testing.assert_allclose(
        state["orientations_wxyz"],
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    )
    np.testing.assert_allclose(
        state["linear_velocities"], [[0.3, 0.1, 0], [0.7, 0.2, 0]]
    )
    np.testing.assert_allclose(
        state["angular_velocities"], [[1.3, 0.1, 0], [1.7, 0.2, 0]]
    )
    np.testing.assert_allclose(
        state["body_positions_world"], view.positions.reshape(2, 2, 3)
    )


def test_tiled_dynamic_chain_reference_body_must_exist_exactly_once() -> None:
    view = _FakeRigidView(
        positions=[[0, 0, 0]],
        orientations=[[1.0, 0.0, 0.0, 0.0]],
        velocities=[[0, 0, 0, 0, 0, 0]],
    )

    with np.testing.assert_raises_regex(ValueError, "identify exactly one body"):
        TiledDynamicChainObjectPoseView(
            view=view,
            body_names=("a",),
            body_paths_by_env=(("/env0/a",),),
            reference_body="missing",
        )


def test_scene_root_and_body_velocity_capture_restore_round_trip(monkeypatch) -> None:
    poses = {
        "/rope": (np.asarray([0.1, 0.2, 0.3]), np.asarray([1.0, 0.0, 0.0, 0.0])),
        "/rope/a": (np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0, 0.0])),
        "/rope/b": (np.asarray([0.2, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0, 0.0])),
    }
    velocities = {
        "/rope": (np.asarray([1.0, 2.0, 3.0]), np.asarray([4.0, 5.0, 6.0])),
        "/rope/a": (np.asarray([0.1, 0.2, 0.3]), np.asarray([1.1, 1.2, 1.3])),
        "/rope/b": (np.asarray([0.4, 0.5, 0.6]), np.asarray([1.4, 1.5, 1.6])),
    }
    writes = []
    monkeypatch.setattr(
        runtime_objects, "read_prim_world_pose", lambda stage, path: poses[path]
    )
    monkeypatch.setattr(
        runtime_objects,
        "_read_prim_rigid_body_velocities",
        lambda stage, path: velocities[path],
    )
    monkeypatch.setattr(
        runtime_objects,
        "_apply_prim_local_pose_and_velocity",
        lambda stage, path, position, orientation, linear, angular, **_kwargs: (
            writes.append((path, linear, angular)) or True
        ),
    )
    handle = SimpleNamespace(
        runtime_handle="rope",
        kind="dynamic_chain",
        config=SimpleNamespace(object_profile="rope"),
        model={"prim_path": "/rope", "bodies": ("/rope/a", "/rope/b")},
    )

    objects = runtime_objects._runtime_object_snapshots(
        stage=object(), handles=(handle,)
    )
    np.testing.assert_allclose(objects["rope"].linear_velocities, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        objects["rope"].body_angular_velocities,
        [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
    )

    mapping = SimpleNamespace(
        source_name="rope",
        bodies=SimpleNamespace(source_indices=np.asarray([0, 1]), names=("a", "b")),
    )
    runtime_objects._restore_runtime_objects(
        SimpleNamespace(
            session=SimpleNamespace(stage=object()), object_handles=(handle,)
        ),
        SimpleNamespace(objects=objects),
        compatibility=SimpleNamespace(object_mappings={"rope": mapping}),
    )
    assert [item[0] for item in writes] == ["/rope", "/rope/a", "/rope/b"]
    np.testing.assert_allclose(writes[0][1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(writes[0][2], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(writes[2][2], [1.4, 1.5, 1.6])


def test_scene_static_object_snapshot_restores_pose_without_velocity(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        runtime_objects,
        "apply_prim_local_pose_and_zero_velocity",
        lambda *args: calls.append(args) or True,
    )

    assert runtime_objects._apply_prim_local_pose_and_velocity(
        object(),
        "/block",
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        None,
        None,
    )
    assert len(calls) == 1
