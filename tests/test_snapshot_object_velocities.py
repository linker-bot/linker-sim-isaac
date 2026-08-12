from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.objects.state_views import (
    SceneObjectStateView,
    create_scene_object_state_views,
)
import linkerbot_sim.snapshots.runtime_objects as runtime_objects
from linkerbot_sim.snapshots.compatibility import (
    ObjectTargetDescriptor,
    SnapshotTargetDescriptor,
    require_snapshot_compatibility,
)
from linkerbot_sim.snapshots.schema import ObjectSnapshot, SceneSnapshot


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
            np.asarray([9.0, 9.0, 9.0]),
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
    np.testing.assert_allclose(objects["block"].positions_local, [0.0, 0.0, 0.0])
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
    assert [write[0] for write in view.writes] == ["pose", "velocity"]
    np.testing.assert_allclose(view.writes[-1][2], [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])


def test_scene_dynamic_chain_live_reference_body_drives_root_and_body_state(
    monkeypatch,
) -> None:
    view = _FakeRigidView(
        positions=[[0.1, 0.0, 0.0], [0.4, 0.2, 0.0]],
        orientations=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        velocities=[
            [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            [0.4, 0.5, 0.6, 1.4, 1.5, 1.6],
        ],
    )
    state_view = SceneObjectStateView(
        body_view=view,
        body_names=("a", "b"),
        reference_body="b",
    )
    handle = SimpleNamespace(
        runtime_handle="rope",
        kind="dynamic_chain",
        config=SimpleNamespace(object_profile="rope"),
        model={"prim_path": "/rope", "bodies": ("/rope/a", "/rope/b")},
    )
    monkeypatch.setattr(
        runtime_objects,
        "read_prim_world_pose",
        lambda *_args: pytest.fail("dynamic pose must not be read from USD"),
    )
    monkeypatch.setattr(
        runtime_objects,
        "apply_prim_local_pose_and_zero_velocity",
        lambda *_args: pytest.fail("dynamic pose must not be restored through USD"),
    )

    objects = runtime_objects._runtime_object_snapshots(
        stage=object(),
        handles=(handle,),
        state_views={"rope": state_view},
    )

    rope = objects["rope"]
    np.testing.assert_allclose(rope.positions_local, [0.4, 0.2, 0.0])
    np.testing.assert_allclose(rope.orientations_wxyz, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rope.linear_velocities, [0.4, 0.5, 0.6])
    np.testing.assert_allclose(rope.body_positions_local, view.positions)
    np.testing.assert_allclose(rope.body_orientations_wxyz, view.orientations)

    mapping = SimpleNamespace(
        source_name="rope",
        bodies=SimpleNamespace(source_indices=np.asarray([0, 1]), names=("a", "b")),
    )
    restored = runtime_objects._restore_runtime_objects(
        SimpleNamespace(
            session=SimpleNamespace(stage=object()),
            object_handles=(handle,),
            object_state_views={"rope": state_view},
        ),
        SimpleNamespace(objects=objects),
        compatibility=SimpleNamespace(object_mappings={"rope": mapping}),
    )

    assert restored == ("rope",)
    assert [write[0] for write in view.writes] == ["pose", "velocity"]
    np.testing.assert_array_equal(view.writes[0][1], [0, 1])
    np.testing.assert_array_equal(view.writes[1][1], [0, 1])
    np.testing.assert_allclose(view.writes[0][2], rope.body_positions_local)


def test_scene_newton_non_strict_partial_chain_preserves_unmapped_body_state() -> None:
    class _AtomicRigidView(_FakeRigidView):
        def __init__(self) -> None:
            super().__init__(
                positions=[[0.1, 0.0, 0.0], [0.4, 0.2, 0.0]],
                orientations=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                velocities=[
                    [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
                    [0.4, 0.5, 0.6, 1.4, 1.5, 1.6],
                ],
            )
            self.atomic_writes: list[dict[str, np.ndarray]] = []

        def set_articulated_body_states(self, **kwargs) -> None:
            self.atomic_writes.append(
                {key: np.asarray(value).copy() for key, value in kwargs.items()}
            )

    view = _AtomicRigidView()
    state_view = SceneObjectStateView(
        body_view=view,
        body_names=("a", "b"),
        reference_body="a",
    )
    handle = SimpleNamespace(
        runtime_handle="rope",
        kind="dynamic_chain",
        config=SimpleNamespace(object_profile="rope"),
        model={"prim_path": "/rope", "bodies": ("/rope/a", "/rope/b")},
    )
    source = ObjectSnapshot(
        name="rope",
        object_profile="rope",
        positions_local=np.asarray([9.0, 9.0, 9.0]),
        orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        body_names=("b",),
        body_positions_local=np.asarray([[8.0, 8.1, 8.2]]),
        body_orientations_wxyz=np.asarray([[0.0, 1.0, 0.0, 0.0]]),
        body_linear_velocities=np.asarray([[2.0, 3.0, 4.0]]),
        body_angular_velocities=np.asarray([[5.0, 6.0, 7.0]]),
    )
    snapshot = SceneSnapshot(robots={}, objects={"rope": source})
    compatibility = require_snapshot_compatibility(
        snapshot,
        SnapshotTargetDescriptor(
            runtime_kind="mirror",
            robots={},
            objects={
                "rope": ObjectTargetDescriptor(
                    name="rope",
                    object_profile="rope",
                    body_names=("a", "b"),
                )
            },
        ),
        strict=False,
    )

    restored = runtime_objects._restore_runtime_objects(
        SimpleNamespace(
            session=SimpleNamespace(stage=object()),
            object_handles=(handle,),
            object_state_views={"rope": state_view},
        ),
        snapshot,
        compatibility=compatibility,
    )

    assert restored == ("rope",)
    assert view.writes == []
    assert len(view.atomic_writes) == 1
    write = view.atomic_writes[0]
    np.testing.assert_array_equal(write["indices"], [0, 1])
    np.testing.assert_allclose(write["positions"], [[0.1, 0.0, 0.0], [8.0, 8.1, 8.2]])
    np.testing.assert_allclose(
        write["velocities"],
        [[0.1, 0.2, 0.3, 1.1, 1.2, 1.3], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
    )


def test_scene_newton_partial_chain_fails_before_write_without_current_state() -> None:
    class _IncompleteAtomicView:
        def __init__(self) -> None:
            self.atomic_writes = 0

        def get_world_poses(self, *, indices):
            return np.zeros((2, 3)), np.zeros((2, 4))

        def set_articulated_body_states(self, **_kwargs) -> None:
            self.atomic_writes += 1

    view = _IncompleteAtomicView()
    state_view = SceneObjectStateView(
        body_view=view,
        body_names=("a", "b"),
        reference_body="a",
    )

    with pytest.raises(RuntimeError, match="complete current body state"):
        state_view.set_body_states(
            body_indices=np.asarray([1]),
            positions=np.asarray([[1.0, 2.0, 3.0]]),
            orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            linear_velocities=np.asarray([[4.0, 5.0, 6.0]]),
            angular_velocities=np.asarray([[7.0, 8.0, 9.0]]),
        )

    assert view.atomic_writes == 0


def test_newton_scene_dynamic_chain_view_creation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linkerbot_sim.objects.state_views as state_views_module

    handle = SimpleNamespace(
        runtime_handle="rope",
        name="rope",
        kind="dynamic_chain",
        config=SimpleNamespace(object_profile="rope", prim_path="/World/rope"),
        model={
            "root": "/World/rope",
            "bodies": (
                "/World/rope/Bodies/a",
                "/World/rope/Bodies/b",
            ),
        },
        state_summary=SimpleNamespace(reference_body="a"),
    )
    monkeypatch.setattr(
        state_views_module,
        "_create_dynamic_chain_or_rigid_view",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("binding failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="required Newton dynamic-chain state view",
    ):
        create_scene_object_state_views(
            (handle,),
            physics_backend="newton",
            immutable_static=True,
        )


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


def test_direct_single_static_object_snapshot_is_immutable(monkeypatch) -> None:
    stage = object()
    handle = SimpleNamespace(
        runtime_handle="workstation",
        name="workstation",
        kind="rigid",
        config=SimpleNamespace(object_profile="workstation"),
        model=SimpleNamespace(
            prim_path="/World/Workstation",
            static=True,
        ),
    )
    monkeypatch.setattr(
        "linkerbot_sim.isaac.scene.pose.read_prim_world_pose",
        lambda actual_stage, prim_path: (
            (
                np.asarray([0.2, -0.1, -0.4], dtype=float),
                np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            if actual_stage is stage and prim_path == "/World/Workstation"
            else None
        ),
    )
    monkeypatch.setattr(
        runtime_objects,
        "apply_prim_local_pose_and_zero_velocity",
        lambda *_args: pytest.fail("Newton immutable object must not use USD restore"),
    )
    views = create_scene_object_state_views(
        (handle,),
        physics_backend="newton",
        stage=stage,
        immutable_static=True,
    )
    objects = runtime_objects._runtime_object_snapshots(
        stage=stage,
        handles=(handle,),
        state_views=views,
    )
    workstation = objects["workstation"]
    np.testing.assert_allclose(workstation.positions_local, [0.2, -0.1, -0.4])
    np.testing.assert_allclose(workstation.linear_velocities, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(workstation.angular_velocities, [0.0, 0.0, 0.0])

    mapping = SimpleNamespace(source_name="workstation", bodies=None)
    runtime = SimpleNamespace(
        session=SimpleNamespace(stage=stage),
        object_handles=(handle,),
        object_state_views=views,
    )
    assert runtime_objects._restore_runtime_objects(
        runtime,
        SimpleNamespace(objects=objects),
        compatibility=SimpleNamespace(object_mappings={"workstation": mapping}),
    ) == ("workstation",)

    relocated = ObjectSnapshot(
        name="workstation",
        object_profile="workstation",
        positions_local=np.asarray([0.3, -0.1, -0.4], dtype=float),
        orientations_wxyz=workstation.orientations_wxyz,
        linear_velocities=np.zeros(3, dtype=float),
        angular_velocities=np.zeros(3, dtype=float),
    )
    with pytest.raises(RuntimeError, match="cannot relocate an immutable"):
        runtime_objects._restore_runtime_objects(
            runtime,
            SimpleNamespace(objects={"workstation": relocated}),
            compatibility=SimpleNamespace(object_mappings={"workstation": mapping}),
        )
