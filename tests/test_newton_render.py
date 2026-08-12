from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.assets.root_pose import RootPoseConfig, apply_root_pose_transform
from linkerbot_sim.assets.robot_import import release_imported_asset_files
from linkerbot_sim.isaac.physics.newton.render import (
    NewtonRenderSync,
    prepare_newton_render_stage,
    prepare_newton_render_subtree,
)
from linkerbot_sim.objects.dynamic_chain.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
)
from linkerbot_sim.objects.rigid.config import RigidObjectConfig
from linkerbot_sim.objects.rigid.importer import add_rigid_objects


class _Array:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)
        self.numpy_calls = 0

    def numpy(self) -> np.ndarray:
        self.numpy_calls += 1
        return self._values.copy()


def _stage():
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    del pxr
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    link = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/link")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    static = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Static")
    static.AddTranslateOp().Set((1.0, 2.0, 3.0))
    camera = UsdGeom.Camera.Define(stage, "/World/envs/env_0/Camera")
    UsdGeom.Xformable(camera).AddTranslateOp().Set((0.5, 0.0, 1.0))
    return stage


def _prepare(
    stage: object,
    *,
    world_transform: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
) -> object:
    prepare_newton_render_subtree(
        stage=stage,
        subtree_root="/World/envs/env_0",
    )
    prepare_newton_render_stage(
        stage=stage,
        prototype_root="/World/envs/env_0",
        world_transform=world_transform,
    )
    return stage


def _model(*, second_label: str = "/World/envs/env_1/Robot/link") -> object:
    return SimpleNamespace(
        body_label=[
            "/World/envs/env_0/Robot/link",
            second_label,
            "/World/global_body",
        ],
        body_world=_Array([0, 1, -1]),
    )


def _xform_topology(stage: object, path: str) -> tuple[tuple[str, ...], bool]:
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    return (
        tuple(str(op.GetOpName()) for op in xform.GetOrderedXformOps()),
        bool(xform.GetResetXformStack()),
    )


def test_render_root_pose_starts_canonical_and_repeated_pose_is_value_only() -> None:
    from pxr import Tf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Object")
    apply_root_pose_transform(
        stage,
        "/World/Object",
        RootPoseConfig(
            xyz=(1.0, 2.0, 3.0),
            rpy=(0.0, 0.0, np.pi / 2.0),
        ),
        prepare_newton_render_topology=True,
    )

    assert _xform_topology(stage, "/World/Object") == (
        ("xformOp:transform:newtonRenderWorld",),
        True,
    )
    prim = stage.GetPrimAtPath("/World/Object")
    assert not prim.GetAttribute("xformOp:translate").IsValid()
    assert not prim.GetAttribute("xformOp:rotateXYZ").IsValid()
    assert tuple(
        UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    ) == pytest.approx((1.0, 2.0, 3.0))

    changed_paths: list[str] = []

    def _record_changed_paths(notice: object, _sender: object) -> None:
        changed_paths.extend(str(path) for path in notice.GetChangedInfoOnlyPaths())

    registration = Tf.Notice.Register(
        Usd.Notice.ObjectsChanged,
        _record_changed_paths,
        stage,
    )
    try:
        apply_root_pose_transform(
            stage,
            "/World/Object",
            RootPoseConfig(xyz=(4.0, 5.0, 6.0)),
            prepare_newton_render_topology=True,
        )
    finally:
        registration.Revoke()

    assert not any(path.endswith(".xformOpOrder") for path in changed_paths)
    assert tuple(
        UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    ) == pytest.approx((4.0, 5.0, 6.0))


def test_non_render_root_pose_keeps_translate_rotate_topology() -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Object")

    apply_root_pose_transform(
        stage,
        "/World/Object",
        RootPoseConfig(xyz=(1.0, 2.0, 3.0)),
        prepare_newton_render_topology=False,
    )

    assert _xform_topology(stage, "/World/Object") == (
        ("xformOp:translate", "xformOp:rotateXYZ"),
        False,
    )


def test_newton_rigid_reference_never_publishes_source_root_op_order(
    tmp_path,
) -> None:
    from pxr import Tf, Usd, UsdGeom, UsdPhysics

    source_path = tmp_path / "rigid.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/Rigid")
    source_root.AddTranslateOp().Set((9.0, 8.0, 7.0))
    source_root.AddRotateXYZOp().Set((10.0, 20.0, 30.0))
    UsdPhysics.RigidBodyAPI.Apply(source_root.GetPrim())
    source_body = UsdGeom.Xform.Define(source_stage, "/Rigid/NestedBody")
    source_body.AddTranslateOp().Set((2.0, 0.0, 0.0))
    source_body.AddScaleOp().Set((0.5, 0.5, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(source_body.GetPrim())
    source_stage.SetDefaultPrim(source_root.GetPrim())
    source_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    observed_nonempty_orders: list[tuple[str, tuple[str, ...]]] = []

    def _record_order(_notice: object, _sender: object) -> None:
        for path in ("/World/TBlock", "/World/TBlock/NestedBody"):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            order = tuple(
                str(op.GetOpName())
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            )
            if order:
                observed_nonempty_orders.append((path, order))

    registration = Tf.Notice.Register(Usd.Notice.ObjectsChanged, _record_order, stage)
    try:
        added = add_rigid_objects(
            stage,
            (
                RigidObjectConfig(
                    name="TBlock",
                    asset_type="usd",
                    asset_path=source_path,
                    prim_path="/World/TBlock",
                    root_pose=RootPoseConfig(xyz=(1.0, 2.0, 3.0)),
                ),
            ),
            physics_backend="newton",
            prepare_newton_render_topology=True,
        )
        # runtime 的最终调用只允许做幂等审计，不能再发布另一份 op order。
        prepare_newton_render_subtree(
            stage=stage,
            subtree_root="/World/TBlock",
        )
    finally:
        registration.Revoke()
        release_imported_asset_files()

    expected = ("xformOp:transform:newtonRenderWorld",)
    assert added[0].prim_path == "/World/TBlock"
    assert observed_nonempty_orders
    assert {order for _, order in observed_nonempty_orders} == {expected}
    assert _xform_topology(stage, "/World/TBlock") == (expected, True)
    assert _xform_topology(stage, "/World/TBlock/NestedBody") == (expected, True)
    assert tuple(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(stage.GetPrimAtPath("/World/TBlock"))
        .ExtractTranslation()
    ) == pytest.approx((1.0, 2.0, 3.0))


def test_newton_dynamic_chain_reference_never_publishes_source_body_op_order(
    tmp_path,
) -> None:
    from pxr import Tf, Usd, UsdGeom, UsdPhysics

    source_path = tmp_path / "rope.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    wrong_default = UsdGeom.Xform.Define(source_stage, "/WrongDefault")
    rope_root = UsdGeom.Xform.Define(source_stage, "/Rope")
    rope_root.AddTranslateOp().Set((9.0, 0.0, 0.0))
    segment = UsdGeom.Xform.Define(source_stage, "/Rope/segment_0")
    segment.AddTranslateOp().Set((2.0, 0.0, 0.0))
    segment.AddScaleOp().Set((0.5, 0.5, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(segment.GetPrim())
    source_stage.SetDefaultPrim(wrong_default.GetPrim())
    source_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    observed_nonempty_orders: list[tuple[str, tuple[str, ...]]] = []

    def _record_order(_notice: object, _sender: object) -> None:
        for path in ("/World/Rope", "/World/Rope/segment_0"):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            order = tuple(
                str(op.GetOpName())
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            )
            if order:
                observed_nonempty_orders.append((path, order))

    registration = Tf.Notice.Register(Usd.Notice.ObjectsChanged, _record_order, stage)
    try:
        model = add_capsule_rope_reference(
            stage,
            CapsuleRopeConfig(
                asset_path=str(source_path),
                prim_path="/World/Rope",
                root_path="/Rope",
            ),
            physics_backend="newton",
            root_pose=RootPoseConfig(xyz=(1.0, 0.0, 0.0)),
            prepare_newton_render_topology=True,
        )
        prepare_newton_render_subtree(stage=stage, subtree_root="/World/Rope")
    finally:
        registration.Revoke()
        release_imported_asset_files()

    expected = ("xformOp:transform:newtonRenderWorld",)
    assert model["root"].IsValid()
    assert len(model["bodies"]) == 1
    assert observed_nonempty_orders
    assert {order for _, order in observed_nonempty_orders} == {expected}
    assert _xform_topology(stage, "/World/Rope") == (expected, True)
    assert _xform_topology(stage, "/World/Rope/segment_0") == (expected, True)
    assert not stage.GetPrimAtPath("/World/Rope/WrongDefault").IsValid()
    assert tuple(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Rope/segment_0"))
        .ExtractTranslation()
    ) == pytest.approx((3.0, 0.0, 0.0))


def test_prepare_freezes_nested_robot_and_object_world_xforms() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/envs")
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    root.AddTranslateOp().Set((10.0, 0.0, 0.0))
    robot = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    robot.AddTranslateOp().Set((0.0, 2.0, 0.0))
    base = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/base")
    base.AddTranslateOp().Set((1.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    link = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/base/link")
    link.AddTranslateOp().Set((0.0, 0.0, 3.0))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    block = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Objects/block")
    block.AddTranslateOp().Set((-2.0, 0.0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(block.GetPrim())
    paths = (
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_0/Robot/base/link",
        "/World/envs/env_0/Objects/block",
    )
    before = {
        path: tuple(
            UsdGeom.XformCache()
            .GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            .ExtractTranslation()
        )
        for path in paths
    }

    prepared = prepare_newton_render_subtree(
        stage=stage,
        subtree_root="/World/envs/env_0",
    )
    audited = prepare_newton_render_stage(
        stage=stage,
        prototype_root="/World/envs/env_0",
        world_transform=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )

    assert prepared == paths
    assert audited == paths
    expected_topology = (("xformOp:transform:newtonRenderWorld",), True)
    for path in ("/World/envs/env_0", *paths):
        assert _xform_topology(stage, path) == expected_topology
    after_cache = UsdGeom.XformCache()
    for path in paths:
        actual = tuple(
            after_cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(path)
            ).ExtractTranslation()
        )
        assert actual == pytest.approx(before[path])

    clone_paths = tuple(path.replace("/env_0/", "/env_1/") for path in paths)
    sync = NewtonRenderSync(
        stage=stage,
        model=SimpleNamespace(
            body_label=[*paths, *clone_paths],
            body_world=_Array([0, 0, 0, 1, 1, 1]),
        ),
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        world_transforms=(
            (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )
    sync.sync(
        _Array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    assert sync.body_count == 6
    for index, path in enumerate((*paths, *clone_paths), start=1):
        assert _xform_topology(stage, path) == expected_topology
        position = (
            UsdGeom.XformCache()
            .GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            .ExtractTranslation()
        )
        assert tuple(position) == pytest.approx((float(index), 0.0, 0.0))


def test_stage_prepare_rejects_body_missed_by_asset_authoring() -> None:
    stage = _stage()

    with pytest.raises(RuntimeError, match="prepared during asset authoring"):
        prepare_newton_render_stage(
            stage=stage,
            prototype_root="/World/envs/env_0",
            world_transform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )

    # 晚边界只允许补 world root；遗漏 body 仍保持原始拓扑，不能被静默修补。
    assert _xform_topology(stage, "/World/envs/env_0") == (
        ("xformOp:transform:newtonRenderWorld",),
        True,
    )
    assert _xform_topology(stage, "/World/envs/env_0/Robot/link") == ((), False)


def test_subtree_prepare_is_idempotent_without_op_order_notice() -> None:
    from pxr import Tf, Usd

    stage = _stage()
    prepare_newton_render_subtree(
        stage=stage,
        subtree_root="/World/envs/env_0",
    )
    changed_paths: list[str] = []

    def _record_changed_paths(notice: object, _sender: object) -> None:
        changed_paths.extend(str(path) for path in notice.GetChangedInfoOnlyPaths())

    registration = Tf.Notice.Register(
        Usd.Notice.ObjectsChanged,
        _record_changed_paths,
        stage,
    )
    try:
        prepare_newton_render_subtree(
            stage=stage,
            subtree_root="/World/envs/env_0",
        )
    finally:
        registration.Revoke()

    assert not any(path.endswith(".xformOpOrder") for path in changed_paths)


def test_render_binding_and_sync_leave_prepared_op_topology_unchanged() -> None:
    stage = _prepare(_stage())
    paths = (
        "/World/envs/env_0",
        "/World/envs/env_0/Robot/link",
    )
    before = {path: _xform_topology(stage, path) for path in paths}
    model = SimpleNamespace(
        body_label=["/World/envs/env_0/Robot/link"],
        body_world=_Array([0]),
    )

    sync = NewtonRenderSync(
        stage=stage,
        model=model,
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0",),
        world_transforms=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),),
    )
    after_binding = {path: _xform_topology(stage, path) for path in paths}
    sync.sync(_Array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]))

    assert after_binding == before
    assert {path: _xform_topology(stage, path) for path in paths} == before


def test_render_binding_rejects_unprepared_body_without_repairing_it() -> None:
    from pxr import UsdGeom

    stage = _prepare(_stage())
    body = UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/link"))
    late_op = body.AddTranslateOp(opSuffix="lateMutation")
    late_op.Set((3.0, 2.0, 1.0))
    body.SetXformOpOrder([late_op], resetXformStack=False)
    before = _xform_topology(stage, "/World/envs/env_0/Robot/link")
    model = SimpleNamespace(
        body_label=["/World/envs/env_0/Robot/link"],
        body_world=_Array([0]),
    )

    with pytest.raises(RuntimeError, match="was not prepared before binding"):
        NewtonRenderSync(
            stage=stage,
            model=model,
            prototype_root="/World/envs/env_0",
            destination_roots=("/World/envs/env_0",),
            world_transforms=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),),
        )

    assert _xform_topology(stage, "/World/envs/env_0/Robot/link") == before


def test_render_sync_materializes_clone_and_authors_world_body_transforms() -> None:
    from pxr import UsdGeom

    stage = _prepare(
        _stage(),
        world_transform=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    sync = NewtonRenderSync(
        stage=stage,
        model=_model(),
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        world_transforms=(
            (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )

    sync.sync(
        _Array(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    assert sync.body_count == 2
    assert sync.sync_count == 1
    assert stage.GetPrimAtPath("/World/envs/env_1/Robot/link").IsValid()
    expected_topology = (("xformOp:transform:newtonRenderWorld",), True)
    assert _xform_topology(stage, "/World/envs/env_0/Robot/link") == expected_topology
    assert _xform_topology(stage, "/World/envs/env_1/Robot/link") == expected_topology
    assert tuple(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(stage.GetPrimAtPath("/World/envs/env_0/Static"))
        .ExtractTranslation()
    ) == pytest.approx((11.0, 2.0, 3.0))
    assert tuple(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(stage.GetPrimAtPath("/World/envs/env_1/Static"))
        .ExtractTranslation()
    ) == pytest.approx((21.0, 2.0, 3.0))
    assert tuple(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(stage.GetPrimAtPath("/World/envs/env_1/Camera"))
        .ExtractTranslation()
    ) == pytest.approx((20.5, 0.0, 1.0))
    for path, expected in (
        ("/World/envs/env_0/Robot/link", (1.0, 2.0, 3.0)),
        ("/World/envs/env_1/Robot/link", (4.0, 5.0, 6.0)),
    ):
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath(path)
        )
        assert tuple(matrix.ExtractTranslation()) == pytest.approx(expected)
        assert UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetResetXformStack()

    cached_ops = tuple(binding.transform_op for binding in sync._bindings)
    sync.sync(
        _Array(
            [
                [7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0],
                [10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    assert tuple(binding.transform_op for binding in sync._bindings) == cached_ops
    assert sync.sync_count == 2
    for path, expected in (
        ("/World/envs/env_0/Robot/link", (7.0, 8.0, 9.0)),
        ("/World/envs/env_1/Robot/link", (10.0, 11.0, 12.0)),
    ):
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        assert len(xform.GetOrderedXformOps()) == 1
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(xform.GetPrim())
        assert tuple(matrix.ExtractTranslation()) == pytest.approx(expected)


def test_render_sync_fails_closed_on_missing_body_prim() -> None:
    stage = _prepare(_stage())

    with pytest.raises(RuntimeError, match="missing=.*not_a_body"):
        NewtonRenderSync(
            stage=stage,
            model=_model(second_label="/World/envs/env_1/not_a_body"),
            prototype_root="/World/envs/env_0",
            destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
            world_transforms=(
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            ),
        )


def test_render_sync_rejects_non_finite_transform() -> None:
    stage = _prepare(_stage())
    sync = NewtonRenderSync(
        stage=stage,
        model=_model(),
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        world_transforms=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        sync.sync(
            _Array(
                [
                    [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        )


def test_render_sync_does_not_normalize_numpy_source_in_place() -> None:
    stage = _prepare(_stage())
    sync = NewtonRenderSync(
        stage=stage,
        model=_model(),
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        world_transforms=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )
    source = np.asarray(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    before = source.copy()

    sync.sync(source)

    # renderer 可以归一化自己的 host 快照，但不能反向修改 CPU 物理/快照源数组。
    np.testing.assert_array_equal(source, before)


def test_render_sync_materializes_and_updates_only_selected_world() -> None:
    from pxr import UsdGeom

    stage = _prepare(_stage())
    sync = NewtonRenderSync(
        stage=stage,
        model=_model(),
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        world_transforms=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        visible_world_indices=(1,),
    )
    body_q = _Array(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    sync.sync(body_q)

    assert body_q.numpy_calls == 1
    assert sync.body_count == 1
    assert sync.diagnostics()["visible_world_indices"] == [1]
    assert sync._bindings[0].world_index == 1
    source = stage.GetPrimAtPath("/World/envs/env_0")
    selected = stage.GetPrimAtPath("/World/envs/env_1")
    assert UsdGeom.Imageable(source).ComputeVisibility() == "invisible"
    assert UsdGeom.Imageable(selected).ComputeVisibility() == "inherited"
    selected_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath("/World/envs/env_1/Robot/link")
    )
    assert tuple(selected_matrix.ExtractTranslation()) == pytest.approx((4.0, 5.0, 6.0))
    # 未选中的 prototype body 不绑定、不更新；100m 的状态值不会写回 USD。
    source_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath("/World/envs/env_0/Robot/link")
    )
    assert tuple(source_matrix.ExtractTranslation()) == pytest.approx((0.0, 0.0, 0.0))
