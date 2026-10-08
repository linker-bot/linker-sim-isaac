from types import SimpleNamespace

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from linkerbot_sim.isaac.physics.newton.replication import _apply_body_collision_filters


def test_body_filter_only_expands_owned_shapes_and_keeps_descendants_collidable():
    stage = Usd.Stage.CreateInMemory()
    paths = ["/World/Arm", "/World/Arm/Flange", "/World/Arm/Flange/Hand"]
    for path in paths:
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(paths[0]))
    pairs.CreateFilteredPairsRel().SetTargets([Sdf.Path(paths[1])])
    captured = set()
    builder = SimpleNamespace(
        body_shapes={0: [0, 1], 1: [2], 2: [3]},
        add_shape_collision_filter_pair=lambda first, second: captured.add(
            (first, second)
        ),
    )
    count = _apply_body_collision_filters(
        builder,
        stage,
        {"path_body_map": dict(zip(paths, range(3))), "path_shape_map": {}},
    )
    assert count == 2
    assert captured == {(0, 2), (1, 2)}
    assert all(3 not in pair for pair in captured)
