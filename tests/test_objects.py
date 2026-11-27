"""Object import (CPU): mass properties, the USD file, and what Newton builds from it."""

import warnings

import numpy as np
import pytest
import trimesh

from linker_lab import objects


def box(size, offset=(0.0, 0.0, 0.0)):
    mesh = trimesh.creation.box(size)
    return mesh.vertices + np.asarray(offset), mesh.faces


@pytest.fixture(autouse=True)
def object_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(objects, "cache_dir", lambda: tmp_path)


def test_box_mass_properties():
    a, b, c, mass = 0.06, 0.04, 0.1, 0.3
    com, inertia = objects.mass_properties([box((a, b, c), (0.01, 0, 0))], mass)
    assert np.allclose(com, (0.01, 0, 0))
    expected = mass / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    assert np.allclose(inertia, expected)


def test_two_parts_add_up_like_one_body():
    halves = [box((0.05, 0.05, 0.05), (-0.025, 0, 0)), box((0.05, 0.05, 0.05), (0.025, 0, 0))]
    whole = [box((0.1, 0.05, 0.05))]
    for a, b in zip(objects.mass_properties(halves, 0.2), objects.mass_properties(whole, 0.2)):
        assert np.allclose(a, b)


def test_load_mesh(tmp_path):
    path = tmp_path / "box.stl"
    trimesh.creation.box((0.05, 0.05, 0.05)).export(path)
    (vertices, faces), = objects.load_mesh(path, scale=2.0)
    assert np.allclose(np.ptp(vertices, axis=0), 0.1) and faces.shape[1] == 3


def test_usd_is_named_by_content():
    parts = [box((0.05, 0.05, 0.05))]
    first = objects.write_usd(parts, 0.1)
    assert objects.write_usd(parts, 0.1) == first
    assert objects.write_usd(parts, 0.2) != first


def test_newton_imports_what_the_file_says():
    """Mass, centre of mass, inertia and contact settings reach Newton unchanged."""
    import newton
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
    from newton.solvers import SolverMuJoCo
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

    parts = [box((0.06, 0.04, 0.1), (0.0, 0.0, 0.01)), box((0.02, 0.02, 0.02), (0.0, 0.0, 0.07))]
    com = (0.001, -0.002, 0.02)
    inertia = np.array([[3e-4, 1e-5, 0], [1e-5, 3.5e-4, 0], [0, 0, 2e-4]])
    path = objects.write_usd(parts, 0.25, com, inertia)

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(
        str(path),
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
        load_visual_shapes=False,  # with visuals, each part also gets a render-only copy
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. "Inertia validation corrected"
        model = builder.finalize(device="cpu")
    assert model.body_count == 1 and model.shape_count == len(parts)
    assert model.body_mass.numpy()[0] == pytest.approx(0.25)
    assert np.allclose(model.body_com.numpy()[0], com, atol=1e-7)
    assert np.allclose(model.body_inertia.numpy()[0], inertia, atol=1e-9)

    m = objects.MATERIAL
    assert np.allclose(model.shape_material_mu.numpy(), m["friction"])
    assert np.allclose(model.shape_material_mu_torsional.numpy(), m["torsional_friction"])
    assert np.allclose(model.shape_material_mu_rolling.numpy(), m["rolling_friction"])
    assert np.allclose(model.shape_gap.numpy(), m["gap"])
    assert np.allclose(model.shape_margin.numpy(), m["margin"])
    assert (model.mujoco.condim.numpy() == m["condim"]).all()
    assert np.allclose(model.mujoco.solref.numpy(), m["solref"])
    assert np.allclose(model.mujoco.geom_solimp.numpy(), m["solimp"])
    assert (model.mujoco.solref_mode.numpy() == SOLREF_MODE_RAW).all()  # used as written
    # Each part collides as its exact convex hull (a box: its 8 corners).
    assert (model.shape_type.numpy() == int(newton.GeoType.CONVEX_MESH)).all()
    assert (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)).all()
    assert [len(model.shape_source[s].vertices) for s in range(model.shape_count)] == [8, 8]


def test_rejects_unphysical_inertia():
    with pytest.raises(ValueError, match="physical inertia"):
        objects.write_usd([box((0.05, 0.05, 0.05))], 0.1, (0, 0, 0), np.diag([1e-4, 1e-4, 3e-4]))


def test_table_and_placement():
    table = objects.table()
    assert table["top"] == pytest.approx(-0.4998)  # 0.5 m below the arm-base origin
    parts = [box((0.04, 0.06, 0.1))]
    x, y, z = objects.on_table(parts, (0.3, 0.1), clearance=0.0)
    assert (x, y) == (0.3, 0.1) and z == pytest.approx(table["top"] + 0.05)
    # Lying on its side (90 degrees about x), the 6 cm edge is vertical.
    q = (np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4))
    assert objects.on_table(parts, (0.3, 0.1), q, clearance=0.0)[2] == pytest.approx(
        table["top"] + 0.03
    )


def test_fixed_object_is_a_static_collider():
    """A fixed object has no body: Newton attaches its colliders to the world, with the
    object material."""
    import newton
    from newton.solvers import SolverMuJoCo
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

    parts = [box((1.0, 1.0, 0.02))]
    path = objects.write_usd(parts, None)
    assert path != objects.write_usd(parts, 0.1)
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(
        str(path),
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
        load_visual_shapes=False,
    )
    model = builder.finalize(device="cpu")
    assert model.body_count == 0 and model.shape_count == 1
    assert model.shape_body.numpy()[0] == -1
    assert np.allclose(model.shape_material_mu.numpy(), objects.MATERIAL["friction"])
    assert (model.mujoco.condim.numpy() == objects.MATERIAL["condim"]).all()
    with pytest.raises(ValueError, match="fixed object"):
        objects.write_usd(parts, None, (0, 0, 0), np.eye(3) * 1e-3)


def test_visual_mesh_is_shown_and_never_collides():
    """With ``visual``, a viewer shows that one mesh; the parts still collide, hidden."""
    import newton
    from newton.solvers import SolverMuJoCo
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

    parts = [box((0.02, 0.02, 0.02), (x, 0, 0)) for x in (-0.03, 0.0, 0.03)]
    visual = box((0.08, 0.02, 0.02))
    path = objects.write_usd(parts, 0.1, visual=visual)
    assert path != objects.write_usd(parts, 0.1)
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(  # as Isaac Lab imports it
        str(path),
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
        load_visual_shapes=True,
        hide_collision_shapes=True,
    )
    model = builder.finalize(device="cpu")
    flags = model.shape_flags.numpy()
    collides = flags & int(newton.ShapeFlags.COLLIDE_SHAPES) != 0
    visible = flags & int(newton.ShapeFlags.VISIBLE) != 0
    assert collides.sum() == 3 and visible.sum() == 1 and not (collides & visible).any()
    assert model.body_mass.numpy()[0] == pytest.approx(0.1)
