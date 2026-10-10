"""Native contact regressions; run explicitly with the simulation interpreter."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.newton.manager import NewtonRuntime
from linkerbot_sim.isaac.physics.newton.replication import (
    _load_newton_dependencies,
    _new_registered_builder,
)
from linkerbot_sim.isaac.spec import IsaacNewtonCpuSpec, IsaacNewtonCudaSpec


def test_mjcf_fallback_keeps_native_inheritance_and_geometry(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from linkerbot_sim.assets.mjcf_defaults import prepare_mjcf_contact_defaults

    source = tmp_path / "model.xml"
    includes = tmp_path / "parts"
    includes.mkdir()
    (includes / "defaults.xml").write_text(
        '<mujocoinclude><default><geom friction=".6 .005 .0001"/>'
        '<default class="soft"><geom solref=".02 1"/>'
        '<default class="nested"/></default></default></mujocoinclude>'
    )
    source.write_text(
        '<mujoco><include file="parts/defaults.xml"/><worldbody>'
        '<geom name="implicit" type="sphere" size=".1"/>'
        '<body childclass="nested"><geom name="inherited" type="sphere" size=".1"/>'
        '<geom name="explicit" type="sphere" size=".1" solref=".03 2"/>'
        "</body></worldbody></mujoco>"
    )
    output = prepare_mjcf_contact_defaults(
        source, tmp_path / "output/model.xml", time_constant_s=0.004
    )
    before = mujoco.MjModel.from_xml_path(str(source))
    after = mujoco.MjModel.from_xml_path(str(output))
    for name, expected in (
        ("implicit", [0.004, 1]),
        ("inherited", [0.02, 1]),
        ("explicit", [0.03, 2]),
    ):
        np.testing.assert_allclose(after.geom(name).solref, expected)
    for field in (
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_friction",
        "geom_solimp",
    ):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))


def _box_runtime(
    execution, *, height, downward_speed=0.0, contact_time_constant_s=None
):
    newton = pytest.importorskip("newton")
    wp = pytest.importorskip("warp")
    wp.init()
    if execution == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA is required")
    spec = IsaacNewtonCpuSpec() if execution == "cpu" else IsaacNewtonCudaSpec()
    if contact_time_constant_s is not None:
        spec = replace(spec, default_contact_time_constant_s=contact_time_constant_s)
    device = "cpu" if execution == "cpu" else "cuda:0"
    builder = _new_registered_builder(
        _load_newton_dependencies(),
        up_axis="Z",
        default_contact_time_constant_s=spec.default_contact_time_constant_s,
    )
    cfg = builder.default_shape_cfg
    cfg.gap = 0.0
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity())
    )
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    joint = builder.add_joint_free(body)
    builder.add_articulation([joint])
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        hx=1.0,
        hy=1.0,
        hz=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()
    qd = state.joint_qd.numpy()
    qd[2] = -downward_speed
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    manager = NewtonRuntime.__new__(NewtonRuntime)
    manager.physics_spec = spec
    manager.physics_dt = 1.0 / 60.0
    manager.stream = wp.Stream(device) if execution == "cuda" else None
    manager.execution = execution
    manager.state = state
    manager.control = model.control()
    manager.solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=execution == "cpu",
        use_mujoco_contacts=execution == "cpu",
        iterations=100,
        ls_iterations=50,
        solver="newton",
    )
    manager._collision_pipeline = (
        newton.CollisionPipeline(model) if execution == "cuda" else None
    )
    manager._contacts = (
        manager._collision_pipeline.contacts()
        if manager._collision_pipeline is not None
        else None
    )
    return manager


@pytest.mark.parametrize("execution", ["cpu", "cuda"])
@pytest.mark.parametrize("contact_time_constant_s", [None, 0.004])
def test_contact_resolves_ten_centimeter_drop(execution, contact_time_constant_s):
    runtime = _box_runtime(
        execution, height=0.12, contact_time_constant_s=contact_time_constant_s
    )
    original = runtime.solver.step
    bottoms = []

    def observed_step(*args):
        original(*args)
        bottoms.append(float(runtime.state.body_q.numpy()[0, 2]) - 0.02)

    runtime.solver.step = observed_step
    for _ in range(120):
        runtime._simulate()
    assert len(bottoms) == 120 * runtime._effective_substeps
    assert np.isfinite(bottoms).all()
    # The 20 ms default permits transient soft-contact compression, but the
    # box center must stay above the support surface and then settle. Keep the
    # former 4 ms penetration bound as a regression for that explicit setting.
    penetration_limit = 0.02 if contact_time_constant_s is None else 0.004
    assert min(bottoms) > -penetration_limit, min(bottoms)
    assert abs(bottoms[-1]) < 0.0005
    assert np.linalg.norm(runtime.state.body_qd.numpy()[0]) < 0.001
    expected_response = 0.02 if contact_time_constant_s is None else 0.004
    assert runtime.solver.mj_model.geom_solref[:, 0] == pytest.approx(expected_response)


def test_new_contact_is_detected_inside_the_outer_step():
    runtime = _box_runtime(
        "cuda", height=0.021, downward_speed=1.0, contact_time_constant_s=0.004
    )
    calls = []
    collide = runtime._collision_pipeline.collide

    def observed_collide(state, contacts):
        collide(state, contacts)
        calls.append(int(contacts.rigid_contact_count.numpy()[0]))

    runtime._collision_pipeline.collide = observed_collide
    runtime._simulate()
    assert len(calls) == runtime._effective_substeps
    assert calls[0] == 0
    assert any(count > 0 for count in calls[1:])
    assert float(runtime.state.body_q.numpy()[0, 2]) > 0.018


def test_upstream_usd_contact_defaults_preserve_authored_values():
    pytest.importorskip("newton")
    from newton._src.usd.schema_resolver import PrimType, SchemaResolverManager
    from linkerbot_sim.isaac.physics.newton.replication import _new_schema_resolvers

    class Prim:
        def __init__(self, values):
            self.values = values

        def GetPath(self):
            return "/test"

        def GetAuthoredPropertiesInNamespace(self, namespace):
            return []

        def GetAttribute(self, name):
            if name not in self.values:
                return None
            return SimpleNamespace(
                HasAuthoredValue=lambda: True, Get=lambda: self.values[name]
            )

    dependencies = _load_newton_dependencies()
    original = dependencies.schema_resolver_mjc_type.mapping[PrimType.SHAPE][
        "ke"
    ].default
    resolvers = _new_schema_resolvers(dependencies)
    manager = SchemaResolverManager(resolvers)
    assert manager.get_value(Prim({}), PrimType.SHAPE, "ke") is None
    assert manager.get_value(Prim({}), PrimType.SHAPE, "kd") is None
    assert manager.get_value(
        Prim({"mjc:solref": [0.05, 1]}), PrimType.SHAPE, "ke"
    ) == pytest.approx(400)
    assert manager.get_value(
        Prim({"mjc:solref": [0.05, 1]}), PrimType.SHAPE, "kd"
    ) == pytest.approx(40)
    authored = Prim({"newton:contact_ke": 123, "mjc:solref": [0.05, 1]})
    assert manager.get_value(authored, PrimType.SHAPE, "ke") == 123
    assert (
        dependencies.schema_resolver_mjc_type.mapping[PrimType.SHAPE]["ke"].default
        == original
    )
    ordinary = SchemaResolverManager(_new_schema_resolvers(dependencies))
    assert ordinary.get_value(Prim({}), PrimType.SHAPE, "ke") is None
