"""Role-scoped materials and the real Newton-to-Warp contact mixing path."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.configuration.physics import NewtonContactSettings
from linkerbot_sim.isaac.physics.newton.contacts import configure_contacts
from linkerbot_sim.isaac.physics.newton.replication import (
    _load_newton_dependencies,
    _new_registered_builder,
)


def _policy():
    return load_mirror_config("newton_cpu").physics.ar5_o6_contact


def _robot(unit, root):
    return SimpleNamespace(
        asset_path=Path(f"/assets/{unit}/workstation.xml"), imported_root_path=root
    )


def test_shipped_contact_policy_matches_devices_and_scope():
    policy = _policy()
    assert isinstance(policy, NewtonContactSettings)
    assert policy == load_mirror_config("newton_cuda").physics.ar5_o6_contact
    assert policy.impratio == 100 and policy.cone == "elliptic"
    assert policy.hand.condim == policy.object.condim == policy.station.condim == 4
    # With no O6, do not even inspect/mutate the builder or import Newton.
    assert (
        configure_contacts(
            None, {"l6": _robot("ar5_08_l6_l", "/L6")}, (), ("/World",), policy
        )
        == {}
    )
    assert configure_contacts(None, {}, (), ("/World",), None) == {}


@pytest.mark.parametrize("execution", ["cpu", "cuda"])
@pytest.mark.parametrize("static", [False, True])
def test_native_hand_object_mixing_preserves_other_robot_and_authored_gap(
    execution, static
):
    newton = pytest.importorskip("newton")
    wp = pytest.importorskip("warp")
    wp.init()
    if execution == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA is required")
    device = "cpu" if execution == "cpu" else "cuda:0"
    builder = _new_registered_builder(_load_newton_dependencies(), up_axis="Z")
    hand_body = builder.add_link(label="/World/O6/hand_rh_index_distal")
    fixed = builder.add_joint_fixed(-1, hand_body)
    builder.add_articulation([fixed])
    hand = builder.add_shape_box(
        hand_body,
        hx=0.05,
        hy=0.05,
        hz=0.02,
        label="/World/O6/hand_rh_index_distal/collision",
    )
    object_body = builder.add_link(
        xform=wp.transform(wp.vec3(0, 0, 0.0399), wp.quat_identity()),
        label="/World/Object/body",
    )
    joint = builder.add_joint_free(object_body)
    builder.add_articulation([joint])
    obj = builder.add_shape_box(
        object_body, hx=0.02, hy=0.02, hz=0.02, label="/World/Object/collision"
    )
    other = builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(2, 0, 0), wp.quat_identity()),
        hx=0.1,
        hy=0.1,
        hz=0.1,
        label="/World/L6/collision",
    )
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0, 0, -0.5), wp.quat_identity()),
        hx=1,
        hy=1,
        hz=0.1,
        label="/World/Table/collision",
    )
    builder.shape_gap[hand], builder.shape_gap[obj] = 0.0, 0.003
    before = {
        name: list(getattr(builder, name))
        for name in (
            "shape_material_mu",
            "shape_margin",
            "shape_gap",
            "body_mass",
            "body_inertia",
            "shape_transform",
        )
    }
    robots = {
        "o6": _robot("ar5_08_o6_r", "/World/O6"),
        "l6": _robot("ar5_08_l6_l", "/World/L6"),
    }
    handles = (
        SimpleNamespace(
            kind="rigid",
            config=SimpleNamespace(
                prim_path="/World/Object", physics=SimpleNamespace(static=static)
            ),
        ),
    )
    counts = configure_contacts(builder, robots, handles, ("/World",), _policy())
    assert counts == {"hand": 1, "object": int(not static), "station": 1 + int(static)}
    for name in ("body_mass", "body_inertia", "shape_transform", "shape_gap"):
        np.testing.assert_array_equal(getattr(builder, name), before[name])
    for name in ("shape_material_mu", "shape_margin"):
        assert getattr(builder, name)[other] == before[name][other]
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        use_mujoco_contacts=False,
        cone="elliptic",
        impratio=100,
        iterations=100,
        ls_iterations=50,
    )
    pipeline = newton.CollisionPipeline(
        model, deterministic=True, contact_matching="disabled"
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, state, model.control(), contacts, 1 / 480)
    wp.synchronize_device(device)
    data = solver.mjw_data
    count = int(data.nacon.numpy()[0])
    assert count > 0
    np.testing.assert_allclose(
        data.contact.friction.numpy()[:count, :3],
        np.tile([4, 4, 1e-5 if static else 0.005], (count, 1)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        data.contact.solref.numpy()[:count], np.tile([0.004, 1], (count, 1)), rtol=1e-6
    )
    np.testing.assert_allclose(
        data.contact.solimp.numpy()[:count],
        np.tile([0.925, 0.95, 0.001, 0.5, 2], (count, 1)),
        rtol=1e-6,
    )
    assert np.all(data.contact.dim.numpy()[:count] == 4)
    assert np.isfinite(state.joint_q.numpy()).all()
    assert np.isfinite(state.joint_qd.numpy()).all()
    other_geom = solver.newton_shape_to_mjc_geom.numpy()[other]
    assert solver.mj_model.geom_condim[other_geom] == 3
    np.testing.assert_allclose(solver.mjw_model.opt.impratio_invsqrt.numpy(), 0.1)
    # Friction reference is generated by kf and body weights, not normal solref.
    assert np.all(data.contact.solreffriction.numpy()[:count, 0] > 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("impratio", 0),
        ("solref", [0, 1]),
        (
            "hand",
            {"friction": [-1, 0, 0], "solimp": [0.9, 0.95, 0.001, 0.5, 2], "condim": 4},
        ),
    ],
)
def test_contact_policy_rejects_invalid_numeric_inputs(field, value):
    from dataclasses import asdict

    data = asdict(_policy())
    data[field] = value
    with pytest.raises(ValueError):
        NewtonContactSettings.from_mapping(data, label="contact")
