"""The LW-v0 workstation config against LW-v0's files (CPU)."""

import numpy as np
import pytest

from linker_lab import assets, physics


@pytest.fixture(scope="module")
def ws():
    from linker_lab import workstation

    return workstation


@pytest.fixture(scope="module")
def mj(ws):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(assets.path("lw_v0.xml")))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_kinematics(model, data)
    return mujoco, model, data


def test_joint_names_are_the_driven_joints(ws, mj):
    mujoco, model, _ = mj
    for side in ws.SIDES:
        driven = {
            model.joint(model.actuator_trnid[i, 0]).name.removeprefix(f"{side}/")
            for i in range(model.nu)
            if model.joint(model.actuator_trnid[i, 0]).name.startswith(f"{side}/")
        }
        assert driven == {*ws.ARM_JOINTS[side], *ws.HAND_JOINTS[side]}
        assert set(assets.home(side)) >= driven
        model.body(f"{side}/{ws.TCP_BODY[side]}")
        model.body(f"{side}/{ws.PALM_BODY[side]}")


def test_palm_point_lies_in_front_of_the_palm(ws, mj):
    """The palm face is the hand base hull's +x face; flexing a finger moves its tip to +x."""
    mujoco, model, data = mj
    for side in ws.SIDES:
        palm = model.body(f"{side}/{ws.PALM_BODY[side]}").id
        hull = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == palm and model.geom_contype[g])
        mesh = model.geom_dataid[hull]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        # mesh vertices are in the geom frame; express them in the body frame
        rotation = data.xmat[palm].reshape(3, 3).T @ data.geom_xmat[hull].reshape(3, 3)
        offset = data.xmat[palm].reshape(3, 3).T @ (data.geom_xpos[hull] - data.xpos[palm])
        local = vertices @ rotation.T + offset
        assert local[:, 0].max() == pytest.approx(ws.PALM_OFFSET[0] - 0.035, abs=1e-3)
        assert local[:, 2].min() < ws.PALM_OFFSET[2] < local[:, 2].max()

        tip = model.body(f"{side}/hand_{side[0]}h_index_distal").id
        before = data.xmat[palm].reshape(3, 3).T @ (data.xpos[tip] - data.xpos[palm])
        flexed = mujoco.MjData(model)
        flexed.qpos[:] = data.qpos
        joint = model.joint(f"{side}/hand_{side[0]}h_index_mcp_pitch")
        flexed.qpos[model.jnt_qposadr[joint.id]] += 1.0
        mujoco.mj_kinematics(model, flexed)
        after = flexed.xmat[palm].reshape(3, 3).T @ (flexed.xpos[tip] - flexed.xpos[palm])
        assert after[0] - before[0] > 0.02


def test_env_cfg(ws):
    cfg = ws.LWv0EnvCfg()
    assert cfg.sim.dt == pytest.approx(1 / physics.PHYSICS_HZ)
    assert cfg.decimation == 8 and cfg.sim.render_interval == 8
    assert cfg.sim.use_newton_actuators is False
    solver = cfg.sim.physics.solver_cfg
    assert (solver.iterations, solver.ls_iterations, solver.cone, solver.impratio) == (30, 50, "elliptic", 10.0)
    assert (solver.nconmax, solver.njmax, solver.enable_multiccd) == (256, 512, True)
    assert solver.use_mujoco_contacts is False and cfg.sim.physics.collision_cfg is not None
    for side in ws.SIDES:
        unit = getattr(cfg.scene, side)
        assert unit.init_state.joint_pos == assets.home(side)
        assert unit.init_state.pos == ((0.0, 0.09, 0.0) if side == "left" else (0.0, -0.09, 0.0))
        drives = unit.actuators["drives"]
        assert drives.stiffness is None and drives.damping is None  # from the file
        assert drives.effort_limit_sim is None and drives.effort_limit is None
    names = [n for n in vars(cfg.actions) if not n.startswith("_")]
    assert names == ["left_arm", "right_arm", "left_hand", "right_hand"]
    assert [len(getattr(cfg.actions, n).joint_names) for n in names] == [7, 7, 6, 6]
    from linker_lab import arm, hand

    for side in ws.SIDES:
        a, h = getattr(cfg.actions, f"{side}_arm"), getattr(cfg.actions, f"{side}_hand")
        assert isinstance(a, arm.ArmActionCfg) and isinstance(h, hand.HandActionCfg)
        assert (a.command, a.interpolation, a.limits) == ("joint", True, "real")
        assert a.joint_names == list(ws.ARM_JOINTS[side]) and h.joint_names == list(ws.HAND_JOINTS[side])


def test_arm_action_cfg_frames_and_options(ws):
    from linker_lab import arm

    palm = ws.arm_action_cfg("right", command="pose", frame="palm")
    assert (palm.body_name, palm.body_offset) == (ws.PALM_BODY["right"], ws.PALM_OFFSET)
    tcp = ws.arm_action_cfg("left", command="pose")
    assert (tcp.body_name, tcp.body_offset) == (ws.TCP_BODY["left"], (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ws.arm_action_cfg("left", frame="wrist")
    servo = ws.arm_action_cfg("left", interpolation=False, stream="samples", limits=None)
    arm._check(servo)
    for bad in (
        dict(interpolation=True, stream="samples"),
        dict(interpolation=True, limits=None),
        dict(command="pose", interpolation=False, stream="points_velocity"),
        dict(stream="other", interpolation=False),
    ):
        with pytest.raises(ValueError):
            arm._check(ws.arm_action_cfg("left", **bad))


def test_registered(ws):
    import gymnasium as gym

    spec = gym.spec("Linker-LW-v0")
    assert spec.kwargs["env_cfg_entry_point"] == "linker_lab.workstation:LWv0EnvCfg"


@pytest.mark.parametrize("name", ["left_rgb", "left_depth", "right_rgb", "right_depth", "top"])
def test_camera_cfg(ws, name):
    part, body, rate_hz, data = ws.CAMERAS[name]
    k = assets.camera(part, body)
    real = ws.camera_cfg(name)
    assert (real.width, real.height) == k["resolution"]
    d = real.spawn.distortion
    assert np.allclose((d.fx, d.fy, d.cx, d.cy), (k["fx"], k["fy"], k["cx"], k["cy"]))
    w, h = (n // 2 for n in k["resolution"])
    cfg = ws.camera_cfg(name, width=w, height=h)
    assert (cfg.width, cfg.height) == (w, h)
    assert cfg.update_period == pytest.approx(1 / rate_hz)
    assert cfg.data_types == list(data)
    assert cfg.prim_path.endswith(f"/{body}/{name}")
    d = cfg.spawn.distortion
    assert np.allclose((d.fx, d.fy, d.cx, d.cy), (k["fx"] / 2, k["fy"] / 2, k["cx"] / 2, k["cy"] / 2))
    assert d.apply_lens_distortion is False
    assert cfg.offset.pos == (0.0, 0.0, ws.CAMERA_FORWARD) and cfg.offset.convention == "ros"


@pytest.mark.parametrize(
    "name, size",
    [
        ("left_rgb", {"width": 640, "height": 480}),  # a 4:3 mode of the 16:10 Gemini 335L
        ("top", {"width": 672, "height": 376}),  # the ZED 2i's VGA mode: not quite 16:9
        ("top", {"width": 640}),
        ("top", {"height": 360}),
        ("top", {"width": 0, "height": 0}),
    ],
)
def test_camera_cfg_refuses_other_sizes(ws, name, size):
    with pytest.raises(ValueError):
        ws.camera_cfg(name, **size)


def test_camera_cfg_rounds_either_side(ws):
    """A size of the real aspect ratio is accepted with either side rounded (1280 x 720: 16:9)."""
    for width, height in ((640, 360), (100, 56), (99, 56), (101, 57)):
        cfg = ws.camera_cfg("top", width=width, height=height)
        assert (cfg.width, cfg.height) == (width, height)


def test_episodes_never_end_by_themselves(ws):
    """No termination terms: an LW-v0 session runs until it is reset (tasks add their own)."""
    assert not ws.LWv0EnvCfg().terminations.to_dict()


def test_home_action_maps_normalized_hands_without_changing_other_terms(ws):
    """Mixed inputs hold the same home angles in either hand convention."""
    from types import SimpleNamespace

    import torch

    from linker_lab import arm, hand

    # Different limits per environment catch accidental use of row zero.
    home = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.2],
                         [0.1, 0.5, 0.75, 1.0, 1.25, 0.6, 0.3]])
    lower = torch.tensor([[0.0] * 7, [0.1] * 7])
    upper = lower + 2.0
    robot = SimpleNamespace(
        find_joints=lambda names, preserve_order: (list(range(len(names))), names),
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=home),
            joint_pos_limits=SimpleNamespace(torch=torch.stack((lower, upper), -1)),
        ),
    )
    terms = {
        "arm": SimpleNamespace(cfg=arm.ArmActionCfg(
            asset_name="robot", joint_names=[str(i) for i in range(7)],
            interpolation=False, stream="points_velocity")),
        "normalized_hand": SimpleNamespace(cfg=hand.HandActionCfg(
            asset_name="robot", joint_names=[str(i) for i in range(6)], normalized=True)),
        "radian_hand": SimpleNamespace(cfg=hand.HandActionCfg(
            asset_name="robot", joint_names=[str(i) for i in range(6)])),
    }
    env = SimpleNamespace(
        action_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__),
        scene={"robot": robot},
    )
    result = ws.home_action(env)
    torch.testing.assert_close(result[:, :7], home)
    torch.testing.assert_close(result[:, 7:14], torch.zeros_like(home))
    normalized = result[:, 14:20]
    restored = lower[:, :6] + (normalized + 1) * 0.5 * (upper[:, :6] - lower[:, :6])
    torch.testing.assert_close(restored, home[:, :6])
    torch.testing.assert_close(result[:, 20:], home[:, :6])
