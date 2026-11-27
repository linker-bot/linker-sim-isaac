"""The playground's jogging, configuration and floor (CPU)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from linker_lab import assets


@pytest.fixture(scope="module")
def pg():
    from linker_lab import playground

    return playground


DT = 1 / 60
FRONT = (1.0, 0.0, -0.5)  # looking along +x and down
HOME = (np.zeros(3), np.eye(3))


def test_keys_move_the_goal_relative_to_the_camera(pg):
    speed = pg.JOG_SPEED[0] * DT
    for keys, direction in (
        ({"i"}, (1, 0, 0)),  # forward: the horizontal view direction
        ({"k"}, (-1, 0, 0)),
        ({"l"}, (0, -1, 0)),  # right of a camera looking along +x
        ({"j"}, (0, 1, 0)),
        ({"u"}, (0, 0, 1)),
        ({"o"}, (0, 0, -1)),
    ):
        p, R = pg.jog(HOME, HOME, keys, FRONT, DT)
        assert np.allclose(p, np.multiply(direction, speed)) and np.allclose(R, np.eye(3))
    p, _ = pg.jog(HOME, HOME, {"i"}, (0.0, 1.0, 0.0), DT)  # camera turned to +y
    assert np.allclose(p, (0, speed, 0))
    p, _ = pg.jog(HOME, HOME, {"i"}, (0.0, 0.0, -1.0), DT)  # straight down: forward is +x
    assert np.allclose(p, (speed, 0, 0))


def test_shift_turns_the_goal_about_the_palm_point(pg):
    p, R = pg.jog(HOME, HOME, {"shift", "j"}, FRONT, DT)  # yaw about up
    assert np.allclose(p, 0)
    assert np.allclose(Rotation.from_matrix(R).as_rotvec(), (0, 0, pg.JOG_SPEED[1] * DT))
    _, R = pg.jog(HOME, HOME, {"shift", "u"}, FRONT, DT)  # roll about forward
    assert np.allclose(Rotation.from_matrix(R).as_rotvec(), (pg.JOG_SPEED[1] * DT, 0, 0))


def test_the_goal_stays_within_the_lead_of_the_palm_point(pg):
    at_lead = (np.array([pg.LEAD[0], 0.0, 0.0]), np.eye(3))
    assert pg.jog(at_lead, HOME, {"i"}, FRONT, DT) is at_lead  # further away: not taken
    p, _ = pg.jog(at_lead, HOME, {"k"}, FRONT, DT)  # back towards the palm point: taken
    assert p[0] < pg.LEAD[0]
    turned = (np.zeros(3), Rotation.from_rotvec((0, 0, pg.LEAD[1])).as_matrix())
    assert pg.jog(turned, HOME, {"shift", "j"}, FRONT, DT) is turned
    _, R = pg.jog(turned, HOME, {"shift", "l"}, FRONT, DT)
    assert Rotation.from_matrix(R).magnitude() < pg.LEAD[1]
    assert pg.jog(HOME, HOME, set(), FRONT, DT) is HOME


def test_environment(pg):
    from linker_lab import workstation

    cfg = pg.env_cfg()
    assert cfg.scene.num_envs == 1
    for side in workstation.SIDES:
        arm = getattr(cfg.actions, f"{side}_arm")
        assert arm.command == "pose" and arm.interpolation and arm.limits == "real"
        assert arm.body_name == workstation.PALM_BODY[side]
        assert tuple(arm.body_offset) == workstation.PALM_OFFSET
    assert not cfg.terminations.to_dict()  # no time limit: the session runs until closed
    assert cfg.sim.visualizer_cfgs.class_type is pg.PlaygroundVisualizer
    assert cfg.scene.floor.init_state.pos[2] + 0.025 == pytest.approx(pg.FLOOR_TOP)


def test_floor_is_at_the_foot_of_the_support(pg):
    """The floor top is the lowest point of the support's visual mesh."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(assets.part_file("support")))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    geom = model.geom("support_visual").id
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    lowest = (vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])[:, 2].min()
    assert pg.FLOOR_TOP == pytest.approx(lowest, abs=5e-4)
