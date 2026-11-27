"""My task: pick up a box with the left hand (an example to change into your own task).

LW-v0's default environment (``linker_lab.workstation.LWv0EnvCfg``) is the workstation: the
scene, the physics, an action per arm and hand, joint-state observations and a reset to home.
A task subclasses it and adds what makes it a task. This one adds:

- **scene**: a 5 cm, 100 g box on the table in front of the left arm;
- **actions**: the left arm (7 joint goals [rad], reached within the real controller's
  limits) and the left hand (6 shaft targets [rad]); the right arm and hand hold home;
- **events**: at every reset the robot goes home and the box is placed at random;
- **observations**: the robot's joint states plus the box and palm positions;
- **rewards**: two examples, the palm approaching the box and the box rising;
- **terminations**: a time limit (5 s; LW-v0's default environment has none) and the box
  falling off the table.

Every position is in the environment frame: x forward over the table, z up, the table top
0.5 m below the origin.
"""

import math

import torch
import trimesh
from isaaclab.envs import mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from linker_lab import objects, workstation

# ---------------------------------------------------------------------------- the box

BOX_EDGE = 0.05
"""Edge length of the box [m]."""
_mesh = trimesh.creation.box((BOX_EDGE, BOX_EDGE, BOX_EDGE))
BOX = [(_mesh.vertices, _mesh.faces)]
BOX_XY = (0.40, 0.30)
"""Centre of the box's start area [m], in front of the left arm."""
BOX_SPREAD = 0.08
"""The box starts up to this far [m] from BOX_XY in x and y, at a random yaw."""
TABLE_TOP = objects.table()["top"]
"""Height of the table top [m] (-0.5)."""


def palm_position(env) -> torch.Tensor:
    """The left palm point (N, 3), in the environment frame."""
    position, _ = workstation.palm_pose_w(env.scene["left"])
    return position - env.scene.env_origins


def palm_to_box(env, std: float) -> torch.Tensor:
    """1 when the palm point is at the box's centre, falling off over ``std`` [m]."""
    box = mdp.root_pos_w(env, SceneEntityCfg("box"))
    return 1 - torch.tanh(torch.linalg.norm(box - palm_position(env), dim=-1) / std)


def box_height(env, cap: float) -> torch.Tensor:
    """How far [m] the box's centre is above where it rests on the table, at most ``cap``."""
    rest = TABLE_TOP + BOX_EDGE / 2
    return (mdp.root_pos_w(env, SceneEntityCfg("box"))[:, 2] - rest).clamp(0, cap)


# ---------------------------------------------------------------------------- the task


@configclass
class MySceneCfg(workstation.LWv0SceneCfg):
    box = objects.object_cfg(
        BOX, mass=0.1, pos=objects.on_table(BOX, BOX_XY), prim_path="{ENV_REGEX_NS}/box"
    )


@configclass
class MyActionsCfg:
    left_arm = workstation.arm_action_cfg("left")
    left_hand = workstation.hand_action_cfg("left")


@configclass
class MyObservationsCfg(workstation.ObservationsCfg):
    @configclass
    class PolicyCfg(workstation.ObservationsCfg.PolicyCfg):
        box_position = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("box")})
        palm_position = ObservationTermCfg(func=palm_position)

    policy = PolicyCfg()


@configclass
class MyEventsCfg(workstation.EventsCfg):
    # Runs after reset_to_home, which puts the box back at its start pose.
    place_box = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "pose_range": {
                "x": (-BOX_SPREAD, BOX_SPREAD),
                "y": (-BOX_SPREAD, BOX_SPREAD),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
        },
    )


@configclass
class MyRewardsCfg:
    reach = RewardTermCfg(func=palm_to_box, weight=1.0, params={"std": 0.1})
    lift = RewardTermCfg(func=box_height, weight=10.0, params={"cap": 0.1})


@configclass
class MyTerminationsCfg(workstation.TerminationsCfg):
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)  # after episode_length_s
    box_dropped = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": TABLE_TOP - 0.1, "asset_cfg": SceneEntityCfg("box")},
    )


@configclass
class MyTaskEnvCfg(workstation.LWv0EnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=16, env_spacing=3.0)
    episode_length_s = 5.0
    actions: MyActionsCfg = MyActionsCfg()
    observations: MyObservationsCfg = MyObservationsCfg()
    events: MyEventsCfg = MyEventsCfg()
    rewards: MyRewardsCfg = MyRewardsCfg()
    terminations: MyTerminationsCfg = MyTerminationsCfg()
