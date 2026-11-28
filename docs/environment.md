# The LW-v0 environment

LW-v0 is one Linker workstation: two AR5 arms, each with an O6 hand and a wrist camera, a
table, and a top camera, as on the real robot. In Isaac Lab it is an ordinary
manager-based environment, registered as `Linker-LW-v0`; your scripts and tasks use it
through Isaac Lab's usual interfaces. This page explains what it contains and how its arms
and hands are commanded. [api.md](api.md) lists every name.

## A first script

```python
from linker_lab import workstation  # first: prepares the physics engine for Isaac Lab
import gymnasium as gym
from isaaclab.app import launch_simulation

cfg = workstation.LWv0EnvCfg()
cfg.scene.num_envs = 16
with launch_simulation(cfg, {"headless": True}):
    env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
    obs, _ = env.reset()
    print(obs["policy"].shape)  # (16, 52): joint positions and velocities of both units
    for _ in range(600):  # 10 s at 60 actions per second
        obs, reward, terminated, truncated, info = env.step(workstation.home_action(env))
    env.close()
```

Import linker-lab before Isaac Lab: `linker_lab.workstation` (and `arm`, `hand`) call
`linker_lab.bootstrap()`, which prepares the pinned physics engine (Newton 1.8 with
MuJoCo-Warp 3.17, newer than the one Isaac Lab 3.0 was released with) and must run before
Isaac Lab's simulation modules are imported. The first run converts LW-v0's files to USD
(about a second, cached in `~/.cache/linker-lab/usd/`) and compiles GPU kernels (about a
minute, cached too).

## What is in the scene

| Scene entity | What it is |
| --- | --- |
| `env.scene["left"]`, `env.scene["right"]` | An arm unit each: the AR5 arm, the O6 hand and the wrist camera, one articulation, starting at home |
| `env.scene["top_camera"]` | The ZED 2i on its mount, fixed |
| `env.scene["support"]` | The table and the column the arms are mounted on, fixed |

They are LW-v0's files from [linker-sim-assets](https://github.com/linker-bot/linker-sim-assets),
unchanged: masses, joint ranges, drives, collision shapes and contact settings. Within a unit
only different fingers collide with each other; everything else collides as on the real
robot. There is no floor (the playground adds one) and no camera sensor until you add one.

**Timing.** The physics runs at 480 Hz with collision detection every step. One action is
a **decision**, applied for 8 physics steps: 60 decisions per second (`env.step_dt` = 1/60 s).

**Frames.** Positions in the environment are in the **environment frame** (per environment,
`env.scene.env_origins` apart): x forward over the table, z up, the origin between the two
arm mounts, the table top 0.5 m below it. Each arm has its own **base frame**, its
articulation's root (`env.scene["left"].data.root_link_pose_w`), the frame the real
controller uses: arm pose commands are in it. Two points on each arm matter:

- the **TCP** (`workstation.TCP_BODY[side]`), the arm's tool point, which the real
  controller reports;
- the **palm point** (`workstation.palm_pose_w(robot)`, a world pose), 3.5 cm in front of
  the middle of the palm's surface: where an object sits in a grasp.

The arms are mounted on the sides of the column, so their base frames are turned against the
environment frame (rolled by about ±90° about x): convert world or environment poses into a
base frame with the base pose, as in the example below.

## Commanding the arms

The real AR5 is commanded through the xCore SDK in one of two ways, and each arm action
(`linker_lab.arm.ArmAction`) models both. The default environment has one per arm:
`left_arm`, `right_arm`, then the hands `left_hand`, `right_hand`, in this order in the
action vector.

**Goals** (`interpolation=True`, the default; the SDK's tracking mode). Each decision's
command is a goal. A jerk-limited trajectory generator moves toward the latest goal within
the real controller's limits, all seven joints arriving together. Its bounded profile search
compares feasible branches without a guarantee of global time optimality. A new goal
replaces the old one smoothly. Use it for scripted moves, teleoperation and policies that
choose where to go.

**Your own path** (`interpolation=False`; the SDK's servo mode). Each command is the next
point of a path you compute, one per decision; consecutive points are joined by a smooth
curve. `stream` says what a point is: `"points"` (7 positions), `"points_velocity"` (7
positions and 7 velocities) or `"samples"` (all 8 physics steps' positions, then their
velocities). Use it when you plan the motion yourself.

**Joint or pose commands.** A command is 7 joint positions [rad] (`command="joint"`, the
default) or a pose of the TCP or the palm point (`command="pose"`: position [m] and quaternion
x, y, z, w, in the arm's base frame), turned into joint positions each decision by
differential inverse kinematics.

```python
cfg.actions.left_arm = workstation.arm_action_cfg("left", command="pose", frame="palm")
cfg.actions.right_arm = workstation.arm_action_cfg("right", interpolation=False)
```

**Limits.** By default (`limits="real"`) the arm keeps the real controller's limits: joint
speeds of 0.32-0.49 rad/s, acceleration 10 rad/s², jerk 40 rad/s³, the joint ranges and the
wrist's octagonal (J6, J7) region (`arm.VELOCITY`, `arm.WRIST_OCTAGON`, ...). The palm point
therefore moves at most about 0.15 m/s. A goal or path point outside the ranges is moved
into them. With goals, the trajectory generator keeps every limit. With your own path, each
physics step's reference is limited in speed and acceleration, velocities you give are
limited in speed, and a jerk above the limit is counted but not changed. A path that reaches
a joint range or the wrist region too fast to stop in time is stopped at the edge, faster
than the acceleration limit: plan paths that slow down before the edge.
`ArmAction.limit_hits` (also `env.extras["arm_limit_hits"][side]`) counts each decision in
which a limit changed the command, so a policy that relies on impossible motion is visible.
`limits=None` turns the limits off (for your own path only).

**Drives.** The arm's joints are position-controlled with the real arm's torque caps (120 N·m
J1-J4, 100 N·m J5-J7). An arm pressed against the table lags its path, as the real one does.

A **non-finite value** (`nan`, `+inf` or `-inf`) in a command keeps that value of the previous command; after the
environment's first reset the previous command is home. An example: the left palm point moves 5 cm up while
everything else holds.

```python
from linker_lab import workstation
import gymnasium as gym
import torch
from isaaclab.app import launch_simulation
from isaaclab.utils.math import subtract_frame_transforms

cfg = workstation.LWv0EnvCfg()
cfg.scene.num_envs = 4
cfg.actions.left_arm = workstation.arm_action_cfg("left", command="pose", frame="palm")
with launch_simulation(cfg, {"headless": True}):
    env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
    env.reset()
    left = env.scene["left"]
    position, orientation = workstation.palm_pose_w(left)
    position = position + torch.tensor([0.0, 0.0, 0.05], device=env.device)  # 5 cm up
    base = left.data.root_link_pose_w.torch
    goal = torch.cat(subtract_frame_transforms(base[:, :3], base[:, 3:7], position, orientation), -1)
    action = torch.full((env.num_envs, env.action_manager.total_action_dim), torch.nan, device=env.device)
    action[:, :7] = goal  # left_arm first; the rest holds
    for _ in range(120):  # 2 s
        env.step(action)
    reached = workstation.palm_pose_w(left)[0]
    print(f"palm point {torch.linalg.norm(reached - position, dim=-1).max() * 1000:.1f} mm from its goal")
    env.close()
```

## Commanding the hands

Each hand action (`linker_lab.hand.HandAction`) takes six targets, one per motor: thumb
pitch, thumb yaw, index, middle, ring, pinky (`hand.SHAFTS`), in radians or, with
`hand_action_cfg(side, normalized=True)`, in [-1, 1] over each joint's range. The other five
finger joints follow through fixed couplings. `workstation.home_action(env)` converts home
angles to normalized inputs when that option is enabled. Non-finite values keep the
previous shaft target in both input modes; finite normalized values are clipped to [-1, 1].

The O6's motors drive through **self-locking gears**, and so does the model (motor model
`geared-shaft-v1`): a motor moves toward its target as fast as its torque allows, and a load
can stop it but never push it back. A finger pushed by an object deflects a little and
springs back when the object is gone, and a closed hand keeps its grasp without a command.
`hand.PRESETS` has the open, closed and pinch targets the playground uses.

## Observations, resets, episodes

The default environment observes the positions and velocities of the 13 driven joints of
each unit (group `policy`) and resets every environment to home on `env.reset()`. It has no
rewards and no terminations: an episode never ends by itself, so a script can drive the robot
for as long as it likes. Tasks add their own rewards and terminations. Tasks subclass `LWv0EnvCfg` and replace its parts; the
[task template](../templates/task/README.md) shows each one.

**Terminations reset inside `env.step()`.** When a task's termination term fires (a time
limit, an object dropped), Isaac Lab resets that environment in the same `env.step()`: the
robot and the objects go back home, the observation it returns is the reset one, and
`terminated` (or `truncated`, for time limits) is True. The task template adds a 5 s time
limit this way:

```python
from isaaclab.envs import mdp
from isaaclab.managers import TerminationTermCfg

cfg.terminations.time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
cfg.episode_length_s = 5.0
```

## Saving and restoring

`linker_lab.snapshot` saves the environment between two `env.step()` calls (one environment),
for example the moment the hand has closed on an object, and restores it later, in the same
process or a new one, without a reset: the arm continues its motion, the hand keeps its grip.

```python
from linker_lab import snapshot

def save_moment(env, path):
    snapshot.save(snapshot.capture(env), path)  # about 40 KB, plus the observations

def back_to(env, path):
    return snapshot.restore(env, snapshot.load(path))  # the observations at that moment
```

A snapshot restores into an environment made with the same linker-lab, the same package
versions and the same configuration (it checks, and refuses otherwise, before changing
anything). The control state, counters and simulated time come back exactly, the physics within
float32 rounding; after that, runs drift apart slowly, as any two GPU runs do. The
observations come back as they were captured, noise included, without being computed again.
Camera sensor buffers and model parameters changed while running (by randomization
events) are not restored.
[api.md](api.md#linker_labsnapshot) lists what is saved and what is refused.

## Objects

`linker_lab.objects` makes rigid objects from meshes: each mesh part collides as its convex
hull, the mass is yours (the centre of mass and inertia follow from it, or give them), and every
object gets the same contact material (friction 1). `on_table` gives the position that rests an
object on the table.

```python
import trimesh
from linker_lab import objects

mesh = trimesh.creation.box((0.05, 0.05, 0.05))  # or objects.load_mesh("your_mesh.stl")
parts = [(mesh.vertices, mesh.faces)]
cfg.scene.box = objects.object_cfg(
    parts, mass=0.1, pos=objects.on_table(parts, (0.4, 0.3)), prim_path="{ENV_REGEX_NS}/box"
)
```

Give each object its own `prim_path`. Read its pose with Isaac Lab's `env.scene["box"].data`. `objects.fixed_cfg` makes a fixed
object (a shelf, a bin) with the same material.

## Cameras

The cameras are off by default. `workstation.camera_cfg(name)` adds one as an Isaac Lab camera
sensor with the real camera's resolution, calibrated intrinsics, rate and image types:
`left_rgb`, `left_depth`, `right_rgb`, `right_depth` (the wrist cameras, 1280 x 800) and `top`
(colour and depth, 1280 x 720). `width` and `height` make another image size of the same aspect
ratio, for example 640 x 400 or 640 x 360; the intrinsics are scaled to it, so the image shows
the same view with fewer pixels. A size of another aspect ratio raises an error: on the real
cameras it is another mode, with its own calibration.

```python
cfg.scene.top = workstation.camera_cfg("top", width=640, height=360)
# in the loop: env.scene["top"].data.output["rgb"], shape (num_envs, height, width, 3)
```

Images are rendered by ray tracing on the GPU; each camera costs simulation speed.

## Many environments

`cfg.scene.num_envs` sets how many workstations run in parallel on the GPU. With the arms
moving within their limits, LW-v0 runs at roughly 28,000 environment steps per second with 400
environments on a current high-end desktop GPU (each step 1/60 s of simulated time). Each
decision also costs a fixed ~4.6 ms of kernel launches whatever the number of environments,
so a few environments run far below that.

## LW-v0's files

The robot's files come from linker-sim-assets, installed with linker-lab as the data package
`linker-lw-v0` from its release tag `v1.0.0` (`assemblies/lw_v0/` and its parts, MJCF and
URDF; `linker_lab.assets.path("lw_v0.xml")`). linker-sim-assets keeps LW-v0 the same robot
throughout 1.x: the files' paths, the names of its bodies, joints, sites, cameras and drives, its
physics, the controllers' constants, and the shape and place of its visual meshes. A 1.x release
changes nothing that is simulated, so tasks and policies written for LW-v0 keep working. A
change to LW-v0's physics or geometry comes only as a new robot or as linker-sim-assets 2.0.0,
adopted in a linker-lab release of its own.
