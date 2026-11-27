# API reference

This page is linker-lab's public API: the modules, names and members listed here. From
2.0.0 they follow semantic versions:

- a change that breaks code using them is a new major version;
- a name to be removed first warns for one minor release;
- a new version of LW-v0's files (the pinned `linker-lw-v0`) is a new minor version, its
  changes listed in the [changelog](../CHANGELOG.md).

Everything else is internal and may change in any release, whether or not its name starts
with an underscore. That includes the modules `engine`, `playground` and `cli`. Each public
module's `__all__` is exactly its list below, and a test keeps the two equal.

linker-lab is an Isaac Lab extension: an LW-v0 environment is an ordinary Isaac Lab
`ManagerBasedRLEnv`, and everything Isaac Lab provides on it (the scene's assets and their
`data`, the managers, sensors, learners) is Isaac Lab's API, not repeated here. linker-lab
promises the names it gives to that environment: the scene entities, the action terms, the
observation, event and termination terms listed below, and the gym id `Linker-LW-v0`. The
names of LW-v0's bodies, joints, sites and cameras are those of its files, which
linker-sim-assets keeps fixed throughout 1.x.

**Conventions.**
- Units: metres, radians, seconds, kilograms, newton metres.
- Quaternions: x, y, z, w (Isaac Lab 3.0's order).
- Frames: the **environment frame** of each environment (x forward over the table, z up,
  the origin between the two arm mounts, the table top 0.5 m below it) is the world frame
  shifted by `env.scene.env_origins`. Names ending in `_w` (Isaac Lab's `root_pos_w`,
  `palm_pose_w`) are world poses: subtract the origins for environment-frame positions. Arm
  pose commands are in **each arm's base frame**, its articulation's root, the frame the
  real controller uses.
- A **decision** is 1/60 s: one action per decision, held for 8 physics steps of 1/480 s.
- **Import linker-lab before Isaac Lab.** Importing `workstation`, `arm`, `hand` or `snapshot` calls
  `linker_lab.bootstrap()`, which prepares the pinned physics engine and must run before
  Isaac Lab simulation modules are imported. `objects`, `physics`, `gallery` and `assets` import
  nothing from Isaac Lab, but `objects.object_cfg`, `objects.fixed_cfg` and
  `physics.simulation_cfg` do when called: call `bootstrap()` (or import `workstation`) first.

```python
from linker_lab import workstation  # first: prepares the engine
import gymnasium as gym
from isaaclab.app import launch_simulation

cfg = workstation.LWv0EnvCfg()
cfg.scene.num_envs = 16
with launch_simulation(cfg, {"headless": True}):
    env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
    env.reset()
    for _ in range(60):
        env.step(workstation.home_action(env))
```

## `linker_lab`

### `bootstrap()`

Prepares the pinned physics engine (Newton 1.8 with MuJoCo-Warp 3.17) for Isaac Lab 3.0.
Call it before importing Isaac Lab; it raises if Isaac Lab's simulation modules are already
imported (Isaac Lab's own command line, `isaaclab train`, may be imported first). Calling it
again after a successful call does nothing. If a patch fails, repair the environment and
restart the Python process before retrying; partial initialization cannot be reused.
The modules that need it call it when imported.

### `__version__`

linker-lab's version, e.g. `"2.0.0"`.

## `linker_lab.workstation`

The LW-v0 scene, its frames, the default environment and the action configurations.

### `LWv0EnvCfg`

The default environment, an Isaac Lab `ManagerBasedRLEnvCfg`, registered as `Linker-LW-v0`.
Tasks subclass it and replace its parts.

- `scene`: an `LWv0SceneCfg` with `num_envs=1` and `env_spacing=3.0`.
- `sim`: `physics.simulation_cfg()`: 480 Hz with LW-v0's solver settings.
- `decimation`: `physics.DECIMATION` (8 physics steps per decision).
- `episode_length_s`: 10 s; used only by a time-limit termination, which the default
  environment does not have.
- `actions`: an `ActionsCfg`.
- `observations`: an `ObservationsCfg`.
- `events`: an `EventsCfg`.
- `rewards`: a `RewardsCfg` (none).
- `terminations`: a `TerminationsCfg`.

### `LWv0SceneCfg`

The workstation's four parts, spawned from LW-v0's files unchanged, at their poses in
`lw_v0.xml`. No floor and no cameras: add cameras with `camera_cfg`, objects with
`linker_lab.objects`.

- `left`, `right`: the arm units (AR5 arm, O6 hand, wrist camera), one articulation each,
  starting at home. Their drives' gains and torque caps come from the files.
- `top_camera`: the ZED 2i on its mount, fixed to the world.
- `support`: the table and the arm-base column, fixed to the world.

### `ActionsCfg`

One action term per arm and hand, in this order in the action vector.

- `left_arm`, `right_arm`: `arm_action_cfg(side)`: 7 joint goals [rad] each, through the
  trajectory generator within the real limits.
- `left_hand`, `right_hand`: `hand_action_cfg(side)`: 6 shaft targets [rad] each.

### `ObservationsCfg`

- `PolicyCfg`, `policy`: one observation group (an Isaac Lab `ObservationGroupCfg`; subclass
  `PolicyCfg` to add terms), concatenated: the positions [rad] of the 13 driven
  joints of the left unit (`ARM_JOINTS["left"]`, then `HAND_JOINTS["left"]`), the same for
  the right unit, then their velocities [rad/s] (terms `left_joint_pos`, `right_joint_pos`,
  `left_joint_vel`, `right_joint_vel`).

### `EventsCfg`

- `reset_to_home`: on reset, every asset back to its default state (the robot at home)
  and every drive target to home.

### `RewardsCfg`

No rewards: LW-v0 is a workstation, not a task. Tasks add their own terms.

### `TerminationsCfg`

No terminations: episodes never end by themselves, so the environment runs as long as you
step it and only resets when you call `env.reset()`. A task that wants episodes adds its own
terms, e.g. a time limit, `time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)`
with the task's `episode_length_s`. When a termination fires, Isaac Lab resets that
environment inside the same `env.step()`: the returned observation is the reset one, and
`terminated` or `truncated` (time limits) is True.

### `arm_action_cfg(side, *, frame="tcp", **options)`

An `arm.ArmActionCfg` for the arm `side` (`"left"` or `"right"`), with its joints and
bodies filled in. `options` are the configuration's fields (`command`, `interpolation`,
`stream`, `limits`, `ik_damping`, and Isaac Lab's `class_type` for a subclass of
`ArmAction`). For pose commands, `frame` is the commanded frame:
`"tcp"`, the arm's tool point (what the real controller reports), or `"palm"`, the palm
point of the hand (`PALM_OFFSET`).

### `hand_action_cfg(side, **options)`

A `hand.HandActionCfg` for the hand `side`, with its joints and motor filled in; `options`:
e.g. `normalized=True`.

### `home_action(env)`

The action (N, dim) that holds every arm and hand at home. For arms with joint commands
(any stream but `"samples"`) and every hand, converting home angles to normalized hand
inputs when configured; raises for other arm commands.

### `palm_pose_w(robot)`

World pose of an arm unit's palm point: position (N, 3) and quaternion (N, 4).
`robot` is the unit's articulation, `env.scene["left"]` or `env.scene["right"]`.

### `camera_cfg(name, width=None, height=None)`

An Isaac Lab `CameraCfg` for the camera stream `name` (a key of `CAMERAS`): the real
camera's resolution, intrinsics, rate and image types, rendered by Newton's Warp renderer.
`width` and `height` [px], given together, set another image size of the real camera's
aspect ratio, and the intrinsics are scaled to it; another aspect ratio (or only one of the
two) raises `ValueError`. Add it to a scene: `cfg.scene.top = camera_cfg("top")`.

### `body_path(part, body)`

The path of a rigid body below a part's root prim (part: `"left"`, `"right"`,
`"top_camera"`, `"support"`), for sensors that need a prim path, e.g.
`f"{{ENV_REGEX_NS}}/left/{body_path('left', 'hand_lh_index_distal')}"`.

### `SIDES`

`("left", "right")`.

### `ARM_JOINTS`

Side → the seven arm joints J1..J7 (from `arms/ar5_08/controller.yaml`).

### `HAND_JOINTS`

Side → the six driven hand joints (thumb pitch, thumb yaw, index, middle, ring, pinky;
from `hands/o6/motor.yaml`). The other five hand joints follow them through fixed
couplings in LW-v0's files.

### `TCP_BODY`

Side → the arm's tool-point body (`frame="tcp"`).

### `PALM_BODY`

Side → the hand's base body, which carries the palm point.

### `PALM_OFFSET`

The palm point in the `PALM_BODY` frame: (0.055, 0, 0.055) m, the centre of the palm,
3.5 cm in front of its surface.

### `CAMERAS`

The camera streams: name → (part, optical body, rate [Hz], image types): `left_rgb`,
`left_depth`, `right_rgb`, `right_depth` (the wrist cameras, Gemini 335L: colour 60 Hz,
depth 30 Hz) and `top` (ZED 2i, colour and depth, 60 Hz).

## `linker_lab.arm`

The AR5 arm: how it is commanded, its servo model and its motion limits. The real arm is
commanded through the xCore SDK in one of two ways, and `ArmAction` models both:

- **Interpolation on** (`interpolation=True`, default; the SDK's PP tracking): each command
  is a **goal**. A jerk-limited trajectory generator moves toward the latest goal,
  time-optimal under the limits, all seven joints arriving together, replanned every
  decision.
- **Interpolation off** (`interpolation=False`; the SDK's CSP servo): each command is the
  **next point of your own path**, joined to the previous one by a cubic Hermite curve.

Then a servo model: LW-v0's drives track the per-step reference with velocity
feed-forward, and the PD torque is kept 1 N·m below each joint's torque cap (120 N·m J1-J4,
100 N·m J5-J7). With the real limits on, every command and reference is kept within the
velocity limits, the joint ranges and the wrist octagon. Acceleration is limited except
when a boundary stop takes priority (see below); each
intervention is counted (`ArmAction.limit_hits`).

### `ArmActionCfg`

The configuration of an `ArmAction` (an Isaac Lab `ActionTermCfg`; `asset_name` is the
side). Make it with `workstation.arm_action_cfg`.

- `joint_names`: the arm joints J1..J7.
- `command`: `"joint"` (default): 7 joint positions [rad]. `"pose"`: a position [m] and a
  quaternion (7 values) of the commanded frame in the arm's base frame, turned into joint
  positions once per decision by damped least-squares IK.
- `interpolation`: `True` (default): the command is a goal. `False`: the next point of your
  path.
- `stream`: with `interpolation=False`: `"points"` (default; 7 positions, the point's
  velocity is the finite difference to the previous point), `"points_velocity"`
  (positions and velocities, 14 values) or `"samples"` (the 8 per-step positions, then the
  8 per-step velocities, 112 values, subject to `limits`).
- `limits`: `"real"` (default): the real controller's limits. `None`: no limits; only with
  `interpolation=False`.
- `body_name`, `body_offset`: pose commands: the commanded frame, a point in a body's
  frame. Filled in by `arm_action_cfg`.
- `ik_damping`: the damping of the IK step (0.01).

With `limits="real"` and streamed paths, supplied velocities are bounded by the speed limits. A boundary stop
prioritizes the joint ranges and wrist octagon over acceleration and jerk, counts the
intervention, and updates the reference state from the accepted motion. Jerk violations
in streamed paths are counted rather than enforced.

A reset projects measured joint positions into the feasible reference region, so solver
drift outside a limit cannot leave the stream stuck there. Physical joint positions are
not changed by the action term's reset.

### `ArmAction`

The Isaac Lab `ActionTerm` made from an `ArmActionCfg`; get it with
`env.action_manager.get_term("left_arm")`.

- `action_dim`: the command's length (7, 14 or 112).
- `raw_actions`, `processed_actions`: the last command (N, action_dim); a non-finite value
  is replaced by the previous command's. The limits act on the joint goal or path point
  made from it, counted in `limit_hits`.
- `reference`: the current physics step's reference `(q_ref, v_ref)`, (N, 7) each.
- `limit_hits`: per environment, the number of decisions since the last reset in which a
  limit changed the command or one of its physics steps' references, or the jerk was
  exceeded. Also in `env.extras["arm_limit_hits"][side]`.
- `get_state()`, `set_state(state)`: everything the term carries from one decision to the
  next (the generator's and the stream's state, this decision's references and servo targets,
  the last command, `limit_hits`, the IK's command), as a dict of copied tensors and ints, and
  its restoration in place; `set_state` raises `ValueError` before changing anything if the
  keys, shapes or dtypes differ. Used by `linker_lab.snapshot`.
- `process_actions(actions)`, `apply_actions()`, `reset(env_ids=None)`: the `ActionTerm`
  interface, called by Isaac Lab's action manager.

### `VELOCITY`

Maximum joint velocity [rad/s], J1..J7: (0.39, 0.32, 0.39, 0.39, 0.49, 0.49, 0.49), the
xCore SDK motion profile of the AR5_08 (`arms/ar5_08/controller.yaml`).

### `ACCELERATION`

Maximum joint acceleration [rad/s²], J1..J7 (10 each).

### `JERK`

Maximum joint jerk [rad/s³], J1..J7 (40 each).

### `WRIST_OCTAGON`

The region the real controller allows for (J6, J7) [rad]: a convex octagon, vertices in
order. The J6 and J7 ranges in LW-v0's files (±60°, ±90°) are its bounding box.

### `TORQUE_GUARD_MARGIN`

The servo model keeps each drive's torque this far [N·m] below its cap (1.0).

## `linker_lab.hand`

The O6 hand. Its six motors drive through self-locking gears: a shaft moves toward its
target, as fast as its torque allows, and a load can stop it but never push it back. A
pushed finger deflects through the transmission's compliance and springs back, and a grasp
holds without a command. The model (`geared-shaft-v1`, defined in linker-sim-assets'
README, "Robot controllers") reads its constants from `hands/o6/motor.yaml`.

### `HandActionCfg`

The configuration of a `HandAction` (an Isaac Lab `ActionTermCfg`). Make it with
`workstation.hand_action_cfg`.

- `joint_names`: the six actuated joints, in `SHAFTS` order.
- `motor`: the hand's motor constants (`hands/o6/motor.yaml`). Filled in by
  `hand_action_cfg`.
- `normalized`: `False` (default): targets in radians. `True`: in [-1, 1] over each joint's
  range.

### `HandAction`

The Isaac Lab `ActionTerm` made from a `HandActionCfg`. A non-finite target keeps that
shaft's previous target; the motor model never moves a shaft outside the joint's range.

- `action_dim`: 6.
- `raw_actions`: the last command (N, 6).
- `processed_actions`: the shaft targets in effect (N, 6) [rad], as commanded.
- `shaft`: the motor shaft positions (N, 6) [rad].
- `get_state()`, `set_state(state)`: the last command, the targets in effect, the guarded
  drive targets and the motor's state (shafts, last measured joints, this decision's per-step
  targets); as on `ArmAction`.
- `process_actions(actions)`, `apply_actions()`, `reset(env_ids=None)`: the `ActionTerm`
  interface.

### `SHAFTS`

The six actuated joints, in action order: thumb pitch, thumb yaw, index, middle, ring,
pinky.

### `PRESETS`

Shaft targets [rad]: `"open"`, `"close"`, `"pinch"`.

### `THUMB_OPPOSE_YAW`

The thumb yaw [rad] that turns the thumb opposite the index finger (1.04).

### `thumb_oppose(current)`

The shaft targets `current` (N, 6) with the thumb turned opposite the index finger.

## `linker_lab.objects`

Rigid objects: import from meshes, their contact material, placement on the table. An
object is one rigid body of one or more mesh **parts** (the pieces of a convex
decomposition, or one mesh); each part collides as its convex hull. Files are written once
into the user cache, named by their content.

```python
import linker_lab
linker_lab.bootstrap()  # object_cfg imports Isaac Lab
import trimesh
from linker_lab import objects

mesh = trimesh.creation.box((0.06, 0.06, 0.06))  # or objects.load_mesh("your_mesh.stl")
box = [(mesh.vertices, mesh.faces)]
pos = objects.on_table(box, (0.35, 0.0))
object_cfg = objects.object_cfg(box, mass=0.1, pos=pos)  # cfg.scene.box = object_cfg
```

### `object_cfg(parts, mass, *, com=None, inertia=None, pos=(0.35, 0, 0), rot=(0, 0, 0, 1), prim_path="{ENV_REGEX_NS}/object", visual=None)`

An Isaac Lab `RigidObjectCfg` for the object at `pos` / `rot` in the environment frame.
Give each object of a scene its own `prim_path` (e.g. `"{ENV_REGEX_NS}/box"`).
Given only `mass` [kg], the centre of mass and inertia are those of a uniform density over
the parts' hulls; give `com` and `inertia` (3×3, about the centre of mass) to set them.
`visual` (a `Part`) is shown instead of the parts and never collides.

### `fixed_cfg(parts, *, pos=(0, 0, 0), rot=(0, 0, 0, 1), prim_path="{ENV_REGEX_NS}/fixed")`

An Isaac Lab `AssetBaseCfg` for a fixed object (a shelf, a bin, a floor): its parts collide
with the object material but never move.

### `write_usd(parts, mass, com=None, inertia=None, visual=None)`

The object's USD file, written into the user cache (e.g. for Isaac Lab's
`MultiUsdFileCfg`, a different object per environment). `mass=None` writes a fixed object.

### `load_mesh(path, scale=1.0)`

A mesh file (STL, OBJ, PLY, ...) as a list of one `Part`, in metres after `scale`.

### `mass_properties(parts, mass)`

Centre of mass (3,) and inertia about it (3×3) [kg m²] for a uniform density over the
parts' hulls.

### `on_table(parts, xy, rot=(0, 0, 0, 1), clearance=0.001)`

The position that puts the object, turned by `rot`, `clearance` above the table top at
`xy`. Released there, it drops onto the table.

### `table()`

The table's collision box in the environment frame: `top` (height), `center` and
`half_size` (x, y) [m].

### `Part`

One mesh part: `(vertices, faces)`, vertices (N, 3) in metres in the object frame,
triangle indices (M, 3).

### `MATERIAL`

The objects' contact material, as in MJCF: `friction` 1, `torsional_friction` 0.005,
`rolling_friction` 0, `condim` 4, `solref`, `solimp`, `margin`, `gap`.

## `linker_lab.physics`

The physics LW-v0 is simulated with. The robot's own properties (masses, ranges, drives,
collision shapes, contact settings) are in LW-v0's files; the time step and the solver
settings are read from `lw_v0.xml`, so Isaac Lab runs what every MuJoCo tool loading that
file runs.

### `PHYSICS_HZ`

Physics steps per second (480). Collision detection runs every step.

### `DECISION_HZ`

Decisions (actions) per second (60).

### `DECIMATION`

Physics steps per decision (8).

### `SOLVER`

The solver settings (names as in MJCF `<option>`): `integrator`, `solver`, `iterations`,
`ls_iterations`, `cone`, `impratio`, `multiccd`.

### `simulation_cfg(device="cuda:0")`

Isaac Lab's `SimulationCfg` with these settings: MuJoCo-Warp solves the dynamics,
Newton's collision pipeline finds the contacts every physics step.

## `linker_lab.snapshot`

Save an environment between two `env.step()` calls and restore it later, in the same process
or in a new one with the same linker-lab, packages and configuration, without a reset: after a
restore, every input of the next `env.step()` is what it was at the capture. One environment
(`num_envs == 1`). Isaac Lab's own `env.reset_to()` resets first, which restarts `ArmAction`'s
and `HandAction`'s controllers from rest (a moving arm replans from standstill, a holding hand
lets go).

```python
from linker_lab import snapshot, workstation

def try_again(env, attempts):
    """Run each attempt (a function of env) from the same moment."""
    moment = snapshot.capture(env)
    for attempt in attempts:
        snapshot.restore(env, moment)
        attempt(env)
```

Saved: every articulation's and rigid object's root pose and velocity, every articulation's
joint positions, velocities and drive targets; each action term's state (`get_state()`); the
action manager's last two actions; the step counters, episode lengths, simulated time, reward
sums, the random generators (torch, NumPy, Python) and the observations as last computed
(`env.obs_buf`, noise included). Not saved: contacts and the physics
solver's buffers (nothing reads them before rewriting them; the solver starts cold), camera
sensor buffers, renderer and viewer, and model parameters changed while running (masses, friction or
gains set by an event): restoring where they differ gives a different simulation.

Accuracy: the action terms' state, the drive targets, the counters and the simulated time are
restored exactly; the physics continues within float32 rounding (measured about 1e-7 rad and
1e-7 m over 8 decisions, a hand holding an object and an arm moving, in each arm mode). Further
on, a restored run drifts from the uninterrupted one as any two GPU runs from one state do:
contact sums are not bit-reproducible.

Calling `capture` or `restore` inside `env.step()` is not supported.

### `capture(env)`

The state of `env` (the unwrapped `ManagerBasedRLEnv`) as a `Snapshot`, copied to the CPU
(size and copy time depend on the configured observations). Raises `SnapshotError` for
what it cannot account for:
`num_envs != 1`; an action term without `get_state()` / `set_state()` (add them to a term of
your own); an observation term with a history; an event term at intervals; command,
curriculum or recorder terms; rigid object collections, deformable objects, cables or surface
grippers; a sensor other than a camera; a permanent external wrench.

### `restore(env, snapshot)`

Writes the snapshot into `env` without a reset and returns the observations at the snapshot
(also stored in `env.obs_buf`). First everything is checked: the same refusals as `capture`,
the header (linker-lab's commit when running from a checkout, the versions of linker-lab,
linker-lw-v0, Isaac Lab, Newton, MuJoCo-Warp, Warp and torch, the configuration fingerprint,
the scene's entities, the action terms and their sizes) and every tensor's shape and dtype.
A mismatch raises `SnapshotError` naming it, with nothing changed.

The observations are copied back as they were captured, noise included; the observation
terms are not computed again, so no random generator advances.

### `save(snapshot, path)`

Writes the snapshot to one file.

### `load(path)`

Reads a file written by `save`; no code in the file is executed (`torch.load` with
`weights_only=True`). A file of another snapshot layout (`header["schema"]`) is refused;
`restore` checks the rest against the environment.

### `Snapshot`

A captured environment, frozen.

- `header`: a dict: `schema`, `linker_lab_commit`, `packages`, `config` (the sha256 of the
  environment configuration, leaving out only `viewer` and `sim.visualizer_cfgs`, which do not
  change what `env.step()` computes), `scene`, `actions`, `created` (UTC).
- `state`: the saved state; its layout is internal and may change with the schema.

### `SnapshotError`

Raised for every refusal and failure. Its attribute `physics_modified` is False when the
environment was not changed (every refusal and mismatch), True when writing had begun and
failed: discard that environment's process.

## `linker_lab.gallery`

The policy gallery: pretrained policies that run on LW-v0. An entry is a folder
`policy-gallery/<entry>/` of this repository with a `manifest.yaml` (name, version, title, description,
license, the linker-lab versions it runs on as `linker_lab`, and its `artifacts`: file name
→ `url` and `sha256`) and a `play.py` whose `run(artifacts)` opens the entry's scene.
An editable install reads the entries from the clone's `policy-gallery/`; a wheel carries
them inside the package. The weights and data are downloads ([the files](gallery.md#the-files)).

### `entries()`

The entries' names.

### `manifest(name)`

An entry's manifest as a dict, checked: every field present, every artifact with a URL and
a sha256, a valid version range.

### `info(name)`

A readable summary of an entry: title, description, licence, the linker-lab versions it
runs on, its files' URLs.

### `supported(manifest)`

Whether this linker-lab's version is in the manifest's range.

### `run(name)`

Opens the entry: checks the version range, fetches its artifacts, calls its `play.run`.
Returns when the window is closed.

### `load(name)`

The entry's `play.py`, loaded as a new module on each call.

### `fetch(name)`

The entry's artifacts in the user cache, name → path: downloaded if missing, each checked
against its sha256 before it is moved into the cache (a mismatch raises `GalleryError` and
keeps no file).

### `artifact_path(name, artifact)`

Where an artifact is kept in the user cache. A file put there by hand is used if its
sha256 matches, e.g. to try an entry before its files are published.

### `GalleryError`

Raised when an entry is missing, malformed, not for this linker-lab, or its files do not
match.

## `linker_lab.assets`

LW-v0's robot files: the data package `linker-lw-v0` from linker-sim-assets, installed
with linker-lab.

### `NAME`

`"lw_v0"`.

### `root()`

The installed `assemblies/lw_v0/` folder.

### `path(name)`

The path of one of LW-v0's files, e.g. `path("lw_v0.xml")`; raises `FileNotFoundError` if
it has no such file.

## Command line

| Command | What it does |
| --- | --- |
| `linker-lab doctor` | Checks the installation: versions, the GPU, the engine patches, one simulated second |
| `linker-lab play [--selftest [--out DIR]]` | The playground: drive LW-v0 by hand; `--selftest` runs a scripted session and reports |
| `linker-lab gallery list` | The gallery's entries |
| `linker-lab gallery info ENTRY` | An entry's description and licences |
| `linker-lab gallery run ENTRY` | Open an entry in the viewer |
