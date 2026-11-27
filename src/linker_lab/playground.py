"""The playground: drive LW-v0 by hand in Isaac Lab's Newton viewer (``linker-lab play``).

The scene is the default environment (``linker_lab.workstation``) with a 5 cm box on the
table and a floor at the foot of the support, so objects knocked off the table stay in
view. You move a goal for the palm point (``workstation.PALM_OFFSET``) of one arm at a
time; the arm follows through ``ArmAction(command="pose", frame="palm")`` with
interpolation on: it moves as the real arm does in the xCore SDK's tracking mode, within
the real controller's limits.

=====================  ==============================================================
Mouse                  fly the camera; right-drag pulls objects; drag the gizmo to
                       move or turn the goal
I/K, J/L, U/O          move the goal forward/back, left/right, up/down (relative to
                       the camera's view)
Shift + the same keys  turn the goal about the palm point (I/K pitch, J/L yaw, U/O roll)
Tab                    switch arm
1 / 2 / 3 / 4          hand open / close / pinch / thumb oppose (``hand.PRESETS``)
R                      reset
=====================  ==============================================================

The viewer keeps W/A/S/D, Q/E and the arrows (camera), Space (pause), H (hide the panel),
F (frame the scene) and Esc (quit) for itself.

Keys move the goal like a jog on a teach pendant: at :data:`JOG_SPEED` while held, and
never more than :data:`LEAD` ahead of the palm point, so the arm stops soon after the key
is released (the arm's own speed is set by its joint limits and is often below the jog
speed). The panel shows the arm TCP pose (the frame the real controller reports and most
learned policies command), how far the palm point is from its goal, limit interventions,
contacts and the simulation speed.

The viewer is a :class:`PlaygroundVisualizer`, a subclass of Isaac Lab's Newton GL
visualizer selected through the visualizer config; Isaac Lab is not patched.

Importing this module applies ``linker_lab.bootstrap()``, so import it before Isaac Lab.
"""

from __future__ import annotations

import linker_lab

linker_lab.bootstrap()

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh
import warp as wp
from isaaclab.utils import configclass
from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonGLVisualizerCfg
from scipy.spatial.transform import Rotation

from linker_lab import hand, objects, physics, workstation

JOG_SPEED = (0.1, 0.5)
"""Speed of the goal while a key is held: [m/s], [rad/s]."""
LEAD = (0.05, 0.3)
"""Keys move the goal at most this far from the palm point: [m], [rad]."""
FLOOR_TOP = -1.267
"""Height of the floor [m] in the environment frame: the foot of the support (the lowest
point of its visual mesh; the table's collision box ends 3 mm lower)."""
BOX = {"edge": 0.05, "mass": 0.1, "xy": (0.42, -0.25)}
"""The box on the table: edge [m], mass [kg], position [m], below the right hand."""
CAMERA = {"eye": (1.2, -0.85, 0.2), "lookat": (0.3, 0.0, -0.35)}
"""Initial viewer camera [m], environment frame: in front of the table, to the right."""

KEYS = ("i", "k", "j", "l", "u", "o")
ACTIONS = {
    "tab": "switch",
    "1": "open",
    "2": "close",
    "3": "pinch",
    "4": "thumb_oppose",
    "r": "reset",
}


# ---------------------------------------------------------------------------- viewer


class PlaygroundVisualizer(NewtonGLVisualizer):
    """Isaac Lab's Newton GL visualizer with one hook: ``draw(viewer)`` is called every
    rendered frame, after the scene and Isaac Lab's markers are logged (gizmo, lines)."""

    draw: Callable | None = None

    def _render_markers(self, num_envs: int) -> None:
        super()._render_markers(num_envs)
        if self.draw is not None:
            self.draw(self._viewer)


@configclass
class PlaygroundVisualizerCfg(NewtonGLVisualizerCfg):
    class_type: type = PlaygroundVisualizer
    streaming_view: bool = False
    window_width: int = 1600
    window_height: int = 900


# ---------------------------------------------------------------------------- environment


def env_cfg() -> workstation.LWv0EnvCfg:
    """The default LW-v0 environment with palm-point pose commands for both arms, a box on
    the table, the floor, no time limit and the playground viewer."""
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = 1
    for side in workstation.SIDES:
        arm = workstation.arm_action_cfg(side, command="pose", frame="palm")
        setattr(cfg.actions, f"{side}_arm", arm)
    table = objects.table()
    floor = trimesh.creation.box((6.0, 6.0, 0.05))
    cfg.scene.floor = objects.fixed_cfg(
        [(floor.vertices, floor.faces)],
        pos=(*table["center"], FLOOR_TOP - 0.025),
        prim_path="{ENV_REGEX_NS}/floor",
    )
    box = trimesh.creation.box((BOX["edge"],) * 3)
    parts = [(box.vertices, box.faces)]
    cfg.scene.box = objects.object_cfg(
        parts, BOX["mass"], pos=objects.on_table(parts, BOX["xy"]), prim_path="{ENV_REGEX_NS}/box"
    )
    cfg.sim.visualizer_cfgs = PlaygroundVisualizerCfg(**CAMERA)
    return cfg


# ---------------------------------------------------------------------------- jogging


def camera_axes(front) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward (the view direction, horizontal), right and up for keys relative to a camera
    looking along ``front``."""
    up = np.array([0.0, 0.0, 1.0])
    forward = np.array([front[0], front[1], 0.0])
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])  # looking straight down
    return forward, np.cross(forward, up), up


def jog(goal, palm, keys: set[str], front, dt: float):
    """The goal after ``dt`` with ``keys`` held (``"shift"`` turns instead of moving).

    ``goal`` and ``palm`` are (position (3,), rotation matrix (3, 3)) in the world frame. A
    step that would put the goal more than :data:`LEAD` from the palm point is not taken,
    unless it brings the goal closer.
    """
    forward, right, up = camera_axes(front)
    p, R = goal
    if "shift" in keys:
        axes = {"i": right, "k": -right, "j": up, "l": -up, "u": forward, "o": -forward}
        w = sum((axes[k] for k in KEYS if k in keys), np.zeros(3))
        if not w.any():
            return goal
        R_new = Rotation.from_rotvec(w * JOG_SPEED[1] * dt).as_matrix() @ R
        angle = lambda Ra: Rotation.from_matrix(Ra @ palm[1].T).magnitude()  # noqa: E731
        if angle(R_new) > LEAD[1] and angle(R_new) >= angle(R):
            return goal
        return p, R_new
    axes = {"i": forward, "k": -forward, "l": right, "j": -right, "u": up, "o": -up}
    v = sum((axes[k] for k in KEYS if k in keys), np.zeros(3))
    if not v.any():
        return goal
    p_new = p + v * JOG_SPEED[0] * dt
    distance = lambda pa: np.linalg.norm(pa - palm[0])  # noqa: E731
    if distance(p_new) > LEAD[0] and distance(p_new) >= distance(p):
        return goal
    return p_new, R


# ---------------------------------------------------------------------------- playground


class Playground:
    """The interactive loop around an LW-v0 environment made from :func:`env_cfg`.

    Call :meth:`step` with the keys held (``keys_down()`` in the viewer) once per decision.
    """

    def __init__(self, env):
        import torch

        self.torch = torch
        self.env = env
        self.viz = next(v for v in env.sim.visualizers if isinstance(v, PlaygroundVisualizer))
        self.active = "right"
        self.pressed: set[str] = set()
        self.goal: dict[str, wp.transform] = {}
        self.hand: dict[str, torch.Tensor] = {}
        self.speed = 0.0  # simulated seconds per wall-clock second
        self.busy = 0.0  # wall-clock seconds spent per decision, without the pacing sleep
        self._clock = (time.perf_counter(), 0)  # (wall time, decisions) at the last speed reading
        self._paced = (time.perf_counter(), 0)
        self.decisions = 0
        self.frames = 0
        self.peak_contacts = 0
        self._joints = {
            s: env.scene[s].find_joints(list(workstation.HAND_JOINTS[s]), preserve_order=True)[0]
            for s in workstation.SIDES
        }
        self._tcp = {
            s: env.scene[s].find_bodies(workstation.TCP_BODY[s])[0][0] for s in workstation.SIDES
        }
        self.reset_goals()
        self.viz.draw = self._draw
        self.viz.register_ui_callback(self._panel, position="panel")

    # -- state

    def palm(self, side):
        """World pose of the palm point: position (3,), rotation matrix (3, 3)."""
        pos, quat = workstation.palm_pose_w(self.env.scene[side])
        return pos[0].double().cpu().numpy(), Rotation.from_quat(quat[0].cpu().numpy()).as_matrix()

    def goal_pose(self, side):
        t = self.goal[side]
        R = Rotation.from_quat(np.array(t.q, dtype=float)).as_matrix()
        return np.array(t.p, dtype=float), R

    def set_goal(self, side, p, R):
        self.goal[side] = wp.transform(wp.vec3(*p), wp.quat(*Rotation.from_matrix(R).as_quat()))

    def reset_goals(self):
        """Goals at the palm points, hands at the home pose."""
        for side in workstation.SIDES:
            self.set_goal(side, *self.palm(side))
            robot = self.env.scene[side]
            self.hand[side] = robot.data.default_joint_pos.torch[0, self._joints[side]].clone()

    def reach_error(self, side) -> tuple[float, float]:
        """Distance [m] and angle [rad] from the palm point to its goal."""
        (p, R), (gp, gR) = self.palm(side), self.goal_pose(side)
        return float(np.linalg.norm(gp - p)), float(Rotation.from_matrix(gR @ R.T).magnitude())

    def tcp_pose(self, side):
        """Arm TCP in the arm's base frame: position [m], extrinsic XYZ Euler angles [rad]
        (the xCore SDK's convention)."""
        robot = self.env.scene[side]
        root = robot.data.root_link_pose_w.torch[0].cpu().numpy()
        body = robot.data.body_link_pose_w.torch[0, self._tcp[side]].cpu().numpy()
        R_root = Rotation.from_quat(root[3:])
        pos = R_root.inv().apply(body[:3] - root[:3])
        return pos, (R_root.inv() * Rotation.from_quat(body[3:])).as_euler("xyz")

    def action(self):
        """The environment action: both palm-point goals in their arms' base frames, then
        both hands' shaft targets."""
        torch = self.torch
        arms = []
        for side in workstation.SIDES:
            root = self.env.scene[side].data.root_link_pose_w.torch[0].cpu().numpy()
            R_root = Rotation.from_quat(root[3:])
            p, R = self.goal_pose(side)
            pos = R_root.inv().apply(p - root[:3])
            quat = (R_root.inv() * Rotation.from_matrix(R)).as_quat()
            arms.append(torch.tensor(np.concatenate((pos, quat)), dtype=torch.float32))
        parts = [*arms, *(self.hand[s].cpu() for s in workstation.SIDES)]
        return torch.cat(parts)[None].to(self.env.device)

    # -- input

    def handle(self, keys: set[str]):
        """Apply one decision's input: ``keys`` are the keys held now."""
        pressed, self.pressed = keys - self.pressed, set(keys)
        side = self.active
        for key in pressed & ACTIONS.keys():
            name = ACTIONS[key]
            if name == "switch":
                self.active = "left" if side == "right" else "right"
            elif name == "reset":
                self.env.reset()
                self.reset_goals()
            elif name == "thumb_oppose":
                self.hand[side] = hand.thumb_oppose(self.hand[side][None])[0]
            else:
                self.hand[side] = self.torch.tensor(hand.PRESETS[name], device=self.env.device)
        if keys & set(KEYS):
            front = self.viz._viewer.camera.get_front() if self.viz._viewer else (1.0, 0.0, 0.0)
            side = self.active
            goal = jog(self.goal_pose(side), self.palm(side), keys, front, self.env.step_dt)
            self.set_goal(side, *goal)

    def keys_down(self) -> set[str]:
        """The playground keys held in the viewer window."""
        from pyglet.window import key

        keys = {k for k in (*KEYS, *ACTIONS) if self.viz.is_key_down(k)}
        if self.viz.is_key_down(key.LSHIFT) or self.viz.is_key_down(key.RSHIFT):
            keys.add("shift")
        return keys

    # -- loop

    def step(self, keys: set[str], realtime: bool = True):
        """One decision: input, the environment step, speed reading, pacing to real time."""
        start = time.perf_counter()
        self.handle(keys)
        _, _, terminated, _, _ = self.env.step(self.action())
        if bool(terminated[0]):  # reset from the viewer's own Reset button
            self.reset_goals()
        self.decisions += 1
        now = time.perf_counter()
        self.busy = 0.9 * self.busy + 0.1 * (now - start) if self.busy else now - start
        if now - self._clock[0] > 0.5:
            self.speed = (
                (self.decisions - self._clock[1]) * self.env.step_dt / (now - self._clock[0])
            )
            self._clock = (now, self.decisions)
        if realtime:  # never faster than real time
            ahead = (self.decisions - self._paced[1]) * self.env.step_dt - (now - self._paced[0])
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.25:  # fell behind (e.g. paused): don't catch up
                self._paced = (now, self.decisions)

    # -- drawing

    def _draw(self, viewer):
        self.frames += 1
        viewer.log_gizmo("palm_goal", self.goal[self.active])
        for (
            side
        ) in workstation.SIDES:  # TCP axes (x red, y green, z blue), a line to the palm point
            body = (
                self.env.scene[side].data.body_link_pose_w.torch[0, self._tcp[side]].cpu().numpy()
            )
            R = Rotation.from_quat(body[3:]).as_matrix()
            starts = np.repeat(body[None, :3], 4, axis=0)
            ends = np.vstack((starts[:3] + 0.1 * R.T, self.palm(side)[0]))
            colors = np.vstack((np.eye(3), (1.0, 0.85, 0.2)))
            viewer.log_lines(
                f"tcp_{side}",
                wp.array(starts, dtype=wp.vec3),
                wp.array(ends, dtype=wp.vec3),
                wp.array(colors, dtype=wp.vec3),
            )

    def _panel(self, imgui):
        if not imgui.collapsing_header("LW-v0 playground", imgui.TreeNodeFlags_.default_open.value):
            return
        side = self.active
        imgui.text(f"Arm: {side}   (Tab switches)")
        pos, euler = self.tcp_pose(side)
        imgui.text("Arm TCP, base frame (as the SDK reports it)")
        imgui.text(f"  x y z  {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f} m")
        imgui.text(f"  rx ry rz  {euler[0]:+.3f} {euler[1]:+.3f} {euler[2]:+.3f} rad")
        distance, angle = self.reach_error(side)
        imgui.text(f"Palm point to goal: {distance * 1000:.0f} mm, {np.degrees(angle):.0f} deg")
        hits = self.env.extras.get("arm_limit_hits", {}).get(side)
        if hits is not None:
            imgui.text(f"Limit interventions: {int(hits[0])} (since reset)")
        contacts = self.contact_count()
        if contacts is not None:
            self.peak_contacts = max(self.peak_contacts, contacts)
            imgui.text(f"Contacts: {contacts} (peak {self.peak_contacts})")
        imgui.text(f"Speed: {self.speed:.2f}x real time")
        imgui.separator()
        imgui.text("I/K J/L U/O  move the goal (camera-relative)")
        imgui.text("Shift + keys  turn it about the palm point")
        imgui.text("1 open  2 close  3 pinch  4 thumb oppose")
        imgui.text("Tab switch arm   R reset   drag the gizmo")

    def contact_count(self) -> int | None:
        from isaaclab_newton.physics import NewtonManager

        contacts = NewtonManager.get_contacts()
        if contacts is None:
            return None
        return int(contacts.rigid_contact_count.numpy()[0])


# ---------------------------------------------------------------------------- entry points


def run() -> int:
    """Open the playground window; returns when it is closed."""
    import gymnasium as gym
    from isaaclab.app import launch_simulation

    cfg = env_cfg()
    with launch_simulation(cfg, {"headless": False}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        playground = Playground(env)
        while playground.viz.is_running():
            playground.step(playground.keys_down())
        env.close()
    return 0


SELFTEST = (
    # (start [s], duration [s], keys held): every move starts from rest
    (0.5, 1.0, {"l"}),  # 10 cm to the camera's right
    (3.0, 0.6, {"shift", "j"}),  # 17 deg yaw
    (5.5, 0.1, {"3"}),  # pinch
    (7.5, 0.1, {"tab"}),  # switch to the left arm
    (8.0, 0.5, {"u"}),  # 5 cm up
    (10.5, 0.1, {"r"}),  # reset
)
SELFTEST_SECONDS = 11.5


def _send_keys(window, keys: set[str], down: bool) -> None:
    """Key press or release events for ``keys``, as the keyboard would send them."""
    from pyglet.window import key

    special = {"tab": key.TAB, "shift": key.LSHIFT}
    for name in keys:
        symbol = special.get(name) or getattr(key, f"_{name}" if name.isdigit() else name.upper())
        window.dispatch_event("on_key_press" if down else "on_key_release", symbol, 0)


def selftest(out: Path | None = None) -> dict:
    """Run the playground with the window open and scripted keys (:data:`SELFTEST`, sent to
    the window as key events); report how the arm and hand followed, whether every frame
    was drawn, and the speed.

    ``out``: also write the report and a few screenshots there.
    """
    import gymnasium as gym
    from isaaclab.app import launch_simulation

    cfg = env_cfg()
    report: dict = {"settle_s": {}}
    with launch_simulation(cfg, {"headless": False}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        pg = Playground(env)
        hz = round(1 / env.step_dt)
        home = {s: env.scene[s].data.joint_pos.torch[0].clone() for s in workstation.SIDES}
        start_palm = {s: pg.palm(s)[0] for s in workstation.SIDES}
        moves = {round(t0 * hz): (round((t0 + d) * hz), keys) for t0, d, keys in SELFTEST}
        window = pg.viz._viewer.renderer.window
        held, release, settling = set(), None, {}
        speeds = []
        for k in range(round(SELFTEST_SECONDS * hz)):
            if k in moves:
                release, held = moves[k]
                _send_keys(window, held, down=True)
                if held & set(KEYS):
                    settling[k] = (pg.active, release)
            if k == release:
                _send_keys(window, held, down=False)
                held, release = set(), None
            pg.step(pg.keys_down())
            for k0, (side, end) in list(settling.items()):
                distance, angle = pg.reach_error(side)
                if k > end and distance < 0.002 and angle < np.radians(1):
                    report["settle_s"][f"{k0 / hz:.1f}"] = round((k - end) / hz, 3)
                    del settling[k0]
            if k >= hz:
                speeds.append(pg.speed)
            if out is not None and k in (round(1.6 * hz), round(7.4 * hz), round(9 * hz)):
                out.mkdir(parents=True, exist_ok=True)
                frame = pg.viz._viewer.get_frame(render_ui=True).numpy()
                from PIL import Image

                Image.fromarray(frame).save(out / f"frame_{k / hz:04.1f}s.png")
            if k == round(2.9 * hz):  # after the first move
                report["moved_right_palm_m"] = float(
                    np.linalg.norm(pg.palm("right")[0] - start_palm["right"])
                )
            if k == round(10.4 * hz):  # before the reset
                report["moved_left_palm_m"] = float(
                    np.linalg.norm(pg.palm("left")[0] - start_palm["left"])
                )
                ids = pg._joints["right"]
                shafts = env.scene["right"].data.joint_pos.torch[0, ids].cpu().numpy()
                report["right_hand_pinch_error_rad"] = float(
                    np.abs(shafts - hand.PRESETS["pinch"]).max()
                )
        report["unsettled"] = [f"{k0 / hz:.1f}" for k0 in settling]
        report["home_error_after_reset_rad"] = max(
            float((env.scene[s].data.joint_pos.torch[0] - home[s]).abs().max())
            for s in workstation.SIDES
        )
        report["decisions"] = pg.decisions
        report["frames_drawn"] = pg.frames
        report["speed_median"] = float(np.median(speeds)) if speeds else 0.0
        report["capacity"] = env.step_dt / pg.busy  # how much faster than real time it could run
        report["window_open"] = bool(pg.viz.is_running())
        report["physics_hz"] = physics.PHYSICS_HZ
        env.close()

    report["passed"] = bool(
        not report["unsettled"]
        and all(t < 2.0 for t in report["settle_s"].values())
        and report["moved_right_palm_m"] > 0.09
        and report["moved_left_palm_m"] > 0.04
        and report["right_hand_pinch_error_rad"] < 0.05
        and report["home_error_after_reset_rad"] < 1e-3
        and report["frames_drawn"] >= report["decisions"] - 1
        and report["window_open"]
    )
    if out is not None:
        (out / "report.json").write_text(json.dumps(report, indent=1))
    return report
