"""Snapshots on the GPU: capture while the arm moves and the hand holds a box, restore, compare
(one JSON line).

    python tests/gpu/snapshot_check.py record DIR          # a pose run's joint goals and samples
    python tests/gpu/snapshot_check.py same MODE DIR       # capture, 8 decisions, restore, 8 again
    python tests/gpu/snapshot_check.py new MODE DIR        # restore the snapshot `same` saved
    python tests/gpu/snapshot_check.py refuse CASE DIR     # two_envs | no_contract | config

The scene: LW-v0 and a 6 cm, 100 g box under the left hand. The left palm point descends to
2 cm above the box centre (the box 4.5 cm ahead of it, under the fingers), the hand closes, the
arm lifts the box 10 cm while moving 5 cm sideways; the snapshot is taken at decision 360,
mid-lift. Left-arm modes: ``pose`` (palm-point poses), ``goal`` (joint goals: the pose run's IK
outputs) and ``samples`` (interpolation off: the pose run's per-step references). Run as a
separate process: Isaac Lab allows one simulation per process.
"""

import json
import sys
from pathlib import Path

import linker_lab

linker_lab.bootstrap()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from linker_lab import hand, objects, snapshot, workstation  # noqa: E402

EDGE, MASS = 0.06, 0.1
TABLE = objects.table()["top"]
BOX_XY = (0.445, 0.319)
PALM_X = BOX_XY[0] - 0.045
GRASP_Z = TABLE + EDGE / 2 + 0.02
CAPTURE, AFTER = 360, 8
_mesh = trimesh.creation.box((EDGE, EDGE, EDGE))
BOX = [(_mesh.vertices, _mesh.faces)]


def env_cfg(mode: str, num_envs: int = 1):
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.box = objects.object_cfg(BOX, mass=MASS, pos=objects.on_table(BOX, BOX_XY), prim_path="{ENV_REGEX_NS}/box")
    if mode == "pose":
        cfg.actions.left_arm = workstation.arm_action_cfg("left", command="pose", frame="palm")
    elif mode == "samples":
        cfg.actions.left_arm = workstation.arm_action_cfg("left", interpolation=False, stream="samples")
    return cfg


def palm_goal(k: int):
    """The palm point's goal (environment frame) and the hand's target at decision k."""
    if k < 120:
        return (PALM_X, BOX_XY[1], GRASP_Z + 0.08), None
    if k < 210:
        return (PALM_X, BOX_XY[1], GRASP_Z), None
    if k < 300:
        return (PALM_X, BOX_XY[1], GRASP_Z), hand.PRESETS["close"]
    s = min((k - 300) / 120, 1.0)
    return (PALM_X, BOX_XY[1] + 0.05 * s, GRASP_Z + 0.10 * s), hand.PRESETS["close"]


class Script:
    def __init__(self, env, mode, recorded=None):
        self.env, self.mode, self.rec = env, mode, recorded
        left = env.scene["left"]
        self.quat_w = workstation.palm_pose_w(left)[1].clone()
        self.base = left.data.root_link_pose_w.torch.clone()
        dims = dict(zip(env.action_manager.active_terms, env.action_manager.action_term_dim))
        names = list(dims)
        start = {n: sum(dims[m] for m in names[: names.index(n)]) for n in names}
        self.arm = slice(start["left_arm"], start["left_arm"] + dims["left_arm"])
        self.hand = slice(start["left_hand"], start["left_hand"] + 6)
        self.dim = env.action_manager.total_action_dim

    def action(self, k: int) -> torch.Tensor:
        env = self.env
        a = torch.full((env.num_envs, self.dim), torch.nan, device=env.device)
        goal, grip = palm_goal(k)
        if self.mode == "pose":
            pos_w = torch.tensor(goal, device=env.device) + env.scene.env_origins
            pos, quat = subtract_frame_transforms(self.base[:, :3], self.base[:, 3:7], pos_w, self.quat_w)
            a[:, self.arm] = torch.cat((pos, quat), -1)
        else:
            a[:, self.arm] = torch.as_tensor(self.rec[self.mode][k], device=env.device)
        if grip is not None:
            a[:, self.hand] = torch.tensor(grip, device=env.device)
        return a


def readout(env) -> dict:
    out = {}
    for side in workstation.SIDES:
        robot = env.scene[side]
        ids, _ = robot.find_joints([*workstation.ARM_JOINTS[side], *workstation.HAND_JOINTS[side]], preserve_order=True)
        out[f"{side}_q"] = robot.data.joint_pos.torch[0, ids].double().cpu().numpy()
    out["box"] = (env.scene["box"].data.root_pos_w.torch[0] - env.scene.env_origins[0]).double().cpu().numpy()
    return out


def clock(env) -> list:
    from isaaclab.physics import PhysicsManager

    return [env._sim_step_counter, env.common_step_counter, int(env.episode_length_buf[0]), PhysicsManager._sim_time]


def run_on(env, script, start):
    rows, clocks = [], []
    for k in range(start, start + AFTER):
        env.step(script.action(k))
        rows.append(readout(env))
        clocks.append(clock(env))
    return {key: np.stack([r[key] for r in rows]) for key in rows[0]}, clocks


def compare(reference, replay) -> dict:
    joints = max(float(np.abs(reference[f"{s}_q"] - replay[f"{s}_q"]).max()) for s in workstation.SIDES)
    return {"joints_rad": joints, "box_m": float(np.abs(reference["box"] - replay["box"]).max())}


def equal(x, y) -> bool:
    """Bit for bit, NaN equal to NaN (NaN in a command means keep)."""
    if isinstance(x, dict):
        return set(x) == set(y) and all(equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(equal(a, b) for a, b in zip(x, y))
    if isinstance(x, torch.Tensor):
        x, y = x.cpu(), y.cpu()
        return x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.nan_to_num(7.0), y.nan_to_num(7.0)) and torch.equal(x.isnan(), y.isnan())
    return x == y


def control(env) -> dict:
    terms = {n: env.action_manager.get_term(n).get_state() for n in env.action_manager.active_terms}
    targets = {s: env.scene[s].data.joint_pos_target.torch.clone() for s in workstation.SIDES}
    return {"terms": terms, "targets": targets}


def scene_state(env) -> dict:
    return {
        name: (asset.data.root_pose_w.torch.clone(), getattr(asset.data, "joint_pos", asset.data.root_pose_w).torch.clone())
        for name, asset in {**env.scene.articulations, **env.scene.rigid_objects}.items()
    }


def start(mode, num_envs=1, directory=None, cfg=None):
    cfg = cfg or env_cfg(mode, num_envs)
    recorded = dict(np.load(Path(directory) / "script.npz")) if mode != "pose" else None
    return cfg, recorded


def record(directory) -> dict:
    cfg, _ = start("pose")
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        script = Script(env, "pose")
        term = env.action_manager.get_term("left_arm")
        goals, samples = [], []
        original = term._joint_command

        def spy(raw):
            q = original(raw)
            goals.append(q[0].float().cpu().numpy())
            return q

        term._joint_command = spy
        for k in range(CAPTURE + AFTER):
            env.step(script.action(k))
            samples.append(torch.cat((term._ref_q[0].flatten(), term._ref_v[0].flatten())).float().cpu().numpy())
        env.close()
    np.savez(Path(directory) / "script.npz", goal=np.stack(goals), samples=np.stack(samples))
    return {"recorded": len(goals)}


def same(mode, directory) -> dict:
    d = Path(directory)
    cfg, recorded = start(mode, directory=d)
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        script = Script(env, mode, recorded)
        for k in range(CAPTURE):
            env.step(script.action(k))
        lift = float(readout(env)["box"][2] - (TABLE + EDGE / 2))
        state = snapshot.capture(env)
        at_capture = control(env), clock(env)
        snapshot.save(state, d / f"{mode}.pt")
        reference, clocks = run_on(env, script, CAPTURE)
        np.savez(d / f"{mode}_reference.npz", **reference)
        json.dump(clocks, open(d / f"{mode}_clocks.json", "w"))
        hits = env.extras["arm_limit_hits"]["left"]
        snapshot.restore(env, snapshot.load(d / f"{mode}.pt"))
        restored = control(env), clock(env)
        same_hits_tensor = env.extras["arm_limit_hits"]["left"] is hits
        replay, clocks2 = run_on(env, script, CAPTURE)
        env.close()
    return {
        "lift_at_capture_m": lift,
        "error": compare(reference, replay),
        "control_exact": equal(restored[0], at_capture[0]),
        "clock_exact": restored[1] == at_capture[1] and clocks2 == clocks,
        "limit_hits_same_tensor": same_hits_tensor,
        "box_lift_end_m": float(replay["box"][-1, 2] - (TABLE + EDGE / 2)),
    }


def new(mode, directory) -> dict:
    d = Path(directory)
    cfg, recorded = start(mode, directory=d)
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()  # a fresh environment at home: everything not in the snapshot starts from scratch
        script = Script(env, mode, recorded)
        state = snapshot.load(d / f"{mode}.pt")
        snapshot.restore(env, state)
        control_exact = equal(
            {n: env.action_manager.get_term(n).get_state() for n in env.action_manager.active_terms},
            {n: {k: v.to(env.device) if isinstance(v, torch.Tensor) else v for k, v in s.items()} for n, s in state.state["control"].items()},
        )
        replay, clocks = run_on(env, script, CAPTURE)
        env.close()
    reference = dict(np.load(d / f"{mode}_reference.npz"))
    return {
        "error": compare(reference, replay),
        "control_exact": control_exact,
        "clock_exact": clocks == json.load(open(d / f"{mode}_clocks.json")),
        "box_lift_end_m": float(replay["box"][-1, 2] - (TABLE + EDGE / 2)),
    }


def refuse(case, directory) -> dict:
    """Each refusal raises SnapshotError(physics_modified=False) and changes nothing."""
    d = Path(directory)
    cfg = env_cfg("pose", 2 if case == "two_envs" else 1)
    if case == "no_contract":
        from isaaclab.envs import mdp

        cfg.actions.right_hand = mdp.JointPositionActionCfg(asset_name="right", joint_names=list(workstation.HAND_JOINTS["right"]))
    if case == "config":
        cfg.episode_length_s = 20.0
    raised = []
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        for _ in range(5):
            env.step(torch.full((env.num_envs, env.action_manager.total_action_dim), torch.nan, device=env.device))
        before = scene_state(env), control(env) if case != "no_contract" else None, clock(env)
        saved = snapshot.load(d / "pose.pt")
        attempts = {"restore": lambda: snapshot.restore(env, saved)}
        if case in ("two_envs", "no_contract"):
            attempts["capture"] = lambda: snapshot.capture(env)
        if case == "config":
            edited = snapshot.Snapshot({**saved.header, "packages": {**saved.header["packages"], "linker-lw-v0": "0.9.0"}}, saved.state)
            attempts["versions"] = lambda: snapshot.restore(env, edited)
            edited_config = snapshot.Snapshot(saved.header, saved.state)
            attempts["restore"] = lambda: snapshot.restore(env, edited_config)
            # a snapshot of this very environment, one tensor's shape changed
            own = snapshot.capture(env)
            bad = {**own.state, "environment": {**own.state["environment"], "episode_length": torch.zeros(3, dtype=torch.long)}}
            attempts["shape"] = lambda: snapshot.restore(env, snapshot.Snapshot(own.header, bad))
        for name, attempt in attempts.items():
            try:
                attempt()
                raised.append([name, None, None])
            except snapshot.SnapshotError as error:
                raised.append([name, str(error), error.physics_modified])
        after = scene_state(env), control(env) if case != "no_contract" else None, clock(env)
        env.close()
    return {"raised": raised, "unchanged": equal(before[0], after[0]) and equal(before[1], after[1]) and before[2] == after[2]}


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    result = {"record": record, "same": same, "new": new, "refuse": refuse}[command](*args)
    print("RESULT " + json.dumps(result), flush=True)
