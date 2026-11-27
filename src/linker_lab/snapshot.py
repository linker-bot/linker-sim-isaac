"""Save and restore an LW-v0 environment between two ``env.step()`` calls.

    from linker_lab import snapshot, workstation   # before Isaac Lab, as every module

    state = snapshot.capture(env)            # between env.step() calls
    ...                                      # step on, try something
    obs = snapshot.restore(env, state)       # back to the captured moment, no reset
    snapshot.save(state, "grasp.pt")         # restore later, also in a new process
    state = snapshot.load("grasp.pt")

After :func:`restore`, every input of the next ``env.step()`` is what it was at :func:`capture`.
Isaac Lab's ``env.reset_to()`` cannot do this: it resets first, which restarts the action
terms' controllers from rest. The state is a written list:

* **Physics**: every articulation's and rigid object's root pose and velocity, every
  articulation's joint positions, velocities and drive targets, written through Isaac Lab's
  writers, then the kinematics recomputed (``env.sim.forward()``), as ``reset_to`` does.
* **Control**: each action term's ``get_state()`` / ``set_state()`` (``ArmAction``,
  ``HandAction``; a term of your own adds the two methods) and the action manager's last two
  actions.
* **Environment**: the step counters, the episode lengths, the simulated time, the reward
  sums and the random generators (torch, NumPy, Python).

Not saved, because nothing reads them before rewriting them: contacts (found from scratch
every physics step) and the physics solver's own buffers (MuJoCo-Warp's controls and applied
forces are rewritten every physics step; its warm start only seeds the solver's iterations, and
a restore starts it cold). Measured, a restored environment follows the uninterrupted one within
float32 rounding (about 1e-7 rad and 1e-7 m over 8 decisions), the same as two restores of one
snapshot; further on, any two runs drift apart as GPU contact sums are not bit-reproducible.

Not saved either: model parameters changed while running (masses, friction or gains set by a
randomization event; restoring where they differ gives a different simulation), camera images,
the renderer and the viewer.

One environment (``num_envs == 1``). :func:`capture` refuses (:class:`SnapshotError`) what it
cannot account for rather than skipping it. Calling these functions inside ``env.step()`` is not
supported.

Importing this module applies ``linker_lab.bootstrap()``, so import it before Isaac Lab.
"""

from __future__ import annotations

import linker_lab

linker_lab.bootstrap()

import datetime
import hashlib
import importlib.metadata as metadata
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["capture", "restore", "save", "load", "Snapshot", "SnapshotError"]

SCHEMA = 1
"""Version of the snapshot's layout; a snapshot of another schema is refused."""

PACKAGES = ("linker-lab", "linker-lw-v0", "isaaclab", "newton", "mujoco-warp", "warp-lang", "torch")
"""Whose versions a snapshot records; restoring needs the same ones."""

UNCOMPARED_CONFIG = ("viewer", "sim.visualizer_cfgs")
"""Configuration fields left out of the fingerprint: they only change what is drawn, never what
``env.step()`` computes."""


class SnapshotError(RuntimeError):
    """A snapshot could not be taken or restored.

    ``physics_modified`` is False if the environment was not changed (every refusal and every
    mismatch is found before the first write), True if writing had begun: then the environment
    is in an undefined state and should be discarded.
    """

    def __init__(self, message: str, physics_modified: bool = False):
        super().__init__(message)
        self.physics_modified = physics_modified


@dataclass(frozen=True)
class Snapshot:
    """A captured environment: ``header`` (versions, configuration fingerprint, scene and
    action layout, creation time) and ``state`` (the layers, CPU tensors and numbers; its
    layout is internal)."""

    header: dict
    state: dict


# ---------------------------------------------------------------------------- public


def capture(env) -> Snapshot:
    """The state of ``env`` (the unwrapped ``ManagerBasedRLEnv``), copied to the CPU."""
    _refusals(env)
    return Snapshot(header=_header(env), state=_cpu(_read(env)))


def restore(env, snapshot: Snapshot) -> dict:
    """Write ``snapshot`` into ``env`` without a reset; return the observations at the snapshot
    (also stored in ``env.obs_buf``). Everything is checked before the first write."""
    _refusals(env)
    _check_header(_header(env), snapshot.header)
    current = _cpu(_read(env))
    try:
        _check_tree(current, snapshot.state, "state")
    except ValueError as error:
        raise SnapshotError(str(error)) from error
    try:
        _write(env, snapshot.state)
    except Exception as error:
        raise SnapshotError(f"restore failed while writing: {error}", physics_modified=True) from error
    env.obs_buf = env.observation_manager.compute(update_history=False)
    return env.obs_buf


def save(snapshot: Snapshot, path: str | Path) -> None:
    """Write the snapshot to one file (plain dicts, lists, numbers, strings and CPU tensors)."""
    torch.save({"header": snapshot.header, "state": snapshot.state}, Path(path))


def load(path: str | Path) -> Snapshot:
    """Read a snapshot written by :func:`save` (no code is executed); :func:`restore` checks it
    against the environment."""
    data = torch.load(Path(path), weights_only=True)
    if not isinstance(data, dict) or set(data) != {"header", "state"}:
        raise SnapshotError(f"{path} is not a linker-lab snapshot")
    if data["header"].get("schema") != SCHEMA:
        raise SnapshotError(f"{path}: snapshot schema {data['header'].get('schema')}, this linker-lab reads {SCHEMA}")
    return Snapshot(header=data["header"], state=data["state"])


# ---------------------------------------------------------------------------- what is refused


def _refusals(env) -> None:
    def refuse(what: str):
        raise SnapshotError(f"cannot snapshot this environment: {what}")

    if env.num_envs != 1:
        refuse(f"num_envs is {env.num_envs}; a snapshot holds one environment")
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if not (callable(getattr(term, "get_state", None)) and callable(getattr(term, "set_state", None))):
            refuse(f"action term {name!r} ({type(term).__name__}) has no get_state() / set_state()")
    for group, cfgs in env.observation_manager._group_obs_term_cfgs.items():
        for cfg in cfgs:
            if (cfg.history_length or 0) > 0:
                refuse(f"observation group {group!r} has a term with a history")
    if "interval" in env.event_manager.available_modes:
        refuse("an event term runs at intervals (its timers are not saved)")
    for manager in ("command_manager", "curriculum_manager", "recorder_manager"):
        terms = getattr(getattr(env, manager, None), "active_terms", None)
        if terms:
            refuse(f"the {manager.split('_')[0]} terms {terms} are not saved")
    scene = env.scene
    for kind in ("rigid_object_collections", "deformable_objects", "surface_grippers", "cable_objects"):
        if getattr(scene, kind, None):
            refuse(f"the scene has {kind.replace('_', ' ')} ({', '.join(getattr(scene, kind))})")
    for name, sensor in scene.sensors.items():
        if "Camera" not in type(sensor).__name__:
            refuse(f"sensor {name!r} ({type(sensor).__name__}) keeps state that is not saved")
    for name, asset in {**scene.articulations, **scene.rigid_objects}.items():
        if asset.permanent_wrench_composer.active:
            refuse(f"{name!r} has a permanent external wrench")


# ---------------------------------------------------------------------------- the header


def _header(env) -> dict:
    packages = {}
    for name in PACKAGES:
        dist = metadata.distribution(name)
        version = dist.version
        direct = dist.read_text("direct_url.json")
        if direct and (commit := json.loads(direct).get("vcs_info", {}).get("commit_id")):
            version += f"+{commit}"
        packages[name] = version
    return {
        "schema": SCHEMA,
        "linker_lab_commit": _commit(),
        "packages": packages,
        "config": _fingerprint(env.cfg),
        "scene": {"articulations": sorted(env.scene.articulations), "rigid_objects": sorted(env.scene.rigid_objects)},
        "actions": dict(zip(env.action_manager.active_terms, env.action_manager.action_term_dim)),
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


def _commit() -> str | None:
    """linker-lab's git commit when it runs from a checkout, else None."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or None


def _fingerprint(cfg) -> str:
    data = cfg.to_dict()
    for dotted in UNCOMPARED_CONFIG:
        *parents, leaf = dotted.split(".")
        node = data
        for key in parents:
            node = node.get(key, {})
        node.pop(leaf, None)
    text = json.dumps(data, sort_keys=True, default=str)
    text = re.sub(r" at 0x[0-9a-fA-F]+", "", text)  # object addresses differ per process
    return hashlib.sha256(text.encode()).hexdigest()


def _check_header(current: dict, saved: dict) -> None:
    for key in ("schema", "packages", "config", "scene", "actions"):
        if current.get(key) != saved.get(key):
            if key == "packages":
                diff = {k: (saved[key].get(k), v) for k, v in current[key].items() if saved[key].get(k) != v}
                raise SnapshotError(f"snapshot taken with other package versions (snapshot, now): {diff}")
            what = "configuration" if key == "config" else key
            raise SnapshotError(f"snapshot does not match this environment: {what} differs")
    if current["linker_lab_commit"] and saved.get("linker_lab_commit") and current["linker_lab_commit"] != saved["linker_lab_commit"]:
        raise SnapshotError(
            f"snapshot taken with linker-lab {saved['linker_lab_commit'][:9]}, this is {current['linker_lab_commit'][:9]}"
        )


# ---------------------------------------------------------------------------- reading


def _read(env) -> dict:
    from isaaclab.physics import PhysicsManager

    physics = {}
    for name, robot in env.scene.articulations.items():
        d = robot.data
        physics[name] = {
            "root_pose": d.root_pose_w.torch, "root_velocity": d.root_vel_w.torch,
            "joint_position": d.joint_pos.torch, "joint_velocity": d.joint_vel.torch,
            "position_target": d.joint_pos_target.torch, "velocity_target": d.joint_vel_target.torch,
            "effort_target": d.joint_effort_target.torch, "joint_names": list(robot.joint_names),
        }
    for name, obj in env.scene.rigid_objects.items():
        physics[name] = {"root_pose": obj.data.root_pose_w.torch, "root_velocity": obj.data.root_vel_w.torch}
    control = {name: env.action_manager.get_term(name).get_state() for name in env.action_manager.active_terms}
    np_state = np.random.get_state()
    environment = {
        "sim_step_counter": int(env._sim_step_counter),
        "common_step_counter": int(env.common_step_counter),
        "episode_length": env.episode_length_buf,
        "sim_time": float(PhysicsManager._sim_time),
        "episode_sums": env.reward_manager._episode_sum_buf,
        "action": env.action_manager._action,
        "prev_action": env.action_manager._prev_action,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(env.device) if "cuda" in str(env.device) else torch.zeros(0, dtype=torch.uint8),
        "rng_numpy": {"keys": torch.from_numpy(np_state[1].astype(np.int64)), "pos": int(np_state[2]),
                      "has_gauss": int(np_state[3]), "gauss": float(np_state[4])},
        "rng_python": [list(random.getstate()[1]), random.getstate()[2]],
    }
    return {"physics": physics, "control": control, "environment": environment}


def _cpu(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().to("cpu", copy=True)
    if isinstance(tree, dict):
        return {k: _cpu(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [_cpu(v) for v in tree]
    return tree


def _check_tree(current, saved, where: str) -> None:
    if isinstance(current, torch.Tensor):
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape or saved.dtype != current.dtype:
            got = (tuple(saved.shape), saved.dtype) if isinstance(saved, torch.Tensor) else type(saved).__name__
            raise ValueError(f"{where}: expected {(tuple(current.shape), current.dtype)}, got {got}")
    elif isinstance(current, dict):
        if not isinstance(saved, dict) or set(saved) != set(current):
            raise ValueError(f"{where}: keys differ")
        for key in current:
            _check_tree(current[key], saved[key], f"{where}.{key}")
    elif isinstance(current, list) and where.endswith("joint_names"):
        if saved != current:
            raise ValueError(f"{where}: {saved} != {current}")
    elif isinstance(current, list):
        if not isinstance(saved, list) or len(saved) != len(current):
            raise ValueError(f"{where}: length differs")
    elif current is not None and type(saved) is not type(current):
        raise ValueError(f"{where}: expected {type(current).__name__}, got {type(saved).__name__}")


# ---------------------------------------------------------------------------- writing


def _write(env, state: dict) -> None:
    from isaaclab.physics import PhysicsManager

    device = env.device

    def dev(t: torch.Tensor) -> torch.Tensor:
        return t.to(device)

    # control
    for name, term_state in state["control"].items():
        term = env.action_manager.get_term(name)
        term.set_state({k: dev(v) if isinstance(v, torch.Tensor) else v for k, v in term_state.items()})
    e = state["environment"]
    env.action_manager._action.copy_(dev(e["action"]))
    env.action_manager._prev_action.copy_(dev(e["prev_action"]))
    # environment
    env._sim_step_counter = e["sim_step_counter"]
    env.common_step_counter = e["common_step_counter"]
    env.episode_length_buf.copy_(dev(e["episode_length"]))
    PhysicsManager._sim_time = e["sim_time"]
    env.reward_manager._episode_sum_buf.copy_(dev(e["episode_sums"]))
    torch.set_rng_state(e["rng_torch"])
    if e["rng_cuda"].numel():
        torch.cuda.set_rng_state(e["rng_cuda"], env.device)
    n = e["rng_numpy"]
    np.random.set_state(("MT19937", n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["gauss"]))
    keys, gauss = e["rng_python"]
    random.setstate((3, tuple(keys), gauss))
    # physics, through Isaac Lab's writers
    for name, s in state["physics"].items():
        if name in env.scene.articulations:
            robot = env.scene.articulations[name]
            robot.write_root_pose_to_sim_index(root_pose=dev(s["root_pose"]))
            robot.write_root_velocity_to_sim_index(root_velocity=dev(s["root_velocity"]))
            robot.write_joint_position_to_sim_index(position=dev(s["joint_position"]))
            robot.write_joint_velocity_to_sim_index(velocity=dev(s["joint_velocity"]))
            robot.set_joint_position_target_index(target=dev(s["position_target"]))
            robot.set_joint_velocity_target_index(target=dev(s["velocity_target"]))
            robot.set_joint_effort_target_index(target=dev(s["effort_target"]))
        else:
            obj = env.scene.rigid_objects[name]
            obj.write_root_pose_to_sim_index(root_pose=dev(s["root_pose"]))
            obj.write_root_velocity_to_sim_index(root_velocity=dev(s["root_velocity"]))
    env.sim.forward()  # kinematics from the written joints (also starts the solver cold)
