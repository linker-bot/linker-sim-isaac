"""Evaluate the selected ycb-grasp objects without a viewer and retain every trial.

Run from a source checkout with its installed environment. This is an inference
evaluation, not a reproduction of training or a camera-based grasp benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def advance_until(scene, states, limit):
    """Advance at least once, so a pending button is processed before checking state."""
    for _ in range(limit):
        scene.step()
        if scene.state in states:
            return
    raise TimeoutError(f"state {scene.state!r} after {limit} decisions")


def run_trial(scene, item, seed, *, settle_steps=400, grasp_steps=800):
    import numpy as np

    scene.rng = np.random.default_rng(seed)
    row = {"object": scene.items[item]["name"], "seed": seed, "success": False}
    scene.press("drop", item)
    try:
        advance_until(scene, {"ready", "empty"}, settle_steps)
    except TimeoutError as exc:
        return {**row, "failure": "settle_timeout", "reason": str(exc)}
    row["placement_retries"] = scene.attempt
    if scene.state != "ready":
        return {**row, "failure": "placement", "reason": scene.message}
    pos, quat = scene._object_pose()
    row["settled_position"] = pos[0].tolist()
    row["settled_quaternion_xyzw"] = quat[0].tolist()
    scene.press("grasp")
    try:
        advance_until(scene, {"done", "empty"}, grasp_steps)
    except TimeoutError as exc:
        return {**row, "failure": "grasp_timeout", "reason": str(exc)}
    if scene.state != "done" or scene.result is None:
        return {**row, "failure": "interrupted", "reason": scene.message}
    return {
        **row,
        **scene.result,
        "failure": None if scene.result["success"] else "grasp",
        "message": scene.message,
    }


def summarize(trials):
    if not trials:
        raise ValueError("an evaluation requires at least one trial")
    successes = sum(row["success"] is True for row in trials)
    return {
        "trials": len(trials),
        "successes": successes,
        "success_rate": successes / len(trials),
        "failures": dict(Counter(row["failure"] for row in trials if row["success"] is not True)),
    }


def write_report(path, report):
    """Replace a checkpoint only after a complete, valid JSON file has been written."""
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_hashes(play):
    import linker_lab

    package = Path(linker_lab.__file__).resolve()
    entry = Path(play.__file__).resolve()
    if package != ROOT / "src/linker_lab/__init__.py" or entry != ROOT / "policy-gallery/ycb-grasp/play.py":
        raise RuntimeError(
            "the loaded linker-lab and gallery entry must belong to this checkout; "
            "run uv sync here before evaluating"
        )
    files = [*package.parent.glob("*.py"), entry, Path(__file__).resolve()]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def metadata(artifacts, play):
    sources = source_hashes(play)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu = []
    packages = ("linker-lab", "linker-lw-v0", "isaaclab", "newton", "mujoco",
                "mujoco-warp", "warp-lang", "torch", "numpy")
    return {
        "commit": revision,
        "tracked_changes": dirty,
        "source_sha256": sources,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu_and_driver": gpu,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "artifacts": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in artifacts.items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", help="object names; defaults to every gallery object")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-success-rate", type=float,
                        help="optional acceptance threshold, chosen before running")
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.min_success_rate is not None and not 0 <= args.min_success_rate <= 1:
        parser.error("min-success-rate must be between 0 and 1")

    from linker_lab import gallery

    play = gallery.load("ycb-grasp")  # bootstrap before Isaac Lab imports
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation

    artifacts = gallery.fetch("ycb-grasp")
    items = play.load_objects(artifacts["objects.npz"])
    names = [item["name"] for item in items]
    selected = names if args.objects is None else args.objects
    if len(set(selected)) != len(selected) or set(selected) - set(names):
        parser.error("objects must be distinct gallery object names")
    actor, demo = play.load_policy(artifacts["policy.npz"])
    report = {
        "schema": 1, "entry": "ycb-grasp", "metadata": metadata(artifacts, play),
        "objects": selected, "seeds": args.seeds,
        "placement_region": [float(x) for x in demo["region"]],
        "criteria": {
            "minimum_rise_m": play.RISE,
            "hold_seconds": (demo["end"] - demo["hold_start"]) / 60,
            "maximum_other_contact_force_n": play.CONTACT_TOLERANCE,
            "joint_limit_tolerance_rad": play.JOINT_LIMIT_TOLERANCE,
            "placement_retries": play.RETRIES,
            "settle_limit_decisions": 400, "grasp_limit_decisions": 800,
        },
        "min_success_rate": args.min_success_rate, "complete": False, "trials": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        write_report(args.output, report)

    save()
    cfg = play.env_cfg(items, viewer=False)
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            scene = play.GraspScene(env, items, actor.to(env.device), demo)
            with torch.inference_mode():
                for name in selected:
                    for seed in args.seeds:
                        report["trials"].append(run_trial(scene, names.index(name), seed))
                        save()
        finally:
            env.close()
    report["summary"] = summarize(report["trials"])
    report["per_object"] = {
        name: summarize([row for row in report["trials"] if row["object"] == name])
        for name in selected
    }
    report["complete"] = True
    save()
    print(json.dumps(report["summary"]))
    return int(args.min_success_rate is not None
               and report["summary"]["success_rate"] < args.min_success_rate)


if __name__ == "__main__":
    raise SystemExit(main())
