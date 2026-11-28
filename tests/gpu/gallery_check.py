"""Drive the ycb-grasp scene through its buttons, without a window: every object comes to
rest when dropped, and a full grasp runs to a result. Prints one JSON line."""

import json
import sys
import time

from linker_lab import gallery

play = gallery.load("ycb-grasp")  # prepares the engine before Isaac Lab is imported

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402

artifacts = gallery.fetch("ycb-grasp")
items = play.load_objects(artifacts["objects.npz"])
actor, demo = play.load_policy(artifacts["policy.npz"])
grasp_items = [i for i, o in enumerate(items) if o["name"] in sys.argv[1:]]
cfg = play.env_cfg(items, viewer=False)
report = {"objects": len(items), "not_at_rest": [], "grasps": {}}
with launch_simulation(cfg, {"headless": True}):
    env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
    env.reset()
    scene = play.GraspScene(env, items, actor.to(env.device), demo)
    scene.rng = np.random.default_rng(0)

    def run_until(states, limit):
        for _ in range(limit):
            scene.step()
            if scene.state in states:
                return

    start, decisions = time.perf_counter(), 0
    for i, o in enumerate(items):
        scene.press("drop", i)
        run_until({"ready", "empty"}, 400)
        if scene.state != "ready":
            report["not_at_rest"].append(o["name"])
        decisions += scene.tick
    report["seconds_per_decision"] = round((time.perf_counter() - start) / decisions, 4)
    for i in grasp_items:
        scene.press("drop", i)
        run_until({"ready", "empty"}, 400)
        scene.press("grasp")
        run_until({"done"}, 800)
        report["grasps"][items[i]["name"]] = scene.message
    scene.press("reset")
    scene.step()
    report["reset_state"] = scene.state
    env.close()
print("REPORT " + json.dumps(report), flush=True)
