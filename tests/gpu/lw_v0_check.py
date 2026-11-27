"""Run LW-v0 with a box dropped onto the table and report what happened (one JSON line).

    python tests/gpu/lw_v0_check.py [num_envs] [seconds]

Every arm and hand is commanded to stay at the home pose. Reports the largest joint
deviation from home, where the box came to rest relative to the table top, and the speed.
Run as a separate process: Isaac Lab allows one simulation per process.
"""

import json
import sys
import time

import gymnasium as gym
import torch
import trimesh

from linker_lab import objects, workstation  # workstation bootstraps the engine

BOX = 0.06  # edge [m]
MASS = 0.1  # [kg]
XY = (0.35, 0.0)


def main(num_envs: int, seconds: float) -> dict:
    from isaaclab.app import launch_simulation  # after linker_lab.workstation

    box = trimesh.creation.box((BOX, BOX, BOX))
    parts = [(box.vertices, box.faces)]
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.object = objects.object_cfg(parts, MASS, pos=objects.on_table(parts, XY, clearance=0.02))
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        action = workstation.home_action(env)
        robots = [env.scene[s] for s in workstation.SIDES]
        home = [r.data.default_joint_pos.torch.clone() for r in robots]
        steps = round(seconds / env.step_dt)
        env.step(action)  # first step compiles and captures the CUDA graph
        torch.cuda.synchronize()
        drift, start = 0.0, time.perf_counter()
        for _ in range(steps - 1):
            env.step(action)
            drift = max(drift, *((r.data.joint_pos.torch - h).abs().max().item() for r, h in zip(robots, home)))
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        obj = env.scene["object"]
        pos = obj.data.root_pos_w.torch - env.scene.env_origins
        speed = obj.data.root_lin_vel_w.torch.norm(dim=-1)
        result = {
            "num_envs": num_envs,
            "simulated_s": steps * env.step_dt,
            "wall_s": wall,
            "env_steps_per_s": num_envs * (steps - 1) / wall,
            "realtime_factor": (steps - 1) * env.step_dt * num_envs / wall,
            "home_max_drift_rad": drift,
            "box_rest_height_error_m": (pos[:, 2] - BOX / 2 - objects.table()["top"]).abs().max().item(),
            "box_xy_drift_m": (pos[:, :2] - torch.tensor(XY, device=pos.device)).norm(dim=-1).max().item(),
            "box_max_speed_m_s": speed.max().item(),
            "finite": bool(torch.isfinite(pos).all() and all(torch.isfinite(r.data.joint_pos.torch).all() for r in robots)),
        }
        env.close()
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    print("RESULT " + json.dumps(main(int(args[0]) if args else 4, float(args[1]) if len(args) > 1 else 2.0)))
