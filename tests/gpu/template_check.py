"""Run the task template's environment and report what its terms computed (one JSON line).

    python tests/gpu/template_check.py

The robot holds home for 1 s: the boxes come to rest where the reset placed them. Then one
box is moved below the table: its environment, and only that one, terminates. Run as a
separate process: Isaac Lab allows one simulation per process.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "templates" / "task" / "src"))

import linker_lab  # noqa: E402

linker_lab.bootstrap()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402
from my_task import env_cfg  # noqa: E402  (importing my_task registers My-Task-LW-v0)

from linker_lab import workstation  # noqa: E402


def main() -> dict:
    cfg = env_cfg.MyTaskEnvCfg()
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("My-Task-LW-v0", cfg=cfg).unwrapped
        obs, _ = env.reset()
        box = env.scene["box"]
        start = (box.data.root_pos_w.torch - env.scene.env_origins).clone()
        rewards = []
        for _ in range(60):
            obs, reward, terminated, truncated, _ = env.step(workstation.home_action(env))
            rewards.append(reward)
        rest = box.data.root_pos_w.torch - env.scene.env_origins
        speed = torch.linalg.norm(box.data.root_lin_vel_w.torch, dim=-1)
        ended_while_holding = bool((terminated | truncated).any())

        pose = box.data.root_link_pose_w.torch.clone()
        pose[0, 2] = env_cfg.TABLE_TOP - 0.5
        box.write_root_pose_to_sim_index(root_pose=pose[:1], env_ids=torch.tensor([0], device=env.device))
        _, _, terminated, _, _ = env.step(workstation.home_action(env))
        result = {
            "num_envs": env.num_envs,
            "observation": list(obs["policy"].shape),
            "action": env.action_manager.total_action_dim,
            "action_terms": env.action_manager.active_terms,
            "start_spread_m": [float(start[:, i].max() - start[:, i].min()) for i in (0, 1)],
            "start_offset_max_m": float((start[:, :2] - torch.tensor(env_cfg.BOX_XY, device=env.device)).abs().max()),
            "rest_height_error_m": float((rest[:, 2] - (env_cfg.TABLE_TOP + env_cfg.BOX_EDGE / 2)).abs().max()),
            "rest_xy_drift_m": float(torch.linalg.norm(rest[:, :2] - start[:, :2], dim=-1).max()),
            "rest_max_speed_m_s": float(speed.max()),
            "rewards_finite": bool(torch.isfinite(torch.stack(rewards)).all()),
            "reward_mean": float(torch.stack(rewards).mean()),
            "ended_while_holding": ended_while_holding,
            "dropped_terminated": [bool(t) for t in terminated.tolist()],
        }
        env.close()
    return result


if __name__ == "__main__":
    print("RESULT " + json.dumps(main()), flush=True)
