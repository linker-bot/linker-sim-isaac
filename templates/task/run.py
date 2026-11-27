"""Run My-Task-LW-v0 with the robot holding home, and print what the task computes.

    uv run python run.py                  # 16 environments, headless, 5 s
    uv run python run.py --viewer         # 4 environments in the viewer, until closed

Replace the action (``workstation.home_action``) with your own controller or policy.
"""

import linker_lab

linker_lab.bootstrap()  # first: prepares the physics engine for Isaac Lab

import argparse  # noqa: E402

import gymnasium as gym  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402
from my_task.env_cfg import MyTaskEnvCfg  # noqa: E402  (importing my_task registers the task)

from linker_lab import workstation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--viewer", action="store_true", help="open the viewer")
parser.add_argument("--seconds", type=float, default=5.0, help="headless: how long to run")
args = parser.parse_args()

cfg = MyTaskEnvCfg()
cfg.scene.num_envs = args.num_envs or (4 if args.viewer else 16)
if args.viewer:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    cfg.sim.visualizer_cfgs = NewtonGLVisualizerCfg(eye=(1.6, -1.2, 0.4), lookat=(0.3, 0.0, -0.4))

with launch_simulation(cfg, {"headless": not args.viewer}):
    env = gym.make("My-Task-LW-v0", cfg=cfg).unwrapped
    obs, _ = env.reset()
    print(f"observation: {tuple(obs['policy'].shape)}, action: {env.action_manager.total_action_dim}")
    decisions = 0
    while (env.sim.visualizers[0].is_running() if args.viewer else decisions * env.step_dt < args.seconds):
        obs, reward, terminated, truncated, _ = env.step(workstation.home_action(env))
        decisions += 1
        if decisions % 60 == 0:
            print(
                f"{decisions * env.step_dt:5.1f} s  reward {reward.mean():.4f}"
                f"  ended {int((terminated | truncated).sum())}/{env.num_envs}"
            )
    env.close()
