"""My task on LW-v0: registers the gym environment ``My-Task-LW-v0``.

Isaac Lab's command line imports this package (the ``isaaclab.tasks`` entry point in
``pyproject.toml``) before it starts the simulation, so it prepares linker-lab's engine first.
"""

import linker_lab

linker_lab.bootstrap()

import gymnasium as gym  # noqa: E402

gym.register(
    id="My-Task-LW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "my_task.env_cfg:MyTaskEnvCfg"},
)
