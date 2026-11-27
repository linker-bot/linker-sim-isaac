# Task template

A task on the LW-v0 workstation, as its own project: copy this folder, rename it, and change
the task. The example task is picking up a box with the left hand: its scene, actions,
resets, observations, rewards and terminations are there; the controller or policy that
picks the box up is yours (`run.py` only holds the robot at home).

```
pyproject.toml        your project: depends on linker-lab; registers the package with Isaac Lab
src/my_task/
  __init__.py         prepares linker-lab's engine, registers the gym id My-Task-LW-v0
  env_cfg.py          the task: MyTaskEnvCfg, a subclass of linker-lab's LWv0EnvCfg
run.py                runs the task with the robot holding home
```

## Start

Use the same [Linux/NVIDIA requirements](https://github.com/linker-bot/linker-sim-isaac/blob/main/docs/requirements.md) as the main project.
Run the copy command from the `linker-sim-isaac` checkout; the `my-task` folder is a
separate project with its own virtual environment and dependency lock.

```bash
cp -r templates/task ~/my-task && cd ~/my-task
uv sync                         # creates .venv with linker-lab, Isaac Lab and the engine (~8 GB)
uv run python run.py            # 16 environments, headless: prints the reward
uv run python run.py --viewer   # 4 environments in the viewer
```

To rename it, change the project `name` and the entry point in `pyproject.toml`, the folder
`src/my_task`, the gym id in `__init__.py`, and the imports in `run.py`.

`uv sync` installs `linker-lab` from its release tag `v2.0.0` in
[linker-bot/linker-sim-isaac](https://github.com/linker-bot/linker-sim-isaac)
(`[tool.uv.sources]` in `pyproject.toml`). To move to a later release, change the tag and run
`uv sync` again.

The task project repeats the Newton override needed by Isaac Lab 3.0 and a
`tinyobjloader` prerelease constraint. Keep those when renaming the template. It resolves
its own versions of unpinned dependencies; commit its generated `uv.lock`, then use
`uv sync --locked` on other machines. The main project's lock and dev dependencies are
not inherited by a project depending on it.

### Use a local checkout

For development against your current source, replace the template's `linker-lab` source
entry with an absolute path (keep the other sections):

```toml
[tool.uv.sources]
linker-lab = { path = "/absolute/path/to/linker-sim-isaac", editable = true }
```

Run `uv sync` after changing it. Python and gallery source changes are then visible
without rebuilding the package. Use the release tag again when sharing the task;
an absolute path is specific to your machine.

## The task

`MyTaskEnvCfg` (`src/my_task/env_cfg.py`) starts from `LWv0EnvCfg`, LW-v0's default
environment: the workstation, its physics (480 Hz, actions at 60 Hz), an action per arm and
hand, joint-state observations and a reset to home. The task changes it in Isaac Lab's usual
way, one configuration class per part:

| Part | Here | To change it |
| --- | --- | --- |
| Scene | a 5 cm box on the table (`linker_lab.objects`) | add objects, fixed objects, cameras (`workstation.camera_cfg`) |
| Actions | left arm: 7 joint goals [rad] within the real controller's limits; left hand: 6 shaft targets [rad]; the right arm and hand hold home | `workstation.arm_action_cfg(side, command="pose", frame="palm")` for palm poses, `interpolation=False` to stream your own path, `hand_action_cfg(side, normalized=True)` for [-1, 1] |
| Events | reset to home, then the box at a random place and yaw | Isaac Lab's `mdp` event terms, or your own functions |
| Observations | joint states, box position, palm position | add terms to `PolicyCfg` |
| Rewards | two examples: the palm approaching the box, the box rising | your task's reward |
| Terminations | 5 s time limit, the box falling off the table | your task's success and failure conditions |

Positions are in the environment frame: x forward over the table, z up, the table top
0.5 m below the origin. [docs/api.md](https://github.com/linker-bot/linker-sim-isaac/blob/main/docs/api.md) lists everything linker-lab provides.

An arm action takes absolute joint goals (or poses), like the real controller. A learned
policy usually outputs small numbers around zero, so map its output to goals around the home
pose, e.g. `home + 0.5 * action`, in your own action term: a subclass of
`linker_lab.arm.ArmAction` whose `process_actions` does that before calling the parent's,
used with `workstation.arm_action_cfg("left", class_type=MyArmAction)`.

## Train it

The template has no learner. Isaac Lab's own trainers work with it: write a learner
configuration in a new module, e.g. `src/my_task/agents.py` with a `MyPPORunnerCfg` subclassing
`RslRlOnPolicyRunnerCfg` from `isaaclab_rl.rsl_rl`, and name it in the registration in
`src/my_task/__init__.py`:

```python
kwargs={
    "env_cfg_entry_point": "my_task.env_cfg:MyTaskEnvCfg",
    "rsl_rl_cfg_entry_point": "my_task.agents:MyPPORunnerCfg",
}
```

then

```bash
uv run isaaclab train --rl_library rsl_rl --task My-Task-LW-v0 --num_envs 512
uv run isaaclab play --rl_library rsl_rl --task My-Task-LW-v0 --num_envs 16
```

Isaac Lab finds the task through the `isaaclab.tasks` entry point in `pyproject.toml`.
Runs are written to `logs/rsl_rl/<experiment>/<time>/` in the current directory. See
Isaac Lab's documentation for the learners and their settings.
