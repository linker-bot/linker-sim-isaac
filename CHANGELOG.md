# Changelog

linker-lab follows [semantic versions](https://semver.org/) from 1.0. The public API is
what [docs/api.md](docs/api.md) lists. A change that breaks it is a new major version, a
removal first warns for one minor release, and a new version of LW-v0's files is a new minor
version.

## 1.0.0 (2026-11-27)

The first release of linker-lab: one Linker workstation, LW-v0, in Isaac Lab 3.0, behaving
like the real robot. It replaces this repository's earlier package, `linkerbot_sim` (last
release `v0.3.0`).

- **LW-v0** (two AR5 arms with O6 hands, the table and three cameras) from the files of
  linker-sim-assets `v1.0.0`, converted to USD on first use; the gym environment
  `Linker-LW-v0` (`workstation.LWv0EnvCfg`) for tasks to subclass.
- **Physics** from LW-v0's files: 480 Hz with collision detection every step, actions at
  60 Hz, on Newton 1.8 with MuJoCo-Warp 3.17 (pinned; `linker_lab.bootstrap()` prepares
  Isaac Lab 3.0 for it).
- **Arm actions** (`arm.ArmAction`) as the real arm is commanded through the xCore SDK: joint
  goals through a jerk-limited trajectory generator (tracking mode), or your own path
  (servo mode); joint or pose commands; the real controller's velocity, acceleration and
  jerk limits, joint ranges and wrist region; the servo model with the real torque caps.
- **Hand actions** (`hand.HandAction`): the O6's self-locking motors (motor model
  `geared-shaft-v1`).
- **Objects** from meshes (`objects`), cameras with the real resolution, intrinsics and rates,
  or a smaller image of the same aspect ratio (`workstation.camera_cfg(name, width, height)`).
- **Playground** (`linker-lab play`): drive either arm and hand from the keyboard and mouse.
- **Policy gallery** (`linker-lab gallery`): `ycb-grasp`, the left hand grasping 31 YCB
  objects it has not seen in training, dropped anywhere on the left half of the table.
- **Installation** (`install.sh`, one `uv.lock`) and `linker-lab doctor`, on Linux x86_64 and
  aarch64 (Jetson Thor).
- **No time limit by default:** `LWv0EnvCfg` has no termination terms, so an LW-v0
  environment runs until you reset it; tasks add their own (the task template adds a 5 s
  time limit).
- **Snapshots** (`linker_lab.snapshot`): save an environment between two steps and restore it,
  in the same or a new process, without a reset; `get_state()` / `set_state()` on `ArmAction` and
  `HandAction`.
- **Documentation** (`docs/`): getting started and troubleshooting, the LW-v0 environment, the
  gallery, requirements, the API reference, development and validation, and a Chinese quick
  start. Every code block runs in the tests.
- **Task template** (`templates/task/`): a task as its own project, with linker-lab as a
  dependency, an `LWv0EnvCfg` subclass and Isaac Lab's `isaaclab.tasks` entry point, so
  `isaaclab train` and `isaaclab play` can run it once a learner configuration is added.
  `linker_lab.bootstrap()` works after Isaac Lab's command line is imported.
