# Getting started

From a fresh clone to driving the robot and running a pretrained policy: install (7 minutes
in a test from an empty home directory on a fast connection; most of it is downloading), the
playground (5 minutes), the gallery (10 minutes).

## 1. Install

You need a Linux computer (x86_64, or aarch64 such as a Jetson Thor) with an NVIDIA GPU (driver 580 or newer), glibc 2.34 or
newer (Ubuntu 22.04 or newer), `git`, internet access and 10 GB of free disk; see
[requirements](requirements.md). Everything else is installed by the script.

```bash
git clone https://github.com/linker-bot/linker-lab
cd linker-lab
./install.sh
```

`install.sh` checks the requirements, installs [uv](https://docs.astral.sh/uv/) if it is
missing, creates `.venv` from `uv.lock` (Python 3.12, Isaac Lab 3.0, the physics engine, about
4 GB to download), then runs `linker-lab doctor`, which ends with `All checks passed.` Run
the doctor again at any time:

```bash
uv run linker-lab doctor
```

If `install.sh` installed uv, this terminal does not find it yet (`uv: command not found`):
open a new terminal, or run `source ~/.local/bin/env`.

The doctor checks the installed versions, the GPU, the engine patches, and simulates one second of
LW-v0 (the first run also compiles GPU kernels, about a minute).

Run every command below with `uv run` from the `linker-lab` folder, or activate the
environment first (`source .venv/bin/activate`).

## 2. Drive the robot: the playground

```bash
uv run linker-lab play
```

A window opens with LW-v0 and a box on the table. You move the **palm point** (3.5 cm in front
of the middle of the palm's surface) of one arm; the arm follows within the real controller's
limits, as the real arm does in the xCore SDK's tracking mode.

| Input | Action |
| --- | --- |
| I/K, J/L, U/O | Move the goal forward/back, left/right, up/down (relative to the camera) |
| Shift + the same keys | Turn the goal about the palm point |
| Mouse | Fly the camera; right-drag pulls objects; drag the gizmo to move or turn the goal |
| Tab | Switch arm |
| 1 / 2 / 3 / 4 | Hand open / close / pinch / thumb oppose |
| R | Reset |

W/A/S/D, Q/E and the arrows move the camera, Space pauses, H hides the panel, Esc quits.
The panel shows the arm's TCP pose as the controller reports it, how far the palm point is
from its goal, contacts and the simulation speed. Try it: move the open hand over the box
(J/L, I/K, then U/O down), close it (2), lift (U).

The arm is slow on purpose: the real controller's speed limits (0.32-0.49 rad/s per joint)
move the palm point at most about 0.15 m/s. Goals at the edge of the arm's reach are approached
slowly.

`uv run linker-lab play --selftest` runs a scripted session instead (a move, a turn, a pinch,
the other arm, a reset) and reports whether the arm and hand followed.

## 3. Run a pretrained policy: the gallery

```bash
uv run linker-lab gallery list               # the entries
uv run linker-lab gallery info ycb-grasp     # what it does, its licences, its files
uv run linker-lab gallery run ycb-grasp      # open it
```

`run` downloads the entry's files (about 12 MB) into `~/.cache/linker-lab/gallery/`,
checks them, and opens the scene: one workstation and a panel. Pick an object (31 YCB
objects, none of them seen in training), press **Drop**: it lands at a random spot on the
left half of the table, in a random resting pose and turn. When it has come to rest press **Grasp**: the
policy looks at the object once and the left hand picks it up and holds it 15 cm up. The
status line says whether it held. **Reset** clears the table. More in [the gallery](gallery.md).

## 4. Next

- [The LW-v0 environment](environment.md): the scene, its frames, how the arm and hand are
  commanded, objects and cameras, in your own scripts.
- [The task template](../templates/task/README.md): your own task on LW-v0, as your own
  project, trained with Isaac Lab's trainers.
- [API reference](api.md): everything linker-lab provides, and what stays stable.
