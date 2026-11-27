# Getting started

From a fresh clone to driving the robot and running a pretrained policy: install (7 minutes
in a test from an empty home directory on a fast connection; most of it is downloading), the
playground (5 minutes), the gallery (10 minutes).

## 1. Install

You need a Linux computer (x86_64, or aarch64 such as a Jetson Thor) with an NVIDIA GPU (driver 580 or newer), glibc 2.34 or
newer (Ubuntu 22.04 or newer), `git`, internet access and 10 GB of free disk; see
[requirements](requirements.md). Everything else is installed by the script.

```bash
git clone https://github.com/linker-bot/linker-sim-isaac.git
cd linker-sim-isaac
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

The doctor reports installed versions, GPU availability and engine patches, then runs a
one-second LW-v0 simulation if the checks pass. A failed check returns exit status 1. The
first simulation also compiles GPU kernels (about a minute).
[PR #85](https://github.com/linker-bot/linker-sim-isaac/pull/85) moves dependency checks before
bootstrap and turns initialization exceptions into readable failures. On older builds, an
import failure can still stop the doctor with a traceback.

The package is named `linker-lab`, although the repository folder is `linker-sim-isaac`.
The script installs the `linker-lw-v0` robot-data dependency too; no `git submodule update`
step is needed. macOS, Windows and CPU-only simulation are not supported here.

Run every command below with `uv run` from the `linker-sim-isaac` folder, or activate the
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
the other arm, a reset) and reports whether the arm and hand followed. It still needs a
display. To save the JSON report and screenshots:

```bash
uv run linker-lab play --selftest --out tests/outputs/playground
```

## 3. Run a pretrained policy: the gallery

The policy files are separate from the source and robot-data package. Check
[artifact availability](gallery.md#artifact-availability) first: the manifest currently
points to an unavailable release location. Listing and inspecting entries do not download
files; running one requires the exact files and checksums in its manifest.

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

## Troubleshooting

- **Unsupported platform or no CUDA device:** use the Linux/NVIDIA requirements in
  [requirements.md](requirements.md). The CPU test subset is not a CPU simulation backend.
- **A missing package or wrong version:** from the repository root run `uv sync --locked`,
  then `uv run --no-sync linker-lab doctor`. Keep the pinned Newton/MuJoCo-Warp stack together;
  the compatibility patches target those versions. A failed bootstrap requires a fresh
  Python process after fixing the environment.
- **Bootstrap/import-order error in a script:** call `linker_lab.bootstrap()` before
  importing Isaac Lab simulation modules, or import `linker_lab.workstation` first.
  Importing `linker_lab` alone does not apply the patches.
- **Viewer cannot open:** check that a display is available. `play`, `play --selftest` and
  `gallery run` open windows and have no headless flag. Use a headless environment script
  or the task template for unattended simulation.
- **Gallery download or checksum failure:** use `gallery info ENTRY` to inspect the URLs
  and follow the [manual cache instructions](gallery.md#artifact-availability). Do not
  substitute other weights or edit the expected hash to bypass the mismatch.
- **USD conversion fails:** the error includes the affected part and cache location.
  Check the converter installation and glibc requirement. If using a compatible machine
  to prepare USDs, copy the entire indicated directory, including `Textures/`.
- **Snapshot is refused:** check `num_envs == 1`, package/configuration compatibility and
  the [supported state](api.md#linker_labsnapshot). When upgrading to schema 2
  ([PR #82](https://github.com/linker-bot/linker-sim-isaac/pull/82)), capture schema 1 snapshots
  again; changing a header does not convert their missing observation data.

Robot USDs, generated object USDs and gallery downloads live in
`~/.cache/linker-lab/{usd,objects,gallery}/`. If `XDG_CACHE_HOME` is set, its value replaces
`~/.cache`. These directories can be copied for an offline run together with the installed
environment; kernel compilation has its own Warp cache and may still run on first use.

## 4. Next

- [The LW-v0 environment](environment.md): the scene, its frames, how the arm and hand are
  commanded, objects and cameras, in your own scripts.
- [The task template](../templates/task/README.md): your own task on LW-v0, as your own
  project, trained with Isaac Lab's trainers.
- [API reference](api.md): everything linker-lab provides, and what stays stable.
- [Development and validation](development.md): CPU, wheel, GPU and release checks.
