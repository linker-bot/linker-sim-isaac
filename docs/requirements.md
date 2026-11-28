# Requirements

| | Needed |
| --- | --- |
| Operating system | Linux on x86_64 or aarch64 (tested on a Jetson Thor with JetPack 7), with glibc 2.35 or newer (Ubuntu 22.04 or newer) |
| Python | Python 3.12 only (`>=3.12,<3.13`); installed by uv |
| GPU | an NVIDIA GPU with driver 580 or newer (CUDA 13); the simulation runs on it |
| Tools | `uv >=0.11.0` (installed if missing); `git` (uv fetches LW-v0's files with it); `curl` or `wget` if uv is not installed |
| Disk | 10 GB free: the environment is about 8 GB, most of it Isaac Lab and PyTorch |
| Network | for the install: PyPI, the PyTorch package index, GitHub, and astral.sh if uv is missing; for the gallery: its files' URLs |
| Display | for the playground, its self-test and the gallery viewer; headless scripts and training need none |

`install.sh` checks the system, glibc, the free disk, `git` and the GPU driver (not the network
or the display), and installs uv (into `~/.local/bin`) and Python 3.12 itself. macOS, Windows and
computers without an NVIDIA GPU are not supported: for those there is
[linker-sim-mujoco](https://github.com/linker-bot/linker-sim-mujoco), the same LW-v0 on the CPU with MuJoCo.

Isaac Lab 3.0 is paired with the project's exact Newton 1.8.0, MuJoCo/MuJoCo-Warp 3.17.0,
Warp 1.17.0 and Torch 2.11.0 pins. `pyproject.toml` declares the requirements and `uv.lock`
records the resolved graph. Isaac Lab's Newton-native actuators and deformable-body path
are not supported by the compatibility patches; LW-v0 uses rigid bodies and implicit
actuators. Installing the package through a different project must preserve the Newton
override; the task template provides it.

LW-v0's files (the `linker-lw-v0` package) are installed with the project; the gallery
downloads its policies' files when an entry first runs ([the files](gallery.md#the-files)).

**Speed.** Reported measurements from the development workstation (a high-end desktop GPU); these
are examples, not performance guarantees:

| | Speed |
| --- | --- |
| Playground (one environment, viewer) | real time, with about 20 % to spare |
| Gallery `ycb-grasp` (viewer) | about 0.5-0.8× real time (slower while grasping) |
| 400 environments, arms moving, headless | about 28,000 environment steps per second |

On a slower GPU the viewers run below real time; the panel shows the simulation speed (on a
Jetson Thor: the playground about 0.5×, the gallery about 0.3× real time). The
first run of each compiles GPU kernels (about a minute; about five minutes on a Jetson Thor;
cached in the home directory).
