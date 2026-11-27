# Requirements

| | Needed |
| --- | --- |
| Operating system | Linux on x86_64 or aarch64 (tested on a Jetson Thor with JetPack 7), with glibc 2.34 or newer (Ubuntu 22.04 or newer) |
| GPU | an NVIDIA GPU with driver 580 or newer (CUDA 13); the simulation runs on it |
| Tools | `git` (uv fetches LW-v0's files with it); `curl` or `wget` if uv is not installed |
| Disk | 10 GB free: the environment is about 8 GB, most of it Isaac Lab and PyTorch |
| Network | for the install: PyPI, the PyTorch package index, GitHub, and astral.sh if uv is missing; for the gallery: its files' URLs |
| Display | for the playground and the gallery viewer; headless scripts and training need none |

`install.sh` checks the system, glibc, the free disk, `git` and the GPU driver (not the network
or the display), and installs uv (into `~/.local/bin`) and Python 3.12 itself. macOS, Windows and
computers without an NVIDIA GPU are not supported: for those there is
[linker-sim](https://github.com/linker-bot/linker-sim), the same LW-v0 on the CPU with MuJoCo.

**Speed.** Measured on the development workstation (a current high-end desktop GPU):

| | Speed |
| --- | --- |
| Playground (one environment, viewer) | real time, with about 20 % to spare |
| Gallery `ycb-grasp` (viewer) | about 0.5-0.8× real time (slower while grasping) |
| 400 environments, arms moving, headless | about 28,000 environment steps per second |

On a slower GPU the viewers run below real time; the panel shows the simulation speed (on a
Jetson Thor: the playground about 0.5×, the gallery about 0.3× real time). The
first run of each compiles GPU kernels (about a minute; about five minutes on a Jetson Thor;
cached in the home directory).
