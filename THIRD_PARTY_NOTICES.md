# Third-Party Notices

This project depends on third-party software and assets with separate
licenses. End users and redistributors are responsible for ensuring full
compliance with all applicable third-party licenses when shipping source,
binaries, containers, or integrated products.

## NVIDIA Isaac Sim

- Project: NVIDIA Isaac Sim
- Upstream: [https://developer.nvidia.com/isaac/sim](https://developer.nvidia.com/isaac/sim)
- Version: 6.0.1 (pinned via the `simulation` extra).
- License: Dual.
  - The Isaac Sim source-code wrapper is released under the Apache
    License 2.0.
  - The underlying NVIDIA Omniverse Kit SDK and proprietary engines
    (PhysX, renderers) are governed by the *NVIDIA Isaac Sim Additional
    Software and Materials License*, which is restrictive and not
    redistributable.
- Notes: Isaac Sim is an **optional** runtime dependency, installed by end
  users via the `simulation` extra with `OMNI_KIT_ACCEPT_EULA=Y`. This
  project does not redistribute any NVIDIA binaries, USDs, or proprietary
  materials. Users installing the `simulation` extra agree to NVIDIA's terms.

## NVIDIA Isaac Lab

- Project: NVIDIA Isaac Lab (formerly Orbit)
- Upstream: [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
- License: BSD 3-Clause License (core).
- Notes: `src/linkerbot_sim/isaac/physics/newton/replication.py` is derived
  from Isaac Lab (`isaaclab_newton.cloner.newton_replicate`,
  `release/3.0.0-beta2.patch1`) and retains its BSD-3-Clause SPDX header.
  Isaac Lab otherwise runs inside Isaac Sim and is installed separately by
  end users; it is not bundled here.

## NVIDIA cuRobo

- Project: NVIDIA cuRobo
- Upstream: [https://github.com/NVlabs/curobo](https://github.com/NVlabs/curobo)
- Version: 0.8.0 (pinned).
- License: Apache License 2.0.
- Notes: The task configuration bundle under
  `src/linkerbot_sim/backends/curobo/resources/task/` is copied verbatim from
  cuRobo v0.8.0. Each file retains its upstream
  `SPDX-FileCopyrightText: NVIDIA` and `SPDX-License-Identifier: Apache-2.0`
  headers; the accompanying `README.md` documents the source commit and a
  SHA-256 integrity gate. Only the Apache-2.0 task configs are vendored, not
  cuRobo's separately-licensed robot assets.

## NVIDIA Warp

- Project: NVIDIA Warp (`warp-lang`)
- Upstream: [https://github.com/NVIDIA/warp](https://github.com/NVIDIA/warp)
- License: Apache License 2.0.
- Notes: Used by the multi-world Newton physics backend. Installed as a
  dependency; not vendored.

## PyTorch

- Project: PyTorch
- Upstream: [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- License: BSD-3-Clause-style with additional terms.

## skrl

- Project: skrl
- Upstream: [https://github.com/Toni-SM/skrl](https://github.com/Toni-SM/skrl)
- License: MIT.
- Notes: CUDA-native reinforcement-learning training path for Kaleidoscope;
  installed via the `training` extra.

## Gymnasium

- Project: Gymnasium
- Upstream: [https://github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- License: MIT.

## trimesh

- Project: trimesh
- Upstream: [https://github.com/mikedh/trimesh](https://github.com/mikedh/trimesh)
- License: MIT.

## NumPy / SciPy / PyYAML / websockets

- NumPy — [https://github.com/numpy/numpy](https://github.com/numpy/numpy) — BSD 3-Clause.
- SciPy — [https://github.com/scipy/scipy](https://github.com/scipy/scipy) — BSD 3-Clause.
- PyYAML — [https://github.com/yaml/pyyaml](https://github.com/yaml/pyyaml) — MIT.
- websockets — [https://github.com/python-websockets/websockets](https://github.com/python-websockets/websockets) — BSD 3-Clause.

## NVIDIA Industrial warehouse asset

- Source: NVIDIA Omniverse / Isaac "ArchVis Industrial" sample content.
- License: Governed by NVIDIA's Omniverse / Isaac asset license terms.
- Notes: The Mirror `scene3` / `industrial_warehouse` scene *references* an
  NVIDIA Industrial warehouse USD by path. **The NVIDIA asset itself is not
  redistributed in this repository** — the referenced payload lives under a
  gitignored `usd-material/` directory that end users must obtain separately
  under NVIDIA's terms. Only lightweight scene wiring is shipped here.

## Robot meshes

The 3D meshes shipped under `assets/` are Linkerbot hardware CAD released as
open-source by the manufacturer:

- AR5V2 arm meshes — Linkerbot.
- L6 (L6V1) hand meshes — Linkerbot.
- workstationV1 bench meshes — Linkerbot.

Each mesh is included in good faith based on the upstream open-source release.
Redistributors should retain the upstream attribution.

## Responsibility

End users and redistributors are responsible for ensuring full compliance
with all applicable third-party licenses when shipping source, binaries,
containers, or integrated products.
