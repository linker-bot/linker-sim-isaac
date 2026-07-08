# LinkerHand Simulation

Language: [English](README.en.md) | [中文](README.zh-CN.md)

LinkerHand Simulation is an Isaac Sim / Isaac Lab project for validating robot-arm, dexterous-hand, and object-manipulation workflows. The current project covers AR5 arms, LinkerHand L6 hands, capsule/cuboid rope approximations, T-shaped rigid objects, and a cuMotion-backed motion stack.

The project is organized around these boundaries:

- `configs/` is versioned with the code and stores robot, env, object, controller, logging, and cuMotion profiles.
- `scripts/` contains runnable simulation and experiment entrypoints.
- `tools/object_assets/` contains offline object-asset builders. Intrinsic geometry and physical properties live there; runtime placement and overrides live in `configs/objects/`.
- `src/linkerbot_sim/app/runtime/` owns Isaac app/session, World, stage, robot import, object import, and controller wiring.
- `src/linkerbot_sim/app/motion/` converts customer-facing motion specs into cuMotion requests, trajectories, and command-space execution data.
- `src/linkerbot_sim/app/interactive/` owns JSON protocols, command queues, and stdin/TCP/WebSocket transports.
- `src/linkerbot_sim/backends/cumotion/` is the only layer that adapts directly to cuMotion Python APIs.
- `src/linkerbot_sim/execution/` only plays already-generated targets or trajectories on physics steps.
- `src/linkerbot_sim/controllers/` converts project targets into Isaac articulation actions and refreshes mimic followers.

## Current Capabilities

- Asset import: robot execution assets use MJCF/URDF; cuMotion planning assets use URDF/XRDF; environment objects support USD/URDF reference or import.
- Single-arm runtime: interactive hand/arm motion, YAML-defined TCP frames, IK, planning, trajectory sampling, and CSV logging.
- Dual-arm runtime: left and right AR5+L6 articulations are imported separately in Isaac and fused into a 14-DOF arm C-space for cuMotion.
- Interactive motion: single-arm and dual-arm runtimes support stdin JSONL, TCP JSONL, and WebSocket JSON commands.
- Tiled environments: a single Isaac scene can host multiple homogeneous env instances with synchronized command stepping, trajectory buffers, and async planner integration.
- Objects: capsule/cuboid rope-like dynamic chains and T-shaped compound rigid objects can be generated from tool configs and referenced by env profiles.
- cuMotion backend: FK, IK, collision-free IK, trajectory optimization, graph search, specified paths, task-space path conversion, and C-space trajectory generation.
- Telemetry: CSV joint tracking, Foxglove live servers, MCAP output, state snapshots, object markers, and sensor camera RGB/depth output.

## Repository Layout

```text
.
├── assets/                   # arm/hand/object meshes and robot/object assets
├── configs/                  # env, robot, object, controller, logging, cuMotion profiles
├── docs/                     # documentation language entry plus zh-CN/en trees
│   ├── README.md
│   ├── zh-CN/
│   └── en/
├── scripts/                  # runnable Isaac Sim / Isaac Lab entrypoints
├── src/linkerbot_sim/        # runtime, motion, controllers, cuMotion backend, telemetry
├── tests/                    # lightweight tests that mostly avoid launching Isaac Sim
├── tools/object_assets/      # offline USD/USDA object asset builders
├── README.md                 # README language entry
├── README.en.md              # English README
├── README.zh-CN.md           # Chinese README
└── pyproject.toml
```

## Environment

Examples assume commands are run from the repository root and that the Python environment already contains Isaac Sim, Isaac Lab, cuMotion, torch, and the regular project dependencies:

```bash
PYTHONPATH=src python <command>
```

If the environment is not activated, replace `python` with the actual interpreter path, for example:

```bash
PYTHONPATH=src env_isaaclab/bin/python <command>
```

The project uses a src-layout. The runnable scripts add `src/` to `sys.path` themselves, but tests and ad-hoc snippets should still set `PYTHONPATH=src`.

The `simulation` extra in `pyproject.toml` records the expected major simulation dependencies:

- `isaacsim[all]==5.1.0.0`
- `cumotion==1.1.0`
- `torch==2.7.0`

Install example:

```bash
python -m pip install -e ".[dev,visualization,simulation]"
```

## Quick Start

Generate or verify object USD assets. `scene1` and `scene2` use the capsule rope; the default dual-arm `scene3` uses the T block:

```bash
PYTHONPATH=src python tools/object_assets/flexible/rope/build_asset.py
PYTHONPATH=src python tools/object_assets/rigid/tblock/build_asset.py
```

Start the single-arm interactive GUI runtime:

```bash
PYTHONPATH=src python scripts/single_arm_interactive.py \
  --env scene1 \
  --gui \
  --foxglove-live-port 8765
```

Start the dual-arm interactive GUI runtime:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold
```

Run a lightweight dual-arm motion semantics test without launching Isaac:

```bash
PYTHONPATH=src python -m pytest tests/test_dual_arm_motion_test.py -q
```

Start the tiled interactive runtime:

```bash
PYTHONPATH=src python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --default-decimation 1
```

Start dual-arm Foxglove telemetry:

```bash
PYTHONPATH=src python scripts/dual_arm_interactive.py \
  --gui --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8766 \
  --state-rate-hz 60 \
  --state-include-objects
```

Connect Foxglove Desktop to `ws://127.0.0.1:8766` using the `Foxglove WebSocket` data source.

## Main Entrypoints

| Mode | Entrypoint | Launches Isaac | Needs cuMotion | Purpose |
| --- | --- | --- | --- | --- |
| Generate rope USD | `tools/object_assets/flexible/rope/build_asset.py` | Yes, headless | No | Writes USD/PhysX schema from the rope config. |
| Generate T block USD | `tools/object_assets/rigid/tblock/build_asset.py` | Yes, headless | No | Writes USD/PhysX schema from the T block config. |
| Single-arm interactive motion | `scripts/single_arm_interactive.py` | Yes | Yes | Long-lived single AR5+L6 runtime controlled by JSON commands. |
| Dual-arm interactive motion | `scripts/dual_arm_interactive.py` | Yes | Yes | Long-lived dual-arm runtime controlled by JSON commands. |
| Tiled interactive runtime | `scripts/tiled_env_interactive.py` | Yes | Optional | Synchronized multi-env command stepping, trajectory buffers, async planning. |
| Dual-arm motion semantics test | `python -m pytest tests/test_dual_arm_motion_test.py -q` | No | No | Checks MoveSpec, TCP, specified-path, and C-space planner data semantics. |

## Documentation

Full English documentation starts at [docs/en/index.md](docs/en/index.md).

Chinese documentation starts at [docs/zh-CN/文档索引.md](docs/zh-CN/文档索引.md).

Topic groups:

- Configuration and naming
- Interaction and runtime
- Tiled environments
- Motion planning
- Telemetry and sensors
- Assets and scenes
- Risks and constraints

## Verification

Syntax check:

```bash
PYTHONPATH=src python -m compileall -q src scripts tests
```

Common lightweight tests:

```bash
PYTHONPATH=src python -m pytest -q tests
```

Diff hygiene before committing:

```bash
git diff --check
```
