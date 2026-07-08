# Known Risks And Design Constraints

Language: [English](known-risks-and-design-constraints.md) | [中文](../../zh-CN/风险与约束/已知风险与设计约束.md)

This document records high-level risks and constraints that should be preserved during refactors.

## Runtime Boundaries

- Execution code should play already-generated targets or trajectories. It should not perform IK or planning during physics-step playback.
- cuMotion backend code should stay focused on cuMotion C-space. Full articulation command-space mapping belongs to runtime/controller layers.
- Mimic follower expansion belongs to controller/runtime logic, not the cuMotion backend.
- Tiled `TiledCommandAdapter` should remain a synchronous command-step adapter. Graph search and trajectory optimization belong to async planner workers or backend planning layers.

## Threading Boundaries

- Isaac stage, articulation, PhysX views, and camera wrappers must be read on the main simulation thread.
- Background threads may publish serialized snapshots, write files, or serve transport responses.
- Foxglove and camera output should consume data captured on the main thread, not access Isaac objects directly.

## Config Boundaries

- Robot placement belongs in env profiles, not robot profiles.
- Robot model resources for cuMotion belong in robot profiles.
- Planner algorithm defaults belong in `configs/cumotion/`.
- Object asset identity and import options belong in object profiles; per-scene placement belongs in env profiles.

## Tiled Constraints

- All envs advance physics synchronously.
- Env-specific commands update selected env target rows only; they do not pause other envs.
- All tiled envs in one profile share the same robot/object set.
- Per-env object differences are pose overrides for same-name objects.

## Telemetry Constraints

- Foxglove state streams are observation-only.
- Command ports and Foxglove live ports must stay separate.
- Camera output ports are configured in env profiles, not in interactive state-stream CLI flags.

## Verification

Useful lightweight checks:

```bash
PYTHONPATH=src python -m pytest tests/test_tiled_*.py -q
PYTHONPATH=src python -m pytest tests/test_env_profile_directory.py tests/test_controller_configs.py tests/test_robot_loader_import_config.py -q
git diff --check
```
