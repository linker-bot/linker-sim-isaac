# Known Risks And Design Constraints

Language: [English](constraints.md) | [中文](../../zh-CN/operations/constraints.md)

This document defines the current runtime, resource, configuration, and safety
boundaries.

## Runtime And Resource Boundaries

- Execution code plays already-generated targets or trajectories. It does not perform IK or planning during physics-step playback.
- cuRobo backend code operates in cuRobo C-space. Full articulation command-space mapping belongs to runtime/controller layers.
- Mimic follower expansion belongs to controller/runtime logic, not the cuRobo backend.
- In Tiled Scene, `TiledCommandAdapter` is a synchronous command-step adapter. Graph search and trajectory optimization belong to async planner workers or backend planning layers.
- Reset, `set_state`, and snapshot restore capture rollback state before the first write. An incomplete rollback or a failure after an irreversible cache/queue reset makes the runtime fail-stop; rebuild it instead of issuing more mutations.
- Shutdown closes transport and publisher threads before planner/camera/IK resources and closes the SimulationApp last. A timeout is an incomplete shutdown, and a live child resource retains ownership of its sink or runtime dependency so shutdown can be retried.

## Threading Boundaries

- Isaac stage, articulation, PhysX views, and camera wrappers are accessed only on the main simulation thread.
- Background threads may publish serialized snapshots, write files, or serve transport responses.
- Foxglove and camera output consume data captured on the main thread; their publisher threads do not access Isaac objects directly.

## Config Boundaries

- Robot placement belongs in env profiles, not robot profiles.
- Robot model resources for cuRobo belong in robot profiles.
- Planner algorithm defaults belong in `configs/curobo/`.
- Object asset identity, import options, physics, and planning collision belong in object profiles; per-scene placement belongs in env profiles.
- Runtime process, resource, transport, telemetry, output, and shutdown policies belong in runtime profiles; env profiles do not accept those fields.
- This checkout is a workspace application. Distribution and editable builds are intentionally rejected because configs, scripts, assets, and vendored task resources are required at runtime.

## Fixed-Base Boundaries

- For a URDF rigid object, an omitted `import.fix_base` follows
  `physics.static`: static objects import fixed and dynamic objects import
  floating.
- A fixed URDF object is not also frozen through kinematic rigid-body
  overrides. A static object with explicit `import.fix_base: false` imports
  floating and is then frozen through kinematic bodies with gravity disabled.
- `physics.static: false` with `import.fix_base: true` is rejected because the
  dynamic and fixed-base declarations conflict.
- A static USD object is frozen through kinematic bodies with gravity disabled;
  USD object references do not accept an `import` section.

## Tiled Scene Constraints

- All envs advance physics synchronously.
- Env-specific commands update selected env target rows only; they do not pause other envs.
- All tiled envs in one profile share the same robot/object set.
- Per-env object differences are pose overrides for same-name objects.

## Network And Telemetry Safety

- Foxglove state streams are observation-only.
- Built-in control, state, and camera listeners accept only `localhost` or a
  numeric loopback address. They provide neither authentication nor TLS; remote
  access requires an authenticated TLS proxy or SSH tunnel terminating on the
  loopback endpoint.
- Command ports, Foxglove state live ports, and camera live ports are distinct
  services and must use separate ports.
- Camera output ports are configured in env profiles, not in interactive state-stream CLI flags.

## Verification

Useful checks:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_tiled_*.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_env_profile_directory.py tests/test_controller_configs.py tests/test_robot_loader_import_config.py -q
git diff --check
```
