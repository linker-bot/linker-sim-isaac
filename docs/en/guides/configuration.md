# Configuration Guide

Language: [English](configuration.md) | [中文](../../zh-CN/guides/configuration.md)

## Layers

- `configs/envs/`: scenes, objects, sensors, and robot instances.
- `configs/robots/`: one articulation's Isaac model, command groups, and cuRobo binding.
- `configs/curobo/`: algorithm defaults such as device, seeds, tolerances, and caches.
- `configs/controllers/<bundle>/`: one arm/hand/default controller bundle.
- `configs/objects/`: object assets and physics settings.
- `configs/logging/`: joint CSV sampling, flushing, output paths, and column switches.
- `configs/runtime/`: entry mode, process resources, telemetry/camera output,
  and file lifecycle policy.

This repository is a workspace application rather than an installable Python
library. Runtime profiles, scripts, assets, and vendored task resources are part
of the application, so `tool.uv.package` is false and the PEP 517 backend
explicitly rejects wheel, source-distribution, and editable builds. Run commands
from the checkout root with `PYTHONPATH=src` after `uv sync --all-extras`; do not
install or copy only the `src/` tree.

Every project profile accepts only the current strict structure. Fixed mappings
reject unknown keys with complete field paths. Booleans do not accept strings or
`0/1`, and periods, sampling intervals, and gains receive strict type and range
checks. A YAML document must resolve to a top-level mapping; empty documents,
duplicate keys at any nesting depth, and non-mapping documents are rejected with
source locations.

Only World settings live under the top-level `env:` mapping. `robots`,
`objects`, `solver`, `visuals`, `sensors`, and optional `tiled` are sibling
top-level fields. A directory-form env profile with `tiled` topology merges `<name>/base.yaml` with
the overrides under `per_env_config_dir`. The effective count is stored in
`tiled.num_envs` and has no CLI override. If a directory base omits it while
fragments exist, the loader derives `max(env_id) + 1`; otherwise it defaults to 1.

The runtime profile owns process-wide Single Scene/Tiled Scene behavior. Its current
top-level fields are:

| Field | Owner |
|---|---|
| `mode` | Select the `single_scene` or `tiled_scene` entry contract |
| `profiles` | Select env, cuRobo, logging, and controller profiles |
| `simulation_app` | GUI, GPU, renderer, and headless launch settings |
| `execution` | Control, idle stepping, decimation, and command defaults |
| `interactive` | stdin, snapshot/history bounds, listeners, and transport limits |
| `planner` | Backend, request defaults, failure policy, workers, and batch limits |
| `playback` | Per-env trajectory queue, sample, and duration limits |
| `camera_output` | Queue, encoding, directory lifecycle, quota, and drain policy |
| `telemetry` | Env selection, rate, payload switches, topics, live endpoint, and MCAP path |
| `output` | CSV and MCAP existing-file policies |
| `paths` | Process cache root |
| `shutdown` | State publisher, camera publisher, and transport timeouts |

The resolution order is code defaults, then the selected runtime YAML, then
explicit CLI overrides. Entry-point parsers leave optional overrides unset, so
changing `--runtime-profile` changes the effective defaults as one unit. Use
`--dump-effective-config` to print the resolved mapping, fingerprint, and source
of every leaf field before Isaac starts. See `configs/runtime/example.yaml` and
[Realtime State Stream](telemetry.md).

Telemetry fields belong under `runtime.telemetry`, and CSV/MCAP existing-file
policy belongs under `runtime.output`; env profiles do not accept these fields.

Every built-in listener host must be `localhost` or a numeric loopback address.
The services provide neither authentication nor TLS. Remote access therefore
requires an authenticated TLS reverse proxy or SSH tunnel; a direct non-loopback
bind is invalid configuration.

MCAP file lifecycle is configured by
`runtime.output.mcap_existing_file_policy`.

## Robot Instances

Top-level `robots` must be a list. Never configure `robot_id`; the loader
assigns dense IDs in list order.

```yaml
robots:
  - label: robot_0
    robot_profile: ar5v2_l6v1_l
    prim_path: /World/Robots/robot_0  # optional
    controller_profile: default      # optional
    root_pose:
      xyz: [0.0, 0.09, 0.0]
      rpy: [-1.5707, 0.0, 0.0]
```

Labels are unique stable configuration identities. `prim_path` belongs to the
scene instance and defaults to `/World/Robots/<label>`; it does not belong in a
robot profile. IDs are session-local control indices and can change when the
list is reordered. Snapshots use label, profile, and fingerprint for persistent
matching.

`controller_profile` selects `configs/controllers/<bundle>/`. Resolution order
is env instance, robot profile, then runtime `profiles.controller_bundle`.

Each bundle contains `arm_controller.yaml` and `hand_controller.yaml`, with an
optional `default_controller.yaml`. An existing but incomplete bundle fails
validation.

The `position_control`, `velocity_control`, and `effort_control` sections each
declare `method`, `active_joints`, and `follower_joints`. A gain or limit accepts
a scalar, a sequence matching the selected joints, or an exact joint-name map;
all values must be finite and non-negative.

## Logging Profiles

Single Scene runtime consumes `profiles.logging` to select
`configs/logging/<name>.yaml`. Tiled Scene runtime does not create a joint-tracking
CSV logger or consume this profile.
`logging.enabled` controls whether the CSV opens, `joint_tracking_path` accepts
a path or `null`, `flush_interval_s` must be a positive finite number of
seconds, and `interval_steps` must be a positive integer. Column switches use
the explicit names `log_actual_position`, `log_actual_velocity`,
`log_command_position`, `log_command_velocity`, `log_command_effort`,
`log_action_effort`, `log_measured_effort`, and `log_applied_effort`. The last
two require more expensive PhysX effort reads and remain disabled by default.

## Robot Profiles

Every profile declares `robot.kind: arm|hand|arm_hand`, disjoint `joint_groups.arm/hand`, and its simulation asset. Profiles with cuRobo planning enabled also bind a planning URDF, TCP frames, and optionally a cuRobo robot YAML with collision spheres.

MJCF remains an Isaac simulation model. cuRobo v0.8.0 receives the planning URDF/robot YAML. A hand-only profile must set `curobo.enabled: false`; `arm_hand` plans the arm group and executes the hand group in command space.

Algorithm values belong in `configs/curobo/*.yaml`, not in robot profiles.

Importer fields are format-specific and unsupported fields fail validation.
MJCF supports `collision_approximation`, `fix_base`, `import_inertia_tensor`,
`import_sites`, `merge_fixed_joints` (default `false`), and robot
`self_collision`. URDF supports `collision_approximation`, `fix_base`,
`merge_fixed_joints` (default `true`), `collision_from_visuals`,
`import_inertia_tensor`, and robot `self_collision`.
Existing USD object references do not accept importer fields. URDF mimic uses
the native PhysX mimic constraint; MJCF equality followers are driven by the
runtime from actual master state.

## Object Instances

Object assets and physics belong in `configs/objects/`; placement belongs in
env `objects[]`. An omitted instance `prim_path` becomes
`/World/Objects/<name>`. Names, runtime handles, and effective prim paths must be
unique, while one object profile may be instantiated more than once.

Object profiles use the following project schema boundary:

```yaml
object:
  name: fixture
  kind: rigid                 # rigid | dynamic_chain
  source: urdf                # usd | urdf
  asset_path: assets/fixture.urdf
  physics:
    static: true
```

Validation eagerly dispatches the kind-specific importer, physics, material,
state-summary, and planning-collision parsers, so nested typos fail during
`validate-config`. Asset source belongs to `object.source`; instance placement
belongs to the env-owned `env.objects[].prim_path`.

## Planner Selection

Single Scene mode reads `runtime.planner.backend` by default and accepts an explicit
command-line override for one launch:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --planner-backend curobo --curobo-profile default
```

`linear` supports only joint-space `plan_cspace_goal` and
`plan_cspace_delta`. It does not create a cuRobo solver and does not provide IK,
collision checking, joint-limit validation, or constrained optimization.

The tiled async planner has no CLI backend override. Its backend, cuRobo profile,
and joint batch mode come from the runtime profile:

```yaml
runtime:
  profiles:
    curobo: default
  planner:
    backend: curobo        # curobo | linear
    joint_batch_mode: auto # auto | per_env | batch_only
```

`joint_batch_mode` applies to cuRobo joint planning. Synchronous tiled `ee_*`
actions use the robot's cuRobo IK binding independently of this async planner
selection.

## Validation

```bash
.venv/bin/python -m pytest -q tests/test_system_configs.py \
  tests/test_robot_instances.py tests/test_robot_capabilities.py
```

Use `.venv/bin/python` for tests that exercise the real Isaac importer,
USD/PhysX, or cuRobo GPU runtime.
