# YAML Configuration Reference

Language: [English](configuration.md) | [中文](../../zh-CN/reference/configuration.md)

This page is the field reference for project-owned YAML. It is sufficient to
author a profile without inferring fields from a bundled example. Every fixed
mapping rejects unknown keys, booleans are strict YAML booleans, and numeric
strings are not numbers. Unless a row explicitly says `null`, a present field
must have the stated type. "Default" means the value produced when the field is
omitted; it is not an instruction to copy the field into every profile.

Empty YAML, a non-mapping document, or a duplicate key at any nesting depth is
rejected before domain parsing, with the file and duplicate-key locations.

## Owners And Resolution

| Owner | Location | Owns | Does not own |
| --- | --- | --- | --- |
| Runtime profile | `configs/runtime/<name>.yaml` | Entry mode, selected profiles, process resources, execution, transports, planning policy, output, telemetry, and shutdown | scene topology, assets, controller gains, robot model resources |
| Env profile | `configs/envs/<name>.yaml` or `<name>/base.yaml` | World facts, visuals, sensors, robot/object instances, tiled layout | Robot/object asset properties, controller gains, cuRobo algorithms |
| Per-env fragment | `configs/envs/<name>/<dir>/*.yaml` | Pose overrides and opaque metadata for an existing tiled env | Topology, assets, physics, controllers, outputs, planning |
| Robot profile | `configs/robots/<name>.yaml` | One articulation's simulation asset, component groups, physics, and cuRobo model binding | Scene `prim_path`/`root_pose`, cuRobo algorithm defaults |
| Object profile | `configs/objects/<name>.yaml` | Object asset, import, physics, planning collision, and dynamic-chain summary | Scene `prim_path`/`root_pose` |
| Controller bundle | `configs/controllers/<name>/` | Arm/hand/default control methods, gains, limits, and follower drives | Contact material and rigid-body damping |
| cuRobo profile | `configs/curobo/<name>.yaml` | Device, seeds, tolerances, cache capacities, and planner algorithms | Robot URDF, robot YAML, frames, and TCP transforms |
| Logging profile | `configs/logging/<name>.yaml` | Single Scene joint-tracking CSV path, cadence, and columns | Telemetry MCAP and camera output |

Profile references are simple file stems, not paths. Runtime resolution is
`code defaults < selected runtime YAML < explicit entry-point CLI overrides`.
Controller bundle selection is `runtime default < robot profile < env robot
instance`. A selected cuRobo algorithm profile is the base and each robot's
`curobo.enabled`, `planning_joint_group`, and `robot` model binding are merged
over it. Object profile properties and env instance placement have no
overlapping fields.

## Complete Validator Option Table

`scripts/validate_config.py` is the pure-Python preflight entrypoint. It accepts
exactly these options:

| Option | Argparse default | Contract |
| --- | --- | --- |
| `--help` | n/a | Print argparse help and exit without loading a profile. |
| `--runtime-profile NAME` | `default_single_scene` | Select the file stem under `configs/runtime/`; paths are not accepted. |
| `--dump-effective-config` | `false` | On success, include resolved runtime values and each leaf's source instead of the minimal summary. |

Success writes one JSON document to stdout and exits `0`. A missing file,
invalid type, unknown field, or cross-profile validation error writes a
`CONFIG_INVALID` line to stderr and exits `1`. The command does not start Isaac,
create output files, or modify configuration.

```bash
.venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
.venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene \
  --dump-effective-config
```

## Runtime Profile

The document root must contain only a `runtime` mapping. Every subsection is
optional and recursively fills omitted leaves from the defaults below.

### Mode, Profiles, And Simulation App

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `runtime.mode` | `single_scene | tiled_scene`; `single_scene` | Must match the entry point. `tiled_scene` also requires the selected env to have `tiled.enabled: true`; `single_scene` requires false. |
| `runtime.profiles.env` | profile stem; `scene1` | Selects a single-file or directory-form env profile. |
| `runtime.profiles.curobo` | profile stem; `default` | Algorithm profile merged into every planning-enabled robot. |
| `runtime.profiles.logging` | profile stem; `default_logger` | Always graph-validated; the Single Scene runtime consumes it for joint CSV. |
| `runtime.profiles.controller_bundle` | bundle stem; `default` | Lowest-priority controller bundle selection. |
| `runtime.simulation_app.gui` | boolean; `false` | `true` launches the interactive window; `false` is headless. |
| `runtime.simulation_app.gpu.multi_gpu` | boolean; `false` | Launch intent; validation does not probe installed devices. |
| `runtime.simulation_app.gpu.max_gpu_count` | positive integer; `1` | Upper bound for both GPU indices. |
| `runtime.simulation_app.gpu.active_gpu` | integer >= 0; `0` | Must be less than `max_gpu_count`. |
| `runtime.simulation_app.gpu.physics_gpu` | integer >= 0; `0` | Must be less than `max_gpu_count`. |
| `runtime.simulation_app.render.gui_size` | `[width, height]`; `[1280, 720]` | Exactly two positive integers. |
| `runtime.simulation_app.render.headless_size` | `[width, height]`; `[640, 480]` | Exactly two positive integers. |
| `runtime.simulation_app.render.window_size` | `[width, height]`; `[1440, 900]` | Exactly two positive integers. |
| `runtime.simulation_app.render.renderer` | non-empty string; `RaytracedLighting` | Isaac renderer name. |
| `runtime.simulation_app.render.anti_aliasing_gui` | integer >= 0; `3` | Anti-aliasing level used in GUI mode. |
| `runtime.simulation_app.render.anti_aliasing_headless` | integer >= 0; `0` | Anti-aliasing level used in headless mode. |
| `runtime.simulation_app.render.samples_per_pixel_per_frame` | positive integer; `1` | Renderer samples per pixel per frame. |
| `runtime.simulation_app.render.denoiser` | boolean; `false` | Renderer denoiser switch. |
| `runtime.simulation_app.render.hide_ui` | boolean or `null`; `null` | `null` lets the launch layer choose from GUI/headless context. |
| `runtime.simulation_app.render.disable_viewport_updates` | boolean or `null`; `null` | `null` delegates the context-dependent choice. |
| `runtime.simulation_app.render.fast_shutdown` | boolean or `null`; `null` | `null` delegates the context-dependent choice. |
| `runtime.simulation_app.render.material_sync_loads` | boolean; `false` | Material synchronous-load setting. |
| `runtime.simulation_app.render.hydra_material_sync_loads` | boolean; `false` | Hydra material synchronous-load setting. |
| `runtime.simulation_app.render.headless_dt_policy` | `camera_aware | physics`; `camera_aware` | `camera_aware` preserves render cadence when an enabled camera has an output consumer; `physics` always follows physics cadence in headless mode. |

Detailed camera creation and render behavior are owned by
[Cameras And Output](../guides/cameras.md).

### Execution And Interactive Transport

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `runtime.execution.control_mode` | `position | velocity | effort`; `position` | Tiled Scene mode currently requires `position`. |
| `runtime.execution.idle_physics_policy` | `pause | hold_step`; `hold_step` | Whether an idle runtime pauses or continues hold steps. |
| `runtime.execution.idle_step_duration_s` | positive finite number; `0.05` | Duration represented by an idle hold interval. |
| `runtime.execution.default_decimation` | positive integer; `2` | Default physics-step multiplier for commands that omit decimation. |
| `runtime.execution.command_defaults.joint_interpolation` | `linear | smoothstep`; `smoothstep` | Used only when a command omits the field. |
| `runtime.execution.command_defaults.pose_frame` | `env | world`; `env` | Default task-space reference frame. |
| `runtime.execution.command_defaults.orientation_mode` | `free | current | target`; `current` | Default task-space orientation treatment. |
| `runtime.interactive.stdin_enabled` | boolean; `true` | Enables the stdin command reader. |
| `runtime.interactive.stdin_eof_policy` | `exit | keep_alive`; `exit` | Process behavior after stdin closes. |
| `runtime.interactive.queue_poll_timeout_s` | positive finite number; `0.05` | Internal command queue poll timeout. |
| `runtime.interactive.snapshot_timeout_s` | positive finite number; `30.0` | Snapshot request completion timeout. |
| `runtime.interactive.command_history_capacity` | integer >= 0; `256` | In-memory command history; zero disables retention. |
| `runtime.interactive.snapshot_request_capacity` | positive integer; `32` | Maximum queued snapshot requests. |
| `runtime.interactive.transport.tcp_jsonl.enabled` | boolean; `false` | Enables the control JSONL TCP listener. |
| `runtime.interactive.transport.tcp_jsonl.host` | loopback host; `127.0.0.1` | Only `localhost` or a numeric loopback address is accepted. |
| `runtime.interactive.transport.tcp_jsonl.port` | `null` or integer 1..65535; `null` | Required when this endpoint is enabled. |
| `runtime.interactive.transport.websocket.enabled` | boolean; `false` | Enables the control WebSocket listener. |
| `runtime.interactive.transport.websocket.host` | loopback host; `127.0.0.1` | Same loopback-only rule. |
| `runtime.interactive.transport.websocket.port` | `null` or integer 1..65535; `null` | Required when this endpoint is enabled. |
| `runtime.interactive.transport.max_message_bytes` | positive integer; `1048576` | Per-message input limit. |
| `runtime.interactive.transport.max_connections` | positive integer; `16` | Concurrent network connection limit. |
| `runtime.interactive.transport.request_queue_capacity` | positive integer; `256` | Accepted request queue bound. |
| `runtime.interactive.transport.event_queue_capacity` | positive integer; `256` | Outbound event queue bound. |
| `runtime.interactive.transport.overflow_policy` | `reject`; `reject` | A full queue rejects new work. |
| `runtime.interactive.transport.startup_timeout_s` | positive finite number; `5.0` | Listener startup timeout. |
| `runtime.interactive.transport.server_poll_interval_s` | positive finite number; `0.1` | Server-side poll interval. |
| `runtime.interactive.transport.response_poll_interval_s` | positive finite number; `0.5` | Response poll interval. |

Listener endpoints do not provide authentication or TLS. A remote client must
connect through an authenticated TLS proxy or SSH tunnel; a non-loopback bind is
invalid configuration. Message schemas are owned by
[Single Scene JSON Protocol](single-scene-json.md) and [Tiled Scene JSON Protocol](tiled-scene-json.md).

### Planner And Playback

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `runtime.planner.backend` | `curobo | linear`; `curobo` | `linear` cannot satisfy task-space or collision-aware requests. |
| `runtime.planner.joint_batch_mode` | `auto | per_env | batch_only`; `auto` | cuRobo joint-planning dispatch policy. |
| `runtime.planner.request_defaults.duration_s` | positive finite number; `1.0` | Default requested motion duration. |
| `runtime.planner.request_defaults.avoid_collisions` | boolean; `false` | Cannot be true with the `linear` backend. |
| `runtime.planner.request_defaults.force_collision_refresh` | boolean; `false` | Unsupported in Tiled Scene mode. |
| `runtime.planner.request_defaults.coordination` | `independent | static_others | coupled`; `independent` | `coupled` is rejected because no coupled backend exists; tiled requires `independent`. |
| `runtime.planner.request_defaults.load_on_success` | boolean; `true` | Load a successful trajectory into playback. |
| `runtime.planner.request_defaults.replace` | boolean; `true` | Replace an existing queued trajectory when loading. |
| `runtime.planner.oversize_request_policy` | `split | reject`; `split` | Handling for requests larger than the batch limit. |
| `runtime.planner.failure_policy` | `hold_failed_env | reject_request`; `hold_failed_env` | Atomic response behavior when some envs fail. |
| `runtime.planner.resources.max_workers` | positive integer; `2` | Async planning worker limit. |
| `runtime.planner.resources.max_pending_requests` | positive integer; `64` | Pending request bound. |
| `runtime.planner.resources.max_completed_results` | integer >= 0; `256` | Completed-result retention; zero disables retention. |
| `runtime.planner.resources.max_batch_problems` | positive integer or `auto`; `64` | `auto` resolves to the env count, capped by selected cuRobo capacities. An explicit cuRobo value cannot exceed the smaller IK/planner `max_batch_size`. |
| `runtime.planner.resources.shutdown_timeout_s` | positive finite number; `30.0` | Planner worker join timeout. |
| `runtime.playback.max_queue_depth_per_env` | positive integer; `32` | Per-env trajectory queue depth. |
| `runtime.playback.max_samples_per_env` | positive integer; `100000` | Per-env queued sample bound. |
| `runtime.playback.max_duration_s_per_env` | positive finite number; `3600.0` | Per-env queued duration bound. |
| `runtime.playback.overflow_policy` | `reject`; `reject` | Rejects work that would exceed any playback bound. |

Planner capabilities and request semantics are owned by
[Motion Planning](../guides/motion-planning.md).

### Camera Output, Telemetry, Paths, And Shutdown

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `runtime.camera_output.queue_size` | positive integer; `128` | Shared async camera publication queue bound. |
| `runtime.camera_output.overflow_policy` | `drop_oldest | drop_newest | block | error`; `block` | Dataset writers should normally use lossless `block` or `error`. |
| `runtime.camera_output.worker_poll_interval_s` | positive finite number; `0.1` | Publisher worker poll interval. |
| `runtime.camera_output.existing_data_policy` | `error | truncate | resume | timestamped_dir`; `error` | Existing camera directory policy. |
| `runtime.camera_output.shutdown_policy` | `drain | abort`; `drain` | Flush or discard pending frames during shutdown. |
| `runtime.camera_output.rgb_format` | `ppm | png | npy`; `ppm` | Encoding for RGB payloads. |
| `runtime.camera_output.depth_format` | `npy | npz`; `npy` | Encoding for depth payloads. |
| `runtime.camera_output.metadata_flush_interval_frames` | positive integer; `1` | Metadata flush cadence. |
| `runtime.camera_output.max_bytes_per_camera` | positive integer; `10737418240` | Per-camera directory quota, including metadata and modalities. |
| `runtime.telemetry.primary_env_id` | integer >= 0; `0` | Source env for standard single-env topics. Tiled Scene runtime profiles must declare it explicitly. |
| `runtime.telemetry.selected_env_ids` | non-empty unique integer list; `[0]` | Values are >= 0. Tiled Scene runtime profiles must declare it explicitly; it must contain `primary_env_id` and stay below `tiled.num_envs`. Single Scene mode requires `[0]`. |
| `runtime.telemetry.publish_decimation` | positive integer; `1` | Tiled Scene global-step publication decimation; Single Scene mode requires `1`. |
| `runtime.telemetry.rate_hz` | finite number >= 0; `60.0` | Telemetry sampling rate. |
| `runtime.telemetry.buffer_size` | positive integer; `1` | State stream buffer bound. |
| `runtime.telemetry.drop_policy` | `latest | drop_oldest | drop_newest`; `latest` | Buffer overflow behavior. |
| `runtime.telemetry.on_error` | `stop | continue`; `stop` | Publisher error behavior. |
| `runtime.telemetry.include_joint_states` | boolean; `true` | Include standard joint state messages. |
| `runtime.telemetry.include_state_json` | boolean; `true` | Include project state JSON. |
| `runtime.telemetry.include_scene_markers` | boolean; `false` | Include scene marker messages. |
| `runtime.telemetry.include_efforts` | boolean; `false` | Read and include effort data. |
| `runtime.telemetry.include_objects` | boolean; `false` | Include runtime object state. |
| `runtime.telemetry.joint_effort_field` | `none | commanded | measured | applied`; `none` | A non-`none` value requires `include_efforts: true` and is supported only in Single Scene mode. |
| `runtime.telemetry.topics.joint_states` | absolute topic; `/joint_states` | Must start with `/`, contain no `//` or `..`, and differ from the other two topics. |
| `runtime.telemetry.topics.scene` | absolute topic; `/scene` | Same topic rules. |
| `runtime.telemetry.topics.state` | absolute topic; `/linkerbot/state` | Same topic rules. |
| `runtime.telemetry.mcap.path` | path string or `null`; `null` | `null` disables this sink; a path cannot contain NUL or a `..` component. |
| `runtime.telemetry.foxglove_live.enabled` | boolean; `false` | Enables the telemetry live server. |
| `runtime.telemetry.foxglove_live.host` | loopback host; `127.0.0.1` | Loopback-only. |
| `runtime.telemetry.foxglove_live.port` | `null` or integer 1..65535; `null` | Required when enabled. |
| `runtime.output.csv_existing_file_policy` | `error | truncate | resume | timestamped_dir`; `error` | Existing joint CSV policy. |
| `runtime.output.mcap_existing_file_policy` | same enum; `error` | Existing telemetry MCAP policy. |
| `runtime.paths.cache_root` | path string or `null`; `null` | `null` delegates cache-root selection; non-null paths cannot contain NUL or `..`. Relative paths use the process working directory. |
| `runtime.shutdown.state_publisher_timeout_s` | positive finite number; `2.0` | State publisher join timeout. |
| `runtime.shutdown.camera_publisher_timeout_s` | positive finite number; `2.0` | Camera publisher join timeout. |
| `runtime.shutdown.transport_timeout_s` | positive finite number; `2.0` | Interactive transport join timeout. |

When a telemetry live or MCAP sink is configured, at least one of joint states,
state JSON, or scene markers must be enabled. In Single Scene mode, a configured sink
with scene markers also requires `include_objects: true`. Detailed payload,
buffer, sink, resume, and file behavior is owned by
[Realtime State Stream](../guides/telemetry.md).

## Env Profile

An env profile accepts only the sibling top-level keys `env`, `solver`,
`visuals`, `sensors`, `robots`, `objects`, and `tiled`. `env` and a non-empty
`robots` list are required. `objects` may be omitted or `null`; all other
present sections must be mappings or lists of the declared type.

### World And Visual Fields

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `env.name` | required non-empty string | Stable scene name. |
| `env.description` | optional string | Human/logging context only. |
| `env.gravity_z` | finite number; `-9.81` | World Z gravity in m/s^2. |
| `env.add_ground` | boolean; `true` | Add the Isaac default ground. |
| `env.ground_height` | finite number; `0.0` | Default-ground Z coordinate. |
| `env.physics_frequency` | positive finite number; `600.0` | Physics steps per second. |
| `env.render_frequency` | positive finite number; `100.0` | Render frames per second when rendering is required. |
| `solver.type` | `PGS | TGS | null`; `null` | Scene-level PhysX solver override. Robot iteration counts do not belong here. |
| `visuals.viewport.enabled` | boolean; `true` | Configure the GUI viewport when enabled. |
| `visuals.viewport.eye` | finite `[x, y, z]`; `[1.35, -1.65, 1.05]` | Viewport eye position. |
| `visuals.viewport.target` | finite `[x, y, z]`; `[0.0, -0.1, 0.42]` | Viewport look-at target. |
| `visuals.viewport.prim_path` | absolute USD path; `/OmniverseKit_Persp` | Viewport camera prim. |
| `visuals.lights.key.enabled` | boolean; `true` | Distant key-light switch. |
| `visuals.lights.key.path` | absolute USD path; `/World/KeyLight` | Key-light prim. |
| `visuals.lights.key.intensity` | finite number >= 0; `1200.0` | Light intensity. |
| `visuals.lights.key.angle` | finite number >= 0; `0.5` | Distant-light angular size. |
| `visuals.lights.key.color` | finite RGB triple or `null`; `null` | `null` preserves the light default. |
| `visuals.lights.key.rotation_rpy` | finite RPY triple or `null`; `null` | Radians. |
| `visuals.lights.fill.enabled` | boolean; `true` | Dome fill-light switch. |
| `visuals.lights.fill.path` | absolute USD path; `/World/FillLight` | Fill-light prim. |
| `visuals.lights.fill.intensity` | finite number >= 0; `250.0` | Light intensity. |
| `visuals.lights.fill.color` | finite RGB triple or `null`; `null` | `null` preserves the light default. |

### Sensor Cameras

`sensors` accepts only `cameras`; `sensors.cameras` is a mapping whose arbitrary
keys are camera names. A name must be non-empty and contain no path separator.

| Path under `sensors.cameras.<name>` | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `enabled` | boolean; `true` | Disabled cameras do not create runtime resources. |
| `prim_path` | required absolute USD path | Camera prim template. |
| `parent_prim_path` | absolute USD path or `null`; `null` | If set, `prim_path` must be below it. |
| `pose.xyz`, `pose.rpy` | finite triples; zero triples | Local pose in metres/radians. `pose: null` is equivalent to omission. |
| `resolution` | two positive integers; `[640, 480]` | `[width, height]`. |
| `frequency` | positive finite number; `30.0` | Capture frequency in Hz. |
| `env_ids` | non-empty unique integer list >= 0; omitted | Must be omitted in env profiles selected by Single Scene and explicitly present for every camera in env profiles with `tiled.enabled: true`; values must be below `tiled.num_envs`. |
| `modalities` | non-empty unique list; `[rgb]` | Values: `rgb`, `depth`, `semantic_segmentation`, `instance_segmentation`. |
| `clipping_range` | finite `[near, far]`; `[0.01, 5.0]` | Must satisfy `0 < near < far`. |
| `intrinsics.fx`, `intrinsics.fy` | required positive finite numbers when `intrinsics` is present | Pixel focal lengths. The whole section may be omitted or `null`. |
| `intrinsics.cx`, `intrinsics.cy` | required finite numbers when `intrinsics` is present | Pixel principal point. |
| `output.save_dir` | non-empty string or `null`; `null` | Camera dataset directory. |
| `output.foxglove_topic_prefix` | absolute topic prefix or `null`; `null` | Must start with `/` when present. |
| `output.foxglove_live_host` | loopback host; `127.0.0.1` | Camera live server bind host. |
| `output.foxglove_live_port` | positive integer or `null`; `null` | Non-null enables a camera live consumer. |
| `output.foxglove_mcap_path` | non-empty string or `null`; `null` | Non-null enables camera MCAP output. |

In Tiled Scene mode, the template is expanded only for `env_ids`. Save directories and
topic prefixes gain an `env_NNN` suffix. A per-env camera pose override is legal
only when that env is in the camera's `env_ids`. Resolved `save_dir` values must
be unique per camera. If any camera writes a directory or MCAP, runtime
`overflow_policy` must be `block` or `error`; lossy policies are live-only.
Camera MCAP also rejects `existing_data_policy: resume`. See
[Cameras And Output](../guides/cameras.md) for creation, cadence, encoding, and
resume behavior.

### Robot And Object Instances

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `robots[].robot_profile` | required profile stem | Selects `configs/robots/<name>.yaml`. |
| `robots[].label` | string matching `[A-Za-z0-9_]+`; `<robot_profile>_<list-index>` | Must be unique. List order generates the session-local dense `robot_id`; `robot_id` is not configurable. |
| `robots[].prim_path` | canonical absolute USD path; `/World/Robots/<label>` | Must be unique and disjoint from every robot/object instance subtree. |
| `robots[].controller_profile` | bundle stem or `null`; `null` | Highest-priority controller bundle selection. |
| `robots[].root_pose.xyz`, `robots[].root_pose.rpy` | required mapping; omitted vectors are zero triples | Finite world pose in metres/radians. |
| `objects[].name` | required `[A-Za-z_][A-Za-z0-9_]*` | Stable scene identity; unique. |
| `objects[].object_profile` | required profile stem | Selects `configs/objects/<name>.yaml`. |
| `objects[].runtime_handle` | non-empty string or `null`; `null` | Optional interactive alias; unique and cannot collide with another object's name. |
| `objects[].prim_path` | canonical absolute USD path; `/World/Objects/<name>` | Must be unique and disjoint from every instance subtree. |
| `objects[].root_pose.xyz`, `objects[].root_pose.rpy` | required mapping; omitted vectors are zero triples | Finite world pose in metres/radians. |

Scene placement never belongs in a robot or object profile.

### Tiled Base

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `tiled.enabled` | boolean; `false` | Directory-form loading forces the effective value to `true`. Must agree with `runtime.mode`. |
| `tiled.num_envs` | positive integer; `1` | Sole effective env-count owner. A directory loader derives `max(env_id)+1` only when base omits it and fragments exist. |
| `tiled.base_env_path` | absolute USD path; `/World/envs` | Cannot be `/` or contain `//`. |
| `tiled.env_prefix` | non-empty string; `env` | Cannot contain `/`; roots become `<base>/<prefix>_<id>`. |
| `tiled.spacing` | positive finite number; `2.0` | XY grid spacing. |
| `tiled.num_per_row` | positive integer or `null`; `null` | `null` uses `ceil(sqrt(num_envs))`. |
| `tiled.per_env_config_dir` | safe relative directory or `null`; `null` | Directory loader uses `envs` when omitted; absolute paths, `.` and `..` are rejected. |
| `tiled.per_env` | sequence of per-env rows; `[]` | A single-file profile may declare the row schema below inline. A directory loader replaces this value with rows materialized from its fragment directory. |
| `tiled.layout.origin_xyz` | finite triple; `[0, 0, 0]` | World translation of the tiled grid. |
| `tiled.clone.replicate_physics` | boolean; `true` | Request PhysX structure replication. |
| `tiled.clone.copy_from_source` | boolean; `false` | GridCloner copy/inherit behavior. |
| `tiled.clone.enable_env_ids` | boolean; `false` | GridCloner env-ID authoring. |
| `tiled.clone.filter_collisions` | boolean; `true` | Enable inter-env collision filtering. |
| `tiled.clone.collision_filter_strategy` | `collision_groups | filtered_pairs`; `collision_groups` | `collision_groups` is the linear-authoring path; `filtered_pairs` is pairwise. |
| `tiled.clone.collision_root_path` | absolute USD path; `/World/collisions` | Cannot be `/`. Non-default collision-group fields require filtering enabled with `collision_groups`. |
| `tiled.clone.physics_scene_path` | absolute USD path or `null`; `null` | `null` discovers the unique PhysicsScene. It is not the string `auto`. Requires active `collision_groups` filtering when explicitly set. |
| `tiled.clone.global_collision_paths` | `auto` or absolute-path list; `auto` | Explicit paths replace automatic standard-ground discovery and require active `collision_groups` filtering. |
| `tiled.clone.extra_global_collision_paths` | absolute-path list; `[]` | Appended after automatic/explicit globals; requires active `collision_groups` filtering when non-empty. |
| `tiled.diagnostics.inspect_env_ids` | unique integer list; `[0]` | Every value must be in `[0, num_envs)`. Affects diagnostics only. |

### Directory Profiles And Per-env Overrides

A directory profile has one shared topology and optional fragments:

```text
configs/envs/<name>/base.yaml
configs/envs/<name>/envs/env_000.yaml
configs/envs/<name>/envs/env_001.yaml
```

The row schema below can be written directly as `tiled.per_env` in a single-file
profile. In a directory profile, use fragment files: the loader reads
`per_env_config_dir`, sorts fragments by `env_id`, and replaces any base
`tiled.per_env` value with those materialized rows before full cross-field
validation. An explicit base `tiled.num_envs` wins and every fragment ID must
fit it.

| Fragment path | Type and requirement | Rules |
| --- | --- | --- |
| `env_id` | required integer >= 0 | Unique and less than effective `tiled.num_envs`. |
| `robots.<label>.root_pose.xyz/rpy` | both required finite triples | Label must already exist in base `robots`; the pose is env-local. |
| `objects.<name>.root_pose.xyz/rpy` | finite triples; omitted vector becomes zero | Name must already exist in base `objects`. Write both vectors to avoid resetting an omitted component to zero. |
| `cameras.<name>.pose.xyz/rpy` | finite triples; omitted vector becomes zero | Camera must exist in base and `env_id` must be in that camera's `env_ids`. |
| `metadata` | JSON-compatible mapping; `{}` | String keys and finite JSON scalar/list/object values only; not interpreted as runtime configuration. |

Fragments cannot add topology or override assets, physics, controllers, output,
or planner settings.

## Robot Profile

The document root accepts only `robot`, `curobo`, `joint_groups`, optional
`rigid_body_groups`, and optional `controlled_joints`.

### Simulation Asset And Physics

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `robot.kind` | required `arm | hand | arm_hand` | Requires the corresponding non-empty joint groups. |
| `robot.name` | non-empty string; `robot` | Logical profile name; an env instance label becomes the runtime articulation name. |
| `robot.controller_profile` | bundle stem or `null`; `null` | Middle-priority controller bundle selection. |
| `robot.asset_type` | `mjcf | urdf`; `mjcf` | Simulation importer type. |
| `robot.asset_path` | required non-empty path string | Repository-relative or absolute asset path. |
| `robot.urdf_drive_type` | `none | position`; `position` | Legal only for a URDF robot. |
| `robot.import.collision_approximation` | `convex_decomposition | convex_hull`; `convex_decomposition` | Importer collision geometry. |
| `robot.import.self_collision` | boolean; `false` | Robot-only articulation self-collision switch. |
| `robot.import.fix_base` | boolean or `null`; `null` | `null` produces the current robot importer default `true`. |
| `robot.import.merge_fixed_joints` | boolean or `null`; `null` | Effective default is `false` for MJCF and `true` for URDF. |
| `robot.import.import_inertia_tensor` | boolean; `true` | Supported by MJCF and URDF. |
| `robot.import.import_sites` | boolean; `true` | MJCF only. |
| `robot.import.collision_from_visuals` | boolean; `false` | URDF only. |
| `robot.physics.gravity.default` | boolean; `false` | Fallback rigid-body gravity policy. |
| `robot.physics.gravity.arm`, `robot.physics.gravity.hand` | boolean or omitted; inherit `default` | Per-component gravity policy. Explicit `null` is not accepted. |
| `robot.physics.solver.arm.position_iterations`, `robot.physics.solver.arm.velocity_iterations` | integer >= 0 or omitted | Per-arm rigid-body PhysX overrides. |
| `robot.physics.solver.hand.position_iterations`, `robot.physics.solver.hand.velocity_iterations` | integer >= 0 or omitted | Per-hand rigid-body PhysX overrides. |
| `robot.planning_collision.spheres[]` | non-empty list when section exists | Backend-neutral conservative robot-root envelopes. |
| `...spheres[].name` | non-empty string; `sphere_<index>` | Unique within the list. |
| `...spheres[].center` | required finite triple | Robot-root-local metres. |
| `...spheres[].radius` | required positive finite number | Metres. |

`robot.physics.physx` accepts common `material`/`rigid_body` fields and optional
`default`, `arm`, and `hand` mappings with the same two submappings. Common
fields are applied first, then `default`, then the component override.

| PhysX leaf below a common/component mapping | Type and omission | Rules |
| --- | --- | --- |
| `material` | mapping, `null`, or `preserve`; omitted means inherit | `null`/`preserve` explicitly keeps the asset material binding. A mapping selects an override. |
| `material.contact_static_friction` | finite number >= 0 or omitted | Robot contact material override. |
| `material.contact_dynamic_friction` | finite number >= 0 or omitted | Robot contact material override. |
| `material.contact_restitution` | finite number in `[0, 1]` or omitted | Robot contact material override. |
| `material.friction_combine_mode` | `average | min | multiply | max | preserve | null`; `average` for a mapping | `preserve`/`null` leaves the combine mode untouched. |
| `rigid_body.linear_damping` | finite number >= 0 or omitted | Rigid-body damping override. |
| `rigid_body.angular_damping` | finite number >= 0 or omitted | Rigid-body damping override. |

### Component Groups And Control Selection

| Path | Type and omission default | Rules |
| --- | --- | --- |
| `joint_groups.arm` | exact-name list; `[]` | Non-empty exactly when `kind` contains an arm. |
| `joint_groups.hand` | exact-name list; `[]` | Non-empty exactly when `kind` contains a hand. |
| `joint_groups.passive` | exact-name list; `[]` | Command-space joints intentionally not written by arm/hand controllers. |
| `rigid_body_groups.arm`, `rigid_body_groups.hand`, `rigid_body_groups.default` | exact-name lists; omitted mapping | Optional explicit component classification. |
| `controlled_joints` | non-empty exact-name list; `[all]` | `[all]` must stand alone; otherwise every name must be in arm/hand groups. |

Names cannot repeat within or across component groups. Group order defines
command-space order. Asset finalization checks names against the articulation;
planning active joints must exactly match `joint_groups.arm`.

### Robot cuRobo Model Binding

`curobo.enabled` is required. With `false`, it must be the only key in the
section. With `true`, `planning_joint_group: arm` and a non-empty `robot` mapping
are required; a hand-only robot cannot enable planning.

| Path under `curobo.robot` | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `robot_config_path` | path string or `null`; `null` | Full cuRobo robot YAML. At least this or `urdf_path` is required. |
| `urdf_path` | path string or `null`; `null` | Planning URDF. |
| `base_link` | non-empty string or `null`; inferred | With a URDF, omission requires exactly one inferable root link. |
| `flange_frame` | non-empty string or `null`; `null` | Default parent candidate for custom TCPs. |
| `tool_frames` | string list; `[]` | Exact existing model frames; no duplicates. |
| `default_tcp_frame` | non-empty string or `null`; `null` | At least `tool_frames` or this field is required. |
| `custom_tcps` | named mapping or frame list; `[]` | Fixed frames materialized before context creation. |
| `custom_tcps.<name>.parent_frame` | non-empty string or omitted | Defaults to `flange_frame`, then `default_tcp_frame`; required if neither exists. |
| `custom_tcps.<name>.xyz`, `custom_tcps.<name>.rpy` | finite triples; zero triples | Parent-local metres/radians. In list form, each row also requires `frame_name`. |
| `load_collision_spheres` | boolean; `true` | Load sphere data from the robot config when present. |

## Object Profile

The root contains only `object`. Instance `prim_path` and `root_pose` are not
legal here.

| Path | Type and omission default | Rules and meaning |
| --- | --- | --- |
| `object.name` | non-empty string; profile stem | Logical asset name. |
| `object.kind` | required `rigid | dynamic_chain` | Selects the strict consumer schema. |
| `object.source` | required `usd | urdf` | `dynamic_chain` requires `usd`. |
| `object.asset_path` | required non-empty path string | Repository-relative or absolute asset path. |
| `object.root_path` | absolute USD path or `null`; `null` | `dynamic_chain` only; the current capsule-rope consumer defaults to `/CapsuleRope`. |
| `object.urdf_drive_type` | `none | position`; `none` | Rigid URDF only. |
| `object.state_summary.reference_body` | required non-empty body name for `dynamic_chain` | A name, not a prim path. Forbidden for rigid objects. |

A rigid URDF accepts `object.import` with the same format-specific fields as the
robot importer except `self_collision`. A rigid USD does not accept `import`.
For a rigid object, omitted `fix_base` follows `physics.static`; explicit
`fix_base: true` conflicts with `physics.static: false`.

| Rigid path | Type and omission default | Rules |
| --- | --- | --- |
| `object.physics.static` | boolean; `false` | Freeze/fix the rigid object. |
| `object.physics.material.static_friction` | finite number >= 0 or omitted | Object material override. |
| `object.physics.material.dynamic_friction` | finite number >= 0 or omitted | Object material override. |
| `object.physics.material.restitution` | finite number in `[0, 1]` or omitted | Object material override. |
| `object.physics.material.friction_combine_mode` | `average | min | multiply | max` or omitted | Object material override. |
| `object.planning_collision.shape` | required `cuboid | sphere | capsule` | Simplified planning geometry; does not change PhysX colliders. |
| `object.planning_collision.size` | required positive-number list | Length 3 for cuboid, 1 (radius) for sphere, 2 (`radius`, `length`) for capsule. |
| `object.planning_collision.xyz`, `object.planning_collision.rpy` | finite triples; zero triples | Object-local collision pose. |
| `object.planning_collision.enabled` | boolean; `true` | Include in planning world. |
| `object.planning_collision.padding` | finite number >= 0; `0.0` | Conservative shape padding. |

The current `dynamic_chain` consumer is a generated capsule rope. It forbids
`import`, `planning_collision`, and `urdf_drive_type`.

| Dynamic-chain path | Type and omission | Rules |
| --- | --- | --- |
| `object.physics.material.static_friction` | finite number >= 0 or omitted | Runtime USD material override. |
| `object.physics.material.dynamic_friction` | finite number >= 0 or omitted | Runtime USD material override. |
| `object.physics.material.restitution` | finite number in `[0, 1]` or omitted | Runtime USD material override. |
| `object.physics.material.friction_combine_mode` | `average | min | multiply | max` or omitted | Runtime USD material override. |
| `object.physics.solver_position_iterations` | positive integer or omitted | Runtime rigid-body position-iteration override. |
| `object.physics.solver_velocity_iterations` | integer >= 0 or omitted | Runtime rigid-body velocity-iteration override. |

Asset generation versus runtime ownership is described in
[Object Assets](../development/object-assets.md); collision behavior is owned by
[Collision Models](../guides/collision-models.md).

## Controller Bundle

A bundle directory must contain `arm_controller.yaml` and
`hand_controller.yaml`; `default_controller.yaml` is optional. Each file has the
same schema. `target` defaults to the file role and, when present, must equal
`arm`, `hand`, or `default` respectively. Bundle names match
`[A-Za-z0-9][A-Za-z0-9_-]*`.

Each of `position_control`, `velocity_control`, and `effort_control` accepts only
`method`, `active_joints`, and `follower_joints`. All three modes are parsed when
the bundle is loaded, not only the selected runtime mode.

| Path pattern | Type and omission default | Rules |
| --- | --- | --- |
| `position_control.method` | `implicit | explicit`; `implicit` | PhysX drive or Python PD effort. |
| `velocity_control.method` | `implicit | explicit`; `implicit` | PhysX velocity drive or Python velocity-error effort. |
| `effort_control.method` | `direct`; `direct` | Direct bounded effort. |
| `<mode>.active_joints.stiffness` | joint parameter; `1000.0` | Used by position control; accepted for every mode. |
| `<mode>.active_joints.damping` | joint parameter; `50.0` | Position/velocity gain. |
| `<mode>.active_joints.max_force` | joint parameter; `100.0` | Drive/effort bound. |
| `<mode>.active_joints.effort_limit` | joint parameter or `null`; `null` | Direct effort symmetric bound. |
| `<mode>.active_joints.joint_friction` | joint parameter; `0.5` | Default joint friction. |
| `<mode>.follower_joints.stiffness` | joint parameter; `50000.0` | Follower position-drive stiffness in every active mode. |
| `<mode>.follower_joints.damping` | joint parameter; `50.0` | Follower position-drive damping. |
| `<mode>.follower_joints.max_force` | joint parameter; `100.0` | Follower drive bound. |
| `<mode>.follower_joints.joint_friction` | joint parameter; `0.5` | Follower joint friction. |

A joint parameter is either one finite non-negative scalar, a non-empty finite
non-negative sequence in selected-joint order, or a non-empty exact
`joint_name: value` mapping. Sequence length and mapping names are resolved
against the imported articulation before execution.

## cuRobo Algorithm Profile

The document root contains only `curobo`. This owner accepts `task_bundle`,
`device`, `kinematics`, and `motion_planner`; it does not accept `enabled`,
`planning_joint_group`, `robot`, or arbitrary task-file paths.

| Path | Type and omission default | Rules |
| --- | --- | --- |
| `curobo.task_bundle` | `curobo_v0_8_default`; same | The installed cuRobo runtime must be `0.8.0`. |
| `curobo.device.device` | non-empty string; `cuda:0` | Torch device used by the backend. |
| `curobo.device.tensor_dtype` | `float32`; `float32` | Project-validated tensor dtype. |
| `curobo.device.collision_geometry_dtype` | `float32`; `float32` | Collision geometry dtype. |
| `curobo.device.collision_gradient_dtype` | `float32`; `float32` | Collision gradient dtype. |
| `curobo.device.collision_distance_dtype` | `float32`; `float32` | Collision distance dtype. |

### IK Algorithm

All fields below are under `curobo.kinematics.ik`.

| Leaf | Type and omission default | Rules |
| --- | --- | --- |
| `num_seeds` | positive integer; `32` | Optimizer seeds per problem. |
| `position_tolerance` | finite number >= 0; `0.002` | Metres. |
| `orientation_tolerance` | finite number >= 0; `0.01` | Radians. |
| `use_cuda_graph` | boolean; `true` | CUDA graph execution switch. |
| `random_seed` | integer >= 0; `123` | Reproducible seed generation. |
| `optimizer_collision_activation_distance` | finite number >= 0; `0.01` | Metres. |
| `store_debug` | boolean; `false` | Retain solver debug data. |
| `override_optimizer_num_iters.particle`, `override_optimizer_num_iters.lbfgs` | integer >= 0 or `null`; `null` | `null` uses task-bundle defaults. No other keys are accepted. |
| `override_iters_for_multi_link_ik` | integer >= 0 or `null`; `null` | Multi-link iteration override. |
| `optimization_dt` | positive finite number or `null`; `null` | Velocity-aware IK timestep. |
| `velocity_regularization_weight` | finite number >= 0 or `null`; `null` | C-space rollout regularization. |
| `acceleration_regularization_weight` | finite number >= 0 or `null`; `null` | C-space rollout regularization. |
| `success_requires_convergence` | boolean; `true` | Require pose-error convergence as well as feasibility. |
| `seed_position_weight` | finite number >= 0; `1.0` | Seed solver weight. |
| `seed_orientation_weight` | finite number >= 0; `1.0` | Seed solver weight. |
| `seed_velocity_weight` | finite number >= 0; `0.0` | Seed solver weight. |
| `seed_acceleration_weight` | finite number >= 0; `0.0` | Seed solver weight. |
| `seed_solver_num_seeds` | positive integer; `32` | Seed solver population. |
| `max_batch_size` | positive integer; `256` | IK resource capacity. |
| `multi_env` | boolean; `false` | Whether each batch problem owns a distinct collision world. |
| `max_goalset` | positive integer; `1` | Goal-set capacity per problem. |
| `self_collision_check` | boolean; `true` | cuRobo model self-collision check. |
| `collision_cache.cuboid`, `collision_cache.mesh` | integer >= 0; omitted/empty | Preallocated obstacle counts; no other geometry keys are accepted. |

### Motion Planner Algorithm

All fields below are under `curobo.motion_planner`.

| Leaf | Type and omission default | Rules |
| --- | --- | --- |
| `warmup` | boolean; `true` | Warm the planner after materialization. |
| `num_ik_seeds` | positive integer; `32` | Goal IK seed count. |
| `num_trajopt_seeds` | positive integer; `4` | Trajectory-optimization seed count. |
| `position_tolerance` | finite number >= 0; `0.002` | Metres. |
| `orientation_tolerance` | finite number >= 0; `0.01` | Radians. |
| `use_cuda_graph` | boolean; `true` | CUDA graph execution switch. |
| `random_seed` | integer >= 0; `123` | Reproducible initialization. |
| `optimizer_collision_activation_distance` | finite number >= 0; `0.01` | Metres. |
| `store_debug` | boolean; `false` | Retain solver debug data. |
| `max_batch_size` | positive integer; `256` | Planner resource capacity. |
| `multi_env` | boolean; `false` | Distinct collision world per problem. |
| `max_goalset` | positive integer; `1` | Goal-set capacity per problem. |
| `self_collision_check` | boolean; `true` | cuRobo model self-collision check. |
| `collision_cache.cuboid`, `collision_cache.mesh` | integer >= 0; omitted/empty | Preallocated obstacle counts. |

## Logging Profile

The document root contains only `logging`. Omitted leaves use these parser
defaults; a bundled profile may explicitly select different values.

| Path | Type and omission default | Meaning |
| --- | --- | --- |
| `logging.enabled` | boolean; `true` | Whether to open/write the Single Scene joint CSV. |
| `logging.joint_tracking_path` | non-empty path string or `null`; `logs/joint_tracking/pinch_grasp.csv` | `null` disables the file target. Relative paths resolve from the repository root. |
| `logging.flush_interval_s` | positive finite number; `0.05` | Simulated-time flush cadence. |
| `logging.interval_steps` | positive integer; `1` | Physics-step sampling decimation. |
| `logging.log_actual_position` | boolean; `true` | Actual-position columns. |
| `logging.log_actual_velocity` | boolean; `true` | Actual-velocity columns. |
| `logging.log_command_position` | boolean; `true` | Position-command columns. |
| `logging.log_command_velocity` | boolean; `true` | Velocity-command columns. |
| `logging.log_command_effort` | boolean; `true` | Semantic effort-command columns. |
| `logging.log_action_effort` | boolean; `false` | Effort action sent to Isaac. |
| `logging.log_measured_effort` | boolean; `false` | PhysX measured effort; more expensive to read. |
| `logging.log_applied_effort` | boolean; `false` | PhysX applied effort; more expensive to read. |

## Complete Graph Validation

Validation follows the actual ownership graph:

```text
runtime
  -> env/base + per-env fragments
  -> robot profiles -> selected controller bundles
                    -> robot resources + selected cuRobo algorithm profile
  -> object profiles
  -> logging profile
```

It also checks runtime/env mode agreement, cross-instance USD subtree overlap,
controller bundle completeness, every object kind-specific consumer, and the
merged cuRobo configuration for each planning-enabled robot. Robots with
`curobo.enabled: false` skip backend materialization checks. Files under
`configs/curobo/task/` are task-bundle resources rather than standalone project
profiles.

Normal success output contains only the fixed `config_validated` event, runtime
profile name, and runtime fingerprint. The fingerprint covers the effective
runtime mapping, not downstream profile file contents; the complete graph is
still validated first. Use `--dump-effective-config` to inspect effective
runtime values and their provenance.
