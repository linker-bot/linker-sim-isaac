# Source Module Map

Language: [English](module-map.md) | [中文](../../zh-CN/development/module-map.md)

This page is the complete navigation map for `src/linkerbot_sim/**/*.py`. It is
for maintainers and advanced in-process users who need to find the source owner
of a behavior. Read the linked task or reference page for the observable
contract. This map does not repeat protocol fields, configuration fields, or
Python symbol signatures.
Supported import paths and exact symbols are owned by the
[Python Facade Reference](../reference/python-api.md).

The runtime label records the strongest requirement of the responsibility named
in a row:

- `pure`: no running Kit/Isaac application is required.
- `Isaac main thread`: call the runtime operation only after Isaac startup and on
  the simulation owner thread.
- `cuRobo/CUDA`: the numerical operation requires the project GPU, Torch, Warp,
  and cuRobo environment. Any associated stage access still belongs on the Isaac
  main thread.

When a backend-neutral module accepts an injected solver, its row labels the
module's own work; the responsibility text identifies branches that inherit the
injected solver's stronger runtime requirement.

The classification is deliberately narrower than source visibility:

- `documented facade`: the package is an intended import surface, but only the
  symbols explicitly listed by the Python reference are dependable.
- `owner path`: the canonical place to understand or maintain a fact. It is an
  advanced navigation path, not a stable API commitment.
- `internal`: composition or implementation detail; callers must enter through a
  documented interface instead.

`linkerbot_sim.tiled.__all__` is empty. Consequently, neither
`linkerbot_sim.tiled` nor any of its 32 descendants is a top-level public API.
The few tiled leaf modules classified as `owner path` below remain navigation
owners only.

## Interface And Owner Registry

This machine-readable registry contains every non-internal inventory entry. The
coverage test requires its classification and runtime label to agree with the
full inventory.

<!-- module-interface-registry:start -->
| Module | Classification | Runtime |
| --- | --- | --- |
| `linkerbot_sim` | documented facade | pure |
| `linkerbot_sim.app.interactive.single_scene` | documented facade | Isaac main thread |
| `linkerbot_sim.app.interactive.tiled_scene` | documented facade | Isaac main thread |
| `linkerbot_sim.app.launch` | owner path | Isaac main thread |
| `linkerbot_sim.app.runtime.collision` | owner path | Isaac main thread |
| `linkerbot_sim.app.runtime.single_scene_runtime` | owner path | Isaac main thread |
| `linkerbot_sim.assets.robot_config` | owner path | pure |
| `linkerbot_sim.assets.robot_instances` | owner path | pure |
| `linkerbot_sim.assets.root_pose` | owner path | Isaac main thread |
| `linkerbot_sim.backends.curobo` | documented facade | cuRobo/CUDA |
| `linkerbot_sim.configs.profiles` | owner path | pure |
| `linkerbot_sim.configs.runtime` | owner path | pure |
| `linkerbot_sim.configs.validator` | owner path | pure |
| `linkerbot_sim.controllers` | documented facade | Isaac main thread |
| `linkerbot_sim.controllers.config` | owner path | pure |
| `linkerbot_sim.envs.config` | owner path | pure |
| `linkerbot_sim.envs.settings` | owner path | pure |
| `linkerbot_sim.envs.visual_settings` | owner path | pure |
| `linkerbot_sim.execution` | documented facade | Isaac main thread |
| `linkerbot_sim.logging.config` | owner path | pure |
| `linkerbot_sim.objects` | documented facade | Isaac main thread |
| `linkerbot_sim.objects.dynamic_chain` | owner path | Isaac main thread |
| `linkerbot_sim.objects.rigid` | owner path | Isaac main thread |
| `linkerbot_sim.planning` | documented facade | pure |
| `linkerbot_sim.robots` | documented facade | pure |
| `linkerbot_sim.sensors` | documented facade | pure |
| `linkerbot_sim.sensors.camera` | owner path | pure |
| `linkerbot_sim.snapshots` | documented facade | Isaac main thread |
| `linkerbot_sim.telemetry.foxglove` | owner path | pure |
| `linkerbot_sim.telemetry.state_snapshot` | owner path | Isaac main thread |
| `linkerbot_sim.telemetry.tiled.config` | owner path | pure |
| `linkerbot_sim.tiled.config` | owner path | pure |
| `linkerbot_sim.tiled.control.types` | owner path | pure |
| `linkerbot_sim.tiled.planning.types` | owner path | pure |
| `linkerbot_sim.tiled.playback.models` | owner path | pure |
| `linkerbot_sim.tiled.scene.types` | owner path | pure |
| `linkerbot_sim.trajectories.joint_trajectory_builder` | owner path | pure |
| `linkerbot_sim.trajectories.retiming` | owner path | pure |
| `linkerbot_sim.trajectories.types` | owner path | pure |
<!-- module-interface-registry:end -->

## Complete Inventory

The inventory is grouped by the first package below `linkerbot_sim`; the package
root is the `root` group. Every row links to the detailed documentation owner
closest to that module's responsibility.

<!-- module-inventory:start -->

### root (1)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| root | `linkerbot_sim` | Repository-root locator and package facade | pure | documented facade | [Python facade reference](../reference/python-api.md) |

### app (50)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| app | `linkerbot_sim.app` | Application launch namespace | pure | internal | [Project overview](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.interactive` | Interactive runtime composition namespace | pure | internal | [Runtime and API chooser](../getting-started/choose-runtime-and-api.md) |
| app | `linkerbot_sim.app.interactive.policies` | Request limits and policy validation | pure | internal | [Runtime constraints](../operations/constraints.md) |
| app | `linkerbot_sim.app.interactive.single_scene.protocol` | Shared JSON response and error helpers | pure | internal | [Single Scene JSON reference](../reference/single-scene-json.md) |
| app | `linkerbot_sim.app.interactive.single_scene.queue` | Bounded motion request queue | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.single_scene` | Single Scene interactive Python entrypoint | Isaac main thread | documented facade | [Python facade reference](../reference/python-api.md) |
| app | `linkerbot_sim.app.interactive.single_scene.cli` | Single Scene CLI parsing and process startup | Isaac main thread | internal | [Single Scene CLI reference](../reference/single-scene-cli.md) |
| app | `linkerbot_sim.app.interactive.single_scene.runtime` | Single Scene request loop and runtime binding | Isaac main thread | internal | [Single Scene JSON reference](../reference/single-scene-json.md) |
| app | `linkerbot_sim.app.interactive.single_scene.state_stream` | Single Scene state sampling and publication | Isaac main thread | internal | [Telemetry guide](../guides/telemetry.md) |
| app | `linkerbot_sim.app.interactive.stdin_reader` | Bounded stdin JSONL reader | pure | internal | [Single Scene JSON reference](../reference/single-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene` | Tiled Scene interactive Python entrypoint | Isaac main thread | documented facade | [Python facade reference](../reference/python-api.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.action_messages` | Tiled Scene synchronous action parsing | pure | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.cli` | Tiled Scene CLI parsing and process startup | Isaac main thread | internal | [Tiled Scene CLI reference](../reference/tiled-scene-cli.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.command_utils` | Tiled Scene command validation helpers | pure | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.hand_messages` | Tiled Scene hand command routing | Isaac main thread | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.message_utils` | Tiled Scene response and request utilities | pure | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.plan_messages` | Tiled Scene asynchronous plan message routing | Isaac main thread | internal | [Motion planning](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.protocol` | Tiled Scene command dispatch | Isaac main thread | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime` | Tiled Scene runtime class export | Isaac main thread | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.core` | Tiled Scene runtime lifecycle owner | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.factory` | Tiled scene and service assembly | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.ik` | Batched IK service integration | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.planning` | Planner submission and result collection | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.state` | Selected-environment state and snapshot access | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.stepping` | Tiled Scene physics stepping and playback | Isaac main thread | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.selectors` | Environment and robot selector parsing | pure | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.telemetry_publish` | Tiled Scene telemetry publication adapter | Isaac main thread | internal | [Telemetry guide](../guides/telemetry.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.trajectory_messages` | Tiled Scene trajectory buffer message routing | Isaac main thread | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.transport` | Tiled Scene stdin, TCP, and WebSocket loops | pure | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.single_scene.transports` | Shared bounded network transports | pure | internal | [Runtime constraints](../operations/constraints.md) |
| app | `linkerbot_sim.app.launch` | SimulationApp settings and launch owner | Isaac main thread | owner path | [Configuration reference](../reference/configuration.md) |
| app | `linkerbot_sim.app.motion` | Application motion namespace | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline` | Single Scene timeline implementation namespace | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.builders` | Timeline tick and executable-track builders | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.compiler` | Atomic request-to-timeline compilation | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.motion.timeline.executor` | Main-thread timeline execution | Isaac main thread | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.model` | Immutable integer-tick timeline model | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.requests` | Backend-neutral timeline request models | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.runtime` | Single Scene runtime composition namespace | pure | internal | [Project overview](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.runtime.collision` | Planning-collision provider registry | Isaac main thread | owner path | [Collision models](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.envelope_provider` | Conservative robot envelope provider | pure | internal | [Collision models](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.object_provider` | Runtime-object collision conversion | pure | internal | [Collision models](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.registry` | Collision provider registry and fingerprints | pure | internal | [Collision models](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.robot_provider` | Robot-state collision sphere provider | Isaac main thread | internal | [Collision models](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.urdf_kinematics` | Lightweight URDF forward kinematics | pure | internal | [Collision models](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.single_scene_reset` | Existing-session reset orchestration | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| app | `linkerbot_sim.app.runtime.robot_registry` | Robot identity and planning-context registry | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.runtime.single_scene_runtime` | Single Scene lifecycle and resource owner | Isaac main thread | owner path | [Project overview](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.runtime.simulation_app_lifecycle` | SimulationApp shutdown coordination | Isaac main thread | internal | [Runtime constraints](../operations/constraints.md) |
| app | `linkerbot_sim.app.runtime.simulation_session` | SimulationApp, World, and scene session assembly | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |

### assets (7)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| assets | `linkerbot_sim.assets` | Robot asset implementation namespace | pure | internal | [Naming](naming.md) |
| assets | `linkerbot_sim.assets.robot_config` | Robot asset and physics configuration owner | pure | owner path | [Configuration reference](../reference/configuration.md) |
| assets | `linkerbot_sim.assets.robot_import` | Isaac robot asset import | Isaac main thread | internal | [Collision approximation](collision-approximation.md) |
| assets | `linkerbot_sim.assets.robot_instances` | Scene robot identity and execution settings | pure | owner path | [Naming](naming.md) |
| assets | `linkerbot_sim.assets.root_pose` | Root-pose model and USD authoring | Isaac main thread | owner path | [Naming](naming.md) |
| assets | `linkerbot_sim.assets.solver_overrides` | PhysX solver override application | Isaac main thread | internal | [Collision approximation](collision-approximation.md) |
| assets | `linkerbot_sim.assets.usd_overrides` | Imported USD and PhysX overrides | Isaac main thread | internal | [Collision approximation](collision-approximation.md) |

### backends (24)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| backends | `linkerbot_sim.backends` | Numerical backend namespace | pure | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo` | cuRobo backend facade | cuRobo/CUDA | documented facade | [Python facade reference](../reference/python-api.md) |
| backends | `linkerbot_sim.backends.curobo.batch` | Batch numerical-kernel namespace | pure | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.ik` | Batched cuRobo IK solver | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.joint_planner` | Batched joint-space planner | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.result_adapter` | Batch result conversion | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.types` | Batch solver data structures | pure | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.call_guard` | Serialized cuRobo call boundary | pure | internal | [Runtime constraints](../operations/constraints.md) |
| backends | `linkerbot_sim.backends.curobo.collision_capability` | Collision-capability selection | pure | internal | [Collision models](../guides/collision-models.md) |
| backends | `linkerbot_sim.backends.curobo.collision_world` | cuRobo collision-world construction | cuRobo/CUDA | internal | [Collision models](../guides/collision-models.md) |
| backends | `linkerbot_sim.backends.curobo.config` | cuRobo configuration models | pure | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.context` | Solver context and capability lifecycle | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.forward_kinematics` | cuRobo forward kinematics | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.inverse_kinematics` | cuRobo inverse kinematics | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.joint_mapping` | Project-to-cuRobo joint mapping | pure | internal | [Naming](naming.md) |
| backends | `linkerbot_sim.backends.curobo.linear_pose_path` | Cartesian line planning | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.motion_planner` | Motion-planner orchestration | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.profile_merge` | Robot and algorithm profile merge | pure | internal | [Configuration guide](../guides/configuration.md) |
| backends | `linkerbot_sim.backends.curobo.robot_model` | cuRobo robot-model materialization | pure | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.runtime_imports` | Lazy cuRobo dependency loading | cuRobo/CUDA | internal | [Runtime constraints](../operations/constraints.md) |
| backends | `linkerbot_sim.backends.curobo.tensor_adapter` | Array-to-device tensor conversion | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.tool_pose` | Tool-pose frame conversion | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.trajectory_adapter` | cuRobo trajectory conversion | cuRobo/CUDA | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| backends | `linkerbot_sim.backends.curobo.warp_compat` | Pinned Warp API adaptation | cuRobo/CUDA | internal | [Runtime constraints](../operations/constraints.md) |

### configs (6)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| configs | `linkerbot_sim.configs` | Project configuration namespace | pure | internal | [Configuration guide](../guides/configuration.md) |
| configs | `linkerbot_sim.configs.cli` | CLI overlay application | pure | internal | [Configuration guide](../guides/configuration.md) |
| configs | `linkerbot_sim.configs.instance_paths` | Instance prim-path validation | pure | internal | [Naming](naming.md) |
| configs | `linkerbot_sim.configs.profiles` | Strict profile loading owner | pure | owner path | [Configuration reference](../reference/configuration.md) |
| configs | `linkerbot_sim.configs.runtime` | Runtime profile model and composition | pure | owner path | [Configuration reference](../reference/configuration.md) |
| configs | `linkerbot_sim.configs.validator` | Complete profile-graph validation | pure | owner path | [Configuration reference](../reference/configuration.md) |

### controllers (4)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| controllers | `linkerbot_sim.controllers` | Joint-control facade | Isaac main thread | documented facade | [Python facade reference](../reference/python-api.md) |
| controllers | `linkerbot_sim.controllers.config` | Controller profile parsing owner | pure | owner path | [Configuration reference](../reference/configuration.md) |
| controllers | `linkerbot_sim.controllers.joint_controller` | Articulation target and mode control | Isaac main thread | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.types` | Control settings and target models | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |

### envs (5)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| envs | `linkerbot_sim.envs` | Environment implementation namespace | pure | internal | [Configuration guide](../guides/configuration.md) |
| envs | `linkerbot_sim.envs.config` | Env profile and fragment validation | pure | owner path | [Configuration reference](../reference/configuration.md) |
| envs | `linkerbot_sim.envs.scene_builder` | Base Isaac world construction | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| envs | `linkerbot_sim.envs.settings` | Environment runtime settings owner | pure | owner path | [Configuration reference](../reference/configuration.md) |
| envs | `linkerbot_sim.envs.visual_settings` | Scene visual settings owner | pure | owner path | [Configuration reference](../reference/configuration.md) |

### execution (4)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| execution | `linkerbot_sim.execution` | Simulation execution facade | Isaac main thread | documented facade | [Python facade reference](../reference/python-api.md) |
| execution | `linkerbot_sim.execution.runtime` | Execution context and step protocol | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| execution | `linkerbot_sim.execution.setup` | Execution-side robot assembly | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| execution | `linkerbot_sim.execution.steps` | Reusable control execution steps | Isaac main thread | internal | [Control and trajectories](../guides/control-and-trajectories.md) |

### logging (5)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| logging | `linkerbot_sim.logging` | CSV logging implementation namespace | pure | internal | [Output reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.config` | Single Scene logging profile owner | pure | owner path | [Output reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.csv_writer` | Bounded joint CSV writer | pure | internal | [Output reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.effort_logger` | Articulation effort sampling | Isaac main thread | internal | [Output reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.joint_logger` | Joint target and state sampling | Isaac main thread | internal | [Output reference](../reference/outputs.md) |

### objects (10)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| objects | `linkerbot_sim.objects` | Scene-object facade | Isaac main thread | documented facade | [Python facade reference](../reference/python-api.md) |
| objects | `linkerbot_sim.objects.config` | Object profile and instance parsing | pure | internal | [Configuration reference](../reference/configuration.md) |
| objects | `linkerbot_sim.objects.dynamic_chain` | Dynamic-chain object owner | Isaac main thread | owner path | [Object assets](object-assets.md) |
| objects | `linkerbot_sim.objects.dynamic_chain.capsule_rope` | Capsule-rope reference and physics setup | Isaac main thread | internal | [Object assets](object-assets.md) |
| objects | `linkerbot_sim.objects.physics` | Object material and root-pose authoring | Isaac main thread | internal | [Collision models](../guides/collision-models.md) |
| objects | `linkerbot_sim.objects.rigid` | Rigid-object owner | Isaac main thread | owner path | [Object assets](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid.config` | Rigid-object and planning-collider models | pure | internal | [Collision models](../guides/collision-models.md) |
| objects | `linkerbot_sim.objects.rigid.importer` | Rigid USD import and physics setup | Isaac main thread | internal | [Object assets](object-assets.md) |
| objects | `linkerbot_sim.objects.runtime` | Runtime object assembly and handles | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| objects | `linkerbot_sim.objects.state_views` | Scene object state read and restore | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |

### planning (8)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| planning | `linkerbot_sim.planning` | Backend-neutral planning facade | pure | documented facade | [Python facade reference](../reference/python-api.md) |
| planning | `linkerbot_sim.planning.backend` | Planner protocol and backend selection | pure | internal | [Motion planning](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.batch_ik` | Batched IK protocol and result model | pure | internal | [Motion planning](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.collision_objects` | Planning collision-object model | pure | internal | [Collision models](../guides/collision-models.md) |
| planning | `linkerbot_sim.planning.frames` | Planning frame transformations | pure | internal | [Motion planning](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.linear_backend` | Backend-neutral linear planner | pure | internal | [Motion planning](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.requests` | IK and motion request models | pure | internal | [Motion planning](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.results` | IK and motion result models | pure | internal | [Motion planning](../guides/motion-planning.md) |

### robots (9)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| robots | `linkerbot_sim.robots` | Robot capability and joint-group facade | pure | documented facade | [Python facade reference](../reference/python-api.md) |
| robots | `linkerbot_sim.robots.capabilities` | Robot planning-capability model | pure | internal | [Motion planning](../guides/motion-planning.md) |
| robots | `linkerbot_sim.robots.classification` | Robot component classification | pure | internal | [Naming](naming.md) |
| robots | `linkerbot_sim.robots.joint_groups` | Named joint-group layout and resolution | pure | internal | [Naming](naming.md) |
| robots | `linkerbot_sim.robots.mimic` | Mimic relation implementation namespace | pure | internal | [Naming](naming.md) |
| robots | `linkerbot_sim.robots.mimic.assets` | Asset mimic-relation parsing | pure | internal | [Naming](naming.md) |
| robots | `linkerbot_sim.robots.mimic.mjcf` | MJCF equality and friction parsing | pure | internal | [Naming](naming.md) |
| robots | `linkerbot_sim.robots.mimic.runtime` | Mimic follower target expansion | Isaac main thread | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| robots | `linkerbot_sim.robots.mimic.urdf` | URDF mimic relation parsing | pure | internal | [Naming](naming.md) |

### sensors (10)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| sensors | `linkerbot_sim.sensors` | Scene sensor-settings facade | pure | documented facade | [Python facade reference](../reference/python-api.md) |
| sensors | `linkerbot_sim.sensors.camera` | Camera configuration owner path | pure | owner path | [Camera guide](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.config` | Camera settings parsing | pure | internal | [Camera guide](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.foxglove` | Camera frame Foxglove encoding | pure | internal | [Output reference](../reference/outputs.md) |
| sensors | `linkerbot_sim.sensors.camera.frame` | Immutable camera frame model | pure | internal | [Camera guide](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.limits` | Camera output resource limits | pure | internal | [Runtime constraints](../operations/constraints.md) |
| sensors | `linkerbot_sim.sensors.camera.observer` | World-step camera observation | Isaac main thread | internal | [Camera guide](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.recorder` | File recording and bounded publication | pure | internal | [Output reference](../reference/outputs.md) |
| sensors | `linkerbot_sim.sensors.camera.runtime` | Isaac camera creation and sampling | Isaac main thread | internal | [Camera guide](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.config` | Scene sensor aggregate settings | pure | internal | [Configuration reference](../reference/configuration.md) |

### snapshots (9)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| snapshots | `linkerbot_sim.snapshots` | Capture and restore facade | Isaac main thread | documented facade | [Python facade reference](../reference/python-api.md) |
| snapshots | `linkerbot_sim.snapshots.compatibility` | Snapshot-to-target identity matching | pure | internal | [Snapshot reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.debug_tiled_scene_adapter` | In-memory tiled snapshot adapter | pure | internal | [Snapshot reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.dispatch` | Runtime-shape snapshot dispatch | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.runtime_objects` | Shared runtime object snapshot helpers | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.single_scene_adapter` | Single Scene capture and transactional restore | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.schema` | Runtime-neutral snapshot data model | pure | internal | [Snapshot reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.tiled_scene_adapter` | Tiled Scene capture, restore, and env cloning | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.transactions` | Restore transaction and fail-stop state | pure | internal | [Snapshot reference](../reference/snapshots.md) |

### telemetry (8)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| telemetry | `linkerbot_sim.telemetry` | Telemetry implementation namespace | pure | internal | [Telemetry guide](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove` | Foxglove live and MCAP sink owner | pure | owner path | [Foxglove guide](../guides/foxglove.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove_state` | State-to-Foxglove serialization | pure | internal | [Telemetry guide](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.state_snapshot` | Single Scene state sampling and handoff | Isaac main thread | owner path | [Telemetry guide](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled` | Tiled Scene telemetry implementation namespace | pure | internal | [Telemetry guide](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled.config` | Tiled Scene telemetry settings owner | pure | owner path | [Telemetry guide](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled.payloads` | Tiled Scene state payload conversion | pure | internal | [Telemetry guide](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled.sink` | Tiled Scene live and MCAP sink lifecycle | pure | internal | [Output reference](../reference/outputs.md) |

### tiled (33)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| tiled | `linkerbot_sim.tiled` | Tiled implementation namespace with no exports | pure | internal | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| tiled | `linkerbot_sim.tiled.config` | Tiled env configuration owner | pure | owner path | [Configuration reference](../reference/configuration.md) |
| tiled | `linkerbot_sim.tiled.control` | Synchronous tiled control namespace | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.control.adapter` | Backend-neutral batched target conversion; EE paths call the injected IK solver | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.control.interpolation` | Fixed-tick joint interpolation | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.control.types` | Tiled Scene action data and value ranges | pure | owner path | [Tiled Scene JSON reference](../reference/tiled-scene-json.md) |
| tiled | `linkerbot_sim.tiled.planning` | Tiled Scene planning implementation namespace | pure | internal | [Motion planning](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.backends` | Tiled Scene planner backend composition | pure | internal | [Motion planning](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.backends.curobo` | Tiled-to-cuRobo planning adapter | cuRobo/CUDA | internal | [Motion planning](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.batching` | Homogeneous request batch layout | pure | internal | [Motion planning](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.linear_backend` | Linear planner batch adaptation | pure | internal | [Motion planning](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.manager` | Planner queue, workers, and cancellation | pure | internal | [Motion planning](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.types` | Tiled Scene planning request and result owner | pure | owner path | [Motion planning](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.playback` | Per-env playback implementation namespace | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.playback.buffer` | Per-env trajectory playback buffer | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.playback.models` | Playback track and cursor models | pure | owner path | [Control and trajectories](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.playback.staging` | Before, main, and after track staging | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.scene` | Tiled Isaac scene namespace | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.builder` | Tiled scene build orchestration | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.cameras` | Per-env camera configuration expansion | pure | internal | [Camera guide](../guides/cameras.md) |
| tiled | `linkerbot_sim.tiled.scene.clone` | Grid cloning and replication setup | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.collision_filter` | Cross-environment collision filtering | Isaac main thread | internal | [Collision models](../guides/collision-models.md) |
| tiled | `linkerbot_sim.tiled.scene.objects` | Per-env object import and overrides | Isaac main thread | internal | [Object assets](object-assets.md) |
| tiled | `linkerbot_sim.tiled.scene.paths` | Env-root paths and grid origins | pure | internal | [Naming](naming.md) |
| tiled | `linkerbot_sim.tiled.scene.robots` | Per-env robot import and identity binding | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.root_pose` | Cloned robot root-pose overrides | Isaac main thread | internal | [Configuration reference](../reference/configuration.md) |
| tiled | `linkerbot_sim.tiled.scene.types` | Immutable tiled scene assembly models | pure | owner path | [Project overview](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.utils` | Lightweight scene build helpers | pure | internal | [Project overview](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.views` | Batched articulation view binding | Isaac main thread | internal | [Project overview](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.state` | Batched state implementation namespace | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| tiled | `linkerbot_sim.tiled.state.object_io` | Selected-env object state orchestration | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| tiled | `linkerbot_sim.tiled.state.object_views` | Batched PhysX object state views | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |
| tiled | `linkerbot_sim.tiled.state.usd_pose` | USD pose read, write, and velocity reset | Isaac main thread | internal | [Snapshot reference](../reference/snapshots.md) |

### trajectories (4)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| trajectories | `linkerbot_sim.trajectories` | Trajectory implementation namespace | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.joint_trajectory_builder` | Joint trajectory construction owner | pure | owner path | [Control and trajectories](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.retiming` | Time-grid and path-progress resampling | pure | owner path | [Control and trajectories](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.types` | Joint trajectory data-model owner | pure | owner path | [Control and trajectories](../guides/control-and-trajectories.md) |

### utils (8)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| utils | `linkerbot_sim.utils` | Side-effect-free utility namespace | pure | internal | [Project overview](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.config` | Strict YAML and mapping helpers | pure | internal | [Configuration reference](../reference/configuration.md) |
| utils | `linkerbot_sim.utils.json` | Strict JSON encoding and decoding | pure | internal | [Runtime constraints](../operations/constraints.md) |
| utils | `linkerbot_sim.utils.math_utils` | Shared numeric transformations | pure | internal | [Motion planning](../guides/motion-planning.md) |
| utils | `linkerbot_sim.utils.output_paths` | Output path planning and application | pure | internal | [Output reference](../reference/outputs.md) |
| utils | `linkerbot_sim.utils.paths` | Repository path resolution | pure | internal | [Project overview](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.rotations` | RPY, quaternion, and matrix conversion | pure | internal | [Motion planning](../guides/motion-planning.md) |
| utils | `linkerbot_sim.utils.timing` | Sample differentiation helpers | pure | internal | [Control and trajectories](../guides/control-and-trajectories.md) |

### visualization (2)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| visualization | `linkerbot_sim.visualization` | Local viewport helper namespace | pure | internal | [USD preview](usd-preview.md) |
| visualization | `linkerbot_sim.visualization.viewport` | GUI viewport camera placement | Isaac main thread | internal | [USD preview](usd-preview.md) |

<!-- module-inventory:end -->
