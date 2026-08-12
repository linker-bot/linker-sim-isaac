# Source module map

Language: [English](module-map.md) | [中文](../../zh-CN/development/module-map.md)

This generated map covers every Python module under `src/linkerbot_sim`. The v2 architecture manifest owns facade, runtime, layer, and ordering facts; run `python scripts/update_architecture_inventory.py --write` after a source move.

- `pure`: no Kit/Isaac application is required.
- `Isaac main thread`: runtime work belongs to the simulation owner thread.
- `cuRobo/CUDA`: numerical work requires the configured CUDA stack.

## Layer direction

Dependencies point downward. Each product owns one runtime and one Isaac session; training consumes only the Kaleidoscope public port.

```text
product interface / training
            ↓
   Mirror | Kaleidoscope
            ↓
Isaac infrastructure | numerical backends
            ↓
configuration + pure domain

Controller/Env → Runtime → IsaacSession → concrete PhysicsRuntime
```

## Interface and owner registry

<!-- module-interface-registry:start -->
| Module | Classification | Runtime |
| --- | --- | --- |
| `linkerbot_sim` | documented facade | pure |
| `linkerbot_sim.configuration` | documented facade | pure |
| `linkerbot_sim.backends.curobo` | documented facade | pure |
| `linkerbot_sim.snapshots` | documented facade | pure |
| `linkerbot_sim.mirror` | documented facade | pure |
| `linkerbot_sim.kaleidoscope` | documented facade | pure |
| `linkerbot_sim.training.skrl` | documented facade | pure |
| `linkerbot_sim.configuration.catalog` | owner path | pure |
| `linkerbot_sim.configuration.modes.kaleidoscope` | owner path | pure |
| `linkerbot_sim.configuration.modes.mirror` | owner path | pure |
| `linkerbot_sim.isaac.physics.factory` | owner path | Isaac main thread |
| `linkerbot_sim.isaac.physics.runtime` | owner path | Isaac main thread |
| `linkerbot_sim.isaac.session` | owner path | Isaac main thread |
| `linkerbot_sim.mirror.controller` | owner path | Isaac main thread |
| `linkerbot_sim.mirror.rendering` | owner path | Isaac main thread |
| `linkerbot_sim.mirror.runtime` | owner path | Isaac main thread |
| `linkerbot_sim.kaleidoscope.env` | owner path | cuRobo/CUDA |
| `linkerbot_sim.kaleidoscope.runtime` | owner path | cuRobo/CUDA |
| `linkerbot_sim.kaleidoscope.state_api` | owner path | cuRobo/CUDA |
<!-- module-interface-registry:end -->

## Complete inventory

<!-- module-inventory:start -->

### root (1)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| root | `linkerbot_sim` | foundation | lightweight repository metadata facade | pure | documented facade | [Architecture reference](../reference/python-api.md) |

### configuration (23)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| configuration | `linkerbot_sim.configuration` | configuration | stable lazy configuration public facade | pure | documented facade | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.catalog` | configuration | sole project profile YAML I/O and composition owner | pure | owner path | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.common` | configuration | shared immutable configuration primitives | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.control` | configuration | configuration implementation owner for control | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.controllers` | configuration | configuration implementation owner for controllers | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.curobo` | configuration | configuration implementation owner for curobo | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.fingerprint` | configuration | configuration implementation owner for fingerprint | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes` | configuration | configuration implementation owner for modes | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes.common` | configuration | shared immutable configuration primitives | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes.kaleidoscope` | configuration | Kaleidoscope strict configuration root | pure | owner path | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes.mirror` | configuration | Mirror strict configuration root | pure | owner path | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.objects` | configuration | configuration implementation owner for objects | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.outputs` | configuration | Mirror output configuration contracts | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.physics` | configuration | configuration implementation owner for physics | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.planning` | configuration | configuration implementation owner for planning | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.robots` | configuration | configuration implementation owner for robots | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.scenes` | configuration | typed scene profile contracts | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.tasks` | configuration | registered task implementation namespace | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.tasks.kaleidoscope` | configuration | Kaleidoscope strict configuration root | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.training` | configuration | configuration implementation owner for training | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.training.skrl` | configuration | configuration implementation owner for skrl | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.visualization` | configuration | configuration implementation owner for visualization | pure | internal | [Architecture reference](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.visualization.kaleidoscope` | configuration | Kaleidoscope strict configuration root | pure | internal | [Architecture reference](../reference/configuration.md) |

### assets (8)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| assets | `linkerbot_sim.assets` | isaac_infrastructure | typed asset profile contracts | pure | internal | [Architecture reference](naming.md) |
| assets | `linkerbot_sim.assets.instance_paths` | isaac_infrastructure | assets implementation owner for instance paths | pure | internal | [Architecture reference](naming.md) |
| assets | `linkerbot_sim.assets.robot_config` | isaac_infrastructure | assets implementation owner for robot config | pure | internal | [Architecture reference](naming.md) |
| assets | `linkerbot_sim.assets.robot_import` | isaac_infrastructure | assets implementation owner for robot import | pure | internal | [Architecture reference](naming.md) |
| assets | `linkerbot_sim.assets.robot_instances` | isaac_infrastructure | assets implementation owner for robot instances | pure | internal | [Architecture reference](naming.md) |
| assets | `linkerbot_sim.assets.root_pose` | isaac_infrastructure | assets implementation owner for root pose | pure | internal | [Architecture reference](naming.md) |
| assets | `linkerbot_sim.assets.solver_overrides` | isaac_infrastructure | assets implementation owner for solver overrides | pure | internal | [Architecture reference](naming.md) |
| assets | `linkerbot_sim.assets.usd_overrides` | isaac_infrastructure | assets implementation owner for usd overrides | pure | internal | [Architecture reference](naming.md) |

### robots (10)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| robots | `linkerbot_sim.robots` | isaac_infrastructure | robots implementation namespace | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.capabilities` | isaac_infrastructure | robots implementation owner for capabilities | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.classification` | isaac_infrastructure | robots implementation owner for classification | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.joint_groups` | isaac_infrastructure | robots implementation owner for joint groups | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.mimic` | isaac_infrastructure | robots implementation owner for mimic | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.mimic.assets` | isaac_infrastructure | typed asset profile contracts | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.mimic.mjcf` | isaac_infrastructure | robots implementation owner for mjcf | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.mimic.runtime` | isaac_infrastructure | resource lifecycle and simulation-step orchestration | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.mimic.urdf` | isaac_infrastructure | robots implementation owner for urdf | pure | internal | [Architecture reference](naming.md) |
| robots | `linkerbot_sim.robots.tcp_binding` | isaac_infrastructure | robots implementation owner for tcp binding | pure | internal | [Architecture reference](naming.md) |

### objects (9)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| objects | `linkerbot_sim.objects` | isaac_infrastructure | objects implementation namespace | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.dynamic_chain` | isaac_infrastructure | objects implementation owner for dynamic chain | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.dynamic_chain.capsule_rope` | isaac_infrastructure | objects implementation owner for capsule rope | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.physics` | isaac_infrastructure | objects implementation owner for physics | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid` | isaac_infrastructure | objects implementation owner for rigid | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid.config` | isaac_infrastructure | objects implementation owner for config | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid.importer` | isaac_infrastructure | objects implementation owner for importer | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.runtime` | isaac_infrastructure | resource lifecycle and simulation-step orchestration | Isaac main thread | internal | [Architecture reference](object-assets.md) |
| objects | `linkerbot_sim.objects.state_views` | isaac_infrastructure | objects implementation owner for state views | Isaac main thread | internal | [Architecture reference](object-assets.md) |

### controllers (7)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| controllers | `linkerbot_sim.controllers` | isaac_infrastructure | controllers implementation namespace | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.control_mode` | isaac_infrastructure | controllers implementation owner for control mode | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.hybrid_force_position` | isaac_infrastructure | controllers implementation owner for hybrid force position | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.joint_controller` | isaac_infrastructure | controllers implementation owner for joint controller | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.projection` | isaac_infrastructure | controllers implementation owner for projection | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.runtime_projection` | isaac_infrastructure | controllers implementation owner for runtime projection | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.types` | isaac_infrastructure | controllers implementation owner for types | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |

### trajectories (4)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| trajectories | `linkerbot_sim.trajectories` | domain | trajectories implementation namespace | pure | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.joint_trajectory_builder` | domain | trajectories implementation owner for joint trajectory builder | pure | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.retiming` | domain | trajectories implementation owner for retiming | pure | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.types` | domain | trajectories implementation owner for types | pure | internal | [Architecture reference](../guides/control-and-trajectories.md) |

### planning (7)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| planning | `linkerbot_sim.planning` | domain | planning implementation namespace | pure | internal | [Architecture reference](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.backend` | domain | planning implementation owner for backend | pure | internal | [Architecture reference](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.collision_objects` | domain | planning implementation owner for collision objects | pure | internal | [Architecture reference](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.frames` | domain | planning implementation owner for frames | pure | internal | [Architecture reference](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.linear_backend` | domain | planning implementation owner for linear backend | pure | internal | [Architecture reference](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.requests` | domain | planning implementation owner for requests | pure | internal | [Architecture reference](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.results` | domain | planning implementation owner for results | pure | internal | [Architecture reference](../guides/motion-planning.md) |

### execution (4)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| execution | `linkerbot_sim.execution` | product | execution implementation namespace | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| execution | `linkerbot_sim.execution.runtime` | product | resource lifecycle and simulation-step orchestration | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| execution | `linkerbot_sim.execution.setup` | product | execution implementation owner for setup | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |
| execution | `linkerbot_sim.execution.steps` | product | execution implementation owner for steps | Isaac main thread | internal | [Architecture reference](../guides/control-and-trajectories.md) |

### backends (23)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| backends | `linkerbot_sim.backends` | numerical_backend | backends implementation namespace | pure | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo` | numerical_backend | stable lazy backends public facade | pure | documented facade | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.call_guard` | numerical_backend | backends implementation owner for call guard | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.collision_capability` | numerical_backend | backends implementation owner for collision capability | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.collision_world` | numerical_backend | backends implementation owner for collision world | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.config` | numerical_backend | backends implementation owner for config | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.context` | numerical_backend | backends implementation owner for context | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.forward_kinematics` | numerical_backend | backends implementation owner for forward kinematics | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.inverse_kinematics` | numerical_backend | backends implementation owner for inverse kinematics | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics` | numerical_backend | backends implementation owner for kinematics | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics.context` | numerical_backend | backends implementation owner for context | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics.device_batch_ik` | numerical_backend | backends implementation owner for device batch ik | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics.types` | numerical_backend | backends implementation owner for types | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.linear_pose_path` | numerical_backend | backends implementation owner for linear pose path | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.motion_planner` | numerical_backend | backends implementation owner for motion planner | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.profile_merge` | numerical_backend | backends implementation owner for profile merge | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.resources` | numerical_backend | backends implementation owner for resources | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.robot_model` | numerical_backend | backends implementation owner for robot model | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.runtime_imports` | numerical_backend | backends implementation owner for runtime imports | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.tensor_adapter` | numerical_backend | backends implementation owner for tensor adapter | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.tool_pose` | numerical_backend | backends implementation owner for tool pose | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.trajectory_adapter` | numerical_backend | backends implementation owner for trajectory adapter | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.warp_compat` | numerical_backend | backends implementation owner for warp compat | cuRobo/CUDA | internal | [Architecture reference](../guides/motion-planning.md) |

### isaac (35)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| isaac | `linkerbot_sim.isaac` | isaac_infrastructure | isaac implementation namespace | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.app` | isaac_infrastructure | process lifecycle and service composition | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.extensions` | isaac_infrastructure | Isaac extension enablement and exclusivity audit | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.gpu_memory_audit` | isaac_infrastructure | isaac implementation owner for gpu memory audit | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.lifecycle` | isaac_infrastructure | isaac implementation owner for lifecycle | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics` | isaac_infrastructure | isaac implementation owner for physics | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.backend` | isaac_infrastructure | isaac implementation owner for backend | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.core_api` | isaac_infrastructure | isaac implementation owner for core api | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.exclusivity` | isaac_infrastructure | isaac implementation owner for exclusivity | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.factory` | isaac_infrastructure | validated concrete runtime construction | Isaac main thread | owner path | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.manager` | isaac_infrastructure | physics owner registry and lifecycle coordination | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton` | isaac_infrastructure | isaac implementation owner for newton | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.constraints` | isaac_infrastructure | isaac implementation owner for constraints | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.integration_state` | isaac_infrastructure | isaac implementation owner for integration state | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.manager` | isaac_infrastructure | physics owner registry and lifecycle coordination | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.render` | isaac_infrastructure | isaac implementation owner for render | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.replication` | isaac_infrastructure | isaac implementation owner for replication | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.views` | isaac_infrastructure | isaac implementation owner for views | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.physx` | isaac_infrastructure | PhysX runtime ownership adapter | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.physx_pipeline` | isaac_infrastructure | isaac implementation owner for physx pipeline | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.physx_task_space` | isaac_infrastructure | isaac implementation owner for physx task space | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.runtime` | isaac_infrastructure | resource lifecycle and simulation-step orchestration | Isaac main thread | owner path | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.provenance` | isaac_infrastructure | isaac implementation owner for provenance | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene` | isaac_infrastructure | isaac implementation owner for replicated scene | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.assets` | isaac_infrastructure | typed asset profile contracts | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.layout` | isaac_infrastructure | isaac implementation owner for layout | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.newton_builder` | isaac_infrastructure | isaac implementation owner for newton builder | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.physx_builder` | isaac_infrastructure | isaac implementation owner for physx builder | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.types` | isaac_infrastructure | isaac implementation owner for types | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.views` | isaac_infrastructure | isaac implementation owner for views | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.scene` | isaac_infrastructure | isaac implementation owner for scene | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.scene.pose` | isaac_infrastructure | isaac implementation owner for pose | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.session` | isaac_infrastructure | SimulationApp, stage, and physics runtime owner | Isaac main thread | owner path | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.spec` | isaac_infrastructure | isaac implementation owner for spec | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.world` | isaac_infrastructure | isaac implementation owner for world | Isaac main thread | internal | [Architecture reference](../operations/constraints.md) |

### sensors (10)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| sensors | `linkerbot_sim.sensors` | isaac_infrastructure | sensors implementation namespace | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera` | isaac_infrastructure | sensors implementation owner for camera | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.config` | isaac_infrastructure | sensors implementation owner for config | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.foxglove` | isaac_infrastructure | sensors implementation owner for foxglove | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.frame` | isaac_infrastructure | sensors implementation owner for frame | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.limits` | isaac_infrastructure | sensors implementation owner for limits | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.observer` | isaac_infrastructure | sensors implementation owner for observer | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.recorder` | isaac_infrastructure | sensors implementation owner for recorder | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.runtime` | isaac_infrastructure | resource lifecycle and simulation-step orchestration | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.config` | isaac_infrastructure | sensors implementation owner for config | Isaac main thread | internal | [Architecture reference](../guides/cameras.md) |

### snapshots (7)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| snapshots | `linkerbot_sim.snapshots` | domain | stable lazy snapshots public facade | pure | documented facade | [Architecture reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.compatibility` | domain | snapshots implementation owner for compatibility | pure | internal | [Architecture reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.mirror_adapter` | domain | snapshots implementation owner for mirror adapter | pure | internal | [Architecture reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.persistence` | domain | snapshots implementation owner for persistence | pure | internal | [Architecture reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.runtime_objects` | domain | snapshots implementation owner for runtime objects | pure | internal | [Architecture reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.schema` | domain | snapshots implementation owner for schema | pure | internal | [Architecture reference](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.transactions` | domain | snapshots implementation owner for transactions | pure | internal | [Architecture reference](../reference/snapshots.md) |

### telemetry (4)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| telemetry | `linkerbot_sim.telemetry` | outputs | telemetry implementation namespace | Isaac main thread | internal | [Architecture reference](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove` | outputs | telemetry implementation owner for foxglove | Isaac main thread | internal | [Architecture reference](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove_state` | outputs | telemetry implementation owner for foxglove state | Isaac main thread | internal | [Architecture reference](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.state_snapshot` | outputs | telemetry implementation owner for state snapshot | Isaac main thread | internal | [Architecture reference](../guides/telemetry.md) |

### logging (5)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| logging | `linkerbot_sim.logging` | outputs | logging implementation namespace | pure | internal | [Architecture reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.csv_writer` | outputs | logging implementation owner for csv writer | pure | internal | [Architecture reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.effort_logger` | outputs | logging implementation owner for effort logger | pure | internal | [Architecture reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.hybrid_control_logger` | outputs | logging implementation owner for hybrid control logger | pure | internal | [Architecture reference](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.joint_logger` | outputs | logging implementation owner for joint logger | pure | internal | [Architecture reference](../reference/outputs.md) |

### visualization (2)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| visualization | `linkerbot_sim.visualization` | isaac_infrastructure | visualization implementation namespace | pure | internal | [Architecture reference](../development/usd-preview.md) |
| visualization | `linkerbot_sim.visualization.viewport` | isaac_infrastructure | visualization implementation owner for viewport | pure | internal | [Architecture reference](../development/usd-preview.md) |

### mirror (41)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| mirror | `linkerbot_sim.mirror` | product | stable lazy mirror public facade | pure | documented facade | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.app` | product | process lifecycle and service composition | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.bootstrap` | product | composition root and resource ownership transfer | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.cli` | product | command-line parsing and process startup | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision` | product | mirror implementation owner for collision | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.envelope_provider` | product | mirror implementation owner for envelope provider | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.object_provider` | product | mirror implementation owner for object provider | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.owner` | product | mirror implementation owner for owner | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.registry` | product | mirror implementation owner for registry | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.robot_provider` | product | mirror implementation owner for robot provider | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.urdf_kinematics` | product | mirror implementation owner for urdf kinematics | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.control_mode` | product | mirror implementation owner for control mode | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.controller` | product | owner-thread command dispatch and safety controls | Isaac main thread | owner path | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.hybrid_parameters` | product | mirror implementation owner for hybrid parameters | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface` | product | product interface namespace | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.admission` | product | bounded request admission and response ownership | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.protocol` | product | strict versioned wire protocol | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.state_stream` | product | mirror implementation owner for state stream | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.transport` | product | stdin, TCP JSONL, and WebSocket ingress | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.lifecycle` | product | mirror implementation owner for lifecycle | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion` | product | mirror implementation owner for motion | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.backend` | product | mirror implementation owner for backend | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.hybrid_executor` | product | mirror implementation owner for hybrid executor | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.owner` | product | mirror implementation owner for owner | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.request_parser` | product | mirror implementation owner for request parser | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline` | product | mirror implementation owner for timeline | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.builders` | product | mirror implementation owner for builders | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.compiler` | product | mirror implementation owner for compiler | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.executor` | product | mirror implementation owner for executor | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.model` | product | mirror implementation owner for model | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.requests` | product | mirror implementation owner for requests | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.rendering` | product | render and camera resource coordination | Isaac main thread | owner path | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.reset` | product | transactional runtime reset orchestration | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.reset_runtime` | product | mirror implementation owner for reset runtime | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.robots` | product | mirror implementation owner for robots | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.runtime` | product | resource lifecycle and simulation-step orchestration | Isaac main thread | owner path | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.scene_assembly` | product | mirror implementation owner for scene assembly | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.scene_settings` | product | mirror implementation owner for scene settings | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.snapshot` | product | owned snapshot schema and restore semantics | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.state` | product | Mirror state access and mutation | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.timing` | product | mirror implementation owner for timing | Isaac main thread | internal | [Architecture reference](../getting-started/project-overview.md) |

### kaleidoscope (32)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| kaleidoscope | `linkerbot_sim.kaleidoscope` | product | stable lazy kaleidoscope public facade | pure | documented facade | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.actions` | product | fixed-shape action validation and application | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.adapters` | product | external API adapter namespace | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.adapters.gymnasium` | product | kaleidoscope implementation owner for gymnasium | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.bootstrap` | product | composition root and resource ownership transfer | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.checkpoint` | product | explicit cold persistent checkpoint boundary | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.control_commands` | product | kaleidoscope implementation owner for control commands | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.control_mode` | product | kaleidoscope implementation owner for control mode | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.control_runtime` | product | kaleidoscope implementation owner for control runtime | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.env` | product | training environment lifecycle | cuRobo/CUDA | owner path | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.geometry` | product | kaleidoscope implementation owner for geometry | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.ik` | product | device-native batched inverse kinematics | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.isaac_adapter` | product | Isaac runtime adapter boundary | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.isaac_views` | product | fixed-shape Isaac tensor views | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.linear_motion` | product | synchronous device-native linear motion | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.newton_ports` | product | kaleidoscope implementation owner for newton ports | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.observations` | product | device-native observation assembly | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.physx_ports` | product | PhysX CUDA tensor port contracts | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.registration` | product | explicit Gymnasium registration | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.resets` | product | batched reset and autoreset semantics | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.rewards` | product | device-native reward computation | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.runtime` | product | resource lifecycle and simulation-step orchestration | cuRobo/CUDA | owner path | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.scene_assembly` | product | kaleidoscope implementation owner for scene assembly | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.snapshot` | product | owned snapshot schema and restore semantics | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.state_api` | product | batched state, snapshot, and clone API | cuRobo/CUDA | owner path | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.task` | product | vector task contract and step semantics | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.task_buffers` | product | owned fixed-shape task state buffers | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.tasks` | product | registered task implementation namespace | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.tasks.tblock_push_v1` | product | kaleidoscope implementation owner for tblock push v1 | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.tensors` | product | CUDA tensor validation and allocation invariants | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.terminations` | product | device-native termination and truncation rules | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.training_port` | product | public CUDA training environment protocol | cuRobo/CUDA | internal | [Architecture reference](../getting-started/choose-runtime-and-api.md) |

### training (6)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| training | `linkerbot_sim.training` | training | training implementation namespace | pure | internal | [Architecture reference](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl` | training | stable lazy training public facade | pure | documented facade | [Architecture reference](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.env` | training | training environment lifecycle | cuRobo/CUDA | internal | [Architecture reference](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.factory` | training | validated concrete runtime construction | cuRobo/CUDA | internal | [Architecture reference](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.final_observation_ppo` | training | training implementation owner for final observation ppo | cuRobo/CUDA | internal | [Architecture reference](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.memory` | training | CUDA rollout memory integration | cuRobo/CUDA | internal | [Architecture reference](../operations/constraints.md) |

### utils (9)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| utils | `linkerbot_sim.utils` | foundation | utils implementation namespace | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.config` | foundation | utils implementation owner for config | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.json` | foundation | utils implementation owner for json | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.math_utils` | foundation | utils implementation owner for math utils | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.output_paths` | foundation | utils implementation owner for output paths | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.paths` | foundation | utils implementation owner for paths | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.rotations` | foundation | utils implementation owner for rotations | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.tensors` | foundation | CUDA tensor validation and allocation invariants | pure | internal | [Architecture reference](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.timing` | foundation | utils implementation owner for timing | pure | internal | [Architecture reference](../getting-started/project-overview.md) |

<!-- module-inventory:end -->
