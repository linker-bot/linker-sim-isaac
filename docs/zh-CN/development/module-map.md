# 源码模块图

语言：[中文](module-map.md) | [English](../../en/development/module-map.md)

本文由目标架构清单生成，完整覆盖 `src/linkerbot_sim` 下每个 Python 模块。facade、运行前提、分层和顺序由 v2 manifest 负责；移动源码后运行 `python scripts/update_architecture_inventory.py --write` 同步。

- `pure`：不需要启动 Kit/Isaac application。
- `Isaac main thread`：runtime 操作属于仿真 owner 线程。
- `cuRobo/CUDA`：数值操作需要配置指定的 CUDA stack。

## 分层方向

依赖只允许向下。每个产品拥有一个 runtime 和一个 Isaac session；训练层只消费 Kaleidoscope public port。

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

## 接口与 owner 登记表

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

## 完整 Inventory

<!-- module-inventory:start -->

### root (1)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| root | `linkerbot_sim` | foundation | 轻量仓库 metadata facade | pure | documented facade | [架构参考](../reference/python-api.md) |

### configuration (23)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| configuration | `linkerbot_sim.configuration` | configuration | 稳定 lazy configuration public facade | pure | documented facade | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.catalog` | configuration | 项目 profile YAML I/O 与组合的唯一 owner | pure | owner path | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.common` | configuration | 共享不可变配置原语 | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.control` | configuration | configuration 层 control 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.controllers` | configuration | configuration 层 controllers 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.curobo` | configuration | configuration 层 curobo 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.fingerprint` | configuration | configuration 层 fingerprint 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes` | configuration | configuration 层 modes 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes.common` | configuration | 共享不可变配置原语 | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes.kaleidoscope` | configuration | Kaleidoscope 严格配置根 | pure | owner path | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.modes.mirror` | configuration | Mirror 严格配置根 | pure | owner path | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.objects` | configuration | configuration 层 objects 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.outputs` | configuration | Mirror 输出配置合同 | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.physics` | configuration | configuration 层 physics 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.planning` | configuration | configuration 层 planning 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.robots` | configuration | configuration 层 robots 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.scenes` | configuration | 强类型 scene profile 合同 | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.tasks` | configuration | 已注册 task 实现命名空间 | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.tasks.kaleidoscope` | configuration | Kaleidoscope 严格配置根 | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.training` | configuration | configuration 层 training 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.training.skrl` | configuration | configuration 层 skrl 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.visualization` | configuration | configuration 层 visualization 实现 owner | pure | internal | [架构参考](../reference/configuration.md) |
| configuration | `linkerbot_sim.configuration.visualization.kaleidoscope` | configuration | Kaleidoscope 严格配置根 | pure | internal | [架构参考](../reference/configuration.md) |

### assets (8)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| assets | `linkerbot_sim.assets` | isaac_infrastructure | 强类型资产 profile 合同 | pure | internal | [架构参考](naming.md) |
| assets | `linkerbot_sim.assets.instance_paths` | isaac_infrastructure | assets 层 instance paths 实现 owner | pure | internal | [架构参考](naming.md) |
| assets | `linkerbot_sim.assets.robot_config` | isaac_infrastructure | assets 层 robot config 实现 owner | pure | internal | [架构参考](naming.md) |
| assets | `linkerbot_sim.assets.robot_import` | isaac_infrastructure | assets 层 robot import 实现 owner | pure | internal | [架构参考](naming.md) |
| assets | `linkerbot_sim.assets.robot_instances` | isaac_infrastructure | assets 层 robot instances 实现 owner | pure | internal | [架构参考](naming.md) |
| assets | `linkerbot_sim.assets.root_pose` | isaac_infrastructure | assets 层 root pose 实现 owner | pure | internal | [架构参考](naming.md) |
| assets | `linkerbot_sim.assets.solver_overrides` | isaac_infrastructure | assets 层 solver overrides 实现 owner | pure | internal | [架构参考](naming.md) |
| assets | `linkerbot_sim.assets.usd_overrides` | isaac_infrastructure | assets 层 usd overrides 实现 owner | pure | internal | [架构参考](naming.md) |

### robots (10)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| robots | `linkerbot_sim.robots` | isaac_infrastructure | robots 实现命名空间 | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.capabilities` | isaac_infrastructure | robots 层 capabilities 实现 owner | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.classification` | isaac_infrastructure | robots 层 classification 实现 owner | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.joint_groups` | isaac_infrastructure | robots 层 joint groups 实现 owner | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.mimic` | isaac_infrastructure | robots 层 mimic 实现 owner | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.mimic.assets` | isaac_infrastructure | 强类型资产 profile 合同 | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.mimic.mjcf` | isaac_infrastructure | robots 层 mjcf 实现 owner | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.mimic.runtime` | isaac_infrastructure | 资源生命周期与仿真步进编排 | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.mimic.urdf` | isaac_infrastructure | robots 层 urdf 实现 owner | pure | internal | [架构参考](naming.md) |
| robots | `linkerbot_sim.robots.tcp_binding` | isaac_infrastructure | robots 层 tcp binding 实现 owner | pure | internal | [架构参考](naming.md) |

### objects (9)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| objects | `linkerbot_sim.objects` | isaac_infrastructure | objects 实现命名空间 | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.dynamic_chain` | isaac_infrastructure | objects 层 dynamic chain 实现 owner | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.dynamic_chain.capsule_rope` | isaac_infrastructure | objects 层 capsule rope 实现 owner | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.physics` | isaac_infrastructure | objects 层 physics 实现 owner | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid` | isaac_infrastructure | objects 层 rigid 实现 owner | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid.config` | isaac_infrastructure | objects 层 config 实现 owner | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid.importer` | isaac_infrastructure | objects 层 importer 实现 owner | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.runtime` | isaac_infrastructure | 资源生命周期与仿真步进编排 | Isaac main thread | internal | [架构参考](object-assets.md) |
| objects | `linkerbot_sim.objects.state_views` | isaac_infrastructure | objects 层 state views 实现 owner | Isaac main thread | internal | [架构参考](object-assets.md) |

### controllers (7)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| controllers | `linkerbot_sim.controllers` | isaac_infrastructure | controllers 实现命名空间 | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.control_mode` | isaac_infrastructure | controllers 层 control mode 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.hybrid_force_position` | isaac_infrastructure | controllers 层 hybrid force position 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.joint_controller` | isaac_infrastructure | controllers 层 joint controller 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.projection` | isaac_infrastructure | controllers 层 projection 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.runtime_projection` | isaac_infrastructure | controllers 层 runtime projection 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.types` | isaac_infrastructure | controllers 层 types 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |

### trajectories (4)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| trajectories | `linkerbot_sim.trajectories` | domain | trajectories 实现命名空间 | pure | internal | [架构参考](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.joint_trajectory_builder` | domain | trajectories 层 joint trajectory builder 实现 owner | pure | internal | [架构参考](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.retiming` | domain | trajectories 层 retiming 实现 owner | pure | internal | [架构参考](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.types` | domain | trajectories 层 types 实现 owner | pure | internal | [架构参考](../guides/control-and-trajectories.md) |

### planning (7)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| planning | `linkerbot_sim.planning` | domain | planning 实现命名空间 | pure | internal | [架构参考](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.backend` | domain | planning 层 backend 实现 owner | pure | internal | [架构参考](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.collision_objects` | domain | planning 层 collision objects 实现 owner | pure | internal | [架构参考](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.frames` | domain | planning 层 frames 实现 owner | pure | internal | [架构参考](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.linear_backend` | domain | planning 层 linear backend 实现 owner | pure | internal | [架构参考](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.requests` | domain | planning 层 requests 实现 owner | pure | internal | [架构参考](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.results` | domain | planning 层 results 实现 owner | pure | internal | [架构参考](../guides/motion-planning.md) |

### execution (4)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| execution | `linkerbot_sim.execution` | product | execution 实现命名空间 | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| execution | `linkerbot_sim.execution.runtime` | product | 资源生命周期与仿真步进编排 | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| execution | `linkerbot_sim.execution.setup` | product | execution 层 setup 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |
| execution | `linkerbot_sim.execution.steps` | product | execution 层 steps 实现 owner | Isaac main thread | internal | [架构参考](../guides/control-and-trajectories.md) |

### backends (23)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| backends | `linkerbot_sim.backends` | numerical_backend | backends 实现命名空间 | pure | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo` | numerical_backend | 稳定 lazy backends public facade | pure | documented facade | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.call_guard` | numerical_backend | backends 层 call guard 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.collision_capability` | numerical_backend | backends 层 collision capability 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.collision_world` | numerical_backend | backends 层 collision world 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.config` | numerical_backend | backends 层 config 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.context` | numerical_backend | backends 层 context 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.forward_kinematics` | numerical_backend | backends 层 forward kinematics 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.inverse_kinematics` | numerical_backend | backends 层 inverse kinematics 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics` | numerical_backend | backends 层 kinematics 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics.context` | numerical_backend | backends 层 context 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics.device_batch_ik` | numerical_backend | backends 层 device batch ik 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.kinematics.types` | numerical_backend | backends 层 types 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.linear_pose_path` | numerical_backend | backends 层 linear pose path 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.motion_planner` | numerical_backend | backends 层 motion planner 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.profile_merge` | numerical_backend | backends 层 profile merge 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.resources` | numerical_backend | backends 层 resources 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.robot_model` | numerical_backend | backends 层 robot model 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.runtime_imports` | numerical_backend | backends 层 runtime imports 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.tensor_adapter` | numerical_backend | backends 层 tensor adapter 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.tool_pose` | numerical_backend | backends 层 tool pose 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.trajectory_adapter` | numerical_backend | backends 层 trajectory adapter 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.warp_compat` | numerical_backend | backends 层 warp compat 实现 owner | cuRobo/CUDA | internal | [架构参考](../guides/motion-planning.md) |

### isaac (35)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| isaac | `linkerbot_sim.isaac` | isaac_infrastructure | isaac 实现命名空间 | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.app` | isaac_infrastructure | 进程生命周期与服务组合 | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.extensions` | isaac_infrastructure | Isaac extension 启用与排他审计 | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.gpu_memory_audit` | isaac_infrastructure | isaac 层 gpu memory audit 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.lifecycle` | isaac_infrastructure | isaac 层 lifecycle 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics` | isaac_infrastructure | isaac 层 physics 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.backend` | isaac_infrastructure | isaac 层 backend 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.core_api` | isaac_infrastructure | isaac 层 core api 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.exclusivity` | isaac_infrastructure | isaac 层 exclusivity 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.factory` | isaac_infrastructure | 验收后的具体 runtime 构造 | Isaac main thread | owner path | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.manager` | isaac_infrastructure | 物理 owner 注册与生命周期协调 | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton` | isaac_infrastructure | isaac 层 newton 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.constraints` | isaac_infrastructure | isaac 层 constraints 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.integration_state` | isaac_infrastructure | isaac 层 integration state 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.manager` | isaac_infrastructure | 物理 owner 注册与生命周期协调 | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.render` | isaac_infrastructure | isaac 层 render 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.replication` | isaac_infrastructure | isaac 层 replication 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.newton.views` | isaac_infrastructure | isaac 层 views 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.physx` | isaac_infrastructure | PhysX runtime 所有权 adapter | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.physx_pipeline` | isaac_infrastructure | isaac 层 physx pipeline 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.physx_task_space` | isaac_infrastructure | isaac 层 physx task space 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.physics.runtime` | isaac_infrastructure | 资源生命周期与仿真步进编排 | Isaac main thread | owner path | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.provenance` | isaac_infrastructure | isaac 层 provenance 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene` | isaac_infrastructure | isaac 层 replicated scene 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.assets` | isaac_infrastructure | 强类型资产 profile 合同 | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.layout` | isaac_infrastructure | isaac 层 layout 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.newton_builder` | isaac_infrastructure | isaac 层 newton builder 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.physx_builder` | isaac_infrastructure | isaac 层 physx builder 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.types` | isaac_infrastructure | isaac 层 types 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.replicated_scene.views` | isaac_infrastructure | isaac 层 views 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.scene` | isaac_infrastructure | isaac 层 scene 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.scene.pose` | isaac_infrastructure | isaac 层 pose 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.session` | isaac_infrastructure | SimulationApp、stage 与物理 runtime owner | Isaac main thread | owner path | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.spec` | isaac_infrastructure | isaac 层 spec 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |
| isaac | `linkerbot_sim.isaac.world` | isaac_infrastructure | isaac 层 world 实现 owner | Isaac main thread | internal | [架构参考](../operations/constraints.md) |

### sensors (10)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| sensors | `linkerbot_sim.sensors` | isaac_infrastructure | sensors 实现命名空间 | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera` | isaac_infrastructure | sensors 层 camera 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.config` | isaac_infrastructure | sensors 层 config 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.foxglove` | isaac_infrastructure | sensors 层 foxglove 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.frame` | isaac_infrastructure | sensors 层 frame 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.limits` | isaac_infrastructure | sensors 层 limits 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.observer` | isaac_infrastructure | sensors 层 observer 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.recorder` | isaac_infrastructure | sensors 层 recorder 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.runtime` | isaac_infrastructure | 资源生命周期与仿真步进编排 | Isaac main thread | internal | [架构参考](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.config` | isaac_infrastructure | sensors 层 config 实现 owner | Isaac main thread | internal | [架构参考](../guides/cameras.md) |

### snapshots (7)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| snapshots | `linkerbot_sim.snapshots` | domain | 稳定 lazy snapshots public facade | pure | documented facade | [架构参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.compatibility` | domain | snapshots 层 compatibility 实现 owner | pure | internal | [架构参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.mirror_adapter` | domain | snapshots 层 mirror adapter 实现 owner | pure | internal | [架构参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.persistence` | domain | snapshots 层 persistence 实现 owner | pure | internal | [架构参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.runtime_objects` | domain | snapshots 层 runtime objects 实现 owner | pure | internal | [架构参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.schema` | domain | snapshots 层 schema 实现 owner | pure | internal | [架构参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.transactions` | domain | snapshots 层 transactions 实现 owner | pure | internal | [架构参考](../reference/snapshots.md) |

### telemetry (4)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| telemetry | `linkerbot_sim.telemetry` | outputs | telemetry 实现命名空间 | Isaac main thread | internal | [架构参考](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove` | outputs | telemetry 层 foxglove 实现 owner | Isaac main thread | internal | [架构参考](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove_state` | outputs | telemetry 层 foxglove state 实现 owner | Isaac main thread | internal | [架构参考](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.state_snapshot` | outputs | telemetry 层 state snapshot 实现 owner | Isaac main thread | internal | [架构参考](../guides/telemetry.md) |

### logging (5)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| logging | `linkerbot_sim.logging` | outputs | logging 实现命名空间 | pure | internal | [架构参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.csv_writer` | outputs | logging 层 csv writer 实现 owner | pure | internal | [架构参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.effort_logger` | outputs | logging 层 effort logger 实现 owner | pure | internal | [架构参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.hybrid_control_logger` | outputs | logging 层 hybrid control logger 实现 owner | pure | internal | [架构参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.joint_logger` | outputs | logging 层 joint logger 实现 owner | pure | internal | [架构参考](../reference/outputs.md) |

### visualization (2)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| visualization | `linkerbot_sim.visualization` | isaac_infrastructure | visualization 实现命名空间 | pure | internal | [架构参考](../development/usd-preview.md) |
| visualization | `linkerbot_sim.visualization.viewport` | isaac_infrastructure | visualization 层 viewport 实现 owner | pure | internal | [架构参考](../development/usd-preview.md) |

### mirror (41)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| mirror | `linkerbot_sim.mirror` | product | 稳定 lazy mirror public facade | pure | documented facade | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.app` | product | 进程生命周期与服务组合 | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.bootstrap` | product | 组合根与资源所有权移交 | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.cli` | product | 命令行解析与进程启动 | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision` | product | mirror 层 collision 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.envelope_provider` | product | mirror 层 envelope provider 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.object_provider` | product | mirror 层 object provider 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.owner` | product | mirror 层 owner 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.registry` | product | mirror 层 registry 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.robot_provider` | product | mirror 层 robot provider 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.collision.urdf_kinematics` | product | mirror 层 urdf kinematics 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.control_mode` | product | mirror 层 control mode 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.controller` | product | owner 线程指令分派与安全控制 | Isaac main thread | owner path | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.hybrid_parameters` | product | mirror 层 hybrid parameters 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface` | product | 产品接口命名空间 | pure | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.admission` | product | 有界请求准入与响应所有权 | pure | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.protocol` | product | 严格版本化 wire protocol | pure | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.state_stream` | product | mirror 层 state stream 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.interface.transport` | product | stdin、TCP JSONL 与 WebSocket ingress | pure | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.lifecycle` | product | mirror 层 lifecycle 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion` | product | mirror 层 motion 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.backend` | product | mirror 层 backend 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.hybrid_executor` | product | mirror 层 hybrid executor 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.owner` | product | mirror 层 owner 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.request_parser` | product | mirror 层 request parser 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline` | product | mirror 层 timeline 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.builders` | product | mirror 层 builders 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.compiler` | product | mirror 层 compiler 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.executor` | product | mirror 层 executor 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.model` | product | mirror 层 model 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.motion.timeline.requests` | product | mirror 层 requests 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.rendering` | product | 渲染与相机资源协调 | Isaac main thread | owner path | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.reset` | product | 事务式 runtime reset 编排 | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.reset_runtime` | product | mirror 层 reset runtime 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.robots` | product | mirror 层 robots 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.runtime` | product | 资源生命周期与仿真步进编排 | Isaac main thread | owner path | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.scene_assembly` | product | mirror 层 scene assembly 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.scene_settings` | product | mirror 层 scene settings 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.snapshot` | product | 自有 snapshot schema 与恢复语义 | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.state` | product | Mirror state 读取与写入 | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |
| mirror | `linkerbot_sim.mirror.timing` | product | mirror 层 timing 实现 owner | Isaac main thread | internal | [架构参考](../getting-started/project-overview.md) |

### kaleidoscope (32)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| kaleidoscope | `linkerbot_sim.kaleidoscope` | product | 稳定 lazy kaleidoscope public facade | pure | documented facade | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.actions` | product | 定形 action 校验与写入 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.adapters` | product | 外部 API adapter 命名空间 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.adapters.gymnasium` | product | kaleidoscope 层 gymnasium 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.bootstrap` | product | 组合根与资源所有权移交 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.checkpoint` | product | 显式冷持久化 checkpoint 边界 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.control_commands` | product | kaleidoscope 层 control commands 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.control_mode` | product | kaleidoscope 层 control mode 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.control_runtime` | product | kaleidoscope 层 control runtime 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.env` | product | 训练环境生命周期 | cuRobo/CUDA | owner path | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.geometry` | product | kaleidoscope 层 geometry 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.ik` | product | 设备原生批量逆运动学 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.isaac_adapter` | product | Isaac runtime adapter 边界 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.isaac_views` | product | 定形 Isaac tensor view | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.linear_motion` | product | 同步设备原生线性运动 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.newton_ports` | product | kaleidoscope 层 newton ports 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.observations` | product | 设备原生 observation 组装 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.physx_ports` | product | PhysX CUDA tensor port 合同 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.registration` | product | 显式 Gymnasium 注册 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.resets` | product | 批量 reset 与 autoreset 语义 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.rewards` | product | 设备原生 reward 计算 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.runtime` | product | 资源生命周期与仿真步进编排 | cuRobo/CUDA | owner path | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.scene_assembly` | product | kaleidoscope 层 scene assembly 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.snapshot` | product | 自有 snapshot schema 与恢复语义 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.state_api` | product | 批量 state、snapshot 与 clone API | cuRobo/CUDA | owner path | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.task` | product | vector task 合同与 step 语义 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.task_buffers` | product | 自有定形 task 状态 buffer | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.tasks` | product | 已注册 task 实现命名空间 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.tasks.tblock_push_v1` | product | kaleidoscope 层 tblock push v1 实现 owner | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.tensors` | product | CUDA tensor 校验与分配不变量 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.terminations` | product | 设备原生终止与截断规则 | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |
| kaleidoscope | `linkerbot_sim.kaleidoscope.training_port` | product | 公开 CUDA 训练环境 protocol | cuRobo/CUDA | internal | [架构参考](../getting-started/choose-runtime-and-api.md) |

### training (6)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| training | `linkerbot_sim.training` | training | training 实现命名空间 | pure | internal | [架构参考](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl` | training | 稳定 lazy training public facade | pure | documented facade | [架构参考](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.env` | training | 训练环境生命周期 | cuRobo/CUDA | internal | [架构参考](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.factory` | training | 验收后的具体 runtime 构造 | cuRobo/CUDA | internal | [架构参考](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.final_observation_ppo` | training | training 层 final observation ppo 实现 owner | cuRobo/CUDA | internal | [架构参考](../operations/constraints.md) |
| training | `linkerbot_sim.training.skrl.memory` | training | CUDA rollout memory 集成 | cuRobo/CUDA | internal | [架构参考](../operations/constraints.md) |

### utils (9)

| Group | Module | Layer | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- | --- |
| utils | `linkerbot_sim.utils` | foundation | utils 实现命名空间 | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.config` | foundation | utils 层 config 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.json` | foundation | utils 层 json 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.math_utils` | foundation | utils 层 math utils 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.output_paths` | foundation | utils 层 output paths 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.paths` | foundation | utils 层 paths 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.rotations` | foundation | utils 层 rotations 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.tensors` | foundation | CUDA tensor 校验与分配不变量 | pure | internal | [架构参考](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.timing` | foundation | utils 层 timing 实现 owner | pure | internal | [架构参考](../getting-started/project-overview.md) |

<!-- module-inventory:end -->
