# 源码模块图

语言：[中文](module-map.md) | [English](../../en/development/module-map.md)

本文完整列出 `src/linkerbot_sim/**/*.py`，供维护者和需要定位实现所有者的高级进程内
调用方导航。可观察契约以每行链接的任务指南或参考页为准；本文不重复协议字段、配置字段
或 Python 符号签名。
受支持的 import path 和精确 symbol 由 [Python Facade 参考](../reference/python-api.md)统一负责。

运行前提记录该行职责所需的最强约束：

- `pure`：不需要运行 Kit/Isaac 应用。
- `Isaac main thread`：必须先启动 Isaac，并在仿真所有者线程调用运行时操作。
- `cuRobo/CUDA`：数值操作需要项目指定的 GPU、Torch、Warp 和 cuRobo 环境；关联的
  stage 访问仍必须在 Isaac 主线程执行。

当后端无关模块接收注入的 solver 时，该行标记模块自身工作的前提；responsibility 会注明继承
所注入 solver 更强运行要求的分支。

分类刻意比源码可见性更严格：

- `documented facade`：该 package 是预期导入面，但只有 Python 参考明确列出的符号才可依赖。
- `owner path`：理解或维护某项事实的规范位置，是高级导航路径，不构成稳定 API 承诺。
- `internal`：组合或实现细节；调用方必须通过已文档化接口进入。

`linkerbot_sim.tiled.__all__` 为空。因此，`linkerbot_sim.tiled` 及其 32 个后代模块都不是
顶层公共 API。下表中少量标为 `owner path` 的 tiled 叶模块仍然只负责源码导航。

## 接口与所有者登记表

以下可机读登记表包含 inventory 中所有非 `internal` 项。覆盖测试要求其分类和运行前提
与完整 inventory 完全一致。

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

## 完整 Inventory

Inventory 按 `linkerbot_sim` 下的第一层 package 分组，package 根归入 `root`。每一行都链接到
与该模块职责最接近的详细文档所有者。

<!-- module-inventory:start -->

### root (1)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| root | `linkerbot_sim` | 仓库根路径定位与 package facade | pure | documented facade | [Python Facade 参考](../reference/python-api.md) |

### app (50)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| app | `linkerbot_sim.app` | 应用启动命名空间 | pure | internal | [项目概览](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.interactive` | 交互 runtime 组合命名空间 | pure | internal | [Runtime 与 API 选择](../getting-started/choose-runtime-and-api.md) |
| app | `linkerbot_sim.app.interactive.policies` | 请求限制与策略校验 | pure | internal | [运行约束](../operations/constraints.md) |
| app | `linkerbot_sim.app.interactive.single_scene.protocol` | 共用 JSON 响应与错误辅助 | pure | internal | [Single Scene JSON 参考](../reference/single-scene-json.md) |
| app | `linkerbot_sim.app.interactive.single_scene.queue` | 有界运动请求队列 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.single_scene` | Single Scene 交互 Python 入口 | Isaac main thread | documented facade | [Python Facade 参考](../reference/python-api.md) |
| app | `linkerbot_sim.app.interactive.single_scene.cli` | Single Scene CLI 解析与进程启动 | Isaac main thread | internal | [Single Scene CLI 参考](../reference/single-scene-cli.md) |
| app | `linkerbot_sim.app.interactive.single_scene.runtime` | Single Scene 请求循环与 runtime 绑定 | Isaac main thread | internal | [Single Scene JSON 参考](../reference/single-scene-json.md) |
| app | `linkerbot_sim.app.interactive.single_scene.state_stream` | Single Scene 状态采样与发布 | Isaac main thread | internal | [遥测指南](../guides/telemetry.md) |
| app | `linkerbot_sim.app.interactive.stdin_reader` | 有界 stdin JSONL 读取 | pure | internal | [Single Scene JSON 参考](../reference/single-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene` | Tiled Scene 交互 Python 入口 | Isaac main thread | documented facade | [Python Facade 参考](../reference/python-api.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.action_messages` | Tiled Scene 同步 action 解析 | pure | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.cli` | Tiled Scene CLI 解析与进程启动 | Isaac main thread | internal | [Tiled Scene CLI 参考](../reference/tiled-scene-cli.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.command_utils` | Tiled Scene 指令校验辅助 | pure | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.hand_messages` | Tiled Scene hand 指令路由 | Isaac main thread | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.message_utils` | Tiled Scene 响应与请求工具 | pure | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.plan_messages` | Tiled Scene 异步 plan 消息路由 | Isaac main thread | internal | [运动规划](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.protocol` | Tiled Scene 指令分派 | Isaac main thread | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime` | Tiled Scene runtime 类导出 | Isaac main thread | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.core` | Tiled Scene runtime 生命周期所有者 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.factory` | Tiled Scene 场景与服务装配 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.ik` | 批量 IK 服务集成 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.planning` | 规划提交与结果收集 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.state` | 选定环境状态与快照访问 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.runtime.stepping` | Tiled Scene 物理步进与回放 | Isaac main thread | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.selectors` | 环境与机器人 selector 解析 | pure | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.telemetry_publish` | Tiled Scene 遥测发布适配 | Isaac main thread | internal | [遥测指南](../guides/telemetry.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.trajectory_messages` | Tiled Scene 轨迹缓冲消息路由 | Isaac main thread | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.interactive.tiled_scene.transport` | Tiled Scene stdin、TCP 与 WebSocket 循环 | pure | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| app | `linkerbot_sim.app.interactive.single_scene.transports` | 共用有界网络 transport | pure | internal | [运行约束](../operations/constraints.md) |
| app | `linkerbot_sim.app.launch` | SimulationApp 设置与启动所有者 | Isaac main thread | owner path | [配置参考](../reference/configuration.md) |
| app | `linkerbot_sim.app.motion` | 应用运动命名空间 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline` | Single Scene timeline 实现命名空间 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.builders` | Timeline tick 与可执行 track 构造 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.compiler` | 请求到 timeline 的原子编译 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| app | `linkerbot_sim.app.motion.timeline.executor` | 主线程 timeline 执行 | Isaac main thread | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.model` | 不可变整数 tick timeline 模型 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.motion.timeline.requests` | 后端无关 timeline 请求模型 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| app | `linkerbot_sim.app.runtime` | Single Scene runtime 组合命名空间 | pure | internal | [项目概览](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.runtime.collision` | 规划碰撞 provider registry | Isaac main thread | owner path | [碰撞模型](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.envelope_provider` | 保守机器人包络 provider | pure | internal | [碰撞模型](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.object_provider` | Runtime 物体碰撞转换 | pure | internal | [碰撞模型](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.registry` | 碰撞 provider registry 与指纹 | pure | internal | [碰撞模型](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.robot_provider` | 机器人状态碰撞球 provider | Isaac main thread | internal | [碰撞模型](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.collision.urdf_kinematics` | 轻量 URDF 正向运动学 | pure | internal | [碰撞模型](../guides/collision-models.md) |
| app | `linkerbot_sim.app.runtime.single_scene_reset` | 已有 session 重置编排 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| app | `linkerbot_sim.app.runtime.robot_registry` | 机器人身份与规划上下文 registry | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.runtime.single_scene_runtime` | Single Scene 生命周期与资源所有者 | Isaac main thread | owner path | [项目概览](../getting-started/project-overview.md) |
| app | `linkerbot_sim.app.runtime.simulation_app_lifecycle` | SimulationApp 关闭协调 | Isaac main thread | internal | [运行约束](../operations/constraints.md) |
| app | `linkerbot_sim.app.runtime.simulation_session` | SimulationApp、World 与场景 session 装配 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |

### assets (7)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| assets | `linkerbot_sim.assets` | 机器人资产实现命名空间 | pure | internal | [命名规范](naming.md) |
| assets | `linkerbot_sim.assets.robot_config` | 机器人资产与物理配置所有者 | pure | owner path | [配置参考](../reference/configuration.md) |
| assets | `linkerbot_sim.assets.robot_import` | Isaac 机器人资产导入 | Isaac main thread | internal | [碰撞近似](collision-approximation.md) |
| assets | `linkerbot_sim.assets.robot_instances` | Scene 机器人身份与执行设置 | pure | owner path | [命名规范](naming.md) |
| assets | `linkerbot_sim.assets.root_pose` | Root pose 模型与 USD 写入 | Isaac main thread | owner path | [命名规范](naming.md) |
| assets | `linkerbot_sim.assets.solver_overrides` | PhysX solver 覆盖应用 | Isaac main thread | internal | [碰撞近似](collision-approximation.md) |
| assets | `linkerbot_sim.assets.usd_overrides` | 导入后 USD 与 PhysX 覆盖 | Isaac main thread | internal | [碰撞近似](collision-approximation.md) |

### backends (24)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| backends | `linkerbot_sim.backends` | 数值后端命名空间 | pure | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo` | cuRobo 后端 facade | cuRobo/CUDA | documented facade | [Python Facade 参考](../reference/python-api.md) |
| backends | `linkerbot_sim.backends.curobo.batch` | 批量数值内核命名空间 | pure | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.ik` | 批量 cuRobo IK solver | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.joint_planner` | 批量关节空间规划器 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.result_adapter` | 批量结果转换 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.batch.types` | 批量 solver 数据结构 | pure | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.call_guard` | 串行 cuRobo 调用边界 | pure | internal | [运行约束](../operations/constraints.md) |
| backends | `linkerbot_sim.backends.curobo.collision_capability` | 碰撞能力选择 | pure | internal | [碰撞模型](../guides/collision-models.md) |
| backends | `linkerbot_sim.backends.curobo.collision_world` | cuRobo 碰撞 world 构建 | cuRobo/CUDA | internal | [碰撞模型](../guides/collision-models.md) |
| backends | `linkerbot_sim.backends.curobo.config` | cuRobo 配置模型 | pure | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.context` | Solver 上下文与能力生命周期 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.forward_kinematics` | cuRobo 正向运动学 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.inverse_kinematics` | cuRobo 逆向运动学 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.joint_mapping` | 项目到 cuRobo 关节映射 | pure | internal | [命名规范](naming.md) |
| backends | `linkerbot_sim.backends.curobo.linear_pose_path` | 笛卡尔直线路径规划 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.motion_planner` | 运动规划器编排 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.profile_merge` | 机器人与算法 profile 合并 | pure | internal | [配置指南](../guides/configuration.md) |
| backends | `linkerbot_sim.backends.curobo.robot_model` | cuRobo 机器人模型物化 | pure | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.runtime_imports` | cuRobo 依赖延迟加载 | cuRobo/CUDA | internal | [运行约束](../operations/constraints.md) |
| backends | `linkerbot_sim.backends.curobo.tensor_adapter` | 数组到 device tensor 转换 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.tool_pose` | Tool pose 坐标转换 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| backends | `linkerbot_sim.backends.curobo.trajectory_adapter` | cuRobo 轨迹转换 | cuRobo/CUDA | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| backends | `linkerbot_sim.backends.curobo.warp_compat` | 项目指定 Warp API 适配 | cuRobo/CUDA | internal | [运行约束](../operations/constraints.md) |

### configs (6)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| configs | `linkerbot_sim.configs` | 项目配置命名空间 | pure | internal | [配置指南](../guides/configuration.md) |
| configs | `linkerbot_sim.configs.cli` | CLI overlay 应用 | pure | internal | [配置指南](../guides/configuration.md) |
| configs | `linkerbot_sim.configs.instance_paths` | 实例 prim path 校验 | pure | internal | [命名规范](naming.md) |
| configs | `linkerbot_sim.configs.profiles` | 严格 profile 加载所有者 | pure | owner path | [配置参考](../reference/configuration.md) |
| configs | `linkerbot_sim.configs.runtime` | Runtime profile 模型与组合 | pure | owner path | [配置参考](../reference/configuration.md) |
| configs | `linkerbot_sim.configs.validator` | 完整 profile 图校验 | pure | owner path | [配置参考](../reference/configuration.md) |

### controllers (4)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| controllers | `linkerbot_sim.controllers` | 关节控制 facade | Isaac main thread | documented facade | [Python Facade 参考](../reference/python-api.md) |
| controllers | `linkerbot_sim.controllers.config` | 控制器 profile 解析所有者 | pure | owner path | [配置参考](../reference/configuration.md) |
| controllers | `linkerbot_sim.controllers.joint_controller` | Articulation target 与模式控制 | Isaac main thread | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| controllers | `linkerbot_sim.controllers.types` | 控制设置与 target 模型 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |

### envs (5)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| envs | `linkerbot_sim.envs` | 环境实现命名空间 | pure | internal | [配置指南](../guides/configuration.md) |
| envs | `linkerbot_sim.envs.config` | Env profile 与片段校验 | pure | owner path | [配置参考](../reference/configuration.md) |
| envs | `linkerbot_sim.envs.scene_builder` | Isaac 基础 world 构建 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| envs | `linkerbot_sim.envs.settings` | 环境 runtime 设置所有者 | pure | owner path | [配置参考](../reference/configuration.md) |
| envs | `linkerbot_sim.envs.visual_settings` | Scene visual 设置所有者 | pure | owner path | [配置参考](../reference/configuration.md) |

### execution (4)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| execution | `linkerbot_sim.execution` | 仿真执行 facade | Isaac main thread | documented facade | [Python Facade 参考](../reference/python-api.md) |
| execution | `linkerbot_sim.execution.runtime` | 执行上下文与 step 协议 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| execution | `linkerbot_sim.execution.setup` | 执行侧机器人装配 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| execution | `linkerbot_sim.execution.steps` | 可复用控制执行步骤 | Isaac main thread | internal | [控制与轨迹](../guides/control-and-trajectories.md) |

### logging (5)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| logging | `linkerbot_sim.logging` | CSV 日志实现命名空间 | pure | internal | [输出参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.config` | Single Scene 日志 profile 所有者 | pure | owner path | [输出参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.csv_writer` | 有界关节 CSV writer | pure | internal | [输出参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.effort_logger` | Articulation effort 采样 | Isaac main thread | internal | [输出参考](../reference/outputs.md) |
| logging | `linkerbot_sim.logging.joint_logger` | 关节 target 与状态采样 | Isaac main thread | internal | [输出参考](../reference/outputs.md) |

### objects (10)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| objects | `linkerbot_sim.objects` | 场景物体 facade | Isaac main thread | documented facade | [Python Facade 参考](../reference/python-api.md) |
| objects | `linkerbot_sim.objects.config` | 物体 profile 与实例解析 | pure | internal | [配置参考](../reference/configuration.md) |
| objects | `linkerbot_sim.objects.dynamic_chain` | 动态链物体所有者 | Isaac main thread | owner path | [物体资产](object-assets.md) |
| objects | `linkerbot_sim.objects.dynamic_chain.capsule_rope` | Capsule rope 引用与物理设置 | Isaac main thread | internal | [物体资产](object-assets.md) |
| objects | `linkerbot_sim.objects.physics` | 物体材质与 root pose 写入 | Isaac main thread | internal | [碰撞模型](../guides/collision-models.md) |
| objects | `linkerbot_sim.objects.rigid` | Rigid object 所有者 | Isaac main thread | owner path | [物体资产](object-assets.md) |
| objects | `linkerbot_sim.objects.rigid.config` | Rigid object 与规划 collider 模型 | pure | internal | [碰撞模型](../guides/collision-models.md) |
| objects | `linkerbot_sim.objects.rigid.importer` | Rigid USD 导入与物理设置 | Isaac main thread | internal | [物体资产](object-assets.md) |
| objects | `linkerbot_sim.objects.runtime` | Runtime 物体装配与 handle | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| objects | `linkerbot_sim.objects.state_views` | Scene 物体状态读取与恢复 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |

### planning (8)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| planning | `linkerbot_sim.planning` | 后端无关规划 facade | pure | documented facade | [Python Facade 参考](../reference/python-api.md) |
| planning | `linkerbot_sim.planning.backend` | 规划器协议与后端选择 | pure | internal | [运动规划](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.batch_ik` | 批量 IK 协议与结果模型 | pure | internal | [运动规划](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.collision_objects` | 规划碰撞物体模型 | pure | internal | [碰撞模型](../guides/collision-models.md) |
| planning | `linkerbot_sim.planning.frames` | 规划 frame 转换 | pure | internal | [运动规划](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.linear_backend` | 后端无关直线规划器 | pure | internal | [运动规划](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.requests` | IK 与运动请求模型 | pure | internal | [运动规划](../guides/motion-planning.md) |
| planning | `linkerbot_sim.planning.results` | IK 与运动结果模型 | pure | internal | [运动规划](../guides/motion-planning.md) |

### robots (9)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| robots | `linkerbot_sim.robots` | 机器人能力与关节组 facade | pure | documented facade | [Python Facade 参考](../reference/python-api.md) |
| robots | `linkerbot_sim.robots.capabilities` | 机器人规划能力模型 | pure | internal | [运动规划](../guides/motion-planning.md) |
| robots | `linkerbot_sim.robots.classification` | 机器人组件分类 | pure | internal | [命名规范](naming.md) |
| robots | `linkerbot_sim.robots.joint_groups` | 命名关节组布局与解析 | pure | internal | [命名规范](naming.md) |
| robots | `linkerbot_sim.robots.mimic` | Mimic 关系实现命名空间 | pure | internal | [命名规范](naming.md) |
| robots | `linkerbot_sim.robots.mimic.assets` | 资产 mimic 关系解析 | pure | internal | [命名规范](naming.md) |
| robots | `linkerbot_sim.robots.mimic.mjcf` | MJCF equality 与 friction 解析 | pure | internal | [命名规范](naming.md) |
| robots | `linkerbot_sim.robots.mimic.runtime` | Mimic follower target 展开 | Isaac main thread | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| robots | `linkerbot_sim.robots.mimic.urdf` | URDF mimic 关系解析 | pure | internal | [命名规范](naming.md) |

### sensors (10)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| sensors | `linkerbot_sim.sensors` | Scene sensor 设置 facade | pure | documented facade | [Python Facade 参考](../reference/python-api.md) |
| sensors | `linkerbot_sim.sensors.camera` | 相机配置 owner path | pure | owner path | [相机指南](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.config` | 相机设置解析 | pure | internal | [相机指南](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.foxglove` | 相机帧 Foxglove 编码 | pure | internal | [输出参考](../reference/outputs.md) |
| sensors | `linkerbot_sim.sensors.camera.frame` | 不可变相机帧模型 | pure | internal | [相机指南](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.limits` | 相机输出资源限制 | pure | internal | [运行约束](../operations/constraints.md) |
| sensors | `linkerbot_sim.sensors.camera.observer` | World step 相机观测 | Isaac main thread | internal | [相机指南](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.camera.recorder` | 文件记录与有界发布 | pure | internal | [输出参考](../reference/outputs.md) |
| sensors | `linkerbot_sim.sensors.camera.runtime` | Isaac 相机创建与采样 | Isaac main thread | internal | [相机指南](../guides/cameras.md) |
| sensors | `linkerbot_sim.sensors.config` | Scene sensor 聚合设置 | pure | internal | [配置参考](../reference/configuration.md) |

### snapshots (9)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| snapshots | `linkerbot_sim.snapshots` | 捕获与恢复 facade | Isaac main thread | documented facade | [Python Facade 参考](../reference/python-api.md) |
| snapshots | `linkerbot_sim.snapshots.compatibility` | 快照到目标的身份匹配 | pure | internal | [快照参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.debug_tiled_scene_adapter` | 内存 tiled 快照适配 | pure | internal | [快照参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.dispatch` | Runtime 形状快照分派 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.runtime_objects` | 共用 runtime 物体快照辅助 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.single_scene_adapter` | Single Scene 捕获与事务式恢复 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.schema` | Runtime 无关快照数据模型 | pure | internal | [快照参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.tiled_scene_adapter` | Tiled Scene 捕获、恢复与 env 复制 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| snapshots | `linkerbot_sim.snapshots.transactions` | 恢复事务与 fail-stop 状态 | pure | internal | [快照参考](../reference/snapshots.md) |

### telemetry (8)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| telemetry | `linkerbot_sim.telemetry` | 遥测实现命名空间 | pure | internal | [遥测指南](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove` | Foxglove live 与 MCAP sink 所有者 | pure | owner path | [Foxglove 指南](../guides/foxglove.md) |
| telemetry | `linkerbot_sim.telemetry.foxglove_state` | 状态到 Foxglove 序列化 | pure | internal | [遥测指南](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.state_snapshot` | Single Scene 状态采样与转交 | Isaac main thread | owner path | [遥测指南](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled` | Tiled Scene 遥测实现命名空间 | pure | internal | [遥测指南](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled.config` | Tiled Scene 遥测设置所有者 | pure | owner path | [遥测指南](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled.payloads` | Tiled Scene 状态 payload 转换 | pure | internal | [遥测指南](../guides/telemetry.md) |
| telemetry | `linkerbot_sim.telemetry.tiled.sink` | Tiled Scene live 与 MCAP sink 生命周期 | pure | internal | [输出参考](../reference/outputs.md) |

### tiled (33)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| tiled | `linkerbot_sim.tiled` | 无导出的 Tiled Scene 实现命名空间 | pure | internal | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| tiled | `linkerbot_sim.tiled.config` | Tiled env 配置所有者 | pure | owner path | [配置参考](../reference/configuration.md) |
| tiled | `linkerbot_sim.tiled.control` | 同步 tiled 控制命名空间 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.control.adapter` | 后端无关批量 target 转换；EE 路径调用注入的 IK solver | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.control.interpolation` | 固定 tick 关节插值 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.control.types` | Tiled Scene action 数据与值域 | pure | owner path | [Tiled Scene JSON 参考](../reference/tiled-scene-json.md) |
| tiled | `linkerbot_sim.tiled.planning` | Tiled Scene 规划实现命名空间 | pure | internal | [运动规划](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.backends` | Tiled Scene 规划后端组合 | pure | internal | [运动规划](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.backends.curobo` | Tiled Scene 到 cuRobo 规划适配 | cuRobo/CUDA | internal | [运动规划](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.batching` | 同构请求 batch 布局 | pure | internal | [运动规划](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.linear_backend` | 直线规划器 batch 适配 | pure | internal | [运动规划](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.manager` | 规划队列、worker 与取消 | pure | internal | [运动规划](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.planning.types` | Tiled Scene 规划请求与结果所有者 | pure | owner path | [运动规划](../guides/motion-planning.md) |
| tiled | `linkerbot_sim.tiled.playback` | Per-env 回放实现命名空间 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.playback.buffer` | Per-env 轨迹回放缓冲 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.playback.models` | 回放 track 与 cursor 模型 | pure | owner path | [控制与轨迹](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.playback.staging` | Before、main 与 after track staging | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| tiled | `linkerbot_sim.tiled.scene` | Tiled Scene Isaac 场景命名空间 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.builder` | Tiled Scene 场景构建编排 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.cameras` | Per-env 相机配置展开 | pure | internal | [相机指南](../guides/cameras.md) |
| tiled | `linkerbot_sim.tiled.scene.clone` | Grid clone 与 replication 设置 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.collision_filter` | 跨环境碰撞过滤 | Isaac main thread | internal | [碰撞模型](../guides/collision-models.md) |
| tiled | `linkerbot_sim.tiled.scene.objects` | Per-env 物体导入与覆盖 | Isaac main thread | internal | [物体资产](object-assets.md) |
| tiled | `linkerbot_sim.tiled.scene.paths` | Env root path 与网格 origin | pure | internal | [命名规范](naming.md) |
| tiled | `linkerbot_sim.tiled.scene.robots` | Per-env 机器人导入与身份绑定 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.root_pose` | Clone 后机器人 root pose 覆盖 | Isaac main thread | internal | [配置参考](../reference/configuration.md) |
| tiled | `linkerbot_sim.tiled.scene.types` | 不可变 tiled 场景装配模型 | pure | owner path | [项目概览](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.utils` | 轻量场景构建辅助 | pure | internal | [项目概览](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.scene.views` | 批量 articulation view 绑定 | Isaac main thread | internal | [项目概览](../getting-started/project-overview.md) |
| tiled | `linkerbot_sim.tiled.state` | 批量状态实现命名空间 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| tiled | `linkerbot_sim.tiled.state.object_io` | 选定 env 物体状态编排 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| tiled | `linkerbot_sim.tiled.state.object_views` | 批量 PhysX 物体状态 view | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |
| tiled | `linkerbot_sim.tiled.state.usd_pose` | USD pose 读写与速度清零 | Isaac main thread | internal | [快照参考](../reference/snapshots.md) |

### trajectories (4)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| trajectories | `linkerbot_sim.trajectories` | 轨迹实现命名空间 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.joint_trajectory_builder` | 关节轨迹构造所有者 | pure | owner path | [控制与轨迹](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.retiming` | 时间网格与路径进度重采样 | pure | owner path | [控制与轨迹](../guides/control-and-trajectories.md) |
| trajectories | `linkerbot_sim.trajectories.types` | 关节轨迹数据模型所有者 | pure | owner path | [控制与轨迹](../guides/control-and-trajectories.md) |

### utils (8)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| utils | `linkerbot_sim.utils` | 无副作用工具命名空间 | pure | internal | [项目概览](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.config` | 严格 YAML 与 mapping 辅助 | pure | internal | [配置参考](../reference/configuration.md) |
| utils | `linkerbot_sim.utils.json` | 严格 JSON 编解码 | pure | internal | [运行约束](../operations/constraints.md) |
| utils | `linkerbot_sim.utils.math_utils` | 共用数值变换 | pure | internal | [运动规划](../guides/motion-planning.md) |
| utils | `linkerbot_sim.utils.output_paths` | 输出路径规划与应用 | pure | internal | [输出参考](../reference/outputs.md) |
| utils | `linkerbot_sim.utils.paths` | 仓库路径解析 | pure | internal | [项目概览](../getting-started/project-overview.md) |
| utils | `linkerbot_sim.utils.rotations` | RPY、四元数与矩阵转换 | pure | internal | [运动规划](../guides/motion-planning.md) |
| utils | `linkerbot_sim.utils.timing` | 采样差分辅助 | pure | internal | [控制与轨迹](../guides/control-and-trajectories.md) |

### visualization (2)

| Group | Module | Responsibility | Runtime | Classification | Related documentation |
| --- | --- | --- | --- | --- | --- |
| visualization | `linkerbot_sim.visualization` | 本地 viewport 辅助命名空间 | pure | internal | [USD 预览](usd-preview.md) |
| visualization | `linkerbot_sim.visualization.viewport` | GUI viewport 相机放置 | Isaac main thread | internal | [USD 预览](usd-preview.md) |

<!-- module-inventory:end -->
