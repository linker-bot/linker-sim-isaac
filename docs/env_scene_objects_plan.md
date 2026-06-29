# Env Scene Objects Plan

## 目标

让 `configs/envs/*.yaml` 可以描述物理世界里的环境物体，例如 workstation、table、预生成
USD 物体。动作脚本只负责选择 env config；场景中有什么、放在哪里，由 env config 决定。

这层只处理“场景布置”，不处理对象生成。像 capsule rope 的段数、半径、质量、D6 joint 等
生成参数仍放在 `configs/objects/*.yaml` 和 `src/linkerbot_sim/objects/` 中。

## 配置边界

- `configs/envs/*.yaml`
  - world 参数：`physics_frequency`、`render_frequency`、`gravity_z`。
  - PhysX solver 覆盖：`solver.type`、`solver.arm_*`、`solver.hand_*`。
  - 场景物体列表：`objects[]`，描述已有资产如何进入当前 stage。
  - 环境固定语义：`objects[].physics.static`，描述该物体是否作为固定环境物体。
  - 可选物理材质覆盖：`objects[].physics.material`，只写需要覆盖的字段。
- `configs/objects/*.yaml`
  - 生成型对象的自身参数，例如 capsule rope 的几何、质量、关节和输出 USD 路径。
- `assets/static_env_objects/`
  - 静态环境资产，例如 workstation/table 的 URDF/USD。
- `assets/dynamic_env_objects/`
  - 预生成动态对象资产，例如 capsule rope USD。

## 第一版配置格式

```yaml
objects:
  - name: workstation_armbase
    asset_type: urdf
    asset_path: assets/static_env_objects/workstationV1_armbase/workstationV1_armbase.urdf
    prim_path: /World/WorkstationArmBase
    root_pose:
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
    physics:
      static: true
      material:
        static_friction: 0.8
        dynamic_friction: 0.6
        restitution: 0.0
        friction_combine_mode: average

  - name: workstation_tablebase
    asset_type: urdf
    asset_path: assets/static_env_objects/workstationV1_tablebase/workstationV1_tablebase.urdf
    prim_path: /World/WorkstationTableBase
    root_pose:
      xyz: [0.03, 0.0, -0.5]
      rpy: [0.0, 0.0, 0.0]
    physics:
      static: true

  - name: rope_reference
    asset_type: usd
    asset_path: assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
    prim_path: /World/CapsuleRopePreview
```

字段语义：

- `name`：日志和诊断用名称。
- `asset_type`：第一版支持 `usd` 和 `urdf`。
- `asset_path`：仓库相对路径或绝对路径。
- `prim_path`：放入当前 USD stage 的目标路径。
- `root_pose.xyz/rpy`：资产根 prim 相对 world 的位姿；不写时为零位姿。
- `physics.static`：是否把该对象作为环境固定物体；URDF 导入时会映射到 importer 的
  `fix_base`，USD/引用资产若包含刚体则会设为 kinematic 并关闭重力。
- `physics.material.static_friction/dynamic_friction/restitution`：可选接触材质属性。
- `physics.material.friction_combine_mode`：可选 PhysX friction combine mode，支持
  `average`、`min`、`multiply`、`max`。

`physics.material` 和其中每个字段都是可选的：不写 `material` 就不创建/绑定 env 侧物理材质；
写了 `material` 但缺某个字段，就不写入对应 USD 属性。这样 env 可以按场景覆盖摩擦/材质，
同时保持缺省配置不修改资产自带属性。

## 调用结构

```text
load_yaml(configs/envs/*.yaml)
  -> scene_objects_from_env_config(...)
     -> add_scene_objects(stage, scene_objects)
       usd  -> define Xform at prim_path + AddReference(asset_path)
       urdf -> URDF importer + apply root_pose to imported root
             + apply physics.static if requested
             + bind physics.material to collision prims if requested
```

脚本接入点：

```text
build_world(...)
stage = omni.usd.get_context().get_stage()
add_scene_objects(stage, scene_objects_from_env_config(env_config))
import_robot_asset(...)
```

环境对象应在机器人导入前进入 stage，便于后续调试、碰撞检查或视觉确认。

## 第一版非目标

- 不在 env config 中生成 capsule rope；仍由 `configs/objects/capsule_rope.yaml` 和
  `add_capsule_rope_reference(...)` 负责。
- 不自动把 env objects 转成 cuMotion `CollisionObject`。cuMotion 避障仍由动作脚本显式同步。
- 不处理动态对象控制、关节驱动或可抓取对象语义。
- 不强制所有脚本都消费 `objects[]`；dry-run 不启动 Isaac 时只做配置解析。

## 后续扩展

- 支持 `mjcf` 静态/场景资产导入。
- 支持 `scale`、`visible`、`collision_enabled` 等场景放置覆盖。
- 支持从 env objects 显式生成 cuMotion collision objects。
- 将 capsule rope 也迁移成 env object 引用，同时保留生成参数在 `configs/objects`。
