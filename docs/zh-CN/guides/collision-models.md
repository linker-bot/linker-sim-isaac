# 碰撞模型选择指南

语言：[中文](collision-models.md) | [English](../../en/guides/collision-models.md)

项目包含三层相互独立的 collision。应配置真正拥有目标行为的层；开启其中一层不表示另外两层
也已生效。

| 层 | 用途 | 主要 owner |
| --- | --- | --- |
| Simulation collision | PhysX 接触、摩擦、穿透响应和刚体运动 | 导入的 USD/URDF/MJCF collider 与 robot/object physics profile |
| Planning collision | cuRobo 使用的机器人 self-collision 和障碍物检查 | Robot cuRobo model、object `planning_collision` 和 Scene collision registry |
| Tiled env filtering | 阻止不同 clone env root 之间发生接触 | Env `tiled.clone` 配置 |

## Simulation Collision

Isaac importer 从 robot 和 rigid-object asset 物化 collider。项目字段 `collision_approximation`
决定导入几何如何生成碰撞几何；合法值和不同资产格式的限制见
[碰撞近似](../development/collision-approximation.md)。

Object profile 拥有 runtime material、static/dynamic 行为、solver iteration 和 import option；Env
profile 拥有每个 object instance 的路径与 pose。Planning obstacle 不会创建 PhysX collider，关闭
planning collision 也不会关闭物理接触。

需求涉及抓取接触、settling、摩擦、恢复系数、穿透或碰撞后物体运动时，应检查 simulation collision。

## Planning Collision

cuRobo collision-aware request 需要同时满足：

- Robot profile 具有有效 cuRobo planning model 和 collision geometry。
- 已物化的 cuRobo context 支持 collision query。
- 所选 Single Scene/Tiled Scene planning path 支持 `avoid_collisions=true`。
- Planning world 包含本次请求需要考虑的障碍物。

Rigid object profile 可以提供简化 planning shape：

```yaml
planning_collision:
  shape: cuboid
  size: [0.04, 0.2, 0.22]
  xyz: [-0.02, 0.0, 0.11]
  rpy: [0.0, 0.0, 0.0]
  padding: 0.0
  enabled: true
```

该 shape 是显式规划近似，不会替换 object 已 author 的 PhysX compound collider。Canonical shape
以 object profile parser 接受的类型为准；cuRobo adapter 会把它们物化到固定的 `cuboid`/`mesh`
cache model。

Single Scene runtime 注册 object/robot geometry provider，为一次 planning transaction 捕获一份 immutable
planning snapshot，并从障碍物列表中排除 target robot。`coordination="static_others"` 可以把其他
机器人作为冻结障碍物。动态状态变化会把 collision registry 标记为 dirty；
`force_collision_refresh` 请求显式读取当前 view。

`avoid_collisions=true` 是严格要求。Robot sphere 缺失、world collision 不可用、request path 不支持
或 backend 能力不足都会使请求失败，runtime 不会静默执行 collision-unaware trajectory。

需求涉及路径可行性、障碍物避让或 IK/planning 期间的 robot self-collision 时，应检查 planning
collision。它不模拟接触力。

## Tiled Scene Env 间过滤

Clone env 共享一个 PhysX scene。没有 env 间过滤时，相邻 env root 下的 collider 在 spacing 或几何
重叠时可能互相接触。

```yaml
tiled:
  clone:
    filter_collisions: true
    collision_filter_strategy: collision_groups
    collision_root_path: /World/collisions
    physics_scene_path: null
    global_collision_paths: auto
    extra_global_collision_paths: []
```

`physics_scene_path: null` 表示自动发现 stage 中唯一的 `UsdPhysics.Scene`。如果 stage 中没有
PhysicsScene 或存在多个 PhysicsScene，发现会失败；多 scene stage 必须把该字段设为目标 prim
的绝对路径。

`collision_groups` 为每个 env 建立一组，并建立共享 global group。开启 PhysX inverted filter 后，env
只与自身和声明的 global ground/fixture 接触，不与其他 env 接触；author relation 数量随 env 数线性
增长。

`filtered_pairs` 直接 author pair filter，复杂度随 env 数量平方增长。它是在无法使用 collision group
的运行环境中显式选择的另一条路径。只有启用过滤并选择 `collision_groups` 时才能使用 group-only
字段，否则配置会被拒绝。

Env 间过滤只改变物理接触，不会给 planner 创建 obstacle world，也不会启用 cuRobo collision check。

## 选择 Global Path

`global_collision_paths: auto` 扫描 env root 外受支持的 ground 位置。显式列表会替换 auto discovery；
`extra_global_collision_paths` 追加共享 fixture。配置路径必须是 clone env root 外的有效绝对 prim path。
把 env-local prim 声明为 global 会重新引入跨 env coupling。

## 诊断

| 现象 | 检查 |
| --- | --- |
| 仿真中机器人穿过物体 | 导入 collider、`physics.static`、material、collision approximation 和 stage pose |
| 物理接触正常但规划忽略物体 | Object `planning_collision`、Scene collision registry 和 cuRobo capability |
| `avoid_collisions=true` 被拒绝 | Robot collision model、context capability、request kind 和 cache/world 可用性 |
| 不同 Tiled env 相互接触 | `filter_collisions`、strategy、global path、spacing 和 authored group diagnostics |
| 开启过滤后 ground 不再碰撞 | Ground/global prim 是否进入 global collision paths |
| 状态修改后 planning view 陈旧 | Registry invalidation 与 `force_collision_refresh` |

## 相关文档

- [运动规划](motion-planning.md)
- [配置](configuration.md)
- [碰撞近似](../development/collision-approximation.md)
- [Tiled Scene JSON 参考](../reference/tiled-scene-json.md)
- [已知约束](../operations/constraints.md)
