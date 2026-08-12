# Isaac 碰撞近似配置

语言：[中文](collision-approximation.md) | [English](../../en/development/collision-approximation.md)

本文定义 Isaac 资产导入使用的碰撞近似字段。

## 配置归属

URDF 和 MJCF robot importer option 属于 robot profile：

```yaml
robot:
  import:
    collision_approximation: convex_decomposition
    self_collision: false
```

Rigid URDF object importer option 属于 object profile：

```yaml
object:
  import:
    collision_approximation: convex_decomposition
```

USD object reference 不接受 `import` 段。Importer collision approximation 不属于 scene 或
controller YAML。

## 支持的取值

项目级 `collision_approximation` 只接受：

| 取值 | Isaac importer 映射 | 含义 |
| --- | --- | --- |
| `convex_decomposition` | Importer 3.0 `collision_type="Convex Decomposition"`，USD `convexDecomposition` | 把非凸 mesh 分解成多个凸碰撞体。对凹形结构更贴合，但导入/cooking 更慢，碰撞体数量也可能更多。 |
| `convex_hull` | Importer 3.0 `collision_type="Convex Hull"`，USD `convexHull` | 每个 collision mesh 生成一个凸包。更快、更简单，但会填平凹陷区域。 |

项目默认值是 `convex_decomposition`。

其他 USD `physics:approximation` token 不属于该 importer 字段，写入 YAML 会在配置校验阶段失败。

## 物理含义

Importer collision approximation 决定 Isaac 生成的 collision prim 和 approximation attribute；
PhysX 再把这份 USD collision description cooking 成运行时数据。这不是需要手动配置的第二次
凸包处理。

Importer option 受资产格式约束。URDF 还接受 `fix_base`、`merge_fixed_joints` 和
`collision_from_visuals`；MJCF 只额外接受 `fix_base`。两种格式的 robot profile 都可以设置
`self_collision`；其默认值为 `false`，控制 articulation contact。Rigid object profile 不接受
`self_collision`。Isaac 5.1 的 `import_inertia_tensor`、`import_sites` 以及 MJCF
`merge_fixed_joints` 已不属于 Importer 3.0，配置校验会明确拒绝。

这些字段只影响 Isaac 物理接触。cuRobo 使用自己的 URDF/robot YAML 和 world description；修改
importer collision approximation 或 `self_collision` 不会更新 cuRobo collision sphere、
self-collision mask 或 planning geometry。

## 验证

修改 collision approximation 后：

- 运行 `tests/test_system_configs.py` 和 `tests/test_robot_loader_import_config.py`。
- 启动 GUI smoke test，检查生成的 collision prim。
- 确认 root pose 和 fixed-base 行为仍符合 scene 配置。
- 运行受影响 robot 或 object profile 的 motion test。
