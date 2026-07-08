# Isaac 碰撞近似配置

本项目暴露两个导入阶段碰撞字段：

```yaml
robot:
  import:
    collision_approximation: convex_decomposition
    self_collision: false

objects:
  - name: fixture
    import:
      collision_approximation: convex_decomposition
```

`collision_approximation` 同时适用于 robot 和 rigid object import；`self_collision` 只适用于
robot import，用于控制 Isaac/PhysX 是否在同一个 articulation 内部 link 之间生成自碰撞接触。
rigid object import 不接受 `self_collision`。

## 碰撞近似取值

当前支持的项目级 `collision_approximation` 取值：

| 取值 | Isaac importer 映射 | 含义 |
| --- | --- | --- |
| `convex_decomposition` | `convex_decomp = True` / `set_convex_decomp(True)` | 把非凸 mesh 分解成多个凸碰撞体。对凹形结构更贴合，但导入/cooking 更慢，碰撞体数量也可能更多。 |
| `convex_hull` | `convex_decomp = False` / `set_convex_decomp(False)` | 每个 collision mesh 生成一个凸包。更快、更简单，但会填平凹陷区域。 |

默认值是 `convex_decomposition`，保持此前 URDF 导入里硬编码的项目行为。

## 机器人自碰撞开关

`robot.import.self_collision` 默认值为 `false`。设置为 `true` 时：

- URDF importer 会写入 `import_config.self_collision = True`。
- MJCF importer 会调用 `import_config.set_self_collision(True)`。
- Isaac/PhysX 会为同一 articulation 内部允许的 link 对生成自碰撞接触。

这个开关只影响 Isaac 物理侧的真实接触生成，不会改变 cuMotion XRDF/URDF 中的 collision spheres
或 self-collision mask。若需要“哪些 link pair 可以/不可以自碰”的细粒度控制，应在资产或未来的
collision filter 配置层处理，而不是放进 `collision_approximation`。

## 不是重复一层凸包

这里容易混淆：**Importer 层**和 **USD/PhysX 层**不是两次独立的凸包生成需求。

更准确地说：

1. URDF/MJCF importer 读取原始资产，并根据 `convex_decomp` 这类导入配置，生成 USD stage 里的 collision prim 和对应的碰撞近似属性。
2. 导入完成后，这些 collision prim 已经存在于 USD stage 中。
3. PhysX 启动仿真或需要碰撞查询时，会读取 USD 上的碰撞 schema 和 `physics:approximation` 属性，把它们 cooking 成 PhysX 运行时真正使用的碰撞数据。

所以“USD/PhysX 层”主要是在说明：**导入结果最后会落到 USD 的 mesh collision 属性上，然后由 PhysX cooking 使用**。它不是说我们已经在 importer 里做了凸包，还必须再手动做第二遍凸包。

## Importer 支持范围

URDF importer：

- URDF importer UI 里把碰撞类型暴露为 `Convex Hull` 和 `Convex Decomposition`。
- 程序接口里对应 `ImportConfig.convex_decomp`：
  - `True` 表示 convex decomposition。
  - `False` 表示 convex hull。
- URDF importer 还有 `collision_from_visuals`，但本项目固定为 `False`，也就是优先使用资产里作者提供的 collision geometry，而不是 visual geometry。

MJCF importer：

- MJCF import config 提供 `set_convex_decomp(...)`。
- Isaac UI 源码里的注释也描述为：true 时把非凸 mesh 分解成凸碰撞形状，false 时使用凸包。
- 本项目现在用同一个 `collision_approximation` 字段显式设置它。

## USD/PhysX 支持的更多 token

导入完成后，USD stage 里的 mesh collider 由 `UsdPhysics.MeshCollisionAPI.physics:approximation` 描述碰撞近似方式。这个属性支持的 token 比 URDF/MJCF importer UI 暴露的选项更多：

| Token | 含义 |
| --- | --- |
| `none` | 三角网格碰撞体，不做近似。通常不适合动态刚体/articulation。 |
| `convexDecomposition` | 多个凸碰撞体。 |
| `convexHull` | 单个凸包碰撞体。 |
| `boundingSphere` | 包围球碰撞体。 |
| `boundingCube` | 最优拟合盒碰撞体。 |
| `meshSimplification` | 简化后的三角网格碰撞体。 |
| `sdf` | SDF 碰撞体。 |
| `sphereFill` | 球填充近似。 |

这些 token 适合未来做“导入后 USD collision override”时使用。但它们不全是 URDF/MJCF importer 的导入选项，所以当前 YAML 只保留 importer 层最明确、最常用的两个值：`convex_decomposition` 和 `convex_hull`。

## 参数情况

对 URDF/MJCF import 来说，当前暴露出来的凸近似基本是模式开关。在本项目使用的 Python import config 中，没有看到按资产配置的 VHACD 风格参数，例如：

- hull 数量
- voxel resolution
- concavity
- simplification error
- 每个 hull 的顶点上限

确实还有一些相关设置，但它们属于别的层：

- PhysX mesh cooking 有缓存和 cooking 行为设置，影响复用和性能，不等于 YAML 里的导入近似模式。
- SDF collider 有自己的 SDF 参数，例如 margin、narrow-band thickness、subgrid resolution、SDF resolution。
- deformable body cooking 也有自己的 mesh simplification 选项。

如果以后需要这些细粒度参数，建议另开一个“导入后 USD/PhysX collision override”配置，而不是把现在的 `collision_approximation` 扩成混合多层语义的大配置。

## 已检查来源

- 本地 Isaac Sim URDF importer 扩展：`<python-env>/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf`
- 本地 Isaac Sim MJCF importer 扩展：`<python-env>/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.mjcf`
- 本地 Isaac Core geometry prim 源码/文档：`GeomPrim.set_collision_approximations(...)`
- 本地 PhysX support UI 源码/文档：static/dynamic collider creation 相关实现
