# 碰撞边界

语言：[中文](collision-models.md) | [English](../../en/guides/collision-models.md)

“Kaleidoscope 不做碰撞规划”不表示禁用物理接触。需要区分三类能力：

1. **物理碰撞**：所选 PhysX CUDA 或 Newton 后端的接触推动 T-block；
2. **跨环境隔离**：PhysX builder 固定用 env IDs，Newton builder 固定用独立 worlds，均阻止不同 env 接触；
3. **规划碰撞/避障**：cuRobo collision world、cache 和 obstacle sampling，只属于 Mirror。

Mirror 可以从 robot sphere、object primitive/mesh 与 stage provider 构建 planning scene，并在 request
边界刷新 fingerprint。Kaleidoscope config/runtime closure 禁止 import planning/collision backend，也不为
每个 env 分配 collision cache。

Mirror 的 `planning.request_defaults` 只决定默认避障与刷新策略；cuRobo planner 是否具备碰撞能力以及
cache 容量由 `curobo.motion_planner` 独立声明。Kaleidoscope 的可选 cuRobo profile 必须省略整个
`motion_planner`。

资产的物理 collision approximation 仍是共享基础事实，见
[碰撞近似](../development/collision-approximation.md)。

`mirror/scene3` scene（文件 `configs/scenes/mirror/scene3.yaml`，内部 `scene.id: scene3`）的仓库地板
展示了这条边界：原始共面视觉 mesh 只负责渲染，包装资产在同一
局部高度声明不可见解析 Plane 负责物理接触。场景因此保持 `add_ground: false`，避免再叠加一层
默认地面；PhysX 与 Newton CPU/CUDA 都消费同一个解析碰撞体。
