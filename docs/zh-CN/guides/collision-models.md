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

### Newton 的明确刚体接缝过滤

当前固定版本的 Newton USD importer 读取 collider 过滤，但不展开 MJCF importer 写出的
body 过滤。Simulator 在 prototype 复制前，将每个明确的 body 对只展开到这两个刚体实际
拥有的 shape；后代刚体仍可碰撞。PhysX 保留原 USD 关系。这样保留 AR5 link5–link7 等
必要接缝，不会关闭整手或相机碰撞，Mirror 与 Kaleidoscope 均使用这一处理。

## 当前 USD 解析器的临时处理

Kit 携带的 OpenUSD 25.11 在一个刚体有多个 mesh 时，`_FinalizeCollisionDescs` 会并行写同一 vector，
本轮分解碰撞资产触发了原生分配器崩溃。Isaac MJCF/URDF 转换和 Newton USD 解析阶段临时使用
一个 USD Work 线程，结束或异常后恢复原值，不改变导入后的物理、渲染或并行采样。此处理仅针对携带的 25.11 版本。

上游 [OpenUSD 修复 ed857d77c95b](https://github.com/PixarAnimationStudios/OpenUSD/commit/ed857d77c95b27dd9cf0919e8ed36366cf3301e6)
已改为线程安全收集再串行合并；携带的 runtime 升级到验证过的修正版后可移除临时处理。
此分配器崩溃与 Hydra `_MarkInstancerDirty` 诊断是不同问题。

Kit 的 `PXR_WORK_THREAD_LIMIT` 会锁住公开 Work setter。本版本的解析保护仅在固定
Linux / USD 25.11 上调用已加载 Kit 扩展导出的 `WorkImpl_SetConcurrencyLimit(unsigned)`，
并核验实际并行度及退出恢复。不会导入第二套 USD、修改二进制或环境变量；符号缺失或限制
未生效就拒绝继续不安全解析。升级到已验证修复版 USD 后移除此版本特定兼容处理。
