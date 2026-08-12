# Mirror 相机

语言：[中文](cameras.md) | [English](../../en/guides/cameras.md)

相机是 Mirror-only 的被动输出能力。Kaleidoscope canonical 训练 scene/mode 不创建 renderer 或
camera；独立 viewport Kit 只能显示一个选中环境，仍排除 camera、SyntheticData、Replicator、录制与
图像 observation。视觉强化学习不在当前产品范围。

## 配置所有权

- `configs/scenes/mirror/scene3.yaml`（selector `mirror/scene3`）：camera prim、parent、pose、resolution、frequency、modality、clip 和 pixel pinhole 内参；
- `configs/outputs/mirror_default.yaml`：是否启用、编码/目录/live sink、队列和关闭策略；
- physics profile：是否具备 render consumer 所需同步能力。

Scene 中没有 camera 时，`outputs.camera.enabled=true` 会在启动前失败。Kaleidoscope scene 出现
camera/viewport 字段同样失败。

内参使用 OpenCV pinhole 约定，单位为 pixel：

```yaml
intrinsics:
  fx: 307.5
  fy: 308.0
  cx: 160.0
  cy: 120.0
```

`fx`、`fy` 必须大于零，四个字段必须同时提供。旧 scene 可省略整个 `intrinsics` mapping，
此时保留 Isaac 的默认 optical 参数；需要与真实相机标定对齐的 scene 应始终显式配置。

## Runtime 所有权

`MirrorRuntime` 拥有 `CameraBundle`，bundle 拥有 camera handle 与相关 sink。Physics manager 只负责
physics-to-USD sync，不注册、不缓存、不关闭 camera。关闭顺序是停止输出 admission → drain/close
camera 与 sink → 关闭 physics/session。

`RenderCoordinator` 拥有完整 render transaction。timeline 与会推进物理的 `hold_step` idle 先执行
`physics.step(render=False)`，再调用 `render_only()`；该方法只推进 renderer，不读取 camera。随后统一的
post-step observer 按频率与背压策略执行唯一一次 capture/publish，避免同一物理 tick 双 readback。

`idle_physics_policy: pause` 不推进物理，也不触发 post-step observer；到达 wall-clock 渲染周期时，owner
loop 显式调用 `MirrorRuntime.render()`，由 `render_frame(capture=True)` 立即返回当前帧。应用代码显式调用
`runtime.render()` 也使用相同的立即 capture 语义。

PhysX 每个 transaction 调用一次具体 runtime 的 `render()`；Newton 每帧只调用一次
`pre_render()`，在 owner stream 可见性边界发布一份不可变物理快照，然后按每个隐藏 camera product 的
预算调用多次纯 `render_update()`。多相机按 viewport 逐个激活，异常后也恢复全部激活状态；这些
renderer-only update 复用同一快照，不重复 D2H/USD 发布，也不推进物理时间。

## 数据与背压

Capture 在 Isaac owner thread 上读取 frame，编码/写盘可在有界 worker 中进行。每个 camera queue、
message bytes、目录 bytes 和 shutdown timeout 都必须有上限。Overflow 策略必须显式，不能扩成无界
队列。已有目录按 output policy 处理，不静默覆盖。

相机 live listener 只绑定 loopback，无认证/TLS。精确输出约定见[输出参考](../reference/outputs.md)。
