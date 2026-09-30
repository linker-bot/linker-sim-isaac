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

CPU PhysX 与 Newton 每帧只调用一次 `pre_render()` 发布当前物理位姿，然后调用
`render_update()`，期间物理时间保持不变。原生相机等待所属 render product 的完成事件，且其
Kit SWH 帧号必须不小于状态发布后的第一个 update。Newton 没有 SWH stage-update owner，
改用原生 product 帧号及 SyntheticData 有理数渲染时钟屏障；该时钟与物理时间严格区分。最多等待 50 次 renderer update；超时抛错，
不返回旧画面。Newton 保留至少四次 update 的 history 预算，多相机按 viewport 逐个激活，异常后
也恢复激活状态。配置频率控制输出采样；显式 `render()` 在暂停时也请求当前状态的新帧。

PhysX 启用渲染的 headless 模式保留活跃的 startup Hydra viewport，不要求可见窗口或相机机械
模型。Mirror PhysX Kit 关闭独立的 sensor/TLAS 时钟，使暂停位姿更新进入本次采集。关闭渲染时
仍禁止创建默认 viewport。

记录新增 `capture` 元数据：`native_frame_id`（`native_frame_source` 在 PhysX 为 `kit_swh`，
Newton 为 `kit_render_product_frame`）、
`render_product_path`、`render_snapshot_index` 和 `physics_time_s`。`frame_index` 仍是输出序号，
不是原生曝光身份；没有匹配渲染的物理步不能把旧像素标成新观测。原生帧号只在 Kit 进程内有效，
快照序号只在当前 coordinator 内有效。相机 handle 的 `get_capture_metadata()` 返回相同身份信息。

定向回归为 `scripts/smoke_mirror_camera.py`：不额外推进物理预热首帧、暂停 set/restore 物块、reset，
以及 120 Hz 物理下连续六个 60 Hz 仿真时刻的采样。可选择 `--gui`、
`--resolution 1920x1080`、`--cameras 3`、`--record-root <空目录>` 验证对应路径。
这些检查不代表墙钟吞吐达到 60 FPS。

## 数据与背压

Capture 在 Isaac owner thread 上读取 frame，编码/写盘可在有界 worker 中进行。每个 camera queue、
message bytes、目录 bytes 和 shutdown timeout 都必须有上限。Overflow 策略必须显式，不能扩成无界
队列。已有目录按 output policy 处理，不静默覆盖。

相机 live listener 只绑定 loopback，无认证/TLS。精确输出约定见[输出参考](../reference/outputs.md)。
