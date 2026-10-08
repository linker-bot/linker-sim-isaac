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

## 法兰、灵巧手与实体相机模块

`camera_workstation` 是可选 Mirror mode/scene 示例，物理频率 120 Hz。已有 L6/O6
左右 profile 默认加入共用法兰并启用自碰撞。选择 `ar5_08_{l6,o6}_gemini335l_{l,r}`
安装法兰、支架和 Gemini 335L；左右独立选配，改用仅法兰 profile 时移除对应相机条目。
关闭渲染不会移除实体硬件，没有只带支架的 profile。重力、增益和 armature 保持原值；
附件继承 default 关闭重力，但惯性仍生效。

静态对象 `workstation_zed2i` 加入顶部相机和支架。示例放在立柱居中安装基准；其他场景
需填写自己的实测根变换。光学 frame 和惯性采用标明来源的 CAD 标称配准/估计，并非设备标定。
旧的零偏移 `pinch_tcp` 仍是臂末端参考，本次装配修改不会自动把它修正成抓取点。

`parent_prim_path` 填机器人/对象实例根，`parent_link` 填导入后唯一的精确 link 名。
资产导入后解析层级，缺失/重名直接失败。`prim_path` 的末段作为新相机名，实际放到解析出的
link 下。`pose_axes: opencv` 表示局部 X 向右、Y 向下、Z 向前；默认 `world` 为 X 向前、
Y 向左、Z 向上。光学 frame 下的零 pose 无需额外旋转。采集元数据记录实际 parent、profile、
型号与标定来源；返回的 camera world pose 仍使用既有 world-camera 轴约定。

```yaml
- id: left_rgb
  camera_profile: gemini335l_rgb_60
  parent_prim_path: /World/Robots/left_arm
  parent_link: camera_rgb_optical
  prim_path: /World/Robots/left_arm/LeftRGB
  pose: {xyz: [0, 0, 0], rpy: [0, 0, 0]}
  pose_axes: opencv
```

相机 leaf 在 `configs/cameras` 单独维护。每个参数只有一个写入方，scene 不能静默覆盖
所选 leaf 的同名参数；需调整时复制/修改 leaf，或提供完整内联配置。catalog 记录所选来源。

| Profile | 原生分辨率 | Hz | 输出 |
| --- | --- | --- | --- |
| `gemini335l_rgb_60` | 1280×800 | 60 | RGB |
| `gemini335l_depth_30` | 1280×800 | 30 | depth |
| `gemini335l_depth_60` | 848×480 | 60 | depth |
| `zed2i_1080p_30` | 1920×1080 | 30 | RGB + 理想深度 |
| `zed2i_720p_60` | 1280×720 | 60 | RGB + 理想深度 |
| `development_1080p_60` | 1920×1080 | 60 | RGB + 理想深度 |

标称 pinhole FOV：Gemini RGB 94×68°、depth 90×65°，ZED 2i 2.1 mm 为 110×70°。
这些预设不模拟双目匹配、畸变、硬件曝光、MinZ 无效域或噪声；真实 MinZ 不是渲染 near plane。
Gemini RGB/depth 使用独立光学 frame，同尺寸不代表已对齐，真机需读取对应 profile 的 SDK 标定。
开发录制采用虚拟相机规格，不是传感器图放大。Hz 指不同仿真时刻的采样数，不保证墙钟 FPS。

当前 O6 + Gemini 装配的拇指伸直侧摆部分路径会碰到相机，符合实物。保留碰撞，并协调弯曲
与侧摆；弯曲不代表所有角度都安全，左手尤其如此。任务需验证完整路径，L6 的手部形状另行判断。

装配回归脚本为 `scripts/smoke_mirror_assemblies.py`，提供 `--hand`、
`--wrists none|left|right|both`、`--profile`、`--gui` 和空目录 `--record-root`。
检查原生图像尺寸、实体腕部随动、暂停恢复、reset 和未改变的重力开关。资产装配与碰撞依据
见资产仓库 `docs/camera-assemblies.zh-CN.md`。

回归还提供 `--depth-hz 30|60`、`--top-hz 30|60` 和 `--development-cameras`
（双腕配置下四台原生 1080p 相机），选择同一套 catalog leaf，不放大采集后的图像。

使用 `--record-root` 时，supervisor 在原生进程关闭后解码已经排空的图像/深度文件，检查
尺寸、配置采样周期、严格递增的原生帧 ID，以及采集时间与物理时间的一致性。
