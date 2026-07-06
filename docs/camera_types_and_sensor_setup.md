# Camera Types and Sensor Camera Setup

本文说明项目里两类容易混淆的 camera 语义，并记录当前仿真传感器摄像机的配置、运行时接入和输出约定。

## 术语边界

项目中应区分两类能力：

| 类型 | 推荐名称 | 当前位置 | 用途 | 是否产生图像数据 |
| --- | --- | --- | --- | --- |
| GUI 观察视角 | viewport view / GUI viewport | `visuals.viewport`、`visualization/viewport.py` | 让 Isaac GUI 打开后更容易观察机械臂、灵巧手和对象 | 否 |
| 仿真传感器摄像机 | sensor camera / RGB-D camera | `sensors.cameras`、`sensors/` | 作为场景中的传感器输出 RGB、depth、segmentation 等数据 | 是 |

GUI 观察视角只是调 `isaacsim.core.utils.viewports.set_camera_view`，修改当前 viewport 的 `eye` 和 `target`。它不改变物理世界、不参与控制、不影响规划、不写日志，也不应该作为算法输入。

仿真传感器摄像机是场景中的一个独立 sensor。它有自己的 prim path、位姿、分辨率、频率和输出 modality，数据可以被保存、发布到 Foxglove/MCAP，或交给视觉算法使用。

## 当前命名与迁移

旧版 env profile 曾经使用：

```yaml
visuals:
  camera:
    enabled: true
    eye: [1.35, -1.65, 1.05]
    target: [0.0, -0.1, 0.42]
    prim_path: /OmniverseKit_Persp
```

这段配置实际表达的是 GUI viewport 视角，不是传感器。当前 schema 已直接迁移为 `visuals.viewport`，并要求同步更新所有 env profile、配置解析、测试和文档。

迁移后的推荐结构：

```yaml
visuals:
  viewport:
    enabled: true
    eye: [1.35, -1.65, 1.05]
    target: [0.0, -0.1, 0.42]
    prim_path: /OmniverseKit_Persp
```

迁移原则：

- 当前版本只读取 `visuals.viewport`。
- 不再 fallback 读取 `visuals.camera`。
- 如果配置里仍出现 `visuals.camera`，应报出清晰错误，提示用户改成 `visuals.viewport`。

## 传感器摄像机配置

仿真传感器摄像机不放在 `visuals` 下。当前使用顶层 `sensors` 配置，把它和 GUI viewport 解耦。

`configs/envs/scene3.yaml` 中的 `world_rgbd` 是世界坐标系下的固定 RGB-D 相机：

```yaml
sensors:
  cameras:
    world_rgbd:
      enabled: true
      parent_prim_path: /World
      prim_path: /World/WorldRGBD
      pose:
        xyz: [0.08, 0.0, 0.08]
        rpy: [0.0, 1.1, 0.0]
      resolution: [640, 480]
      frequency: 30.0
      modalities: [rgb, depth]
      clipping_range: [0.01, 5.0]
      output:
        save_dir: logs/cameras/world_rgbd
        foxglove_topic_prefix: /cameras/world_rgbd
        foxglove_live_host: 127.0.0.1
        foxglove_live_port: 8766
        # foxglove_mcap_path: logs/cameras/world_rgbd.mcap
```

腕部或工具相机仍可用 `wrist_rgbd` 这类名字，但应把 `parent_prim_path` 指到对应 link，并让 `prim_path` 位于该 link 路径下。固定世界相机不要继续命名为 `wrist_*`，避免把坐标系语义混在一起。

字段语义：

| 字段 | 含义 |
| --- | --- |
| `enabled` | 是否创建并采样该摄像机。 |
| `prim_path` | 摄像机自身 USD prim path，必须是绝对路径；设置 `parent_prim_path` 时必须位于父 prim 下面。 |
| `parent_prim_path` | 可选父 prim。设置为 `/World` 时表示世界固定相机；设置为机器人 link 时表示腕部或工具相机。 |
| `pose.xyz` | 相对于父 prim 的平移；没有父 prim 时为世界坐标。单位 m。 |
| `pose.rpy` | 相对于父 prim 的姿态。单位 rad，沿用项目 RPY 约定。 |
| `resolution` | 图像宽高，例如 `[640, 480]`。 |
| `frequency` | 采样频率，单位 Hz。应小于或等于 GUI render frequency 或离屏渲染能力。 |
| `modalities` | 需要输出的数据类型，例如 `rgb`、`depth`、`semantic_segmentation`。 |
| `clipping_range` | 近远裁剪面，单位 m。 |
| `output.save_dir` | 可选离线图像输出目录。 |
| `output.foxglove_topic_prefix` | 可选实时或 MCAP topic 前缀。 |
| `output.foxglove_live_host` | 可选 Foxglove live server 监听地址。 |
| `output.foxglove_live_port` | 可选 Foxglove live server 监听端口；设置后发布 camera RawImage。这个字段来自 env profile，不是交互脚本的 `--foxglove-live-port`。 |
| `output.foxglove_mcap_path` | 可选 Foxglove MCAP 输出路径；设置后写入 camera RawImage。 |

## 运行时接入位置

传感器相机放在 `src/linkerbot_sim/sensors/`，不放进 `visualization/`：

```text
src/linkerbot_sim/sensors/
├── __init__.py
├── camera_config.py
├── camera_frame.py
├── camera_foxglove.py
├── camera_observer.py
├── camera_recorder.py
├── camera_runtime.py
```

职责建议：

- `camera_config.py`: 解析 `sensors.cameras`，只依赖标准 Python，不导入 Isaac。
- `camera_runtime.py`: 在 Isaac runtime 已启动后创建 camera prim/render product，并在主线程采样。
- `camera_frame.py`: 把 Isaac 输出规范化为 RGB `uint8` 或 depth `float32` frame，并处理 annotator 暂未产出图像的首帧情况。
- `camera_observer.py`: 在 world step 后按 camera `frequency` 触发采样。
- `camera_recorder.py`: 负责离线保存 RGB/depth/metadata。
- `camera_foxglove.py`: 把 RGB/depth frame 发布为 Foxglove `RawImage`，并发布 JSON info。

接入顺序建议：

1. 启动 `SimulationApp` 并创建 `World`。
2. 导入 robot/object runtime，使 `parent_prim_path` 指向的 link 或对象已经存在。
3. 创建传感器摄像机。
4. `world.reset()` 后初始化 camera wrapper 并挂载所需 annotator。
5. 在 physics step 或 render step 后按 `frequency` 采样。
6. 将数据写到磁盘、Foxglove/MCAP 或调用方提供的 callback。

如果摄像机挂在机器人腕部，必须在机器人导入后创建或绑定，并让 `prim_path` 位于对应 link 的 USD path 下；如果是固定场景相机，可以在对象导入后创建，便于检查它是否被桌面、工装或机器人遮挡。

## Headless 和 GUI 的关系

仿真传感器摄像机不应依赖 `--gui`。它应支持：

- GUI 模式：用户可以同时看到 viewport，传感器仍独立输出数据。
- Headless 模式：不显示窗口，但可以使用离屏渲染输出图像数据。

因此不要用 `gui` 作为是否创建传感器摄像机的唯一条件。当前 runtime 在存在 camera output 时会驱动 `world.step(render=True)`，即使没有 GUI 窗口，也给 Isaac camera annotator 一个产出图像的 render 更新机会。相机刚初始化后的空帧会被跳过，不会占用 frame index，也不会中断仿真。

## 数据输出建议

RGB 图像和 depth 图像都需要稳定的时间戳和帧号。推荐每帧至少包含：

| 字段 | 含义 |
| --- | --- |
| `camera_name` | 配置中的摄像机名称，例如 `world_rgbd`。 |
| `frame_index` | 从 0 开始递增的采样帧号。 |
| `simulation_step` | 对应的 physics step。 |
| `time_s` | 仿真时间，单位 s。 |
| `modality` | `rgb`、`depth` 等。 |
| `intrinsics` | 相机内参。 |
| `camera_position_world` | 采样时相机在世界坐标系下的位置。 |
| `camera_orientation_world` | 采样时相机在世界坐标系下的四元数姿态。 |

保存到磁盘时可以使用结构化目录：

```text
logs/cameras/world_rgbd/
├── metadata.jsonl
├── rgb/
│   └── 000000.ppm
└── depth/
    └── 000000.npy
```

发布到 Foxglove/MCAP 时，建议 topic 使用配置中的前缀，例如：

```text
/cameras/world_rgbd/rgb
/cameras/world_rgbd/depth
/cameras/world_rgbd/info
```

Foxglove 中 RGB 和 depth 是两个独立 Image topic：

- `/cameras/world_rgbd/rgb`: `RawImage`，encoding 为 `rgb8`。
- `/cameras/world_rgbd/depth`: `RawImage`，encoding 为 `32FC1`，显示时可能需要在 Image panel 里调 depth/color scale 的 min/max。
- `/cameras/world_rgbd/info`: JSON metadata，用于检查 frame index、shape、内参和相机世界 pose。

如果 topic 列表中能看到 `/cameras/world_rgbd/rgb`，但右侧图像仍是深度图，通常只是 Image panel 仍选中了 `/cameras/world_rgbd/depth`，需要在该面板顶部切换 topic。

## 不建议的做法

- 不要把 `visuals.camera` 当成传感器配置。
- 不要从 GUI viewport 截图作为算法输入；viewport 会被用户操作、窗口大小、UI 状态影响。
- 不要把传感器采样逻辑放进 `visualization/` 包；该包只服务调试显示。
- 不要在后台线程直接读取 Isaac stage、articulation 或渲染资源；项目现有状态流约定是在主线程采样后再交给后台输出。
- 不要让传感器频率隐式等于 physics frequency；图像渲染通常更贵，应有独立 `frequency`。

## 最小实现检查项

添加第一版仿真传感器摄像机时，至少检查：

- 配置解析能拒绝非法 `prim_path`、非法 `resolution`、非正 `frequency` 和未知 `modality`。
- GUI 观察视角关闭时，传感器摄像机仍可创建。
- Headless 模式下可以输出至少一帧 RGB 图像。
- Foxglove Image panel 可以分别显示 `/cameras/<name>/rgb` 和 `/cameras/<name>/depth`。
- 挂载到机器人 link 的相机在机器人运动后 pose 会随 link 更新。
- 输出帧包含 `simulation_step`、`time_s`、camera position/orientation 和内参。
- 单元测试覆盖纯配置解析；Isaac 相关行为用可选 smoke test 或手工验证命令记录在 PR 中。
