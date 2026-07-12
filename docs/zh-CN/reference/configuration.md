# YAML 配置参考

语言：[中文](configuration.md) | [English](../../en/reference/configuration.md)

本页是项目自有 YAML 的字段参考。人类或大模型无需从随仓示例猜测字段，即可据此编写
profile。所有固定 mapping 都拒绝未知 key；boolean 必须是严格 YAML boolean，数字字符串
不会被当作数字。除非某行明确写出 `null`，已经出现的字段必须符合所述类型。“缺省值”指
省略字段时 parser 产生的值，并不表示每份 profile 都要复制该字段。

空 YAML、非 mapping 文档以及任意嵌套层级的重复 key 会在领域解析前被拒绝，并报告文件与
重复声明位置。

## 所有权与解析顺序

| Owner | 位置 | 拥有 | 不拥有 |
| --- | --- | --- | --- |
| Runtime profile | `configs/runtime/<name>.yaml` | 入口模式、profile 选择、进程资源、执行、transport、规划策略、输出、telemetry 和关闭 | 场景拓扑、资产、controller gain、机器人模型资源 |
| Env profile | `configs/envs/<name>.yaml` 或 `<name>/base.yaml` | World 事实、visual、sensor、robot/object instance、tiled 布局 | robot/object 资产属性、controller gain、cuRobo 算法 |
| Per-env fragment | `configs/envs/<name>/<dir>/*.yaml` | 已有 tiled env 的 pose override 和不透明 metadata | 拓扑、资产、物理、controller、输出、规划 |
| Robot profile | `configs/robots/<name>.yaml` | 单个 articulation 的仿真资产、部件分组、物理和 cuRobo 模型绑定 | 场景 `prim_path`/`root_pose`、cuRobo 算法默认值 |
| Object profile | `configs/objects/<name>.yaml` | 对象资产、import、物理、规划碰撞和 dynamic-chain 摘要 | 场景 `prim_path`/`root_pose` |
| Controller bundle | `configs/controllers/<name>/` | arm/hand/default 控制方法、gain、limit 和 follower drive | 接触材质与 rigid-body damping |
| cuRobo profile | `configs/curobo/<name>.yaml` | device、seed、tolerance、cache capacity 和 planner 算法 | robot URDF、robot YAML、frame 和 TCP transform |
| Logging profile | `configs/logging/<name>.yaml` | Single Scene joint-tracking CSV 路径、节拍和列 | telemetry MCAP 与相机输出 |

Profile 引用是简单文件 stem，不是路径。Runtime 解析顺序是
`code defaults < selected runtime YAML < explicit entry-point CLI overrides`。
Controller bundle 选择顺序是 `runtime default < robot profile < env robot instance`。
选中的 cuRobo 算法 profile 是 base，各机器人自己的 `curobo.enabled`、
`planning_joint_group` 和 `robot` 模型绑定覆盖其上。Object profile 属性与 env instance
摆放不存在重叠字段。

## 校验器完整参数表

`scripts/validate_config.py` 是 pure-Python 预检入口，只接受以下参数：

| 参数 | Argparse 默认值 | 契约 |
| --- | --- | --- |
| `--help` | 不适用 | 输出 argparse help，不加载 profile，随后退出。 |
| `--runtime-profile NAME` | `default_single_scene` | 选择 `configs/runtime/` 下的文件 stem；不接受路径。 |
| `--dump-effective-config` | `false` | 成功时输出 resolved runtime 值及每个叶字段的来源，而不是最小摘要。 |

成功时向 stdout 写一个 JSON document 并返回 `0`。文件缺失、类型错误、未知字段或跨 profile
校验失败时向 stderr 写 `CONFIG_INVALID` 行并返回 `1`。该命令不启动 Isaac、不创建输出文件，
也不修改配置。

```bash
.venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
.venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene \
  --dump-effective-config
```

## Runtime Profile

文档根节点只能包含 `runtime` mapping。所有子段都可省略，省略的叶字段会按下表默认值递归
补齐。

### 模式、Profile 与 Simulation App

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `runtime.mode` | `single_scene | tiled_scene`；`single_scene` | 必须匹配入口。`tiled_scene` 还要求所选 env 的 `tiled.enabled: true`；`single_scene` 要求 false。 |
| `runtime.profiles.env` | profile stem；`scene1` | 选择单文件或目录型 env profile。 |
| `runtime.profiles.curobo` | profile stem；`default` | 合入每个启用规划的 robot 的算法 profile。 |
| `runtime.profiles.logging` | profile stem；`default_logger` | 始终参与完整 graph 校验；Single Scene runtime 用它创建 joint CSV。 |
| `runtime.profiles.controller_bundle` | bundle stem；`default` | 最低优先级 controller bundle。 |
| `runtime.simulation_app.gui` | boolean；`false` | `true` 启动交互窗口，`false` 为 headless。 |
| `runtime.simulation_app.gpu.multi_gpu` | boolean；`false` | 启动意图；校验不探测实际设备。 |
| `runtime.simulation_app.gpu.max_gpu_count` | 正整数；`1` | 两个 GPU index 的上界。 |
| `runtime.simulation_app.gpu.active_gpu` | 整数 >= 0；`0` | 必须小于 `max_gpu_count`。 |
| `runtime.simulation_app.gpu.physics_gpu` | 整数 >= 0；`0` | 必须小于 `max_gpu_count`。 |
| `runtime.simulation_app.render.gui_size` | `[width, height]`；`[1280, 720]` | 恰好两个正整数。 |
| `runtime.simulation_app.render.headless_size` | `[width, height]`；`[640, 480]` | 恰好两个正整数。 |
| `runtime.simulation_app.render.window_size` | `[width, height]`；`[1440, 900]` | 恰好两个正整数。 |
| `runtime.simulation_app.render.renderer` | 非空字符串；`RaytracedLighting` | Isaac renderer 名称。 |
| `runtime.simulation_app.render.anti_aliasing_gui` | 整数 >= 0；`3` | GUI 模式抗锯齿级别。 |
| `runtime.simulation_app.render.anti_aliasing_headless` | 整数 >= 0；`0` | Headless 模式抗锯齿级别。 |
| `runtime.simulation_app.render.samples_per_pixel_per_frame` | 正整数；`1` | 每帧每像素采样数。 |
| `runtime.simulation_app.render.denoiser` | boolean；`false` | Renderer denoiser 开关。 |
| `runtime.simulation_app.render.hide_ui` | boolean 或 `null`；`null` | `null` 由启动层根据 GUI/headless context 决定。 |
| `runtime.simulation_app.render.disable_viewport_updates` | boolean 或 `null`；`null` | `null` 委托 context-dependent 选择。 |
| `runtime.simulation_app.render.fast_shutdown` | boolean 或 `null`；`null` | `null` 委托 context-dependent 选择。 |
| `runtime.simulation_app.render.material_sync_loads` | boolean；`false` | Material 同步加载设置。 |
| `runtime.simulation_app.render.hydra_material_sync_loads` | boolean；`false` | Hydra material 同步加载设置。 |
| `runtime.simulation_app.render.headless_dt_policy` | `camera_aware | physics`；`camera_aware` | `camera_aware` 在启用的相机存在输出 consumer 时保留 render cadence；`physics` 让 headless 始终跟随 physics cadence。 |

详细的相机创建和渲染行为由[相机与输出](../guides/cameras.md)负责说明。

### 执行与交互 Transport

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `runtime.execution.control_mode` | `position | velocity | effort`；`position` | Tiled Scene 模式当前只接受 `position`。 |
| `runtime.execution.idle_physics_policy` | `pause | hold_step`；`hold_step` | 空闲时暂停或继续 hold step。 |
| `runtime.execution.idle_step_duration_s` | 正有限数；`0.05` | 一次空闲 hold interval 表示的时长。 |
| `runtime.execution.default_decimation` | 正整数；`2` | 命令省略 decimation 时的 physics-step 倍数。 |
| `runtime.execution.command_defaults.joint_interpolation` | `linear | smoothstep`；`smoothstep` | 仅在命令省略该字段时使用。 |
| `runtime.execution.command_defaults.pose_frame` | `env | world`；`env` | 缺省 task-space 参考系。 |
| `runtime.execution.command_defaults.orientation_mode` | `free | current | target`；`current` | 缺省 task-space 姿态处理。 |
| `runtime.interactive.stdin_enabled` | boolean；`true` | 启用 stdin command reader。 |
| `runtime.interactive.stdin_eof_policy` | `exit | keep_alive`；`exit` | stdin 关闭后的进程行为。 |
| `runtime.interactive.queue_poll_timeout_s` | 正有限数；`0.05` | 内部 command queue poll timeout。 |
| `runtime.interactive.snapshot_timeout_s` | 正有限数；`30.0` | Snapshot request 完成 timeout。 |
| `runtime.interactive.command_history_capacity` | 整数 >= 0；`256` | 内存 command history；0 禁用保留。 |
| `runtime.interactive.snapshot_request_capacity` | 正整数；`32` | Snapshot request queue 上限。 |
| `runtime.interactive.transport.tcp_jsonl.enabled` | boolean；`false` | 启用 control JSONL TCP listener。 |
| `runtime.interactive.transport.tcp_jsonl.host` | loopback host；`127.0.0.1` | 只接受 `localhost` 或数值 loopback 地址。 |
| `runtime.interactive.transport.tcp_jsonl.port` | `null` 或 1..65535 整数；`null` | Endpoint 启用时必填。 |
| `runtime.interactive.transport.websocket.enabled` | boolean；`false` | 启用 control WebSocket listener。 |
| `runtime.interactive.transport.websocket.host` | loopback host；`127.0.0.1` | 同样仅允许 loopback。 |
| `runtime.interactive.transport.websocket.port` | `null` 或 1..65535 整数；`null` | Endpoint 启用时必填。 |
| `runtime.interactive.transport.max_message_bytes` | 正整数；`1048576` | 单条输入消息上限。 |
| `runtime.interactive.transport.max_connections` | 正整数；`16` | 并发网络连接上限。 |
| `runtime.interactive.transport.request_queue_capacity` | 正整数；`256` | 已接收 request queue 上限。 |
| `runtime.interactive.transport.event_queue_capacity` | 正整数；`256` | 发送 event queue 上限。 |
| `runtime.interactive.transport.overflow_policy` | `reject`；`reject` | Queue 满时拒绝新工作。 |
| `runtime.interactive.transport.startup_timeout_s` | 正有限数；`5.0` | Listener 启动 timeout。 |
| `runtime.interactive.transport.server_poll_interval_s` | 正有限数；`0.1` | Server-side poll interval。 |
| `runtime.interactive.transport.response_poll_interval_s` | 正有限数；`0.5` | Response poll interval。 |

Listener endpoint 不提供认证或 TLS。远程 client 必须通过带认证的 TLS proxy 或 SSH tunnel；
直接绑定非 loopback 地址属于无效配置。消息结构由 [Single Scene JSON 协议](single-scene-json.md)和
[Tiled Scene JSON 协议](tiled-scene-json.md)负责说明。

### Planner 与 Playback

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `runtime.planner.backend` | `curobo | linear`；`curobo` | `linear` 不能满足 task-space 或 collision-aware request。 |
| `runtime.planner.joint_batch_mode` | `auto | per_env | batch_only`；`auto` | cuRobo joint planning dispatch 策略。 |
| `runtime.planner.request_defaults.duration_s` | 正有限数；`1.0` | 缺省运动时长。 |
| `runtime.planner.request_defaults.avoid_collisions` | boolean；`false` | 不能与 `linear` backend 的 true 组合。 |
| `runtime.planner.request_defaults.force_collision_refresh` | boolean；`false` | Tiled Scene 模式不支持。 |
| `runtime.planner.request_defaults.coordination` | `independent | static_others | coupled`；`independent` | 没有 coupled backend，故 `coupled` 会被拒绝；tiled 必须为 `independent`。 |
| `runtime.planner.request_defaults.load_on_success` | boolean；`true` | 成功后把 trajectory 装入 playback。 |
| `runtime.planner.request_defaults.replace` | boolean；`true` | 装入时替换现有 queued trajectory。 |
| `runtime.planner.oversize_request_policy` | `split | reject`；`split` | Request 超出 batch 上限时的行为。 |
| `runtime.planner.failure_policy` | `hold_failed_env | reject_request`；`hold_failed_env` | 部分 env 失败时的原子 response 行为。 |
| `runtime.planner.resources.max_workers` | 正整数；`2` | Async planning worker 上限。 |
| `runtime.planner.resources.max_pending_requests` | 正整数；`64` | Pending request 上限。 |
| `runtime.planner.resources.max_completed_results` | 整数 >= 0；`256` | Completed-result 保留上限；0 禁用保留。 |
| `runtime.planner.resources.max_batch_problems` | 正整数或 `auto`；`64` | `auto` 解析为 env 数，并受所选 cuRobo capacity 限制。显式 cuRobo 值不得超过 IK/planner 两个 `max_batch_size` 中较小者。 |
| `runtime.planner.resources.shutdown_timeout_s` | 正有限数；`30.0` | Planner worker join timeout。 |
| `runtime.playback.max_queue_depth_per_env` | 正整数；`32` | 每 env trajectory queue 深度。 |
| `runtime.playback.max_samples_per_env` | 正整数；`100000` | 每 env queued sample 上限。 |
| `runtime.playback.max_duration_s_per_env` | 正有限数；`3600.0` | 每 env queued duration 上限。 |
| `runtime.playback.overflow_policy` | `reject`；`reject` | 超过任意 playback 上限时拒绝工作。 |

Planner capability 与 request 语义由[运动规划](../guides/motion-planning.md)负责说明。

### 相机输出、Telemetry、路径与关闭

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `runtime.camera_output.queue_size` | 正整数；`128` | 共享 async camera publication queue 上限。 |
| `runtime.camera_output.overflow_policy` | `drop_oldest | drop_newest | block | error`；`block` | Dataset writer 通常应使用无损 `block` 或 `error`。 |
| `runtime.camera_output.worker_poll_interval_s` | 正有限数；`0.1` | Publisher worker poll interval。 |
| `runtime.camera_output.existing_data_policy` | `error | truncate | resume | timestamped_dir`；`error` | 已有相机目录策略。 |
| `runtime.camera_output.shutdown_policy` | `drain | abort`；`drain` | 关闭时 flush 或丢弃 pending frame。 |
| `runtime.camera_output.rgb_format` | `ppm | png | npy`；`ppm` | RGB payload 编码。 |
| `runtime.camera_output.depth_format` | `npy | npz`；`npy` | Depth payload 编码。 |
| `runtime.camera_output.metadata_flush_interval_frames` | 正整数；`1` | Metadata flush cadence。 |
| `runtime.camera_output.max_bytes_per_camera` | 正整数；`10737418240` | 单相机目录配额，包含 metadata 与所有 modality。 |
| `runtime.telemetry.primary_env_id` | 整数 >= 0；`0` | 标准单 env topic 的来源。Tiled Scene runtime profile 必须显式声明。 |
| `runtime.telemetry.selected_env_ids` | 非空、唯一整数列表；`[0]` | 值 >= 0。Tiled Scene runtime profile 必须显式声明，必须包含 `primary_env_id` 且小于 `tiled.num_envs`；Single Scene 必须为 `[0]`。 |
| `runtime.telemetry.publish_decimation` | 正整数；`1` | Tiled Scene global-step publication decimation；Single Scene 必须为 `1`。 |
| `runtime.telemetry.rate_hz` | 有限数 >= 0；`60.0` | Telemetry sampling rate。 |
| `runtime.telemetry.buffer_size` | 正整数；`1` | State stream buffer 上限。 |
| `runtime.telemetry.drop_policy` | `latest | drop_oldest | drop_newest`；`latest` | Buffer overflow 行为。 |
| `runtime.telemetry.on_error` | `stop | continue`；`stop` | Publisher error 行为。 |
| `runtime.telemetry.include_joint_states` | boolean；`true` | 包含标准 joint state message。 |
| `runtime.telemetry.include_state_json` | boolean；`true` | 包含项目 state JSON。 |
| `runtime.telemetry.include_scene_markers` | boolean；`false` | 包含 scene marker message。 |
| `runtime.telemetry.include_efforts` | boolean；`false` | 读取并包含 effort。 |
| `runtime.telemetry.include_objects` | boolean；`false` | 包含 runtime object state。 |
| `runtime.telemetry.joint_effort_field` | `none | commanded | measured | applied`；`none` | 非 `none` 值要求 `include_efforts: true`，且只支持 Single Scene。 |
| `runtime.telemetry.topics.joint_states` | absolute topic；`/joint_states` | 必须以 `/` 开头，不含 `//` 或 `..`，且与另外两个 topic 不同。 |
| `runtime.telemetry.topics.scene` | absolute topic；`/scene` | 同一 topic 规则。 |
| `runtime.telemetry.topics.state` | absolute topic；`/linkerbot/state` | 同一 topic 规则。 |
| `runtime.telemetry.mcap.path` | 路径字符串或 `null`；`null` | `null` 禁用 sink；路径不能含 NUL 或 `..` component。 |
| `runtime.telemetry.foxglove_live.enabled` | boolean；`false` | 启用 telemetry live server。 |
| `runtime.telemetry.foxglove_live.host` | loopback host；`127.0.0.1` | 只允许 loopback。 |
| `runtime.telemetry.foxglove_live.port` | `null` 或 1..65535 整数；`null` | 启用时必填。 |
| `runtime.output.csv_existing_file_policy` | `error | truncate | resume | timestamped_dir`；`error` | 已有 joint CSV 策略。 |
| `runtime.output.mcap_existing_file_policy` | 同一 enum；`error` | 已有 telemetry MCAP 策略。 |
| `runtime.paths.cache_root` | 路径字符串或 `null`；`null` | `null` 委托 cache-root 选择；非 null 路径不能含 NUL 或 `..`。相对路径使用进程 cwd。 |
| `runtime.shutdown.state_publisher_timeout_s` | 正有限数；`2.0` | State publisher join timeout。 |
| `runtime.shutdown.camera_publisher_timeout_s` | 正有限数；`2.0` | Camera publisher join timeout。 |
| `runtime.shutdown.transport_timeout_s` | 正有限数；`2.0` | Interactive transport join timeout。 |

配置 telemetry live 或 MCAP sink 时，joint states、state JSON、scene markers 至少启用一种。
Single Scene 模式下，已配置 sink 且启用 scene markers 时还要求 `include_objects: true`。详细 payload、
buffer、sink、resume 和文件行为由[实时状态流](../guides/telemetry.md)负责说明。

## Env Profile

Env profile 只接受并列顶层 key：`env`、`solver`、`visuals`、`sensors`、`robots`、
`objects`、`tiled`。`env` 和非空 `robots` list 必填。`objects` 可省略或为 `null`；其余已
出现 section 必须是声明类型的 mapping 或 list。

### World 与 Visual 字段

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `env.name` | 必填非空字符串 | 稳定 scene 名称。 |
| `env.description` | 可选字符串 | 仅用于人工/日志 context。 |
| `env.gravity_z` | 有限数；`-9.81` | World Z 重力，单位 m/s^2。 |
| `env.add_ground` | boolean；`true` | 添加 Isaac 默认地面。 |
| `env.ground_height` | 有限数；`0.0` | 默认地面 Z 坐标。 |
| `env.physics_frequency` | 正有限数；`600.0` | 每秒 physics step 数。 |
| `env.render_frequency` | 正有限数；`100.0` | 需要渲染时每秒 render frame 数。 |
| `solver.type` | `PGS | TGS | null`；`null` | Single Scene 级 PhysX solver override。Robot iteration 不放这里。 |
| `visuals.viewport.enabled` | boolean；`true` | 启用时配置 GUI viewport。 |
| `visuals.viewport.eye` | 有限 `[x, y, z]`；`[1.35, -1.65, 1.05]` | Viewport eye position。 |
| `visuals.viewport.target` | 有限 `[x, y, z]`；`[0.0, -0.1, 0.42]` | Viewport look-at target。 |
| `visuals.viewport.prim_path` | absolute USD path；`/OmniverseKit_Persp` | Viewport camera prim。 |
| `visuals.lights.key.enabled` | boolean；`true` | Distant key-light 开关。 |
| `visuals.lights.key.path` | absolute USD path；`/World/KeyLight` | Key-light prim。 |
| `visuals.lights.key.intensity` | 有限数 >= 0；`1200.0` | Light intensity。 |
| `visuals.lights.key.angle` | 有限数 >= 0；`0.5` | Distant-light angular size。 |
| `visuals.lights.key.color` | 有限 RGB triple 或 `null`；`null` | `null` 保留 light 默认值。 |
| `visuals.lights.key.rotation_rpy` | 有限 RPY triple 或 `null`；`null` | 单位 rad。 |
| `visuals.lights.fill.enabled` | boolean；`true` | Dome fill-light 开关。 |
| `visuals.lights.fill.path` | absolute USD path；`/World/FillLight` | Fill-light prim。 |
| `visuals.lights.fill.intensity` | 有限数 >= 0；`250.0` | Light intensity。 |
| `visuals.lights.fill.color` | 有限 RGB triple 或 `null`；`null` | `null` 保留 light 默认值。 |

### Sensor Camera

`sensors` 只接受 `cameras`；`sensors.cameras` 是 mapping，其任意 key 是 camera 名称。名称
必须非空且不能含路径分隔符。

| `sensors.cameras.<name>` 下路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `enabled` | boolean；`true` | Disabled camera 不创建 runtime resource。 |
| `prim_path` | 必填 absolute USD path | Camera prim template。 |
| `parent_prim_path` | absolute USD path 或 `null`；`null` | 设置时 `prim_path` 必须位于其下。 |
| `pose.xyz`、`pose.rpy` | 有限 triple；零 triple | 局部 pose，单位 m/rad。`pose: null` 等同省略。 |
| `resolution` | 两个正整数；`[640, 480]` | `[width, height]`。 |
| `frequency` | 正有限数；`30.0` | Capture frequency，单位 Hz。 |
| `env_ids` | 非空、唯一的整数列表 >= 0；省略 | Single Scene 所选 env profile 必须省略；启用 `tiled` 的 env profile 中每个 camera 必须显式提供，且值小于 `tiled.num_envs`。 |
| `modalities` | 非空、唯一列表；`[rgb]` | 可选 `rgb`、`depth`、`semantic_segmentation`、`instance_segmentation`。 |
| `clipping_range` | 有限 `[near, far]`；`[0.01, 5.0]` | 必须满足 `0 < near < far`。 |
| `intrinsics.fx`、`intrinsics.fy` | `intrinsics` 出现时为必填正有限数 | 像素焦距。整个 section 可省略或为 `null`。 |
| `intrinsics.cx`、`intrinsics.cy` | `intrinsics` 出现时为必填有限数 | 像素主点。 |
| `output.save_dir` | 非空字符串或 `null`；`null` | Camera dataset 目录。 |
| `output.foxglove_topic_prefix` | absolute topic prefix 或 `null`；`null` | 出现时必须以 `/` 开头。 |
| `output.foxglove_live_host` | loopback host；`127.0.0.1` | Camera live server bind host。 |
| `output.foxglove_live_port` | 正整数或 `null`；`null` | 非 null 时启用 camera live consumer。 |
| `output.foxglove_mcap_path` | 非空字符串或 `null`；`null` | 非 null 时启用 camera MCAP output。 |

Tiled Scene 模式只为 `env_ids` 展开 template。Save directory 与 topic prefix 会增加 `env_NNN`
后缀。Per-env camera pose override 只允许作用于 camera `env_ids` 中的 env。每个 camera 解析后的
`save_dir` 必须唯一。只要任一 camera 写目录或 MCAP，runtime `overflow_policy` 就必须为
`block` 或 `error`；有损策略只允许 live-only 输出。Camera MCAP 还拒绝
`existing_data_policy: resume`。创建、cadence、编码与 resume 行为见
[相机与输出](../guides/cameras.md)。

### Robot 与 Object Instance

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `robots[].robot_profile` | 必填 profile stem | 选择 `configs/robots/<name>.yaml`。 |
| `robots[].label` | 匹配 `[A-Za-z0-9_]+` 的字符串；`<robot_profile>_<list-index>` | 必须唯一。List 顺序生成 session-local 连续 `robot_id`；禁止配置 `robot_id`。 |
| `robots[].prim_path` | canonical absolute USD path；`/World/Robots/<label>` | 必须唯一，并与所有 robot/object instance 子树互不包含。 |
| `robots[].controller_profile` | bundle stem 或 `null`；`null` | 最高优先级 controller bundle。 |
| `robots[].root_pose.xyz`, `robots[].root_pose.rpy` | 必填 mapping；省略 vector 为零 triple | 有限 world pose，单位 m/rad。 |
| `objects[].name` | 必填 `[A-Za-z_][A-Za-z0-9_]*` | 稳定 scene identity；唯一。 |
| `objects[].object_profile` | 必填 profile stem | 选择 `configs/objects/<name>.yaml`。 |
| `objects[].runtime_handle` | 非空字符串或 `null`；`null` | 可选交互 alias；唯一，且不能与另一对象 name 冲突。 |
| `objects[].prim_path` | canonical absolute USD path；`/World/Objects/<name>` | 必须唯一，并与所有 instance 子树互不包含。 |
| `objects[].root_pose.xyz`, `objects[].root_pose.rpy` | 必填 mapping；省略 vector 为零 triple | 有限 world pose，单位 m/rad。 |

Single Scene 摆放永远不属于 robot 或 object profile。

### Tiled Base

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `tiled.enabled` | boolean；`false` | 目录型加载会把 effective 值强制为 `true`。必须与 `runtime.mode` 一致。 |
| `tiled.num_envs` | 正整数；`1` | 唯一有效 env 数 owner。仅当 base 省略且 fragment 存在时，目录 loader 派生 `max(env_id)+1`。 |
| `tiled.base_env_path` | absolute USD path；`/World/envs` | 不能为 `/` 或包含 `//`。 |
| `tiled.env_prefix` | 非空字符串；`env` | 不能含 `/`；root 形如 `<base>/<prefix>_<id>`。 |
| `tiled.spacing` | 正有限数；`2.0` | XY 网格间距。 |
| `tiled.num_per_row` | 正整数或 `null`；`null` | `null` 使用 `ceil(sqrt(num_envs))`。 |
| `tiled.per_env_config_dir` | 安全相对目录或 `null`；`null` | 目录 loader 省略时使用 `envs`；拒绝绝对路径、`.`、`..`。 |
| `tiled.per_env` | per-env row sequence；`[]` | 单文件 profile 可直接内联下述 row schema。目录 loader 会用 fragment 目录生成的 row 替换该值。 |
| `tiled.layout.origin_xyz` | 有限 triple；`[0, 0, 0]` | Tiled Scene 网格的 world translation。 |
| `tiled.clone.replicate_physics` | boolean；`true` | 请求 PhysX structure replication。 |
| `tiled.clone.copy_from_source` | boolean；`false` | GridCloner copy/inherit 行为。 |
| `tiled.clone.enable_env_ids` | boolean；`false` | GridCloner env-ID authoring。 |
| `tiled.clone.filter_collisions` | boolean；`true` | 启用 env 间 collision filtering。 |
| `tiled.clone.collision_filter_strategy` | `collision_groups | filtered_pairs`；`collision_groups` | `collision_groups` 是线性 authoring 路径；`filtered_pairs` 为两两配置。 |
| `tiled.clone.collision_root_path` | absolute USD path；`/World/collisions` | 不能为 `/`。非默认 collision-group 字段要求启用 filtering 且使用 `collision_groups`。 |
| `tiled.clone.physics_scene_path` | absolute USD path 或 `null`；`null` | `null` 自动发现唯一 PhysicsScene，不能写字符串 `auto`。显式设置要求启用 `collision_groups` filtering。 |
| `tiled.clone.global_collision_paths` | `auto` 或 absolute-path list；`auto` | 显式 path 取代标准地面自动发现，并要求启用 `collision_groups` filtering。 |
| `tiled.clone.extra_global_collision_paths` | absolute-path list；`[]` | 追加到自动/显式 global 后；非空时要求启用 `collision_groups` filtering。 |
| `tiled.diagnostics.inspect_env_ids` | 唯一整数列表；`[0]` | 每个值必须位于 `[0, num_envs)`，只影响诊断。 |

### 目录型 Profile 与 Per-env Override

目录型 profile 由一份共享拓扑和可选 fragment 组成：

```text
configs/envs/<name>/base.yaml
configs/envs/<name>/envs/env_000.yaml
configs/envs/<name>/envs/env_001.yaml
```

单文件 profile 可直接把下述 row schema 写在 `tiled.per_env`。目录型 profile 应使用
fragment 文件：loader 读取 `per_env_config_dir`，按 `env_id` 排序 fragment，并用生成的 row
替换 base 中可能存在的 `tiled.per_env`，再执行完整跨字段校验。Base 中显式
`tiled.num_envs` 始终优先，每个 fragment ID 都必须落在该范围内。

| Fragment 路径 | 类型与要求 | 规则 |
| --- | --- | --- |
| `env_id` | 必填整数 >= 0 | 唯一且小于 effective `tiled.num_envs`。 |
| `robots.<label>.root_pose.xyz/rpy` | 两者均为必填有限 triple | Label 必须已存在于 base `robots`；pose 为 env-local。 |
| `objects.<name>.root_pose.xyz/rpy` | 有限 triple；省略 vector 变为零 | Name 必须已存在于 base `objects`。建议两者都写，避免省略分量被置零。 |
| `cameras.<name>.pose.xyz/rpy` | 有限 triple；省略 vector 变为零 | Camera 必须存在于 base，且 `env_id` 必须在其 `env_ids` 中。 |
| `metadata` | JSON-compatible mapping；`{}` | 只允许字符串 key 与有限 JSON scalar/list/object；不解释为 runtime 配置。 |

Fragment 不能新增拓扑，也不能覆盖资产、物理、controller、输出或 planner 设置。

## Robot Profile

文档根节点只接受 `robot`、`curobo`、`joint_groups`、可选 `rigid_body_groups` 和可选
`controlled_joints`。

### 仿真资产与物理

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `robot.kind` | 必填 `arm | hand | arm_hand` | 要求对应非空 joint group。 |
| `robot.name` | 非空字符串；`robot` | 逻辑 profile 名；env instance label 会成为 runtime articulation name。 |
| `robot.controller_profile` | bundle stem 或 `null`；`null` | 中优先级 controller bundle。 |
| `robot.asset_type` | `mjcf | urdf`；`mjcf` | 仿真 importer 类型。 |
| `robot.asset_path` | 必填非空路径字符串 | 仓库相对或绝对 asset path。 |
| `robot.urdf_drive_type` | `none | position`；`position` | 仅 URDF robot 合法。 |
| `robot.import.collision_approximation` | `convex_decomposition | convex_hull`；`convex_decomposition` | Importer collision geometry。 |
| `robot.import.self_collision` | boolean；`false` | Robot-only articulation self-collision 开关。 |
| `robot.import.fix_base` | boolean 或 `null`；`null` | `null` 产生当前 robot importer 默认值 `true`。 |
| `robot.import.merge_fixed_joints` | boolean 或 `null`；`null` | MJCF effective 默认 `false`，URDF 为 `true`。 |
| `robot.import.import_inertia_tensor` | boolean；`true` | MJCF 与 URDF 均支持。 |
| `robot.import.import_sites` | boolean；`true` | 仅 MJCF。 |
| `robot.import.collision_from_visuals` | boolean；`false` | 仅 URDF。 |
| `robot.physics.gravity.default` | boolean；`false` | Rigid-body gravity fallback。 |
| `robot.physics.gravity.arm`, `robot.physics.gravity.hand` | boolean 或省略；继承 `default` | Per-component gravity。显式 `null` 不合法。 |
| `robot.physics.solver.arm.position_iterations`, `robot.physics.solver.arm.velocity_iterations` | 整数 >= 0 或省略 | Arm rigid-body PhysX override。 |
| `robot.physics.solver.hand.position_iterations`, `robot.physics.solver.hand.velocity_iterations` | 整数 >= 0 或省略 | Hand rigid-body PhysX override。 |
| `robot.planning_collision.spheres[]` | section 出现时为非空 list | Backend-neutral robot-root 保守包络。 |
| `...spheres[].name` | 非空字符串；`sphere_<index>` | List 内唯一。 |
| `...spheres[].center` | 必填有限 triple | Robot-root-local，单位 m。 |
| `...spheres[].radius` | 必填正有限数 | 单位 m。 |

`robot.physics.physx` 接受通用 `material`/`rigid_body` 字段，以及具有相同两个子 mapping 的
可选 `default`、`arm`、`hand`。应用顺序为 common、`default`、component override。

| 通用/component mapping 下 PhysX 叶字段 | 类型与省略语义 | 规则 |
| --- | --- | --- |
| `material` | mapping、`null` 或 `preserve`；省略为 inherit | `null`/`preserve` 明确保留 asset material binding；mapping 选择 override。 |
| `material.contact_static_friction` | 有限数 >= 0 或省略 | Robot contact material override。 |
| `material.contact_dynamic_friction` | 有限数 >= 0 或省略 | Robot contact material override。 |
| `material.contact_restitution` | `[0, 1]` 内有限数或省略 | Robot contact material override。 |
| `material.friction_combine_mode` | `average | min | multiply | max | preserve | null`；mapping 缺省 `average` | `preserve`/`null` 不改 combine mode。 |
| `rigid_body.linear_damping` | 有限数 >= 0 或省略 | Rigid-body damping override。 |
| `rigid_body.angular_damping` | 有限数 >= 0 或省略 | Rigid-body damping override。 |

### 部件分组与控制选择

| 路径 | 类型与缺省值 | 规则 |
| --- | --- | --- |
| `joint_groups.arm` | 精确名称 list；`[]` | 仅且必须在 `kind` 含 arm 时非空。 |
| `joint_groups.hand` | 精确名称 list；`[]` | 仅且必须在 `kind` 含 hand 时非空。 |
| `joint_groups.passive` | 精确名称 list；`[]` | 明确不由 arm/hand controller 写入的 command-space joint。 |
| `rigid_body_groups.arm`, `rigid_body_groups.hand`, `rigid_body_groups.default` | 精确名称 list；mapping 可省略 | 可选显式 component classification。 |
| `controlled_joints` | 非空精确名称 list；`[all]` | `[all]` 必须单独出现；否则每个名称都必须在 arm/hand group。 |

名称不能在一个或多个 component group 中重复。Group 顺序定义 command-space 顺序。Asset
finalization 会对 articulation 核对名称；planning active joints 必须与 `joint_groups.arm` 完全一致。

### Robot cuRobo 模型绑定

`curobo.enabled` 必填。值为 `false` 时，它必须是该 section 唯一 key。值为 `true` 时，必须
提供 `planning_joint_group: arm` 和非空 `robot` mapping；hand-only robot 不能启用规划。

| `curobo.robot` 下路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `robot_config_path` | 路径字符串或 `null`；`null` | 完整 cuRobo robot YAML；它与 `urdf_path` 至少提供一个。 |
| `urdf_path` | 路径字符串或 `null`；`null` | 规划 URDF。 |
| `base_link` | 非空字符串或 `null`；推断 | 使用 URDF 时，省略要求能推断出唯一 root link。 |
| `flange_frame` | 非空字符串或 `null`；`null` | Custom TCP 的第一默认 parent。 |
| `tool_frames` | 字符串 list；`[]` | 模型中精确 frame，不得重复。 |
| `default_tcp_frame` | 非空字符串或 `null`；`null` | `tool_frames` 与该字段至少提供一个。 |
| `custom_tcps` | 命名 mapping 或 frame list；`[]` | Context 创建前 materialize 的 fixed frame。 |
| `custom_tcps.<name>.parent_frame` | 非空字符串或省略 | 依次默认 `flange_frame`、`default_tcp_frame`；两者都没有时必填。 |
| `custom_tcps.<name>.xyz`, `custom_tcps.<name>.rpy` | 有限 triple；零 triple | Parent-local，单位 m/rad。List 形态的每项还要求 `frame_name`。 |
| `load_collision_spheres` | boolean；`true` | 存在时从 robot config 加载 sphere 数据。 |

## Object Profile

根节点只能包含 `object`。Instance `prim_path` 与 `root_pose` 在这里不合法。

| 路径 | 类型与缺省值 | 规则与含义 |
| --- | --- | --- |
| `object.name` | 非空字符串；profile stem | 逻辑 asset 名。 |
| `object.kind` | 必填 `rigid | dynamic_chain` | 选择严格 consumer schema。 |
| `object.source` | 必填 `usd | urdf` | `dynamic_chain` 要求 `usd`。 |
| `object.asset_path` | 必填非空路径字符串 | 仓库相对或绝对 asset path。 |
| `object.root_path` | absolute USD path 或 `null`；`null` | 仅 `dynamic_chain`；当前 capsule-rope consumer 默认 `/CapsuleRope`。 |
| `object.urdf_drive_type` | `none | position`；`none` | 仅 rigid URDF。 |
| `object.state_summary.reference_body` | `dynamic_chain` 必填非空 body name | 是名称而不是 prim path；rigid object 禁止该字段。 |

Rigid URDF 接受 `object.import`，格式字段与 robot importer 相同但没有 `self_collision`。Rigid
USD 不接受 `import`。Rigid object 省略 `fix_base` 时跟随 `physics.static`；显式
`fix_base: true` 与 `physics.static: false` 冲突。

| Rigid 路径 | 类型与缺省值 | 规则 |
| --- | --- | --- |
| `object.physics.static` | boolean；`false` | 冻结/固定 rigid object。 |
| `object.physics.material.static_friction` | 有限数 >= 0 或省略 | Object material override。 |
| `object.physics.material.dynamic_friction` | 有限数 >= 0 或省略 | Object material override。 |
| `object.physics.material.restitution` | `[0, 1]` 内有限数或省略 | Object material override。 |
| `object.physics.material.friction_combine_mode` | `average | min | multiply | max` 或省略 | Object material override。 |
| `object.planning_collision.shape` | 必填 `cuboid | sphere | capsule` | 简化规划 geometry，不改变 PhysX collider。 |
| `object.planning_collision.size` | 必填正数 list | Cuboid 长度 3；sphere 长度 1（radius）；capsule 长度 2（`radius`, `length`）。 |
| `object.planning_collision.xyz`, `object.planning_collision.rpy` | 有限 triple；零 triple | Object-local collision pose。 |
| `object.planning_collision.enabled` | boolean；`true` | 是否进入 planning world。 |
| `object.planning_collision.padding` | 有限数 >= 0；`0.0` | 保守 shape padding。 |

当前 `dynamic_chain` consumer 是生成好的 capsule rope。它禁止 `import`、
`planning_collision` 和 `urdf_drive_type`。

| Dynamic-chain 路径 | 类型与省略语义 | 规则 |
| --- | --- | --- |
| `object.physics.material.static_friction` | 有限数 >= 0 或省略 | Runtime USD material override。 |
| `object.physics.material.dynamic_friction` | 有限数 >= 0 或省略 | Runtime USD material override。 |
| `object.physics.material.restitution` | `[0, 1]` 内有限数或省略 | Runtime USD material override。 |
| `object.physics.material.friction_combine_mode` | `average | min | multiply | max` 或省略 | Runtime USD material override。 |
| `object.physics.solver_position_iterations` | 正整数或省略 | Runtime rigid-body position-iteration override。 |
| `object.physics.solver_velocity_iterations` | 整数 >= 0 或省略 | Runtime rigid-body velocity-iteration override。 |

资产生成与 runtime 所有权见[对象资产](../development/object-assets.md)；碰撞行为由
[碰撞模型](../guides/collision-models.md)负责说明。

## Controller Bundle

Bundle 目录必须包含 `arm_controller.yaml` 和 `hand_controller.yaml`，可选
`default_controller.yaml`。每个文件采用相同结构。`target` 缺省为文件角色；显式提供时
必须分别等于 `arm`、`hand` 或 `default`。Bundle 名匹配
`[A-Za-z0-9][A-Za-z0-9_-]*`。

`position_control`、`velocity_control`、`effort_control` 都只接受 `method`、
`active_joints`、`follower_joints`。Bundle 加载时会解析全部三种模式，而不只是当前 runtime
模式。

| 路径 pattern | 类型与缺省值 | 规则 |
| --- | --- | --- |
| `position_control.method` | `implicit | explicit`；`implicit` | PhysX drive 或 Python PD effort。 |
| `velocity_control.method` | `implicit | explicit`；`implicit` | PhysX velocity drive 或 Python velocity-error effort。 |
| `effort_control.method` | `direct`；`direct` | 直接有界 effort。 |
| `<mode>.active_joints.stiffness` | joint parameter；`1000.0` | Position control 使用；每种模式都接受。 |
| `<mode>.active_joints.damping` | joint parameter；`50.0` | Position/velocity gain。 |
| `<mode>.active_joints.max_force` | joint parameter；`100.0` | Drive/effort 上限。 |
| `<mode>.active_joints.effort_limit` | joint parameter 或 `null`；`null` | Direct effort 对称上限。 |
| `<mode>.active_joints.joint_friction` | joint parameter；`0.5` | 缺省 joint friction。 |
| `<mode>.follower_joints.stiffness` | joint parameter；`50000.0` | 所有 active mode 下的 follower position-drive stiffness。 |
| `<mode>.follower_joints.damping` | joint parameter；`50.0` | Follower position-drive damping。 |
| `<mode>.follower_joints.max_force` | joint parameter；`100.0` | Follower drive 上限。 |
| `<mode>.follower_joints.joint_friction` | joint parameter；`0.5` | Follower joint friction。 |

Joint parameter 可以是一个有限非负 scalar、按 selected-joint 顺序的非空有限非负 sequence，
或非空精确 `joint_name: value` mapping。执行前会根据导入的 articulation 解析 sequence 长度与
mapping 名称。

## cuRobo 算法 Profile

文档根节点只能包含 `curobo`。该 owner 接受 `task_bundle`、`device`、`kinematics`、
`motion_planner`，不接受 `enabled`、`planning_joint_group`、`robot` 或任意 task file path。

| 路径 | 类型与缺省值 | 规则 |
| --- | --- | --- |
| `curobo.task_bundle` | `curobo_v0_8_default`；同值 | 安装的 cuRobo runtime 必须为 `0.8.0`。 |
| `curobo.device.device` | 非空字符串；`cuda:0` | Backend 使用的 Torch device。 |
| `curobo.device.tensor_dtype` | `float32`；`float32` | 项目验证的 tensor dtype。 |
| `curobo.device.collision_geometry_dtype` | `float32`；`float32` | Collision geometry dtype。 |
| `curobo.device.collision_gradient_dtype` | `float32`；`float32` | Collision gradient dtype。 |
| `curobo.device.collision_distance_dtype` | `float32`；`float32` | Collision distance dtype。 |

### IK 算法

以下字段都位于 `curobo.kinematics.ik`。

| 叶字段 | 类型与缺省值 | 规则 |
| --- | --- | --- |
| `num_seeds` | 正整数；`32` | 每 problem optimizer seed 数。 |
| `position_tolerance` | 有限数 >= 0；`0.002` | 单位 m。 |
| `orientation_tolerance` | 有限数 >= 0；`0.01` | 单位 rad。 |
| `use_cuda_graph` | boolean；`true` | CUDA graph execution 开关。 |
| `random_seed` | 整数 >= 0；`123` | 可复现 seed generation。 |
| `optimizer_collision_activation_distance` | 有限数 >= 0；`0.01` | 单位 m。 |
| `store_debug` | boolean；`false` | 保留 solver debug data。 |
| `override_optimizer_num_iters.particle`, `override_optimizer_num_iters.lbfgs` | 整数 >= 0 或 `null`；`null` | `null` 使用 task-bundle 默认；不接受其它 key。 |
| `override_iters_for_multi_link_ik` | 整数 >= 0 或 `null`；`null` | Multi-link iteration override。 |
| `optimization_dt` | 正有限数或 `null`；`null` | Velocity-aware IK timestep。 |
| `velocity_regularization_weight` | 有限数 >= 0 或 `null`；`null` | C-space rollout regularization。 |
| `acceleration_regularization_weight` | 有限数 >= 0 或 `null`；`null` | C-space rollout regularization。 |
| `success_requires_convergence` | boolean；`true` | 除可行性外还要求 pose-error 收敛。 |
| `seed_position_weight` | 有限数 >= 0；`1.0` | Seed solver weight。 |
| `seed_orientation_weight` | 有限数 >= 0；`1.0` | Seed solver weight。 |
| `seed_velocity_weight` | 有限数 >= 0；`0.0` | Seed solver weight。 |
| `seed_acceleration_weight` | 有限数 >= 0；`0.0` | Seed solver weight。 |
| `seed_solver_num_seeds` | 正整数；`32` | Seed solver population。 |
| `max_batch_size` | 正整数；`256` | IK resource capacity。 |
| `multi_env` | boolean；`false` | 每个 batch problem 是否有独立 collision world。 |
| `max_goalset` | 正整数；`1` | 每 problem goal-set capacity。 |
| `self_collision_check` | boolean；`true` | cuRobo 模型 self-collision check。 |
| `collision_cache.cuboid`, `collision_cache.mesh` | 整数 >= 0；省略/空 | 预分配 obstacle 数；不接受其它 geometry key。 |

### Motion Planner 算法

以下字段都位于 `curobo.motion_planner`。

| 叶字段 | 类型与缺省值 | 规则 |
| --- | --- | --- |
| `warmup` | boolean；`true` | Materialize 后预热 planner。 |
| `num_ik_seeds` | 正整数；`32` | Goal IK seed 数。 |
| `num_trajopt_seeds` | 正整数；`4` | Trajectory-optimization seed 数。 |
| `position_tolerance` | 有限数 >= 0；`0.002` | 单位 m。 |
| `orientation_tolerance` | 有限数 >= 0；`0.01` | 单位 rad。 |
| `use_cuda_graph` | boolean；`true` | CUDA graph execution 开关。 |
| `random_seed` | 整数 >= 0；`123` | 可复现 initialization。 |
| `optimizer_collision_activation_distance` | 有限数 >= 0；`0.01` | 单位 m。 |
| `store_debug` | boolean；`false` | 保留 solver debug data。 |
| `max_batch_size` | 正整数；`256` | Planner resource capacity。 |
| `multi_env` | boolean；`false` | 每 problem 独立 collision world。 |
| `max_goalset` | 正整数；`1` | 每 problem goal-set capacity。 |
| `self_collision_check` | boolean；`true` | cuRobo 模型 self-collision check。 |
| `collision_cache.cuboid`, `collision_cache.mesh` | 整数 >= 0；省略/空 | 预分配 obstacle 数。 |

## Logging Profile

文档根节点只能包含 `logging`。省略叶字段采用下列 parser 默认值；随仓 profile 可以显式选择
不同值。

| 路径 | 类型与缺省值 | 含义 |
| --- | --- | --- |
| `logging.enabled` | boolean；`true` | 是否打开/写入 Single Scene joint CSV。 |
| `logging.joint_tracking_path` | 非空路径字符串或 `null`；`logs/joint_tracking/pinch_grasp.csv` | `null` 禁用 file target；相对路径从仓库根目录解析。 |
| `logging.flush_interval_s` | 正有限数；`0.05` | 仿真时间 flush cadence。 |
| `logging.interval_steps` | 正整数；`1` | Physics-step sampling decimation。 |
| `logging.log_actual_position` | boolean；`true` | Actual-position 列。 |
| `logging.log_actual_velocity` | boolean；`true` | Actual-velocity 列。 |
| `logging.log_command_position` | boolean；`true` | Position-command 列。 |
| `logging.log_command_velocity` | boolean；`true` | Velocity-command 列。 |
| `logging.log_command_effort` | boolean；`true` | 语义 effort-command 列。 |
| `logging.log_action_effort` | boolean；`false` | 发送给 Isaac 的 effort action。 |
| `logging.log_measured_effort` | boolean；`false` | PhysX measured effort；读取成本较高。 |
| `logging.log_applied_effort` | boolean；`false` | PhysX applied effort；读取成本较高。 |

## 完整依赖图校验

校验沿实际 owner graph 进行：

```text
runtime
  -> env/base + per-env fragments
  -> robot profiles -> selected controller bundles
                    -> robot resources + selected cuRobo algorithm profile
  -> object profiles
  -> logging profile
```

校验还覆盖 runtime/env 模式一致性、跨 instance USD 子树重叠、controller bundle 完整性、每种
object 的 kind-specific consumer，以及每个启用规划的 robot 合并后的 cuRobo 配置。
`curobo.enabled: false` 的 robot 跳过 backend materialization 校验。
`configs/curobo/task/` 下文件是 task-bundle resource，不是独立项目 profile。

普通成功输出只包含固定 `config_validated` event、runtime profile 名和 runtime fingerprint。
Fingerprint 只覆盖 effective runtime mapping，不包含下游 profile 文件内容；完整 graph 仍会先
校验。使用 `--dump-effective-config` 检查 effective runtime 值及其来源。
