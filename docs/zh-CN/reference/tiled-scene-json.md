# Tiled Scene 并行环境使用方式与指令格式

语言：[中文](tiled-scene-json.md) | [English](../../en/reference/tiled-scene-json.md)

本文负责独立 Isaac Tiled Scene runtime 的 JSON transport、消息、selector、生命周期和响应契约。
Checkout 准备与第一次完整请求见 [Tiled Scene 快速开始](../getting-started/tiled-scene-quickstart.md)；全部
启动参数、最终默认值和进程标记见 [Tiled Scene CLI 参考](tiled-scene-cli.md)。cuRobo 算法和资源行为见
[运动规划](../guides/motion-planning.md)。

## 1. 运行模型

Tiled Scene runtime 在一个 Isaac stage 中克隆 `num_envs` 个同构环境。所有 env 共享一个 physics clock，
但拥有独立 articulation/object 状态、episode counter 和 trajectory playback：

```text
JSON request
  -> 主线程解析 robot_id/env_ids
  -> 为 selected env 生成目标
  -> 合成 full-batch command target
  -> 同一 physics tick 写所有 selected articulations
  -> world.step() 一次
```

一个高层 `step` 展开成固定 `decimation` 个 physics ticks；`ee_linear_path` 也可用 `duration_s`
指定逻辑时长，执行 tick 数向上对齐 physics dt。未选择的 env 不停止全局物理时间，它们保持自己的 command
target，并和 selected env 一起随共享 World 推进。

异步 planner worker 只接收主线程复制出的 numpy current state、目标和 selector，不访问 Isaac
`World`、stage、articulation view 或 PhysX handle。planner 完成并不自动推进仿真，轨迹必须进入
`TiledTrajectoryBuffer` 后由 `step_trajectory` 回放。

## 2. Transport Framing 与并发

启动和 endpoint 启用规则见 [Tiled Scene CLI 参考](tiled-scene-cli.md)。启用后，各 control transport
采用以下协议边界：

| Transport | 请求 framing | 直接响应 | 额外交付 |
|---|---|---|---|
| stdin | 每行一个 JSON object | stdout 一行 JSON | 无 |
| TCP JSONL | 每行一个 JSON object | 同一连接一行响应 | 无；生命周期用对应 API 轮询 |
| WebSocket | 每个 text message 一个 JSON object | 同一连接返回 JSON text | 广播已处理响应的副本 |

每个请求都有一个直接响应。TCP 客户端应保持长连接，按“一行请求、一行响应”交替读写，不能依赖
一次连接只发一个请求。异步 planner 状态仍通过 `planner_status` 查询。

Transport thread 只解析严格 JSON 并排队；Isaac、USD 和 PhysX 访问留在仿真主线程。
TCP/WebSocket 共享 runtime `max_connections` 名额，stdin 不占用。所有输入共享有界 request
queue 和 `max_message_bytes`，每个 WebSocket 还有有界 event queue。超长、非 UTF-8、重复
key、`NaN`、`Infinity` 或带尾随内容的 JSON 都会在主线程 dispatch 前被拒绝。

## 3. Runtime 配置边界

Tiled Scene 入口要求 env profile 设置 `tiled.enabled: true`，并至少包含一个同构 env 行。Clone 数量、layout、per-env
pose/metadata override、camera env 范围和碰撞过滤属于 env profile；CLI 不覆盖 `num_envs`。
Planner 选择、容量、失败策略、playback 限制和 telemetry 属于 runtime profile。

配置编写流程见[配置指南](../guides/configuration.md)，精确字段与校验见[配置参考](configuration.md)。
碰撞策略与 planning world 的区分由[碰撞模型](../guides/collision-models.md)说明；planner 算法和
batch 资源由[运动规划](../guides/motion-planning.md)说明。

## 4. 消息与 Selector 总览

### 4.1 全部消息类型

<!-- tiled-message-index:start -->
| `type` | 同步/异步 | 用途 |
|---|---|---|
| `status` | 同步 | runtime、env、robot discovery |
| `step` | 同步 | 固定 tick 的 batched command action |
| `reset` | 同步 | selected env 恢复初始状态 |
| `get_state` | 同步 | 读取临时 batched runtime state |
| `set_state` | 同步 | 写 selected env command state |
| `get_snapshot` | 同步 | 读取单 env 持久 snapshot |
| `set_snapshot` | 同步 | 广播 snapshot 到 target env |
| `clone_state` | 同步 | tiled 内 source env 克隆 |
| `load_trajectory` | 同步 | 载入已有关节轨迹 |
| `hand` | 同步提交 | 排队一个独立关节子轨，默认 append |
| `step_trajectory` | 同步 | 回放 trajectory buffer |
| `trajectory_status` | 同步 | 查询 buffer |
| `clear_trajectory` | 同步 | 清理 buffer |
| `plan` | 异步提交 | 复制状态并进入 planner queue |
| `planner_status` | 同步 collect | 派发/收集 planner result |
| `cancel_plan` | 同步 | 取消 queued/running plan |
| `clear_completed` | 同步 | 清理 completed 摘要 |
| `quit` | 同步 | 请求退出主循环 |
<!-- tiled-message-index:end -->

异常统一返回：

```json
{"event":"rejected","error":"..."}
```

输入错误不会终止主循环。

### 4.2 Selector 规则

| 目标 | 字段 | 规则 |
|---|---|---|
| 单机器人 | `robot_id` | 非负当前会话 ID |
| 多机器人 step/回放 | `robot_ids` | 非空、无重复 ID array |
| 所有机器人 | `robot_ids: "all"` | 只用于多机器人语义消息 |
| 单环境 | `env_id` | 只用于 `get_snapshot` source |
| 多环境 | `env_ids` | 非空、无重复 ID array；所有作用于环境的命令都必须显式提供 |
| clone source | `source_env_id` | 单个 env ID |
| clone targets | `target_env_ids` | 非空 env ID array |

以下消息始终要求显式 `env_ids`，不能用省略字段表达“全部环境”：

<!-- tiled-env-ids-required-index:start -->
| `type` | 作用范围 |
|---|---|
| `reset` | Selected env 状态 |
| `get_state` | Selected env 状态 |
| `set_state` | Selected env 状态 |
| `set_snapshot` | Snapshot 恢复目标 |
| `load_trajectory` | Playback 目标 |
| `step_trajectory` | Playback 目标 |
| `trajectory_status` | 查询的 playback 行 |
| `clear_trajectory` | 清理的 playback 行 |
| `hand` | Hand motion 目标 |
| `plan` | Planning problem 行 |
| `step` | Control 目标 |
<!-- tiled-env-ids-required-index:end -->

`cancel_plan` 按 `request_id` 精确取消时不要求 env selector；没有 `request_id`、按 robot/env 交集
取消时必须提供 `env_ids`。

`step` 和 `step_trajectory` 在场景有多台机器人时必须显式 `robot_id/robot_ids`。`plan`、
`load_trajectory`、`hand` 是单机器人命令，场景有多台机器人时必须用 `robot_id`。公开协议不接受
label/name/side/role selector；响应边界把内部 label 转为 `robot_id`。

### 4.3 通用响应与 `quit`

除 `status` 外，成功响应通常包含：

| 字段 | 说明 |
|---|---|
| `event` | 接口对应的结果类型，例如 `step`、`state`、`plan_submitted` |
| `accepted` | 请求是否成功通过校验并完成同步操作或提交异步操作 |
| `backend` | 运行后端；`TiledSceneRuntime` 为 `isaac` |
| `step/time_s` | 响应时的全局 physics step 与仿真时间；部分纯队列响应不带 |
| `env_ids` | 请求显式选择并由 runtime 校验后的 env |
| `robot_id/robot_ids` | public response boundary 从内部 label 转换得到的会话 ID |

输入结构、selector、shape 或 runtime 操作失败统一返回：

```json
{"event":"rejected","error":"env_ids contains out-of-range env id"}
```

`rejected` 不会终止主循环，但调用方不能假定失败前完全无副作用；只有文中明确说明原子校验的接口
才承诺失败前不写状态。

退出请求与响应：

```json
{"type":"quit"}
```

```json
{"event":"quit","accepted":true}
```

`quit` 设置 runtime 的退出事件，主循环在当前消息处理完成后退出。它不是暂停命令，也不会保存
trajectory buffer、planner cache 或 episode 状态。

入口关闭顺序是 stdin reader -> WebSocket/TCP ingress -> telemetry publisher -> runtime。runtime 内部
再按 planner -> camera -> IK context -> `SimulationApp` 关闭。`runtime.shutdown.transport_timeout_s`
约束 stdin 和 listener 线程，`state_publisher_timeout_s` 和 `camera_publisher_timeout_s` 分别
约束 telemetry/camera publisher，planner 使用 `runtime.planner.resources.shutdown_timeout_s`。关闭超时会
打印 `TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT` 和仍存活资源；runtime 只会在子资源成功关闭后标记
closed，保留的 planner/camera/IK 句柄可在后续 `close()` 中重试。

## 5. Status 与会话发现

```json
{"type":"status"}
```

下面用一个假设的 4-env、单机器人 profile 展示响应结构；它不是内置 `scene3_tiled` 的实际规模。
当前 `scene3_tiled` 配置为 64 个 env、两台机器人，真实响应中的数组和 `robots[]` 会相应展开。

关键响应字段：

```json
{
  "event": "status",
  "backend": "isaac",
  "env": "demo_tiled",
  "num_envs": 4,
  "step": 0,
  "time_s": 0.0,
  "episode_steps": [0, 0, 0, 0],
  "episode_ids": [0, 0, 0, 0],
  "env_roots": [
    "/World/envs/env_0",
    "/World/envs/env_1",
    "/World/envs/env_2",
    "/World/envs/env_3"
  ],
  "env_origins": [
    [0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [6.0, 0.0, 0.0],
    [9.0, 0.0, 0.0]
  ],
  "runtime": {"inspect_env_ids": [0]},
  "robots": [
    {
      "robot_id": 0,
      "label": "ar5v2_l6v1_0",
      "robot_profile": "ar5v2_l6v1_l",
      "kind": "arm_hand",
      "supports_planning": true,
      "count": 4,
      "num_dof": 30,
      "command_joints": ["..."],
      "ik_tcp_frame": "AR5V2_L_pinch_tcp"
    }
  ],
  "sensors": {"cameras": []}
}
```

`env_id` 与 `robot_id` 是独立维度：robot 0 表示同一个机器人实例定义，batched articulation 中有
`num_envs` 行。和普通 Single Scene 一样，robot ID 只在当前进程内有效。

`status` 不使用 selector，也不带 `accepted`。关键字段含义：

| 字段 | 说明 |
|---|---|
| `num_envs` | batched articulation 的 env 行数 |
| `env_roots/env_origins` | 每个 env 的 USD 根路径和 world-frame 平移原点 |
| `episode_steps/episode_ids` | 每个 env 的 episode 局部步数与 reset 代次 |
| `robots[].robot_id` | 公开命令使用的会话级机器人 ID |
| `robots[].command_joints` | `step/load_trajectory/plan/set_state` 的 command-space 列顺序 |
| `robots[].supports_planning` | 是否存在有效 cuRobo binding；不代表异步 linear backend 不可用 |
| `robots[].ik_tcp_frame` | 同步 `ee_*` 和省略 `tcp_frame_name` 时使用的默认 TCP |

响应固定包含 `per_env_metadata`（没有 metadata 时为空数组）。runtime 对应 provider/资源启用时还会
加入 `transport`、`telemetry`、`planner` 和 `camera_output` 诊断：分别报告请求队列与连接容量、遥测
publisher、规划队列/缓存和相机输出状态。客户端应按字段名读取，不应假定 status 只有上例字段。

## 6. Canonical `step` Action

同步 action 的唯一外层形状是顶层 `type="step"` 和显式 `kind`。普通 action 使用统一
`values`；`ee_linear_path` 还可在同一顶层使用命名的相对/绝对目标字段：

```json
{
  "type": "step",
  "kind": "joint_delta_pos",
  "robot_ids": [0, 1],
  "env_ids": [0, 1],
  "values": [0.01, 0.2, 0.0],
  "decimation": 2,
  "interpolation": "smoothstep"
}
```

外层 `type` 固定为 `step`；`kind` 和 action 参数都像上例一样直接位于消息顶层，不存在嵌套的
`action` mapping。当前 `kind` 未消费的额外字段会被拒绝。

### 6.1 Action 字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `kind` | 必填 | 下表七种 action |
| `values` | hold 外通常必填 | `(D,)` 或 `(E,D)`；`ee_linear_path` 中是 compact world-frame offset，与命名目标字段互斥 |
| `decimation` | runtime execution 默认 | 正整数 physics tick 数；在 `ee_linear_path` 中是显式替代 `duration_s` 的方式 |
| `duration_s` | runtime planner request 默认 | 仅 `ee_linear_path`；正数逻辑时长，不能与 `decimation` 同时提供 |
| `sample_dt_s` | physics dt | 仅 `ee_linear_path`；顺序 batched IK 的采样周期 |
| `interpolation` | runtime command 默认（内置 profile 为 `smoothstep`） | `linear` 或 `smoothstep` |
| `tcp_frame_name` | robot 默认 TCP | 仅 `ee_*`，必须已注册 |
| `pose_reference_frame` | runtime command 默认（内置 profile 为 `env`） | `env/base/world` |
| `target_offset` | 无 | 仅 `ee_linear_path`；相对起点位移，与 `target_position/values` 互斥 |
| `target_position` | 无 | 仅 `ee_linear_path`；绝对终点，与 `target_offset/values` 互斥 |
| `orientation_mode` | runtime command 默认（内置 profile 为 `current`） | 仅 `ee_linear_path`；允许 `free/current/target` |
| `target_orientation_quat_wxyz` | 无 | `orientation_mode=target` 时必填；wxyz 四元数 |

同步 `ee_linear_path` 中，JSON 显式值优先于 `runtime.planner.request_defaults` 或
`runtime.execution.command_defaults`。JSON 显式 `null` 不等于省略；`duration_s: null` 和
`sample_dt_s: null` 会被拒绝。

<!-- tiled-action-index:start -->
| `kind` | 单行宽度 | 语义 |
|---|---:|---|
| `hold` | 不带 `values` | 保持上次 command target |
| `joint_position_target` | `1..command_dim` | 绝对 command-space 前缀目标 |
| `joint_delta_pos` | `1..command_dim` | 相对当前 command-space 前缀目标 |
| `ee_pose_target` | 7 | `[x,y,z,qw,qx,qy,qz]` |
| `ee_delta_pos` | 3 | TCP 世界平移增量，保持当前姿态 |
| `ee_delta_pose` | 6 或 7 | `[dx,dy,dz,rx,ry,rz]` rotvec，或平移 + 目标 wxyz |
| `ee_linear_path` | 3 | TCP 直线/线性位姿路径；每个 sampled waypoint 做 batched IK |
<!-- tiled-action-index:end -->

关节 action 可写 command joint 前缀，未覆盖后缀保持当前 target；若需要非前缀子集，使用
`load_trajectory/plan` 的 `joint_names`，`step` 不提供 name scatter。

### 6.2 通用 `step` 响应

所有 7 种 action 都在返回前完成目标转换、固定 tick 执行和 TCP cache 刷新。关节 action 响应示例：

```json
{
  "event": "step",
  "accepted": true,
  "backend": "isaac",
  "kind": "joint_position_target",
  "env_ids": [0, 1],
  "robot_ids": [0],
  "ticks": 20,
  "step": 20,
  "time_s": 0.0833333333,
  "episode_steps": [20, 20, 20, 20],
  "info": [
    {"robot_id": 0, "command_width": 2}
  ]
}
```

`ticks` 是这条 action 实际推进的 physics tick 数；全局 `step/time_s` 和所有 env 的
`episode_steps` 都会推进，因为所有 env 共享一个 World。未选择 env 只保持既有 target。`info[]`
按 `robot_id` 给出实际 command width；`ee_*` 还包含 `ik` 和 `ik_backend`。只要输入能解析，单个
env 的数值 IK 失败不会把整个响应改成 `rejected`，而是在 `info[].ik` 中标记并保持该行 seed。

### 6.3 Hold 案例

```json
{
  "type": "step",
  "kind": "hold",
  "robot_ids": "all",
  "env_ids": [0, 1, 2, 3],
  "decimation": 2
}
```

`hold` 不接受 `values`，保持各机器人 adapter 的上次 command target；首次使用时保持当前关节位置。
它仍推进 `decimation` 个 physics ticks，适合等待接触稳定或刷新 GUI/telemetry。

### 6.4 绝对关节目标案例

```json
{
  "type": "step",
  "kind": "joint_position_target",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [[0.3, -0.4], [0.4, -0.4]],
  "decimation": 20,
  "interpolation": "linear"
}
```

`values` 的一维形式广播到 selected env，二维形式第一维必须是 1 或 `len(env_ids)`。宽度 D 写入
`status.robots[].command_joints` 的前 D 列；未覆盖后缀保持原 target。目标在 `decimation` ticks 内
按 `interpolation` 从各 env 当前 command position 插值到终点。

### 6.5 相对关节目标案例

```json
{
  "type": "step",
  "kind": "joint_delta_pos",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [[0.05, 0.0], [-0.05, 0.0]],
  "decimation": 4,
  "interpolation": "smoothstep"
}
```

delta 以消息执行时读取到的 selected-env 当前 command joint position 为基准，不以上一次请求提交时的
状态为基准。shape 和 command-prefix 规则与绝对关节目标相同。

### 6.6 绝对 TCP Pose 案例

```json
{
  "type": "step",
  "kind": "ee_pose_target",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "values": [0.35, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "pose_reference_frame": "env",
  "decimation": 2
}
```

- `env`：position 是每个 env 的局部坐标，内部加各自 env origin。
- `base`：pose 是各 robot base-local，position 和 orientation 都转到 world。
- `world`：所有 selected env 使用同一个世界 pose，通常只适合检查单 env。

### 6.7 TCP 平移增量案例

```json
{
  "type": "step",
  "kind": "ee_delta_pos",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "values": [0.0, 0.0, 0.01],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "decimation": 2
}
```

`values` 是 world-frame TCP 平移增量 `[dx,dy,dz]`，姿态保持命令开始时的当前 TCP 姿态。
`pose_reference_frame` 不改变该 compact delta 的方向；需要 frame-aware 路径 offset 时使用
`ee_linear_path.target_offset`。

### 6.8 TCP 位姿增量案例

旋转向量形式：

```json
{
  "type": "step",
  "kind": "ee_delta_pose",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [0.0, 0.0, 0.01, 0.0, 0.0, 0.0872664626],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "decimation": 4
}
```

7 维形式把后四维解释为绝对目标 wxyz，而不是四元数增量：

```json
{
  "type": "step",
  "kind": "ee_delta_pose",
  "robot_id": 0,
  "env_ids": [0, 1],
  "values": [0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "decimation": 4
}
```

6 维形式为 `[dx,dy,dz,rx,ry,rz]`，rotvec 单位为 rad，并以左乘方式作用到当前 world-frame TCP
姿态。两种形式的平移都是 world-frame 增量，均可用 `(E,6)` 或 `(E,7)` 逐 env 指定。

除 `ee_linear_path` 外，`ee_*` action 一次调用 batched cuRobo IK。`failure_policy=hold_failed_env` 时
成功行写 IK 解，失败的 selected 行保持 seed/current target；响应 `info[].ik` 包含
`ik_success/failed_env_ids/ik_position_error/ik_orientation_error`。`failure_policy=reject_request` 时，
任一 selected env 失败都会在所有机器人 target/physics 写入前拒绝整条同步请求，rejected 响应返回
排序后的 `failed_env_ids`。

### 6.9 固定时长批量 TCP 直线

```json
{
  "type": "step",
  "kind": "ee_linear_path",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "target_offset": [0.0, 0.0, 0.10],
  "orientation_mode": "free",
  "duration_s": 0.4,
  "sample_dt_s": 0.02,
  "interpolation": "linear",
  "tcp_frame_name": "AR5V2_L_pinch_tcp"
}
```

`ee_linear_path` 必须在 `values/target_offset/target_position` 中恰好提供一个。命名字段一行可
广播，也可传 `(len(env_ids),3)`。绝对终点案例：

```json
{
  "type": "step",
  "kind": "ee_linear_path",
  "robot_id": 0,
  "env_ids": [0, 1],
  "target_position": [0.35, 0.0, 0.4],
  "pose_reference_frame": "base",
  "orientation_mode": "target",
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "interpolation": "smoothstep"
}
```

三种姿态模式与异步 `linear_pose_path` 一致：

- `free`：只约束 TCP 位置，IK 不检查姿态误差。
- `current`：整条路径保持命令开始时的 TCP 姿态。
- `target`：从起点姿态 Slerp 到必填的 wxyz 目标姿态。

显式提供 `target_orientation_quat_wxyz` 而省略 `orientation_mode` 时，请求自动推导为
`target`，并优先于 runtime command 默认。显式 `free/current` 不能与目标四元数组合，`target`
但没有四元数也会被拒绝。

`pose_reference_frame` 同时解释命名目标的位置、offset 方向和目标姿态；`env` 的轴与 world 对齐、
position 加各 env origin，`base` 按每个机器人 base pose 转到 world。compact
`values: [dx,dy,dz]` 固定表示 world-frame offset；省略 `orientation_mode` 时使用 resolved runtime
command 默认。需要显式 frame、绝对位置或目标姿态时使用命名字段。

`linear` 表示等时间间隔的等距离/等角度 waypoint；`smoothstep` 使用同一个平滑进度参数化位置和
姿态，不改变直线与 Slerp 的几何路径。

时序规则：

1. 显式 `duration_s` 与 `decimation` 不能同时提供；都省略时注入
   `runtime.planner.request_defaults.duration_s`。显式 `decimation` 会抑制该 duration 默认并选择固定
   tick 数。
2. IK 次数为 `ceil(duration_s / sample_dt_s)`；未给 `sample_dt_s` 时使用 physics dt。末 waypoint
   固定落在逻辑 `duration_s`，不要求二者整除。
3. physics tick 数为 `ceil(duration_s / physics_dt)`。例如 100 Hz physics 下 `duration_s=0.405`
   执行 41 ticks，响应实际 `duration_s=0.41`。
4. runtime 在第一个 `world.step()` 前生成全部 IK waypoint，上一 waypoint 的关节解作为下一次 seed，
   再按相同 `linear/smoothstep` 路径进度把关节轨迹重采样到 physics tick endpoints。
5. 所有 selected env 执行完全相同的 ticks 和实际 duration；共享 World 中未选择的 env 也推进相同
   physics ticks，但保持已有 command target。
6. `hold_failed_env` 下某个 env 首次 IK 失败后，该行从该 waypoint 起保持最后成功 target，其它 env
   继续；`reject_request` 下任一 selected env 失败都会在 physics 执行前拒绝所有机器人。求解器抛出
   配置、shape 或 CUDA 异常时同样整条 action rejected。

响应中的 `duration_s` 是实际 physics duration，`sample_dt_s` 是有效 IK 周期，`ik_waypoints` 是
顺序 batched IK 次数。`info[].ik` 额外包含：

| 字段 | 说明 |
|---|---|
| `ik_success` | 每个 env 是否完成全部 waypoint |
| `ik_first_failure_step` | 首个失败 IK waypoint，1-based；成功为 `-1` |
| `ik_completed_steps` | 失败前完成的 waypoint 数 |
| `ik_position_error` | 已尝试 waypoint 的最大位置误差 |
| `ik_orientation_error` | 后端提供时为最大姿态误差 |

该 action 适合需要严格统一 rollout horizon 的批量控制。它不是 collision-aware trajectory optimizer，
也不提供圆弧或自动绕障；这些需求仍使用异步 planner 或离线轨迹。

## 7. State、Reset 与 Episode

### 7.1 Reset

```json
{"type":"reset","env_ids":[0,1]}
```

```json
{
  "event": "reset",
  "accepted": true,
  "env_ids": [0, 1],
  "step": 120,
  "time_s": 0.5,
  "episode_steps": [0, 0, 37, 42],
  "episode_ids": [3, 3, 2, 2],
  "objects_reset": 4
}
```

selected env 的机器人位置/速度和对象状态恢复为启动初值，`episode_steps` 清零，`episode_ids` 加一；
对应 trajectory 被清理，相关 pending planner request 被取消。全局 `step/time_s` 不回退，也不会为
reset 自动调用 `world.step()`。`objects_reset` 是成功恢复的 object/env 组合条目数，即对象数乘以
本次 selected env 数，不按对象内部 prim/body 数计数。`env_ids` 必须显式提供；重置全部环境时也要
列出全部 ID。

### 7.2 `get_state`

```json
{"type":"get_state","env_ids":[0,1]}
```

完整响应外层与 robot array 形状：

```json
{
  "event": "state",
  "accepted": true,
  "backend": "isaac",
  "env_ids": [0, 1],
  "step": 120,
  "time_s": 0.5,
  "state": {
    "robots": [
      {
        "robot_id": 0,
        "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
        "joint_positions": [[0.1, 0.2], [0.3, 0.4]],
        "joint_velocities": [[0.0, 0.0], [0.0, 0.0]],
        "tcp_positions_world": [[0.35, 0.0, 0.4], [3.35, 0.0, 0.4]],
        "tcp_orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
      }
    ],
    "objects": {
      "Tblock": {
        "positions_world": [[0.2, 0.0, -0.4], [3.2, 0.0, -0.4]],
        "orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
      }
    },
    "episode_steps": [12, 15],
    "episode_ids": [2, 2]
  }
}
```

可用 `fields` 裁剪：

```json
{
  "type": "get_state",
  "env_ids": [0],
  "fields": [
    "robots.ar5v2_l6v1_0.joint_positions",
    "objects.Tblock.positions_world",
    "episode_steps"
  ]
}
```

`fields` 中一段式名称选择顶层字段；robot/object 子字段使用
`robots.<internal_label>.<field>` 或 `objects.<object_name>.<field>`。robot 路径中的名称来自
`status.robots[].label`，不是 `robot_id`。不存在的 field 会被忽略，因此调用方应检查响应中是否
真的出现请求字段。

公开响应中的 `state.robots` 是 array，每项含 `robot_id`，并包含：

- `joint_names`
- `joint_positions: (E,D)`
- `joint_velocities: (E,D)`
- `tcp_positions_world: (E,3)`
- `tcp_orientations_wxyz: (E,4)`

对象字段按 object runtime 类型返回 root/body pose。`get_state` 是当前进程内的调试和 telemetry 格式；
持久恢复和 Single Scene/tiled 交换使用 snapshot。

### 7.3 `set_state`

```json
{
  "type": "set_state",
  "env_ids": [0, 1],
  "state": {
    "robots": [
      {
        "robot_id": 0,
        "joint_positions": [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "joint_velocities": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
      }
    ],
    "episode_steps": [5, 8],
    "episode_ids": [2, 3]
  }
}
```

成功响应不回显大数组：

```json
{
  "event": "set_state",
  "accepted": true,
  "backend": "isaac",
  "env_ids": [0, 1],
  "step": 120,
  "time_s": 0.5
}
```

一行 robot state 可广播到 selected env；宽度必须是该 robot 完整 command dimension，列顺序以
`get_state` 返回的 `state.robots[].joint_names` 为准。公开输入使用
`robots[] + robot_id`，不接受内部 label-keyed robot map。未出现的机器人/字段保持原值。写回后
清理 selected env trajectory，并取消与 selected env 相交的 planner request，避免 stale result。

`state` 当前可写字段只有 `robots[].joint_positions`、`robots[].joint_velocities`、`episode_steps` 和
`episode_ids`。episode 数组可传单值广播或 `len(env_ids)` 个值；robot 数组不能包含重复
`robot_id`。`set_state` 不写 object pose，持久对象恢复使用 `set_snapshot`。

## 8. Snapshot 与 Clone

Canonical payload、身份匹配、shape、单位、恢复结果和事务规则统一由
[Snapshot 数据与恢复参考](snapshots.md)拥有。本节只定义 Tiled Scene 消息外层和 env selector。

`get_snapshot` 只读取一个 source env：

```json
{"type":"get_snapshot","env_id":0}
```

```json
{
  "event": "snapshot",
  "accepted": true,
  "backend": "isaac",
  "env_id": 0,
  "step": 120,
  "time_s": 0.5,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

上面的缩略主体只展示消息外层。把真实返回的完整 snapshot 广播到明确 targets：

```json
{
  "type": "set_snapshot",
  "env_ids": [1, 2],
  "strict": true,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

`env_ids` 必填、非空、无重复并执行范围校验。`label_map` 可选；`strict` 默认 true。
公开成功响应为：

```json
{
  "event": "snapshot_restored",
  "accepted": true,
  "backend": "isaac",
  "robot_ids": [0],
  "objects": [],
  "env_ids": [1, 2],
  "partial": false,
  "step": 120,
  "time_s": 0.5
}
```

`set_snapshot` 只允许 `type/snapshot/env_ids/label_map/strict`。匹配使用稳定 label、profile 和
`strict`。Runtime 把单个 source payload 广播到每个 selected env；结果中的 object name 只列一次，
不是每个 object-env pair 一项。

同一 runtime 内复制状态可用：

```json
{
  "type": "clone_state",
  "source_env_id": 0,
  "target_env_ids": [1, 2, 3],
  "strict": true
}
```

```json
{
  "event": "state_cloned",
  "accepted": true,
  "backend": "isaac",
  "robot_ids": [0, 1],
  "objects": ["Tblock"],
  "env_ids": [1, 2, 3],
  "partial": false,
  "source_env_id": 0,
  "target_env_ids": [1, 2, 3],
  "step": 120,
  "time_s": 0.5
}
```

它等价于主线程内 `get_snapshot(source) + set_snapshot(targets)`。`get_snapshot` 使用 `env_id`，
`clone_state` 使用 `target_env_ids` 指定目标环境且不接受 `label_map`。两种写回都使用 canonical
Snapshot 参考定义的事务语义。

## 9. Trajectory Buffer

### 9.1 载入轨迹

```json
{
  "type": "load_trajectory",
  "request_id": "trajectory-1",
  "source": "offline_planner",
  "robot_id": 0,
  "env_ids": [0, 1],
  "times": [0.0, 0.1, 0.2],
  "positions": [
    [0.0, 0.0],
    [0.1, 0.05],
    [0.2, 0.1]
  ],
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "replace": true,
  "queue": false
}
```

成功载入响应：

```json
{
  "event": "trajectory_loaded",
  "accepted": true,
  "backend": "isaac",
  "robot_id": 0,
  "env_ids": [0, 1],
  "samples": 3,
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "step": 120,
  "time_s": 0.5
}
```

字段和 shape：

| 字段 | 规则 |
|---|---|
| `times` | 非空有限一维数组；多样本时严格递增，允许第一项为 0 |
| `positions` | `(T,D)` 广播，或 `(E,T,D)`/`(1,T,D)` |
| `joint_names` | 可选，长度 D、无重复、必须属于 command joints |
| `request_id` | 可选，保存在 playback status |
| `source` | 可选，默认 `interactive` |
| `replace` | 默认 true；替换 selected env 当前/队列 |
| `queue` | 默认 false；true 时 append 优先于 replace |

省略 `joint_names` 时 D 列写 command-space 前缀，其余关节以载入时 target 填充。指定 names 时按名称
散射，未指定列同样保持载入时 target。不接受嵌套 `trajectory`、`joint_positions` 或 `overlays`。

### 9.2 回放和管理

回放请求：

```json
{"type":"step_trajectory","robot_ids":[0],"env_ids":[0,1],"decimation":2}
```

```json
{
  "event": "trajectory_step",
  "accepted": true,
  "backend": "isaac",
  "env_ids": [0, 1],
  "robot_ids": [0],
  "ticks": 2,
  "step": 122,
  "time_s": 0.5083333333,
  "episode_steps": [22, 22, 20, 20],
  "trajectory": [
    {
      "robot_id": 0,
      "env_ids": [0, 1],
      "active_env_ids": [0, 1],
      "completed_env_ids": [],
      "idle_env_ids": [],
      "dt_s": 0.0041666667
    }
  ],
  "planner_ready": [],
  "planner_loaded": [],
  "load_rejected": []
}
```

`decimation` 默认使用 CLI `--default-decimation` 且必须大于 0。场景有多机器人时必须显式
`robot_id/robot_ids`；`robot_ids="all"` 可一次回放所有机器人。`planner_ready/planner_loaded` 和
`load_rejected` 是该调用顺便 collect 到的异步规划摘要。
`trajectory_step.load_rejected` 与 `planner_status` 使用相同的逐 request playback admission 拒绝结构；
本次收集结果均成功载入时为空数组。

查询 buffer：

```json
{"type":"trajectory_status","robot_id":0,"env_ids":[0,1]}
```

```json
{
  "event": "trajectory_status",
  "accepted": true,
  "backend": "isaac",
  "step": 122,
  "time_s": 0.5083333333,
  "trajectory": {
    "num_envs": 64,
    "limits": {
      "max_queue_depth_per_env": 32,
      "max_samples_per_env": 100000,
      "max_duration_s_per_env": 3600.0,
      "overflow_policy": "reject"
    },
    "queued_trajectories": 2,
    "queued_samples": 6,
    "queued_duration_s": 0.4,
    "rejected_loads": 0,
    "rejected_loads_scope": "robot",
    "robots": [
      {
        "robot_id": 0,
        "count": 2,
        "queued_trajectories": 2,
        "queued_samples": 6,
        "queued_duration_s": 0.4,
        "rejected_loads": 0,
        "active_env_ids": [0, 1],
        "completed_env_ids": [],
        "envs": [
          {
            "env_id": 0,
            "request_id": "trajectory-1",
            "source": "offline_planner",
            "stage": "trajectory",
            "completed": false,
            "elapsed_s": 0.0083333333,
            "duration_s": 0.2,
            "progress": 0.0416666667,
            "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
            "samples": 3,
            "joint_track_names": [],
            "queue_length": 1,
            "queued_trajectories": 1,
            "queued_samples": 3,
            "queued_duration_s": 0.2
          },
          {
            "env_id": 1,
            "request_id": "trajectory-1",
            "source": "offline_planner",
            "stage": "trajectory",
            "completed": false,
            "elapsed_s": 0.0083333333,
            "duration_s": 0.2,
            "progress": 0.0416666667,
            "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
            "samples": 3,
            "joint_track_names": [],
            "queue_length": 1,
            "queued_trajectories": 1,
            "queued_samples": 3,
            "queued_duration_s": 0.2
          }
        ]
      }
    ]
  }
}
```

清理 buffer：

```json
{"type":"clear_trajectory","robot_id":0,"env_ids":[0,1]}
```

```json
{
  "event": "trajectory_cleared",
  "accepted": true,
  "backend": "isaac",
  "cleared": [{"robot_id": 0, "env_ids": [0, 1]}],
  "step": 122,
  "time_s": 0.5083333333
}
```

`step_trajectory` 每次请求只在进入 tick loop 前 collect 一次 ready planner result，然后每个 physics
tick 从 selected env 的队首 playback 采样；本次 loop 期间才完成的 future 留到下次 collect。没有轨迹的
env 保持 target。响应列出 `active_env_ids/completed_env_ids/idle_env_ids`，以及 planner 本次
ready/loaded/load-rejected 摘要。

`trajectory_status` 必须提供 `env_ids`；可以省略 robot selector 查询全部机器人。响应返回 configured
`limits`、聚合 `queued_trajectories/queued_samples/queued_duration_s` 和累计 `rejected_loads`。对应的
`rejected_loads_scope` 在无 robot selector 时为 `buffer`，按 robot 查询时为 `robot`；env selector
不会缩小这个 request counter。robot/env 条目也报告
对应队列容量。append 校验 existing+new，replace 只校验新 sequence；一次 load 会先原子预检全部
selected env，任一 depth/sample/duration 超限就整体拒绝，绝不静默淘汰正在执行的轨迹。每个 env
条目还包含 request/source、当前 stage、elapsed/duration/progress、完整轨迹的 `joint_names`，以及
内部部分关节轨实际覆盖的 `joint_track_names`。普通完整轨迹的 `joint_track_names` 为空数组。
`clear_trajectory` 同样必须提供 `env_ids`；可以省略 robot selector，清理这些 env 的全部机器人。

### 9.3 独立 `hand` 子轨

```json
{
  "type": "hand",
  "request_id": "close-hand",
  "robot_id": 0,
  "env_ids": [0, 1],
  "duration_s": 0.2,
  "joint_positions": {
    "L6V1_L_hand_index_mcp_pitch": [0.7, 0.8]
  },
  "queue": true,
  "replace": false
}
```

```json
{
  "event": "hand_motion_queued",
  "accepted": true,
  "backend": "isaac",
  "motions": [
    {
      "robot_id": 0,
      "env_ids": [0, 1],
      "duration_s": 0.2,
      "joint_names": ["AR5V2_L_arm_joint_1", "L6V1_L_hand_index_mcp_pitch"],
      "joint_track_count": 1,
      "queued": true
    }
  ],
  "step": 122,
  "time_s": 0.5083333333
}
```

| `hand` 字段 | 默认/规则 |
|---|---|
| `request_id` | 可选，写入 playback status |
| `source` | 默认 `interactive_hand` |
| `robot_id` | 多机器人场景必填；只允许一个机器人 |
| `env_ids` | 必填；非空、唯一且不越界 |
| `duration_s` | 必填、非负；0 表示单点立即目标 |
| `joint_positions` | 必填非空 mapping；key 为 command joint name，value 为标量或每 env 一值 |
| `queue` | 默认 true；在现有 playback 后 append |
| `replace` | 默认 false；true 时替换且不 append |

每个 joint target 可为标量（广播）或长度 `len(env_ids)` 的数组。`duration_s` 必填且非负；默认
`queue=true/replace=false`，因此会排在现有 arm/hand playback 后。所有 hand 字段都必须位于该消息
顶层，不会从外层 plan/request 继承。成功响应的 `motions[]` 包含 `robot_id/env_ids/duration_s`、完整
command-space `joint_names`、`joint_track_count=1` 和 `queued`。

该接口内部使用 `PlaybackJointTrack` 表达只覆盖指定 command 列的部分关节轨。队列追加时，子轨在
真正开始播放的一刻读取当前 command target 作为动态基线，避免载入时的旧 arm target 覆盖前一段
轨迹的终点。`trajectory_status` 通过 `joint_track_names` 暴露被覆盖列，但公开协议不接受内部
`PlaybackJointTrack` 对象，也没有 overlay 输入。

`hand` 当前按 joint name 选择稀疏 command 列，调用方应只传 status 中确认属于 hand 的关节；它不做
cuRobo planning，也不提供 arm/hand 同 tick 的完整 timeline。需要严格同步臂手的复杂动作时，普通
Single Scene 使用 canonical `group_tracks`；tiled 侧应直接载入完整 command-space trajectory。

## 10. 异步 `plan`

`plan` 字段全部位于消息顶层，`kind` 必填。它只复制提交时的 selected-env target，不持续读取正在
变化的 runtime state。

共同字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `request_id` | 自动生成 `plan-<uuid>` | planner 生命周期、取消和 completed cache 的唯一 ID |
| `robot_id` | 单机器人场景可省略 | 只允许一个机器人 |
| `env_ids` | 必填 | 规划问题行；非空、唯一且不越界 |
| `kind` | 必填 | `joint_position_target/joint_delta_pos/linear_pose_path` |
| `duration_s` | runtime planner request 默认（内置 profile 为 `1.0`） | 正数逻辑时长 |
| `sample_dt_s` | physics dt | planner 输出采样周期 |
| `avoid_collisions` | false | true 时要求完整 cuRobo collision capability |
| `load_on_success` | true | ready 成功结果是否自动写 trajectory buffer |
| `replace` | true | 自动载入时是否替换 selected env 现有 playback |
| `source` | `interactive_plan` | 写入 result/playback status 的来源字符串 |

任一种 `kind` 通过校验后的立即响应都只表示进入 planner manager，不表示 GPU 已完成：

```json
{
  "event": "plan_submitted",
  "accepted": true,
  "backend": "isaac",
  "request_id": "plan-1",
  "robot_id": 0,
  "env_ids": [0, 1],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "segments": ["joint_position_target"],
  "load_on_success": true
}
```

### 10.1 绝对关节目标

```json
{
  "type": "plan",
  "request_id": "plan-1",
  "robot_id": 0,
  "env_ids": [0, 1],
  "kind": "joint_position_target",
  "joint_positions": [[0.2, 0.1], [0.3, 0.1]],
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false,
  "load_on_success": true,
  "replace": true,
  "source": "policy"
}
```

### 10.2 相对关节目标

```json
{
  "type": "plan",
  "request_id": "delta-1",
  "robot_id": 0,
  "env_ids": [0, 1],
  "kind": "joint_delta_pos",
  "joint_deltas": [0.1, 0.0],
  "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
  "duration_s": 0.8,
  "sample_dt_s": 0.02
}
```

目标 `(D,)` 广播，`(E,D)` 逐 env 使用。省略 `joint_names` 时写 command prefix；指定 names 时按名称
散射。绝对目标未指定列保持 current，delta 未指定列按 0 增量处理。

### 10.3 线性 TCP 路径

相对终点：

```json
{
  "type": "plan",
  "request_id": "path-1",
  "robot_id": 0,
  "env_ids": [0],
  "kind": "linear_pose_path",
  "target_offset": [0.0, 0.0, 0.1],
  "orientation_mode": "free",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

绝对终点与姿态：

```json
{
  "type": "plan",
  "request_id": "path-2",
  "robot_id": 0,
  "env_ids": [0],
  "kind": "linear_pose_path",
  "target_position": [0.35, 0.0, 0.4],
  "orientation_mode": "target",
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "sample_dt_s": 0.02
}
```

`target_position/target_offset` 必须二选一。`orientation_mode`：

- `free`：只约束位置。
- `current`：保持起点姿态。
- `target`：Slerp 到必填 wxyz 目标姿态。

这里也遵循相同的显式四元数规则：有四元数但省略 mode 时推导为 `target`；显式
`free/current` 加四元数、或 `target` 不带四元数都会被拒绝。其它情况下使用 runtime command
默认。

异步 `plan` 的 `target_position/target_offset` 只接受 shape `(3,)`，目标姿态只接受
`(4,)`；同一目标作用于所有 selected env。该 canonical 接口不定义 `pose_reference_frame`，payload
携带该字段会作为 unknown field 被拒绝。目标数值直接按 cuRobo context 的 robot-base-local 坐标解释。
需要 world/env 目标时，调用方必须先按各机器人
base pose 转换；需要每个 env 不同目标时，必须拆成多条 request。这与同步
`step/ee_linear_path` 的可广播 `(E,3)` 命名目标不同。

Task-space path 当前逐 env 调用 planner facade，不进入 joint batch。每个 JSON `plan` 只表达一个
segment，字段均位于 request 顶层。

## 11. Planner 生命周期与批量调度

### 11.1 查询、取消、清理

派发 queued 请求并等待/收集 ready result：

```json
{"type":"planner_status","wait_timeout_s":0.1}
```

```json
{
  "event": "planner_status",
  "accepted": true,
  "backend": "isaac",
  "ready": [
    {
      "request_id": "plan-1",
      "robot_id": 0,
      "env_ids": [0, 1],
      "success": true,
      "status": "SUCCESS",
      "message": "",
      "samples": 51,
      "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
      "source": "policy",
      "load_on_success": true
    }
  ],
  "loaded": [
    {"request_id": "plan-1", "robot_id": 0, "env_ids": [0, 1]}
  ],
  "load_rejected": [],
  "planner": {
    "pending": [],
    "pending_count": 0,
    "completed_count": 1,
    "max_pending_requests": 64,
    "max_completed_results": 256,
    "max_batch_problems": 64,
    "oversize_request_policy": "split",
    "max_workers": 2,
    "running_batch_count": 0,
    "rejected_requests": 0,
    "split_requests": 0,
    "evicted_completed_results": 0,
    "shutdown_requested": false,
    "shutdown_timed_out": false,
    "queued_request_ids": [],
    "running_request_ids": [],
    "live_request_ids": [],
    "completed": [
      {
        "request_id": "plan-1",
        "robot_id": 0,
        "env_ids": [0, 1],
        "success": true,
        "status": "SUCCESS",
        "message": "",
        "samples": 51,
        "joint_names": ["AR5V2_L_arm_joint_1", "AR5V2_L_arm_joint_2"],
        "source": "policy",
        "load_on_success": true
      }
    ]
  }
}
```

`wait_timeout_s` 默认 0，只做非阻塞 collect；正数表示最多等待第一个 future 完成，不保证等到所有
pending。`ready` 只包含本次新收集结果，`planner.completed` 是仍保留在 cache 的全部摘要。响应从不
返回大型 `times/positions` 数组；成功轨迹只通过自动载入后的 trajectory buffer 消费。
`load_rejected` 逐 request 返回成功规划结果被 playback admission 拒绝的
`request_id/robot_id/env_ids/code/error`。单条 load 拒绝不会丢弃同批后续 ready result；本次自动载入
尝试分别出现在 `loaded` 与 `load_rejected`。

按 request ID 取消：

```json
{"type":"cancel_plan","request_id":"plan-1"}
```

```json
{
  "event": "plan_cancelled",
  "accepted": true,
  "backend": "isaac",
  "result": {
    "request_id": "plan-1",
    "accepted": true,
    "status": "cancel_requested",
    "future_cancelled": false
  }
}
```

按 robot/env 交集批量取消：

```json
{"type":"cancel_plan","robot_id":0,"env_ids":[0,1]}
```

此形式的 `result` 是逐 request 结果 array。省略 `request_id` 时 `env_ids` 必填，`robot_id` 可省略以
匹配这些 env 上所有机器人的 pending 请求；三个 selector 全部省略会被拒绝。queued 请求状态直接
变为 `cancelled`；running GPU future 通常只能标记 `cancel_requested`，完成后才以 `CANCELLED`
result 被 collect。

按一个或多个 ID 清理 completed cache：

```jsonl
{"type":"clear_completed","request_id":"plan-1"}
{"type":"clear_completed","request_ids":["plan-1","plan-2"]}
{"type":"clear_completed"}
```

```json
{
  "event": "completed_cleared",
  "accepted": true,
  "backend": "isaac",
  "result": {
    "cleared": ["plan-1"],
    "missing": ["plan-2"],
    "count": 1
  }
}
```

- `submit_plan` 立即返回 `plan_submitted`，不会等待 GPU。
- `planner_status` 先 dispatch queued request，再等待最多 `wait_timeout_s` 收集第一个完成 future。
- `step_trajectory` 也会 collect ready result。
- 成功且 `load_on_success=true` 的结果自动载入 buffer；false 时只保留摘要。
- queued request 可立即取消；running GPU 调用通常不能强制中断，完成后转为 `CANCELLED`。
- `cancel_plan` 省略 request ID 时按必填 env 与可选 robot 的交集取消；全部省略会被拒绝。
- `clear_completed` 支持 `request_id`、`request_ids`；全部省略清空 completed cache。

`planner_status.planner` 返回 `pending_count/completed_count/max_*`、oversize policy、累计拒绝/split/淘汰
计数、关闭状态、live/queued/running ID、逐请求 queued/running 状态和不含大轨迹矩阵的 completed
摘要。`max_completed_results=0` 时 ready 仍在当前响应返回，但不进缓存。

### 11.2 两级 Batch 调度

```text
公开 request_id 队列
  -> manager 按 FIFO 连续同构请求分组
  -> 每组一个 future，受 planner-workers 并发限制
  -> TiledCuroboPlanningBackend.plan_many 合并 request rows
  -> CuroboBatchJointProblem（无 env/request 字段）
  -> CuroboBatchJointPlanner -> cuRobo BatchMotionPlanner
  -> 按 row slice 拆回原 request_id
```

manager batch key 包含：

- robot label（公开侧对应同一 `robot_id`）
- 完整 command joint names
- `duration_s/sample_dt_s`
- `avoid_collisions`
- joint-space segment kind/结构

只合并队列中连续、key 完全相同的 joint-space 请求。遇到异构请求立即切组，不跨过它重排 FIFO。
每组默认最多 64 个 problem 行，计数是各 request 的 `len(env_ids)` 之和，不是 request 条数。多条
请求超过上限时 manager 会切成多个 future。单条请求超过上限时，
`oversize_request_policy=split` 按 env rows 切分 current/goal 和每个 segment goal，并只在所有有界
chunk 成功且结构一致后恢复原 request ID；任一 chunk 失败则整 request 失败，不会部分载入。
`oversize_request_policy=reject` 在进入 pending queue 前拒绝。任何 backend 调用都不会超过
`max_batch_problems`。

后端把一组请求 `vstack` 成 batch rows，一次创建独享 planner/context，并直接构造不含 env/request
字段的 `CuroboBatchJointProblem`。若真实 rows 小于 cuRobo 固定 `BatchMotionPlanner.batch_size`，用
最后一行 padding；padding 结果不会公开。runtime profile 会校验显式 `max_batch_problems` 不超过所选
cuRobo profile 容量，manager 在进入 backend 前完成单请求分 chunk。

`--planner-workers > 1` 允许多个 batch future 并发，但每个 worker 都会创建独享 cuRobo context、
CUDA graph/cache/tensor。增加 worker 会提高吞吐，也会显著增加显存；先按 1 或 2 测量。

## 12. 端到端流程

可直接执行的 [Tiled Scene 快速开始](../getting-started/tiled-scene-quickstart.md)统一负责 discovery、同步操作、
结果检查和关闭。该流程成功后，按[控制与轨迹](../guides/control-and-trajectories.md)和
[运动规划](../guides/motion-planning.md)完成 trajectory 与异步 planning 任务，再回到本文查询精确
消息字段。

响应边界按原字段把内部 robot-keyed mapping 转成 `robots[]`、`info[]` 或 `trajectory[]`，客户端应按
`robot_id` 查找对应项，不依赖 array 只有一个元素。

## 13. 常见失败与诊断

| 状态/错误 | 原因与处理 |
|---|---|
| `rejected` selector 越界 | 重新读取 status，检查 env/robot ID |
| `values first dimension` | 行数必须为 1 或 `len(env_ids)` |
| `robots is required` | 多机器人 step/回放显式传 selector |
| `COLLISION_UNSUPPORTED` | robot spheres、checker、cache 或 scene sync 不完整 |
| `BATCH_UNAVAILABLE` | `batch_only` 下 request 不是 batch-capable 或 planner 无 batch API |
| `BATCH_TOO_SMALL` | merged batch row 数超过 cuRobo batch size；拆小公开 request 或提高 profile batch |
| `FAILED env rows [...]` | 至少一个真实 env 没有成功 seed；检查目标、seed、碰撞与容差 |
| `too many pending` | 先 collect/cancel，或合理提高 pending 上限 |
| `duplicate request_id` | pending 或 completed cache 仍有同名 ID；换 ID 或 clear completed |
| stale 结果被取消 | reset/set_state/set_snapshot 与 pending request 的 robot/env 相交 |
| GUI/Foxglove 不刷新 | 设置 `--idle-physics-policy hold_step`，并确认 telemetry rate 大于 0 且已配置 live/MCAP sink |

## 14. 当前边界

- batched IK 不回退到 per-env IK loop。
- 同步 `ee_linear_path` 的每个 IK waypoint 跨 env batch，重采样后每个 physics tick 下发
  full-batch target；异步
  `linear_pose_path` 仍逐 env 顺序 IK。
- 单条公开 request 超过 `max_batch_problems` 时按 `oversize_request_policy` 处理：内置 profile 的
  `split` 会按 env row 切成有界 chunk，`reject` 则在进入 pending queue 前拒绝。
- `multi_env=false` 时 batch problem 共享一个 collision world；各 env 障碍物不同不能假设已独立避碰。
- `avoid_collisions=true` 不会静默退化。
- trajectory payload 不接受部分关节轨对象；`hand` 是唯一稀疏子轨便利接口。
- 固定步长只能执行整数 physics ticks。`duration_s/sample_dt_s` 非整数时 IK 数向上取整，
  `duration_s/physics_dt` 非整数时执行时长可能多出不足一个 physics tick。
- `get_state/set_state` 是运行时调试格式；持久恢复使用 snapshot。
