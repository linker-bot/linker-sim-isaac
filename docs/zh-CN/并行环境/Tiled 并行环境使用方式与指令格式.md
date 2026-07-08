# Tiled 并行环境使用方式与指令格式

本文说明当前项目中 Isaac Lab 风格 tiled envs 的启动方式、配置约定、交互协议和 JSON 指令格式。

tiled 模式的定位是“单个 Isaac/PhysX scene 中并行运行多个同构 env，并用同步 command step 推进”。它不是旧单臂/双臂 motion runtime 的并行版本；规划能力通过 runtime 外围的 trajectory buffer 和 async planner manager 接入，`TiledCommandAdapter` 本身仍只处理同步 step-control。

## 1. 当前能力边界

支持:

- 一个 `SimulationApp` / 一个 `World` / 一个 PhysX scene。
- `/World/envs/env_0 ... /World/envs/env_{N-1}` 下放置 N 份同构任务实例。
- 单臂 tiled envs 和双臂 tiled envs；双臂机器人逻辑名通常为 `left`、`right`。
- 所有 env 拥有相同机器人和相同物体集合，但每个 env 可以覆盖同名物体的初始 `root_pose`。
- batched `Articulation` view 读写多个 env 的机器人关节状态。
- `env_ids` 局部 reset、局部 get/set state、局部 action target 更新。
- `load_trajectory` / `step_trajectory` 回放已规划好的 batched 关节轨迹。
- `plan` / `planner_status` / `cancel_plan` 提交后台规划，并把 ready result 写入轨迹缓冲。默认 `linear` backend 支持单段关节空间目标；`--planner-backend cumotion` 可接入 task-space line/arc 和 specified path。
- `load_trajectory` 和 async planner ready result 支持 `before` / `sync` / `after` 手部 overlay；overlay 只通过显式 `joint_positions` mapping 覆盖 command-space 中存在的手部关节列。
- `hand` / `dual_hand` 可作为独立 hand-only motion 默认追加到 selected robot/env 的 trajectory playback queue。
- 交互脚本通过 stdin JSONL 或 TCP JSONL 接收指令。
- 可选 Foxglove live / MCAP telemetry。

不支持:

- 每个 env 拥有不同物体集合。
- 旧 motion runtime 的 IK pose/offset 队列、`cancel_current`/`estop` 语义。
- 在 `TiledCommandAdapter` 内部做 graph planner、trajectory optimizer 或每 env 变长调度。
- 旧 motion runtime 的通用 running/pending motion queue；tiled 只保留 trajectory playback queue，其中包含 hand-only queue。
- 每个 env 推进不同数量的 physics steps。即使只给部分 env 下发 action，physics 仍同步推进所有 env。
- tiled 交互脚本的 WebSocket command transport。当前只有 stdin JSONL 和 TCP JSONL。

## 2. 运行入口

### 2.1 交互入口

交互入口用于手工调试、上层程序逐条发送 command、episode reset 和状态检查。它始终启动真实 Isaac tiled scene，并通过 batched articulation view 写入 selected robot 的关节目标:

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --default-decimation 1
```

启动成功后会打印:

```text
TILED_INTERACTIVE_READY
```

如果启用 TCP JSONL，会额外打印:

```text
TILED_INTERACTIVE_TCP_JSONL host=127.0.0.1 port=9003
```

退出时打印:

```text
TILED_INTERACTIVE_EXIT
```

## 3. env profile 配置方式

旧的单文件 env profile 仍然可用。tiled 示例推荐使用目录型 profile:

```text
configs/envs/scene3_tiled/
  base.yaml
  envs/
    env_000.yaml
    env_001.yaml
    env_002.yaml
    env_003.yaml
```

`base.yaml` 保存所有 env 共用设置:

- `env`: 重力、physics/render frequency 等。
- `visuals`: 视角、灯光。
- `robots`: 单臂或双臂机器人集合。
- `objects`: 所有 env 都有的同构物体集合。
- `sensors.cameras`: 所有 env 共用的相机参数，例如分辨率、模态和输出端口。
- `tiled`: tiled 拓扑和 clone/runtime 选项。

`envs/env_XXX.yaml` 只保存单个 env 的差异，当前支持同名对象的 `root_pose`
覆盖，以及同名相机的 `pose` 覆盖:

```yaml
env_id: 1
objects:
  Tblock:
    root_pose:
      xyz: [0.12, 0.04, -0.4]
      rpy: [0.0, 1.5707, 0.18]
cameras:
  world_rgbd:
    pose:
      xyz: [0.08, 0.0, 0.08]
      rpy: [0.0, 1.1, 0.0]
metadata:
  replay_id: scene3_tiled_001
```

tiled runtime 会为每个 env 创建一份相机，例如
`/World/envs/env_0/WorldRGBD`。离线输出目录会自动追加 `env_000` 这类后缀，
Foxglove topic prefix 也会追加同样的 env 后缀，避免多个 env 写到同一帧文件或 topic。

关键 `tiled` 字段:

```yaml
tiled:
  enabled: true
  num_envs: 4
  base_env_path: /World/envs
  env_prefix: env
  spacing: 2.0
  per_env_config_dir: envs
  clone:
    replicate_physics: false
    copy_from_source: false
    enable_env_ids: false
    filter_collisions: true
    collision_root_path: /World/collisions
  runtime:
    inspect_env_ids: [0]
```

说明:

- env 数量只由 YAML 中的 `tiled.num_envs` 决定；交互 CLI 不提供 `--num-envs` 覆盖。
- `base_env_path` 和 `env_prefix` 决定 env root，例如 `/World/envs/env_0`。
- `spacing` 决定 env root 的网格间距，需要足够大以避免可视和物理重叠。
- `filter_collisions: true` 用于关闭 env 间碰撞。
- `replicate_physics: true` 会请求 PhysX replication 性能路径；如果 scene builder 检测到 MJCF fixed-base root joint 不兼容，会在构建时自动关闭 replication。
- `inspect_env_ids` 只表示 GUI/telemetry 调试时重点关注哪些 env，并会出现在 `status.runtime.inspect_env_ids`；它不是物理或渲染性能过滤开关。

## 4. 交互脚本 CLI 参数

常用参数:

| 参数 | 含义 |
| --- | --- |
| `--env` | env profile 名称，例如 `scene3_tiled`。 |
| `--gui` | 打开 Isaac GUI。 |
| `--default-decimation` | action 未指定 `decimation` 时展开的 physics tick 数。 |
| `--planner-backend` | async planner backend，`linear` 或 `cumotion`。默认 `linear`；`cumotion` 支持 task-space/specified-path。 |
| `--planner-workers` | tiled async planner worker 数；worker 只消费状态快照，不访问 Isaac runtime。 |
| `--max-pending-requests` | in-flight planner 请求上限，默认 `64`。 |
| `--max-completed-results` | completed result 摘要缓存上限，默认 `256`；设为 `0` 表示不保留。 |
| `--stdin / --no-stdin` | 是否从 stdin 读取 JSONL。默认开启。 |
| `--hold` | 空闲时保持当前 target 并持续刷新 GUI/Foxglove；stdin EOF 后仍保持进程，适合 IDE/后台启动、TCP-only 或只看 Foxglove。 |
| `--tcp-jsonl-host` | TCP JSONL host，默认 `127.0.0.1`。 |
| `--tcp-jsonl-port` | TCP JSONL port；不传则不启动 TCP server。 |

Telemetry 参数:

| 参数 | 含义 |
| --- | --- |
| `--foxglove-live-host` | Foxglove live server host。 |
| `--foxglove-live-port` | Foxglove live server port；tiled 日常调试建议 `8767`，不传则不开 live。 |
| `--foxglove-mcap-path` | 写入 MCAP 的路径；不传则不写 MCAP。 |
| `--telemetry-env-ids` | 逗号分隔 selected env ids，默认 `0`。 |
| `--telemetry-decimation` | 每隔多少 global step 发布一次 telemetry；reset/set_state 总会发布。 |
| `--telemetry-rate-hz` | 开启 Foxglove/MCAP 后的周期状态发布频率，默认 `10`；设为 `0` 时关闭周期发布。 |
| `--telemetry-topic-prefix` | topic 前缀，默认 `/tiled`。 |
| `--telemetry-full-batch-json / --no-telemetry-full-batch-json` | 是否发布 selected state JSON。 |
| `--telemetry-joint-states / --no-telemetry-joint-states` | 是否为第一个 selected env 发布标准 JointStates。 |

## 5. Transport

### 5.1 stdin JSONL

默认从 stdin 读取 JSONL。每一行必须是一个 JSON object，每一行返回一个 JSON response。

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --default-decimation 1
```

在终端输入:

```json
{"type":"status"}
{"type":"step","kind":"joint_delta_pos","env_ids":[1],"robots":["left"],"values":[[0.01,0,0]],"decimation":1}
{"type":"quit"}
```

### 5.2 TCP JSONL

启动 TCP server:

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --no-stdin \
  --tcp-jsonl-host 127.0.0.1 \
  --tcp-jsonl-port 9003
```

发送一行 JSON:

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 9003
```

TCP server 每行 request 返回一行 response。多个 TCP 客户端可以连接，但 command 仍作用于同一个 runtime；不要假设它提供事务隔离。

## 6. JSON 消息通用规则

所有交互消息都是 JSON object。

通用字段:

| 字段 | 类型 | 适用范围 | 含义 |
| --- | --- | --- | --- |
| `type` | string | 所有消息 | 控制消息类型或 action 类型。 |
| `env_ids` | int array | reset/get_state/set_state/action/trajectory/plan | 选中的 env id；不传表示全部 env。 |
| `robot` | string | 单机器人 trajectory/plan/action | 单个机器人逻辑名，例如 `"left"`。 |
| `robots` | string 或 string array | action/trajectory/plan | 选中的机器人逻辑名，`all`、`"left,right"` 或 `["left"]`。 |
| `decimation` | int | action/step_trajectory | 本条 command step 展开成多少个 physics ticks。 |
| `interpolation` | string | action | `smoothstep` 或 `linear`。 |
| `tcp_frame_name` | string | `ee_*` action | 覆盖默认 TCP frame。 |
| `pose_reference_frame` | string | `ee_pose_target` | `env`、`base` 或 `world`。默认 `env`。 |

单位和姿态:

- 长度单位是 m。
- 关节角单位是 rad。
- 四元数顺序是 `wxyz`，即 `[w, x, y, z]`。
- `delta_rotvec` 是旋转向量，单位 rad。
- tiled action 数组第一维是 env 维度。第一维为 1 时允许广播。

`env_ids` 语义:

- 不传 `env_ids` 表示选择全部 env。
- `env_ids` 必须是一维、非空、无重复、在 `[0, num_envs)` 范围内。
- 对 action 而言，只更新 selected env 的目标；未选中的 env 保持当前目标或当前位置。
- 无论选中几个 env，physics 仍同步推进所有 env。
- 对 `reset` 而言，只重置 selected env。
- 对 `get_state` / `set_state` 而言，只读写 selected env 的状态行。

`robots` 语义:

- `robots` 选择的是 robot 逻辑名，不是 env id。`left` / `right` 来自 env profile 的 `robots.dual.left/right`。
- 要选择第几个环境，用 `env_ids`；例如 `env_ids:[1]` 表示只更新 env 1。
- 交互 CLI 不提供启动级 robot allowlist；runtime 会创建 env profile 中的全部 robot。
- 单机器人 env 的 action 可以省略 `robots`；双臂/多机器人 env 的 action 和 `step_trajectory` 必须在消息里显式写 `robot`、`robots:["left"]` 或 `robots:"all"`。

## 7. 消息格式总览

控制消息:

```json
{"type":"status"}
{"type":"reset","env_ids":[0,2]}
{"type":"get_state","env_ids":[0],"fields":["robots.left.joint_positions","episode_steps"]}
{"type":"set_state","env_ids":[1],"state":{...}}
{"type":"load_trajectory","robot":"left","env_ids":[0],"times":[0.0,0.5],"positions":[[0,0,0],[0.2,0,0]]}
{"type":"step_trajectory","robot":"left","decimation":4}
{"type":"plan","kind":"joint_position_target","robot":"left","env_ids":[0],"joint_positions":[[0.2,0,0]],"duration_s":0.5}
{"type":"planner_status","wait_timeout_s":0.1}
{"type":"cancel_plan","request_id":"plan-123"}
{"type":"quit"}
```

action 消息支持两种外形。

第一种: `type` 直接写 action 类型:

```json
{"type":"joint_delta_pos","robots":["left"],"values":[[0.01,0,0]],"decimation":1}
```

第二种: `type:"step"`，用 `kind` 或 `action` 描述动作:

```json
{"type":"step","kind":"joint_delta_pos","robots":["left"],"values":[[0.01,0,0]],"decimation":1}
```

```json
{"type":"step","robots":["left"],"action":{"kind":"joint_delta_pos","values":[[0.01,0,0]],"decimation":1}}
```

注意: 如果使用 `{"type":"step","action":{...}}`，当前解析器只从 `action` object 中读取 action 字段。建议把 `env_ids`、`robots` 这类选择字段放在最外层，把 `kind/values/decimation` 放在 `action` 内或全部使用 `type:"step", kind:...` 的扁平格式。

## 8. 控制消息

### 8.1 status

请求:

```json
{"type":"status"}
```

返回中会包含 `backend:"isaac"` 和 `robots` 摘要:

```json
{
  "event": "status",
  "backend": "isaac",
  "env": "scene3_tiled",
  "num_envs": 2,
  "robots": {
    "left": {
      "count": 2,
      "num_dof": 18,
      "command_joints": ["joint1","joint2"]
    }
  }
}
```

### 8.2 reset

请求:

```json
{"type":"reset","env_ids":[1]}
```

返回:

```json
{
  "event": "reset",
  "accepted": true,
  "env_ids": [1],
  "step": 120,
  "time_s": 0.5,
  "episode_steps": [120,0],
  "episode_ids": [0,1],
  "objects_reset": 2
}
```

说明:

- 不传 `env_ids` 会 reset 全部 env。
- reset 后 selected env 的 `episode_steps` 清零，`episode_ids` 加 1。
- reset 会把 selected env 的机器人 joint position/velocity 写回初始化状态，并按启动时缓存的 env-local pose 恢复 selected env 的 runtime objects；对已有 RigidBodyAPI 的对象会尽量清零线速度和角速度。

### 8.3 get_state

请求全部字段:

```json
{"type":"get_state","env_ids":[0]}
```

返回示例:

```json
{
  "event": "state",
  "accepted": true,
  "backend": "isaac",
  "env_ids": [0],
  "state": {
    "robots": {
      "left": {
        "joint_names": ["joint1","joint2"],
        "joint_positions": [[0.1,0.2]],
        "joint_velocities": [[0.0,0.0]],
        "tcp_positions_world": [[0.22,0.01,0.18]],
        "tcp_orientations_wxyz": [[1.0,0.0,0.0,0.0]]
      }
    },
    "objects": {
      "Tblock": {
        "env_ids": [0],
        "positions_world": [[0.15,0.0,-0.4]],
        "positions_local": [[0.15,0.0,-0.4]],
        "orientations_wxyz": [[0.707,0.0,0.707,0.0]]
      }
    },
    "episode_steps": [8],
    "episode_ids": [0]
  }
}
```

字段裁剪:

```json
{"type":"get_state","env_ids":[0],"fields":["robots.left.joint_positions","episode_steps"]}
```

`fields` 支持 `robots.left.joint_positions` 和 `objects.Tblock.positions_world` 这类嵌套字段。

### 8.4 set_state

```json
{
  "type": "set_state",
  "env_ids": [1],
  "state": {
    "robots": {
      "left": {
        "joint_positions": [[0.3,0.2,0.1]],
        "joint_velocities": [[0.0,0.0,0.0]]
      }
    },
    "episode_steps": [7],
    "episode_ids": [3]
  }
}
```

返回:

```json
{"event":"set_state","accepted":true,"env_ids":[1],"step":10,"time_s":0.0416666667}
```

说明:

- `joint_positions` / `joint_velocities` 支持单行广播到多个 selected env。
- `joint_positions` / `joint_velocities` 宽度必须等于该 robot 的 command joint 数。
- set_state 后建议下一条 action 明确给目标，避免旧 hold target 语义干扰。

### 8.5 quit

请求:

```json
{"type":"quit"}
```

返回:

```json
{"event":"quit","accepted":true}
```

## 9. Action 消息

当前支持的 action kind:

| kind | 含义 |
| --- | --- |
| `hold` | 保持上一帧 target 或当前位置。 |
| `joint_position_target` | 绝对关节目标。 |
| `joint_delta_pos` | 基于当前关节位置的增量目标。 |
| `ee_pose_target` | 通过 batched cuMotion IK 求解绝对 TCP 位姿目标。 |
| `ee_delta_pos` | 通过 batched cuMotion IK 求解 TCP 平移微动，保持当前姿态。 |
| `ee_delta_pose` | 通过 batched cuMotion IK 求解 TCP 位姿微动。 |

`ee_*` action 通过 `BatchedCuMotionIKSolver` 调用 cuMotion batch API；运行环境缺少 cuMotion batch API 时不会回退到 per-env `solve_ik` loop。`pose_reference_frame="env"` 表示 env-local 目标，`"world"` 表示世界目标，`"base"` 表示当前 robot base-local 目标。

runtime 会按 `decimation` 逐 physics tick 下发插值后的 target；`interpolation` 支持 `smoothstep` 和 `linear`。

关节类 action 的 `values` 宽度表示本次要写入的 command joint 前缀宽度，不表示 env 数量，也不一定等于机械臂理论 DOF 数。对于 `joint_position_target` 和 `joint_delta_pos`：

- 如果 selected robot 有 7 个 command joints，`values:[[0.01,-0.01,0]]` 只作用于前 3 个 command joints。
- 需要控制 7 个 command joints 时必须传 7 列，例如 `values:[[0.01,-0.01,0,0,0,0,0]]`。
- 返回里的 `info.<robot>.command_width` 是本次 action 实际写入的列数；要查看该 robot 的完整 command joint 列表，用 `status` 返回中的 `robots.<name>.command_joints`。

### 9.1 hold

请求:

```json
{"type":"hold","decimation":4}
```

或:

```json
{"type":"step","kind":"hold","decimation":4}
```

说明:

- `hold` 不需要 `values`。
- runtime 保持每个 selected robot 的内部 target。

### 9.2 joint_position_target

绝对关节目标。

```json
{
  "type": "joint_position_target",
  "robots": ["left"],
  "values": [[0.2,-0.1,0.05]],
  "decimation": 2,
  "interpolation": "smoothstep"
}
```

也可以使用别名字段:

```json
{"type":"joint_position_target","robots":["left"],"joint_positions":[[0.2,-0.1,0.05]]}
```

带 `robots`/`env_ids` 的 step 格式:

```json
{
  "type": "step",
  "kind": "joint_position_target",
  "env_ids": [0,1],
  "robots": ["left"],
  "values": [[0.2,-0.1,0.05]],
  "decimation": 1
}
```

shape 规则:

- `values` 宽度 `K` 可以小于或等于 selected robot 的 command joint 数；脚本会写前 K 个 command joints。
- `K` 小于 command joint 数时，未写到的尾部 command joints 保持当前 target。
- 第一维为 1 时广播到所有 selected env。
- 第二维是关节列，单位 rad。

### 9.3 joint_delta_pos

基于当前关节位置的增量目标。

```json
{
  "type": "joint_delta_pos",
  "env_ids": [1],
  "robots": ["left"],
  "values": [[0.01,0.0,0.0]],
  "decimation": 1
}
```

别名字段:

```json
{"type":"joint_delta_pos","robots":["left"],"joint_deltas":[[0.01,0.0,0.0]]}
```

说明:

- `values` 宽度 `K` 可以小于或等于 selected robot 的 command joint 数；脚本只对前 K 个 command joints 做 `target = current + delta`。
- `K` 小于 command joint 数时，未写到的尾部 command joints 保持当前 target。
- 未选中的 env 不接收增量，保持当前目标。

### 9.4 ee_pose_target

绝对 TCP 位姿目标，通过 batched cuMotion IK 求解。

```json
{
  "type": "ee_pose_target",
  "position": [0.2,0.0,0.1],
  "orientation_quat_wxyz": [1.0,0.0,0.0,0.0],
  "pose_reference_frame": "env",
  "decimation": 2
}
```

也可以直接给 `values`:

```json
{
  "type": "ee_pose_target",
  "values": [[0.2,0.0,0.1,1.0,0.0,0.0,0.0]],
  "pose_reference_frame": "env"
}
```

字段:

- `position`: `(N,3)` 或 `(3,)`，单位 m。
- `orientation_quat_wxyz`: `(N,4)` 或 `(4,)`，可省略；省略时使用单位四元数。
- `values`: `(N,7)`，等价于 `[x,y,z,qw,qx,qy,qz]`。
- `pose_reference_frame`: 默认 `env`。

参考系:

- `env`: `position` 是 env-local 坐标。广播同一 action 时，每个 env 到达各自局部坐标中的相同位置。
- `world` / `base`: 当前 adapter 不额外加 env origin，按传入坐标求解。日常广播控制推荐优先使用 `env`。

### 9.5 ee_delta_pos

TCP 平移微动，保持当前 TCP 姿态，通过 batched cuMotion IK 求解。

```json
{
  "type": "ee_delta_pos",
  "offset": [0.01,0.0,0.0],
  "decimation": 2
}
```

或:

```json
{"type":"ee_delta_pos","values":[[0.01,0.0,0.0]]}
```

shape 规则:

- `offset` / `values` 是 `(N,3)` 或 `(3,)`。
- 第一维为 1 时广播到所有 selected env。
- 目标位置为 `current_tcp_positions_world + offset`。

### 9.6 ee_delta_pose

TCP 位姿微动，通过 batched cuMotion IK 求解。

使用平移增量 + 旋转向量增量:

```json
{
  "type": "ee_delta_pose",
  "offset": [0.01,0.0,0.0],
  "delta_rotvec": [0.0,0.0,0.05],
  "decimation": 2
}
```

使用平移增量 + 目标四元数:

```json
{
  "type": "ee_delta_pose",
  "offset": [0.01,0.0,0.0],
  "target_orientation_quat_wxyz": [1.0,0.0,0.0,0.0]
}
```

直接给 `values`:

```json
{"type":"ee_delta_pose","values":[[0.01,0.0,0.0,0.0,0.0,0.05]]}
```

```json
{"type":"ee_delta_pose","values":[[0.01,0.0,0.0,1.0,0.0,0.0,0.0]]}
```

shape 规则:

- `values` 宽度为 6 时表示 `[dx,dy,dz,rx,ry,rz]`，后三维是 rotvec delta。
- `values` 宽度为 7 时表示 `[dx,dy,dz,qw,qx,qy,qz]`，后四维是目标四元数。
- 如果只给 `offset`，不传姿态字段，则旋转增量为零，保持当前姿态。

## 10. 轨迹回放和异步规划

### 10.1 load_trajectory

`load_trajectory` 用于把外部已经规划好的关节轨迹放入 per-env/per-robot buffer。

```json
{
  "type": "load_trajectory",
  "robot": "left",
  "env_ids": [0,1],
  "times": [0.0,0.5,1.0],
  "positions": [
    [[0.0,0.0,0.0],[0.1,0.0,0.0],[0.2,0.0,0.0]],
    [[0.0,0.0,0.0],[0.2,0.0,0.0],[0.4,0.0,0.0]]
  ],
  "joint_names": ["joint1","joint2","joint3"],
  "request_id": "offline-001"
}
```

规则:

- `positions` 可为 `(T,D)`，广播到 selected env；也可为 `(E,T,D)`，每个 selected env 一条轨迹。
- 不传 `joint_names` 时，列顺序按 robot command joint 前缀解释；宽度小于 command joint 数时，其它关节保持载入时的当前 target。
- 传 `joint_names` 时会按名称映射到 command-space；未知关节名会被拒绝。
- `replace` 默认 `true`；设为 `false` 时，如果目标 env 已有未完成轨迹会拒绝。

### 10.2 step_trajectory

`step_trajectory` 在同步 physics tick 中消费 ready trajectory。没有 ready trajectory 的 env 会 hold 当前 target，不会阻塞 `world.step()`。

```json
{"type":"step_trajectory","robot":"left","env_ids":[0,1],"decimation":4}
```

返回中的 `trajectory` 会列出本次 selected env 里 active/completed/idle 的 env id。`planner_ready` 和 `planner_loaded` 表示本次 step 前是否收到了后台 planner result 并自动写入 buffer。

### 10.3 trajectory_status / clear_trajectory

```json
{"type":"trajectory_status","robot":"left"}
{"type":"clear_trajectory","robot":"left","env_ids":[0]}
```

`reset` 和 `set_state` 会自动清理 selected env 的 trajectory buffer，避免旧 episode 的轨迹继续回放。

### 10.4 plan

`plan` 是 tiled async planner 的入口。主线程只复制 selected env 当前 command target、robot/env 选择和路径参数；真正规划在 planner worker 中完成。`planner_status` 或后续 `step_trajectory` 会收集 ready result，并把成功结果写入 trajectory buffer。

所有 tiled 规划请求都使用 `type:"plan"`，运动类型写在 `kind`。旧 motion runtime 的顶层 `cspace_goal` / `cspace_delta` / `task_space_line` / `task_space_arc` / `specified_path` 消息、`plan_queue`、顶层 `moves` 队列、`move_type` 字段和 `side` 别名都不再是 tiled command 协议的一部分；tiled 里请选择 `robot` 或 `robots`。

默认 `--planner-backend linear` 只支持单段关节空间目标。需要 `task_space_line`、`task_space_arc` 或 `specified_path` 时，用 `--planner-backend cumotion` 启动；cuMotion backend 会把路径请求转换成 `MotionRequest` 或 `SpecifiedPathRequest`，最后仍输出统一的 `times + positions(E,T,D)` 关节轨迹。

绝对关节目标:

```json
{
  "type": "plan",
  "robot": "left",
  "env_ids": [0,1],
  "request_id": "plan-001",
  "duration_s": 0.5,
  "sample_dt_s": 0.02,
  "kind": "joint_position_target",
  "joint_positions": [[0.2,0.0,0.0]]
}
```

关节增量目标:

```json
{
  "type": "plan",
  "robot": "left",
  "env_ids": [1],
  "duration_s": 0.5,
  "kind": "joint_delta_pos",
  "joint_deltas": [[0.05,-0.02,0.0]]
}
```

Task-space line/arc 会提交 specified-path planning segment:

```json
{
  "type": "plan",
  "kind": "task_space_line",
  "robot": "left",
  "env_ids": [0,1],
  "request_id": "line-001",
  "duration_s": 1.0,
  "target_offset": [0.0,0.0,0.08],
  "orientation_mode": "none"
}
```

```json
{
  "type": "plan",
  "kind": "task_space_arc",
  "robot": "right",
  "env_ids": [0],
  "duration_s": 1.2,
  "target_offset": [0.0,0.05,0.0],
  "intermediate_offset": [0.0,0.03,0.02],
  "arc_mode": "three_point",
  "constant_orientation": true
}
```

通用 `specified_path` 支持三种 JSON 形态:

```json
{
  "type": "plan",
  "kind": "specified_path",
  "robot": "left",
  "duration_s": 1.0,
  "path": {
    "type": "cspace_waypoints",
    "waypoints": [[0.0,0.0,0.0],[0.2,0.0,0.0]]
  }
}
```

```json
{
  "type": "plan",
  "kind": "specified_path",
  "robot": "left",
  "duration_s": 1.0,
  "path": {
    "type": "task_space_segments",
    "segments": [
      {"type":"task_space_line","target_offset":[0.0,0.0,0.05],"orientation_mode":"none"},
      {"type":"task_space_arc","target_offset":[0.0,0.04,0.0],"intermediate_offset":[0.0,0.02,0.02]}
    ]
  }
}
```

规则:

- `joint_positions` 是绝对目标；`joint_deltas` 基于提交 plan 时的状态快照。
- 第一维为 1 时广播到 selected env。
- `joint_names` 可选；传入后按名称映射，未指定关节保持当前 target 或零增量。
- `duration_s` 必须为正；`sample_dt_s` 默认使用 physics dt。
- `load_on_success` 默认 `true`，ready 后自动载入 trajectory buffer；设为 `false` 时只保留 planner result 摘要。
- `task_space_line` / `task_space_arc` 的 `target_offset` 和 `intermediate_offset` 是相对路径起点的偏移，最适合 tiled 多 env 广播；绝对 `target_position` 由 cuMotion specified-path backend 按机器人 base/frame 语义解释。

### 10.5 planner_status / cancel_plan

```json
{"type":"planner_status","wait_timeout_s":0.1}
{"type":"cancel_plan","request_id":"plan-001"}
{"type":"cancel_plan","robot":"left","env_ids":[1]}
{"type":"clear_completed","request_id":"plan-001"}
{"type":"clear_completed"}
```

`planner_status` 会收集已完成请求，并把成功且 `load_on_success=true` 的结果载入 buffer。`cancel_plan` 可以按 `request_id` 取消，也可以按 `robot`/`env_ids` 批量取消 in-flight 请求。正在运行的线程无法被强制中断时，manager 会标记取消；结果回来后作为 `CANCELLED` 处理，不会载入 buffer。
`clear_completed` 只清理 manager 的 completed result 摘要缓存，不影响已载入的 trajectory buffer。

### 10.6 cuMotion 接入边界

代码层提供 `linkerbot_sim.backends.cumotion.tiled_planner.CuMotionJointPlannerBackend` adapter，可把已有 cuMotion `MotionPlanner.plan(MotionRequest | SpecifiedPathRequest)` 接入 `TiledPlannerManager`。真实使用时应传入“每个 worker 独享”的 planner factory，不要多个线程共享同一个带内部状态的 cuMotion planner/context。

交互 CLI 通过 `--planner-backend cumotion` 创建该 adapter。它按 segment 顺序逐 env 调用 cuMotion planner，关节目标段使用 `MotionRequest(goal_q=...)`，task-space/specified-path 段使用 `SpecifiedPathRequest(path=...)`，然后统一重采样到 request 的时间网格。默认 `linear` backend 不做 path conversion、避障或 trajectory optimization。

## 11. Step 返回格式

```json
{
  "event": "step",
  "accepted": true,
  "backend": "isaac",
  "kind": "joint_delta_pos",
  "env_ids": [1],
  "robots": ["left"],
  "ticks": 1,
  "step": 121,
  "time_s": 0.5041666667,
  "episode_steps": [120,121],
  "info": {
    "left": {"command_width": 3}
  }
}
```

这里的 `command_width` 表示本次 action 的 `values` 宽度，即实际写入了 selected robot 的前 3 个 command joints；它不表示 left robot 总共只有 3 个 command joints。完整 command joint 名称请用 `status` 查看。

末端 IK action 的 response 会在 `info` 中包含:

```json
{
  "ik_success": [true,false],
  "ik_position_error": [0.0,0.0],
  "ik_orientation_error": [0.0,0.0]
}
```

IK 失败的 env 会保持 seed/current target，并通过 `ik_success` mask 标出。

## 12. 错误返回

任何解析、shape、backend 能力或 runtime 错误都会返回:

```json
{"event":"rejected","error":"..."}
```

常见错误:

- `unsupported tiled action`: 发送了不属于 action 或 planning 的旧 motion runtime 命令，例如 `ik_pose`。
- `env_ids contains out-of-range env id`: env id 越界。
- `env_ids cannot contain duplicates`: env id 重复。
- `values must have shape ...`: action/state 数组 shape 不符合要求。
- `plan requires joint_positions or joint_deltas`: `plan` 没有给目标。
- `linear planner only supports joint-space segments`: 默认 linear backend 收到了 task-space/specified-path segment；用 `--planner-backend cumotion` 或改用 `load_trajectory`。
- `unknown plan joint_names`: `plan` 中的关节名不属于 selected robot command joints。
- `requires a batched IK solver`: 创建 cuMotion batch IK solver 失败，或当前机器人没有可用 TCP/IK 配置。
- `robots is required when multiple tiled robots are available`: 双臂/多机器人 action 没有在消息层指定机器人。
- `Unknown tiled robot names`: `robots` 中有当前 runtime 未创建的机器人名。

## 13. Foxglove / MCAP telemetry

启动 MCAP:

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --hold \
  --foxglove-mcap-path logs/tiled_interactive.mcap \
  --telemetry-env-ids 0 \
  --telemetry-decimation 2 \
  --telemetry-rate-hz 10
```

启动 live server:

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --hold \
  --foxglove-live-host 127.0.0.1 \
  --foxglove-live-port 8767 \
  --telemetry-rate-hz 10
```

默认 topic:

- `/tiled/state`: JSON，保存 selected state response、`env_ids`、`step/time_s` 和触发事件摘要。
- `/tiled/env_000/joint_states`: Foxglove 标准 JointStates，只为第一个 selected env 发布。
- `/tiled/env_000/scene`: selected env object marker 和 TCP marker，entity id 形如 `env_000/object/Tblock`、`env_000/tcp/left`。

说明:

- 不配置 `--foxglove-live-port` 或 `--foxglove-mcap-path` 时，telemetry 完全关闭。
- telemetry 开启后会先发布一帧 selected state，并按 `--telemetry-rate-hz` 周期发布；每次 command/reset/set_state 后也会发布最新 selected state。`--telemetry-rate-hz 0` 只关闭周期发布，不关闭启动帧和命令触发发布。
- `--telemetry-env-ids` 可以选多个 env，但标准 JointStates 只发布第一个 selected env。
- 完整 selected state 会写入 `/tiled/state`。
- command/get_state 等交互响应触发 telemetry 时，会重新按 `--telemetry-env-ids` 采样 runtime；如果某个 response 或 object state 不包含第一个 selected env，标准 JointStates/marker 会跳过，不会回退到 row 0。
- `reset` 和 `set_state` 事件总会尝试发布，不受 step decimation 跳过。
- 大规模 env benchmark 默认不要打开全量 telemetry。

## 14. 推荐调试流程

1. 启动真实 Isaac tiled 交互进程:

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/tiled_env_interactive.py \
  --env scene3_tiled \
  --default-decimation 1 \
  --tcp-jsonl-port 9003 \
  --hold
```

2. 用 `status` 查看 robot command joints 和 env roots:

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 9003
```

3. 先只控制一个 env:

```json
{"type":"step","kind":"joint_delta_pos","env_ids":[0],"robots":["left"],"values":[[0.01,0,0]],"decimation":1}
```

4. 再广播到所有 env:

```json
{"type":"step","kind":"joint_delta_pos","robots":["left"],"values":[[0.01,0,0]],"decimation":1}
```

5. 用 `get_state` 检查 selected env:

```json
{"type":"get_state","env_ids":[0],"fields":["robots.left.joint_positions","episode_steps","episode_ids"]}
```

6. 需要重新开始某个 episode 时只 reset 该 env:

```json
{"type":"reset","env_ids":[0]}
```

## 15. 与旧交互协议的区别

旧 `dual_arm_interactive.py` 面向单个双臂 scene，支持 motion queue、IK pose、IK offset、task-space path、cancel 等语义。

tiled interactive 面向 batched step-control 和外部 trajectory/planner 调度:

- 每条 action 必须在进入 physics tick 前变成整批关节 target。
- 一个 command step 只允许固定 `decimation` 个 physics ticks。
- 关节空间和指定路径规划通过 async planner manager 生成 ready trajectory，再由 `step_trajectory` 同步回放。
- tiled command 协议不接收旧 `moves`/MoveSpec 队列；旧 motion runtime 的 running/pending 执行队列、`cancel_current`、`estop` 或每 env 不同步推进语义不进入 tiled 热路径。
- `env_ids` 只裁剪 target/state/reset，不能让 env 的仿真时间不同步。
- 末端 `ee_*` 通过 batched cuMotion IK 执行；日常连通性检查建议使用 `status`、`get_state`、`hold` 或一条小幅 `joint_delta_pos`，避免把启动检查和 IK 依赖检查混在一起。

当前实现已经按“planner 是 runtime 外部生产者、trajectory buffer 是同步消费者”的结构落地；后续接 collision-aware cuMotion planner 时仍应保持这个边界，不应把 planner 塞进 `TiledCommandAdapter` 的 step 热路径。
