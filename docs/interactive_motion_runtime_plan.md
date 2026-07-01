# Interactive Motion Runtime Plan

本文给出把一次性 motion test 升级成可交互仿真的修改方案。目标是启动 Isaac 后保持仿真运行，外部逐条发送运动消息，机械臂按消息执行，执行完继续等待下一条。

## 目标

- 客户能启动一个长生命周期仿真进程。
- 第一版同时支持 stdin JSONL、TCP socket JSONL 和 WebSocket JSON。
- 运动命令解析成现有 `IkOffsetMoveSpec`、`SpecifiedPathMoveSpec`、`CSpaceDeltaPlanMoveSpec`、`CumotionMoveSpec`，并补充需要的一等客户 spec。
- 所有 arm/cuMotion motion 都支持 hand overlays；hand motion 也可以作为单独命令执行。
- hand motion 支持单手和双手同步，走 controller command-space，不进入 cuMotion planner。
- 支持命令队列状态查询，返回 `pending`、`running`、`done`、`failed`、`cancelled`。
- 支持取消 pending 命令、取消当前 running 命令，以及急停。
- 使用长生命周期 cuMotion context，避免每条命令重复创建 context。
- Isaac world stepping 和 controller 下发保持在主线程。
- 消息接收和解析可以在后台线程做，但只把解析后的命令放进 queue。
- `src` 继续只处理通用 TCP 变换和通用运动 spec，不引入具体 pinch TCP 语义。

## 非目标

- 第一版不把运动参数写入 YAML。
- 第一版不支持外部直接传任意 Python 对象。
- 第一版不重构 cuMotion planner 内部 pipeline。
- 第一版不做 ROS2。ROS2 可以在同一 command service 之上另接 adapter。
- 第一版不做多命令并行执行。队列可以并发接收，但实际运动仍串行执行。

## 建议入口

保留当前一次性脚本行为，同时新增交互模式：

```bash
env_isaaclab/bin/python scripts/dual_arm_motion_test.py --gui --interactive
env_isaaclab/bin/python scripts/dual_arm_motion_test.py --gui --interactive --tcp-jsonl-port 8765
env_isaaclab/bin/python scripts/dual_arm_motion_test.py --gui --interactive --websocket-port 8766
```

启动后，进程打印：

```text
DUAL_ARM_INTERACTIVE_READY
```

用户可以从 stdin、TCP socket 或 WebSocket 发送 JSON 命令，仿真按队列顺序执行。

## Transport

三种 transport 共享同一套 command schema 和同一个 command queue。

### stdin JSONL

- 适合本地调试。
- 每行一个 JSON object。
- 输出仍打印到 stdout。

### TCP socket JSONL

- 适合外部控制程序。
- 一个 TCP 连接中每行一个 JSON object。
- 服务端每处理一行，回写一行 JSON response。
- 多连接可以同时提交命令，但命令进入同一个队列串行执行。

启动参数建议：

```bash
--tcp-jsonl-host 127.0.0.1 --tcp-jsonl-port 8765
```

### WebSocket JSON

- 适合浏览器控制面板。
- 每条 WebSocket message 是一个 JSON object。
- 服务端用 JSON message 推送 accepted/running/done/failed/status event。

启动参数建议：

```bash
--websocket-host 127.0.0.1 --websocket-port 8766
```

## JSONL 协议

每条消息是一个 JSON object。stdin/TCP JSONL 中每行一条；WebSocket 中每个 message 一条。支持四类消息：

### 单条 move

```json
{"type":"ik_offset","side":"left","offset":[0.03,0,0.02],"duration_s":1.0}
```

### 批量 moves

```json
{
  "moves": [
    {"type":"ik_pose","side":"left","position":[0.35,-0.2,0.4],"duration_s":1.0},
    {
      "type":"ik_offset",
      "side":"left",
      "offset":[0.03,0,0.02],
      "duration_s":1.0,
      "overlays":[
        {
          "timing":"sync",
          "left_hand":{
            "joint_positions":{"L6V1_L_hand_index_mcp_pitch":0.7},
            "duration_s":1.0
          }
        }
      ]
    },
    {"type":"task_space_line","side":"left","target_offset":[0,0,0.05],"duration_s":1.2}
  ]
}
```

### 控制消息

```json
{"type":"quit"}
```

或：

```json
{"type":"hold","duration_s":0.5}
```

### 状态消息

```json
{"type":"status"}
```

或查询指定命令：

```json
{"type":"status","id":"cmd-12"}
```

### 取消和急停

取消某个命令：

```json
{"type":"cancel","id":"cmd-12"}
```

取消当前 running 命令：

```json
{"type":"cancel_current"}
```

急停：

```json
{"type":"estop"}
```

## 通用字段

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `type` | 是 | 运动类型或控制类型 |
| `side` | move 必填 | `left` 或 `right` |
| `tcp_frame_name` | 否 | 使用哪个 TCP；不传则按 `side` 使用启动时默认 TCP |
| `duration_s` | move 必填 | 执行时长 |
| `phase` | 否 | 日志/轨迹阶段名 |
| `id` | 否 | 客户指定 command id；不传由服务端生成 |
| `overlays` | 否 | arm/cuMotion motion 的 hand overlay 列表 |

## 支持的 move 类型

### `hand`

单手 controller command-space motion，不经过 cuMotion。

```json
{
  "type": "hand",
  "side": "left",
  "joint_positions": {
    "L6V1_L_hand_index_mcp_pitch": 0.7,
    "L6V1_L_hand_thumb_cmc_pitch": 0.5
  },
  "duration_s": 0.5,
  "phase": "left_hand_close"
}
```

字段：

- `side`：`left` 或 `right`。
- `joint_positions`：mapping 或 array。mapping 键是 controller command-space 关节名；array 表示按该侧 hand command joints 约定顺序给值。
- `duration_s`：执行时长。

### `dual_hand`

双手同步 controller command-space motion，不经过 cuMotion。

```json
{
  "type": "dual_hand",
  "left": {
    "joint_positions": {
      "L6V1_L_hand_index_mcp_pitch": 0.7
    }
  },
  "right": {
    "joint_positions": {
      "L6V1_R_hand_index_mcp_pitch": 0.7
    }
  },
  "duration_s": 0.5,
  "phase": "dual_hand_close"
}
```

字段：

- `left`：可选左手目标。
- `right`：可选右手目标。
- `duration_s`：双手同步执行时长；子项不传时使用这个时长。
- `phase`：可选阶段名。

### `ik_pose`

映射到 `CumotionMoveSpec(request=IKRequest(...))`。这是绝对 TCP 目标。

```json
{
  "type": "ik_pose",
  "side": "left",
  "tcp_frame_name": "left_demo_tcp",
  "position": [0.35, -0.20, 0.40],
  "orientation": [1.0, 0.0, 0.0, 0.0],
  "position_tolerance": 0.0001,
  "orientation_tolerance": 0.001,
  "avoid_collisions": false,
  "duration_s": 1.0,
  "phase": "left_ik_pose"
}
```

带 hand overlay：

```json
{
  "type": "ik_pose",
  "side": "left",
  "position": [0.35, -0.20, 0.40],
  "duration_s": 1.0,
  "overlays": [
    {
      "timing": "sync",
      "left_hand": {
        "joint_positions": {
          "L6V1_L_hand_index_mcp_pitch": 0.7,
          "L6V1_L_hand_thumb_cmc_pitch": 0.5
        },
        "duration_s": 1.0
      }
    }
  ]
}
```

字段：

- `position`：TCP 绝对目标位置。
- `orientation`：可选 TCP 绝对目标姿态，wxyz 四元数。不传时只约束位置。
- `position_tolerance`：可选位置容差。
- `orientation_tolerance`：可选姿态容差。
- `avoid_collisions`：可选 collision-aware IK。

### Hand overlays

所有 arm/cuMotion move 都可带 `overlays`：

```json
"overlays": [
  {
    "timing": "sync",
    "left_hand": {
      "joint_positions": {
        "L6V1_L_hand_index_mcp_pitch": 0.7
      },
      "duration_s": 1.0
    },
    "right_hand": {
      "joint_positions": {
        "L6V1_R_hand_index_mcp_pitch": 0.7
      },
      "duration_s": 1.0
    }
  }
]
```

字段：

- `timing`：`sync`、`before` 或 `after`。
- `left_hand`：可选左手目标。
- `right_hand`：可选右手目标。
- hand 子项的 `duration_s` 可省略；省略时 `sync` 使用 arm move 的 duration，`before/after` 使用 overlay 所在命令的默认 duration。

### `ik_offset`

映射到 `IkOffsetMoveSpec`。第一版现有 spec 只支持位置 offset；交互协议应预留目标姿态字段，并在实现时同步扩展 spec。

```json
{
  "type": "ik_offset",
  "side": "left",
  "tcp_frame_name": "left_demo_tcp",
  "offset": [0.03, 0.0, 0.02],
  "orientation_mode": "current",
  "orientation": [1.0, 0.0, 0.0, 0.0],
  "duration_s": 1.0,
  "phase": "left_ik_offset"
}
```

字段：

- `offset`：当前 TCP pose 的相对位置偏移。
- `orientation_mode`：可选，`current` 保持当前姿态，`target` 使用 `orientation`，`none` 只约束位置。
- `orientation`：`orientation_mode="target"` 时的目标姿态，wxyz 四元数。

### `cspace_goal`

表示把选定侧 arm joints 规划到给定绝对角度。实现时建议新增 `CSpaceGoalPlanMoveSpec`；在新增 spec 之前，也可以解析成高级 `CumotionMoveSpec + MotionRequest(goal_q=...)`。

```json
{
  "type": "cspace_goal",
  "side": "right",
  "tcp_frame_name": "right_demo_tcp",
  "joint_positions": [0.2, -0.5, 0.3, -1.0, 0.1, 0.2, 0.0],
  "duration_s": 1.2,
  "phase": "right_cspace_goal"
}
```

字段：

- `joint_positions`：选定侧 arm C-space 的绝对目标角度。可以少于 arm joints，未给出的尾部关节保持当前值。

### `cspace_delta`

映射到 `CSpaceDeltaPlanMoveSpec`。

```json
{
  "type": "cspace_delta",
  "side": "right",
  "joint_deltas": [0.1, -0.06, 0.04],
  "duration_s": 1.2,
  "phase": "right_cspace_delta"
}
```

字段：

- `joint_deltas`：选定侧 arm C-space 的关节增量。

### `task_space_line`

映射到 `SpecifiedPathMoveSpec(TaskSpacePath(TcpLineSegment(...)))`。

```json
{
  "type": "task_space_line",
  "side": "left",
  "target_offset": [0.0, 0.0, 0.05],
  "orientation_mode": "none",
  "duration_s": 1.2
}
```

字段：

- `target_offset` 或 `target_position` 二选一。
- `orientation_mode` 可选，默认 `current`。
- `target_orientation` 在 `orientation_mode="target"` 时需要。

绝对直线终点示例：

```json
{
  "type": "task_space_line",
  "side": "left",
  "target_position": [0.35, -0.20, 0.45],
  "orientation_mode": "target",
  "target_orientation": [1.0, 0.0, 0.0, 0.0],
  "duration_s": 1.2
}
```

### `task_space_arc`

映射到 `SpecifiedPathMoveSpec(TaskSpacePath(TcpArcSegment(...)))`。

```json
{
  "type": "task_space_arc",
  "side": "right",
  "target_offset": [0.0, 0.05, 0.0],
  "intermediate_offset": [0.0, 0.03, 0.02],
  "arc_mode": "three_point",
  "constant_orientation": true,
  "duration_s": 1.6
}
```

字段：

- `target_offset` 或 `target_position` 二选一。
- `arc_mode` 默认 `three_point`，可选 `tangent`。
- `intermediate_offset` 或 `intermediate_position` 在 `three_point` 模式下二选一。
- `constant_orientation` 默认 `true`。
- `target_orientation` 可选。

绝对圆弧终点示例：

```json
{
  "type": "task_space_arc",
  "side": "right",
  "target_position": [0.40, 0.15, 0.35],
  "intermediate_position": [0.38, 0.12, 0.38],
  "arc_mode": "three_point",
  "constant_orientation": true,
  "target_orientation": [1.0, 0.0, 0.0, 0.0],
  "duration_s": 1.6
}
```

## 输出协议

stdout 继续保留稳定单行日志，便于人工 grep：

```text
DUAL_ARM_INTERACTIVE_READY
DUAL_ARM_INTERACTIVE_ACCEPTED id=3 moves=2
DUAL_ARM_INTERACTIVE_RUNNING id=3
DUAL_ARM_INTERACTIVE_DONE id=3 steps=720
DUAL_ARM_INTERACTIVE_FAILED id=4 error=...
DUAL_ARM_INTERACTIVE_CANCELLED id=5
DUAL_ARM_INTERACTIVE_ESTOP id=6
DUAL_ARM_INTERACTIVE_EXIT
```

如果输入 JSON 无法解析：

```text
DUAL_ARM_INTERACTIVE_REJECTED error=invalid_json line=...
```

TCP socket 和 WebSocket 应返回 JSON response/event：

```json
{"event":"accepted","id":"cmd-3","state":"pending","queue_index":2}
{"event":"running","id":"cmd-3","state":"running"}
{"event":"done","id":"cmd-3","state":"done","steps":720}
{"event":"failed","id":"cmd-4","state":"failed","error":"..."}
{"event":"cancelled","id":"cmd-5","state":"cancelled"}
{"event":"status","commands":[{"id":"cmd-3","state":"done"}]}
```

## 代码结构建议

新增模块：

```text
src/linkerbot_sim/app/interactive_motion_protocol.py
src/linkerbot_sim/app/interactive_motion_queue.py
src/linkerbot_sim/app/interactive_motion_transports.py
src/linkerbot_sim/app/dual_arm_interactive_motion.py
```

### `interactive_motion_protocol.py`

职责：

- 解析 JSON dict。
- 校验基础字段。
- 将消息转换为 `MoveSpec` tuple 或控制命令。
- 不 import Isaac。
- 不创建 runtime。
- 不调用 cuMotion。

建议公开：

```python
@dataclass(frozen=True)
class InteractiveMotionCommand:
    kind: Literal[
        "moves",
        "hold",
        "status",
        "cancel",
        "cancel_current",
        "estop",
        "quit",
    ]
    command_id: str | None = None
    moves: tuple[MoveSpec, ...] = ()
    duration_s: float | None = None
    cancel_id: str | None = None


def parse_interactive_motion_message(
    message: Mapping[str, object],
    *,
    default_tcp_by_side: Mapping[str, str],
) -> InteractiveMotionCommand:
    ...
```

motion protocol 还应定义 hand/overlay spec：

```python
@dataclass(frozen=True)
class HandMoveSpec:
    side: Literal["left", "right"]
    joint_positions: Mapping[str, float] | tuple[float, ...]
    duration_s: float
    phase: str | None = None


@dataclass(frozen=True)
class DualHandMoveSpec:
    left: HandMoveSpec | None = None
    right: HandMoveSpec | None = None
    duration_s: float | None = None
    phase: str | None = None


@dataclass(frozen=True)
class CommandOverlaySpec:
    timing: Literal["sync", "before", "after"] = "sync"
    left_hand: HandMoveSpec | None = None
    right_hand: HandMoveSpec | None = None
```

### `interactive_motion_queue.py`

职责：

- 生成 command id。
- 维护命令状态：`pending`、`running`、`done`、`failed`、`cancelled`。
- 支持入队、取下一条、状态查询。
- 支持取消 pending 命令。
- 维护 `cancel_current` 和 `estop` 标志，供执行循环检查。

建议公开：

```python
CommandState = Literal["pending", "running", "done", "failed", "cancelled"]


@dataclass
class QueuedMotionCommand:
    command_id: str
    moves: tuple[MoveSpec, ...]
    state: CommandState = "pending"
    error: str | None = None


class InteractiveMotionQueue:
    def submit(self, command: InteractiveMotionCommand) -> QueuedMotionCommand:
        ...

    def next_pending(self) -> QueuedMotionCommand | None:
        ...

    def cancel(self, command_id: str) -> bool:
        ...

    def request_cancel_current(self) -> None:
        ...

    def request_estop(self) -> None:
        ...
```

### `interactive_motion_transports.py`

职责：

- stdin JSONL reader。
- TCP socket JSONL server。
- WebSocket JSON server。
- 只解析 JSON、调用 protocol、提交 queue、返回 accepted/rejected/status。
- 不碰 Isaac runtime。
- 不调用 cuMotion。

实现建议：

- stdin reader 用后台 thread。
- TCP socket server 用后台 thread，每个 connection 一个轻量 client thread。
- WebSocket 可以用 `websockets` 包；如果运行环境没有该依赖，则启动时报清晰错误。
- 三种 transport 共享 `InteractiveMotionQueue`。

### `dual_arm_interactive_motion.py`

职责：

- 持有 `DualRobotAppRuntime`。
- 持有 `DualArmTcpSpec`、`cumotion_profile`、`dual_arm_profile`。
- 启动并持有长生命周期 cuMotion context。
- 从 queue 取命令并执行。
- 等待命令时保持当前姿态并刷新 GUI。
- 在每个 trajectory sample 前检查 `cancel_current` 和 `estop`。
- 在执行 arm trajectory 时同步叠加 `CommandOverlaySpec(timing="sync")` 中的 hand command trajectory。
- 在 arm move 前后执行 `timing="before"` / `timing="after"` hand overlay。
- 支持单独 `HandMoveSpec` 和 `DualHandMoveSpec`。
- 将状态事件广播给 transports。

建议公开：

```python
@dataclass
class DualArmInteractiveMotionSession:
    runtime: DualRobotAppRuntime
    tcp: DualArmTcpSpec
    cumotion_profile: str
    dual_arm_profile: str
    step: int = 0

    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def execute_moves(self, moves: Sequence[MoveSpec]) -> int:
        ...
        return self.step

    def hold_once(self, duration_s: float = 0.05) -> int:
        ...
```

## 长生命周期 cuMotion Context

交互模式不应该每条命令都调用一次完整的 `run_dual_arm_cumotion_motion(...)` 并重建 context。应把当前函数中的 context 创建逻辑拆成可复用 session：

```text
DualArmCuMotionExecutionSession.open()
  ├── load_profile_yaml("cumotion", ...)
  ├── merged_robot_config_with_cumotion_profile(...)
  ├── robot_cumotion_config(...)
  ├── load_dual_arm_semantic_config(...)
  ├── tcp_transforms_from_dual_spec(tcp)
  ├── make_cumotion_context(...)
  ├── context.clear_collision_world()
  ├── joint_names = context.joint_names()
  └── partitions = DualArmJointPartitions.from_joint_names(...)
```

每条命令执行时只刷新：

- 当前左右 controller command。
- 当前 dual C-space `current_q`。
- 当前 TCP pose 或 planner request。
- 当前 hand command target。
- command queue 状态和累计 `step`。

### 什么时候不需要重建 context

只要以下内容不变，就不需要重建：

- `cumotion_profile`
- `dual_arm_profile`
- robot/env 中影响 cuMotion URDF/XRDF 的配置
- 左右 robot root pose
- TCP frame name
- TCP 相对末端的 `xyz/rpy`
- TCP parent/flange frame

也就是说，普通运动命令只改变目标 pose、目标关节角、路径段、速度/时长时，不需要重建 context。

### 什么时候必须重建 context

以下变化需要重建 context：

- 更换 TCP frame name。
- 修改 TCP 相对末端的 `xyz/rpy`。
- 更换 robot/cumotion/dual-arm profile。
- 更换 scene root pose 或 robot asset。
- 重新生成/替换用于 cuMotion 的 URDF/XRDF。
- 需要重新注入或删除 TCP fixed link。

如果只是“选择已经注入的 left/right TCP”，不需要重建；planner/IK 调用时传不同 `tcp_frame_name` 即可。

### Collision world 刷新

第一版可以在 session open 时 `context.clear_collision_world()`。如果后续要支持动态障碍物，需要增加显式消息：

```json
{"type":"refresh_collision_world"}
```

该消息只刷新 collision world，不一定需要重建 robot context。

## 主线程和队列

transport reader 放后台线程：

```text
transport thread:
  readline()
  json.loads(...)
  parse_interactive_motion_message(...)
  queue.put(command)

main thread:
  while app.is_running():
    if command available:
      execute command
    else:
      hold_once()
```

关键约束：

- Isaac `world.step(...)` 只在 main thread 里调用。
- cuMotion context、IK、planner 只在 main thread 里使用。
- 后台线程不碰 runtime、articulation、controller、world、cuMotion context。

## 取消和急停

取消/急停需要执行层可中断。具体做法：

- 将当前 `DualCommandPositionTrajectoryStep.run(...)` 的长 for-loop 拆成可逐样本 step 的 executor，或给 `run(...)` 增加 `should_stop` callback。
- 每个 trajectory sample 前检查：
  - `queue.estop_requested`
  - `queue.cancel_current_requested`
  - `simulation_app.is_running()`
- `cancel_current`：停止当前 trajectory，保持当前关节位置，状态标记为 `cancelled`。
- `estop`：立即停止当前 trajectory，向 controller 下发当前位置/零速度保持，清空 pending queue，状态广播 `estop`。

第一版急停是仿真控制急停，不是硬件安全急停。

## Hand Execution

hand motion 不需要 cuMotion context。执行层应把 hand targets 映射到 controller command-space：

- mapping 输入按 joint name 更新目标。
- array 输入按该侧 hand command joints 约定顺序更新目标。
- 未指定的 command joints 保持当前值。
- hand trajectory 使用与 arm trajectory 相同的 physics step 网格做 smooth interpolation。

同步 overlay 的执行方式：

```text
arm trajectory sample i:
  arm target from cuMotion trajectory
  hand target from overlay interpolation at same normalized time
  apply left/right controller targets
  world.step(render=...)
```

单独 `hand` / `dual_hand` 可复用 `DualCommandPositionTargetStep` 或新增更明确的 hand command step。关键是不要把 hand joints 混进 cuMotion C-space。

## `dual_arm_motion_test.py` 修改点

新增参数：

```python
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--tcp-jsonl-host", default="127.0.0.1")
parser.add_argument("--tcp-jsonl-port", type=int, default=None)
parser.add_argument("--websocket-host", default="127.0.0.1")
parser.add_argument("--websocket-port", type=int, default=None)
```

入口分支：

```python
if args.interactive:
    run_interactive_dual_arm_motion(
        runtime,
        tcp=tcp,
        cumotion_profile=args.cumotion_profile,
        dual_arm_profile=args.dual_arm_profile,
        stdin_enabled=True,
        tcp_jsonl_host=args.tcp_jsonl_host,
        tcp_jsonl_port=args.tcp_jsonl_port,
        websocket_host=args.websocket_host,
        websocket_port=args.websocket_port,
    )
else:
    steps = run_dual_arm_cumotion_motion(...)
```

`--interactive` 建议隐式保持 app 生命周期。用户通常会配合 `--gui` 使用；headless 下也可以用于自动化 stdin 测试。

## 第一版实现步骤

1. 新增或扩展 motion spec：
   - `IkOffsetMoveSpec` 增加可选 `orientation_mode` 和 `target_orientation`，或在协议层将带姿态的 offset 转成 `CumotionMoveSpec(IKRequest)`。
   - 新增 `CSpaceGoalPlanMoveSpec`，表达选定侧绝对关节角目标。
   - 新增 `HandMoveSpec`、`DualHandMoveSpec`、`CommandOverlaySpec`。
   - 所有 arm/cuMotion move 增加 `overlays: tuple[CommandOverlaySpec, ...] = ()`。
2. 新增 `interactive_motion_protocol.py`。
3. 为协议解析添加单元测试：
   - `hand`
   - `dual_hand`
   - arm move 带 `overlays`
   - `ik_pose`
   - `ik_offset`
   - `ik_offset` 带 `orientation_mode="target"`
   - `cspace_goal`
   - `cspace_delta`
   - `task_space_line`
   - `task_space_line` 使用 `target_position`
   - `task_space_arc`
   - `task_space_arc` 使用 `target_position` 和 `intermediate_position`
   - 批量 `moves`
   - `status`
   - `cancel`
   - `cancel_current`
   - `estop`
   - `quit`
   - 缺少必填字段时失败
4. 新增 `interactive_motion_queue.py`，覆盖状态流转、取消 pending、取消 current、急停清队列。
5. 改造双臂 cuMotion 执行逻辑，抽出长生命周期 context/session。
6. 改造 trajectory 执行层，支持 `should_stop` callback 或逐样本 executor，并支持 hand overlay 同步插值。
7. 新增 `interactive_motion_transports.py`：
   - stdin JSONL
   - TCP socket JSONL
   - WebSocket JSON
8. 新增 `dual_arm_interactive_motion.py`，组装 queue、transports、session loop。
9. 在 `scripts/dual_arm_motion_test.py` 增加 `--interactive`、socket、WebSocket 参数。
10. 更新 README 和调用链文档。
11. 手动验证：

```bash
env_isaaclab/bin/python scripts/dual_arm_motion_test.py --gui --interactive
```

输入：

```json
{"type":"ik_offset","side":"left","offset":[0.03,0,0.02],"duration_s":1.0}
{"type":"ik_pose","side":"left","position":[0.35,-0.2,0.4],"orientation":[1,0,0,0],"duration_s":1.0}
{"type":"hand","side":"left","joint_positions":{"L6V1_L_hand_index_mcp_pitch":0.7},"duration_s":0.5}
{"type":"dual_hand","left":{"joint_positions":{"L6V1_L_hand_index_mcp_pitch":0.0}},"right":{"joint_positions":{"L6V1_R_hand_index_mcp_pitch":0.0}},"duration_s":0.5}
{"type":"cspace_goal","side":"right","joint_positions":[0.2,-0.5,0.3,-1.0,0.1,0.2,0],"duration_s":1.0}
{"type":"task_space_line","side":"right","target_offset":[0,0,0.03],"orientation_mode":"none","duration_s":1.0}
{"type":"status"}
{"type":"cancel_current"}
{"type":"quit"}
```

TCP JSONL 验证：

```bash
env_isaaclab/bin/python scripts/dual_arm_motion_test.py --gui --interactive --tcp-jsonl-port 8765
```

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 8765
```

WebSocket 验证可用浏览器控制台或轻量 client 发送同一 JSON schema。
