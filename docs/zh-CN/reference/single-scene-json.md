# Single Scene 交互式仿真使用说明

语言：[中文](single-scene-json.md) | [English](../../en/reference/single-scene-json.md)

本文负责普通（非 tiled）`SingleSceneRuntime` 的 JSON transport、消息、selector、生命周期和响应
契约。Checkout 准备与第一次完整请求见 [Single Scene 快速开始](../getting-started/single-scene-quickstart.md)；
全部启动参数、最终默认值和进程标记见 [Single Scene CLI 参考](single-scene-cli.md)。并行克隆环境见
[Tiled Scene JSON 参考](tiled-scene-json.md)。

## 1. Runtime 边界

`SingleSceneRuntime` 管理一个物理 World，以及所选 env profile 声明的全部机器人和对象。它没有
克隆 `env_id` 维度；Single Scene 也不表示单机器人，discovery 可以返回任意配置数量的机器人。启动、
EULA、最终配置、endpoint 启用和进程标记由 [Single Scene CLI 参考](single-scene-cli.md)统一说明。

## 2. Transport 与并发边界

三种 transport 共用 `parse_interactive_motion_message()` 和同一个 `InteractiveMotionQueue`：

| transport | framing | 即时响应 | 后续状态变化 |
|---|---|---|---|
| stdin | 每行一个 JSON object | stdout 一行 JSON | 用 `status` 轮询 |
| TCP JSONL | 每行一个 JSON object | 同一连接返回一行 JSON | 用 `status` 轮询 |
| WebSocket | 每个 text message 一个 JSON object | 同一连接返回 JSON text | 同一连接还会收到 `running/done/failed/cancelled` 等推送 |

所有 Isaac、USD 和 PhysX 读写都在仿真主线程发生。TCP/WebSocket 线程只做 JSON 解析、排队和
等待结果，不能直接访问 `World`、stage 或 articulation。snapshot 请求也先进入主线程队列。
`runtime.interactive.snapshot_timeout_s`（内置 profile 为 30 秒）只约束主线程开始执行前的
等待；此时超时会取消请求，runtime 不会随后执行它。请求一旦被原子标记为
executing，正常运行期间 transport 必须等待真实的成功或失败结果，不会在后台继续写入时返回
假超时；shutdown 抢先时的明确例外见第 9.3 节。

Transport 资源全部有界。TCP 和 WebSocket 共享进程级 `max_connections`，stdin 不占连接
名额；`max_message_bytes` 在 JSON dispatch 前生效，motion/snapshot 请求队列有容量上限，每个
WebSocket 也有独立的有界 event queue。超长、非 UTF-8、重复 key、`NaN`、`Infinity`
和尾随内容都会在到达 Isaac 之前被拒绝。event queue 满时按当前 `reject` 策略拒绝新广播
并累计诊断计数，请求的直接响应与广播队列分开。

### 2.1 stdin 案例

```bash
export OMNI_KIT_ACCEPT_EULA=Y
printf '%s\n' \
  '{"type":"status"}' \
  '{"type":"quit"}' \
| PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
    --env scene1
```

这条预先缓冲的 pipe 只演示 stdin framing 和有序关闭。需要等待运动命令终态时，应使用交互式
stdin、TCP 或 WebSocket，并在发送 `quit` 前确认命令进入终态。

### 2.2 TCP JSONL 案例

启动服务后可用 `nc` 保持一条长连接：

```bash
nc 127.0.0.1 8765
```

然后逐行输入：

```jsonl
{"type":"status"}
{"type":"status","id":"move-1"}
```

TCP 每个请求只有一个直接响应。运动完成事件不会异步插入连接，需要轮询 `status`。

### 2.3 WebSocket 案例

```python
import asyncio
import json

import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:8766") as ws:
        await ws.send(json.dumps({"type": "status"}))
        print(json.loads(await ws.recv()))

        await ws.send(json.dumps({
            "type": "joint_delta",
            "id": "move-1",
            "robot_id": 0,
            "group": "arm",
            "joint_deltas": {"AR5V2_L_arm_joint_1": 0.05},
            "duration_s": 0.5,
        }))
        while True:
            event = json.loads(await ws.recv())
            print(event)
            if event.get("id") == "move-1" and event.get("state") in {
                "done", "failed", "cancelled"
            }:
                break


asyncio.run(main())
```

WebSocket 的 `accepted` 可能同时作为直接响应和队列广播出现，客户端应按 `event + id + state`
做幂等处理，而不是按消息条数推断状态。

## 3. 首次连接：发现会话身份

每次连接新进程后先发送：

```json
{"type":"status"}
```

响应主体示例（省略本节后文说明的 `queue`、`transport` 诊断字段）：

```json
{
  "event": "status",
  "commands": [],
  "current_id": null,
  "estop": false,
  "resetting": false,
  "last_reset": null,
  "config_fingerprint": "...",
  "robots": [
    {
      "robot_id": 0,
      "label": "ar5v2_l6v1_0",
      "robot_profile": "ar5v2_l6v1_l",
      "profile_fingerprint": "...",
      "kind": "arm_hand",
      "supports_planning": true,
      "supports_collision_aware_planning": false,
      "planning_joint_group": "arm",
      "joint_groups": {
        "arm": ["AR5V2_L_arm_joint_1"],
        "hand": ["L6V1_L_hand_index_mcp_pitch"],
        "passive": []
      }
    }
  ],
  "collision": {},
  "planning": {}
}
```

机器人身份字段：

| 字段 | 契约 |
|---|---|
| `robot_id` | 本次会话内由 env `robots[]` 顺序生成的稠密整数 |
| `label` | 稳定配置身份，用于日志和 snapshot 匹配 |
| `robot_profile` | `configs/robots/` profile 名称 |
| `profile_fingerprint` | profile 内容指纹 |
| `kind` | `arm`、`hand` 或 `arm_hand` |
| `supports_planning` | 机器人自身 cuRobo model/joint binding 是否有效，可否创建 cuRobo context |
| `supports_collision_aware_planning` | 已 materialize context 对当前 scene version 是否具备碰撞规划能力 |
| `planning_joint_group` | 当前只能是 `arm` 或 `null` |
| `joint_groups` | articulation 的 arm/hand/passive 显式关节顺序 |

`robot_id` 不是持久 ID。进程重启、env 配置重排或机器人增删后都必须重新发现。请求可带
`robot_label` 作为一致性断言，但不能只用 label、side、role 或 name 选择机器人。

`supports_planning=false` 不等于 `--planner-backend linear` 不可用。`linear` 不创建机器人模型，
只按 arm group 执行显式关节空间插值；Task-space、碰撞检查和 cuRobo C-space 规划仍受上述能力字段
约束。

## 4. 命令总表与状态机

<!-- scene-message-index:start -->
| `type` | 用途 | 主要响应 |
|---|---|---|
| `plan_timeline` | 完整多机器人 timeline | `accepted`，随后进入状态机 |
| `hold` | 单 robot/group hold 简写 | `accepted` |
| `joint_goal` | direct 绝对关节目标 | `accepted` |
| `joint_delta` | direct 相对关节目标 | `accepted` |
| `joint_trajectory` | direct 离散关节轨迹 | `accepted` |
| `plan_cspace_goal` | 当前 planner backend 的 C-space 绝对目标 | `accepted` |
| `plan_cspace_delta` | 当前 planner backend 的 C-space 相对目标 | `accepted` |
| `ik_pose` | cuRobo TCP 位姿目标轨迹规划 | `accepted` |
| `ik_offset` | cuRobo TCP 相对位移轨迹规划 | `accepted` |
| `plan_linear_pose_path` | 顺序 IK 的 TCP 直线路径 | `accepted` |
| `status` | 查询全部或指定命令 | `status` |
| `cancel` | 按 ID 取消 pending/running 命令 | `cancel` |
| `cancel_current` | 中断当前命令 | `cancel_current` |
| `reset` | 主线程安全 reset | `reset`，完成事件为 `reset_done/reset_failed` |
| `get_snapshot` | 读取 runtime-neutral snapshot | `snapshot` |
| `set_snapshot` | 恢复 snapshot | `snapshot_restored/snapshot_failed` |
| `estop` | 取消队列并结束交互循环 | `estop` |
| `quit` | 正常退出 | `quit` |
<!-- scene-message-index:end -->

### 4.1 运动命令的统一响应与终态

所有单 segment 简写和 `plan_timeline` 都进入同一队列。`id` 可省略，此时 runtime 生成
`move-<n>`；生产客户端应显式提供会话内唯一 ID。提交成功的直接响应固定为：

```json
{
  "event": "accepted",
  "id": "timeline-1",
  "state": "pending",
  "queue_index": 0
}
```

`queue_index` 是当前 pending 子队列中的零基位置，不是全局命令序号。随后 WebSocket 会广播状态
变化；TCP/stdin 客户端用 `status` 轮询同样的状态：

```jsonl
{"event":"running","id":"timeline-1","state":"running"}
{"event":"done","id":"timeline-1","state":"done","steps":480}
{"event":"failed","id":"timeline-1","state":"failed","error":"planning failed: ..."}
{"event":"cancelled","id":"timeline-1","state":"cancelled","error":"interrupted","steps":231}
```

| 响应字段 | 说明 |
|---|---|
| `event` | 当前事件；终态为 `done/failed/cancelled` |
| `id` | 请求 `id` 或 runtime 自动生成的 ID |
| `state` | `pending/running/done/failed/cancelled` |
| `queue_index` | 只在 `accepted` 中出现 |
| `error` | 失败原因；成功时为 `null` 或省略 |
| `steps` | 命令终止时的全局 simulation step，不是该命令自己的 tick 数 |

JSON 解析、字段校验或队列提交失败不会创建命令，直接返回：

```json
{"event":"rejected","error":"robot_id is required"}
```

timeline 编译在执行前原子完成。任一 robot、unit、group 或 segment 规划失败时整条命令进入
`failed`，不会先执行已成功编译的其它 track。

### 4.2 `status` 查询格式

不带 `id` 返回全部命令和第 3 节的完整 runtime discovery；带 `id` 时 `commands` 最多一个元素：

```jsonl
{"type":"status"}
{"type":"status","id":"timeline-1"}
```

定向查询响应示例（省略 `queue`、`transport` 的具体内容）：

```json
{
  "event": "status",
  "commands": [
    {
      "id": "timeline-1",
      "state": "running",
      "error": null,
      "steps": null,
      "command_kind": "timeline"
    }
  ],
  "config_fingerprint": "...",
  "robots": [
    {"robot_id": 0, "label": "ar5v2_l6v1_0"}
  ]
}
```

未知 `id` 返回空 `commands`，不是 `rejected`。不带 `id` 的额外字段
`current_id/estop/resetting/last_reset` 分别表示当前执行项、急停状态、reset 是否进行中和最近一次
reset 结果。全部查询（包括带 `id` 的定向查询）都返回 `queue`：其中给出 motion active、pending、
running、terminal history、snapshot 和 reset 请求的实时深度、容量，以及累计拒绝、超时或淘汰计数。
transport 已启动时还返回 `transport`，其中给出 TCP/WebSocket 共享连接数、消息拒绝/超长计数、
WebSocket event queue 深度/容量/丢弃计数和关闭状态；这些计数均为当前进程生命周期内的诊断值。

## 5. 单 segment 简写

单 segment 命令必须带 `robot_id`，最终仍被规范化为一条单 track timeline。共同字段如下：

| 字段 | 必填 | 说明 |
|---|---|---|
| `type` | 是 | 下列 canonical segment type |
| `id` | 建议 | 队列命令 ID |
| `robot_id` | 是 | 当前会话机器人 ID |
| `robot_label` | 否 | ID 对应 label 的一致性断言 |
| `group` | 否 | `arm` 或 `hand`，默认 `arm` |
| `duration_s` | 是 | 非负有限秒数，执行时转换成整数 tick |
| `sample_dt_s` | 否 | 仅规划 segment；规划采样周期，默认 physics dt |
| `coordination` | 否 | `independent` 或 `static_others`，默认 `independent` |
| `force_collision_refresh` | 否 | 是否强制重建当前 context 的碰撞视图 |
| `phase` | 否 | 轨迹/telemetry 阶段名，默认使用 kind |
| `metadata` | 否 | 透传到编译轨迹的 JSON object |

类型专属字段必须使用下表的 canonical 名称：

| `type` | 必填业务字段 | 可选业务字段与约束 |
|---|---|---|
| `hold` | 无 | 不带 joint/task-space 目标 |
| `joint_goal` | `joint_positions` | 一维完整顺序数组或 joint-name 到标量的 mapping |
| `joint_delta` | `joint_deltas` | 一维完整顺序数组或 joint-name 到标量的 mapping |
| `joint_trajectory` | `joint_positions` | `(T,D)`、joint-name 到 T 个样本的 mapping；`times_s` 可选 |
| `plan_cspace_goal` | `joint_positions` | `sample_dt_s/avoid_collisions/force_collision_refresh` |
| `plan_cspace_delta` | `joint_deltas` | `sample_dt_s/avoid_collisions/force_collision_refresh` |
| `ik_pose` | `target_position/reference_frame` | `target_orientation_quat_wxyz/tcp_frame_name/sample_dt_s/avoid_collisions` |
| `ik_offset` | `offset/offset_frame` | `target_orientation_quat_wxyz/tcp_frame_name/sample_dt_s/avoid_collisions` |
| `plan_linear_pose_path` | `target_position+reference_frame` 或 `offset+offset_frame` | 两组目标恰好选一；可带目标姿态、TCP、采样和碰撞字段 |

`sample_dt_s` 只允许规划类 type。`joint_positions` 与 `joint_deltas` 不混用；Single Scene 相对位移字段固定
为 `offset`。所有下列成功案例的立即响应都采用 4.1 节的 `accepted` 形状。

### 5.1 Hold

```json
{
  "type": "hold",
  "id": "hold-arm",
  "robot_id": 0,
  "group": "arm",
  "duration_s": 0.2
}
```

`hold` 不调用 planner，也不改变 group target；它只把编译开始时的 command 保持指定时长。零时长
hold 是合法 no-op，常用于显式表达 unit 内的时序边界。

### 5.2 绝对关节目标

mapping 只修改列出的关节，其余 group 关节保持当前值：

```json
{
  "type": "joint_goal",
  "id": "close-finger",
  "robot_id": 0,
  "group": "hand",
  "joint_positions": {
    "L6V1_L_hand_index_mcp_pitch": 0.7
  },
  "duration_s": 0.8
}
```

也可传按 `status.joint_groups[group]` 完整顺序排列的一维数组。

### 5.3 相对关节目标

```json
{
  "type": "joint_delta",
  "id": "arm-nudge",
  "robot_id": 0,
  "group": "arm",
  "joint_deltas": {
    "AR5V2_L_arm_joint_2": 0.2
  },
  "duration_s": 0.4
}
```

增量以该 segment 编译时的当前 group target 为基准。

### 5.4 离散关节轨迹

```json
{
  "type": "joint_trajectory",
  "id": "hand-curve",
  "robot_id": 0,
  "group": "hand",
  "joint_positions": {
    "L6V1_L_hand_index_mcp_pitch": [0.2, 0.5, 0.7]
  },
  "times_s": [0.1, 0.3, 0.6],
  "duration_s": 0.6
}
```

`joint_positions` 也可为 `(samples, group_dim)` 矩阵。mapping 中每列样本数必须相同；`times_s`
若提供，长度必须等于样本数、全部大于 0 且严格递增。省略 `times_s` 时，样本在
`duration_s` 内均匀分布；若 `duration_s=0`，按 physics dt 为每个样本分配一个间隔。

### 5.5 Planned C-space 目标

这两类请求由 `--planner-backend` 选择的后端执行。以关节 label 作为索引：

```json
{
  "type": "plan_cspace_goal",
  "id": "arm-plan-goal",
  "robot_id": 0,
  "group": "arm",
  "joint_positions": {
    "AR5V2_L_arm_joint_1": 0.2
  },
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

`curobo` 会进入机器人 planning model；`linear` 直接生成从当前关节到目标关节的线性轨迹。
`linear` 不检查 joint limits，且 `avoid_collisions=true` 会明确失败。

默认关节列表：

```jsonl
{"type":"plan_cspace_goal","id":"arm-plan-goal0","robot_id":0,"group":"arm","joint_positions":[1.64,1.2,-1.5707,1.57,-0.37,0.0,0.0],"duration_s":1.0,"avoid_collisions":false}
{"type":"plan_cspace_goal","id":"arm-plan-goal1","robot_id":1,"group":"arm","joint_positions":[1.2,-1.2,-1.5707,1.57,0.37,0.0,0.0],"duration_s":1.0,"avoid_collisions":false}
```

上面是两条独立 JSONL 指令，每行提交一个机器人目标；它们不是一个 JSON array。需要两台机器人
同 tick 同步执行时，应改用一条包含两个 robot unit 的 canonical timeline 请求。

相对目标形式只改 `type` 和目标字段：

```json
{
  "type": "plan_cspace_delta",
  "id": "arm-plan-delta",
  "robot_id": 0,
  "group": "arm",
  "joint_deltas": {"AR5V2_L_arm_joint_1": -0.1},
  "duration_s": 1.0,
  "avoid_collisions": true
}
```

### 5.6 绝对 TCP 位姿规划

```json
{
  "type": "ik_pose",
  "id": "ik-absolute",
  "robot_id": 0,
  "group": "arm",
  "target_position": [0.35, 0.0, 0.40],
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "reference_frame": "world",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "avoid_collisions": false
}
```

姿态可省略，此时只约束 TCP 位置。

### 5.7 相对 TCP 位移规划

```json
{
  "type": "ik_offset",
  "id": "ik-up",
  "robot_id": 0,
  "group": "arm",
  "offset": [0.0, 0.0, 0.05],
  "offset_frame": "robot_base",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 0.8,
  "avoid_collisions": false
}
```

`ik_offset` 只改变位置，保持起点 TCP 姿态。

`ik_pose` 和 `ik_offset` 是 Single Scene JSON 的 segment kind 名称，不表示 runtime 只返回一个 IK
关节解。`TimelinePlanningSession` 会把目标转换到 robot base，构造
`MotionRequest(goal_pose=...)`，再由 `CuroboMotionPlanner.plan()` 调用 cuRobo
`MotionPlanner.plan_pose()` 生成可执行轨迹。直接调用
`CuroboInverseKinematics.solve()` 的单目标 IK 只属于 Python facade，不是这两条 Single Scene 命令的
执行链路。

### 5.8 TCP 直线路径

相对路径：

```json
{
  "type": "plan_linear_pose_path",
  "id": "tcp-line-up",
  "robot_id": 0,
  "group": "arm",
  "offset": [0.0, 0.0, 0.10],
  "offset_frame": "robot_base",
  "duration_s": 1.0,
  "avoid_collisions": false
}
```

绝对路径：

```json
{
  "type": "plan_linear_pose_path",
  "id": "tcp-line-target",
  "robot_id": 0,
  "group": "arm",
  "target_position": [0.35, 0.0, 0.40],
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "reference_frame": "world",
  "duration_s": 1.2
}
```

`target_position` 与 `offset` 必须二选一。未给目标姿态时只约束位置；提供姿态时从当前姿态
Slerp 到目标姿态。后端离散 TCP waypoint，并用前一点 IK 解 warm-start 下一点。

## 6. 完整多机器人 Timeline

层级固定为：

```text
plan_timeline
  robot track                 不同 robot track 从 tick 0 并行
    motion unit               同一 robot 的 units 串行
      arm/hand group track    同一 unit 内共享起点并行
        segment               同一 group 内串行
```

顶层字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `type` | 是 | 固定为 `plan_timeline` |
| `id` | 建议 | 整条 timeline 的队列 ID |
| `tracks` | 是 | 非空 robot track array；同一 `robot_id` 只能出现一次 |
| `coordination` | 否 | `independent/static_others`，默认 `independent`；`coupled` 当前拒绝 |
| `force_collision_refresh` | 否 | boolean；规划前强制同步碰撞视图 |

track 有两种且只能二选一的 canonical 形状：

| track 形状 | 字段 | 语义 |
|---|---|---|
| 简写单 group | `robot_id`、可选 `robot_label/group`、`segments[]` | 整个 track 只有一个串行 group；`group` 默认 `arm` |
| 完整 unit | `robot_id`、可选 `robot_label`、`units[]` | 同一机器人多个 unit 串行；每个 unit 内可并行 arm/hand |

完整 unit 的固定结构为：

```json
{
  "robot_id": 0,
  "robot_label": "ar5v2_l6v1_0",
  "units": [
    {
      "group_tracks": [
        {
          "group": "arm",
          "segments": [
            {"kind": "hold", "duration_s": 0.2}
          ]
        }
      ]
    }
  ]
}
```

timeline 内部 segment 使用 `kind`，字段集合与第 5 节对应的顶层单 segment type 完全相同；例如
顶层 `{"type":"ik_pose",...}` 放入 `segments[]` 后写成 `{"kind":"ik_pose",...}`。unit 的
`group_tracks` 必须非空，同一 unit 不能有两个 writer 写同一 group；`segments` 也必须非空。

完整案例：robot 0 同时移动手臂并闭合手指，robot 1 保持静止。

```json
{
  "type": "plan_timeline",
  "id": "coordinated-1",
  "coordination": "static_others",
  "force_collision_refresh": true,
  "tracks": [
    {
      "robot_id": 0,
      "robot_label": "ar5v2_l6v1_0",
      "units": [
        {
          "group_tracks": [
            {
              "group": "arm",
              "segments": [
                {
                  "kind": "ik_pose",
                  "target_position": [0.35, 0.0, 0.4],
                  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                  "reference_frame": "world",
                  "duration_s": 1.0,
                  "avoid_collisions": true
                }
              ]
            },
            {
              "group": "hand",
              "segments": [
                {"kind": "hold", "duration_s": 0.2},
                {
                  "kind": "joint_goal",
                  "joint_positions": {
                    "L6V1_L_hand_index_mcp_pitch": 0.7
                  },
                  "duration_s": 0.8
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "robot_id": 1,
      "group": "arm",
      "segments": [
        {"kind": "hold", "duration_s": 1.0}
      ]
    }
  ]
}
```

成功提交的直接响应示例：

```json
{
  "event": "accepted",
  "id": "coordinated-1",
  "state": "pending",
  "queue_index": 0
}
```

每个 robot track 必须且只能提供 `units` 或 `segments`。`segments` 是一个单 group unit 的简写，
`group` 默认 `arm`。每个 `group_tracks` 必须非空，且同一 unit 中不能有两个 track 写同一个 group。
同一个 `robot_id` 不能出现两次。

`arm_hand` 的 direct `joint_goal/joint_delta/joint_trajectory` 如果按完整 command-space 提供目标，
编译器会按显式 joint names 拆成同一 unit 的 arm/hand track。规划 segment 只能作用于 `arm`。

Single Scene 与 Tiled Scene 的具体 planner 都消费共享的 canonical request/result。`curobo` 支持 joint-space、IK
和 TCP 直线路径；`linear` 是可直接执行的 joint-space 线性插值策略，支持
`plan_cspace_goal/plan_cspace_delta`，但不做 IK、避碰、关节限位校验或受约束轨迹优化。
`sample_dt_s` 控制规划轨迹的采样网格，最终 scene 执行仍会重采样到 physics grid。

### 6.1 整数 tick 时间规则

- `duration_s` 按 physics dt 向上取整为 tick 数。
- `sample_dt_s` 不改变 physics dt；省略时使用 physics dt。
- 正时长非平凡运动至少占一个 tick；零时长 hold 可以是 no-op。
- 同一 group 的 segment 串行，后一个从前一个终点开始。
- 同一 unit 的结束 tick 取最长 group track，较短 group 保持终点。
- 全局结束 tick 取最长 robot track，较短 robot 保持终点。
- 每个 tick 先计算并应用所有机器人目标，再调用一次 `world.step()`。
- executor 使用编译后的整数 tick，不用浮点秒数重新推导进度。

## 7. 坐标系、四元数与 TCP

Task-space 请求不补默认 frame：

| 字段 | 允许值 | 用途 |
|---|---|---|
| `reference_frame` | `world/env/robot_base/tcp` | 解释绝对 `target_position` 和目标姿态 |
| `offset_frame` | `world/env/robot_base/tcp` | 解释相对 `offset` |
| `tcp_frame_name` | 当前 cuRobo model 已注册 frame | 选择被约束 TCP；省略时使用 robot 默认 TCP |

唯一公开姿态字段为 `target_orientation_quat_wxyz`，顺序严格是 `[w,x,y,z]`。`world` 和 `env`
目标会通过 env origin 与机器人 root pose 转成 robot base-local；position 和 orientation 使用同一个
刚体变换。`tcp` offset 使用当前 TCP 方向解释位移。

TCP 名称必须来自 robot profile 的 `default_tcp_frame/tool_frames/custom_tcps` 最终 materialize 后的
model。JSON 不能临时声明一个 cuRobo 不认识的 frame。

## 8. 碰撞规划与多机器人 Coordination

| `coordination` | 行为 |
|---|---|
| `independent` | 每台机器人独立规划，不把其它 robot track 当作协同对象 |
| `static_others` | 规划当前机器人时，把其它机器人当前几何冻结为静态障碍 |
| `coupled` | 当前无动态 coupled backend，明确拒绝 |

`avoid_collisions=true` 要求同时满足：robot model 有 collision spheres、对应 solver 有 scene collision
checker、collision cache 容量足够、context 已同步当前 scene version。缺任一项都返回能力错误，不会
静默退化为无碰撞规划。

`force_collision_refresh=true` 可放在 timeline 顶层或 segment 上。场景对象位姿改变、snapshot
恢复或需要排查陈旧 collision view 时使用；频繁强制刷新会增加规划延迟。

## 9. 控制、Reset 与 Snapshot

### 9.1 取消与退出

按 ID 取消命令：

```json
{"type":"cancel","id":"timeline-1"}
```

```json
{"event":"cancel","id":"timeline-1","accepted":true}
```

`accepted=true` 只表示找到了可取消的 pending/running 命令。pending 会立即进入 `cancelled`；running
在后续 tick 边界中断并产生终态事件。未知 ID 或已经终止的 ID 返回 `accepted=false`。

取消当前 running 命令而不关心 ID：

```json
{"type":"cancel_current"}
```

```json
{"event":"cancel_current","accepted":true}
```

没有 running 命令时 `accepted=false`。它不取消其它 pending 项。

急停并结束交互循环：

```json
{"type":"estop"}
```

```json
{"event":"estop","accepted":true}
```

`estop` 取消全部 pending，要求当前命令尽快停止，并结束当前交互循环；当前协议没有解除 estop 的
resume 命令，需要重新启动进程。WebSocket 还可能收到队列广播
`{"event":"estop","state":"cancelled"}`。

正常退出：

```json
{"type":"quit"}
```

```json
{"event":"quit","accepted":true}
```

`quit` 唤醒主循环并正常关闭 transport；它不等同于可恢复的暂停，也不保证正在执行的轨迹继续到
终点。

关闭按依赖顺序且每步有界：交互循环先唤醒命令队列、中断 stdin reader 并停止
TCP/WebSocket ingress，再排空 state publisher。`SingleSceneRuntime.close()` 随后关闭保留的异步资源、
planning context、camera output 和 CSV logger，最后才关闭 `SimulationApp`。
`runtime.shutdown.transport_timeout_s`、`state_publisher_timeout_s` 和
`camera_publisher_timeout_s` 分别约束这些等待。超时资源保留原句柄供后续 `close()`
重试；子资源仍存活时不会从它下方关闭 `SimulationApp`。任何 `*_SHUTDOWN_TIMEOUT`
日志都表示关闭尚未完成，即使外层脚本后续打印了最终 step。

### 9.2 Reset

<!-- scene-reset-request:start -->
```json
{
  "type": "reset",
  "id": "reset-1",
  "clear_queue": true,
  "hold_after_reset": true
}
```
<!-- scene-reset-request:end -->

直接响应只确认 reset 已进入主线程安全队列：

<!-- scene-reset-response:start -->
```json
{
  "event": "reset",
  "accepted": true,
  "id": "reset-1",
  "clear_queue": true,
  "hold_after_reset": true
}
```
<!-- scene-reset-response:end -->

`clear_queue=true` 取消 pending 命令；running 命令总会被请求中断。`hold_after_reset=true`
会在 reset 后执行一个短 hold，让 drive target 和物理状态稳定。WebSocket 等待以下终态；TCP
和 stdin 用 `status.last_reset` 轮询：

```jsonl
{"event":"reset_done","id":"reset-1","state":"done","step":120}
{"event":"reset_failed","id":"reset-1","state":"failed","error":"..."}
```

`clear_queue=false` 保留 pending 队列，但 running 命令仍被中断，reset 完成后队列继续执行。省略
`id` 时生成 `reset-<n>`；省略两个 boolean 时都默认为 true。

reset 会在第一次 USD 写入前完整读取并校验 robot/object root pose 回滚值。进入
`World.reset()` 之前的写入失败会逆序补偿；`World.reset()` 已开始后，PhysX 内部状态无法通过
逐 prim 写回来证明完整恢复，任何后续异常都会让 runtime 进入 fail-stop 并请求退出。

### 9.3 Snapshot

Canonical payload、身份匹配、shape、单位、恢复结果和事务规则统一由
[Snapshot 数据与恢复参考](snapshots.md)拥有。本节只定义 Single Scene 消息外层和 admission 行为。

Runtime 读取当前完整场景，没有 env selector：

```json
{"type":"get_snapshot","id":"snapshot-1"}
```

```json
{
  "event": "snapshot",
  "accepted": true,
  "backend": "isaac",
  "id": "snapshot-1",
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

上面的缩略主体只展示消息外层；真实响应会包含完整状态。恢复时应原样传递该 `snapshot`：

```json
{
  "type": "set_snapshot",
  "id": "restore-1",
  "strict": true,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

`set_snapshot` 只接受 `type`、可选 `id`、`snapshot`、可选 `label_map` 和可选 `strict`；
`strict` 默认 true。成功响应为：

```json
{
  "event": "snapshot_restored",
  "accepted": true,
  "backend": "isaac",
  "id": "restore-1",
  "robots": ["robot_0"],
  "objects": [],
  "env_ids": [],
  "partial": false
}
```

配置的 snapshot timeout 只在主线程原子标记 executing 前生效。此后的快照校验或写回失败
返回 `snapshot_failed` 和 `error`，不会在写入后台继续时报告 timeout。执行前等待超时返回：

```json
{"event":"snapshot_timeout","accepted":false,"id":"snapshot-1"}
```

shutdown 是进入 executing 后不再等待终态的唯一例外。关闭请求抢先时，尚未执行的 snapshot 会被
取消并返回 `snapshot_cancelled`；已经 executing 的请求不能再诚实地报告取消，等待方会立即得到
`snapshot_running`：

```jsonl
{"event":"snapshot_cancelled","accepted":false,"reason":"shutdown","id":"snapshot-1"}
{"event":"snapshot_running","accepted":true,"state":"running","id":"restore-1"}
```

`snapshot_running` 不是成功终态，只表示 runtime 已接纳并开始执行，但 shutdown 使当前同步响应不再
等待最终结果。客户端不能把它当成 `snapshot_restored`，也不应自动重发同一个写入请求。

## 10. 端到端流程

可直接执行的 [Single Scene 快速开始](../getting-started/single-scene-quickstart.md)统一负责 discovery、提交、
终态轮询、关闭和进程标记检查。该流程成功后，再回到本文查询精确协议字段。

## 11. 常见拒绝原因

- 缺失、越界或跨会话缓存的 `robot_id`。
- `robot_label` 与当前 ID 的 label 不一致。
- robot track 同时提供或同时缺少 `units/segments`。
- group 不存在、目标包含 group 外关节，或同一 unit 有重复 writer。
- task-space 请求缺失显式 frame，四元数不是 wxyz 四维数组，或 TCP 不在 model 中。
- hand-only robot 或 hand group 请求 cuRobo planning。
- `avoid_collisions=true` 但 collision capability 不完整。
- command `id` 重复，或已触发 estop/quit。
