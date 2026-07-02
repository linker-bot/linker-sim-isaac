# Interactive Simulation Usage

本文说明如何启动双臂可交互仿真，并通过 JSON 消息逐条发送运动命令。

## 启动

最常用的 GUI 调试启动方式：

```bash
env_isaaclab/bin/python scripts/dual_arm_interactive.py --gui --hold
```

启动成功后会打印：

```text
DUAL_ARM_INTERACTIVE_READY
```

此时 Isaac 进程保持运行，外部可以继续发送运动消息。`--hold` 会让窗口和仿真循环在没有命令时继续保持当前姿态。

## Transport

### stdin JSONL

默认启用 stdin。每行发送一个 JSON object：

```bash
env_isaaclab/bin/python scripts/dual_arm_interactive.py --gui --hold
```

然后在终端输入：

```json
{"type":"ik_offset","side":"left","offset":[0.02,0,0.02],"duration_s":1.0}
```

### TCP JSONL

启动 TCP JSONL 服务：

```bash
env_isaaclab/bin/python scripts/dual_arm_interactive.py \
  --gui --hold \
  --tcp-jsonl-host 127.0.0.1 \
  --tcp-jsonl-port 8765
```

客户端每行发送一个 JSON object，服务端每行返回一个 JSON response：

```bash
printf '%s\n' '{"type":"status"}' | nc 127.0.0.1 8765
```

### WebSocket JSON

启动 WebSocket 服务：

```bash
env_isaaclab/bin/python scripts/dual_arm_interactive.py \
  --gui --hold \
  --websocket-host 127.0.0.1 \
  --websocket-port 8766
```

每条 WebSocket message 是一个 JSON object。WebSocket 适合浏览器控制面板使用。

## 通用规则

- 长度单位是 m。
- 角度单位是 rad。
- 姿态默认使用 RPY 欧拉角，格式为 `[roll, pitch, yaw]`，单位 rad。
- `side` 为 `left` 或 `right`。
- `duration_s` 是本条 motion 的执行时长。
- `tcp_frame_name` 可省略；省略时按 `side` 使用启动脚本里的默认 TCP。
- 运动命令串行执行。多个客户端可以同时提交命令，但执行队列仍按顺序播放。
- cuMotion context 是长生命周期对象；普通目标位置、关节角、路径参数变化不会重建 context。

## 姿态字段

交互协议面向人工调试，默认姿态字段使用 RPY 欧拉角。解析层会把 RPY 转成内部 IK/planner 使用的
`wxyz` 四元数。如果只想约束 TCP 位置，不想约束姿态，省略对应姿态字段即可。

| 字段 | 当前支持 | 含义 |
| --- | --- | --- |
| `orientation` | 是 | 目标姿态 RPY，`[roll, pitch, yaw]`。用于 `ik_pose`，以及 `ik_offset` 的 `orientation_mode: "target"`。 |
| `target_orientation` | 是 | 目标姿态 RPY，`[roll, pitch, yaw]`。用于 task-space line/arc 的目标姿态。 |
| `orientation_quat_wxyz` | 是 | 显式四元数姿态，`[w, x, y, z]`。用于调用方已经有四元数的情况。 |
| `target_orientation_quat_wxyz` | 是 | 显式 task-space 四元数姿态，`[w, x, y, z]`。 |

不要在同一条消息里同时写 RPY 字段和对应的 `*_quat_wxyz` 字段；协议会拒绝这种歧义输入。

## 返回事件

提交成功：

```json
{"event":"accepted","id":"cmd-1","state":"pending","queue_index":0}
```

执行中：

```json
{"event":"running","id":"cmd-1","state":"running"}
```

完成：

```json
{"event":"done","id":"cmd-1","state":"done","steps":240}
```

失败：

```json
{"event":"failed","id":"cmd-2","state":"failed","error":"..."}
```

取消：

```json
{"event":"cancelled","id":"cmd-3","state":"cancelled"}
```

## 状态、取消和退出

查询全部命令：

```json
{"type":"status"}
```

查询指定命令：

```json
{"type":"status","id":"cmd-1"}
```

取消 pending 命令或指定 running 命令：

```json
{"type":"cancel","id":"cmd-1"}
```

取消当前 running 命令：

```json
{"type":"cancel_current"}
```

急停。急停会取消当前 running 命令并取消所有 pending 命令：

```json
{"type":"estop"}
```

退出交互循环：

```json
{"type":"quit"}
```

保持当前姿态一段时间：

```json
{"type":"hold","duration_s":0.5}
```

## Arm Motion 示例

### IK Offset

从当前 TCP pose 出发，对 TCP 位置加相对位移并求 IK：

```json
{
  "type": "ik_offset",
  "side": "left",
  "offset": [0.03, 0.0, 0.02],
  "orientation_mode": "current",
  "duration_s": 1.0,
  "phase": "left_lift"
}
```

`orientation_mode` 可选：

| 值 | 含义 |
| --- | --- |
| `current` | 保持当前 TCP 姿态 |
| `target` | 使用 `orientation` 作为目标姿态 |
| `none` | 只约束位置 |

带目标姿态：

```json
{
  "type": "ik_offset",
  "side": "left",
  "offset": [0.03, 0.0, 0.02],
  "orientation_mode": "target",
  "orientation": [0.0, 0.0, 0.0],
  "duration_s": 1.0
}
```

### Absolute IK Pose

指定 TCP 绝对目标位置，可选目标姿态：

```json
{
  "type": "ik_pose",
  "side": "left",
  "position": [0.35, -0.40, 0.10],
  "orientation": [0.3, 0.0, -1.5707],
  "duration_s": 2.0,
  "phase": "left_ik_pose"
}
```

不传 `orientation` 时只约束位置。

只约束位置：

```json
{
  "type": "ik_pose",
  "side": "left",
  "position": [0.35, -0.40, 0.10],
  "duration_s": 2.0,
  "phase": "left_ik_pose_position_only"
}
```

### Absolute C-Space Goal

把选定侧 arm joints 规划到绝对关节角目标：

```json
{
  "type": "cspace_goal",
  "side": "right",
  "joint_positions": [0.2, -0.5, 0.3, -1.0, 0.1, 0.2, 0.0],
  "duration_s": 1.2,
  "phase": "right_cspace_goal"
}
```

`joint_positions` 可以少于 arm joints，未给出的尾部关节保持当前值。

### C-Space Delta

在当前选定侧 arm C-space 上叠加关节增量并规划：

```json
{
  "type": "cspace_delta",
  "side": "right",
  "joint_deltas": [0.1, -0.06, 0.04],
  "duration_s": 1.2,
  "phase": "right_cspace_delta"
}
```

### TCP Line

指定 TCP 直线路径。终点可以用相对 offset：

```json
{
  "type": "task_space_line",
  "side": "left",
  "target_offset": [0.0, 0.0, 0.05],
  "orientation_mode": "none",
  "duration_s": 1.2,
  "phase": "left_tcp_line"
}
```

也可以用绝对位置：

```json
{
  "type": "task_space_line",
  "side": "left",
  "target_position": [0.35, -0.20, 0.45],
  "orientation_mode": "target",
  "target_orientation": [0.0, 0.0, 0.0],
  "duration_s": 1.2
}
```

### TCP Arc

三点圆弧：

```json
{
  "type": "task_space_arc",
  "side": "right",
  "target_offset": [0.0, 0.05, 0.0],
  "intermediate_offset": [0.0, 0.03, 0.02],
  "arc_mode": "three_point",
  "constant_orientation": true,
  "duration_s": 1.6,
  "phase": "right_tcp_arc"
}
```

圆弧会经过 cuMotion task-space path conversion。若路径过大、经过奇异位形、碰撞或关节限制，planner 可能失败；这时缩小 offset 或调整 intermediate point。

## Hand Motion 示例

### Single Hand

手部动作走 controller command-space，不进入 cuMotion planner：

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

`joint_positions` 推荐用 mapping。也可以传 array，array 会按该侧 hand command joints 的顺序写入：

```json
{
  "type": "hand",
  "side": "left",
  "joint_positions": [0.7, 0.6, 0.6, 0.6],
  "duration_s": 0.5
}
```

### Dual Hand

双手同步动作：

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

## Raw Joint Sequence

如果外部控制器已经按 physics step 生成 command-space 关节角 target，可以使用
`raw_joint_sequence` 直接刷新目标。该模式不走 cuMotion，不插值，不做加减速规划，也不检查
速度/加速度限制；每个样本就是要下发给 controller 的 position target。

完整 command-space 矩阵示例：

```json
{
  "type": "raw_joint_sequence",
  "left": {
    "joint_positions": [
      [0.1, 0.2, 0.3],
      [0.11, 0.21, 0.31]
    ]
  },
  "step_interval": 1,
  "phase": "left_policy"
}
```

`step_interval` 表示每个样本保持多少个 physics step。`1` 表示每步刷新一个新 target；
`3` 表示每个 target 连续保持 3 个 physics step 后再切到下一个样本。

也可以用 mapping 只覆盖部分 command-space 关节，未给出的关节保持当前 command 值：

```json
{
  "type": "raw_joint_sequence",
  "right": {
    "joint_positions": {
      "AR5V2_R_arm_joint1": [0.0, 0.02, 0.04],
      "AR5V2_R_arm_joint2": [-0.3, -0.31, -0.32]
    }
  },
  "step_interval": 2
}
```

左右两侧可以同时发送，但样本数必须一致；只发送一侧时，另一侧会保持当前姿态。

## Hand Overlay

所有 arm/cuMotion motion 都可以带 `overlays`。overlay timing 支持：

| `timing` | 含义 |
| --- | --- |
| `sync` | 手部轨迹和 arm 轨迹同步执行 |
| `before` | 先执行手部动作，再执行 arm motion |
| `after` | arm motion 完成后执行手部动作 |

示例：左臂 IK offset 同步闭合左手：

```json
{
  "type": "ik_offset",
  "side": "left",
  "offset": [0.03, 0.0, 0.02],
  "duration_s": 1.0,
  "overlays": [
    {
      "timing": "sync",
      "left_hand": {
        "joint_positions": {
          "L6V1_L_hand_index_mcp_pitch": 0.7,
          "L6V1_L_hand_thumb_cmc_pitch": 0.5
        }
      }
    }
  ]
}
```

`sync` overlay 的 hand `duration_s` 可以省略，默认使用 arm motion 的 `duration_s`。

## 批量命令

一次提交多个 move，队列会把它们作为同一个 command 串行执行：

```json
{
  "id": "pick-sequence-1",
  "moves": [
    {
      "type": "ik_pose",
      "side": "left",
      "position": [0.35, -0.20, 0.40],
      "duration_s": 1.0
    },
    {
      "type": "ik_offset",
      "side": "left",
      "offset": [0.0, 0.0, -0.03],
      "duration_s": 0.8,
      "overlays": [
        {
          "timing": "after",
          "left_hand": {
            "joint_positions": {
              "L6V1_L_hand_index_mcp_pitch": 0.7
            },
            "duration_s": 0.4
          }
        }
      ]
    },
    {
      "type": "ik_offset",
      "side": "left",
      "offset": [0.0, 0.0, 0.05],
      "duration_s": 1.0
    }
  ]
}
```

## 常见问题

### 没有看到机械臂继续动

确认已经打印 `DUAL_ARM_INTERACTIVE_READY`。如果使用 GUI，建议加 `--hold`，让窗口关闭前仿真持续 step。

### 命令 accepted 但没有立刻执行

命令进入同一个串行队列。先发：

```json
{"type":"status"}
```

查看是否有 running 命令占用执行器。

### TCP frame 不存在

不传 `tcp_frame_name` 时会用脚本默认的 `left_demo_tcp` / `right_demo_tcp`。如果传了自定义 TCP，需要确保启动时的 TCP spec 里创建了该 frame。

### WebSocket 启动失败

WebSocket transport 需要 Python 环境里有 `websockets` 包。没有该包时会打印：

```text
DUAL_ARM_INTERACTIVE_WEBSOCKET_UNAVAILABLE
```
