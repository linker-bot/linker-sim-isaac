# 控制与轨迹选择指南

语言：[中文](control-and-trajectories.md) | [English](../../en/guides/control-and-trajectories.md)

本文帮助应用用户选择正确的控制与轨迹路径，不重新定义消息字段。精确 schema 见
[Single Scene JSON 参考](../reference/single-scene-json.md)和
[Tiled Scene JSON 参考](../reference/tiled-scene-json.md)。

## 选择执行路径

| 需求 | 使用 | Runtime |
| --- | --- | --- |
| 对一个机器人/group 保持、移动关节或执行采样关节曲线 | Single Scene 单 segment 命令 | Single Scene |
| 让多个机器人和 arm/hand group 在共同 tick 轴上启动 | `plan_timeline` | Single Scene |
| 通过关节空间或 task space 规划一个 Single Scene arm 目标 | Single Scene planning segment | Single Scene |
| 对 selected clone env row 应用固定 tick 命令 | `step` | Tiled Scene |
| 载入已有轨迹并由调用方显式推进 | `load_trajectory` 后接 `step_trajectory` | Tiled Scene |
| 追加按名称选择的稀疏 hand subtrack | `hand` | Tiled Scene |
| 不阻塞 physics 地规划，检查完成后再载入和回放 | `plan`、`planner_status`、`step_trajectory` | Tiled Scene |

Single Scene timeline 与 Tiled Scene playback 解决不同的同步问题。Single Scene 在执行前原子编译全部 track，并在一次
World step 前应用所有机器人 target。Tiled Scene playback 为每个 robot/env row 持有队列，只在调用方发送
`step_trajectory` 时推进。

## Command Space 与关节顺序

公开关节向量不是任意 articulation 数组：

- Single Scene group 向量遵循 `status.robots[].joint_groups.arm` 或 `.hand`。
- Tiled Scene command 向量遵循 `status.robots[].command_joints`。
- Planning 向量遵循所选 backend 的 planning-joint 顺序。
- `JointTrajectory` 的列顺序严格由 `joint_names` 定义。

请求有意控制子集时，优先使用名称 mapping 或 `joint_names`。Tiled Scene `step` 的 prefix action 写入前
`D` 个 command column，其余 target 保持不变。Controller/runtime 负责把 active command joint 扩展
到 mimic follower；除非接口明确要求，client 不应单独发送 follower target。

旋转关节位置单位为 rad，移动关节位置单位为 m；速度使用对应单位每秒。Effort 量纲由 PhysX
关节类型决定。

## Single Scene Timeline 模型

Single Scene 使用固定层级：

```text
timeline
  robot track                 从全局 tick 0 并行
    motion unit               同一机器人内串行
      arm/hand group track    同一 unit 内并行
        segment               同一 group 内串行
```

一个 unit 在最长 group track 结束时结束，较短 group 保持末端 target；整条 timeline 在最长 robot
track 结束时结束。编译是原子的：任一规划或校验失败都会阻止全部 track 启动。

最小协同请求：

```json
{
  "type": "plan_timeline",
  "id": "arm-and-hand",
  "tracks": [
    {
      "robot_id": 0,
      "units": [
        {
          "group_tracks": [
            {
              "group": "arm",
              "segments": [
                {
                  "kind": "joint_goal",
                  "joint_positions": {"AR5V2_L_arm_joint_1": 0.2},
                  "duration_s": 0.5
                }
              ]
            },
            {
              "group": "hand",
              "segments": [
                {"kind": "hold", "duration_s": 0.5}
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

单 track 使用 Single Scene 单 segment 命令；只要需求包含同 tick 协同，就使用 `plan_timeline`。分别发送
多条 JSONL 请求不会让它们同时开始。

## Tiled Scene 同步 `step`

`step` 会完成 target 转换并推进固定 physics tick 后再返回。所有 env-scoped 请求都显式提供
`env_ids`；多机器人场景还必须提供该接口要求的 robot selector。

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

闭环 policy 根据最新观测生成下一 target 时使用 `step`。Joint action 支持绝对/相对 command-space
prefix。末端 action 使用 batch IK；`ee_linear_path` 会在第一次 physics 写入前算完全部 waypoint。

所有 env 共享同一个 World step，因此未选 env 也会在保持最新 target 的同时推进时间。

## Tiled Scene Trajectory Buffer

已有轨迹采样，或异步 planner 结果需要稍后回放时使用 buffer：

1. `load_trajectory` 原子校验并 stage 全部 selected row。
2. `step_trajectory` 按显式 physics tick 推进 playback。
3. `trajectory_status` 返回 active、queued、completed、capacity 和 rejection 数据。
4. `clear_trajectory` 删除 selected robot/env entry。

Buffer 按 env 限制 queue depth、sample 数和 duration。`replace` 只校验 replacement sequence；append
校验 existing 与 new 的总和。容量不足会拒绝完整 selected-env load，不会淘汰 active trajectory。

`hand` 是稀疏 named-joint 便利接口，可以追加 hand subtrack，并在 playback 真正开始时避免覆盖 arm
末端。它不能替代 Single Scene 中同 tick 的 arm/hand timeline。

## Tiled Scene 异步规划

异步请求具有独立的 planning 与 playback 生命周期：

```text
plan submission
  -> queued planner request
  -> planner_status dispatch/collect
  -> optional atomic playback load
  -> step_trajectory playback
```

`plan` 只入队。`planner_status` 和 `step_trajectory` 会 dispatch/collect ready work。
`load_on_success=true` 的成功结果只有在 buffer admission 通过后才会载入。必须同时检查 `ready`、
`loaded` 和 `load_rejected`；planner success 不代表 playback 一定有容量。

取消和 completed-result 管理应使用 request ID。Reset、状态恢复等重叠 mutation 会取消受影响
robot/env row 的 stale planning work。

## 时间与插值

- Single Scene 将 `duration_s` 转成 `ceil(duration_s / physics_dt)` 个 tick。
- Single Scene `sample_dt_s` 控制 planning grid，不改变 World physics dt。
- Tiled Scene joint `step` 使用正整数 `decimation` physics tick。
- Tiled Scene `ee_linear_path` 接受逻辑 `duration_s` 或显式 `decimation`，两者不能同时提供。
- Tiled Scene trajectory 多采样数据的时间必须有限且严格递增。
- `linear` 使用均匀 progress；`smoothstep` 平滑 progress，但不改变请求的几何端点。

响应中的实际 tick 数是权威结果。不要用 wall clock 推导完成，也不要假设十进制 duration 能被
physics dt 整除。

## Frame、Orientation 与 TCP

公开位置单位为 m，四元数顺序统一为 `wxyz`。Frame 字段由具体接口定义：

- Single Scene task-space 命令显式声明 `world`、`env`、`robot_base` 或文档规定的 offset frame。
- Tiled Scene 同步 named end-effector target 解析 `env`、`base` 或 `world`。
- Tiled Scene 异步 linear pose goal 直接使用 robot-base-local，不接受 `pose_reference_frame` 字段。

`free` 只约束位置，`current` 保持起始姿态，`target` 必须提供目标四元数。通过 status 确认机器人注册
的 `tcp_frame_name`，不要从 link 名称推断 TCP。

## 控制模式

Single Scene controller profile 可以按 component 配置 position、velocity 或 effort 控制，以及对应支持的
implicit、explicit 或 direct method；所选 runtime mode 与 controller bundle 必须一致。Tiled Scene runtime
只接受 position control，velocity 或 effort 会在配置解析时被拒绝。

Mimic follower 始终是 controller 所有的 position drive。Planning 通常只针对 arm group；hand motion
使用 direct command-space control。

## 失败与恢复

- 被拒绝的 JSON 请求不会产生 command mutation。
- Single Scene timeline 在执行前全量原子编译。
- Tiled Scene `reject_request` IK policy 会在 selected robot target 或 physics 写入前拒绝。
- Tiled Scene `hold_failed_env` 保留失败 row 的最后成功 target，并返回诊断。
- 轨迹载入失败不会只填充部分 selected env buffer。
- Fail-stop runtime 会拒绝后续 mutation；应重建 runtime，而不是重试控制。

使用与所选生命周期对应的 `status`、`trajectory_status` 或 `planner_status`。Submission response 只证明
admission，不证明执行终态。

## 相关文档

- [Single Scene JSON 参考](../reference/single-scene-json.md)
- [Tiled Scene JSON 参考](../reference/tiled-scene-json.md)
- [运动规划](motion-planning.md)
- [配置](configuration.md)
- [已知约束](../operations/constraints.md)
