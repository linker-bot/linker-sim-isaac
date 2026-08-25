# Mirror v1/v2/v3 JSON 参考

语言：[中文](mirror-json.md) | [English](../../en/reference/mirror-json.md)

Mirror 的三个 ingress 使用同一个 strict envelope。v1 的 20 项 operation 保持冻结，v2 是增加
3 项的兼容 superset，v3 再增加 4 项混合力/位控制 operation。每个 stdin/TCP JSONL 请求占一行；
WebSocket 每个 text frame 是一个请求。

## Request 与 response

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "request-42",
  "operation": "runtime.status",
  "arguments": {}
}
```

`protocol` 只接受 `linkerbot.mirror.v1`、`linkerbot.mirror.v2` 或 `linkerbot.mirror.v3`。四个字段全部必需，且不接受
未知/重复字段、空 request ID、NaN 或 Infinity。成功 response 回显已接受请求的协议版本：

```json
{"protocol":"linkerbot.mirror.v1","request_id":"request-42","ok":true,"result":{}}
```

失败：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "request-42",
  "ok": false,
  "error": {"code": "invalid_arguments", "message": "..."}
}
```

同一 request ID 在 pending、active 或 terminal history 中重复都会被拒绝。所有普通业务请求进入
有界 admission；cancel、cancel-current、estop 和 quit 只改变线程安全控制状态，可以由 ingress
立即处理，但不从后台线程访问 Isaac。

## 固定 operation 集

| 类别 | Operation |
| --- | --- |
| Timeline/关节 | `motion.plan_timeline`, `motion.joint_goal`, `motion.joint_delta`, `motion.joint_trajectory`, `motion.hold`, `motion.joint_effort`（仅 v2） |
| cuRobo planning | `motion.plan_cspace_goal`, `motion.plan_cspace_delta`, `motion.plan_linear_pose_path` |
| IK | `motion.ik_pose`, `motion.ik_offset` |
| Runtime | `runtime.reset`, `runtime.status` |
| State | `state.get`, `state.set` |
| Snapshot | `snapshot.get`, `snapshot.set` |
| Admission/safety | `queue.cancel`, `queue.cancel_current`, `runtime.estop`, `runtime.quit` |
| Control（仅 v2） | `control.get_mode`, `control.set_mode` |
| Hybrid（仅 v3） | `control.get_hybrid_parameters`, `control.set_hybrid_parameters`, `control.tare_wrench`, `motion.hybrid_force_position` |

其中原有 20 个名称是 wire v1 的协议常量，不由 YAML 修改；v2 接受全部 v1 operation，并增加
control query/switch 与显式 effort；v3 接受全部 v2 operation，并增加上述 4 项。无法可靠解析
protocol 的 malformed payload 可使用 v1 生成
transport-level failure；已接受请求绝不会被改写协议版本。Motion 参数和坐标 frame 见
[控制与轨迹](../guides/control-and-trajectories.md)及[运动规划](../guides/motion-planning.md)。

wire planning segment 可覆盖 `duration_s`、`sample_dt_s`、`avoid_collisions` 和
`force_collision_refresh`；省略时读取 `planning.request_defaults`。`coordination` 只能在单 segment
wrapper 或 timeline 顶层覆盖，不是 segment 字段。`timeout_s` 不属于 wire schema，每次规划始终使用
`planning.request_defaults.timeout_s`。该 planning profile 不选择或扩容 cuRobo；IK batch 容量以及
单请求 MotionPlanner 的 seed、CUDA graph、碰撞能力和 cache 容量来自 Mirror mode 单独选择的
`profiles.curobo`。

## 常用请求

Reset（默认清空 pending queue、保持目标并解除 estop）：

```json
{
  "protocol":"linkerbot.mirror.v1",
  "request_id":"reset-1",
  "operation":"runtime.reset",
  "arguments":{"clear_queue":true,"hold_after_reset":true}
}
```

控制模式查询和切换都是普通 owner queue 请求：

```json
{
  "protocol": "linkerbot.mirror.v2",
  "request_id": "mode-get-1",
  "operation": "control.get_mode",
  "arguments": {}
}
```

query result 包含 `initial_mode`、`active_mode`、`generation`、`supported_modes` 和
`scope: "all"`。带 optimistic generation 的全机器人切换：

```json
{
  "protocol": "linkerbot.mirror.v2",
  "request_id": "mode-set-1",
  "operation": "control.set_mode",
  "arguments": {"mode": "velocity", "expected_generation": 0}
}
```

`mode` 只接受 `position`、`velocity`、`effort`。真实切换成功后 generation 加一；切到当前模式
是无 engine write 的幂等操作，但 generation 不匹配时仍拒绝。切换只位于两次完整运动之间，estop
latch 期间禁止，且不会重建 runtime、physics world、planner 或 collision context。

真实切换先中和旧通道、应用全部 robot profile，再中和新通道：position 写预检时的当前 q，
velocity/effort 写零。前向失败会按逆 robot 顺序 rollback；rollback 也失败时 runtime 永久
fail-stop，只能关闭并由调用方重建。

混合控制参数也是普通 owner queue 状态。先读取 YAML 初值、上限和 generation：

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "hybrid-parameters-get-1",
  "operation": "control.get_hybrid_parameters",
  "arguments": {}
}
```

在两次完整运动之间可原子更新任意非空参数子集：

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "hybrid-parameters-set-1",
  "operation": "control.set_hybrid_parameters",
  "arguments": {
    "expected_generation": 0,
    "motion_stiffness": [180.0, 180.0, 220.0, 8.0, 8.0, 8.0],
    "motion_damping": [28.0, 28.0, 32.0, 1.8, 1.8, 1.8],
    "force_proportional": [0.2, 0.2, 0.3, 0.1, 0.1, 0.1],
    "force_integral": [0.4, 0.4, 0.6, 0.08, 0.08, 0.08]
  }
}
```

四个数组分别对应笛卡尔位控 `Kp`、`Kd` 与力控 `Kf`、`Ki`；
`posture_stiffness`、`posture_damping` 是零空间标量增益。值必须有限、非负且不超过 YAML
`tuning_limits`。真实变化增加独立的 hybrid parameter generation，幂等更新不增加。由于更新和
motion 共用 admission queue，它不可能插入正在运行的控制循环。

启动 hybrid motion 前先对物理 TCP 做 tare：

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "hybrid-tare-left-1",
  "operation": "control.tare_wrench",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "reference_frame": "world"
  }
}
```

成功结果返回 `tare_generation`。reset 会失效全部 tare；单纯修改增益不会改变 sensor/frame 绑定，
因此不会失效 tare。

读取当前逻辑状态：

```json
{"protocol":"linkerbot.mirror.v1","request_id":"state-1","operation":"state.get","arguments":{}}
```

成功响应的 `result` 直接是 Mirror state service 返回的 owned JSON object，外层不会再套一层
`state`。修改客户端解码后的结果不会通过引用反向修改 runtime。

写入状态：

```json
{
  "protocol":"linkerbot.mirror.v1",
  "request_id":"state-2",
  "operation":"state.set",
  "arguments":{
    "state":{
      "schema":"linkerbot.scene-snapshot.v1",
      "metadata":{
        "source_runtime":"mirror",
        "coordinate_frame":"scene-local",
        "info":{}
      },
      "robots":[],
      "objects":{}
    },
    "strict":true
  }
}
```

上例只展示 state object 的外层形状；实际请求应使用 `state.get` 返回的完整 object。`state` 必需且
必须是 JSON object；`strict` 可选、默认为 `true`，并且只能是 JSON boolean。缺少字段、未知字段、
数组形式的 state，以及用 `0`/`1`/字符串冒充 boolean，都会在调用 adapter 前返回
`invalid_arguments`。

生产 state adapter 的 `state.set` 成功结果包含：

| Result 字段 | 类型 | 语义 |
| --- | --- | --- |
| `event` | string | restore 事件名。 |
| `accepted` | boolean | 事务恢复是否被接纳。 |
| `robots` | string array | 已恢复的机器人 label。 |
| `objects` | string array | 已恢复的对象名。 |
| `env_ids` | integer array | Mirror 是单场景，该数组为空。 |
| `partial` | boolean | 非 strict 恢复是否跳过了未匹配状态。 |
| `message` | optional string | adapter 提供的附加信息。 |

两个 state operation 都是普通有界队列请求，不是 out-of-band engine call。Ingress 线程只冻结并
入队 payload，只有 runtime owner thread 会调用 state service。Pending state 请求可以被
`queue.cancel` 清除；`state.set` 一旦进入事务 adapter，就不能被 cancel/estop 从中间打断，只能完整
提交或回滚。

Snapshot capture：

```json
{"protocol":"linkerbot.mirror.v1","request_id":"snap-1","operation":"snapshot.get","arguments":{}}
```

Snapshot restore 接受 `snapshot`，可选 `label_map` 与 boolean `strict`。snapshot 是
`linkerbot.scene-snapshot.v1` 的 owned JSON mapping；恢复会在第一次 mutation 前完成兼容性检查。

取消指定请求：

```json
{
  "protocol":"linkerbot.mirror.v1",
  "request_id":"cancel-1",
  "operation":"queue.cancel",
  "arguments":{"request_id":"motion-9"}
}
```

`runtime.estop` 会清除 pending 请求，并让 active motion 在下一个取消检查点停止。estop 时物理
冻结；`state.get` 仍可用于检查状态，新的 `state.set` 返回 `runtime_estopped`。如果 state transaction
在 latch 设置前已经进入 adapter，它会原子完成或回滚，但绝不会解除 latch。

现有 snapshot 语义不随本次 state RPC 改变：`snapshot.set` 在 estop 时仍可用于物理冻结状态下的
versioned 冷恢复，但同样不会解除 latch。只有成功的 `runtime.reset` 才会解除 estop，并重新允许
motion 和新的 `state.set`。

## 运动指令公共规则

下面的请求全部面向内置 `mirror/scene3`。stdin/TCP JSONL 必须把一个请求编码在一行中；文档为了
可读性才展开为多行。`robot_id` 是 session-local 整数，scene3 按场景声明顺序使用：

所有示例都通过当前 strict envelope 与 motion parser；但 IK/规划能否执行成功仍取决于请求时的
机器人状态、碰撞场景和目标可达性。task-space 坐标是写法示例，应用应根据标定和当前状态生成目标。

| `robot_id` | `robot_label` | 机器人 profile |
| --- | --- | --- |
| `0` | `left_arm` | `ar5v2_l6v1_l` |
| `1` | `right_arm` | `ar5v2_l6v1_r` |

`robot_label` 是可选的一致性断言，不是 `robot_id` 的替代 selector。`group` 省略时为 `arm`；内置
arm-hand profile 还接受 `hand`。

### 关节 mapping 与列表

`joint_positions`/`joint_deltas`/`joint_efforts` 接受两种形式：

- 名称 mapping：允许只写 group 的一部分关节；未写关节保持当前 command；
- flat 列表：必须完整覆盖 group，并严格采用 robot profile 的 `joint_groups` 顺序。

scene3 机械臂列表顺序如下：

```text
left_arm/arm:
  AR5V2_L_arm_joint_1, AR5V2_L_arm_joint_2, AR5V2_L_arm_joint_3,
  AR5V2_L_arm_joint_4, AR5V2_L_arm_joint_5, AR5V2_L_arm_joint_6,
  AR5V2_L_arm_joint_7

right_arm/arm:
  AR5V2_R_arm_joint_1, AR5V2_R_arm_joint_2, AR5V2_R_arm_joint_3,
  AR5V2_R_arm_joint_4, AR5V2_R_arm_joint_5, AR5V2_R_arm_joint_6,
  AR5V2_R_arm_joint_7
```

左右手 `hand` 列表顺序均为 index、middle、ring、pinky、thumb roll、thumb pitch；完整名称分别带
`L6V1_L_` 或 `L6V1_R_` 前缀。列表顺序来自配置，不使用 USD articulation 的内部 DOF 顺序。

`motion.joint_trajectory` 的名称形式把每个 joint 映射到等长 sample 列表；矩阵形式则是
`[sample][group_joint]`，每一行都必须完整覆盖 group。`times_s` 数量必须等于 sample 数，所有值必须
大于零且严格递增。

### 单位、frame 与公共字段

- revolute joint position/delta 使用 rad，位置和 offset 使用 m，时间使用 s；
- 四元数顺序固定为 `wxyz`；
- `reference_frame`/`offset_frame` 只接受 `world`、`env`、`robot_base`、`tcp`；
- 直接 `hold`/joint/trajectory 必须显式给出 `duration_s`；planning operation 可省略并读取
  `planning.request_defaults.duration_s`；
- planning segment 可覆盖 `duration_s`、`sample_dt_s`、`avoid_collisions`、
  `force_collision_refresh`；`timeout_s` 禁止出现在 wire request；
- `coordination` 只放在单段 wrapper 或 timeline 顶层；`independent` 不把其它机器人当规划障碍，
  `static_others` 把其它机器人当静态障碍，`coupled` 当前明确拒绝；
- `interpolation` 只适用于 `joint_goal`/`joint_delta`，可选 `linear` 或 `smoothstep`。

| Operation | 必填 payload | 主要可选字段 |
| --- | --- | --- |
| `motion.hold` | `robot_id`, `duration_s` | `robot_label`, `group` |
| `motion.joint_goal` | `robot_id`, `duration_s`, `joint_positions` | `interpolation` |
| `motion.joint_delta` | `robot_id`, `duration_s`, `joint_deltas` | `interpolation` |
| `motion.joint_trajectory` | `robot_id`, `duration_s`, `joint_positions`, `times_s` | 无 kind-specific 字段 |
| `motion.joint_effort` | `robot_id`, `duration_s`, `joint_efforts` | `robot_label`, `group`, `phase` |
| `motion.plan_cspace_goal` | `robot_id`, `joint_positions` | planning override |
| `motion.plan_cspace_delta` | `robot_id`, `joint_deltas` | planning override |
| `motion.ik_pose` | `robot_id`, `target_position` | orientation、TCP、frame、planning override |
| `motion.ik_offset` | `robot_id`, `offset` | orientation、TCP、frame、planning override |
| `motion.plan_linear_pose_path` | `robot_id`，以及 target/offset 二选一 | orientation mode、TCP、frame、planning override |
| `motion.plan_timeline` | `tracks` | `coordination`, `force_collision_refresh` |

## 单段运动指令示例

### 1. 保持左臂：`motion.hold`

保持当前 arm command 一秒；它仍按 physics tick 推进，不等于暂停整个仿真：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-hold-left-arm-1",
  "operation": "motion.hold",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 1.0
  }
}
```

### 2. 左臂列表回零：`motion.joint_goal`

列表必须正好包含七个 arm 值。本例用 `smoothstep` 在两秒内到达配置零位，不重置对象或手部：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-joint-goal-left-home-1",
  "operation": "motion.joint_goal",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 2.0,
    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "interpolation": "smoothstep"
  }
}
```

### 3. 左臂局部增量：`motion.joint_delta`

名称 mapping 只改变 joint 1 和 joint 4，其余 arm command 保持不变：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-joint-delta-left-1",
  "operation": "motion.joint_delta",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 1.0,
    "joint_deltas": {
      "AR5V2_L_arm_joint_1": 0.1,
      "AR5V2_L_arm_joint_4": -0.1
    },
    "interpolation": "smoothstep"
  }
}
```

### 4. 左臂采样轨迹：`motion.joint_trajectory`

每个名称对应三个 sample；未写关节在整个轨迹中保持当前 command。`times_s` 与 sample 数一致：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-joint-trajectory-left-1",
  "operation": "motion.joint_trajectory",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 1.5,
    "joint_positions": {
      "AR5V2_L_arm_joint_1": [0.1, 0.2, 0.0],
      "AR5V2_L_arm_joint_2": [-0.1, -0.2, 0.0]
    },
    "times_s": [0.5, 1.0, 1.5]
  }
}
```

### 5. 左臂显式 effort：`motion.joint_effort`

该 v2-only 请求保持指定 effort，按当前 controller profile 限幅，不访问 planner/collision，并在
结束时写零 effort：

```json
{
  "protocol": "linkerbot.mirror.v2",
  "request_id": "motion-joint-effort-left-1",
  "operation": "motion.joint_effort",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "duration_s": 0.2,
    "joint_efforts": {
      "AR5V2_L_arm_joint_1": 2.5,
      "AR5V2_L_arm_joint_2": -1.0
    },
    "phase": "contact_push"
  }
}
```

### 6. 混合力/位控制：`motion.hybrid_force_position`

该 v3-only 请求在整个执行期间冻结 generation 1 的增益。Z 平移使用力控，其余五个笛卡尔轴使用
显式运动阻抗；下一条请求可以独立选择另一组 `force_axes`：

```json
{
  "protocol": "linkerbot.mirror.v3",
  "request_id": "motion-hybrid-force-position-left-1",
  "operation": "motion.hybrid_force_position",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "duration_s": 0.5,
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "reference_frame": "world",
    "target_position": [0.35, 0.0, 0.25],
    "target_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
    "force_axes": [false, false, true, false, false, false],
    "target_wrench": [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
    "tare_generation": 1,
    "hybrid_parameter_generation": 1,
    "phase": "normal_force_hold"
  }
}
```

`force_axes` 是单条请求状态，不进入持久 tuning 参数。motion 在 preflight 时一次性冻结六组增益；
排在它后面的参数更新不会影响当前 loop，下一段 motion 必须携带新 generation。PhysX raw wrench 的
符号是 environment-on-tool，`target_wrench` 和返回 feedback 则是 tool-on-environment。

### 7. C-space 规划回零：`motion.plan_cspace_goal`

该 operation 调用 MotionPlanner，而不是直接 joint interpolation。`static_others` 会把右臂作为静态
规划障碍；`force_collision_refresh` 要求本次规划前刷新 collision view：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-plan-cspace-goal-left-1",
  "operation": "motion.plan_cspace_goal",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "coordination": "static_others",
    "duration_s": 2.0,
    "sample_dt_s": 0.02,
    "joint_positions": {
      "AR5V2_L_arm_joint_1": 0.0,
      "AR5V2_L_arm_joint_2": 0.0,
      "AR5V2_L_arm_joint_3": 0.0,
      "AR5V2_L_arm_joint_4": 0.0,
      "AR5V2_L_arm_joint_5": 0.0,
      "AR5V2_L_arm_joint_6": 0.0,
      "AR5V2_L_arm_joint_7": 0.0
    },
    "avoid_collisions": true,
    "force_collision_refresh": true
  }
}
```

### 6. C-space 增量规划：`motion.plan_cspace_delta`

省略 `duration_s` 与 `sample_dt_s` 时读取 planning profile 默认值：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-plan-cspace-delta-left-1",
  "operation": "motion.plan_cspace_delta",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "joint_deltas": {
      "AR5V2_L_arm_joint_2": 0.05,
      "AR5V2_L_arm_joint_4": -0.05
    },
    "avoid_collisions": false
  }
}
```

### 7. TCP 绝对位姿 IK：`motion.ik_pose`

位置按 `reference_frame` 解释，姿态是 `wxyz`。`ik_pose` 不接受 `orientation_mode`；提供四元数即
表示约束目标姿态：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-ik-pose-left-1",
  "operation": "motion.ik_pose",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "target_position": [0.35, 0.0, 0.25],
    "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "reference_frame": "robot_base",
    "duration_s": 2.0,
    "sample_dt_s": 0.02,
    "avoid_collisions": false
  }
}
```

### 8. TCP 相对偏移 IK：`motion.ik_offset`

该请求沿当前 TCP 自身 Z 轴移动 3 cm，只约束目标位置并保留 planner 的其余默认值：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-ik-offset-left-1",
  "operation": "motion.ik_offset",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "offset": [0.0, 0.0, 0.03],
    "offset_frame": "tcp",
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "duration_s": 1.0,
    "avoid_collisions": false
  }
}
```

### 9. TCP 直线路径：`motion.plan_linear_pose_path`

相对形式必须写 `offset`，不能同时写 `target_position`。本例沿 TCP Z 轴直线移动 5 cm，并保持当前
TCP 姿态：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-linear-offset-left-1",
  "operation": "motion.plan_linear_pose_path",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "offset": [0.0, 0.0, 0.05],
    "offset_frame": "tcp",
    "orientation_mode": "current",
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "duration_s": 1.5,
    "sample_dt_s": 0.02,
    "avoid_collisions": true,
    "force_collision_refresh": true
  }
}
```

绝对形式改用 `target_position`/`reference_frame`。`orientation_mode: target` 必须同时给出目标四元数：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-linear-target-left-1",
  "operation": "motion.plan_linear_pose_path",
  "arguments": {
    "robot_id": 0,
    "robot_label": "left_arm",
    "group": "arm",
    "target_position": [0.35, 0.0, 0.30],
    "reference_frame": "robot_base",
    "orientation_mode": "target",
    "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "tcp_frame_name": "AR5V2_L_pinch_tcp",
    "duration_s": 2.0,
    "sample_dt_s": 0.02,
    "avoid_collisions": false
  }
}
```

`orientation_mode: free` 只约束位置；`current` 保持请求开始时的 TCP 姿态；`target` 使用给定目标姿态。

## 多机器人 Timeline 示例

### 10. 双臂同步回零：`motion.plan_timeline`

不同 robot track 都从全局 tick 0 开始。同一个 `unit` 内的 arm/hand `group_tracks` 同时开始；每个
group track 内的 `segments` 按数组顺序串行。下面让左右臂与左手同步回到各自的配置零位：

```json
{
  "protocol": "linkerbot.mirror.v1",
  "request_id": "motion-timeline-dual-home-1",
  "operation": "motion.plan_timeline",
  "arguments": {
    "coordination": "independent",
    "force_collision_refresh": false,
    "tracks": [
      {
        "robot_id": 0,
        "robot_label": "left_arm",
        "units": [
          {
            "group_tracks": [
              {
                "group": "arm",
                "segments": [
                  {
                    "kind": "joint_goal",
                    "duration_s": 2.0,
                    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "interpolation": "smoothstep"
                  }
                ]
              },
              {
                "group": "hand",
                "segments": [
                  {
                    "kind": "joint_goal",
                    "duration_s": 2.0,
                    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "interpolation": "smoothstep"
                  }
                ]
              }
            ]
          }
        ]
      },
      {
        "robot_id": 1,
        "robot_label": "right_arm",
        "group": "arm",
        "segments": [
          {
            "kind": "joint_goal",
            "duration_s": 2.0,
            "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "interpolation": "smoothstep"
          }
        ]
      }
    ]
  }
}
```

Timeline 会先完成全部结构校验和所需规划，任一 segment 编译失败时整条 timeline 都不执行。成功
response 的 `result` 包含 `event: motion_completed`、原 operation 和 runtime 累计 physics `steps`；
`steps` 不是本次请求的局部 tick 数。

## Transport 安全

Server 只绑定 loopback，无认证、授权或 TLS。不要将端口直接暴露到非受信网络。每个连接和消息
大小均有上限。response timeout 会在 admission 锁内移除仍在 pending 的请求，保证它不会稍后被
owner 取出执行；已经 active 的请求只会收到协作取消，且可能已越过状态修改边界，因此重试前仍需
查询状态。timeout 不转移或提前销毁仍由 owner 持有的 runtime 资源。
