# cuRobo 使用与批量调度

语言：[中文](motion-planning.md) | [English](../../en/guides/motion-planning.md)

本文说明当前 cuRobo v0.8.0 集成、可执行 linear planner、SingleSceneRuntime planning、Tiled Scene 同步 IK
以及 Tiled Scene 异步批调度。

全部 YAML 字段见[配置参考](../reference/configuration.md)，受支持的 facade 与结果类型见
[Python API 参考](../reference/python-api.md)，完整协议字段与状态清单见
[Tiled Scene JSON 参考](../reference/tiled-scene-json.md)。本指南负责说明这些契约如何组合成规划工作流。

## Backend 选择

Single Scene 与 Tiled Scene 共用项目级 planning request、result 和 `runtime.planner.backend` 所有者。Single Scene 还
提供本次启动专用的 CLI override，Tiled Scene 不提供：

| Runtime 路径 | 选择方式 | 支持的工作 |
|---|---|---|
| Single Scene `curobo` | `runtime.planner.backend: curobo`，可用 `--planner-backend curobo` 覆盖 | Joint goal、pose-goal trajectory、TCP 直线路径和可选碰撞感知规划 |
| Single Scene `linear` | `runtime.planner.backend: linear`，可用 `--planner-backend linear` 覆盖 | 仅 `plan_cspace_goal` 与 `plan_cspace_delta` |
| Tiled Scene 异步 `curobo` | `runtime.planner.backend: curobo` | Joint batch/per-env planning 和 per-env task-space path |
| Tiled Scene 异步 `linear` | `runtime.planner.backend: linear` | 可执行的 joint-space 插值 |
| Tiled Scene 同步 EE action | Robot 的 cuRobo IK binding | Batched `ee_*` action，与异步 planner 选择无关 |

Linear backend 不创建 cuRobo solver，也不提供 IK、碰撞检查、关节限位校验或速度/加速度约束优化。
它是明确的 joint interpolation 策略，不是碰撞规划器。

`status.supports_planning` 表示 robot 是否具有有效 cuRobo binding，不表示无模型 linear 策略是否
可用。

## 配置层

算法默认值属于 `configs/curobo/<profile>.yaml`：

- CUDA device 与 tensor dtype。
- IK/planner seed 数量与容差。
- CUDA graph 设置。
- Batch size 与 `multi_env`。
- Self-collision 与场景 collision cache 容量。
- 经过校验的 `task_bundle` 与 planner `warmup` 生命周期策略。

项目拥有的 cuRobo profile 只接受当前严格结构。固定 mapping 会拒绝 unknown field 并报告完整
嵌套路径；boolean 与数值字段使用严格 YAML 类型；非法范围会在创建 CUDA context 前失败。四个
tensor dtype 字段当前只接受项目测试过的 `float32` 组合。

只支持 `task_bundle: curobo_v0_8_default`。Raw optimizer、rollout、transition-model 或
graph-planner 文件路径会被拒绝，因为这些文件属于 cuRobo 版本契约。Context 创建时会验证已安装
版本精确等于 `0.8.0`，不会假设其它 `0.8.x` patch 兼容该 bundle。
`motion_planner.warmup` 默认为 `true`；设为 `false` 会跳过 lazy creation 后的显式 warmup，并把
cold-start 成本移到第一次真实请求。

`configs/curobo/task/**/*.yml` 是随仓库固定的 cuRobo 0.8.0 资源。精确的
`curobo_v0_8_default` bundle 拥有并校验完整文件集合。

Robot 资源属于 `configs/robots/<robot>.yaml`：

```yaml
curobo:
  enabled: true
  planning_joint_group: arm
  robot:
    urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
    robot_config_path: assets/single_system/arm/AR5V2_L/AR5V2_L_curobo.yml
    flange_frame: AR5V2_L_arm_flan_link
    default_tcp_frame: AR5V2_L_pinch_tcp
    load_collision_spheres: true
```

合并顺序是 algorithm profile 在前、robot profile 在后，因此显式 robot 值优先。MJCF 仍是 Isaac
仿真资产；cuRobo 接收 planning URDF 和可选 robot YAML。Hand-only profile 必须设置
`curobo.enabled: false`。

Custom TCP URDF 写入 `runtime.paths.cache_root/curobo`。该 runtime 值为 null 时，依次解析
`LINKERBOT_SIM_CACHE_ROOT`、`XDG_CACHE_HOME/linkerbot_sim`、`~/.cache/linkerbot_sim`，然后追加
`curobo`。相对 runtime 或环境 cache root 会展开并相对进程 working directory 解析。仓库
`.cache` 目录永远不是 fallback。

## SingleSceneRuntime

需要明确选择时，用显式 backend 启动 Single Scene：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --planner-backend curobo --curobo-profile default \
  --tcp-jsonl-port 8765 --gui
```

JSON 边界上的 planned joint goal 与 backend 无关：

```json
{
  "type": "plan_cspace_goal",
  "id": "joint-plan-1",
  "robot_id": 0,
  "group": "arm",
  "joint_positions": {"AR5V2_L_arm_joint_1": 0.2},
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

使用 `--planner-backend linear` 时，只接受两个 `plan_cspace_*` kind，且
`avoid_collisions` 必须为 false。Task-space kind 需要 cuRobo：

```json
{
  "type": "ik_pose",
  "id": "pose-1",
  "robot_id": 0,
  "target_position": [0.35, 0.0, 0.4],
  "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "reference_frame": "world",
  "tcp_frame_name": "AR5V2_L_pinch_tcp",
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false
}
```

`sample_dt_s` 是可选的正数 planner 输出间隔，默认使用 Single Scene physics dt。它不会修改 World
physics step。执行前，编译器会把 trajectory 重采样为整数 physics tick。

Runtime request 值在命令进入队列前解析。显式 JSON 优先于
`runtime.planner.request_defaults` 和 `runtime.execution.command_defaults`。
`duration_s`、`avoid_collisions`、`force_collision_refresh` 与 `coordination` 属于 planner default；
joint `interpolation`、task-space frame 与 linear-path `orientation_mode` 属于 command default。
`coupled` 不受支持。无模型 linear planner 使用解析后的 duration；省略 `sample_dt_s` 时使用实际
Single Scene physics dt。

`ik_pose` 与 `ik_offset` 是 Single Scene 协议 kind 名，不会调用单解
`CuroboInverseKinematics.solve()` facade。Single Scene compiler 把目标转换到 robot-base 坐标，构造
`MotionRequest(goal_pose=...)`，再由 `CuroboMotionPlanner.plan()` 调用 cuRobo
`MotionPlanner.plan_pose()` 生成可执行 trajectory。直接单目标 IK 只通过 Python facade 提供。

## Tiled Scene 异步 Planner

Tiled Scene entrypoint 没有 planner backend CLI override。Runtime profile 拥有 backend、cuRobo profile
与 joint batch mode：

```yaml
runtime:
  profiles:
    curobo: default
  planner:
    backend: curobo
    joint_batch_mode: auto
```

`joint_batch_mode` 接受：

- `auto`：优先 `BatchMotionPlanner`，request 不能使用 batch path 时改用 per-env planning。
- `per_env`：禁用 joint batch planning。
- `batch_only`：batch 不可用时以 `BATCH_UNAVAILABLE` 失败。

提交请求时，全部字段位于顶层：

```json
{
  "type": "plan",
  "request_id": "batch-plan-1",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "kind": "joint_position_target",
  "joint_positions": [0.2, 0.1],
  "duration_s": 1.0,
  "sample_dt_s": 0.02,
  "avoid_collisions": false,
  "load_on_success": true,
  "replace": true
}
```

主线程冻结 selected env 的当前 command row，然后返回 `plan_submitted`。Worker 永远不访问 Isaac
`World`、stage、articulation view 或 PhysX handle。

省略 `duration_s`、`avoid_collisions`、`load_on_success` 与 `replace` 时，使用
`runtime.planner.request_defaults`；省略 `sample_dt_s` 时使用实际 physics dt。显式 JSON 字段始终
优先。轮询并回放：

```jsonl
{"type":"planner_status","wait_timeout_s":0.1}
{"type":"step_trajectory","robot_id":0,"env_ids":[0,1,2,3],"decimation":4}
```

`load_on_success=true` 的成功结果会进入 per-env trajectory buffer。Planner 完成本身不会推进
physics。

异步 `kind="linear_pose_path"` 的 `target_position` 或 `target_offset` 是 selected env 共用的单个
三元素向量；可选目标姿态是一个 wxyz quaternion。这些值直接按 cuRobo robot-base-local frame
解释。Canonical async plan API 不定义也不应用 `pose_reference_frame`；提供该字段会被拒绝，而
不是静默忽略。它也不接受 per-env `(E,3)/(E,4)` target。提交前应转换 world/env target；不同
env 需要不同 target 时应拆分请求。

Tiled Scene async API 还会拒绝 Single Scene-only 的 `coordination` 与 `force_collision_refresh`，并按 plan kind
拒绝其它 unknown field。异步 request 保持原子语义，每个 cuRobo worker request 拥有隔离 context。

## Tiled Scene 同步 TCP 直线运动

`type="step", kind="ee_linear_path"` 是同步 control action，不是异步 `MotionPlanner` request。它
恰好接受一种 target 表示：

- `target_offset`：带名称的相对 target，按 `pose_reference_frame` 解释。
- `target_position`：带名称的绝对 target，按 `pose_reference_frame` 解释。
- `values`：紧凑的 world-frame offset 形式。

```json
{
  "type": "step",
  "kind": "ee_linear_path",
  "robot_id": 0,
  "env_ids": [0, 1, 2, 3],
  "target_offset": [0.0, 0.0, 0.1],
  "orientation_mode": "free",
  "duration_s": 0.4,
  "sample_dt_s": 0.02,
  "interpolation": "linear",
  "tcp_frame_name": "AR5V2_L_pinch_tcp"
}
```

`orientation_mode` 可为 `free`、`current` 或 `target`。`target` 要求
`target_orientation_quat_wxyz`，`free` 执行 position-only IK。省略 mode 但显式提供 target
quaternion 时，parser 推导为 `target`；否则使用解析后的
`runtime.execution.command_defaults.orientation_mode`，内置默认 profile 为 `current`。显式 action
字段始终优先。完整字段与 failure-policy 契约由 [Tiled Scene JSON 参考](../reference/tiled-scene-json.md)拥有。

Action 在每个 sampled waypoint 对 selected env 执行 batched IK。时间维顺序执行，因此上一个
waypoint 解会 warm-start 下一个。稀疏 IK 网格使用
`ceil(duration_s / sample_dt_s)` 个 waypoint，再重采样为
`ceil(duration_s / physics_dt)` 个执行 tick。全部 IK 在 physics 推进前完成。
`failure_policy: hold_failed_env` 时，一个 env 在首次数值 IK 失败后冻结，但所有 env 仍执行相同
physics tick 数；`failure_policy: reject_request` 时，任一数值失败都会在 target cache 或 physics
写入前原子拒绝整条 action。

## 跨请求批调度

`TiledPlannerManager` 只在连续 FIFO joint-space request 的 batch key 相同时合并。Key 包含 robot
identity、command joint names、duration、sample dt、collision requirement 与 segment structure。
Manager 不会跨过不兼容项重排请求。

Manager 计数的是 problem row，不是 request object。内置 runtime profile 设置
`runtime.planner.resources.max_batch_problems: 64`。`TiledCuroboPlanningBackend.plan_many()` 把有界
group 堆叠为 `CuroboBatchJointProblem`；cuRobo batch core 不包含 `env_id`、`request_id`、source 或
playback 字段。规划结束后，result 按原 request ID 与 env row 拆分。

少于固定 cuRobo batch size 的 row 会通过复制最后一个真实 row 来 padding。内置
`runtime.planner.oversize_request_policy: split` 会把超过 `max_batch_problems` 的单个公开 request
按有界 env-row chunk 派发，再原子合并回原 request result。设置为 `reject` 会在 dispatch 前拒绝。
Backend 调用不得超过 `max_batch_problems`，解析后的上限也必须适配所选 cuRobo batch capacity。

每个 active cuRobo planner future 拥有自己的 context、CUDA graph/cache 和 tensor。增加
`--planner-workers` 可能提高吞吐，也会成倍增加显存与 warmup 成本。增加 worker 前先测量一个或
两个 worker 的表现。

## Collision 能力

`avoid_collisions=true` 是严格要求。除非以下能力全部可用，否则 request 会失败，不会静默降级：

- Robot collision sphere。
- 所选 solver/planner 的 scene collision checker。
- 足够的 `cuboid`/`mesh` cache 容量。
- 与当前 scene version 同步的物化 collision view。

`multi_env=false` 表示一个 Tiled batch 共用一个 collision world。当 obstacle pose 不同时，不要假设
各 env 障碍物彼此独立，除非 backend 已物化等价的 per-env world。

## 当前边界

- Tiled Scene joint-space planning 可使用 `BatchMotionPlanner`；异步 `linear_pose_path` 仍对每个 env
  分别执行顺序 warm-start IK。
- 同步 `ee_linear_path` 在每个 waypoint 对 env 维度做 batch，但不是 collision-aware graph search
  或 trajectory optimization。
- 超大公开 request 按 `runtime.planner.oversize_request_policy` 处理；内置默认在
  `max_batch_problems` 处拆分，`reject` 则在 dispatch 前拒绝。
- 公开 task-space quaternion 始终使用 wxyz。
- 经过校验的 Isaac/cuRobo stack 需要 Warp API adapter。

## 代码索引

- 共享 request/result：`src/linkerbot_sim/planning/`
- Linear backend：`src/linkerbot_sim/planning/linear_backend.py`
- cuRobo config/context：`src/linkerbot_sim/backends/curobo/config.py`、`context.py`
- cuRobo IK/planning：`inverse_kinematics.py`、`motion_planner.py`、`linear_pose_path.py`
- cuRobo batch core：`src/linkerbot_sim/backends/curobo/batch/`
- Tiled Scene integration：`src/linkerbot_sim/tiled/planning/backends/curobo.py`
- Async manager：`src/linkerbot_sim/tiled/planning/manager.py`
- Single Scene compiler：`src/linkerbot_sim/app/motion/timeline/compiler.py`
