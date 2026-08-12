# 控制、轨迹与 Action

语言：[中文](control-and-trajectories.md) | [English](../../en/guides/control-and-trajectories.md)

Mirror 面向业务 motion；Kaleidoscope 面向固定 shape RL action。两者不共享 command envelope。只有
Mirror 拥有 control profile；Kaleidoscope action 语义来自 task，默认 controller bundle 由 physics 派生。

## Mirror

Mirror controller 在 owner thread 执行 10 类 motion operation。Timeline 把多个 robot track 编译到
共同整数 tick；每个 tick 所有 target 写入后只推进一次 physics。Joint trajectory 必须带明确 joint
顺序和时间，cancel/estop 在每个 tick 检查。

v1 的 20 项 operation 保持冻结；v2 增加 `control.get_mode`、`control.set_mode` 与
`motion.joint_effort`。完整可解析请求维护在
[Mirror JSON 与运动示例](../reference/mirror-json.md)。

Position/velocity/effort 能力由 controller profile 和 backend capability 共同决定。名称 mapping
允许只写部分 group joints；flat 列表则必须完整覆盖 group，并按 robot profile 的 `joint_groups`
顺序解释，绝不依赖 articulation 内部 DOF 顺序。IK/规划细节见[运动规划](motion-planning.md)。
wire planning segment 可覆盖 `duration_s`、`sample_dt_s`、`avoid_collisions` 和
`force_collision_refresh`；`coordination` 只能在单 segment wrapper 或 timeline 顶层覆盖。`timeout_s`
不是 wire 字段，每次规划始终读取 `planning.request_defaults.timeout_s`。cuRobo
`kinematics.max_batch_size` 只属于 IK；Mirror MotionPlanner 固定单请求，同时由单独的
`profiles.curobo` 持有 warmup、seed、CUDA graph、碰撞能力和 cache 容量。

`control.sync_simulation_to_wall_clock` 只控制执行 pacing，不做轨迹重定时。开启后，idle hold 与 motion
timeline 使用同一个墙钟 deadline 序列；启动或 reset 后的第一 tick 立即执行，后续 tick 只等待剩余的
physics 间隔。tick 落后时会从当前时间重定位，不会突发补跑。关闭后两条路径都不执行墙钟 sleep，但
physics dt 保持不变。`idle_physics_policy: pause` 仍会停止仿真时间，因为此时没有 physics tick 可同步。

## 运行时关节控制模式

关节控制模式不是 action variant。Mirror v2 与原生 `TorchKaleidoscopeEnv` 可在一次完整运动/decision
结束后、下一次开始前，把全部 robot/env 一起切换到 `position`、`velocity` 或 `effort`。切换不会重建
session、physics runtime、robot view、action term、task、IK 或 planner；运动执行中及 SAME_STEP
事务未完成时拒绝切换。

Mirror 的 position/velocity 模式可执行位置型 timeline；effort 模式只允许 hold 与显式、受 profile
限幅的 effort segment。真实切换使 generation 加一；幂等切换不写 engine，也不增加 generation。
切换先中和旧通道，再事务式应用全部 controller profile，最后写新通道 neutral：position 使用当前 q，
velocity/effort 使用零。前向失败且完整补偿后仍可使用旧模式；补偿失败则永久 fail-stop，必须 close
并由调用方重建。

## 混合力/位控制

专用 `physx_cpu_hybrid` Mirror profile 以 240 Hz 运行。一条
`motion.hybrid_force_position` 执行期间，目标 robot 的全部 arm joint 临时使用 direct effort drive；
`force_axes=false` 的笛卡尔方向执行显式位姿阻抗（`Kp/Kd`），`force_axes=true` 的方向执行显式力 PI
（`Kf/Ki`）。hand 继续使用 position/implicit，其它 robot 保持原模式。不能在 arm 上把某个笛卡尔
方向的 implicit position drive 与另一个方向的 explicit effort drive 直接混用，因为 PhysX drive mode
属于 joint，不属于笛卡尔轴。

`force_axes` 属于每条 motion request，相邻运动可选择不同方向。六组增益是运行时 tuning 状态；Mirror
v3 把 `control.get_hybrid_parameters`、`control.set_hybrid_parameters` 与 motion 放在同一 owner queue。
motion 必须携带 `hybrid_parameter_generation`，并在 preflight 时冻结唯一一份不可变增益 snapshot；后续
更新不能改变正在运行的 loop，下一条 motion 必须使用更新结果的新 generation。filter、contact、sensor、
effort/rate/displacement 等安全限幅仍由 YAML 固定，不允许通过 wire 修改。

运动前还必须为相同 robot、物理 TCP 和 world frame 成功执行 `control.tare_wrench`。PhysX raw feedback
采用 environment-on-tool；减 tare、滤波后再换号，对 target、result、CSV 与 telemetry 暴露统一的
tool-on-environment 语义。reset 会失效 tare。正常完成、cancel 或异常都会先把 effort 渐降到零，再恢复
原 position controller，并以最终实测关节位置 handover；恢复失败是 fatal，必须关闭并重建 runtime。

## Kaleidoscope

Action 是 `(num_envs, action_dim)` float32 CUDA tensor，全部 env 同步推进固定
`physics_ticks_per_action`。Canonical task 使用 `joint_control`；其 position 分支保持原 joint-delta
累加语义，velocity 分支输出有界 rad/s，effort 分支按 controller profile limit 的配置比例输出。
已有 `joint_delta` 与 EE/linear variant 支持 position 及 position reference 的有界速度差分，但拒绝
effort。action variant 在构造期冻结，切换 control mode 不改变 action shape 或 tick 数。

Kaleidoscope 没有 Python trajectory list、逐 env playback queue 或异步 planner。固定 tick target buffer
在同一 GPU 上预分配，step 内不做 CPU selector、JSON parsing 或动态 action mode 切换。

Gymnasium、skrl 与 `KaleidoscopeTrainingPort` 不暴露 mode setter，训练 rollout 固定使用初始 position
语义。Schema 2 snapshot 记录 mode 与来源 generation；restore 不自动切换 mode，只能恢复到相同 active
mode，且不会回退 runtime generation。

## Hold 与 reset

Mirror `motion.hold` 保持当前 target，`runtime.reset` 可清 queue 并在 reset 后 hold。Kaleidoscope done
行由 `reset_idx` 或 training SAME_STEP handshake 重置；失败 action 行 hold 并产生 mask/penalty。

继续阅读：[Mirror JSON 与运动示例](../reference/mirror-json.md)及
[运动规划](motion-planning.md)。
