# 控制、轨迹与 Action

语言：[中文](control-and-trajectories.md) | [English](../../en/guides/control-and-trajectories.md)

Mirror 面向业务 motion；Kaleidoscope 面向固定 shape RL action。两者不共享 command envelope。只有
Mirror 拥有 control profile；Kaleidoscope action 语义来自 task，默认 controller bundle 由 physics 派生。

## 法兰与任务 TCP

两个产品的 AR5 臂手 profile 默认使用 `AR5V2_L_flange_tcp` / `AR5V2_R_flange_tcp`。
它们相对配置中的机械臂 `flange_frame` 均为零位移、零旋转；即使安装了实体法兰或腕部相机，
也不表示指尖、捏取点或经过标定的接触点。

原来显式选择 `AR5V2_L_pinch_tcp` / `AR5V2_R_pinch_tcp` 的请求需改用对应的 `flange_tcp`
名称，不保留旧名兼容别名。省略 `tcp_frame_name` 时仍使用配置默认值。本次改名不改变数值位姿、
物理绑定、IK 目标或 wrench 参考点。单独机械臂 profile 保留原有 `AR5V2_L_tool_tcp` 名称。

实际工具/接触点应在 `curobo.robot.custom_tcps` 中另行命名，给出相对指定 `parent_frame` 的实测
`xyz`（m）和 `rpy`（rad），再通过 `default_tcp_frame` 或操作支持的 TCP selector 选择。
固定偏移只描述定义时的手型，手指关节变化可能改变接触点。混合力位控制与 tare 必须使用配置的物理 TCP。

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

专用 `physx_cpu_hybrid` Mirror profile 默认以 60 Hz 指令 / 240 Hz 反馈 / 480 Hz 物理运行。一条
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

## L6/O6 手部命令顺序

所有 L6、O6 profile 均采用以下六维顺序，覆盖独立 L6、AR5 组合及相机附件版本，
适用于 PhysX 和 Newton：

| 位置（从 1 开始） | 动作 | L6 关节后缀 | O6 关节后缀 |
| --- | --- | --- | --- |
| 1 | 拇指弯曲 | `thumb_cmc_pitch` | `thumb_cmc_pitch` |
| 2 | 拇指侧摆 | `thumb_cmc_roll` | `thumb_cmc_yaw` |
| 3 | 食指弯曲 | `index_mcp_pitch` | `index_mcp_pitch` |
| 4 | 中指弯曲 | `middle_mcp_pitch` | `middle_mcp_pitch` |
| 5 | 无名指弯曲 | `ring_mcp_pitch` | `ring_mcp_pitch` |
| 6 | 小指弯曲 | `pinky_mcp_pitch` | `pinky_mcp_pitch` |

完整名称由左手 `hand_lh_`、右手 `hand_rh_` 前缀与后缀拼成。保留真实资产名，
L6 的 `roll` 不改名为 O6 的 `yaw`。旧 L6 的 index/middle/ring/pinky/roll/pitch
数组需要重排；按名称发送的指令含义不变。L6 增益、主从联动和驱动选择保持不变。

## Newton AR5/O6 柔顺驱动

Newton 对 AR5/O6（含 Gemini 相机附件组合）自动选择 `newton_compliant` 驱动，
无需 scene 覆盖，该组合不再选择旧直接 PD 路径。仅支持 CPU/CUDA **位置控制**；
其他机器人组合保留原驱动配置，PhysX 不变。

该 bundle 的 `arm_controller.yaml` / `hand_controller.yaml` 设置原生输出，
`newton_drive.yaml` 设置虚拟电机及驱动动力学。参数数组按 robot profile 显式声明的
arm/hand 关节组顺序解释，导入前绑定为精确关节名映射；USD 遍历和 articulation 顺序
变化不会把拇指与四指增益错配。所有控制器 bundle 都采用此名称绑定。
L6 与 O6 采用上述相同语义的六维手部顺序，保留各自真实关节名。
默认完整命令向量按声明的 arm + hand 排列，通过索引映射到原生引擎。

| 关节组 | 原生 Kp / Kd | 原生限矩（N·m） | 被动阻尼（N·m·s/rad） |
| --- | --- | --- | --- |
| AR5 J1–J7 | 16000 / 614.224、613.711、603.791、600.333、604.137、604.749、603.668 | 200 | 0 |
| O6 拇指侧摆 | 600 / 24 | 10 | 0 |
| O6 拇指屈曲 | 2500 / 40 | 60 | 4 |
| O6 四指 MCP | 1250 / 40 | 60 | 4 |
| O6 从动关节 | 无独立电机；原生 equality | 无独立电机限矩 | 0.4 |

手部虚拟电机 Kp=600、虚拟轴最大速度 1 rad/s。**推进限矩**为侧摆 3 N·m、屈曲
5 N·m，与上表原生保持/输出限矩不同。侧摆、拇指屈曲、四指屈曲的粘性分别为
20、20、4.2，输出阻尼分别为 20、40、40。虚拟轴采用后向 Euler 更新，只向指令
方向推进，并限制自由速度及位移；受阻时停滞，不无限积累目标角。它读取主动关节运动，
从动接触反作用通过原生 equality 回传。控制器不直接裁剪或覆写物理 q/qd。

默认 **60 Hz 指令 / 480 Hz 物理**：每指令更新一次虚拟轴，八个物理步线性插值前后
虚拟轴位置，并保持本周期 `输出阻尼 / Kp * 虚拟轴速度` 偏置。Mirror 指令频率来自
scene；Kaleidoscope 来自物理频率除以 `physics_ticks_per_action`。自定义合法整数
比仍按指令更新一次虚拟轴，不会随物理频率提高而把电机推进加快。

机械臂使用相邻指令位置及其速度作三次 Hermite 插值，并保留速度前馈。直接位置目标
端点速度为零；轨迹应提供采样速度。这是通用的逐指令插值，不是实验中的 T 块阶段
PCHIP/重定时策略。每个原生物理步前将估计 PD 输出保护在 199 N·m，低于原生 200 N·m
上限；该估计不是实测驱动力，也不保证积分后约束反力受同一限制。

选定机器人本体及附件启用重力补偿，物体保留重力。臂/手 armature 为 0.01/0.001，
关节库仑摩擦为零，主从 equality 使用 `solref=(0.002,1)`、
`solimp=(0.999,0.999,0.001,0.5,2)`。这些驱动设置只覆盖选定机器人根节点；
几何、质量惯量和关节限角继续由资产定义。独立的 [AR5/O6 接触配置](../reference/configuration.md#ar5o6-接触配置)
设置场景求解参数和按部件区分的碰撞材质。CPU/CUDA 管线、gap 回退值及
求解器默认值见[Newton 共用设置](../reference/configuration.md)。

运行 `just smoke-newton-drive` 验证 CPU/CUDA 的正式 Mirror 运动入口：手部声明顺序、
机械臂速度传递、每条指令八个物理子步、手部目标插值、原生仿真时间及历史状态恢复。

控制历史（包括指令内子步相位）参与 reset、snapshot、restore 及选定环境 clone。
冷位置重置从实际关节位置初始化虚拟轴。CPU/CUDA 使用同一公式和 float64 历史；
CUDA 使用物理 owner 的 stream 和图捕获。Kaleidoscope 以 float64 字段
`controller.virtual_shaft` 暴露此状态，保存快照时不能降精度。外部请求目标与物理
子步生成的原生目标分开保留。
