# 运动规划与批量运动学

语言：[中文](motion-planning.md) | [English](../../en/guides/motion-planning.md)

## 能力归属

| 能力 | Mirror | Kaleidoscope |
| --- | --- | --- |
| FK/IK | 支持 | 支持 device-native batch |
| 同步笛卡尔直线动作 | 支持 | 支持固定 waypoint batch IK |
| 关节/笛卡尔 trajectory planning | 支持 | 不创建 |
| planning collision world / avoidance | 支持 | 不创建 |
| 图搜索、trajopt、异步 planner worker | 可按 profile 创建 | 禁止 |

Kaleidoscope 保留 batch IK 和同步直线 motion，是为了把 action 转成固定 tick joint targets；它们不是
trajectory planner。一次 linear action 在 GPU 上生成 waypoint，并调用一次 batch waypoint IK；失败行
从首次失败点 hold，附加 penalty/truncate。不得为它加载 MotionPlanner、collision checker 或 per-env
trajectory buffer。

## Mirror planning

Mirror motion owner 处理 timeline、joint goal/delta/trajectory、c-space goal/delta、IK、linear pose
path 与 hold。cuRobo context、planner、collision world 和 worker 全由 Mirror runtime 拥有，并在 session
前关闭。Request 使用 robot identity 与明确 frame；四元数为 `wxyz`，长度 m、角度 rad。

Planning collision snapshot 在提交 request 前从当前 scene 捕获。其它机器人可以按协调策略成为静态
障碍，物体 collision geometry 由 robot/object profile 与 stage provider 生成。配置所有权明确分成两层：
`configs/planning/mirror.yaml` 只拥有 duration、采样周期、timeout、避障、刷新与 coordination 等
`request_defaults`；`configs/curobo/mirror.yaml` 拥有 IK batch 容量，以及 MotionPlanner 的 seed、
CUDA graph、碰撞能力和 cache。`kinematics.max_batch_size` 只属于 IK；Mirror MotionPlanner context 固定
`max_batch_size=1`，一次只规划一个请求。wire planning segment 可覆盖 duration、采样周期、避障和刷新，
coordination 只能在 wrapper/timeline 顶层覆盖；timeout 不能由 wire 覆盖，始终来自
`planning.request_defaults.timeout_s`。planning 配置不选择 backend，也不配置数值容量。

`coordination: independent` 是默认值，规划时不把其它机器人加入障碍；`static_others` 使用同一份
scene snapshot，把其它机器人作为静态碰撞障碍。当前没有 coupled multi-robot optimizer，
`coupled` 会在规划前被明确拒绝。

Mirror 可以启用 IK CUDA graph，但 MotionPlanner CUDA graph 必须关闭。项目固定的 cuRobo 0.8 runtime
默认未启用实验性 solver graph reset；Mirror 又会让同一个 planner 处理 pose 与 cspace 请求，并且不依赖
该全局开关。`MirrorConfig` 会在启动前拒绝无效配置，避免把失败推迟到第一次真实规划。

## 显存边界

Planner 的 graph/trajopt seed、collision cache、候选 trajectory 和 debug buffer 会按 env/seed/horizon
放大显存，因此不进入大规模 RL runtime。Kaleidoscope task 不选择 backend；EE/直线 mode 通过可选
`profiles.curobo` 加载 kinematics-only 数值 profile，`joint_control`/`joint_delta` mode 则必须省略它并
完全不加载 cuRobo。

## 失败策略

- Mirror：请求失败返回结构化 error，不提交半条 trajectory；cancel/estop 在 tick 检查点停止；
- Kaleidoscope：逐 env failure mask，失败行 hold，reward penalty，并按 task policy truncate；
- 两者都禁止把 NaN、shape mismatch 或未知 frame 猜成默认值。

完整请求与 wire operation 见 [Mirror JSON 与运动示例](../reference/mirror-json.md)，动作节奏见
[控制与轨迹](control-and-trajectories.md)。
