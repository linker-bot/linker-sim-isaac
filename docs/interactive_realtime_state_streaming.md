# Interactive Realtime State Streaming

本文记录双臂交互实时模式中，如何读取并通过 Foxglove live server 或 MCAP 对外提供机器人关节状态和环境对象位姿。

## 结论

实时模式可以另外开线程，但不建议让新线程直接读取 Isaac/PhysX runtime 状态。

推荐做法是：

1. 仿真主线程在每次 `world.step()` 后采样 Isaac 状态。
2. 主线程把采样结果转换成普通 Python/numpy 快照。
3. 后台线程只消费快照，负责通过 Foxglove live server 实时发布，或写入 Foxglove MCAP。

这样可以避免跨线程直接访问 `SingleArticulation`、USD stage、PhysX view 等对象带来的线程安全风险。

第一阶段只支持 Foxglove 输出。项目自己的 JSONL/TCP/WebSocket 状态查询、CSV 分流和自定义 Web UI 状态流先不纳入本设计的落地范围。

## 当前实时模式结构

实时交互入口是 `scripts/dual_arm_interactive.py`。

核心循环在 `src/linkerbot_sim/app/interactive/dual_arm.py`：

- transport 线程读取 stdin/TCP/WebSocket JSON 命令。
- 命令进入 `InteractiveMotionQueue`。
- 主循环从队列取命令，并串行执行 motion。
- 空闲时通过 hold step 持续刷新当前姿态。

transport 线程已经存在于 `src/linkerbot_sim/app/interactive/transports.py`，但它们只处理命令收发，不推进仿真，也不读取 Isaac 状态。

双臂真正推进物理仿真的地方在 `src/linkerbot_sim/execution/dual_steps.py` 的 `_apply_dual_targets_once()`：

```text
left controller apply action
right controller apply action
world.step()
write optional logs
return step + 1
```

因此，状态采样最适合挂在 `world.step()` 后面。

## 为什么不直接在后台线程读状态

目标状态包括：

- 关节角度
- 关节速度
- 关节加速度
- 关节 effort，包括控制器命令 effort、PhysX measured effort 和 Isaac applied effort
- 环境对象位置和姿态

这些数据分别来自：

- `articulation.get_joint_positions()`
- `articulation.get_joint_velocities()`
- `controller.last_commanded_efforts`
- `get_measured_joint_efforts()`
- `get_applied_joint_efforts()`
- USD prim 或 PhysX rigid body 的 world transform

这些 API 背后可能访问 Isaac Sim、PhysX、USD stage 或 GPU/torch/warp buffer。它们通常应和仿真 step 保持同一线程访问。后台线程直接轮询可能造成：

- 读到半步状态或与控制目标不同步的状态。
- 和 `world.step()`、render 或 PhysX buffer 更新竞争。
- 在 GUI/Kit runtime 中触发难复现崩溃。
- effort 和 object pose 的读取开销干扰实时控制。

所以后台线程应该读项目自己的快照，而不是读 Isaac runtime 对象。

## 推荐架构

新增一个状态采样和发布层，例如：

```text
src/linkerbot_sim/telemetry/state_snapshot.py
src/linkerbot_sim/telemetry/foxglove_state.py
src/linkerbot_sim/app/interactive/state_stream.py
```

建议职责拆分如下：

### StateSampler

运行在仿真主线程。

职责：

- 在 `world.step()` 后采样左右 articulation。
- 读取对象 prim 的 world pose。
- 用上一帧速度计算关节加速度。
- 生成不可变状态快照。

### StateStream

线程安全快照通道。

职责：

- 保存最新快照。
- 可选保存固定长度 ring buffer。
- 用 `Condition` 或 `Queue` 唤醒后台消费者。
- 不持有 Isaac runtime 对象。

### FoxgloveStateSink

Foxglove 输出适配器。

职责：

- 包装现有 `FoxgloveLogger.open_live_server(...)` 和 `FoxgloveLogger.open_mcap(...)`。
- 把 `StateSnapshot` 映射到 Foxglove topics。
- live server 和 MCAP 使用同一套 `publish(snapshot)` 逻辑，区别只在 sink 初始化方式。
- 关闭时统一释放 Foxglove server 或 MCAP writer。

### StatePublisher

运行在后台线程。

职责：

- 读取 `StateStream` 中的快照。
- 调用 `FoxgloveStateSink.publish(snapshot)`。
- 控制发布频率、队列长度和丢帧策略。
- 不直接访问 Isaac runtime 对象。

第一阶段只内置 Foxglove sink。以后如果要加 CSV、JSONL 或项目 WebSocket 状态流，可以复用同一个 `StateSnapshot`，但不要改变主线程采样边界。

## 快照内容

建议快照结构包含：

```json
{
  "step": 120,
  "time_s": 0.5,
  "robots": {
    "left": {
      "joint_names": ["..."],
      "positions_rad": [0.0],
      "velocities_rad_s": [0.0],
      "accelerations_rad_s2": [0.0],
      "commanded_efforts": [0.0],
      "measured_efforts": [0.0],
      "applied_efforts": [0.0]
    },
    "right": {
      "joint_names": ["..."],
      "positions_rad": [0.0],
      "velocities_rad_s": [0.0],
      "accelerations_rad_s2": [0.0],
      "commanded_efforts": [0.0],
      "measured_efforts": [0.0],
      "applied_efforts": [0.0]
    }
  },
  "objects": {
    "Tblock": {
      "prim_path": "/World/TBlock",
      "position_m": [0.15, 0.0, -0.4],
      "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

字段说明：

- `step`: 全局 physics step。
- `time_s`: 仿真时间，通常为 `step * physics_dt` 或 `(step + 1) * physics_dt`，需要和现有日志保持一致。
- `positions_rad`: 实际关节位置。
- `velocities_rad_s`: 实际关节速度。
- `accelerations_rad_s2`: 由相邻两帧速度差分得到。
- `commanded_efforts`: Python 控制器显式下发的 effort；implicit drive 下可能为 `nan`。
- `measured_efforts`: PhysX 求解器测得/计算的关节 effort。
- `applied_efforts`: Isaac runtime 当前 actuation effort。
- `objects`: 由 env `objects[]` 导入的运行时对象位姿。

## Effort 字段区别

`get_measured_joint_efforts()` 和 `get_applied_joint_efforts()` 不是同一个物理量。

建议在状态流里保留三个不同字段：

- `commanded_efforts`: 项目控制器在 Python 侧显式下发的 effort action。只有 explicit position/velocity 或 direct effort 控制时才有明确数值；implicit drive 通常为 `nan`。
- `measured_efforts`: PhysX 求解器在关节 DOF 上计算/测得的 generalized force，常用于观察实际约束、接触和 drive 求解后的关节负载。
- `applied_efforts`: Isaac articulation runtime 当前记录的 actuation effort，表示施加到关节 actuator/drive 侧的 effort 状态。

因此 Foxglove 状态输出不应只写一个模糊的 `efforts` 字段，除非调用方明确选择了某一种语义。默认更安全的做法是把三类 effort 分开输出。

## 加速度计算

Isaac articulation 通常可以直接读位置和速度，但关节加速度可以先用有限差分：

```text
ddq = (dq_current - dq_previous) / physics_dt
```

注意事项：

- 第一帧没有上一帧速度，可以填 `nan` 或 0。
- 如果采样频率低于 physics 频率，应使用两次采样之间的实际时间差。
- 差分加速度会有噪声，必要时可在 publisher 侧做平滑，但不要让滤波结果影响控制闭环。

## 采样频率

`scene3` 默认 physics frequency 是 240 Hz。每个 physics step 都采样最完整状态是可行的，但 effort 和对象位姿读取可能较贵。

建议支持这些策略：

- `state_rate_hz=0`: 关闭实时状态流。
- `state_rate_hz=60`: 默认推荐，适合 GUI 和外部监控。
- `state_rate_hz=120`: 更细的调试频率。
- `state_rate_hz=240`: 每个 physics step 发布，适合短时间诊断。

采样判断可基于 step interval：

```text
interval_steps = max(1, round(physics_frequency / state_rate_hz))
should_sample = step % interval_steps == 0
```

## 对象位姿读取

对象来源在 `runtime.object_handles` 中。每个 handle 包含：

- `name`
- `kind`
- `runtime_handle`
- `config`
- `model`

刚体对象可以优先记录对象根 prim 的 world pose。动态链式对象例如 capsule rope，可能需要记录根节点、端块或全部 segment body，建议分层支持：

```text
rigid object:
  object root pose

dynamic_chain object:
  root pose
  optional body poses
  optional segment centerline
```

对象 pose 采样也应在主线程完成。后台线程只接收已经序列化好的 `position_m` 和 `orientation_wxyz`。

## Foxglove 输出设计

Foxglove live 和 MCAP 统一通过同一个 sink 接口输出：

```text
FoxgloveStateSink.live(host, port)
FoxgloveStateSink.mcap(path)

publish(snapshot)
close()
```

建议 topic 分层：

- `/joint_states`: 使用 Foxglove 标准 `JointStates`，发布实际 `q/dq`，并把一个选定 effort 语义写入标准 `effort` 字段。
- `/scene`: 使用 Foxglove `SceneUpdate`，发布环境对象、TCP 轨迹、目标点或调试 marker。
- `/linkerbot/state`: Foxglove JSON 编码的项目完整状态 topic，用于保留 `ddq`、三类 effort、对象 pose 和 step/time 等完整快照。

标准 `JointStates` 只有一个 `effort` 字段，不足以同时表达 commanded、measured 和 applied 三种 effort。因此：

- `JointStates.effort` 应做成可配置项，例如 `none`、`commanded`、`measured`、`applied`。
- 完整三类 effort 应保留在 `/linkerbot/state`，不要挤进一个模糊的 `effort` 字段。
- 第一版使用 JSON channel 发布完整快照，后续如需更强 schema 约束，再把 `/linkerbot/state` 升级为显式 JSON schema。

Foxglove live server 和项目交互 WebSocket/TCP 是不同协议，不能共用同一个端口。项目交互 transport 继续只负责 motion command/status，Foxglove 专门负责 telemetry。

## 暂不支持

第一阶段不实现以下能力：

- `{"type": "state"}` 这类交互协议查询。
- TCP JSONL 或项目 WebSocket 持续状态推送。
- 新增 CSV state logger。
- 自定义 Web UI 状态流。

这些能力以后可以作为新的 sink 接到 `StateSnapshot` 后面，但不应改变 Isaac 状态只在主线程采样的原则。

## 推荐落地步骤

1. 新增状态快照 dataclass 和序列化函数。
2. 新增 `StateStream`，内部用 lock/condition 保存最新快照。
3. 新增 `FoxgloveStateSink`，统一 live server 和 MCAP 输出。
4. 新增后台 `StatePublisher`，只消费 `StateStream` 快照并调用 Foxglove sink。
5. 给 `DualRobotRuntime` 增加可选 `state_observer` 或 `state_stream` 字段。
6. 在 `_apply_dual_targets_once()` 的 `world.step()` 后调用状态采样。
7. 给 interactive runtime 增加 CLI 参数：
   - `--state-rate-hz`
   - `--state-include-efforts`
   - `--state-include-objects`
   - `--foxglove-live-host`
   - `--foxglove-live-port`
   - `--foxglove-mcap-path`
   - `--foxglove-joint-effort-field`
8. 补充不依赖 Isaac 的单元测试，测试快照序列化、加速度差分、stream 线程同步和 Foxglove sink 映射。

## 风险和边界

- 不要在后台线程直接访问 `runtime.world`、`runtime.left.articulation`、`runtime.right.articulation` 或 USD stage。
- measured/applied effort 读取失败时应填 `nan`，不能打断仿真主循环。
- 状态流必须是观测功能，不能反向影响 motion execution。
- Foxglove 是可选依赖，未安装 `foxglove-sdk` 时应给出明确错误，不影响未开启 telemetry 的交互模式。
- Foxglove live port 不能和交互 TCP/WebSocket 端口冲突。
- Foxglove 标准 `JointStates` 无法表达三类 effort 和加速度；完整快照通过 `/linkerbot/state` JSON topic 保留。
- publisher 线程不能反压主仿真循环；队列满时应优先丢旧帧或只保留最新快照。

## 和现有日志的关系

现有 `JointTrackingLogger` 已经支持记录：

- 目标位置
- 实际位置
- 目标速度
- 实际速度
- command/action/measured/applied effort

实时状态流不是替代 CSV logger，而是补充 Foxglove 在线观测和 MCAP 录制能力：

- CSV logger 适合离线分析。
- Foxglove live 适合在线监控和 3D/曲线联动调试。
- Foxglove MCAP 适合事后回放和复盘。

第一阶段不重构 CSV logger。后续如果要统一 logger 和 telemetry，应只合并主线程采样层，让 CSV logger 和 Foxglove sink 都从同一个 `StateSnapshot` 取数，避免重复读取 expensive effort。
