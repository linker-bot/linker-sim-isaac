# 实时状态流

语言：[中文](telemetry.md) | [English](../../en/guides/telemetry.md)

Single Scene 与 Tiled Scene runtime 可以将 immutable 仿真状态 snapshot 发布到 Foxglove live server、
MCAP 文件或同时发布到两者。Telemetry 只用于观测，不接收控制命令，也不修改仿真状态。

## 安装与配置

Foxglove 输出需要可选 SDK 依赖：

```bash
uv sync --all-extras
```

进程级 telemetry 配置属于 `configs/runtime/*.yaml`。显式 CLI 值只覆盖本次启动所选 runtime
profile。

```yaml
runtime:
  telemetry:
    primary_env_id: 0
    rate_hz: 30.0
    buffer_size: 1
    drop_policy: latest
    on_error: stop
    include_joint_states: true
    include_state_json: true
    include_scene_markers: false
    include_efforts: false
    include_objects: false
    joint_effort_field: none
    topics:
      joint_states: /joint_states
      scene: /scene
      state: /linkerbot/state
    mcap:
      path: null
    foxglove_live:
      enabled: true
      host: 127.0.0.1
      port: 8767
```

Tiled Scene runtime profile 还使用 `selected_env_ids` 和 `publish_decimation`：

```yaml
runtime:
  telemetry:
    primary_env_id: 0
    selected_env_ids: [0, 2]
    publish_decimation: 2
```

`primary_env_id` 必须属于选中的 Tiled env。三个 topic 名必须是互不相同的绝对路径。
`rate_hz: 0` 会完全关闭 telemetry，因此配置的 live endpoint 和 MCAP path 都不会打开。

## Endpoint 边界

控制 TCP JSONL、控制 WebSocket、Foxglove 状态 live 与相机 live 是不同服务，不能共用端口；
向 Foxglove endpoint 发送控制 JSON object 不会执行命令。

所有内置 listener 只接受 `localhost` 或数值 loopback 地址，且不提供认证或 TLS。远程访问必须
通过以该 loopback endpoint 为上游的认证 TLS proxy 或 SSH tunnel。

并发运行的每个 runtime 应使用不同端口和不同 topic namespace。相机 live port 在 env profile
中按相机配置，不是状态流的 `--foxglove-live-port`。

## 采样与线程

Single Scene 在共享 `world.step()` 后采样。仿真线程把全部 robot 和可选 object 读取为一个 immutable
`StateSnapshot`，publisher 线程不访问 Isaac 或 USD object。因此同一 snapshot 中的全部 robot
具有相同 physics time。

Single Scene 将请求频率量化为整数 physics-step 间隔：

```text
interval_ticks = max(1, round(1 / (physics_dt * rate_hz)))
actual_rate_hz = 1 / (physics_dt * interval_ticks)
```

请求频率高于 physics frequency 时最多每个 physics step 一帧。关节加速度由相邻两个已采样
snapshot 的速度差分得到；第一帧和 reset 后第一帧没有前值，JSON 使用 `null`。

Tiled Scene telemetry 在主线程读取完成的 `get_state` 结果，并只冻结 `selected_env_ids`。`rate_hz`
调度 idle-loop 发布，`publish_decimation` 按 global step 过滤普通事件。`reset` 与 `set_state`
事件不受 decimation 限制，成功的交互响应也会尝试发布当前状态。后台 publisher 只消费冻结帧。

## Topic 与 Schema

Single Scene 与 Tiled Scene 使用相同的三个配置 topic 名：

| 配置字段 | 编码 | Single Scene | Tiled Scene |
| --- | --- | --- | --- |
| `topics.joint_states` | Foxglove `JointStates` | 全部 robot | `primary_env_id` 中的全部 robot |
| `topics.scene` | Foxglove `SceneUpdate` | Runtime object marker | `primary_env_id` 的 robot TCP marker 与可选 object marker |
| `topics.state` | JSON | 完整 Single Scene snapshot | 完整 selected-env state |

标准 `JointStates` 按稳定 runtime 顺序拼接 robot。名称使用
`<robot_label>/<joint_name>`，避免不同 robot 的同名关节冲突。需要 robot 边界或全部选中
Tiled env 时应使用 JSON topic。

全部时间戳使用仿真时间。关节位置单位为 rad，速度为 rad/s，加速度为 rad/s2，世界位置为 m。
非有限数编码为 JSON `null`。

## Single Scene Telemetry

启动 Single Scene live 状态流：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --gui \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --state-rate-hz 30 \
  --state-include-efforts --state-include-objects \
  --foxglove-joint-effort-field measured \
  --foxglove-live-port 8766
```

设置 `--foxglove-mcap-path` 可以把同一状态流写入 MCAP；live 与 MCAP 可以同时启用。至少配置
其中一个目的地才会创建状态流。

重要 Single Scene CLI override 如下：

| 参数 | 含义 |
| --- | --- |
| `--state-rate-hz` | 非负目标采样频率；`0` 关闭 telemetry |
| `--state-include-efforts` | 采样 commanded、measured 与 applied effort array |
| `--state-include-objects` | 采样 runtime object root 世界位姿 |
| `--foxglove-live-host` / `--foxglove-live-port` | Loopback live endpoint |
| `--foxglove-mcap-path` | 状态 MCAP 目的地 |
| `--foxglove-joint-effort-field` | 标准 JointStates effort 来源：`none`、`commanded`、`measured` 或 `applied` |

启用 Single Scene telemetry sink 时，`include_scene_markers: true` 要求同时设置
`include_objects: true`，因为 scene marker 来自 runtime object。

Single Scene JSON payload 结构如下：

```json
{
  "step": 120,
  "time_s": 0.5041666667,
  "phase": "ik_offset",
  "robots": [
    {
      "robot_id": 0,
      "label": "ar5v2_l6v1_0",
      "joint_names": ["AR5V2_L_arm_joint_1"],
      "positions_rad": [0.12],
      "velocities_rad_s": [0.03],
      "accelerations_rad_s2": [0.4],
      "commanded_efforts": [null],
      "measured_efforts": [0.8],
      "applied_efforts": [0.75]
    }
  ],
  "objects": {
    "Tblock": {
      "prim_path": "/World/TBlock",
      "position_m": [0.15, 0.0, -0.4],
      "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

空闲时可能没有 `phase`。关闭 effort 采样时 effort array 为 `null`；单个不可用值也使用
`null`。读取可选 effort API 失败不会终止状态流。

## Tiled Scene Telemetry

启动 Tiled Scene 状态流：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --env scene3_tiled \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --foxglove-live-port 8767 \
  --telemetry-env-ids 0,2 \
  --telemetry-rate-hz 10 \
  --telemetry-decimation 2
```

重要 Tiled Scene CLI override 如下：

| 参数 | 含义 |
| --- | --- |
| `--telemetry-env-ids` | 非空、唯一、范围内的逗号分隔 env ID |
| `--telemetry-primary-env-id` | 标准 topic 使用的单个 selected env |
| `--telemetry-rate-hz` | Idle-loop 发布频率；`0` 关闭全部 telemetry sink |
| `--telemetry-decimation` | 普通事件的正整数 global-step filter |
| `--telemetry-full-batch-json` | 启用 selected-env JSON topic |
| `--telemetry-joint-states` | 启用 primary env 标准 JointStates |
| `--foxglove-live-port` | 启用状态 live sink |
| `--foxglove-mcap-path` | 启用状态 MCAP sink |

自定义 JSON 保留 selected-env 维度：

```json
{
  "event": "step",
  "step": 42,
  "time_s": 0.175,
  "env_ids": [0, 2],
  "state": {
    "robots": {
      "ar5v2_l6v1_0": {
        "joint_names": ["AR5V2_L_arm_joint_1"],
        "joint_positions": [[0.1], [0.2]],
        "joint_velocities": [[0.0], [0.0]],
        "tcp_positions_world": [[0.3, 0.0, 0.4], [6.3, 0.0, 0.4]],
        "tcp_orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
      }
    },
    "episode_steps": [42, 42],
    "episode_ids": [0, 0]
  },
  "trigger": {
    "event": "step",
    "accepted": true,
    "kind": "joint_delta_pos"
  }
}
```

robot map key 是稳定 runtime label。`include_objects: false` 省略 object state 和 object
marker，但不会删除 robot TCP marker。`include_efforts: true` 采样 measured/applied effort；
标准 Tiled Scene JointStates 在可用时使用 measured effort。

## 输出受压与持久化

Telemetry 使用有界、producer 非阻塞的 buffer。默认 `buffer_size: 1` 与
`drop_policy: latest` 保留最新 snapshot；另外可选 `drop_oldest` 和 `drop_newest`。
publisher 出错后由 `on_error` 选择 `stop` 或 `continue`。

状态 MCAP 使用 `runtime.output.mcap_existing_file_policy`，并拒绝 `resume`。正常关闭时先停止
接纳，再排空已接纳 snapshot，最后关闭 live/MCAP sink。已有数据策略、联合路径预检、buffer
行为、join timeout 与失败状态的精确定义见[输出与持久化](../reference/outputs.md)。

## Status 与故障排查

Single Scene 与 Tiled Scene telemetry status 包含 `buffer_depth`、`buffer_capacity`、
`dropped_snapshots`、`error_count`、`last_published_sequence`、`last_error`、
`thread_alive`、`shutdown_timed_out` 和 `sink_closed`。

没有状态流启动事件
: 确认 `rate_hz > 0`，并至少配置一个 live port 或 MCAP path。

Foxglove 无法连接
: 确认客户端连接的是 Foxglove live port，而不是控制端口或相机端口；检查 rate 为正，并确保
远程 proxy 最终连接到配置的 loopback 地址。

JointStates effort 为空
: Single Scene 需要启用 effort 采样并选择非 `none` effort field；Tiled Scene 需要启用
`include_efforts`。当前 Isaac articulation wrapper 也必须能提供所需 effort。

第一帧 acceleration 为 `null`
: 在存在两个采样 velocity frame 之前这是正常结果；reset 会清除 velocity history。

Tiled Scene 空闲时 live 数据停止
: 使用 `--idle-physics-policy hold_step` 让主循环持续 pump runtime；stdin 长期运行还应使用
`--stdin-eof-policy keep_alive`。

Topic 选择和连接步骤见 [Foxglove 快速参考](foxglove.md)。控制 payload 见
[Single Scene JSON](../reference/single-scene-json.md)与
[Tiled Scene JSON](../reference/tiled-scene-json.md)。
