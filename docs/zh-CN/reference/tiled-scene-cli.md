# Tiled Scene CLI 参考

语言：[中文](tiled-scene-cli.md) | [English](../../en/reference/tiled-scene-cli.md)

Tiled Scene 入口是 `scripts/tiled_scene_interactive.py`。它创建独立的
`TiledSceneRuntime`，不会创建或包装 `SingleSceneRuntime`。所选 env profile
负责 clone 数量和拓扑；runtime profile 负责进程、planner、transport、playback、telemetry
和输出策略。

第一次完整运行请从 [Tiled Scene 快速开始](../getting-started/tiled-scene-quickstart.md)开始；消息和 selector
见 [Tiled Scene JSON 参考](tiled-scene-json.md)。

## 解析顺序

Runtime 值依次来自强类型代码默认值、所选 runtime YAML，以及本次调用显式传入的 CLI 字段。
除了 `--runtime-profile` 和 `--dump-effective-config`，省略的 parser 值都为 `None`，
因此不会覆盖 YAML。

下面的命令不会创建 Isaac，可查看最终值及字段来源：

```bash
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene --dump-effective-config
```

## 完整参数表

| 参数 | 省略时的 argparse 值 | 内置 `default_tiled_scene` 最终值 | 约束 |
|---|---|---|---|
| `-h`, `--help` | 不适用 | 不适用 | 输出 argparse help，在解析配置或启动 Isaac 前退出。 |
| `--runtime-profile NAME` | `default_tiled_scene` | `default_tiled_scene` | 选择 `configs/runtime/NAME.yaml`；必须解析为 `mode: tiled_scene`。 |
| `--dump-effective-config` | `false` | `false` | 输出最终配置、字段来源和 fingerprint，在启动 Isaac 前退出。 |
| `--env NAME` | `None` | `scene3_tiled` | 覆盖 `runtime.profiles.env`；clone 数量和 per-env 结构仍来自该 env profile。 |
| `--gui`, `--no-gui` | `None` | `false` | 显式开启或关闭 Isaac GUI。 |
| `--default-decimation TICKS` | `None` | `2` | 兼容 action 省略 `decimation` 时使用的正 physics tick 数。 |
| `--planner-workers COUNT` | `None` | `2` | 异步 planner 的正 worker 数；每个并发 cuRobo worker 都持有独立 GPU context/cache 资源。 |
| `--max-pending-requests COUNT` | `None` | `64` | queued 和 running 异步规划请求的正上限。 |
| `--max-completed-results COUNT` | `None` | `256` | completed planner 摘要缓存上限；`0` 不保留。 |
| `--stdin`, `--no-stdin` | `None` | `true` | 显式开启或关闭 stdin JSONL。 |
| `--stdin-eof-policy {exit,keep_alive}` | `None` | `exit` | 决定 stdin 自然 EOF 是否可请求退出，或让进程继续存活。 |
| `--idle-physics-policy {pause,hold_step}` | `None` | `pause` | 空闲时暂停，或保持 target 并继续推进共享 World。 |
| `--tcp-jsonl-host HOST` | `None` | `127.0.0.1` | 覆盖 TCP bind host；只覆盖 host 不会启用 TCP。 |
| `--tcp-jsonl-port PORT` | `None` | 不启用（`null`） | 设置 TCP JSONL port，并启用该 endpoint。 |
| `--websocket-host HOST` | `None` | `127.0.0.1` | 覆盖 WebSocket bind host；只覆盖 host 不会启用 WebSocket。 |
| `--websocket-port PORT` | `None` | 不启用（`null`） | 设置 WebSocket port，并启用该 endpoint。 |
| `--foxglove-live-host HOST` | `None` | `127.0.0.1` | 覆盖 Foxglove live bind host；只覆盖 host 不会启用 live 输出。 |
| `--foxglove-live-port PORT` | `None` | 不启用（`null`） | 设置 Foxglove live port，并启用该 endpoint。 |
| `--foxglove-mcap-path PATH` | `None` | 不启用（`null`） | 按输出策略把 Tiled Scene telemetry 写入该 MCAP 路径。 |
| `--telemetry-env-ids IDS` | `None` | `0` | 逗号分隔的 selected env IDs；解析与 runtime resolution 会拒绝空、重复、负数或越界选择。 |
| `--telemetry-primary-env-id ID` | `None` | `0` | 标准单 env topic 使用的 env，必须包含在 selected env IDs 中。 |
| `--telemetry-decimation TICKS` | `None` | `1` | 常规 telemetry publication 之间的正 global-step 间隔。 |
| `--telemetry-rate-hz HZ` | `None` | `10.0` | Telemetry 采样频率；只有 `0` 会关闭 Tiled Scene telemetry 且不打开 live 或 MCAP sink，负值会被拒绝。 |
| `--telemetry-full-batch-json`, `--no-telemetry-full-batch-json` | `None` | `true` | 包含或省略 selected-env JSON state。 |
| `--telemetry-joint-states`, `--no-telemetry-joint-states` | `None` | `true` | 包含或省略 primary env 的标准 JointStates。 |

成对的布尔参数写同一个字段。若两种形式同时出现，argparse 采用最后一次；每次调用只使用一种
形式。

## 启用与组合校验

CLI listener port 会同时设置端口并启用对应 endpoint；只改 host 会保留 YAML 中原有的启用
状态。TCP、WebSocket 和 Foxglove live 是三个独立 service。不要让两个 service 使用相同
host/port；配置解析不会预检这个冲突，后启动的 listener 会在 bind 时失败，使用不同 port 最清楚。
Listener host 只允许 `localhost` 或数值 loopback 地址；内置 service 不提供认证或 TLS。

Telemetry selector 作为整体校验：`primary_env_id` 必须属于 `selected_env_ids`，每个 ID
必须小于所选 env 的 `tiled.num_envs`。只覆盖其中一侧可能使原本有效的 profile 变成无效。
即使配置了 live port 或 MCAP path，`--telemetry-rate-hz 0` 也不会创建 telemetry sink。

stdin 存活策略与 physics idle 策略相互独立。长期 network service 可用 `--no-stdin`；
GUI、相机或持续采样 telemetry 通常需要 `hold_step`，纯请求驱动的 headless service 可用
`pause`。

CLI 不提供 `num_envs`、机器人选择、planner backend、cuRobo profile、control mode、碰撞
策略或 planner batch mode 参数。这些值属于 env 或 runtime YAML。Tiled Scene 仅支持 position
control。见[配置参考](configuration.md)、[碰撞模型](../guides/collision-models.md)和
[运动规划](../guides/motion-planning.md)。

## 启动与退出标记

每次实际启动 Isaac 前都必须显式接受 EULA：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene --no-stdin --tcp-jsonl-port 8765
```

| 标记 | 含义 |
|---|---|
| `TILED_SCENE_INTERACTIVE_CONFIG runtime_profile=<name> fingerprint=<hash>` | 配置已解析，即将创建 runtime。 |
| `TILED_SCENE_INTERACTIVE_READY` | 已启用 network transport 可以入队请求；配置的 stdin reader 和 main loop 紧接该标记启动。 |
| `TILED_SCENE_INTERACTIVE_EXIT` | 入口到达最终关闭；正常完成随后以状态 0 退出，Tiled Scene 没有额外成功标记。 |
| `TILED_SCENE_INTERACTIVE_FAILED <Exception>: <message>` | 启动或运行异常逃逸，wrapper 以非零状态退出。 |

任意 `TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT` 都表示清理未完成，即使之后输出 `EXIT` 也需要
排查。见[故障排查](../operations/troubleshooting.md)。
