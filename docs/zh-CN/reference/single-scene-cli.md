# Single Scene CLI 参考

语言：[中文](single-scene-cli.md) | [English](../../en/reference/single-scene-cli.md)

Single Scene 入口是 `scripts/single_scene_interactive.py`。它解析 Single Scene runtime
profile，只应用本次命令显式传入的 CLI 覆盖，创建一个 `SingleSceneRuntime`，然后提供
Single Scene JSON 协议。一个 Single Scene runtime 可以包含所选 env profile 声明的任意数量机器人。

第一次完整运行请从 [Single Scene 快速开始](../getting-started/single-scene-quickstart.md)开始；请求与响应
字段见 [Single Scene JSON 参考](single-scene-json.md)。

## 解析顺序

Runtime 值按以下顺序解析：

1. 强类型代码默认值。
2. `--runtime-profile` 选择的 YAML。
3. 本次调用显式出现的 CLI 字段。

除了 `--runtime-profile` 和 `--dump-effective-config`，未传入的 option 在 parser
中都保持 `None`。`None` 的含义是“不覆盖所选 YAML”，并不是最终运行值。下表因此同时列出
argparse 原始结果和内置 `default_single_scene` profile 的最终值。

下面的命令不会启动 Isaac，可查看最终值及每个字段的来源：

```bash
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene --dump-effective-config
```

它解析所选 runtime 和 env 配置、输出 JSON，然后在创建 `SimulationApp` 前退出。

## 完整参数表

| 参数 | 省略时的 argparse 值 | 内置 `default_single_scene` 最终值 | 约束 |
|---|---|---|---|
| `-h`, `--help` | 不适用 | 不适用 | 输出 argparse help，在解析配置或启动 Isaac 前退出。 |
| `--runtime-profile NAME` | `default_single_scene` | `default_single_scene` | 选择 `configs/runtime/NAME.yaml`；必须解析为 `mode: single_scene`。 |
| `--dump-effective-config` | `false` | `false` | 输出最终配置、字段来源和 fingerprint，在启动 Isaac 前退出。 |
| `--env NAME` | `None` | `scene1` | 覆盖 `runtime.profiles.env`，值来自 `configs/envs/`。 |
| `--curobo-profile NAME` | `None` | `default` | 覆盖 `runtime.profiles.curobo`；创建 cuRobo 能力时消费。 |
| `--planner-backend {curobo,linear}` | `None` | `curobo` | 选择 cuRobo 或可执行的关节空间插值；`linear` 不提供 IK 或碰撞检查。 |
| `--logging-profile NAME` | `None` | `default_logger` | 选择 `configs/logging/` 下的 Single Scene joint CSV profile。 |
| `--control-mode MODE` | `None` | `position` | 使用 `position`、`velocity` 或 `effort` 覆盖 articulation 控制；不受支持的 controller 组合会被拒绝。 |
| `--gui`, `--no-gui` | `None` | `false` | 显式开启或关闭 Isaac GUI。 |
| `--stdin-eof-policy {exit,keep_alive}` | `None` | `exit` | 决定 stdin 自然 EOF 是否可请求退出，或让进程继续存活。 |
| `--idle-physics-policy {pause,hold_step}` | `None` | `hold_step` | 空闲时暂停，或保持当前 target 并继续推进 World。 |
| `--tcp-jsonl-host HOST` | `None` | `127.0.0.1` | 覆盖 TCP bind host；只覆盖 host 不会启用 TCP。 |
| `--tcp-jsonl-port PORT` | `None` | 不启用（`null`） | 设置 TCP JSONL port，并启用该 endpoint。 |
| `--websocket-host HOST` | `None` | `127.0.0.1` | 覆盖 WebSocket bind host；只覆盖 host 不会启用 WebSocket。 |
| `--websocket-port PORT` | `None` | 不启用（`null`） | 设置 WebSocket port，并启用该 endpoint。 |
| `--state-rate-hz HZ` | `None` | `60.0` | 覆盖状态采样频率；只有 `0` 会关闭状态输出且不打开状态 sink，负值会被拒绝。 |
| `--state-include-efforts`, `--no-state-include-efforts` | `None` | `false` | 包含或省略 commanded、measured、applied effort。 |
| `--state-include-objects`, `--no-state-include-objects` | `None` | `false` | 在状态输出中包含或省略 runtime object pose。 |
| `--foxglove-live-host HOST` | `None` | `127.0.0.1` | 覆盖 Foxglove live bind host；只覆盖 host 不会启用 live 输出。 |
| `--foxglove-live-port PORT` | `None` | 不启用（`null`） | 设置 Foxglove live port，并启用该 endpoint。 |
| `--foxglove-mcap-path PATH` | `None` | 不启用（`null`） | 按 runtime 输出策略把状态流写入该 MCAP 路径。 |
| `--foxglove-joint-effort-field {none,commanded,measured,applied}` | `None` | `none` | 选择 `/joint_states.effort` 来源；非 `none` 值要求开启 effort 采样。 |

成对的布尔参数写同一个字段。若两种形式同时出现，argparse 采用最后一次；每次调用只使用一种
形式，启动日志会更明确。

## Endpoint 与进程存活关系

`--tcp-jsonl-port`、`--websocket-port`、`--foxglove-live-port` 分别启用三个不同
service。不要让两个 service 使用相同 host/port；配置解析不会预检这个冲突，后启动的 listener
会在 bind 时失败，使用不同 port 最清楚。内置 listener 只接受 `localhost` 或数值 loopback
地址，而且不提供认证或 TLS。远程访问应让上游继续监听 loopback，再使用认证 TLS 代理或
SSH tunnel。

进程存活和空闲物理推进是两个独立设置：

- `stdin_eof_policy=exit` 仅在没有 TCP、WebSocket、状态流或相机输出继续持有进程时，
  才把自然 EOF 转成退出请求。
- `stdin_eof_policy=keep_alive` 允许纯 stdin 进程在 EOF 后继续存活。
- GUI、相机或 live state 在无控制请求时仍需刷新，应使用 `idle_physics_policy=hold_step`。
- 纯请求驱动服务可用 `idle_physics_policy=pause` 避免空闲物理计算。

Single Scene CLI 不暴露 `stdin_enabled`、transport queue 容量、相机输出策略、输出冲突策略或
shutdown timeout；这些值属于 runtime YAML。见[配置参考](configuration.md)和
[Telemetry](../guides/telemetry.md)。

## 启动与退出标记

每次实际启动 Isaac 前，部署环境都必须显式接受 EULA：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene --tcp-jsonl-port 8765
```

| 标记 | 含义 |
|---|---|
| `SINGLE_SCENE_INTERACTIVE_CONFIG runtime_profile=<name> fingerprint=<hash>` | 配置已解析，即将创建 runtime。 |
| `SINGLE_SCENE_INTERACTIVE_READY` | Transport 和命令队列已可接收请求；延迟渲染工作仍可能在后续 rendered step 发生。 |
| `SINGLE_SCENE_INTERACTIVE_EXIT` | 交互循环进入最终关闭。 |
| `SINGLE_SCENE_INTERACTIVE_OK steps=<n>` | 关闭正常返回；`<n>` 是最终全局仿真 step。 |
| `SINGLE_SCENE_INTERACTIVE_FAILED <Exception>: <message>` | 启动或运行异常逃逸，wrapper 以非零状态退出。 |

Shutdown timeout 诊断可能出现在 `EXIT` 和 `OK` 之间；即使 main function 返回，也应把它
当作资源清理失败处理。见[故障排查](../operations/troubleshooting.md)。
