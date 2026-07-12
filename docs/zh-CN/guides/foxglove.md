# Foxglove 快速参考

语言：[中文](foxglove.md) | [English](../../en/guides/foxglove.md)

使用 `uv sync --all-extras` 安装可选 Foxglove SDK。内置 live server 只绑定 loopback，且不提供
认证或 TLS。

## Single Scene

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --gui \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --foxglove-live-port 8766 \
  --state-rate-hz 30 --state-include-objects
```

在 Foxglove 中连接 `ws://127.0.0.1:8766`。使用 Plot 或 Raw Messages 查看
`/joint_states`，使用 Raw Messages 查看 `/linkerbot/state`；runtime profile 启用
`include_scene_markers` 时，使用 3D panel 查看 `/scene`。

JointStates 名称使用 `<robot_label>/<joint_name>`。JSON state 保留 robot 边界，并包含 session
`robot_id` 与稳定 `label`。

## Tiled Scene

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --env scene3_tiled \
  --stdin-eof-policy keep_alive --idle-physics-policy hold_step \
  --foxglove-live-port 8767 \
  --telemetry-env-ids 0,2 --telemetry-rate-hz 10
```

连接 `ws://127.0.0.1:8767`。JSON state 保留两个 requested env；标准 JointStates 与
SceneUpdate 只使用 `telemetry.primary_env_id`。

## Topic 速查

实际状态 topic 名来自 `runtime.telemetry.topics`。相机 prefix 来自各
`sensors.cameras.<name>.output.foxglove_topic_prefix`。

| Topic | 编码 | 内容 |
| --- | --- | --- |
| `/joint_states` | Foxglove `JointStates` | 标准关节数组 |
| `/scene` | Foxglove `SceneUpdate` | Scene/runtime marker |
| `/linkerbot/state` | JSON | 完整 Single Scene 或 selected-env state |
| `<camera-prefix>/rgb` | `RawImage`，`rgb8` | RGB frame |
| `<camera-prefix>/depth` | `RawImage`，`32FC1` | Float32 depth frame |
| `<camera-prefix>/info` | JSON | 每个相机 modality 的 metadata |

分割 modality 会发布 `/info` metadata，但不创建 RawImage channel。相机 live endpoint 在 env
profile 中配置，可以使用与状态 live 不同的端口。

## Endpoint 与录制边界

控制 TCP JSONL、控制 WebSocket、状态 Foxglove live 与相机 Foxglove live 是不同协议，必须
分配不同端口。Foxglove 连接只用于观测，不能执行控制 JSON。

`--foxglove-mcap-path` 录制状态 telemetry。相机 MCAP path 在 env profile 中按相机配置。
live 与 MCAP 可以同时启用。

完整状态 schema 与采样行为见[实时状态流](telemetry.md)，相机配置与 modality 行为见
[相机类型与传感器](cameras.md)。已有文件、队列、配额与关闭语义统一见
[输出与持久化](../reference/outputs.md)。
