# 故障排查

语言：[中文](troubleshooting.md) | [English](../../en/operations/troubleshooting.md)

本文用于先定位故障所属领域。精确字段和错误响应仍由链接的参考页唯一负责。

## Isaac 启动前

| 现象 | 检查 |
| --- | --- |
| `OMNI_KIT_ACCEPT_EULA=Y` 错误 | 阅读并接受适用的 NVIDIA/Kit EULA，再设置无首尾空白的 `Y`、`YES` 或 `1` |
| Import 或资源路径失败 | 从 checkout 根目录以 `PYTHONPATH=src` 运行，不要只安装 `src/` |
| Profile 或 unknown-field 错误 | 运行 `scripts/validate_config.py --runtime-profile <name>`，检查完整字段路径 |
| Runtime mode 不匹配 | Single Scene 入口使用 `single_scene` runtime profile，Tiled Scene 入口使用 `tiled_scene` runtime profile |
| `--dump-effective-config` 后退出 | 这是正常行为：它在 Isaac 启动前解析、输出并退出 |
| Wheel/editable build 被拒绝 | 本项目按设计只作为 checkout workspace 应用运行 |

先阅读[项目概览](../getting-started/project-overview.md)和
[配置](../guides/configuration.md)。

## 进程已启动但没有 Ready

`SINGLE_SCENE_INTERACTIVE_READY` 或 `TILED_SCENE_INTERACTIVE_READY` 表示 command transport 已可接收请求。
在该标记前，应检查第一条启动异常，不要发送控制消息。

- Asset/import 错误：检查所选 env、robot、object、controller 和 cuRobo 路径。
- CUDA/cuRobo 错误：检查锁定环境、GPU 可用性和 robot planning binding。
- Camera 启动错误：按照[相机指南](../guides/cameras.md)区分 fatal exception 与已知 RTX/Fabric warning。
- Port 错误：TCP、WebSocket、state Foxglove 和 camera Foxglove endpoint 需要不同的可用端口。

## Transport 与进程存活

| 现象 | 常见原因 |
| --- | --- |
| 非 loopback host 被拒绝 | 内置 listener 只允许 loopback，且不提供认证/TLS |
| TCP 请求没有返回 | TCP JSONL 要求完整 JSON object 后跟 `\n` |
| WebSocket binary frame 被拒绝 | 只接受 JSON text message |
| JSON 在 dispatch 前被拒绝 | 检查 UTF-8、重复 key、尾随内容、`NaN`/infinity、消息大小和 object shape |
| 新连接被拒绝 | Single Scene TCP 与 WebSocket 共享一份有界连接名额 |
| stdin EOF 后进程退出 | 选择文档规定的 EOF policy，或保持已启用的服务/输出 consumer |
| 空闲时 GUI/camera/telemetry 不刷新 | 使用 `hold_step`；`pause` 会按设计停止空闲 World step |

不要把 listener 直接暴露给不可信网络。远程访问使用认证 TLS proxy 或 SSH tunnel，并让 upstream
保持 loopback。

## Single Scene 命令

- 先用 `status` 发现本会话的 `robot_id`、label、profile、joint group 和 capability；ID 只在会话内有效。
- `rejected` 表示命令没有进入 queue。
- `accepted` 只证明 admission，不证明完成。轮询 `status` 获取 `done`、`failed` 或 `cancelled`，
  或消费 WebSocket lifecycle event。
- 任一 segment 规划失败都会让完整 timeline 在执行前失败。
- 多条独立 JSONL 命令不提供同 tick 协同；应使用 `plan_timeline`。

见 [Single Scene JSON](../reference/single-scene-json.md)和
[控制与轨迹](../guides/control-and-trajectories.md)。

## Tiled Scene 命令与 Selector

- 每个 env-scoped 命令都要求显式、非空、唯一的 `env_ids`。
- 多机器人命令必须提供该消息定义的 robot selector。
- `values` 行数必须是 1 或 `len(env_ids)`，列宽遵循 command-space/action 规则。
- `get_state` 是当前进程内调试 shape；持久恢复使用 Snapshot。
- 共享 World step 时，未选 env 也会在保持 target 的同时推进。

通过 `status` 检查 `num_envs`、robot command joint、env origin、queue resource、telemetry、planner
和 camera diagnostics。见 [Tiled Scene JSON](../reference/tiled-scene-json.md)。

## Planning 与 IK

| 现象 | 检查 |
| --- | --- |
| Task-space request 不支持 | Robot cuRobo binding、TCP frame 和 request kind |
| `avoid_collisions=true` 失败 | 完整 robot collision model、planning world、cache 和 backend capability |
| Linear backend 拒绝请求 | 它只支持 joint interpolation，不提供 IK 或 collision avoidance |
| Batch 过大 | Runtime `max_batch_problems` 和 `oversize_request_policy` |
| Planning request 一直 queued | 调用 `planner_status` 或 `step_trajectory` dispatch/collect Tiled Scene work |
| Planner 成功但没有轨迹 | 检查 `loaded` 与 `load_rejected` 中的 playback admission 失败 |
| 第一次请求很慢 | cuRobo context、kernel compilation 和 warmup 按资源 owner lazy 创建 |

见[运动规划](../guides/motion-planning.md)和
[碰撞模型](../guides/collision-models.md)。

## Trajectory Playback

- 检查有限且严格递增的 `times`、position shape 和 `joint_names` 顺序。
- 在 `trajectory_status` 中检查每 env queue depth、sample 数和 duration 上限。
- Append 计算 existing + new；replace 校验 new sequence。
- Playback 只通过 `step_trajectory` 推进；load 本身不会 step physics。
- 稀疏 `hand` path 不是 Single Scene 风格的同步 arm/hand timeline。

## Snapshot 与状态修改

- 使用稳定 label/profile/fingerprint 匹配，不使用缓存的会话 robot ID。
- `strict=true` 要求每个已映射 entry 的 joint/body name set 相等；目标中额外 robot/object 保持不动。
  只有有意改名时才使用 `label_map`。
- Pending snapshot request 可以在执行前 timeout；executing 后等待真实结果，shutdown 返回显式 running
  状态是唯一例外。
- Rollback 失败或不可逆 commit 后失败会让 runtime 进入 fail-stop。必须重建，不能在无法证明一致的
  状态上重试 mutation。

见 [Snapshot 参考](../reference/snapshots.md)。

## Telemetry、Camera 与文件

| 现象 | 检查 |
| --- | --- |
| Telemetry sink 没有打开 | Effective rate 必须大于 0，且配置 live port 或 MCAP path |
| Foxglove 连到错误服务 | Control TCP/WebSocket 与 Foxglove 使用不同端口和协议 |
| Joint effort 为空 | 开启 effort sampling，并选择目标 effort field |
| Foxglove 没有 segmentation image | Segmentation 本地保存为 `.npy`；RawImage channel 只支持 RGB/depth |
| Output target 已存在 | 启动前选择该 sink 允许的显式 existing-data policy |
| Camera publisher 达到 quota 后停止 | 增加已规划的每相机 budget 或使用新 target，并检查 orphan cleanup 状态 |
| 持久化输出丢帧 | Persistent sink 会拒绝 lossy queue policy，应使用有界 block 或 fail-fast 行为 |

见[遥测](../guides/telemetry.md)、[相机](../guides/cameras.md)和
[输出参考](../reference/outputs.md)。

## 资产工具

Rope 与 T block builder 需要接受 EULA，并启动 headless SimulationApp。应从 checkout 根目录运行
`build_asset.py`，不是库文件 `builder.py`。若 runtime 仍载入旧几何，检查 `--output` 是否与 object
profile 的 `asset_path` 一致，并在启动场景前预览生成的 USD。

见[物体资产](../development/object-assets.md)和
[USD 预览](../development/usd-preview.md)。

## Shutdown

任何 `*_SHUTDOWN_TIMEOUT` 都表示关闭未完成，即使外层脚本打印了最终 step。Owner 会保留超时资源
供重试，不能在 live child 下方关闭 Kit。应分别检查 transport、state publisher、camera publisher、
planner 和 runtime 状态；它们使用独立 timeout budget。

完整不变量和恢复边界见[已知约束](constraints.md)。
