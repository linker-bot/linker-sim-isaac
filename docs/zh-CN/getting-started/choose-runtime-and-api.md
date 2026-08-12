# 选择 Mirror 或 Kaleidoscope

语言：[中文](choose-runtime-and-api.md) | [English](../../en/getting-started/choose-runtime-and-api.md)

先按数据形状和产品能力选择模式，再选择外层接口。两个模式没有兼容 alias，也不能在同一个
runtime 中切换。

## 决策表

| 需求 | 选择 |
| --- | --- |
| 映射现实工作站、交互调试、多机器人协同 | Mirror |
| 需要完整 cuRobo trajectory planning 或规划避障 | Mirror |
| 需要 camera、CSV、MCAP、Foxglove 或网络控制 | Mirror |
| 单个现实映像需要 Newton | Mirror `newton_cpu` 或 `newton_cuda` profile |
| 并行强化学习需要 Newton | Kaleidoscope `newton_cuda` profile |
| 数百/数千同构环境进行强化学习 | Kaleidoscope native Torch |
| skrl 全 CUDA rollout | Kaleidoscope + `training.skrl` |
| 必须接入 Gymnasium 工具 | `GymnasiumKaleidoscopeAdapter` |
| 需要 GPU 内 env state/snapshot/clone | Kaleidoscope |

## Mirror 接口

- **命令行进程**：`scripts/mirror.py`，适合人工交互和外部 loopback client。
- **strict wire**：`linkerbot.mirror.v1` envelope，可经 stdin JSONL、TCP JSONL 或 WebSocket。
- **embedded Python**：`create_mirror_runtime(config)` 构造，`MirrorController` 同步分派，
  `run_mirror` 运行 owner-thread loop。

Mirror 请求有明确 request ID、有界 admission、cancel、estop 和 quit。它适合低频业务控制，
不是强化学习逐步 IPC。参见 [Mirror CLI](../reference/mirror-cli.md)与
[Mirror v1 JSON](../reference/mirror-json.md)。

## Kaleidoscope 接口

Kaleidoscope 的 public tensor contract 不随物理后端改变。`physx_cuda` 使用 PhysX CUDA/Fabric，
`newton_cuda` 使用项目自有的 multi-world Newton；后者不是 Isaac Newton extension。
两个训练 composition 都是 headless、GPU-native，并从同一个 `compute.cuda_device` 派生 physics、
Torch、cuRobo 与 trainer device；人工调试可通过 `make_viewport_env()` 为任一后端显式显示一个环境。

### Native Torch（推荐训练路径）

`make_torch_env` 返回 `TorchKaleidoscopeEnv`。`reset`/`step` 直接交换 CUDA tensor，避免
Gymnasium 的主机数组边界。skrl adapter 只消费 `KaleidoscopeTrainingPort`，并在 SAME_STEP
reset 前保存 terminal observation。

### Gymnasium

显式调用 `register_gymnasium_envs()` 后，可使用项目 vector entry point。adapter 批量执行
CUDA↔CPU/NumPy 转换，因此语义兼容但不用于最高吞吐训练。支持 `disabled` 与 `same_step`
autoreset，不实现含糊的 next-step 行为。

### Human viewport

`make_viewport_env()` 读取独立 launch-only profile，只把 `selected_env` 投影到 renderer-facing USD。
训练 step 仍固定 `render=False`，调用方显式调用 `env.render()`；该边界不增加 camera、SyntheticData、
Replicator、录制或 image observation，也不改变 snapshot/clone fingerprint。

### 状态扩展

`get_state`、`set_state`、`snapshot`、`restore_snapshot` 和 `clone_state` 是进程内 CUDA API，
selector 必须是同一 device 上的 `torch.int64` tensor。它们不是 RPC，也不接受 Python env ID list。
物理、task/history/counter 与 RNG 字段的 capture、restore 和 row-to-row clone 均在设备内完成，
只有显式 persistent checkpoint 才允许经过 CPU。

参见 [Kaleidoscope API](../reference/kaleidoscope-api.md)。

## 不要交叉使用

- 不要给 Kaleidoscope 增加 trajectory planner、planning collision cache、avoidance、camera、transport
  或 telemetry worker；物理接触与跨环境隔离仍由所选后端负责；
- 不要把 Mirror scene snapshot 传给 Kaleidoscope episode restore；
- 不要让 training package 直接 import Kaleidoscope 实现子模块或 Isaac owner；
- 不要在 Gymnasium adapter 外隐式把 native CUDA tensor 转为 NumPy。

## 继续

- [Mirror 快速入门](mirror-quickstart.md)
- [Kaleidoscope 快速入门](kaleidoscope-quickstart.md)
- [配置指南](../guides/configuration.md)
