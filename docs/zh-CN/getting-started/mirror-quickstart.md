# Mirror 快速入门

语言：[中文](mirror-quickstart.md) | [English](../../en/getting-started/mirror-quickstart.md)

Mirror 将 selector `mirror/scene3`（文件 `configs/scenes/mirror/scene3.yaml`，内部
`scene.id: scene3`）映射为一个交互仿真 World。CLI 默认的 `physx_cpu` 使用 PhysX CPU；
`physx_cpu_hybrid` 使用 240 Hz PhysX CPU 专用 hybrid composition；`newton_cpu` 与 `newton_cuda`
分别选择 Newton CPU/CUDA，四者都由 Mirror 派生一个 world。首次使用新的 checkout 时，
请先完成[安装与环境准备](installation.md)。

## 1. 准备

```bash
uv sync --extra simulation --extra visualization
export OMNI_KIT_ACCEPT_EULA=Y
```

默认 `mirror/scene3` 引用本仓库不再分发的 NVIDIA Warehouse 视觉素材。具体目标路径和
检查命令见[安装与环境准备](installation.md)。配置图校验并不会下载该素材，因此需要仓库视觉
效果时应单独确认文件存在。

先做不启动 Isaac 的配置校验：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
```

四个 Mirror mode 都引用 `configs/curobo/mirror.yaml` 和 `configs/planning/mirror.yaml`。前者拥有
IK batch 容量，以及单请求 MotionPlanner 的 warmup、seed、CUDA graph、碰撞能力与 cache 容量；
`kinematics.max_batch_size` 不控制 planner，MotionPlanner context 固定 `max_batch_size=1`。后者只拥有
duration、采样周期、默认避障、刷新和 coordination，以及不可由 wire 覆盖的每请求 timeout；planning
profile 不选择 backend。wire planning segment 可覆盖 duration、采样周期、避障和刷新，coordination
只能在 wrapper/timeline 顶层覆盖。两者也统一引用
`configs/control/mirror.yaml`，默认 controller bundle 由 `physics.engine` 派生。

Hybrid 模式另选可选 `profiles.hybrid_control: hybrid_force_position`，通过 v3 tare 和 motion operation
使用；普通默认 `physx_cpu` 不会隐式开启该能力。配置校验命令为：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu_hybrid
```

## 2. 启动 stdin JSONL

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py --profile physx_cpu
```

`physx_cpu` 是 Mirror CLI 默认值，因此可以省略 `--profile physx_cpu`。Canonical control profile 还默认
开启墙钟同步；将 `control.sync_simulation_to_wall_clock` 设为 `false` 可取消 pacing。

当 output profile 设置 `outputs.render.gui: true` 时，Newton scene3 会同时出现人工观察
`Viewport` 和数据采集窗口 `NewtonCamera:world_rgbd`。只在 `Viewport` 中调整观察视角：
`Alt + 左键` 旋转、中键平移、右键环视、滚轮缩放。`NewtonCamera:*` 的相机导航会按窗口关闭，
鼠标操作不会改写 RGB-D 传感器外参。

看到 `MIRROR_INTERACTIVE_READY` 后，每行发送一个完整 JSON object：

```json
{"protocol":"linkerbot.mirror.v1","request_id":"status-1","operation":"runtime.status","arguments":{}}
```

正常响应包含相同 `request_id`、`ok` 和 `result`。退出：

```json
{"protocol":"linkerbot.mirror.v1","request_id":"quit-1","operation":"runtime.quit","arguments":{}}
```

stdin EOF 默认也请求退出。不要直接 kill 正在写 camera/MCAP 的进程；正常关闭会先停止 ingress，
再关闭 output/camera/planner，最后释放 Isaac session。

## 3. 可选 loopback transport

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py \
  --profile physx_cpu \
  --no-stdin \
  --tcp-jsonl 127.0.0.1:8765 \
  --websocket 127.0.0.1:8766
```

内置 listener 只接受 loopback，不提供认证或 TLS。远程使用必须经认证代理或 SSH tunnel。

## 4. Embedded Python

```python
from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror import create_mirror_runtime, run_mirror

config = load_mirror_config("physx_cpu")
runtime = create_mirror_runtime(config)
result = run_mirror(runtime, max_iterations=100)
assert result.close_report is None or result.close_report.stopped
```

`MirrorRuntime`、其 controller、planner、camera bundle 和 session 有严格线程/关闭所有权；不要把
runtime handle 交给后台 transport 线程调用。

## 5. Newton smoke

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/smoke_mirror_physics.py \
  --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cuda
PYTHONPATH=src .venv/bin/python scripts/smoke_mirror_physics.py \
  --profile newton_cuda
```

Newton experience 不继承 PhysX experience。渲染时每帧只允许一次 physics-to-USD sync；camera
由 Mirror `CameraBundle` 拥有，不由 Newton manager 关闭。

维护的完整 Mirror 门禁同时运行四个正式 mode profile；七 Kit closure 门禁还会覆盖无渲染的 Newton
physics-only experience：

```bash
just smoke-mirror
just smoke-runtime-kits
```

`just test-simulation` 会把它们与 Kaleidoscope 双后端、Newton 容量和 PhysX 显存预算一起执行。

继续阅读：[Mirror CLI](../reference/mirror-cli.md)、
[Mirror v1/v2/v3 JSON 与运动示例](../reference/mirror-json.md)、
[运动规划](../guides/motion-planning.md)。
