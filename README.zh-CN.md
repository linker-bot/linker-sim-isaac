# linker-sim-isaac

语言：[中文](README.zh-CN.md) | [English](README.en.md)

linker-sim-isaac 是基于 checkout 运行的 Isaac Sim 机器人操作工作区。项目不再提供可互换的
“单场景/批量场景”兼容层，而是把两种目标拆成边界明确的产品：

- **Mirror**：把一个真实工作站映射到一个仿真 World，保留交互控制、完整运动规划、碰撞模型、
  相机、遥测、日志以及冷路径场景快照。
- **Kaleidoscope**：把一个任务 scene prototype 复制成大量隔离环境，面向 Torch/Gymnasium/skrl
  强化学习。物理、状态、snapshot、clone、批量 IK 和同步直线动作均驻留在同一 CUDA device；
  默认使用 PhysX CUDA，也可选择项目自有的 multi-world Newton。两个后端的训练路径均为
  headless GPU-native，并另有只显示一个选中环境的显式调试 viewport；均不创建轨迹 planner、规划
  避障、相机、SyntheticData、Replicator、录制、transport 或遥测。

产品工厂只会从以下 7 个正式 Kit experience 中选择一个：

| 产品 | engine / execution | 渲染闭包 | 正式 Kit experience |
| --- | --- | --- | --- |
| Mirror | PhysX / CPU | 由 outputs profile 控制 | `apps/linkerbot_sim.mirror.physx.python.kit` |
| Mirror | Newton / CPU 或 CUDA | 关闭 | `apps/linkerbot_sim.mirror.newton.python.kit` |
| Mirror | Newton / CPU 或 CUDA | 开启 | `apps/linkerbot_sim.mirror.newton_render.python.kit` |
| Kaleidoscope | PhysX / CUDA | 训练 headless | `apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` |
| Kaleidoscope | Newton / CUDA | 训练 headless | `apps/linkerbot_sim.kaleidoscope.newton.python.kit` |
| Kaleidoscope | PhysX / CUDA | 显式单环境 viewport | `apps/linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit` |
| Kaleidoscope | Newton / CUDA | 显式单环境 viewport | `apps/linkerbot_sim.kaleidoscope.newton_viewport.python.kit` |

调用方启动 Mirror 或 Kaleidoscope 产品入口，不直接拼装 Kit；factory 根据已校验的 physics 与 render
规格作唯一选择。
公开 selector 显式包含合法 execution：Mirror 为 `physx_cpu/physx_cpu_hybrid/newton_cpu/newton_cuda`，
Kaleidoscope 为 `physx_cuda/newton_cuda`。
两个产品根分别引用带产品命名空间的 scene selector `mirror/scene3` 与
`kaleidoscope/tblock_push`。

## 环境

- Linux x86-64、Python 3.12
- Isaac Sim 6.0.1、PyTorch 2.11 cu128
- Kaleidoscope 和 cuRobo action 需要兼容 CUDA GPU

创建仿真环境：

```bash
uv sync --extra simulation --extra visualization --extra training
```

CPU 开发检查必须使用独立环境，不能把 PyPI `usd-core` 与 Kit 提供的 `pxr` 混装：

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev --extra visualization
```

启动 Isaac 前由部署者显式接受 EULA：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

本仓库是 workspace application，不构建 wheel；命令均从 checkout 根目录执行。

## 校验 canonical 配置

以下命令只解析严格配置图，不启动 Isaac：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile physx_cpu_hybrid
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile newton_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode kaleidoscope --profile newton_cuda
```

Mirror 四个物理 profile 统一引用 `configs/control/mirror.yaml`，默认 PhysX/Newton controller bundle 由
`physics.engine` 派生。Kaleidoscope 完全没有 control slot：根只引用 `scene/physics/task`，EE/直线 action
才额外引用可选 `curobo`。Mirror 日志唯一入口是 `outputs.logging`。cuRobo 后端固定使用已验证的 0.8.0
task bundle 与 float32 dtype，YAML 只保留真实可调的数值容量。

## 快速运行

Mirror 交互进程：

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py --profile physx_cpu
```

Kaleidoscope 原生 Torch 环境：

```python
from linkerbot_sim.kaleidoscope import make_torch_env

env = make_torch_env(profile="physx_cuda", num_envs=256)
try:
    observations, info = env.reset()
    # actions 必须已经是 env.device 上的 float32 CUDA tensor。
finally:
    env.close()
```

native/debug `step` 会同步读取一次 done scalar，让未 reset 的 terminal row 在 physics 推进前被拒绝；
skrl SAME_STEP 路径不执行该 guard。Gymnasium 是显式 NumPy 边界；高吞吐训练应使用原生 Torch port 或
`training.skrl` adapter。
将 `profile` 改为 `"newton_cuda"` 即可选择 Newton。工厂分别加载
`apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` 与
`apps/linkerbot_sim.kaleidoscope.newton.python.kit`；后者使用项目 Python wheel/runtime
直接拥有 Newton worlds，不加载 Isaac Newton extension。

两个真实物理 smoke 使用同一入口：

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py --profile physx_cuda --num-envs 2 --steps 2
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py --profile newton_cuda --num-envs 2 --steps 2 --exercise-training-adapters
```

`just smoke-kaleidoscope` 还会用正式 Newton Kit 重复验证 batch IK 与同步直线 action；完整命令见
[Kaleidoscope 快速入门](docs/zh-CN/getting-started/kaleidoscope-quickstart.md)。

显式查看任一后端时运行：

```bash
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py --profile physx_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py --profile newton_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
```

viewer 单独读取 `configs/visualization/kaleidoscope.yaml`；该冷配置不进入 episode
snapshot/clone fingerprint。训练 step 始终使用 `render=False`，只有显式 `env.render()` 更新 viewport。

## 文档

- [项目概览](docs/zh-CN/getting-started/project-overview.md)
- [选择 Mirror 或 Kaleidoscope](docs/zh-CN/getting-started/choose-runtime-and-api.md)
- [Mirror 快速入门](docs/zh-CN/getting-started/mirror-quickstart.md)
- [Kaleidoscope 快速入门](docs/zh-CN/getting-started/kaleidoscope-quickstart.md)
- [Mirror CLI](docs/zh-CN/reference/mirror-cli.md)
- [Mirror JSON](docs/zh-CN/reference/mirror-json.md)
- [Kaleidoscope API](docs/zh-CN/reference/kaleidoscope-api.md)
- [配置参考](docs/zh-CN/reference/configuration.md)
- [状态、快照与克隆](docs/zh-CN/reference/snapshots.md)
- [完整中文文档索引](docs/zh-CN/index.md)

## 质量门禁

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev just quality
```

仿真与 GPU 门禁单独运行：

```bash
just test-simulation
```

该聚合门禁包含 `smoke-runtime-kits`（七个正式 Kit closure）、`smoke-mirror`（四个 Mirror mode
profile）、Kaleidoscope 双后端/动作 smoke、Newton 256-world 容量和 PhysX 进程显存预算。

## 许可

本项目基于 [MIT License](LICENSE) 发布，© Linkerbot (Beijing) Technology Co., Ltd.。
第三方软件与素材许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
