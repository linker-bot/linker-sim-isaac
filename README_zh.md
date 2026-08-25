# linker-sim-isaac

[![Quality](https://github.com/linker-bot/linker-sim-isaac/actions/workflows/quality.yml/badge.svg)](https://github.com/linker-bot/linker-sim-isaac/actions/workflows/quality.yml)

语言：[中文](README_zh.md) | [English](README.md)

快速入口：[安装与环境准备](docs/zh-CN/getting-started/installation.md) ·
[选择产品与接口](docs/zh-CN/getting-started/choose-runtime-and-api.md) ·
[完整文档](docs/zh-CN/index.md)

linker-sim-isaac 是一个面向机器人操作、现实回放以及 GPU 并行强化学习的 Isaac Sim
工作区。仓库以两种边界明确、契约刻意不同的产品模式对外暴露：

- **Mirror**：把一个真实工作站映射到一个仿真 World。它拥有交互控制、完整的运动规划与
  碰撞避障、相机、遥测以及 JSON transport。
- **Kaleidoscope**：通过 PhysX CUDA 或项目自有的 multi-world Newton runtime 运行大量同构的
  强化学习环境。两个后端都保留 headless GPU-native 的训练路径，并提供一个显式的单环境
  调试 viewport。该产品拥有 device-resident state、snapshot 与 clone、批量 IK、同步直线
  end-effector action、Gymnasium 集成以及 CUDA-native 的 skrl 路径；它不包含批量轨迹
  planner、避障服务、相机、SyntheticData、Replicator、录制、transport 或遥测。

应先选择产品，再选择物理后端和调用接口：

| 如果需要…… | 建议入口 |
| --- | --- |
| 映射一个真实工作站、交互控制、运动规划、相机或 JSON 接口 | **Mirror** |
| 运行大量同构 GPU 强化学习环境 | **Kaleidoscope** |
| 跨语言进程控制 | **Mirror JSON** |
| 保持训练数据在 CUDA | **Kaleidoscope 原生 Torch 或 skrl** |
| 接入 Gymnasium 生态 | **Kaleidoscope Gymnasium adapter** |

这些名称即为公开 API；对已废弃的 mode 名称或入口不提供任何兼容性契约。

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

请调用 Mirror 或 Kaleidoscope 的产品入口，而不是手动拼装 Kit；factory 会根据已校验的
physics 与 render 规格作出唯一选择。公开 selector 显式声明合法的 execution：Mirror 提供
`physx_cpu`、`physx_cpu_hybrid`、`newton_cpu` 与 `newton_cuda`；Kaleidoscope 提供 `physx_cuda`
与 `newton_cuda`。两个产品根分别引用带产品命名空间的 scene selector `mirror/scene3` 与
`kaleidoscope/tblock_push`。

## 环境要求

- Linux x86_64
- Python 3.12
- Isaac Sim 6.0.1
- PyTorch 2.11，配合 CUDA 12.8
- 规划或 end-effector action 需要 NVIDIA cuRobo 0.8.0
- Kaleidoscope 与 Newton 需要兼容的 NVIDIA GPU

关于仓库克隆、`uv`、依赖 extra、GPU 预检和可选 NVIDIA Warehouse 素材，请先阅读
[安装与环境准备](docs/zh-CN/getting-started/installation.md)。

从 checkout 根目录安装完整仿真工作区：

```bash
uv sync --extra simulation --extra visualization --extra training
```

CPU 开发环境必须独立，因为它的 `usd-core` 包不能遮蔽 Kit 的 `pxr` 模块：

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev --extra visualization
```

启动任一产品前先接受 Isaac EULA：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

本仓库是 workspace application，而非可安装的 wheel。请从 checkout 根目录、带
`PYTHONPATH=src` 运行命令。

> 默认 Mirror `scene3` 引用了本仓库不再分发的 NVIDIA Industrial Warehouse 原始素材。
> 项目自有的包装层与解析地面仍在仓库中，但在依赖仓库视觉效果前，需要按文档将获得许可的
> 素材放到指定路径。Kaleidoscope 不依赖该素材。

## 校验配置（不启动 Isaac）

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu_hybrid
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile newton_cuda
```

校验器会报告精确的源文件以及确定性的配置 fingerprint。Mode 根位于 `configs/modes/` 下；
被引用的 leaf profile 始终是其事实的唯一所有者。

所有 Mirror 物理 profile 统一引用 `configs/control/mirror.yaml`，默认的 PhysX 或 Newton
controller bundle 由 `physics.engine` 派生。Kaleidoscope 完全没有 control slot：它的根包含
`scene`、`physics` 与 `task`，只有在 end-effector 或直线 action 时才额外引用可选的 `curobo`。
Mirror 日志唯一在 `outputs.logging` 下配置。cuRobo 后端固定使用其已验证的 0.8.0 task bundle
与 float32 dtype；YAML profile 只保留真实的数值容量选择。

## 启动 Mirror

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/mirror.py --profile physx_cpu
```

Mirror 通过 stdin 接受严格的 `linkerbot.mirror.v1`、`v2` 与 `v3` JSON。可选的 TCP JSONL 与
WebSocket 监听器仅限 loopback，既不提供认证也不提供 TLS。参见
[Mirror 快速入门](docs/zh-CN/getting-started/mirror-quickstart.md)。

## 启动 Kaleidoscope

```python
import torch

from linkerbot_sim.kaleidoscope import make_torch_env

env = make_torch_env(profile="physx_cuda", num_envs=256)
observations, info = env.reset()
actions = torch.zeros(
    (env.num_envs, env.action_dim), device=env.device, dtype=torch.float32
)
observations, rewards, terminated, truncated, info = env.step(actions)
env.close()
```

原生接口返回 CUDA tensor。它的 debug `step` 会同步读取一次 done scalar，让未 reset 的
terminal row 在 physics 推进前被拒绝；skrl SAME_STEP 路径不执行该 guard。仅当需要 NumPy
边界时才使用 Gymnasium 适配器。参见
[Kaleidoscope 快速入门](docs/zh-CN/getting-started/kaleidoscope-quickstart.md)。

将 `profile` 设为 `"newton_cuda"` 即可选择 Newton。工厂会根据 profile 选择
`apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` 或
`apps/linkerbot_sim.kaleidoscope.newton.python.kit`。Newton experience 使用项目自有的 Python
runtime 作为 multi-world 物理所有者，不加载 Isaac Newton extension。

对两个 profile 运行真实物理 smoke：

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile physx_cuda --num-envs 2 --steps 2
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 2 \
  --exercise-training-adapters
```

`just smoke-kaleidoscope` 还会执行真实的 Newton batch IK 与同步直线 action。完整命令见
[Kaleidoscope 快速入门](docs/zh-CN/getting-started/kaleidoscope-quickstart.md)。

为任一后端启动显式 viewport：

```bash
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile physx_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile newton_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
```

viewer 单独读取 `configs/visualization/kaleidoscope.yaml`，与 task/physics 图相互独立，因此
显示选择不会改变 episode 的 snapshot/clone 兼容性。训练 step 仍使用 `render=False`，只有显式
的 `env.render()` 调用才更新 viewport。不添加任何相机、SyntheticData、Replicator 或录制管线。

## 文档

- [文档索引](docs/zh-CN/index.md)
- [安装与环境准备](docs/zh-CN/getting-started/installation.md)
- [项目概览](docs/zh-CN/getting-started/project-overview.md)
- [选择 mode 与 API](docs/zh-CN/getting-started/choose-runtime-and-api.md)
- [Mirror CLI](docs/zh-CN/reference/mirror-cli.md)
- [Mirror JSON](docs/zh-CN/reference/mirror-json.md)
- [Kaleidoscope API](docs/zh-CN/reference/kaleidoscope-api.md)
- [配置参考](docs/zh-CN/reference/configuration.md)
- [Python API](docs/zh-CN/reference/python-api.md)
- [故障排查](docs/zh-CN/operations/troubleshooting.md)
- [源码模块地图](docs/zh-CN/development/module-map.md)
- [贡献指南](CONTRIBUTING_zh.md)

## 质量门禁

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev just quality
```

在仿真环境中运行 Isaac、CUDA、真实物理、Newton 容量以及 PhysX 进程显存门禁：

```bash
just test-simulation
```

该聚合门禁包含针对全部七个正式 Kit closure 的 `smoke-runtime-kits`、针对全部四个 Mirror mode
profile 的 `smoke-mirror`、Kaleidoscope 两个后端及其动作变体、Newton 的 256-world 容量以及
PhysX 进程显存预算。
受信任 NVIDIA runner 的自动化方式、触发边界和配置要求见
[Simulation CI](docs/zh-CN/operations/simulation-ci.md)。

## 许可

本项目基于 [MIT License](LICENSE) 发布，© Linkerbot (Beijing) Technology Co., Ltd.。
第三方软件与素材许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
