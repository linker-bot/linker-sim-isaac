# 安装与环境准备

语言：[中文](installation.md) | [English](../../en/getting-started/installation.md)

本项目是面向 Linux x86_64 的 checkout application，不构建可安装 wheel。命令、配置 profile、
资产和 Kit experience 都以仓库根目录为解析基准。

## 1. 前置条件

创建项目环境前，需要准备：

- Git；
- [uv](https://docs.astral.sh/uv/)；
- 使用 Kaleidoscope、Newton CUDA、RTX 渲染或 cuRobo 时，与锁定 Isaac Sim/CUDA 组合兼容的
  NVIDIA 驱动和 GPU；
- 足以容纳 Isaac Sim wheel 与 extension cache 的本地空间。

仓库当前锁定 Python 3.12、Isaac Sim 6.0.1、PyTorch 2.11/cu128、Warp 1.13.0 和
cuRobo 0.8.0。准确版本以 `pyproject.toml` 与 `uv.lock` 为准。

## 2. 获取工作区

```bash
git clone https://github.com/linker-bot/linker-sim-isaac.git
cd linker-sim-isaac
uv python install 3.12
```

后续维护命令都应从该 checkout 根目录执行。

## 3. 创建仿真环境

根据实际工作流选择 extra：

| Extra | 主要内容 | 典型用途 |
| --- | --- | --- |
| `simulation` | Isaac Sim、PyTorch CUDA、cuRobo、Warp、CUDA bindings | 两个产品的运行基础 |
| `visualization` | Foxglove SDK | Mirror 遥测可视化 |
| `training` | Gymnasium 与 skrl | Kaleidoscope adapter 与训练 |
| `test` | pytest 与 coverage | CPU 与仿真环境共用的测试工具 |
| `dev` | pytest、coverage、Ruff、PyPI USD | 仅用于 CPU 开发检查 |

完整运行环境：

```bash
uv sync --extra simulation --extra visualization --extra training
```

较精简的 Mirror 环境：

```bash
uv sync --extra simulation --extra visualization
```

Kaleidoscope 训练环境：

```bash
uv sync --extra simulation --extra training
```

不要使用 `--all-extras`，也不要混用 `dev` 与 `simulation`。`dev` 会安装 PyPI
`usd-core`，而 Isaac 必须从 Kit 加载 `pxr`。项目已经把两者声明为 uv conflict，使错误组合
直接失败，而不是静默污染运行环境。

## 4. 创建 CPU 开发环境

lint、纯 CPU 测试、架构检查和文档检查统一使用 `.venv-dev`：

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev   uv sync --extra dev --extra visualization
```

维护的 `just quality` 命令会使用该环境，不会改写仿真 `.venv`。

## 5. 接受 Isaac EULA

每个启动 Kit application 的进程都需要显式设置：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

设置前应阅读 NVIDIA 许可条款。本项目不分发 Isaac Sim 二进制文件。

## 6. 准备可选 Warehouse 素材

默认 Mirror 场景 `mirror/scene3` 使用项目自有包装层：

```text
assets/rigid_env_objects/industrial_warehouse_meters/industrial_warehouse_meters.usda
```

该包装层引用以下 NVIDIA 原始素材：

```text
usd-material/extracted/Industrial_NVD_10012/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd
```

NVIDIA 素材受其许可条款约束，已被 Git 排除，需要用户另行合法获取。请保持它的关联资产与纹理
目录结构，并放到上述精确路径。可使用下面的命令检查入口文件：

```bash
test -f usd-material/extracted/Industrial_NVD_10012/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd
```

配置校验只检查项目配置图，不会下载或授权外部素材。Kaleidoscope 不依赖该 Warehouse。如果无法
获得素材，应使用不引用 `industrial_warehouse` 的 Mirror scene profile，而不是把第三方内容提交
到本仓库。

## 7. 运行预检

检查锁定的 Python，并在使用 GPU profile 时检查 CUDA 可见性：

```bash
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)'
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

在不启动 Isaac 的情况下校验两个产品的配置图：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
```

校验器会输出解析后的源文件和确定性配置 fingerprint，但不会证明 GPU、外部素材或所有 Kit
extension 一定能启动。

## 8. 从最小运行检查开始

Mirror：

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_mirror_physics.py --profile physx_cpu --steps 8
```

Kaleidoscope PhysX CUDA：

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_kaleidoscope_physics.py \
  --profile physx_cuda --num-envs 2 --steps 2
```

完整 GPU/Isaac 验收矩阵使用 `just test-simulation`，它与 CPU `just quality` 门禁刻意分离。该 recipe
会自行加入兼容的 `test` extra，绝不加入 `dev` 或 PyPI `usd-core`。受信任 runner 合同见
[Simulation CI](../operations/simulation-ci.md)。

## 后续阅读

- [选择产品与接口](choose-runtime-and-api.md)
- [Mirror 快速入门](mirror-quickstart.md)
- [Kaleidoscope 快速入门](kaleidoscope-quickstart.md)
- [故障排查](../operations/troubleshooting.md)
