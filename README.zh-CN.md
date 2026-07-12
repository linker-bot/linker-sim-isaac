# LinkerHand Simulation

语言：[中文](README.zh-CN.md) | [English](README.en.md)

LinkerHand Simulation 是一个基于 checkout 运行的 Isaac Sim 机器人操作工作区，覆盖多机器人
场景、Tiled Scene 模式的克隆环境、cuRobo 规划、轨迹、快照、遥测、相机和实验持久化输出。

项目包含两种 runtime 形态。`SingleSceneRuntime` 管理一个场景图，场景内可以配置任意数量机器人；
`TiledSceneRuntime` 管理批量克隆环境。两者在适合的领域共享配置和数据模型，但
Tiled Scene 不通过 `SingleSceneRuntime` 运行。

## 环境要求

- Linux x86_64
- Python 3.11
- Isaac Sim 5.1
- PyTorch 2.7
- cuRobo 操作需要 NVIDIA cuRobo 0.8.0 和兼容的 CUDA GPU

依赖由 `pyproject.toml` 声明、`uv.lock` 锁定：

```bash
uv sync --all-extras
```

所有 Isaac 入口都要求部署环境显式接受 EULA：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

本仓库是 workspace 应用，不是可安装 wheel。所有命令都应从 checkout 根目录以
`PYTHONPATH=src` 运行；runtime profile、脚本、资产和内置 task 资源都是应用组成部分。

## 校验配置

下面两条命令不会启动 Isaac，可先验证内置 Single Scene/Tiled Scene 配置依赖图：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_tiled_scene
```

## 运行

Single Scene runtime：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene --env scene1 --gui
```

Tiled Scene runtime：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene --env scene3_tiled --gui
```

两个进程都接受 stdin 严格 JSON，也可以开启仅限 loopback 的 TCP JSONL 和 WebSocket
listener。内置 listener 不提供认证或 TLS；远程访问必须使用终止在 loopback 上游的认证 TLS
代理或 SSH tunnel。

## 文档

- [项目概览](docs/zh-CN/getting-started/project-overview.md)
- [选择 Single Scene、Tiled Scene、JSON 或 Python](docs/zh-CN/getting-started/choose-runtime-and-api.md)
- [Single Scene 快速入门](docs/zh-CN/getting-started/single-scene-quickstart.md)
- [Tiled Scene 快速入门](docs/zh-CN/getting-started/tiled-scene-quickstart.md)
- [Single Scene CLI 参考](docs/zh-CN/reference/single-scene-cli.md)
- [Single Scene JSON 与 runtime 参考](docs/zh-CN/reference/single-scene-json.md)
- [Tiled Scene CLI 参考](docs/zh-CN/reference/tiled-scene-cli.md)
- [Tiled Scene JSON 与 runtime 参考](docs/zh-CN/reference/tiled-scene-json.md)
- [Python facade 参考](docs/zh-CN/reference/python-api.md)
- [配置指南](docs/zh-CN/guides/configuration.md)
- [运动规划与 cuRobo](docs/zh-CN/guides/motion-planning.md)
- [相机指南](docs/zh-CN/guides/cameras.md)
- [源码模块图](docs/zh-CN/development/module-map.md)
- [完整中文文档索引](docs/zh-CN/index.md)

## 质量检查

```bash
just quality
```

该命令覆盖格式、静态检查、文档链接、完整 CPU 测试、覆盖率和两套内置配置依赖图。
