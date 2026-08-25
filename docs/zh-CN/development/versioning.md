# 版本与修订标识

语言：[中文](versioning.md) | [English](../../en/development/versioning.md)

linker-sim-isaac 是 workspace application，不构建可安装 wheel，但运行记录和问题报告仍需要
稳定的兼容标识。以下命令不会启动 Isaac：

```bash
PYTHONPATH=src python scripts/mirror.py --version
PYTHONPATH=src python scripts/kaleidoscope_viewer.py --version
```

Python 集成可以读取 `linkerbot_sim.__version__`。开发 checkout 还必须附带
`git rev-parse HEAD`：兼容版本表示预期的 API/配置系列，commit 才能标识精确修订。

## 更新版本

权威 release 值声明在 `pyproject.toml`，并同步写入纯 Python、可安全导入的
`linkerbot_sim` facade。这是因为当前 workspace 明确不安装 distribution metadata。CPU
质量门禁要求两处完全一致。

版本发布应在同一个聚焦变更中更新这两个位置。不要在 import 时读取 checkout 来推导版本；
顶层 facade 必须继续保持无文件 I/O、无 Kit 启动、无 CUDA 初始化、无可选运行时依赖。
