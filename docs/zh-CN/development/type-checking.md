# 静态类型检查

语言：[中文](type-checking.md) | [English](../../en/development/type-checking.md)

仓库使用 Pyright 作为必须通过的 CPU 开发门禁。单独运行：

```bash
just type-check
```

`just quality` 会执行同一项检查。命令固定使用已锁定的 `dev` 环境和
`pyrightconfig.ci.json`，因此本地与 CI 都按相同的 Python 3.12 依赖集合解析类型。

## 配置与范围

两个 Pyright 配置各有职责：

- `pyrightconfig.json` 为编辑器保留 `src`、`scripts` 和 `tests` 的宽范围发现能力。
- `pyrightconfig.ci.json` 定义必须保持零诊断的基线，包括配置包、依赖与文档检查、纯模块
  覆盖率与架构 inventory 工具、mode 校验和 workspace build backend。

CI 配置采用 standard 检查，不包含全局 `ignore` 或被关闭的 `report*` 规则。唯一一处行级
例外是惰性 `linkerbot_sim.configuration` facade 动态计算的 `__all__`；架构测试会独立冻结
并校验这组公开导出。

## 扩大门禁

只有当模块或目录在 CPU `dev` 环境中达到零诊断后，才能将其加入
`pyrightconfig.ci.json`。应修复真实类型边界或缩窄类型，不要全局关闭诊断。只有运行时
契约确实需要、注释准确且另有独立测试保护时，才保留行级例外。

当前范围刻意避开由 Isaac Sim、Kit、CUDA 或其他仿真专用发行包拥有 import 和类型的模块。
扩展到这些区域前，需要兼容的运行时包或持续维护的 stub。静态分析不能替代
[Simulation CI](../operations/simulation-ci.md) 所述的仿真与 GPU 门禁。
