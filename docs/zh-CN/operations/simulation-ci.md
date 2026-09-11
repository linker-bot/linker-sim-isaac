# Simulation CI

语言：[中文](simulation-ci.md) | [English](../../en/operations/simulation-ci.md)

`Simulation` GitHub Actions 工作流在专用 NVIDIA 主机上运行仓库维护的 GPU/Isaac 验收矩阵。
它不替代 GitHub-hosted CPU `Quality` 工作流：两者使用互不兼容的 Python 环境，验证的合同也不同。

## 触发与信任边界

工作流只在维护者显式手动触发时运行。在 self-hosted runner 的稳定性问题解决前，`master`
push 自动执行已经暂时关闭。它没有 `push`、`pull_request` 或 `pull_request_target` 触发器。
本仓库公开，未经审查的 Pull Request 代码不得进入可能保留机器状态的 self-hosted runner。

需要合并前证据时，进入 **Actions → Simulation → Run workflow**，选择本仓库内已经审查的分支，
并把运行链接记录到 Pull Request。来自 fork 的代码必须先完成审查，再复制到受信任的仓库内分支。
发布验收必须选择 annotated tag 或同一 commit 上的仓库内分支；
[发布工作流](releases.md)会校验这个精确修订。

## Runner 合同

Runner 必须同时具有以下标签：

| 标签 | 合同 |
| --- | --- |
| `self-hosted` | 主机不属于 GitHub-hosted 基础设施，由项目自行维护。 |
| `linux` | Isaac 运行于 Linux。 |
| `x64` | 锁定的 wheel 依赖图面向 x86-64。 |
| `nvidia-gpu` | `nvidia-smi` 能识别兼容的 NVIDIA GPU 与驱动。 |
| `isaac-sim` | 主机专门承载本仓库的 Isaac 工作负载。 |

应使用只授权给本仓库的组织级 runner group；策略支持时，再限制为仅允许
`.github/workflows/simulation.yml`。优先使用临时或可重置主机。持久 package/Isaac cache 可以放在
Actions workspace 之外，但其中不得保存仓库凭据、无关 secret 或可变源码 checkout。

主机还需要为 Isaac wheel 和 extension cache 预留足够磁盘。如果测试需要可选的授权 Warehouse
素材，应在 Git 之外按[安装文档](../getting-started/installation.md)中的精确布局准备；工作流不会下载
或再分发这些内容。

## 仓库 Environment

创建名为 `simulation` 的 GitHub environment。审阅 NVIDIA 条款后，添加非 secret 环境变量：

```text
OMNI_KIT_ACCEPT_EULA=Y
```

变量缺失或不是上述值时，工作流会在安装依赖前失败。checkout step 没有仓库写权限，也不会保留
Git credential。

## 执行的门禁

工作流依次：

1. 检查 Linux x86-64、EULA 决策、`nvidia-smi` 和依赖锁；
2. 同步 `simulation`、`visualization`、`training` 与 `test`，不安装不兼容的 `dev` extra；
3. 检查锁定的 Python、pytest、coverage、Torch、CUDA 与 GPU stack；
4. 运行 `bash ci/simulation.sh`，与维护者使用同一入口。

Job 上限为四小时。Concurrency 不会强制取消正在运行的 job，因为在 native teardown 中途终止 Kit
可能污染持久 runner。失败或超时不能作为验收证据；应定位第一个失败 recipe，存在残留 native
进程时先重置 runner，再重新执行完整矩阵。
