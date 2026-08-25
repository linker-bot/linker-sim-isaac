# 依赖安全与更新

语言：[中文](dependency-security.md) | [English](../../en/operations/dependency-security.md)

`pyproject.toml` 声明受支持的依赖 profile，`uv.lock` 冻结完整的 Linux x86-64 依赖图。
依赖不能只因为版本更新就合并：Isaac、CUDA、Torch、Warp 与 cuRobo 构成一组兼容性闭包，
变更后仍须通过对应运行时验收。

## 自动更新策略

`.github/dependabot.yml` 每周检查 `uv` 与 GitHub Actions 两个 ecosystem。Dependabot 不会
自动合并，只会创建可审查的 Pull Request，并同步更新 `pyproject.toml`、`uv.lock`、Action
commit pin 以及同一行的版本注释。

Python 更新按审查边界分组：

| 分组 | 内容 | 必需验证 |
| --- | --- | --- |
| `simulation-runtime` | Isaac Sim、Torch、Warp、CUDA bindings、cuRobo、Newton 及相关兼容包 | CPU quality 与受信任的 Simulation CI |
| `development-tooling` | pytest、coverage、Ruff 与 CPU-only USD 工具 | CPU quality |
| `application-dependencies` | 其余基础与可选应用依赖 | CPU quality；如果包进入仿真 profile，再运行 Simulation CI |
| `workflow-actions` | 仓库工作流使用的 GitHub Actions | 工作流策略测试及受影响工作流 |

兜底的 application 分组必须位于最后，避免属于仿真或工具边界的包被吸收到宽泛更新中。

## Pull Request 增量门禁

`Dependency Audit` 工作流检查 `pyproject.toml` 与 `uv.lock` 的变更。它按精确 commit ID
分别 checkout Pull Request 的 base 与 head，然后从受信任的 base checkout 执行
`scripts/check_dependency_audit_delta.py`。head checkout 只提供待审计数据，不能在同一个
Pull Request 中替换比较策略。唯一例外是首次上线：如果 base 尚不含检查器，工作流会使用
已审查的候选副本并输出明确 notice。

检查器对两个 commit 的完整锁图分别执行 `uv audit --frozen`，不会安装或 import 依赖。
head 出现以下任一情况时门禁失败：

- 同一个包出现 base 中不存在的新漏洞标识或 alias；
- 新增 archived 等 adverse project status。

比较会合并 advisory alias，因此同一漏洞的 GHSA、CVE 和 PYSEC 记录只计一次。现有 finding
仍显示在审计数量中，但不会让所有无关依赖 Pull Request 永久失败。审计执行、网络、schema 或
锁文件错误都会 fail closed，不能被当成干净结果。

这是防退化门禁，不表示当前锁图不存在已知 finding。维护者仍须持续处置现有 advisory，
包括受模拟器版本约束的传递依赖。

## 审查更新

1. 阅读上游 release 与安全说明。GitHub Action 更新还要确认候选 commit 确实属于注释中的
   release tag。
2. 直接依赖声明与 `uv.lock` 必须放在同一个变更中。解析完成后运行 `uv lock --check`，
   不要手改锁文件。
3. 运行 `UV_PROJECT_ENVIRONMENT=.venv-dev just quality`。
4. `simulation-runtime` 或任何由 Kit/CUDA 加载的包都要运行受信任的
   [Simulation CI](simulation-ci.md)。Isaac Sim、Torch、torchvision、torchaudio、Warp、
   CUDA bindings 与 cuRobo 必须作为一个闭包测试。
5. 确认 Dependency Audit 没有新增 finding。总数下降是有效证据，但不能替代兼容性测试。

本地比较需要保留一个干净的 base checkout：

```bash
python scripts/check_dependency_audit_delta.py \
  --base-project ../linker-sim-isaac-base \
  --head-project .
```

需要完整处置列表时，可直接运行 `uv audit --frozen --python-version 3.12`；当前依赖图存在
任意 finding 时该命令会返回非零状态。

## 例外与局限

不要为了让 Pull Request 变绿而增加仓库级 advisory 忽略清单。如果上游约束导致暂时无法修复，
应在 Pull Request 中记录受影响 profile、可达性、缓解措施、上游 issue 和重新验证条件。
增量门禁会在处置期间继续阻止无关的新 finding。

审计完整性受包 metadata 与 advisory 服务覆盖范围限制，Git 依赖尤其可能没有 package index
层面的 advisory。cuRobo 源码因此继续锁定到完整不可变 commit；即使审计干净，commit 变化也
必须接受源码审查并运行 Simulation CI。
