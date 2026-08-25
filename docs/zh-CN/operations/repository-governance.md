# 仓库治理

语言：[中文](repository-governance.md) | [English](../../en/operations/repository-governance.md)

CPU quality 工作流会在每个 Pull Request 上运行，但只有默认分支将其设为必检项后，工作流结果
才是合并约束。项目维护的策略位于 `.github/rulesets/master.json`，面向仓库默认分支并要求：

- 变更必须通过 Pull Request；
- 至少一名审查者批准，新 push 会使旧批准失效；
- 最新 push 必须由提交者以外的人员批准；
- 所有 review conversation 已解决；
- 严格执行始终存在的 `CPU quality` 检查；
- 禁止删除默认分支和 non-fast-forward push。

策略明确不设置 bypass actor。它不要求按路径触发的 dependency audit，因为无关变更不会创建
该检查；也不会把 Pull Request 代码发送到自托管 Simulation runner。相关变更仍需按
[Simulation CI](simulation-ci.md) 的要求，对已审查的仓库内分支执行手动门禁。

## 应用 Ruleset

合并 JSON 文件不会自动修改 GitHub 仓库设置。仓库管理员需要通过
**Settings → Rules → Rulesets** 创建一次，或使用具备仓库 Administration write 权限的
fine-grained token 执行：

```bash
gh api --method POST \
  repos/linker-bot/linker-sim-isaac/rulesets \
  --input .github/rulesets/master.json
```

不要重复执行 `POST`。策略变更时，先列出现有 ruleset，核对准确的数字 ID，再通过 `PATCH`
更新。更新前必须比较 GitHub 上的生效设置和已审查 JSON。必检 context 必须准确写为
`CPU quality`，与 `.github/workflows/quality.yml` 的 job 名称一致；来源必须限定为 GitHub
Actions app（`integration_id` 15368）。

## 检查漂移

应用 ruleset 后运行：

```bash
just check-repository-policy
```

`Repository Policy` 工作流每周运行，也支持手动触发。它以只读方式检查默认分支目标、review、
分支最新状态、禁止删除和禁止 force push，不会创建或更新仓库设置。

没有 ruleset write 权限的公共 metadata reader 无法看到 bypass actor。仓库 owner 或管理员
发生变化后，应在设置页面人工复核生效规则仍未配置 bypass actor。
