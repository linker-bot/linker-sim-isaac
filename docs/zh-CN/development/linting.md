# Lint 与格式化策略

语言：[中文](linting.md) | [English](../../en/development/linting.md)

仓库使用 Ruff 执行 Python lint 与格式化门禁：

```bash
just lint
just format-check
```

两项命令都属于 `just quality`，并使用 `dev` 依赖组中固定的 Ruff 版本。

## 稳定的规则契约

`pyproject.toml` 显式选择 `E4`、`E7`、`E9` 和 `F` 规则，并将目标版本设为 Python
3.12。这样，仓库已接受的基线不依赖 Ruff 每个版本的默认值，依赖升级不能静默增删策略。

扩大规则集应作为独立变更处理：修复新增诊断，记录确有必要的局部例外，并保持
`just quality` 通过。不要用仓库级 ignore 隐藏整个新规则族。

格式化门禁只处理 Python 源码。Markdown 继续遵循
[文档维护](../maintenance/documentation-guide.md)中的双语目录与本地链接契约；Ruff 升级不能
附带改写 fenced code 示例。
