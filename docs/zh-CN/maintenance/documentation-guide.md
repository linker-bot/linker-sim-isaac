# 文档维护指南

语言：[中文](documentation-guide.md) | [English](../../en/maintenance/documentation-guide.md)

## 信息架构

- `getting-started/`：产品选择、概览和最短可运行流程；
- `guides/`：按任务解释决策与工作流；
- `reference/`：精确 CLI、wire、Python、配置、状态和输出合同；
- `operations/`：部署约束与故障处理；
- `development/`：源码 owner、命名和资产维护。

文档只使用 **Mirror** 与 **Kaleidoscope** 产品名。破坏性迁移历史只放在 design plan 或明确标记的
Mirror migration record，不在用户 quickstart 中建立旧入口跳转页。

## 事实 owner

| 事实 | 源码/配置 owner | 文档 owner |
| --- | --- | --- |
| Mirror CLI options | `mirror.cli` | `reference/mirror-cli.md` |
| Mirror operation/envelope | `mirror.interface.protocol` | `reference/mirror-json.md` |
| Kaleidoscope Torch/Gymnasium/state | `kaleidoscope` facade/ports | `reference/kaleidoscope-api.md` |
| Mode/profile schema | `configuration` + `configs/` | `reference/configuration.md` |
| Scene/episode snapshot | product state owner | `reference/snapshots.md` |
| Camera/planning/telemetry 归属 | Mirror composition | 对应 guide |
| 模块、layer、runtime、facade | `architecture/module_disposition.yaml` | generated module map |

不要在多个页面复制完整字段表。Guide 链接 reference，reference 再链接源码 owner。

## Module map 与 inventory

源码移动后运行：

```bash
just update-architecture
```

脚本同时生成中英文 module map、hardcoded candidate count/hash 和 v2 file inventory。不要手改 marker
内表格。发布前 `just test-architecture` 要求 facade 已冻结、7 个 Kit 精确匹配、旧产品名与 shim
归零、所有 count/hash 当前。

## 文档变更检查

任何 CLI/wire/config/facade 变更至少同步：

1. 对应 reference；
2. 受影响 quickstart/guide；
3. 双语 index 与 README（若入口变化）；
4. module map generator policy（若 owner/layer/facade 变化）。

运行：

```bash
just check-docs
just test-architecture
```

本地链接必须存在并受 Git 跟踪；双语 module map 的事实列必须完全一致。示例命令必须使用当前
`scripts/mirror.py`、`scripts/validate_mode_config.py` 或 public Python facade。
