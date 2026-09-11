# 支持

语言：[中文](SUPPORT_zh.md) | [English](SUPPORT.md)

可复现的项目问题和缺陷应使用公开 tracker。请先检索已有 Issue，再选择对应模板：

- [缺陷报告](https://github.com/linker-bot/linker-sim-isaac/issues/new?template=bug_report.md)：
  报告可复现的错误行为。
- [功能建议](https://github.com/linker-bot/linker-sim-isaac/issues/new?template=feature_request.md)：
  建议公开能力或契约变更。
- [使用问题](https://github.com/linker-bot/linker-sim-isaac/issues/new?template=question.md)：
  询问文档覆盖的安装和使用问题。
- 漏洞请按[安全策略](SECURITY.md)私下报告，不要在公开 Issue 中披露。

报告中应包含 `python scripts/mirror.py --version`、`git rev-parse HEAD`、所选产品和 profile、
相关的 OS/Python/Isaac/GPU 版本、配置校验 fingerprint、最小复现步骤和完整错误文本。发布前必须
删除 credential、token、客户数据、内部 hostname 和私有文件路径。

公开 tracker 覆盖本仓库维护的源码、文档化配置和可复现 runtime 行为；它不承诺提供私有部署
运维、硬件采购、定制场景制作或未发布本地修改的支持。缺少所选模板要求的信息或无法复现的请求，
维护者可以将其关闭。
