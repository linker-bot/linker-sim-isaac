# 发布

语言：[中文](releases.md) | [English](../../en/operations/releases.md)

linker-sim-isaac 通过 GitHub Releases 发布带版本的源码 workspace 归档。项目不发布 wheel、
sdist、container image 或 PyPI distribution；runtime profile、Kit experience、配置、脚本和素材
必须保留在同一个 checkout 中。

## 一次性仓库设置

创建名为 `release` 的 GitHub environment，将 deployment 权限限制给项目维护者；仓库计划支持时
配置 required reviewer。工作流只有 `actions: read` 和 `contents: write` 权限，checkout 不保留
credential。

## 准备版本

1. 将 `pyproject.toml` 的 `project.version` 与 `linkerbot_sim.__version__` 更新为相同的语义版本。
2. 把用户可见条目从 `[Unreleased]` 移入 `CHANGELOG.md` 和 `CHANGELOG_zh.md` 中版本完全一致、
   带日期的 section。
3. `just quality` 通过后合并版本变更。
4. 在合并后的 commit 上创建并推送 annotated tag：

   ```bash
   git tag -a v0.3.0 -m "linker-sim-isaac 0.3.0"
   git push origin v0.3.0
   ```

不得移动或替换已发布 tag。已发布缺陷应通过新的 patch 版本修复。

## 产生验收证据

进入 **Actions → Simulation → Run workflow**，选择 tag 或指向同一 commit、已经审查的仓库内
分支。等待完整矩阵成功，并从 URL 中记录数字 run ID。失败、取消、跳过或超时的运行不能作为
发布证据。

在 runner 稳定性问题解决前，GPU 工作流暂时只允许手动触发。因此 Release 工作流强制要求一个
显式成功、且 `headSha` 与 annotated tag commit 完全一致的 Simulation run，不能复用其他修订的
结果。

## 发布

进入 **Actions → Release → Run workflow**，填写 annotated tag、Simulation run ID 和 prerelease
选项。工作流随后：

1. checkout 精确 tag，并校验它是 annotated tag 且与两处版本声明一致；
2. 校验 changelog 条目和同一 commit 上成功的 Simulation 证据；
3. 在锁定依赖下运行 CPU `just quality` 门禁；
4. 用 `git archive` 创建 `linker-sim-isaac-VERSION.tar.gz` 和 `SHA256SUMS`；
5. 只为已经存在的 tag 创建 GitHub Release。

工作流不会创建缺失的 tag。校验或上传失败时，应保持 tag 不变，在经过审查的变更中修正发布
工具后重跑。如果错误来自已打 tag 的源码，则发布新的 patch 版本，不要重定向 tag。

下载后执行以下命令校验归档：

```bash
sha256sum --check SHA256SUMS
```
