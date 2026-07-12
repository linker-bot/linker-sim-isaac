# USD 资产预览指南

语言：[中文](usd-preview.md) | [English](../../en/development/usd-preview.md)

本文说明如何预览仓库中的 `.usd` / `.usda` 资产。示例命令假设已经激活 Isaac Sim 环境，并且
`isaacsim` 命令在当前 `PATH` 中。默认流程是先启动 Isaac Sim GUI，然后通过 `File -> Open`
打开资产文件。

## 推荐流程：GUI 打开资产

从仓库根目录先启动 Isaac Sim：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
isaacsim isaacsim.exp.full
```

启动后在菜单栏选择 `File -> Open`，再选择要预览的 `.usd` / `.usda` 文件。

检查 T block 时打开运行时实际引用的资产：

```text
assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda
```

绳体资产：

```text
assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

如果资产打开后看不到物体，先在 Stage 面板选中根 prim，然后按 `F` 聚焦视图。

## 命令行启动 GUI

从仓库根目录启动 GUI：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
isaacsim isaacsim.exp.full
```

然后使用 `File -> Open`。`isaacsim` 的第一个非选项参数是 Isaac Sim/Kit experience
配置（`.kit`），不是 USD 文件；直接运行 `isaacsim <asset.usda>` 会把资产误当 experience。
仓库当前也没有 `open_stage.py` helper，因此 `--exec "open_stage.py ..."` 不是本项目可用命令。

## 从对象配置找资产

运行时对象 profile 在 `configs/objects/` 下。先看 `object.asset_path`：

```bash
sed -n '1,80p' configs/objects/TblockV1_default.yaml
sed -n '1,80p' configs/objects/capsule_rope.yaml
```

然后在 Isaac Sim GUI 中使用 `File -> Open` 选择该 `asset_path`。

## 预览时重点检查

- `Stage` 面板里的 default prim 是否是预期根节点，例如 `/TBlock` 或 `/CapsuleRope`。
- 几何尺寸、方向和局部原点是否符合生成配置。
- `PhysicsCollisionAPI`、`PhysicsRigidBodyAPI`、`MassAPI` 等 schema 是否在需要的 prim 上。
- 可视材质和颜色是否正确。
- 对链式对象，检查 joints 是否存在并连接到正确 body。

## 常见问题

普通命令行 Python 里可能无法直接导入 `pxr`：

```bash
python -c "from pxr import Usd"
```

这是因为 USD/PhysX schema 通常由 Isaac/Kit 启动后注册。预览和交互检查优先启动
Isaac Sim 后用 `File -> Open` 打开资产。
