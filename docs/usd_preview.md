# USD/USDA 预览指南

本文说明如何预览仓库中的 `.usd` / `.usda` 资产。示例命令假设已经激活 Isaac Sim 环境，并且
`isaacsim` 命令在当前 `PATH` 中。默认流程是先启动 Isaac Sim GUI，然后通过 `File -> Open`
打开资产文件。

## 推荐流程：GUI 打开资产

从仓库根目录先启动 Isaac Sim：

```bash
isaacsim isaacsim.exp.full
```

启动后在菜单栏选择 `File -> Open`，再选择要预览的 `.usd` / `.usda` 文件。

如果只是想看 T block 形状，推荐打开带相机和灯光的预览 stage：

```text
assets/rigid_env_objects/TblockV1_default/TblockV1_preview.usda
```

要检查实际运行时引用的 T block 资产，打开：

```text
assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda
```

绳体资产：

```text
assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

如果资产打开后看不到物体，先在 Stage 面板选中根 prim，然后按 `F` 聚焦视图。

## 可选：命令行打开资产

优先使用 GUI 打开。需要脚本化预览时，可以从仓库根目录运行：

```bash
isaacsim isaacsim.exp.full --exec "open_stage.py assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda"
```

注意：`isaacsim` 的第一个非选项参数是 Isaac Sim/Kit 的 experience
配置（`.kit`），不是 USD 文件。直接运行
`isaacsim <asset.usda>` 会把 `<asset.usda>` 当作 `.kit` 去查找，
并报 `Unable to find experience (.kit) file`。

`--exec` 后面的脚本和参数也要放在同一组引号里。否则 Kit 只会把 `open_stage.py`
传给脚本执行器，后面的路径不会进入脚本的 `argparse`。

## 从对象配置找资产

运行时对象 profile 在 `configs/objects/` 下。先看 `object.asset_path`：

```bash
sed -n '1,80p' configs/objects/TblockV1_default.yaml
sed -n '1,80p' configs/objects/capsule_rope.yaml
```

然后把 `asset_path` 传给 Isaac Sim：

在 GUI 里使用 `File -> Open` 选择该路径。需要命令行打开时，可使用
`isaacsim isaacsim.exp.full --exec "open_stage.py <asset_path>"`。

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
