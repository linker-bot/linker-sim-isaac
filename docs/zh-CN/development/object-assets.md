# 物体资产生成指南

语言：[中文](object-assets.md) | [English](../../en/development/object-assets.md)

本文说明如何用 `tools/object_assets/` 生成环境物体 USD/USDA 资产，并把生成结果接入
`configs/objects/` 和所属产品的 `configs/scenes/{mirror,kaleidoscope}/`。

## 构建器完整参数表

两个受支持的 `build_asset.py` 入口接受相同参数：

| 参数 | 默认值 | 契约 |
| --- | --- | --- |
| `--help` | 不适用 | 输出 argparse help，不启动 Kit，随后退出。 |
| `--config PATH` | 对应入口同目录的 `config.yaml` | 从指定路径读取资产生成 YAML。 |
| `--output PATH` | 省略 | 覆盖 `object.asset_path`；省略时写入生成配置声明的路径。 |

路径从 checkout 解析。Rope 构建成功时输出 `BUILD_CAPSULE_ROPE_ASSET_OK`，T block 构建成功时
输出 `BUILD_T_BLOCK_ASSET_OK`。两个命令都要求先接受 Kit EULA，并在资产写完后关闭 headless
`SimulationApp`。

## 基本边界

项目把“资产生成”和“仿真运行”分成四层：

| 层级 | 路径 | 职责 |
| --- | --- | --- |
| 生成配置 | `tools/object_assets/<rigid|flexible>/<name>/config.yaml` | 描述资产固有属性，例如几何、质量、阻尼、关节限制和可视颜色 |
| 生成入口 | `tools/object_assets/<rigid|flexible>/<name>/build_asset.py` | 启动 headless Isaac/Omni，写出 USD/PhysX schema |
| 运行时对象 | `configs/objects/*.yaml` | 引用已经生成好的 USD/URDF，并配置资产来源、导入参数、物理属性、规划碰撞和适用时的资产内部 `root_path` |
| 场景实例 | `configs/scenes/<mode>/<scene>.yaml` 的 `objects[]` | 选择 object profile，并设置每个场景里的 stage `prim_path` 和 `root_pose` |

运行脚本不会自动重新生成物体资产。修改 `tools/object_assets/.../config.yaml` 后，需要手动运行
对应的 `build_asset.py`。

## 运行环境

先激活包含 Isaac/Omni 扩展的 Python 环境，再从仓库根目录执行生成命令：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python tools/object_assets/flexible/rope/build_asset.py
PYTHONPATH=src .venv/bin/python tools/object_assets/rigid/tblock/build_asset.py
```

注意：

- 直接运行 `builder.py` 不会生成资产；它只是库模块。
- 使用 `build_asset.py`，因为写 USD/PhysX schema 需要先启动 Isaac/Omni 扩展。
- 生成脚本默认 headless，不会导入机器人，也不会执行 motion。

## Capsule Rope

生成配置：

```text
tools/object_assets/flexible/rope/config.yaml
```

默认输出：

```text
assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

常用命令：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python tools/object_assets/flexible/rope/build_asset.py \
  --config tools/object_assets/flexible/rope/config.yaml \
  --output assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

`rope` 段主要控制：

- `segments`、`length`、`radius`：绳段数量、总长度和半径。
- `shape`：中间段几何，当前可用 `capsule` 或 `cuboid`。
- `endpoint_box_*`：两端端块尺寸、质量和阻尼。
- `bend_*`、`twist_*`、`lock_twist`：D6 joint 角度限制和弹簧阻尼。
- `disable_adjacent_collisions`：是否关闭相邻段碰撞，通常保持 `true`。
- `endpoint_color`、`rope_color`：资产内置可视材质。

运行时引用通常在：

```text
configs/objects/capsule_rope.yaml
```

这里配置 `kind: dynamic_chain`、`source: usd`、`asset_path`、`root_path`、接触材质和 solver
iteration。`asset_path` 要指向生成出来的 `.usda` 文件；dynamic chain 的 `root_path` 要和
生成配置中的 `object.root_path` 一致。

运行时物理字段按后端分层：`object.physics.material` 只保存通用
`static_friction/dynamic_friction/restitution`；`friction_combine_mode` 位于
`object.physics.physx.material`；绳体的 `position_iterations/velocity_iterations` 位于
`object.physics.physx.solver`。Newton 只消费通用材质，PhysX leaf 不进入 Newton runtime。

## T Block

生成配置：

```text
tools/object_assets/rigid/tblock/config.yaml
```

默认输出：

```text
assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda
```

常用命令：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python tools/object_assets/rigid/tblock/build_asset.py \
  --config tools/object_assets/rigid/tblock/config.yaml \
  --output assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda
```

`tblock` 段主要控制：

- `center`：资产局部中心。
- `stem_size`、`stem_offset`：stem cuboid 的尺寸和局部中心偏移。
- `cap_size`、`cap_offset`：cap cuboid 的尺寸和局部中心偏移。
- `total_mass`：T block compound rigid body 的总质量。
- `linear_damping`、`angular_damping`：刚体阻尼。
- `color`：资产内置可视材质。

运行时引用通常在：

```text
configs/objects/TblockV1_default.yaml
```

刚体是否静态冻结、通用接触材质和 PhysX combine mode 属于运行时对象 profile，不写在生成配置里。T block
这类 rigid USD object 的 stage 路径和位姿分别写在 scene profile 的 `scene.objects[].prim_path` 与
`scene.objects[].root_pose`；实例名称来自 `scene.objects[].name`。当前 strict scene schema 要求显式
`prim_path`，不从已删除的旧 env 配置推导路径。

## 预览生成结果

推荐先启动 Isaac Sim GUI：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
isaacsim isaacsim.exp.full
```

再用 `File -> Open` 打开
`assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda`。仓库当前没有
`open_stage.py` helper，因此不要照搬带 `--exec open_stage.py ...` 的外部示例。更多检查项见
[USD 资产预览指南](usd-preview.md)。

预览时重点检查：

- 物体尺寸、方向和局部原点是否符合生成配置。
- `root_path` 是否存在，例如 `/TBlock` 或 `/CapsuleRope`。
- 刚体、collision、mass schema 是否存在。
- 可视材质是否能正常显示。

## 接入场景

生成资产后，按下面顺序接入运行时：

1. 确认 `tools/object_assets/.../config.yaml` 的 `object.asset_path` 指向目标资产路径。
2. 运行对应 `build_asset.py` 生成或刷新 `.usda`。
3. 在 `configs/objects/<profile>.yaml` 中引用同一个 `asset_path`；如果该运行时类型使用
   `root_path`，也要和生成配置保持一致。
4. 在 `configs/scenes/<mode>/<scene>.yaml` 的 `objects[]` 中引用 `object_profile`，并设置
   `root_pose`；mode root 使用 `<mode>/<scene>` selector 引用该文件，文件内 `scene.id` 只写
   `<scene>` basename。
5. 运行 dry-run 或仿真脚本验证配置链路。

示例 scene 片段：

```yaml
objects:
  - name: tblock
    object_profile: TblockV1_default
    prim_path: /World/TBlock
    root_pose:
      xyz: [0.0, -0.45, 0.12]
      rpy: [0.0, 0.0, 0.0]
```

## 验证命令

运行离线入口、配置和实例测试：

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_object_asset_entrypoints.py \
  tests/test_object_instances.py \
  tests/test_system_configs.py -q
```

不启动 Isaac，校验配置依赖图：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile physx_cpu
```

Mirror 场景交互导入检查：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/mirror.py --profile physx_cpu
```

## 常见问题

`builder.py` 跑完没有生成文件
: `builder.py` 是库模块；请运行同目录的 `build_asset.py`。

`ModuleNotFoundError: linkerbot_sim`
: 从仓库根目录运行，并设置 `PYTHONPATH=src`。

`pxr`、`PhysxSchema` 或 USD schema 相关错误
: 确认当前 `python` 来自 Isaac/Omni 环境，再运行 `build_asset.py`。这些 schema 依赖 Isaac/Omni
  扩展加载。

资产生成了，但仿真中找不到 prim
: 检查 `configs/objects/*.yaml` 的 `asset_path` 和 scene `objects[].prim_path`；如果该对象 profile
  使用 `root_path`，也要确认它和生成配置中的 `object.root_path` 一致。

修改了生成配置但仿真没有变化
: 重新运行 `build_asset.py`，并确认输出路径就是运行时 object profile 引用的 `asset_path`。
