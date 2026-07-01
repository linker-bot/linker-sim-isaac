# 配置 Profile 与 Runtime 初始化下沉说明

本文说明当前推荐的配置边界和代码职责划分。目标是让脚本只通过 scene/env 名称进入 runtime，
由 env 选择机器人实例、对象和世界设置，不再显式指定 `configs/...yaml` 路径，也不在脚本中处理
对象资产路径、摩擦力、solver iteration 或 USD 生成。

## 总目标

脚本入口使用稳定名称：

```bash
python scripts/pinch_grasp.py --env scene1
```

而不是：

```bash
python scripts/pinch_grasp.py \
  --env-config configs/envs/scene1.yaml
```

`linkerbot_sim` 内部负责：

- profile 名称到 YAML 文件的解析。
- env/object/robot/cumotion/logging 配置读取；robot profile 从 env `robots.single.robot_profile`
  或 `robots.dual.left/right.robot_profile` 得到。
- runtime 初始化，包括 Isaac session、rigid/runtime objects、robot、controller、logger。
- 已生成资产的引用和运行时物理覆盖。

脚本只负责动作语义，例如抓取点、手型、IK、轨迹和执行步骤。

## 三层配置边界

配置分成三层，不能互相混用。

### `configs/envs/scene*.yaml`

env 只描述当前世界、机器人实例和对象实例。单机器人 runtime 读取 `robots.single`，双机器人
runtime 读取 `robots.dual.left/right`。每个机器人实例只能有：

- `robot_profile`
- `root_pose`

对象实例只能有这些字段：

- `name`
- `object_profile`
- `runtime_handle`
- `root_pose`

示例：

```yaml
# scene1: 单左臂 AR5 + L6 抓取 capsule rope 端块的默认桌面场景。
env:
  name: scene1
  description: single left arm rope pinch grasp scene
  gravity_z: -9.81
  add_ground: false
  physics_frequency: 300.0
  render_frequency: 60.0

solver:
  type: TGS

robots:
  single:
    robot_profile: ar5v2_l6v1_l
    root_pose:
      xyz: [0.0, 0.09, 0.0]
      rpy: [-1.5707, 0.0, 0.0]
  dual:
    left:
      robot_profile: ar5v2_l6v1_l
      root_pose:
        xyz: [0.0, 0.09, 0.0]
        rpy: [-1.5707, 0.0, 0.0]
    right:
      robot_profile: ar5v2_l6v1_r
      root_pose:
        xyz: [0.0, -0.09, 0.0]
        rpy: [1.5707, 0.0, 0.0]

objects:
  - name: workstation_armbase
    object_profile: workstation_armbase
    root_pose:
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]

  - name: rope
    object_profile: capsule_rope
    runtime_handle: rope
    root_pose:
      xyz: [0.4, -0.55, 0.05]
      rpy: [0.0, 0.0, 0.0]
```

env 的 `objects[]` 不写这些字段：

- `kind`
- `source`
- `asset_path`
- `prim_path`
- `root_path`
- `import`
- `physics`

`root_pose` 属于 env，因为它描述“同一个 profile 在当前 scene 中放在哪里”。同一个
robot/object profile 可以在 `scene1`、`scene2` 中摆到不同位置。

env 文件使用编号命名：

```text
configs/envs/scene1.yaml
configs/envs/scene2.yaml
configs/envs/scene3.yaml
```

场景语义写在 YAML 顶部注释和 `env.description` 中，不写进文件名。脚本和外部命令只依赖
`scene1`、`scene2` 这些稳定编号。

### `configs/objects/*.yaml`

object profile 描述 `src/linkerbot_sim` 运行时需要读取的对象属性。刚性固定物体、刚性可动物体、
自身形状会动的动态链式对象都使用同一入口：

```yaml
object:
  name: workstation_armbase
  kind: rigid
  source: urdf
  asset_path: assets/static_env_objects/workstationV1_armbase/workstationV1_armbase.urdf
  prim_path: /World/WorkstationArmBase
  import:
    collision_approximation: convex_decomposition
  physics:
    static: true
```

动态链式对象也保持同样结构：

```yaml
object:
  name: capsuleropeV1_default
  kind: dynamic_chain
  source: usd
  asset_path: assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
  prim_path: /World/CapsuleRope
  root_path: /CapsuleRope
  physics:
    material:
      static_friction: 0.7
      dynamic_friction: 0.5
      restitution: 0.0
      friction_combine_mode: average
    solver_position_iterations: 48
    solver_velocity_iterations: 4
```

这里的 `physics` 是运行时仿真属性，例如：

- 刚性物体是否固定：`physics.static`
- 接触摩擦、恢复系数、combine mode
- solver iteration 覆盖
- URDF 导入配置：`object.import`

object profile 不写 `root_pose`，也不写 capsule rope 的段数、长度、质量、阻尼、关节限制或视觉材质。

### `tools/assets/configs/*.yaml`

资产生成固有属性放在 tools 下。例如 capsule rope 的生成配置：

```yaml
object:
  name: capsuleropeV1_default
  asset_path: assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
  root_path: /CapsuleRope

rope:
  segments: 12
  length: 0.75
  radius: null
  center: [0.0, 0.0, 0.0]
  total_mass: 0.2
  shape: capsule
  endpoint_box_mass: 0.5
  endpoint_box_size: [0.04, 0.03, 0.1]
  endpoint_linear_damping: 0.015
  endpoint_angular_damping: 0.05
  segment_linear_damping: 0.1
  segment_angular_damping: 0.12
  bend_limit: 2.0943951023931953
  bend_stiffness: 0.1
  bend_damping: 0.1
  lock_twist: true
  twist_limit: null
  twist_stiffness: 0.1
  twist_damping: 0.05
  disable_adjacent_collisions: true
  endpoint_color: [0.12, 0.34, 0.95]
  rope_color: [0.78, 0.62, 0.22]
```

这些字段是资产固有性质，生成 USD 时决定。`src/linkerbot_sim` 运行时不读取它们，也不负责生成 USD。

## 是否所有对象都用 URDF

不建议强行把所有对象都转换成 URDF。统一点应该是 object profile 和 runtime object interface，
不是底层资产格式。

推荐策略：

- 刚性固定物体：`kind: rigid`，通常 `source: urdf` 或 `source: usd`。
- 刚性可动物体：`kind: rigid`，按需要使用 URDF articulation 或 USD。
- 自身形状会动的动态对象：`kind: dynamic_chain`，优先引用离线生成好的 USD。

capsule rope 这类对象需要 PhysX D6 joint、相邻碰撞过滤、每段 solver iteration、twist/bend
limit 和材质绑定。标准 URDF 很难完整表达这些信息，强行 URDF 会导致导入后仍要大量 USD/PhysX
patch，表面统一，实际更复杂。

## `src/linkerbot_sim/objects` 职责

`src/linkerbot_sim/objects` 是对象配置读取和运行时处理层，不是资产生成层。

它负责：

- 解析 `configs/objects/*.yaml`。
- 解析 env `objects[]` 中的 scene instance。
- 校验 scene instance 只包含 `name/object_profile/runtime_handle/root_pose`。
- 校验 object profile 包含 `kind/source/asset_path/prim_path`。
- 引用已有 USD/URDF 资产到 stage。
- 对已引用对象应用运行时物理覆盖。
- 返回 `RuntimeObjectHandle`，供动作脚本通过 `runtime.objects[...]` 读取。

它不负责：

- 生成 USD。
- 计算 capsule rope 段块位置。
- 在运行时改变 rope 拓扑。
- 保存 tools 资产生成参数。

USD 生成代码放在：

```text
tools/assets/capsule_rope_builder.py
tools/assets/configs/capsule_rope.yaml
scripts/build_capsule_rope_asset.py
```

对象层当前结构：

```text
src/linkerbot_sim/objects/
  config.py
  dynamic_chain/
    capsule_rope.py
```

`config.py` 保存通用对象配置解析；`dynamic_chain/capsule_rope.py` 保存 capsule rope 这种具体
dynamic-chain 对象的运行时实现。

`src/linkerbot_sim/objects/dynamic_chain/capsule_rope.py` 只保留运行时能力：

- `CapsuleRopeConfig.from_mapping(...)`
- `asset_file()` 和参数校验
- `add_capsule_rope_reference(...)`
- `apply_capsule_rope_runtime_physics(...)`

## Runtime 初始化下沉

通用单机器人 runtime 入口是：

```python
from linkerbot_sim.app.single_robot_runtime import create_single_robot_runtime

runtime = create_single_robot_runtime(
    env="scene1",
    cumotion_profile="default",
    logging_profile="default_logger",
    control_mode="position",
    gui=False,
    status_prefix="RUN_PINCH_GRASP",
)
```

`create_single_robot_runtime(...)` 负责：

- 读取 profile。
- 从 env `robots.single` 选择 robot profile 和 root pose。
- 合并 robot 和 cuMotion profile。
- 构造 `EnvRuntimeSettings`。
- 解析并导入 env 中声明的 runtime objects。
- 启动 Isaac session。
- 导入 robot。
- `world.reset()` 后设置 gravity。
- 创建 controller。
- 创建 logger。
- 返回 `SingleRobotRuntime`。
- 异常或结束时关闭 logger/app。

它不负责：

- pinch TCP 计算。
- 手型目标。
- IK 或 motion planning。
- grasp/lift/wiggle 动作流程。

## `pinch_grasp.py` 变化

`pinch_grasp.py` 会明显变短。它不再需要：

- `--robot-config`
- `--robot`
- `--env-config`
- `--rope`
- 读取 `configs/...yaml`
- 导入 rigid runtime objects
- 导入 capsule rope
- 创建 Isaac session
- 创建 controller/logger
- 读取 rope 段块或端块位置

它保留：

- `--env`
- `--grasp-world`
- 动作参数
- 手型和运动序列

当前 pinch grasp 不考虑 rope 段块位置；抓取点由 `--grasp-world X Y Z` 给出，默认值在脚本中作为动作参数。

## 新增或修改场景

新增 `scene2` 时：

1. 新建 `configs/envs/scene2.yaml`。
2. 在顶部注释说明场景内容。
3. `env.name` 写 `scene2`。
4. `robots.single` 或 `robots.dual.left/right` 每项只写 `robot_profile/root_pose`。
5. `objects[]` 每项只写 `name/object_profile/runtime_handle/root_pose`。
6. 不在 env 的 scene instance 中写 `asset_path`、`prim_path`、`kind`、`source`、`physics` 或 `import`。

新增对象 profile 时：

1. 新建 `configs/objects/<profile>.yaml`。
2. 在 `object:` 中声明 `kind/source/asset_path/prim_path`。
3. 运行时物理属性写在 `object.physics`。
4. 导入参数写在 `object.import`。
5. 不写 `root_pose`。
6. 不写资产生成固有属性。

新增 capsule rope 资产变体时：

1. 在 `tools/assets/configs/` 新增生成配置。
2. 运行 `scripts/build_capsule_rope_asset.py` 生成 USD。
3. 在 `configs/objects/` 新增对应 runtime profile，指向生成后的 USD。
4. 在需要的 `scene*.yaml` 中通过 `object_profile` 引用它。

## 验证

建议改完后运行：

```bash
PYTHONPATH=src env_isaaclab/bin/python -m py_compile \
  scripts/pinch_grasp.py \
  scripts/dual_arm_motion_test.py \
  scripts/build_capsule_rope_asset.py \
  src/linkerbot_sim/configs/profiles.py \
  src/linkerbot_sim/app/runtime_objects.py \
  src/linkerbot_sim/app/single_robot_runtime.py \
  src/linkerbot_sim/execution/hold.py \
  src/linkerbot_sim/objects/rigid/runtime.py \
  src/linkerbot_sim/objects/rigid/__init__.py \
  src/linkerbot_sim/objects/config.py \
  src/linkerbot_sim/objects/dynamic_chain/capsule_rope.py \
  src/linkerbot_sim/objects/dynamic_chain/__init__.py \
  tools/assets/capsule_rope_builder.py
```

```bash
PYTHONPATH=src env_isaaclab/bin/python -m pytest \
  tests/test_rotations.py \
  tests/test_system_configs.py \
  tests/test_dual_arm_motion_test.py \
  tests/test_controller_configs.py \
  -q
```

边界搜索：

```bash
rg -n "write_capsule_rope_asset|create_rope_model" src/linkerbot_sim
rg -n "asset_path:|prim_path:|kind:|source:|physics:|import:" configs/envs
rg -n -e "--robot" scripts src tests README.md
```

第一条不应该有结果。第二条不应该在 env scene instance 中命中对象属性；`root_pose` 是 env 中唯一的摆放配置。
第三条不应该命中脚本入口参数。
