# Configs 配置和使用说明

本文说明 `configs/` 目录下各类 YAML profile 的职责、引用关系和常用运行方式。完整字段模板以各目录中的 `example.yaml` 为准；本文重点说明怎么组合和使用这些配置。

## 总体规则

`configs/` 里的文件按 profile 名称使用。脚本参数通常传文件名 stem，而不是 YAML 路径：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --env scene1 \
  --cumotion-profile default \
  --logging-profile default_logger
```

上面的参数会读取：

```text
configs/envs/scene1.yaml
configs/cumotion/default.yaml
configs/logging/default_logger.yaml
```

profile 名称必须是简单文件名，不包含 `/` 或 `\`。配置中的相对路径，例如 `assets/...` 和 `logs/...`，都按仓库根目录解析，不依赖当前 shell 的工作目录。

## 目录职责

| 目录 | 用途 | 选择方式 |
| --- | --- | --- |
| `configs/envs/` | scene profile：世界频率、重力、solver、灯光相机、机器人实例和对象实例摆放 | 脚本参数 `--env <name>` |
| `configs/robots/` | robot profile：单个 Isaac articulation 的资产、物理覆盖、cuMotion 模型资源 | 由 env 中的 `robot_profile` 引用 |
| `configs/objects/` | object profile：环境对象的资产路径、导入方式和运行时物理属性 | 由 env 中的 `object_profile` 引用 |
| `configs/controllers/` | arm/hand 控制模式、增益、限幅和 mimic follower drive | runtime 固定读取 `arm_controller.yaml` 和 `hand_controller.yaml` |
| `configs/cumotion/` | cuMotion 算法 profile：IK 容差、planner pipeline、trajectory generation 等 | 脚本参数 `--cumotion-profile <name>` |
| `configs/logging/` | 关节跟踪 CSV 日志 profile | 脚本参数 `--logging-profile <name>` |

## Env Profile

env profile 描述“这个场景怎么摆”。它不直接写机器人资产路径和对象资产路径，而是引用 robot/object profile。

典型结构：

```yaml
env:
  name: scene1
  gravity_z: -9.81
  add_ground: false
  physics_frequency: 240.0
  render_frequency: 60.0

solver:
  type: PGS

robots:
  single:
    robot_profile: ar5v2_l6v1_l
    root_pose:
      xyz: [0.0, 0.09, 0.0]
      rpy: [-1.5707, 0.0, 0.0]

objects:
  - name: workstation_armbase
    object_profile: workstation_armbase
    root_pose:
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
```

关键边界：

- `root_pose` 属于 env，因为同一个 robot/object profile 可以在不同场景放到不同位置。
- `robots.single` 给单臂脚本使用；`robots.dual.left/right` 给双臂脚本使用。二者可以共存在同一 env 文件里。
- `objects[]` 只写实例名、`object_profile`、可选 `runtime_handle` 和 `root_pose`。
- object 的 `kind/source/asset_path/import/physics` 不写在 env，放在 `configs/objects/`。

## Robot Profile

robot profile 描述一个 Isaac articulation，以及 cuMotion 规划层需要的机器人模型资源。

典型结构：

```yaml
robot:
  name: ar5v2_l6v1_l
  asset_type: mjcf
  asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
  prim_path: /World/AR5V2_L6V1_L
  import:
    collision_approximation: convex_decomposition
    self_collision: false
  physics:
    gravity:
      default: false
      arm: false
      hand: false

cumotion:
  xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
  flange_frame: AR5V2_L_arm_flan_link
```

关键边界：

- 一份 robot profile 只描述一个 articulation。双臂不是写成一个 robot profile，而是在 env 的 `robots.dual.left/right` 中引用两份单侧 robot profile。
- `robot.asset_type` 当前支持 `mjcf` 和 `urdf`。
- `robot.root_pose` 不允许写在这里，必须写在 env 的机器人实例下。
- `robot.import.collision_approximation` 只影响 Isaac importer 生成碰撞几何。
- `robot.physics.physx` 和 `robot.physics.solver` 是机器人导入后的 USD/PhysX 覆盖，不属于 controller 配置。
- `cumotion` 段放 XRDF/URDF/frame 这类模型资源；IK/planner 算法参数放在 `configs/cumotion/`。

## Object Profile

object profile 描述一个可复用的环境对象。env 只负责实例化和摆放它。

rigid URDF 示例：

```yaml
object:
  name: workstation_armbase
  kind: rigid
  source: urdf
  asset_path: assets/rigid_env_objects/workstationV1_armbase/workstationV1_armbase.urdf
  prim_path: /World/WorkstationArmBase
  import:
    collision_approximation: convex_decomposition
  physics:
    static: true
```

USD dynamic chain 示例：

```yaml
object:
  name: capsuleropeV1_default
  kind: dynamic_chain
  source: usd
  asset_path: assets/flexible_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
  prim_path: /World/CapsuleRope
  root_path: /CapsuleRope
```

当前环境对象支持：

- `kind: rigid`，`source: usd` 或 `source: urdf`。
- `kind: dynamic_chain`，当前运行时用于 capsule rope，走 `source: usd`。

环境对象当前不支持 `source: mjcf`。MJCF 支持在 robot profile 的 `robot.asset_type` 中。

## Controller 配置

controller 配置固定从 `configs/controllers/arm_controller.yaml` 和 `configs/controllers/hand_controller.yaml` 读取，目前没有 `--controller-profile` 参数。

脚本通过 `--control-mode` 选择读取哪个控制段：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --control-mode position
```

支持的 mode：

- `position`：读取 `position_control`，method 可为 `implicit` 或 `explicit`。
- `velocity`：读取 `velocity_control`，method 可为 `implicit` 或 `explicit`。
- `effort`：读取 `effort_control`，method 当前为 `direct`。

每个控制段中：

- `active_joints` 面向主动命令关节。
- `follower_joints` 面向 MJCF equality/mimic follower，运行时会按 master 状态刷新 follower 目标。
- 机器人接触材质和刚体阻尼不要放在 controller YAML，放到 robot profile 的 `robot.physics.physx`。

## cuMotion Profile

cuMotion profile 只放算法参数，例如 IK 容差、planner pipeline、graph search、trajectory generation 和 specified path 默认值。

使用方式：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --cumotion-profile default
```

合并边界：

```text
configs/cumotion/*.yaml  <  configs/robots/*.yaml  <  脚本动作参数
```

含义是：cuMotion profile 提供算法默认值，robot profile 提供具体机器人资源，动作脚本提供任务目标和阶段参数。

不要把这些字段放进 `configs/cumotion/`：

- `xrdf_path`
- `urdf_path`
- `flange_frame`
- 具体抓取点、阶段时长、手型目标

## Logging Profile

logging profile 控制关节跟踪 CSV 是否写入、输出路径、flush 周期、采样降频和列开关。

使用方式：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --logging-profile default_logger
```

常用命令行覆盖：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --log logs/joint_tracking/test.csv \
  --log-interval-steps 1 \
  --log-measured-effort
```

注意：

- `--log` 会覆盖 profile 中的 `joint_tracking_path`。
- `measured_effort` 和 `applied_effort` 读取成本更高，只有分析力矩时建议打开。
- `enabled: false` 会禁用 CSV 写入。

## 运行时加载链路

单臂 runtime：

```text
--env scene1
  -> configs/envs/scene1.yaml
  -> robots.single.robot_profile
  -> configs/robots/<robot_profile>.yaml

--cumotion-profile default
  -> configs/cumotion/default.yaml

--logging-profile default_logger
  -> configs/logging/default_logger.yaml

controllers
  -> configs/controllers/arm_controller.yaml
  -> configs/controllers/hand_controller.yaml
```

双臂 runtime：

```text
--env scene2
  -> configs/envs/scene2.yaml
  -> robots.dual.left.robot_profile
  -> robots.dual.right.robot_profile
  -> configs/robots/<left>.yaml
  -> configs/robots/<right>.yaml
```

env 中的 `objects[]` 在单臂和双臂 runtime 中都会加载：

```text
objects[].object_profile
  -> configs/objects/<object_profile>.yaml
```

## 常用操作

新增一个 scene：

1. 复制 `configs/envs/example.yaml` 为 `configs/envs/my_scene.yaml`。
2. 修改 `env.name`、世界频率、地面、solver 和 visuals。
3. 在 `robots.single` 或 `robots.dual.left/right` 中引用已有 robot profile。
4. 在 `objects[]` 中引用已有 object profile，并设置每个实例的 `root_pose`。
5. 运行 `scripts/dual_arm_motion_test.py --dry-run --env my_scene` 做配置链路检查；单臂场景可用 `scripts/pinch_grasp.py --env my_scene --no-grasp` 做导入检查。

新增一个机器人 profile：

1. 复制 `configs/robots/example.yaml` 为 `configs/robots/my_robot.yaml`。
2. 修改 `robot.name`、`robot.asset_type`、`robot.asset_path` 和 `robot.prim_path`。
3. 按机器人命名和物理需求调整 `robot.physics`。
4. 设置 `cumotion.xrdf_path`、`cumotion.urdf_path` 和 `cumotion.flange_frame`。
5. 在 env 的 `robot_profile` 中引用 `my_robot`。

新增一个环境对象 profile：

1. 复制 `configs/objects/example.yaml` 或现有相近对象 profile。
2. 选择 `kind` 和 `source`。
3. 设置 `asset_path`、`prim_path`，必要时设置 `root_path`。
4. rigid 对象可设置 `physics.static`、接触材质和 importer 碰撞近似。
5. 在 env 的 `objects[]` 中通过 `object_profile` 引用，并设置 `root_pose`。

调整控制参数：

1. 修改 `configs/controllers/arm_controller.yaml` 或 `configs/controllers/hand_controller.yaml`。
2. 根据运行命令的 `--control-mode` 调整对应 section。
3. 如果只想临时比较 position/velocity/effort 模式，优先改命令行的 `--control-mode`。

调整规划参数：

1. 复制 `configs/cumotion/default.yaml` 为新的 profile。
2. 修改 IK tolerance、planner pipeline、graph search 或 trajectory generation 参数。
3. 用 `--cumotion-profile <name>` 选择它。

## 配置检查命令

双臂 dry-run 不启动 Isaac，适合快速检查 env、左右 robot profile 和 cuMotion 资源组合：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py \
  --dry-run \
  --env scene2 \
  --cumotion-profile default
```

单臂导入检查会启动 Isaac、导入对象和机器人，但不执行抓取：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --env scene1 \
  --no-grasp
```

打开 GUI 检查 stage、碰撞体和 root pose：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py \
  --env scene1 \
  --gui \
  --hold \
  --no-grasp
```

## 常见错误

`Profile name must be a simple file stem`

: 脚本参数传了路径。把 `--env configs/envs/scene1.yaml` 改成 `--env scene1`。

`robot root_pose belongs under env robots`

: 把机器人安装位姿写进了 robot profile。移动到 `configs/envs/*.yaml` 的 `robots.single.root_pose` 或 `robots.dual.<side>.root_pose`。

`objects[index] contains scene-level unsupported keys`

: env 的 `objects[]` 里写了资产属性。env 只写实例摆放；资产属性移动到 `configs/objects/*.yaml`。

`object.source must be one of ['urdf', 'usd']`

: 环境对象不支持 MJCF。MJCF 资产用于机器人 `robot.asset_type: mjcf`。

`Controller profile 'arm' has mismatched target`

: `arm_controller.yaml` 或 `hand_controller.yaml` 中的 `target` 和文件角色不一致。

