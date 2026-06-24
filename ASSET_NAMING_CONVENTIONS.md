# 资产命名规范

本文档规定 `assets/`、配置文件和资产内部实体的命名方式。目标是让文件路径、URDF/MJCF/USD 实体名、配置关节名和 Isaac runtime DOF 名保持一致。

## 核心规则

- 只使用 ASCII 字符。
- 正式资产名、目录名、文件名、关节名、link/body 名和配置键不使用连字符 `-`。
- 设备系列和版本写成紧凑字段，例如 `AR5V2`、`L6V1`、`D435`。
- 使用下划线 `_` 分隔左右侧、组件、类别和语义段。
- 文件名应与所在资产目录的主标识一致。
- 不在正式资产名中使用 `new`、`test`、`final`、`copy` 等临时状态词。
- 如果确实需要导出给某个工具链的特殊命名，应生成导出副本，不反向污染主资产命名。

不使用连字符的原因：Isaac importer 可能把 `-` 自动转换成 `_`，这会导致配置中的关节名和 articulation 暴露的 DOF 名不一致。

## 单体系统

格式：

```text
<device-family><device-version>_<side-or-variant>
```

示例：

```text
AR5V2_L
AR5V2_R
L6V1_L
L6V1_R
```

字段说明：

- `device-family`：设备系列，例如 `AR5`、`L6`、`D435`。
- `device-version`：硬件或资产版本，例如 `V1`、`V2`。
- `side-or-variant`：左右侧、安装方向或几何变体，例如 `L`、`R`、`front`、`short`。

## 组合系统

格式：

```text
<component-a>_<component-b>[_<component-c>...]
```

组合时不要拆开单体组件内部字段。

示例：

```text
AR5V2_L6V1_L
mobilebaseV1_AR5V2_L6V1_L
```

如果组合系统包含左右侧信息，优先让每个组件自己携带侧别，而不是在末尾追加一个全局侧别。

## 环境对象

格式：

```text
<object-family><object-version>_<variant>
```

示例：

```text
capsuleropeV1_default
tableV1_lab
boxendpointV1_left
```

对象资产如果由脚本生成，配置文件中应显式写出输出路径，例如：

```yaml
object:
  name: capsuleropeV1_default
  asset_path: assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda
```

## 目录结构

推荐目录结构：

```text
assets/
  mesh/
    <category>/
      <single-system-name>/
  single_system/
    <category>/
      <single-system-name>/
  combined_system/
    <combined-system-name>/
  static_env_objects/
    <object-name>/
  dynamic_env_objects/
    <object-name>/
```

当前项目示例：

```text
assets/mesh/arm/AR5V2_L/
assets/mesh/hand/L6V1_L/
assets/single_system/arm/AR5V2_L/
assets/single_system/hand/L6V1_L/
assets/combined_system/AR5V2_L6V1_L/
assets/dynamic_env_objects/capsuleropeV1_default/
```

## 文件命名

主资产文件使用目录名作为文件名前缀：

```text
<asset-name>.urdf
<asset-name>.xml
<asset-name>.usd
<asset-name>.usda
<asset-name>.xrdf
```

示例：

```text
AR5V2_L.urdf
AR5V2_L.xml
AR5V2_L.xrdf
AR5V2_L6V1_L.xml
capsuleropeV1_default.usda
```

派生文件在主名前追加用途后缀：

```text
AR5V2_L_collision_model.yaml
AR5V2_L_collision.usda
AR5V2_L_visual.usda
AR5V2_L6V1_L_calibrated.xml
```

mesh 文件优先使用对应 link/body 的规范名：

```text
AR5V2_L_arm_link1.stl
L6V1_R_hand_thumb_metacarpals_base1.stl
```

如需保留 CAD 原始文件名，应放在独立来源目录或在配置/注释中说明，不作为主运行资产名。

## 资产内部实体名

URDF、MJCF、XRDF 和 USD 内部的 `robot`、`model`、`link`、`joint`、`body`、`actuator`、`mesh` 等实体名使用稳定前缀。

格式：

```text
<single-system-name>_<category>_<local-name>
```

常用 `category`：

- `arm`
- `hand`
- `gripper`
- `sensor`
- `tool`
- `base`

示例：

```text
AR5V2_L_arm
AR5V2_L_arm_base
AR5V2_L_arm_joint_1
AR5V2_L_arm_flan_link
L6V1_R_hand
L6V1_R_hand_base_link
L6V1_R_hand_thumb_metacarpals_base1
L6V1_R_hand_index_mcp_pitch
```

组合系统中禁止只使用局部名，例如 `thumb_tip`、`joint_1`。应保留完整前缀：

```text
L6V1_L_hand_thumb_tip
L6V1_R_hand_thumb_tip
AR5V2_L_arm_joint_1
AR5V2_R_arm_joint_1
```

## 关节命名

机械臂关节：

```text
<single-system-name>_arm_joint_<index>
```

示例：

```text
AR5V2_L_arm_joint_1
AR5V2_L_arm_joint_7
```

灵巧手关节：

```text
<single-system-name>_hand_<finger>_<joint-role>
```

示例：

```text
L6V1_L_hand_thumb_cmc_roll
L6V1_L_hand_thumb_cmc_pitch
L6V1_L_hand_index_mcp_pitch
L6V1_L_hand_index_dip
```

mimic/equality 名称也保留完整前缀：

```text
L6V1_L_hand_couple_index
L6V1_L_hand_couple_thumb
```

## 配置引用

配置文件应引用最终运行资产，而不是中间目录：

```yaml
robot:
  asset_type: mjcf
  asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
```

同一系统的多个描述文件保持同目录、同前缀：

```yaml
cumotion:
  xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
  flange_frame: AR5V2_L_arm_flan_link
```

关节组和轨迹目标必须使用资产内部真实关节名：

```yaml
controlled_joints:
  - AR5V2_L_arm_joint_1
  - L6V1_L_hand_thumb_cmc_roll
```

## 版本和修订

硬件版本写入组件名：

```text
AR5V2
L6V1
```

资产修订建议写成后缀：

```text
AR5V2_L_assetr1
AR5V2_L6V1_L_assetr2
```

如果版本需要被代码读取，应在 YAML 中单独提供结构化字段，不依赖文件名解析。

## 特殊导出

主资产命名保持本规范。如果某个工具链要求全小写、全下划线或其它限制，可以只对导出副本转换：

```text
ar5v2_l6v1_l
AR5_V2_L6_V1_L
```

转换后的名字不应回写到主资产目录、URDF/MJCF 实体名或默认配置。

## 提交前检查

改动资产命名后，至少检查：

- `assets/` 路径中没有旧命名残留。
- URDF/MJCF/XML 可解析。
- URDF/MJCF 中引用的 mesh 文件存在。
- 关节组、轨迹目标、TCP frame、IK 描述里的名称同步更新。
- `scripts/run_pinch_grasp.py --no-grasp --short-smoke` 能通过导入和 controller 初始化。

常用扫描命令：

```bash
rg "AR5-V2|L6-V1|capsule-rope" assets configs source scripts tests README.md ASSET_NAMING_CONVENTIONS.md
```
