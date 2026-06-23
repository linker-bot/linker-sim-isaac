# 资产命名规范

本文档定义机器人、末端执行器、传感器、环境对象和组合系统的通用命名格式。目标是让文件名、目录名和配置中的资产标识具备可读性、可排序性和可扩展性。

## 总体原则

- 使用 ASCII 字符，避免空格、中文、括号和特殊符号。
- 使用小写或大写需在同一项目内保持一致；型号名可保留厂商原始大小写。
- 不在正式资产标识、URDF/MJCF/USD 实体名和配置关节名中使用连字符 `-`；Isaac importer 可能把 `-` 自动转换为 `_`，导致运行时 DOF 名与配置不一致。
- 设备系列和版本写成紧凑字段，例如 `AR5V2`、`L6V1`。
- 使用下划线 `_` 分隔不同组件或不同语义段。
- 文件名应与所在资产目录的主标识一致。
- 不把临时状态写进正式资产名，例如 `new`、`test`、`final`、`copy`。

## 推荐格式

单体系统：

```text
<device-family><device-version>_<side-or-variant>
```

组合系统：

```text
<component-a>_<component-b>[_<component-c>...]
```

环境对象：

```text
<object-family><object-version>_<variant>
```

示例：

```text
AR5V2_L
L6V1_L
AR5V2_L6V1_L
capsuleropeV1_default
depthcameraD435_front
```

## 字段说明

`device-family`  
设备系列或产品族，例如 `AR5`、`L6`、`D435`。

`device-version`  
硬件、模型或资产版本，例如 `V1`、`V2`、`R3`。

`side-or-variant`  
左右手、左右臂、安装方向或几何变体，例如 `L`、`R`、`front`、`short`。

`component`  
组合系统中的单个设备完整标识。组合时不要拆开组件内部字段。

## 分隔符规则

设备系列、型号和版本在组件内部保持紧凑，不额外使用分隔符：

```text
AR5V2
L6V1
cameraD435
```

下划线 `_` 表示左右侧、组件之间或语义段之间的分隔：

```text
AR5V2_L
AR5V2_L6V1_L
mobilebaseV1_AR5V2_L6V1_L
```

## 目录结构建议

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

单体资产目录示例：

```text
assets/single_system/arm/AR5V2_L/
  AR5V2_L.urdf
  AR5V2_L.xml
  AR5V2_L.xrdf
```

组合资产目录示例：

```text
assets/combined_system/AR5V2_L6V1_L/
  AR5V2_L6V1_L.urdf
  AR5V2_L6V1_L.xml
  AR5V2_L6V1_L.xrdf
```

mesh 目录示例：

```text
assets/mesh/arm/AR5V2_L/
assets/mesh/hand/L6V1_L/
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

派生文件在主名前追加用途后缀：

```text
AR5V2_L_lula.yaml
AR5V2_L_collision.usda
AR5V2_L_visual.usda
AR5V2_L6V1_L_calibrated.xml
```

mesh 文件优先使用对应 link/body 的规范名，便于从 URDF/MJCF 直接追踪到几何文件；如需保留 CAD 原始名，应在配置或注释中说明来源。

## 资产内部实体名

URDF、MJCF、XRDF 和 USD 内部的 `robot`、`model`、`link`、`joint`、`body`、`actuator`、`mesh` 等实体名应使用稳定前缀：

```text
<single-system-name>_<category>_<local-name>
```

其中 `category` 建议使用 `arm`、`hand`、`gripper`、`sensor`、`tool` 等类别词，`local-name` 使用设备内部的局部语义名。

示例：

```text
AR5V2_L_arm
AR5V2_L_arm_joint_1
AR5V2_L_arm_flan_link
L6V1_R_hand
L6V1_R_hand_thumb_metacarpals_base1
L6V1_R_hand_index_mcp_pitch
```

这类前缀让组合系统中的同名局部结构不会冲突，例如左手和右手都可以有 `thumb_tip`，但组合后分别是：

```text
L6V1_L_hand_thumb_tip
L6V1_R_hand_thumb_tip
```

实体名中不要使用连字符。这样配置里的关节名、资产描述里的关节名和 Isaac runtime 暴露的 DOF 名可以保持一致。

## 版本与变体

硬件版本写在组件内部：

```text
AR5V2
L6V1
```

资产修订版本可作为后缀：

```text
AR5V2_L_assetr1
AR5V2_L6V1_L_assetr2
```

如果版本会被代码读取，建议在配置文件中单独提供结构化字段，而不是只依赖文件名解析。

## 配置引用

配置文件中的路径应引用最终资产文件，而不是中间目录：

```yaml
robot:
  asset_type: mjcf
  asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
```

如果一个系统需要多个描述文件，保持同一目录、同一前缀：

```yaml
ik:
  robot_description: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
  base_urdf: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
```

## 何时使用额外转换

仅在目标环境要求全小写或全下划线时，在导出副本中做额外转换，例如 Python module 名、某些代码生成标识符或特定工具链限制：

```text
AR5_V2_L6_V1_L
```

文件目录、资产内部实体名和普通配置路径优先使用本项目统一格式：

```text
AR5V2_L6V1_L
```
