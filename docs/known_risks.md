# 潜在隐患记录

## 1. URDF 静态环境物体不要叠加 kinematic/static 覆盖

### 背景

`configs/envs/*.yaml` 可以通过 `objects[]` 添加环境物体。对于 URDF 环境物体，
`physics.static: true` 通过 URDF importer 的 `fix_base=True` 实现。Importer
可能会创建一个固定根关节，例如：

```text
/World/WorkstationArmBase/root_joint
```

对于 USD/reference 环境物体，`physics.static: true` 通过把刚体设为 kinematic 并关闭
重力实现，因为这类资产没有 URDF importer 生成的 root joint。

### 风险

不要对同一个导入后的 URDF 环境物体同时使用 fixed-base 导入和 kinematic/static 刚体覆盖。
PhysX 可能会拒绝创建 importer 生成的 root joint，因为 joint 两端在物理上都相当于是
static body：

```text
PhysicsUSD: CreateJoint - cannot create a joint between static bodies
```

一次实际运行中，这个错误之后还出现了：

```text
malloc(): mismatching next->prev_size (unsorted)
```

这个 malloc 崩溃更像是无效 PhysX joint 设置之后的下游失败，不是根因。

### 当前保护

`src/linkerbot_sim/objects/rigid/runtime.py` 会区分处理 URDF 静态 rigid object 和 USD 静态 rigid object：

- URDF + `physics.static: true`：使用 importer `fix_base=True`；不要再额外把导入的刚体标成
  kinematic/static。
- URDF + `physics.static: false`：使用 importer `fix_base=False`；物体保持动态。
- USD + `physics.static: true`：把刚体标成 kinematic，并关闭重力。

### 后续注意

如果以后重构静态物体处理逻辑，需要保持这两套固定机制分开。若新增资产类型自带 root joint
或 articulation-root 行为，在泛化应用 kinematic/static 覆盖之前，要先确认是否会再次形成
static-static joint 条件。

## 2. 机械臂和桌面合并为同一个 URDF 可能隐藏固定关节风险

### 背景

为了方便整体导入，有时会考虑把机械臂安装座、桌面、机械臂本体等合并成一个 URDF。
这种做法会把原本应分别建模的静态环境和机器人 articulation 放进同一棵 link/joint 树。

### 风险

如果合并后的 URDF 没有明确区分桌面/安装座的静态基座、机器人 base link，以及它们之间的
连接 joint，可能出现以下问题：

- base link 被 importer 当成固定根处理，导致机械臂整体不能按预期作为独立 articulation 控制。
- 桌面和机械臂之间的 joint 被误设为 fixed，运行时 root pose、fixed-base joint 或控制器初始化
  可能和实际安装位姿冲突。
- 一些资产在生成时没有把 base link 和相关 joint 设置成适合仿真的非固定连接，后续再通过代码
  叠加 root pose 或静态覆盖时，容易触发 PhysX joint/transform 异常。

### 当前建议

优先把桌面/工装等静态环境物体和机械臂机器人资产分开导入：

- 桌面、安装座作为 `objects[]` 中的环境物体管理。
- 机械臂/手作为独立机器人 articulation 管理。
- 机器人安装位姿通过 env scene 的 `robots.single.root_pose` 或
  `robots.dual.left/right.root_pose` 管理。

这种拆分可以降低 URDF 合并资产中 fixed joint、base link 和 importer 固定基座语义混在一起的风险。
