# 双臂双手协作第一版修改建议：融合规划模型 + 按阶段选择 TCP

本文记录当前工程从单手单臂扩展到双手双臂协作的第一版落地方案。根据前面的讨论，第一版不直接做双 TCP 同时约束，也不强制把 Isaac 仿真 articulation 立即融合成一个机器人；优先实现一个更稳的中间形态：

```text
cuMotion 规划模型：由双臂配置临时生成融合 URDF，XRDF 提供 14-DOF C-space
TCP 模型：同一个 cuMotion context 中同时存在 left_pinch_tcp / right_pinch_tcp
动作策略：每次规划/运动时显式选择一个 tcp_frame_name
执行语义：第一版只采纳选定侧关节，另一侧保持当前目标；后续再升级双臂同时联合运动
```

这样既能验证双臂融合规划描述、14-DOF 关节顺序和多 TCP context，又避免一开始进入双末端同时约束、闭链协作和复杂避碰的高风险区域。

## 1. 为什么选择按阶段选择 TCP

当前项目的 cuMotion 封装和动作脚本以“一个 TCP 目标”为中心。cuMotion 本身可以处理任意宽度的 C-space，因此双臂融合后从 `q[7]` 变成 `q[14]` 是自然的；但双 TCP 同时到位不是现有接口的直接形态。

第一版采用按阶段选择 TCP 的原因：

- 每个阶段仍然只规划一个 TCP，和现有 `pinch_grasp.py` 结构接近。
- 同一个 cuMotion context 可以包含左右两个 TCP frame，每次调用 IK/planner 时显式选择一个；同一侧可以连续选择多次，不要求左右轮流。
- 左右臂的 C-space 已经融合成 14 维，为后续联合规划、自碰撞配置和双臂同时运动打基础。
- 未被选中的一侧在第一版中保持当前目标，便于观察和调试。
- 同侧手和臂不需要碎片化拆开，手预成型可以随手臂 approach 同步，真正闭合单独成阶段。

## 2. 目标动作流程

第一版动作可以按如下阶段组织：

```text
left_pre_shape_and_approach:
  tcp_frame_name = left_pinch_tcp
  左手预成型，左臂到 approach pose
  右臂右手保持

left_descend:
  tcp_frame_name = left_pinch_tcp
  左臂沿目标方向靠近抓取点
  右侧保持

left_close:
  左臂保持，左手闭合
  右侧保持

left_adjust_or_retry:
  tcp_frame_name = left_pinch_tcp
  如有需要，可以继续选择左侧做修正或重试
  右侧保持

right_pre_shape_and_approach:
  tcp_frame_name = right_pinch_tcp
  右手预成型，右臂到 approach pose
  左侧保持

right_close:
  右臂保持，右手闭合
  左侧保持

optional_lift_or_pull:
  可以继续任意选择 left/right TCP，也可以在目标构型验证后升级为 14-DOF 同时规划
```

第一版的关键约束是：当前阶段只让选定侧的 7 个 arm joints 更新，另一侧 7 个 arm joints 用当前值覆盖回去。即使某次单 TCP IK 返回了完整 `q[14]`，也只采纳选定侧的那 7 个关节。

## 3. 规划模型

### 3.1 配置驱动的融合 URDF

双臂规划 URDF 不再手工维护固定文件。`ar5v2_l6v1_dual.yaml` 作为左右臂相对位姿的唯一来源：

```yaml
robots:
  left:
    robot:
      asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
      prim_path: /World/AR5V2_L6V1_L
    root_pose:
      xyz: [0.0, 0.2, 0.0]
      rpy: [-1.5707, 0.0, 0.0]
  right:
    robot:
      asset_path: assets/combined_system/AR5V2_L6V1_R/AR5V2_L6V1_R.xml
      prim_path: /World/AR5V2_L6V1_R
    root_pose:
      xyz: [0.0, -0.2, 0.0]
      rpy: [1.5707, 0.0, 0.0]
```

这份 `root_pose` 同时用于两处：

- Isaac 导入左右 MJCF articulation 后，在 `world.reset()` 前把各自 root prim 摆到对应世界位姿。
- 创建 cuMotion context 前，根据左右单臂 URDF/XRDF 生成缓存的双臂规划 URDF/XRDF。

生成后的 cuMotion URDF 拓扑为：

```text
world
  fixed -> AR5V2_L_arm_base -> left arm chain
  fixed -> AR5V2_R_arm_base -> right arm chain
```

该 URDF 不包含手部 DOF。灵巧手仍由 Isaac/MJCF controller 负责；pinch TCP 通过闭合手型离线计算后，作为 fixed TCP link 注入到生成的双臂 URDF：

```text
AR5V2_L_arm_flan_link -> left_pinch_tcp
AR5V2_R_arm_flan_link -> right_pinch_tcp
```

`cumotion.left/right` 延续单臂 cuMotion 配置结构，分别描述左右单臂 XRDF、URDF 和 flange；
双臂配置不再直接写单个 `cumotion.xrdf_path/urdf_path/flange_frame`。运行时按左右 `root_pose`
生成缓存 URDF，并把左右 XRDF 融合成 14-DOF XRDF：

```yaml
cumotion:
  left:
    xrdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf
    urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
    flange_frame: AR5V2_L_arm_flan_link
  right:
    xrdf_path: assets/single_system/arm/AR5V2_R/AR5V2_R.xrdf
    urdf_path: assets/single_system/arm/AR5V2_R/AR5V2_R.urdf
    flange_frame: AR5V2_R_arm_flan_link
  output_dir: .cache/cumotion
```

缓存文件名由左右单臂 URDF/XRDF 路径和左右 `root_pose` 计算 hash，避免每次运行无意义重建，也避免 pose 改动后误用旧资产。
融合模型不设置默认 TCP/frame；每次 IK 或规划都必须显式传入左侧或右侧的 `tcp_frame_name`。

### 3.2 融合 XRDF

融合 XRDF 由运行时从左右单臂 XRDF 生成。它合并 C-space、默认关节位置、加速度限制、jerk 限制
和 tool frames；C-space 为左右臂关节拼接：

```yaml
cspace:
  joint_names:
    - AR5V2_L_arm_joint_1
    - ...
    - AR5V2_L_arm_joint_7
    - AR5V2_R_arm_joint_1
    - ...
    - AR5V2_R_arm_joint_7
```

默认关节位置、加速度限制和 jerk 限制从左右单臂 XRDF 合并。

### 3.3 多 TCP 注入

`make_cumotion_context(...)` 扩展为支持多个 `TcpFrame`。这样同一个 context 可以同时加载左右 TCP：

```python
with make_cumotion_context(
    dual_cumotion_config,
    tcp=[left_pinch_tcp, right_pinch_tcp],
) as context:
    left_ik = context.make_inverse_kinematics(tcp_frame_name="left_pinch_tcp")
    right_ik = context.make_inverse_kinematics(tcp_frame_name="right_pinch_tcp")
```

## 4. IK 和 C-space 目标拼接

选定侧阶段的目标生成规则：

```text
base_q[14] = 当前双臂关节
solution_q[14] = 单 TCP IK 返回的完整解

如果选定侧是 left:
  q_goal[left_indices] = solution_q[left_indices]
  q_goal[right_indices] = base_q[right_indices]

如果选定侧是 right:
  q_goal[left_indices] = base_q[left_indices]
  q_goal[right_indices] = solution_q[right_indices]
```

这个规则避免普通单 TCP IK 意外移动未选中的一侧。后续如果希望另一侧主动让路，可以把这个保持策略放宽成阈值检查或软约束。

## 5. 避碰策略

第一版不声称“自动免碰撞”。它只做三件事：

- 在融合 XRDF 中保留 14-DOF 统一 C-space，为 self-collision 配置留出入口。
- 用 cuMotion collision-aware planner 时，让左右臂属于同一个 robot description。
- 对阶段终点和采样轨迹保留后验检查入口。

真正免碰撞需要：

- XRDF 中配置有效 collision spheres。
- self-collision mask 不错误屏蔽左右臂之间的碰撞对。
- IK 使用 collision-free IK，或者 planner 使用碰撞约束。
- 轨迹采样后做后验碰撞检查，失败则换 seed、换 waypoint 或调整目标。

## 6. 同侧手臂关系

同一侧的手和臂没有必要频繁拆成多个互相等待的小阶段。推荐：

```text
手预成型 + 手臂 approach 可以同步
手臂 descend 时手保持 pre-pinch
真正接触后单独 close hand
抓住后手保持 closed，手臂执行 lift/pull
```

原因是机械臂由 cuMotion 规划，手指是 scripted command-space smooth target。把闭合动作单独成阶段能更清楚地区分“到位”和“接触/夹持”。

## 7. 第一版代码修改范围

第一版实现以下基础设施：

- 新增双臂按阶段选择运动所需的关节分组和目标拼接工具。
- 新增双臂按阶段选择 TCP 配置。
- `make_cumotion_context(...)` 支持多个 TCP frame。
- 新增 smoke 脚本，用于验证配置拼接，并在可用时验证左右 TCP 能同时进入同一个 context。
- 增加纯 Python 测试覆盖多 TCP 注入和选定侧 C-space 拼接。

暂不实现：

- 双 TCP 同时 IK。
- 双臂同时 task-space 约束。
- 完整 Isaac 双臂执行入口。
- 自动碰撞恢复。

## 8. 后续升级路径

第一版跑通后，可以按下面顺序升级：

1. 在 Isaac 中导入左右两个 AR5+L6 articulation，共用双臂 cuMotion context 做规划。
2. 将选定侧阶段生成的 `q_goal[14]` 拆回左右 controller command-space 执行。
3. 增加 trajectory 后验自碰撞检查。
4. 放开未选中侧保持约束，允许 14-DOF planner 在阈值内微调另一侧避障。
5. 实现左右分别 IK 后拼接目标，再做双臂同时 C-space 规划。
6. 最后再考虑真正双 TCP 同时约束或闭链协作。

## 9. 第二版：完整 Isaac 双机器人执行

第二版要解决第一版遗留的执行侧问题：原来的 `pinch_grasp.py` 是单 articulation、单
controller、单 logger 的结构，不能把左右 AR5+L6 同时作为 Isaac 中的两个机器人执行。第二版
的目标是补齐双机器人执行骨架，并把双臂同步运动测试拆到独立入口
`scripts/dual_arm_motion_test.py`；`pinch_grasp.py` 继续保持单臂抓取 demo：

```text
规划侧：
  一个融合 cuMotion robot description
  C-space = left_arm_7 + right_arm_7
  每次规划选择 left_pinch_tcp 或 right_pinch_tcp

执行侧：
  Isaac stage 中导入 left_robot articulation
  Isaac stage 中导入 right_robot articulation
  每侧各自拥有 JointController 和 command-space
  每个 physics step 先下发左右目标，再 world.step()
```

### 9.1 配置结构

双臂配置不再让顶层 `robot/controlled_joints` 假装代表整个 dual robot，而是显式写成左右两侧：

```yaml
robots:
  left:
    robot:
      name: ar5v2_l6v1_l
      asset_type: mjcf
      asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
      prim_path: /World/AR5V2_L
    controlled_joints:
      - AR5V2_L_arm_joint_1
      - ...
      - L6V1_L_hand_pinky_mcp_pitch
  right:
    robot:
      name: ar5v2_l6v1_r
      asset_type: mjcf
      asset_path: assets/combined_system/AR5V2_L6V1_R/AR5V2_L6V1_R.xml
      prim_path: /World/AR5V2_R
    controlled_joints:
      - AR5V2_R_arm_joint_1
      - ...
      - L6V1_R_hand_pinky_mcp_pitch
```

`cumotion.left/right` 分别指向左右单臂 XRDF/URDF；运行时根据左右 `root_pose` 生成 cuMotion
融合 URDF，并把左右 XRDF 融合成同名缓存 XRDF。`configs/dual_arm/...` 的 `dual_arm.left/right` 只保存规划 C-space、flange/TCP frame 和侧别 MJCF 路径。pinch grasp
的预夹/闭合手型属于动作脚本，留在 `scripts/pinch_grasp.py`。
这让“左右安装位姿”“规划模型”“Isaac 执行模型”和“任务动作参数”分开：`root_pose` 是左右安装位姿的
单一来源，cuMotion 看一个 14-DOF robot，Isaac 看两个 articulation，pinch_grasp 自己决定手型。

### 9.2 执行模型

新增双机器人执行层，不复用单机器人 `ExecutionRuntime` 硬塞两个 controller：

```text
DualRobotRuntime
  left:  articulation + JointController + optional logger
  right: articulation + JointController + optional logger
  shared world + ArticulationAction + SimulationApp

DualCommandPositionTargetStep
  left command target 可选
  right command target 可选
  对两侧都 build/apply targets
  然后只调用一次 world.step()

DualCommandPositionTrajectoryStep
  输入左右 command-space trajectory
  按同一采样 index 同步播放
  缺失侧保持上一帧/当前目标
```

关键规则是：左右 action 必须在同一个 physics step 之前全部下发，再统一 `world.step()`。不能先左臂
step 完一段、再右臂 step 一段，否则就退回了时间上串行执行。

### 9.3 14-DOF 规划结果拆分

融合 cuMotion 轨迹的列顺序是：

```text
left_arm_7 + right_arm_7
```

执行前按关节名拆分：

```text
dual q[14] -> left arm command columns
dual q[14] -> right arm command columns
```

手部目标仍然是各侧 controller command-space 的一部分。手可以按阶段插值到 pre-pinch 或 closed；
机械臂列来自 cuMotion 采样轨迹，手部列由动作脚本按阶段补齐。

### 9.4 第二版暂不解决

- 双 TCP 同时 IK。
- 复杂双臂闭链协作。
- 自动碰撞恢复。
- 把旧 `pinch_grasp.py` 一次性改成完整双臂抓取 demo。

第二版先提供可测试的执行基础设施和 smoke 脚本：能在 Isaac 中导入左右机器人、创建两个 controller、
同步下发左右目标，并验证配置结构不会再出现“dual 配置只控制左臂”的误导。`scripts/dual_arm_motion_test.py`
按类似单臂 pinch grasp 的阶段执行预夹、机械臂 reach、闭合手和回初始位；完整双臂抓取 demo
后续可以在这个 motion test 分支上继续扩展。
