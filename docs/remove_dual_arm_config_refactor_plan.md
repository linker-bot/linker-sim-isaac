# 移除 configs/dual_arm 的重构方案

状态：已完成。本文保留重构前后的设计背景，因此正文会提到已删除的旧接口名和旧配置路径；
运行时代码、脚本、测试和用户文档不应再引用这些旧接口。

## 目标

删除 `configs/dual_arm/` 这一层独立 profile，让双臂运行时只从 scene 中声明的
left/right robot profile 和各自 `cumotion` 资源推导双臂规划语义。

重构后：

- 双臂和单臂都以 `configs/robots/*.yaml` 作为机器人资源事实来源。
- 不再需要 `--dual-arm-profile`。
- 不再手写 `dual_arm.left/right.arm_joints`。
- `configs/dual_arm/ar5v2_l6v1_dual.yaml` 和 `configs/dual_arm/example.yaml` 可以删除。

## 背景问题

当前 `configs/dual_arm/ar5v2_l6v1_dual.yaml` 保存：

- `left/right.arm_joints`
- `left/right.flange_frame`
- `left/right.tcp_frame`
- `left/right.combined_mjcf_path`

其中大部分已经和 robot profile 重复：

- `combined_mjcf_path` 等价于对应 `configs/robots/*.yaml` 的 `robot.asset_path`。
- `flange_frame` 已经在对应 robot profile 的 `cumotion.flange_frame`。
- 左右 cuMotion URDF/XRDF 已经在对应 robot profile 的 `cumotion.urdf_path` /
  `cumotion.xrdf_path`。

真正让双臂代码依赖 `configs/dual_arm` 的核心只有：

- selected-side 规划时，需要知道融合 C-space 中哪些关节属于左臂、哪些属于右臂。
- 默认 TCP 名当前从 `dual_arm.left/right.tcp_frame` 读取。

但这两者都不需要独立 profile：

- 左右 arm joints 可以从 left/right robot profile 的 `cumotion.xrdf_path` 中读取
  `cspace.joint_names` 推导。
- 默认 TCP frame 可以继续由入口代码/CLI 构造 `DualArmTcpSpec` 时提供；双臂 runtime 不应为了
  默认值再读取一份 profile。

## 目标结构

### 保留的配置来源

`configs/envs/*.yaml`

- 决定 scene 中使用哪些 left/right robot profile。
- 决定 left/right root pose。

`configs/robots/*.yaml`

- `robot.asset_path`：Isaac 导入资产。
- `robot.prim_path`：Isaac stage 路径。
- `cumotion.urdf_path`：cuMotion 单臂运动学 URDF。
- `cumotion.xrdf_path`：cuMotion 单臂 C-space/XRDF。
- `cumotion.flange_frame`：该侧机械臂法兰 frame。

`configs/cumotion/*.yaml`

- IK、planner、trajectory generation 等算法参数。

### 删除的配置来源

`configs/dual_arm/*.yaml`

不再作为运行时输入。

## 推导规则

### 左右 arm joints

从 each side robot profile 的 `cumotion.xrdf_path` 读取 XRDF：

```yaml
cspace:
  joint_names:
    - ...
```

得到：

- `left_arm_joints = left_xrdf["cspace"]["joint_names"]`
- `right_arm_joints = right_xrdf["cspace"]["joint_names"]`

然后继续调用现有：

```python
DualArmJointPartitions.from_joint_names(
    fused_joint_names,
    left_joint_names=left_arm_joints,
    right_joint_names=right_arm_joints,
)
```

这里的 `fused_joint_names` 仍来自 cuMotion dual context。由于 dual XRDF 本来就是由
left/right XRDF 合并生成，名称应完全一致；不一致时让现有校验抛错即可。

### 左右 flange frame

从 each side robot profile 的 `cumotion.flange_frame` 读取：

- `left_flange_frame = left_robot_config["cumotion"]["flange_frame"]`
- `right_flange_frame = right_robot_config["cumotion"]["flange_frame"]`

用于构造 `tcp_parent_frames`：

```python
tcp_parent_frames = {
    tcp.left.frame_name: left_flange_frame,
    tcp.right.frame_name: right_flange_frame,
}
```

### 默认 TCP frame

不要迁移到 robot config。

当前 `DualArmCuMotionExecutionSession` 接收 `tcp: DualArmTcpSpec`，运行时实际需要的是
调用方传进来的 `tcp.left.frame_name` 和 `tcp.right.frame_name`。如果某些入口需要默认值，
应在入口层保持常量或 CLI 默认，例如：

```python
DualArmTcpSpec(
    left=CartesianTcpFrameSpec(frame_name="left_pinch_tcp", ...),
    right=CartesianTcpFrameSpec(frame_name="right_pinch_tcp", ...),
)
```

这样 robot profile 只描述机器人资源，不承担动作脚本默认参数。

## 代码修改计划

### 1. 新增双臂语义推导 helper

建议位置：

`src/linkerbot_sim/app/motion/dual_arm.py`

或新建：

`src/linkerbot_sim/app/motion/dual_arm_semantics.py`

建议数据结构：

```python
@dataclass(frozen=True)
class DualArmRobotSemantics:
    left_arm_joints: tuple[str, ...]
    right_arm_joints: tuple[str, ...]
    left_flange_frame: str
    right_flange_frame: str
```

入口：

```python
def dual_arm_semantics_from_robot_configs(
    side_robot_configs: Mapping[str, Mapping[str, object]],
) -> DualArmRobotSemantics:
    ...
```

内部逻辑：

- 校验 `side_robot_configs["left"]` 和 `side_robot_configs["right"]` 存在。
- 读取每侧 `cumotion.xrdf_path`，解析 YAML。
- 从 `cspace.joint_names` 取关节名 tuple。
- 从 `cumotion.flange_frame` 取 flange frame。
- 抛出清晰错误：
  - 缺少 side robot config。
  - 缺少 `cumotion`。
  - 缺少 `xrdf_path`。
  - XRDF 文件不存在。
  - XRDF 缺少 `cspace.joint_names`。
  - 缺少 `flange_frame`。

### 2. 改 `DualArmCuMotionExecutionSession`

当前构造函数参数：

```python
dual_arm_profile: str = "ar5v2_l6v1_dual"
```

改为不再需要该参数。

构造函数内部：

- 删除 `self.dual_arm_profile`。
- 删除 `self.dual_arm = load_dual_arm_semantic_config(...)`。
- 改用 `runtime.side_robot_configs` 推导 `self.dual_arm_semantics`。
- `tcp_parent_frames` 使用推导出的 left/right flange frame。
- `DualArmJointPartitions.from_joint_names(...)` 使用推导出的 left/right arm joints。

注意：`DualRobotAppRuntime` 已经包含：

```python
side_robot_configs: Mapping[str, Mapping[str, object]]
```

所以 session 不需要额外加载 profile。

### 3. 改 summary 和调用入口

`dual_arm_cumotion_summary(...)` 当前接收 `dual_arm_profile` 并读取
`configs/dual_arm`。

建议改成：

- 从 env 加载 left/right robot profile。
- 从 `side_robot_configs` 推导双臂语义。
- 若传入 `tcp`，summary 用传入 TCP。
- 若未传入 `tcp`，summary 使用入口默认 TCP 常量，或只显示 `None`/`<provided by caller>`。

`run_dual_arm_cumotion_motion(...)` 和 `run_interactive_dual_arm_motion(...)`：

- 删除 `dual_arm_profile` 参数。
- 透传调用处一并删除。

### 4. 改 CLI

`scripts/dual_arm_motion_test.py`

- 删除 `--dual-arm-profile` 参数。
- 删除调用 `dual_arm_cumotion_summary(...)` / session / interactive 时的
  `dual_arm_profile=args.dual_arm_profile`。
- 保留 TCP 相关 CLI 或默认 TCP 构造逻辑。

### 5. 删除旧配置和旧 helper

删除：

- `configs/dual_arm/ar5v2_l6v1_dual.yaml`
- `configs/dual_arm/example.yaml`
- 空目录 `configs/dual_arm/`

删除或替换：

- `load_dual_arm_semantic_config(...)`
- `_side_arm_joints(...)`
- `_side_tcp_frame_name(...)`
- `_side_flange_frame(...)`

若 `_side_tcp_frame_name(...)` 只服务 summary 默认值，也应删除，避免继续暗示存在 dual_arm profile。

### 6. 更新测试

需要修改：

- `tests/test_dual_arm_motion_test.py`
  - 删除 `test_load_dual_arm_semantic_config_reads_default_profile`。
  - 新增 `test_dual_arm_semantics_from_robot_configs_reads_xrdf_and_flange`。
- `tests/test_dual_arm_selectable_tcp.py`
  - 不再直接读 `configs/dual_arm/ar5v2_l6v1_dual.yaml`。
  - 改用 left/right robot config 推导 joints。
- `tests/test_system_configs.py`
  - 删除遍历 `configs/dual_arm/*.yaml` 的测试。
  - 可新增 robot cumotion config 校验：每个 dual scene 引用的 left/right robot profile 都有
    `cumotion.xrdf_path`、`cumotion.urdf_path`、`cumotion.flange_frame`，且 XRDF 中存在
    `cspace.joint_names`。
- `tests/test_dual_cumotion_urdf.py`
  - 如果使用 dual_arm config 读取 joint names，改为从 side robot XRDF 读取。

### 7. 更新文档

需要更新：

- `README.md`
  - 删除 `configs/dual_arm/` 目录说明。
  - 删除 `--dual-arm-profile` 示例。
  - 说明双臂规划语义来自 left/right robot profile 的 `cumotion` 资源。
- `docs/cumotion_motion_modes_examples.md`
  - 删除 `dual_arm_profile` 参数示例。
- 归档类 `docs/trush/*` 可以不改，或标注历史设计。

## 兼容性和风险

### 风险 1：XRDF joint_names 和 cuMotion context joint_names 不一致

风险原因：

- dual context 的 fused XRDF 是由 left/right XRDF 合并生成。
- 如果生成逻辑或缓存复用异常，可能导致名称不一致。

处理：

- 继续依赖 `DualArmJointPartitions.from_joint_names(...)` 的缺失关节校验。
- 错误信息应包含缺失关节和来源 side。

### 风险 2：默认 TCP frame 来源不清晰

处理：

- 不把 default TCP 塞进 robot config。
- 入口层显式构造 `DualArmTcpSpec`。
- summary 中也不要假装 robot config 有默认 TCP。

### 风险 3：raw-only interactive 仍会初始化 cuMotion context

本次重构只删除 `configs/dual_arm`，不改变 interactive loop 架构。

如果后续希望 raw 模式完全不依赖 cuMotion，应另做：

- `raw-only` interactive runtime。
- 不创建 `DualArmCuMotionExecutionSession`。
- 不读取 XRDF/URDF，不创建 cuMotion context。

## 验收标准

- 仓库中不存在 `configs/dual_arm/`。
- `rg "dual_arm_profile|configs/dual_arm|load_dual_arm_semantic_config" src scripts tests README.md docs/cumotion_motion_modes_examples.md configs`
  无引用。
- `scripts/dual_arm_motion_test.py --interactive` 不再接受 `--dual-arm-profile`。
- 双臂 cuMotion session 可以从 env 选定的 left/right robot profile 推导：
  - left/right arm joints。
  - left/right flange frame。
- 现有 selected-side IK、C-space goal、task-space path、raw_joint_sequence 测试通过。
- `pytest` 相关测试通过。
