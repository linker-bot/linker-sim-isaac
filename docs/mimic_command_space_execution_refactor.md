# 主动关节命令空间执行改造计划

本文档面向后续代码生成模型。目标不是继续讨论分层，而是给出可执行的改造规格：让动作脚本只表达主动关节和动作意图，让 mimic follower 的运行时处理统一留在控制器/执行层边界内。

## 0. 给代码生成模型的执行规则

必须遵守：

- 只围绕主动关节命令空间、mimic follower 执行边界和 `scripts/pinch_grasp.py` 简化改动，不重构无关模块。
- 当前包路径是 `src/linkerbot_sim/...`。如果看到旧文档里的 `source/...` 路径，只把它当作迁移前命名。
- 不把 mimic 公式写进动作脚本。
- 不把 mimic 公式写进 cuMotion 后端。
- 不让 execution 层解析 MJCF。execution 只负责按时间播放目标；mimic 关系仍由 `JointController` 根据 MJCF 和实际 master 状态计算。
- 保留 `JointController` 当前语义：follower 每帧根据 master 实际位置/速度推导，而不是根据 master 命令目标推导。
- 保留 pinch TCP 几何计算中的 mimic 展开。那是“闭合手型几何建模”，不是运行时 follower 控制。
- 每个阶段都运行相关测试，避免一次性大改。

推荐工作方式：

1. 先补 execution/controller 单元测试，固定命令空间执行语义。
2. 再补少量 helper，让 command-space 目标构造变简单。
3. 再改 `scripts/pinch_grasp.py`，把完整 DOF 目标逐步收缩到主动命令空间。
4. 最后更新 README 和相关文档，确认没有旧 full-DOF/follower 叙述残留。

## 1. 当前问题

当前 `scripts/pinch_grasp.py` 仍然大量使用完整 Isaac articulation DOF：

- 读取 `robot.dof_names`。
- 用 `target_vector_from_mapping(...)` 构造完整 DOF 目标。
- 把 cuMotion C-space trajectory 嵌回完整 DOF trajectory。
- 对手部目标也按完整 DOF base 叠加。
- 通过 `SmoothJointTargetStep`、`FullJointTrajectoryStep`、`HoldJointTargetStep` 播放完整 DOF。

这会让动作脚本关心太多执行细节：

- 哪些 DOF 是主动关节。
- 哪些 DOF 是 mimic follower。
- follower 在完整 DOF 中的位置。
- 非本阶段关节如何保持。
- 完整 DOF trajectory 如何给 controller 消费。

但项目里 `JointController` 已经有正确的运行时 follower 处理：

- `build_control_targets(...)` 接收 command-space 主动关节目标。
- `targets_from_full_state(...)` 接收完整 DOF 目标。
- 两个入口都会调用 `_apply_follower_targets(...)`。
- `_apply_follower_targets(...)` 使用 master 实际状态生成 follower target。
- `apply_targets(...)` 将主动关节和 follower 分组下发，避免 effort action 覆盖 follower position drive。

因此，外层动作脚本继续构造完整 DOF 目标已经不是必须的。

## 2. 分层目标

目标分层如下：

| 层级 | 应负责 | 不应负责 |
|---|---|---|
| `scripts/pinch_grasp.py` | 动作阶段、主动关节目标、TCP 目标、cuMotion 请求 | mimic follower 展开、完整 DOF action、每帧 follower target |
| `execution` | 按 physics dt 播放 command-space 或 full-DOF 目标，调用 controller | 解析 MJCF、计算 mimic 关系、运行 IK |
| `JointController` | 命令空间到完整 DOF control target、mimic follower 每帧跟随、Isaac action 分组下发 | 生成动作阶段、规划 cuMotion |
| `tcp/pinch_tcp.py` | 为闭合手型几何计算一次性展开 mimic，生成 pinch TCP | 每帧控制 follower |
| `backends/cumotion` | 机械臂 C-space IK/path/trajectory 和 TCP context 装配 | Isaac full DOF、手部 mimic follower |

最终理想状态：

- `pinch_grasp.py` 中不再导入 `expand_targets_with_mjcf_equalities`。
- `pinch_grasp.py` 不再把 follower 写入任何目标数组。
- `pinch_grasp.py` 中的普通手型目标只包含 master/主动手部关节。
- `pinch_grasp.py` 能用 controller command space 播放手部和机械臂命令。
- follower 只通过 `JointController` 在执行时生成。

## 3. 非目标

本次不要做：

- 不修改 mimic 数学模型。继续使用 `src/linkerbot_sim/robots/mimic.py` 中的 MJCF equality 解析和 `MimicFollowerTargetMapper`。
- 不把完整 articulation DOF 映射移入 cuMotion 后端。
- 不要求所有 execution API 立刻删除 full-DOF 入口。full-DOF 入口仍可用于测试、调试或未来其它动作。
- 不改变 `JointTrajectory` 数据结构。它本来已经允许列表示完整 DOF 或 command space。
- 不改变 cuMotion 输出语义。cuMotion trajectory 仍是机械臂 C-space trajectory，需要由调用方决定如何放入执行命令空间。
- 不在真实 Isaac/cuMotion 环境中才能验证；新增逻辑必须有 fake 单元测试。

## 4. 推荐最终接口

### 4.1 Controller 暴露命令空间名称

建议在 `src/linkerbot_sim/controllers/joint_controller.py` 增加只读属性：

```python
@property
def command_joint_names(self) -> tuple[str, ...]:
    return tuple(self.dof_names[int(index)] for index in self.command_indices)
```

用途：

- 动作脚本可以按 `controller.command_joint_names` 构造 command-space 目标。
- execution 测试可以断言 trajectory 列顺序和 controller command space 一致。
- 避免动作脚本自己重复推导 follower set。

不要暴露可变列表。

### 4.2 关节目标 helper

建议新增一个 command-space 稀疏目标 helper，可以放在 `src/linkerbot_sim/robots/joint_groups.py`：

```python
def target_vector_for_names(
    joint_names: Sequence[str],
    targets: Mapping[str, float],
    *,
    base: np.ndarray | None = None,
) -> np.ndarray:
    ...
```

也可以复用现有 `target_vector_from_mapping(...)`，但要改注释，不要继续把它描述成“完整 DOF 专用”。如果复用现有函数，首选做法：

- 保持函数名不变，避免大范围改调用。
- 修改 docstring：`dof_names` 改成 `joint_names` 语义，说明可用于完整 DOF 或 command-space。
- 不改变行为。

### 4.3 Command-space execution step

当前已有函数：

- `execute_command_joint_trajectory(...)`

但脚本现在主要使用 step dataclass：

- `SmoothJointTargetStep`
- `FullJointTrajectoryStep`
- `HoldJointTargetStep`

建议增加 command-space step dataclass，并导出到 `src/linkerbot_sim/execution/__init__.py`：

```python
@dataclass(frozen=True)
class SmoothCommandJointTargetStep:
    start_command: np.ndarray
    target_command: np.ndarray
    duration: float
    phase: str
    base_positions: np.ndarray | None = None

@dataclass(frozen=True)
class CommandJointTrajectoryStep:
    trajectory: JointTrajectory

@dataclass(frozen=True)
class HoldCommandJointTargetStep:
    target_command: np.ndarray
    duration: float
    phase: str
    base_positions: np.ndarray | None = None
```

命名可以微调，但要满足：

- 名称明确带 `Command`，表示输入列顺序是 controller command space。
- step 内部调用 `joint_controller.build_control_targets(...)`。
- 不直接处理 follower。
- 日志仍通过 `_apply_control_targets_once(...)` 记录 `joint_controller.driven_indices`。

### 4.4 Command-space smooth/hold 函数

建议在 `src/linkerbot_sim/execution/steps.py` 增加：

```python
def execute_smooth_command_joint_target(...):
    ...

def execute_command_joint_hold(...):
    ...
```

语义：

- `start_command` 和 `target_command` 长度等于 `joint_controller.command_indices.size`。
- 每个 physics step 先用 smoothstep 得到 command-space position/velocity。
- 再调用 `_apply_command_joint_target_once(...)`。
- `_apply_command_joint_target_once(...)` 会进入 `JointController.build_control_targets(...)`，follower 在那里被覆盖。
- `base_positions` 缺省时读取 articulation 当前完整位置；每步更新为上一帧 `targets.positions`，保持非 command DOF 连续。

不要在 execution 里写 follower 公式。

## 5. Pinch grasp 改造规格

### 5.1 去掉脚本内的 mimic 展开导入

当前 `src/linkerbot_sim/tcp/pinch_tcp.py` 的 `make_pinch_tcp(...)` 已经会调用 `fingertip_pinch_local_offset(...)`，后者内部会 `expand_targets_with_mjcf_equalities(...)`。

因此 `scripts/pinch_grasp.py` 中这段是重复的：

```python
closed_geometry_targets = expand_targets_with_mjcf_equalities(
    DEFAULT_CLOSED_PINCH_HAND_TARGETS, mjcf_path
)
tcp = make_pinch_tcp(..., closed_geometry_targets, ...)
```

应改为：

```python
tcp = make_pinch_tcp(
    mjcf_path,
    DEFAULT_CLOSED_PINCH_HAND_TARGETS,
    parent_frame=cumotion_config.flange_frame,
    frame_name=tcp_frame_name,
)
```

然后删除 `expand_targets_with_mjcf_equalities` import。

这一步只影响 TCP 几何输入，不改变运行时 follower 控制。

### 5.2 用 command space 表达手部目标

动作脚本应以 `controller.command_joint_names` 为列顺序构造手部目标：

```python
command_names = controller.command_joint_names
initial_command = current_full[controller.command_indices]
pre_pinch_command = target_vector_from_mapping(
    command_names,
    DEFAULT_PRE_PINCH_HAND_TARGETS,
    base=initial_command,
)
closed_command = target_vector_from_mapping(
    command_names,
    DEFAULT_CLOSED_PINCH_HAND_TARGETS,
    base=grasp_open_command,
)
```

注意：

- `DEFAULT_*_HAND_TARGETS` 仍只写主动手部关节。
- 如果目标 dict 中包含 follower，`target_vector_from_mapping(command_names, ...)` 会报 missing joint，这正好能防止脚本误写 follower。
- arm 关节命令也在同一个 command vector 中占列。

### 5.3 cuMotion C-space 与 command space 映射

cuMotion 只输出机械臂 C-space。执行层需要 command-space trajectory。

动作脚本中应建立：

```python
command_index_by_name = {name: i for i, name in enumerate(controller.command_joint_names)}
arm_command_indices = np.asarray(
    [command_index_by_name[name] for name in context.joint_names()],
    dtype=int,
)
```

同时仍需要从 articulation 当前完整状态取当前 C-space：

```python
dof_index_by_name = {name: i for i, name in enumerate(robot.dof_names)}
arm_dof_indices = np.asarray([dof_index_by_name[name] for name in context.joint_names()])
current_cspace = current_full[arm_dof_indices]
```

边界：

- `arm_dof_indices` 只用于读取真实 articulation 状态给 cuMotion。
- `arm_command_indices` 用于把 cuMotion trajectory 写回 command-space trajectory。
- 不再把 cuMotion trajectory 写回完整 DOF trajectory。

### 5.4 轨迹嵌入 helper 改名/改语义

当前脚本内 helper `_full_trajectory_from_cumotion_trajectory(...)` 和 `_full_positions_from_cspace_path(...)` 是 full-DOF 语义。

建议改成 command-space 语义：

```python
def _command_trajectory_from_cumotion_trajectory(
    cumotion_trajectory,
    *,
    motion_planner,
    command_joint_names: tuple[str, ...],
    arm_command_indices: np.ndarray,
    start_command: np.ndarray,
    target_command: np.ndarray,
    requested_duration_s: float,
    phase: str,
    physics_dt: float | None,
) -> JointTrajectory:
    ...
```

内部逻辑基本同现有实现：

- 采样/retime cuMotion C-space trajectory。
- 非 arm command columns 在 start/target 之间线性补齐。
- arm command columns 用 cuMotion positions/velocities/accelerations/jerks 覆盖。
- `joint_names=command_joint_names`。

同时改：

```python
_full_positions_from_cspace_path -> _command_positions_from_cspace_path
```

### 5.5 Specified TCP line trajectory 也返回 command-space trajectory

当前 `build_specified_tcp_line_trajectory(...)` 输入 `start_all`，输出完整 DOF trajectory。

改造后应输入 command-space 起点和映射：

```python
def build_specified_tcp_line_trajectory(
    *,
    context: CuMotionContext,
    tcp_frame_name: str,
    command_joint_names: tuple[str, ...],
    arm_command_indices: np.ndarray,
    current_q: np.ndarray,
    start_command: np.ndarray,
    target_position: np.ndarray,
    ...
) -> JointTrajectory:
```

关键点：

- `current_q` 是 cuMotion C-space 起点，来自 command 或 actual 均可，但必须明确。推荐使用 `start_command[arm_command_indices]`，保证当前阶段轨迹连续。
- `result.path[-1]` 写入 `target_command[arm_command_indices]`。
- 返回 command-space `JointTrajectory`。

### 5.6 执行步骤使用 command-space step

`pinch_grasp_execution_steps(...)` 应改成 command-space step：

- `SmoothCommandJointTargetStep` 播放 pre-pinch。
- `CommandJointTrajectoryStep` 播放 move_to_approach、approach_line、lift、wiggle、post sweep。
- `SmoothCommandJointTargetStep` 播放 close_fingers。
- `HoldCommandJointTargetStep` 播放 final hold。

不要在脚本里调用 full-DOF step。

### 5.7 运行结果只返回摘要

保留前一轮简化后的结果语义：

```python
return {"steps": step, "ik": plan["ik"]}
```

不要重新把完整 plan、完整目标数组或后端 `MotionResult` 暴露出去。

## 6. 测试计划

### 6.1 Controller 测试

修改或新增 `tests/test_joint_controller.py`：

- `command_joint_names` 不包含 follower。
- `command_joint_names` 顺序与 `command_indices` 一致。
- 即使 `joint_names=["all"]` 或配置里包含 follower，command space 仍剔除 follower。
- `build_control_targets(...)` 中调用方传入的 command positions 只覆盖主动关节，follower 最终由 actual master 状态覆盖。

### 6.2 Execution 测试

修改或新增 `tests/test_execution_steps.py`：

- `SmoothCommandJointTargetStep` 每帧调用 fake controller 的 `build_control_targets(...)`，不是 `targets_from_full_state(...)`。
- `CommandJointTrajectoryStep` 按 physics dt 插值播放 command-space trajectory，而不是按采样点硬播放。如果保留旧 `execute_command_joint_trajectory(...)` 采样点播放语义，也要明确测试两个入口差异。
- `HoldCommandJointTargetStep` 持续下发同一个 command-space 目标，并让 fake controller 扩展成完整 targets。
- logger 记录 indices 使用 `driven_indices`，即主动关节 + follower。

### 6.3 Pinch grasp motion planning 测试

修改 `tests/test_pinch_grasp_motion_planning.py`：

- helper 输出 trajectory 的 `joint_names` 是 command-space 名称。
- arm columns 使用 cuMotion trajectory。
- hand/master columns 按 start/target 插值。
- 不再断言完整 DOF 中的 follower 列。
- specified TCP line helper 只通过 C-space current_q 调 cuMotion，不要求完整 DOF 输入。

### 6.4 TCP 测试

保留或增强 `tests/test_tcp_frames.py`：

- `make_pinch_tcp(...)` 接收主动手部 targets 时，内部会展开 mimic follower。
- 不要求调用方预先展开 follower。

### 6.5 系统配置测试

修改 `tests/test_system_configs.py`：

- 如果测试手型目标，确认默认手型目标不包含 follower joint。
- 如果需要检查 controlled joints，确认 controller 会剔除 follower，而不是要求 YAML 手动剔除。

### 6.6 回归命令

每个阶段至少运行：

```bash
env_isaaclab/bin/python -m pytest -q tests/test_joint_controller.py tests/test_execution_steps.py tests/test_tcp_frames.py
env_isaaclab/bin/python -m pytest -q tests/test_pinch_grasp_motion_planning.py tests/test_system_configs.py
```

最终运行：

```bash
env_isaaclab/bin/python -m compileall -q src scripts tests
env_isaaclab/bin/python -m pytest -q tests
git diff --check
```

## 7. 推荐实施阶段

### Phase 1: 固定 controller command-space 语义

改动文件：

- `src/linkerbot_sim/controllers/joint_controller.py`
- `tests/test_joint_controller.py`

任务：

- 增加 `command_joint_names` 属性。
- 增加测试，确保 follower 不在 command space。
- 不改动作脚本。

完成条件：

- controller 测试通过。
- 没有行为回退到 follower command input。

### Phase 2: 增加 command-space step

改动文件：

- `src/linkerbot_sim/execution/steps.py`
- `src/linkerbot_sim/execution/__init__.py`
- `tests/test_execution_steps.py`

任务：

- 增加 `SmoothCommandJointTargetStep`。
- 增加 `CommandJointTrajectoryStep`。
- 增加 `HoldCommandJointTargetStep`。
- 复用 `_apply_command_joint_target_once(...)` 和 `_apply_control_targets_once(...)`。
- 不解析 MJCF。

完成条件：

- execution 单元测试证明 command-space step 通过 controller 生成完整 targets。

### Phase 3: 简化 pinch TCP 调用

改动文件：

- `scripts/pinch_grasp.py`
- `tests/test_tcp_frames.py`

任务：

- 删除脚本中的 `expand_targets_with_mjcf_equalities` import。
- 直接把 `DEFAULT_CLOSED_PINCH_HAND_TARGETS` 传给 `make_pinch_tcp(...)`。
- 确认 `make_pinch_tcp(...)` 内部展开 mimic。

完成条件：

- TCP 测试和 pinch grasp motion planning 测试通过。

### Phase 4: 把 pinch grasp 规划改成 command-space trajectory

改动文件：

- `scripts/pinch_grasp.py`
- `tests/test_pinch_grasp_motion_planning.py`

任务：

- 建立 `command_joint_names`、`arm_command_indices`、`arm_dof_indices`。
- 把 `_full_trajectory_from_cumotion_trajectory(...)` 改成 command-space 语义。
- 把 `build_planned_joint_motion_trajectory(...)` 改成输入/输出 command-space trajectory。
- 把 `build_specified_tcp_line_trajectory(...)` 改成输入/输出 command-space trajectory。
- 不再在脚本中构造完整 DOF 阶段目标。

完成条件：

- `pinch_grasp.py` 不再调用 `target_vector_from_mapping(robot.dof_names, ...)` 构造手部动作。
- `pinch_grasp.py` 不再直接写 follower 目标。
- motion planning 测试覆盖 command-space trajectory 嵌入。

### Phase 5: 切换执行步骤

改动文件：

- `scripts/pinch_grasp.py`
- `tests/test_pinch_grasp_motion_planning.py`

任务：

- `pinch_grasp_execution_steps(...)` 使用 command-space step。
- `run_pinch_grasp_action(...)` 保持只返回 `steps` 和 `ik`。
- 删除脚本中不再使用的 full-DOF helper/import。

完成条件：

- `scripts/pinch_grasp.py` 中不再导入 `FullJointTrajectoryStep`、`SmoothJointTargetStep`、`HoldJointTargetStep`，除非仍有明确 full-DOF 分支。
- `scripts/pinch_grasp.py` 中 full-DOF 目标只用于读取 robot 当前实际状态或给 cuMotion 当前 C-space seed，不用于执行动作阶段。

### Phase 6: 文档更新

改动文件：

- `README.md`
- `docs/cumotion_interface.md`
- 其它提到 pinch grasp full-DOF/follower 旧边界的文档。

任务：

- 更新分层说明：script 表达主动命令空间，controller 负责 follower。
- 明确 pinch TCP 几何计算仍会在 TCP helper 内部展开 mimic。
- 明确执行层 command-space trajectory 和 full-DOF trajectory 都支持，但 pinch grasp 走 command-space。

完成条件：

- `rg -n "完整 DOF|full DOF|follower|mimic|PinchGraspAction|action_config" docs README.md scripts src tests` 没有误导性旧叙述。

## 8. 关键风险和规避方式

### 8.1 不要把 follower 从 TCP 几何里删掉

运行时控制可以不在脚本处理 follower，但 pinch TCP 几何必须考虑 follower。否则指尖中心会偏向未闭合手型。

正确边界：

- `tcp/pinch_tcp.py`: 为几何计算展开 mimic。
- `controller/joint_controller.py`: 为每帧执行展开 mimic。
- `scripts/pinch_grasp.py`: 不展开 mimic。

### 8.2 command-space trajectory 的列顺序必须稳定

列顺序必须由 `controller.command_joint_names` 定义。不要用 dict insertion order 临时拼列。

### 8.3 cuMotion C-space 和 controller command space 不是同一个空间

cuMotion C-space 通常只包含机械臂关节。controller command space 包含机械臂主动关节和手部 master 关节。

必须显式建立名字映射，不要假设 arm joints 在 command space 的前 N 列。

### 8.4 follower 使用 actual master，而不是 command master

不要“修正”为根据 command target 生成 follower。当前根据 actual master 可以减少从动关节超前导致的接触抖动，这个语义要保留。

### 8.5 不要删除 full-DOF execution 能力

本次目标是让 pinch grasp 不依赖 full-DOF 执行，不是删除 full-DOF 通用能力。保留 full-DOF step 方便其它脚本、测试和调试。

## 9. 最终验收标准

改造完成时应满足：

- `scripts/pinch_grasp.py` 不导入 `expand_targets_with_mjcf_equalities`。
- `scripts/pinch_grasp.py` 不直接写 follower 目标。
- `scripts/pinch_grasp.py` 的动作执行步骤使用 command-space step。
- `JointController` 仍然每帧根据 actual master 状态计算 follower target。
- `make_pinch_tcp(...)` 仍然能基于主动手型目标计算包含 follower 几何影响的 pinch TCP。
- 所有单元测试通过。
- `README.md` 和相关 docs 清楚说明：外层动作脚本只考虑主动关节；从动关节在 controller/execution 边界内统一处理。
