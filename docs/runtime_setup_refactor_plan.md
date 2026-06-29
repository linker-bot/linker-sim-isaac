# Runtime Setup 抽取计划

本文整理 `scripts/pinch_grasp.py` 和 `scripts/dual_arm_motion_test.py` 中可复用的配置读取、Isaac 运行时装配、机器人导入和 controller 初始化逻辑，并给出分阶段抽取到 `src/linkerbot_sim/` 的计划。

## 1. 背景

当前两个脚本都承担了两类职责：

- 任务或 demo 逻辑：pinch grasp 的接近、夹取、抬升、wiggle；双臂 scripted motion 的预成型、reach、闭合和回位。
- 运行时装配逻辑：读取 YAML、解析 env 覆盖、启动 Isaac、创建 world、导入 scene objects、导入机器人、应用 USD/PhysX 覆盖、初始化 controller。

第二类逻辑已经不是具体任务语义，继续留在 `scripts/` 中会带来几个问题：

- 新增 demo 时容易复制 env/world/robot/controller 初始化代码。
- 单臂和双臂脚本的 `world.reset()` 前后边界需要人工保持一致。
- cuMotion profile 合并规则写在动作脚本里，后续其它动作复用时入口不清晰。
- 机械臂或灵巧手是否关闭重力应由 robot 执行配置声明，而不是由动作脚本临时决定。

目标是让脚本主要表达“这个 demo 要做什么动作”，把“如何按配置把仿真系统搭起来”沉到 `linkerbot_sim` 包内。

## 2. 设计原则

- 不抽成一个巨大的万能 loader；优先拆成小 dataclass 和小函数。
- Isaac/Omni 相关 import 继续放在函数内部，避免普通单元测试 import 包时依赖 Isaac runtime。
- 明确 `world.reset()` 边界：stage 资产导入和 USD/PhysX 覆盖在 reset 前，controller runtime 初始化在 reset 后。
- 保留脚本对 `SimulationApp` 生命周期的控制，尤其是 `try/finally: simulation_app.close()`。
- 动作语义继续留在脚本里，不把 pinch grasp 阶段参数或双臂 scripted motion 变成通用配置层。
- `argparse` 可以抽公共参数 helper，但不要强行统一两个脚本的全部命令行接口。
- 每一步抽取都应保持行为等价，并配套轻量测试。

## 3. 当前重复点

### 3.1 Env 运行参数解析

两个脚本都在做：

- 读取 `env_config["env"]`
- 解析 `physics_frequency`
- 解析 `render_frequency`
- 解析 `gravity_z`
- 解析 `add_ground`
- 用 CLI 的 `--physics-frequency`、`--render-frequency`、`--gravity-z` 覆盖
- 校验频率为正数

这部分是纯 Python 逻辑，最适合先抽。

### 3.2 Isaac session 创建

两个脚本都在做：

- 设置 `OMNI_KIT_ACCEPT_EULA`
- `launch_simulation_app(gui=...)`
- 延迟 import `SingleArticulation`、`ArticulationAction`、`omni.usd`
- 根据频率创建 `World`
- GUI 模式下调用 `configure_visuals()`
- 获取当前 USD stage

这部分适合抽成 session helper，但需要保留延迟 import 和 app close 边界。

### 3.3 Scene objects

两个脚本都在做：

- `scene_objects_from_env_config(env_config)`
- `add_scene_objects(stage, scene_objects)`
- 打印已加入对象摘要

建议抽“解析和加入 stage”逻辑，打印由脚本保留，或让 helper 返回 summary 数据给脚本打印。

### 3.4 机器人导入和 USD/PhysX 覆盖

单臂和双臂单侧都遵循同一流程：

- `import_robot_asset(...)`
- `apply_root_pose(...)`
- `apply_robot_usd_overrides(...)`
- `solver_settings(...)`
- `apply_solver_iteration_overrides(...)`
- `apply_robot_gravity_policy(...)`
- `world.scene.add(SingleArticulation(...))`

这部分适合抽到 execution setup 层。注意 controller 不能在这里创建，因为它依赖 `world.reset()` 之后稳定的 articulation view。
同时，重力启用/禁用策略不应长期依赖脚本 CLI，而应由 robot 配置声明，并由 setup 统一应用。

### 3.5 Controller 初始化

reset 后两个脚本都在做：

- 根据 robot 配置确认 runtime gravity 状态
- 清零 joint velocities
- 创建 `JointController`
- 调用 `controller.configure_runtime()`

这部分适合和机器人导入 helper 配套抽取。

### 3.6 cuMotion profile 合并

`pinch_grasp.py` 中的下列函数已经具备包级工具属性：

- `merged_robot_config_with_cumotion_profile(...)`
- `robot_cumotion_config(...)`
- `motion_planner_config_from_profile(...)`

它们只依赖配置结构，不依赖具体动作阶段，应移到 cuMotion 配置 helper 模块。

### 3.7 机器人重力策略配置

机械臂或灵巧手的重力策略属于 robot execution 语义，已经统一放到 robot YAML：

- 不同机器人可能希望机械臂和灵巧手分别启用或禁用重力。
- 同一个 env 可以复用不同 robot gravity policy。
- setup 层需要在 stage prim 和 reset 后 articulation runtime 两处保持一致。

建议在 `configs/robots/*.yaml` 中增加明确配置，例如：

```yaml
robot:
  physics:
    gravity:
      default: false
      arm: false
      hand: false
```

双机器人配置中写在各侧 `robots.left.robot.physics.gravity` / `robots.right.robot.physics.gravity` 下。

语义约定：

- `false` 表示导入后禁用对应刚体重力。
- `true` 表示保留重力。
- `default` 作为未知部件或未分类刚体的回退。
- `arm` / `hand` 可覆盖对应部件。

核心原则是：重力策略只从 robot 配置读取，动作脚本不再提供额外重力开关。

### 3.8 手型和手部原子动作

`dual_arm_motion_test.py` 目前从 `pinch_grasp.py` import 默认手型函数，但本次 runtime setup 抽取不处理手型抽象。

后续计划是把手型整理成“手的原子动作”或更完整的 hand action primitive，例如预夹、闭合、张开、保持等；这属于任务/动作抽象，不属于当前运行时装配抽取范围。

## 4. 目标模块

建议新增或扩展以下模块：

```text
src/linkerbot_sim/app/runtime_settings.py
src/linkerbot_sim/app/simulation_session.py
src/linkerbot_sim/execution/setup.py
src/linkerbot_sim/backends/cumotion/profile_config.py
```

## 5. 模块设计

### 5.1 `app/runtime_settings.py`

职责：解析 env YAML 和命令行覆盖，生成启动 world 需要的运行参数。不依赖 Isaac。

建议接口：

```python
@dataclass(frozen=True)
class EnvRuntimeSettings:
    physics_frequency: float
    render_frequency: float
    gravity_z: float
    add_ground: bool

    @classmethod
    def from_env_config(
        cls,
        env_config: Mapping[str, object],
        *,
        physics_frequency_override: float | None = None,
        render_frequency_override: float | None = None,
        gravity_z_override: float | None = None,
    ) -> "EnvRuntimeSettings":
        ...

    @property
    def physics_dt(self) -> float:
        ...

    def rendering_dt(self, *, gui: bool) -> float:
        ...
```

解析规则：

- `env_config` 必须包含顶层 `env` mapping。
- CLI override 优先于 YAML。
- `physics_frequency` 和 `render_frequency` 必须大于 0。
- `add_ground` 默认值可沿用当前脚本逻辑：缺省为 `True`。

### 5.2 `app/simulation_session.py`

职责：启动 Isaac app，创建 world，配置可视化，返回 stage 和 Isaac 类型句柄。

建议接口：

```python
@dataclass(frozen=True)
class SimulationSession:
    app: object
    world: object
    stage: object
    articulation_action_type: object
    single_articulation_type: object


def create_simulation_session(
    *,
    gui: bool,
    settings: EnvRuntimeSettings,
) -> SimulationSession:
    ...
```

实现要求：

- 在函数内部设置 `OMNI_KIT_ACCEPT_EULA` 默认值。
- 在 `launch_simulation_app(...)` 之后再 import Isaac/Omni 模块。
- 使用 `build_world(...)` 创建 world。
- GUI 模式下调用 `configure_visuals()`.
- 不在 helper 内关闭 app，由调用脚本负责 `finally: session.app.close()`。

### 5.3 `execution/setup.py`

职责：封装 Isaac 执行侧的机器人导入和 controller 初始化。

建议分成 reset 前和 reset 后两段。

reset 前：

```python
@dataclass(frozen=True)
class ImportedRobot:
    articulation: object
    articulation_path: str
    imported_root_path: str
    asset_path: Path
    asset_type: str
    controlled_joints: tuple[str, ...]


def import_execution_robot_to_stage(
    *,
    world: object,
    stage: object,
    single_articulation_type: object,
    robot_execution: RobotExecutionConfig,
    controller_profiles: ControllerProfiles,
    env_config: Mapping[str, object],
) -> ImportedRobot:
    ...
```

该函数负责：

- 导入 MJCF/URDF 机器人资产。
- 应用 `root_pose`。
- 应用 USD/PhysX override。
- 应用 env solver iteration。
- 根据 robot 配置应用机器人刚体重力策略。
- 创建并加入 `SingleArticulation`。

reset 后：

```python
@dataclass(frozen=True)
class PreparedRobotRuntime:
    articulation: object
    joint_controller: JointController
    asset_path: Path
    mjcf_path: Path | None


def finalize_robot_controller(
    *,
    imported: ImportedRobot,
    controller_profiles: ControllerProfiles,
    control_mode: str,
) -> PreparedRobotRuntime:
    ...
```

该函数负责：

- 根据 robot 配置确认 articulation runtime gravity 状态。
- 清零关节速度。
- 创建 `JointController`。
- 调用 `controller.configure_runtime()`。

单臂脚本调用一次；双臂脚本左右各调用一次，并在两侧 import 完成后统一 `world.reset()`。

### 5.4 `backends/cumotion/profile_config.py`

职责：保存 cuMotion profile 与 robot YAML 合并、解析相关 helper。

建议迁移接口：

```python
def merged_robot_config_with_cumotion_profile(
    robot_config: Mapping[str, object],
    cumotion_profile: Mapping[str, object],
) -> dict:
    ...


def robot_cumotion_config(robot_config: Mapping[str, object]) -> CuMotionConfig:
    ...


def motion_planner_config_from_profile(
    cumotion_profile: Mapping[str, object],
) -> MotionPlannerBackendConfig:
    ...
```

迁移后：

- `pinch_grasp.py` 从该模块 import。
- 现有测试中对 `pinch_grasp.py` 内函数的直接引用应同步更新。
- 合并优先级保持不变：`configs/cumotion/*.yaml < configs/robots/*.yaml`。

### 5.5 robot gravity policy

职责：把机器人执行配置中的重力策略解析成 stage/runtime 可应用的结构。

该能力可以放在 `assets/robot_loader.py` 的执行配置 dataclass 中，或在 `execution/setup.py` 中定义专用 dataclass。建议先保持靠近 `RobotExecutionConfig`，因为它属于 robot YAML 的执行语义。

建议结构：

```python
@dataclass(frozen=True)
class RobotGravityPolicy:
    default: bool = False
    arm: bool | None = None
    hand: bool | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
    ) -> "RobotGravityPolicy":
        ...

    def enabled_for_component(self, component: str) -> bool:
        ...
```

建议 YAML：

```yaml
robot:
  physics:
    gravity:
      default: false
      arm: false
      hand: false
```

当前策略：

- 缺省行为保持当前脚本默认：机器人重力关闭。
- 只接受 `default` / `arm` / `hand` mapping，不再支持 `gravity: false` / `gravity: true` 简写。
- 动作脚本不再提供重力 CLI override；如需改变行为，修改 robot YAML。

执行策略：

- reset 前对 USD 子树中的 rigid body 写入 `disableGravity`。
- reset 后对 articulation runtime 做一致性处理。
- 如需区分 arm/hand，应基于 `component_for_name(...)` 判断 prim 或 DOF 所属部件。

## 6. 分阶段实施

### 阶段一：抽取 env runtime settings

改动：

- 新增 `src/linkerbot_sim/app/runtime_settings.py`。
- 替换 `pinch_grasp.py` 和 `dual_arm_motion_test.py` 中重复的 env 解析逻辑。
- 新增或扩展测试，覆盖默认值、CLI 覆盖、非法频率、缺失 `env`。

风险：低。该阶段不依赖 Isaac。

### 阶段二：迁移 cuMotion profile helper

改动：

- 新增 `src/linkerbot_sim/backends/cumotion/profile_config.py`。
- 迁移 `merged_robot_config_with_cumotion_profile(...)`、`robot_cumotion_config(...)`、`motion_planner_config_from_profile(...)`。
- 更新脚本 import。
- 更新测试 import。

风险：低到中。需要保证 profile merge 行为完全一致。

### 阶段三：增加 robot gravity policy

改动：

- 扩展 robot 配置解析，支持 `robot.physics.gravity`。
- 更新 robot example YAML，说明 `default` / `arm` / `hand` 重力策略。
- setup 层读取 robot gravity policy，并应用到 stage/runtime。
- 移除动作脚本中的重力 CLI override，避免 robot YAML 之外出现第二套入口。

风险：中。需要确认 USD stage 级 `disableGravity` 和 articulation runtime gravity 行为一致。

### 阶段四：抽取机器人执行 setup

改动：

- 新增 `src/linkerbot_sim/execution/setup.py`。
- 抽取 reset 前的机器人导入和 USD/PhysX 覆盖逻辑。
- 抽取 reset 后的 controller 初始化逻辑。
- `pinch_grasp.py` 使用单臂 setup。
- `dual_arm_motion_test.py` 左右两侧复用同一 setup。

风险：中。重点检查：

- `world.reset()` 仍在所有机器人导入完成之后。
- 单臂和双臂 gravity policy 行为保持一致。
- `mjcf_path` 只在 asset type 为 `mjcf` 时传给 controller 和 USD override。
- solver iteration 覆盖仍按 env config 生效。

### 阶段五：抽取 Simulation session

改动：

- 新增 `src/linkerbot_sim/app/simulation_session.py`。
- 替换两个脚本中的 app/world/stage 创建代码。
- 保留脚本层 `try/finally` 关闭 app。

风险：中。重点检查 Isaac import 时机和 GUI/headless 行为。

### 阶段六：整理 README 和文档

改动：

- 更新 README 的运行时装配层说明。
- 在本计划文档中标记完成状态，或新增实际抽取后的接口说明。

风险：低。

## 7. 不抽取内容

以下内容继续留在脚本中：

- `run_pinch_grasp_action(...)`。
- pinch grasp 的阶段顺序、目标偏置、lift/wiggle 参数、specified path 逻辑。
- `run_dual_arm_motion_sequence(...)`。
- 当前默认手型函数，以及后续“手的原子动作”抽象。
- 具体 demo 的输出前缀，例如 `RUN_PINCH_GRASP_*`、`DUAL_ARM_MOTION_*`。
- 任务级动作参数和调试流程。

## 8. 测试计划

阶段一到三建议运行：

```bash
env_isaaclab/bin/pytest tests/test_system_configs.py tests/test_controller_configs.py tests/test_dual_arm_motion_test.py -q
```

涉及 cuMotion profile 时额外运行：

```bash
env_isaaclab/bin/pytest tests/test_system_configs.py tests/test_dual_cumotion_urdf.py -q
```

阶段四到五完成后运行：

```bash
env_isaaclab/bin/pytest tests/test_system_configs.py tests/test_controller_configs.py tests/test_effort_logger.py tests/test_dual_arm_selectable_tcp.py tests/test_dual_cumotion_urdf.py tests/test_dual_arm_motion_test.py -q
```

可选 dry-run：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py --dry-run
```

可选 Isaac smoke：

```bash
PYTHONPATH=src env_isaaclab/bin/python scripts/pinch_grasp.py --no-grasp --short-smoke
PYTHONPATH=src env_isaaclab/bin/python scripts/dual_arm_motion_test.py --short-smoke
```

## 9. 验收标准

- `pinch_grasp.py` 主流程变短，主要保留 pinch grasp 任务编排。
- `dual_arm_motion_test.py` 不再复制 env/world/robot/controller 装配细节。
- 新增包内模块可被非 Isaac 单元测试 import。
- `world.reset()` 前后的职责边界清晰，没有把 controller 初始化提前到 reset 前。
- 机器人重力策略由 robot 配置驱动，不再通过脚本调试参数绕过。
- robot/env/controller/cumotion/logging 配置读取行为使用当前 schema，旧接口入口已清理。
- 现有轻量测试全部通过。

## 10. 建议提交拆分

建议按以下提交粒度推进，便于 review 和回退：

1. `refactor: add env runtime settings`
2. `refactor: move cumotion profile helpers`
3. `refactor: add robot gravity policy`
4. `refactor: extract robot execution setup`
5. `refactor: extract simulation session setup`
6. `docs: document runtime setup layering`
