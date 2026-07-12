# Python Facade 参考

语言：[中文](python-api.md) | [English](../../en/reference/python-api.md)

本文定义应用与算法代码可以依赖的进程内 Python 接口，面向明确在本仓库 checkout 内持有
Python 对象的调用方。跨进程客户端应使用 [Single Scene JSON 参考](single-scene-json.md)或
[Tiled Scene JSON 参考](tiled-scene-json.md)。

只有下文点名的 import path 与入口属于 Python 接口边界。其他模块存在、名称未以下划线开头，
或出现在 package 的 `__all__` 中，都不能单独构成接口承诺。下文标为 opaque 的具体返回类型
只用于传给另一个已记录的入口，调用方不能依赖其实现属性。

## 1. Checkout 与运行前提

本项目是 checkout application，不是可独立安装的 SDK。请在 Linux x86-64、Python 3.11
环境中从仓库根目录运行：

```bash
uv sync --all-extras
export PYTHONPATH=src
```

阅读并接受适用的 NVIDIA/Kit EULA 后，只对需要启动 Isaac Sim 的命令设置：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

本文的运行标签含义固定如下：

| 标签 | 前提 |
| --- | --- |
| `pure` | import 与调用不需要启动 Kit/Isaac。若契约明确写明文件操作，函数仍可能读写路径。 |
| `Isaac main thread` | 必须在拥有 `SimulationApp`、World、stage 和 articulation/PhysX view 的线程创建、读取、修改、step 与关闭对象。 |
| `cuRobo/CUDA` | 需要项目锁定的 cuRobo、Torch、Warp、可用的目标 CUDA device，并要求显式释放资源；不代表必须存在 Isaac stage。 |

本文所有 facade 本身都可以在普通项目 Python 环境中 import。import 时不会加载 `omni`、
`isaacsim`、`pxr`、`torch` 或 `curobo`；这些依赖只在下文标明的调用边界加载。因此可在
Kit 启动前执行以下 import smoke test：

```python
from linkerbot_sim import REPO_ROOT
from linkerbot_sim.app.interactive.single_scene import run_single_scene_interactive_motion
from linkerbot_sim.app.interactive.tiled_scene import TiledSceneRuntime
from linkerbot_sim.backends.curobo import CuroboConfig
from linkerbot_sim.controllers import JointController
from linkerbot_sim.execution import ExecutionRuntime
from linkerbot_sim.objects import ObjectProfileConfig
from linkerbot_sim.planning import MotionRequest
from linkerbot_sim.robots import JointGroupLayout
from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.snapshots import SimulationSnapshot
```

import 成功不会放宽调用标签。例如，上述两个 runtime 符号的解析很轻量，但构造或运行它们
仍是 `Isaac main thread` 操作。

通用数值约定为米、秒、弧度、弧度每秒，公共四元数顺序为 `wxyz`。关节数组顺序必须由
显式 `joint_names`、command-joint 列表或 articulation `dof_names` 确定，不能从资产文件
隐式推断。

## 2. Facade 总览

| Import path | 标签 | 所有权 |
| --- | --- | --- |
| `linkerbot_sim` | `pure` | 只提供仓库根路径 |
| `linkerbot_sim.planning` | `pure` | 后端无关 request、result、frame、collision DTO 与 linear backend |
| `linkerbot_sim.backends.curobo` | 混合 `pure` / `cuRobo/CUDA` | cuRobo config、context、solver、mapping、collision world 与 adapter |
| `linkerbot_sim.controllers` | 混合 `pure` / `Isaac main thread` | Controller setting、完整 DOF target 与 articulation 控制 |
| `linkerbot_sim.execution` | 执行时为 `Isaac main thread` | 在已有 World 上执行 command-space step |
| `linkerbot_sim.objects` | 混合 `pure` / `Isaac main thread` | Object profile 解析与 stage 插入 |
| `linkerbot_sim.robots` | `pure` | Robot kind、joint group 与 planning capability 诊断 |
| `linkerbot_sim.sensors` | `pure` | Camera 配置选择，不采集 frame |
| `linkerbot_sim.snapshots` | 混合 `pure` / `Isaac main thread` | Snapshot 数据、兼容性、采集、恢复与 clone |
| `linkerbot_sim.app.interactive.single_scene` | `Isaac main thread` | Single Scene CLI 与 canonical interactive loop |
| `linkerbot_sim.app.interactive.tiled_scene` | 混合 | Tiled Scene runtime、protocol 解析/分派与 transport 所有权 |

## 3. 仓库根路径

`from linkerbot_sim import REPO_ROOT` 是完整的顶层 facade。`REPO_ROOT` 是根据源码位置推导的
绝对 `pathlib.Path`，不依赖进程当前目录，也不保证任一子路径一定存在。各 domain 名称应从
对应 facade import，不能依赖 `linkerbot_sim` 的传递导入。

## 4. 后端无关 Planning

从 `linkerbot_sim.planning` import。本节所有入口都是 `pure`。只有实现明确说明时 dataclass
才会复制或规整数组；否则应把 NumPy 输入视为调用方持有。

### 4.1 Protocol 与名称

| 入口 | 签名与契约 |
| --- | --- |
| `PlannerBackendName` | `Literal["curobo", "linear"]`。 |
| `PlanningRequest` | `MotionRequest | LinearPosePathRequest`。 |
| `PlannerBackend` | Runtime-checkable protocol：`joint_names() -> Sequence[str]` 与 `plan(request: PlanningRequest) -> MotionResult`。 |
| `normalize_planner_backend` | `(value: object) -> PlannerBackendName`；去除首尾空白并转小写，空值取 `curobo`，其他名称抛 `ValueError`。 |
| `BatchIKBackend` | Protocol 方法 `solve(*, target_positions, target_orientations_wxyz, seeds, tcp_frame_name) -> BatchIKResult`；行表示 env，所有数组的 `N` 必须相同。 |
| `OrientationMode` | `Literal["free", "current", "target"]`；控制线性 TCP 段忽略姿态、保持起点姿态或到达目标四元数。 |

### 4.2 Request 与几何

| 入口 | 构造函数 / 公共方法 | Shape、单位、frame 与拒绝条件 |
| --- | --- | --- |
| `PoseTarget` | `(position, orientation=None)` | Position `(3,)`，单位 m，位于调用方与后端约定的 task frame；可选 orientation `(4,)`、`wxyz`。由包含它的 request 执行校验。 |
| `IKRequest` | `(target_position, target_orientation=None, tcp_frame_name=None, warm_start_ik_cspace_seed=None, position_tolerance=None, orientation_tolerance=None, avoid_collisions=False)`；`validate_structure()` | Target `(3,)` m、四元数 `(4,)` `wxyz`、seed `(C,)` 且使用后端 C-space 顺序，容差非负（m/rad）。空 frame、非有限值、宽度或容差错误抛 `ValueError`；目标不可达由 result 表达。 |
| `MotionRequest` | `(current_q, goal_q=None, goal_pose=None, tcp_frame_name=None, duration_s=None, sample_dt_s=None, avoid_collisions=False)`；`validate_structure()` | `current_q`/`goal_q` 为 `(C,)`，顺序由 `backend.joint_names()` 给出，转动关节单位 rad。`goal_q` 与 `goal_pose` 必须且只能给一个。Duration 有限且非负，采样间隔有限且为正；结构错误抛 `ValueError`。 |
| `TcpLineSegment` | `(start_position=None, target_position=None, target_offset=None, orientation_mode="free", target_orientation=None)` | 向量 `(3,)`、单位 m、使用 request task frame。两种终点形式必须且只能给一个。`target` 要求 `(4,)` `wxyz`，其他 mode 禁止携带该四元数。显式 start 在 cuRobo 采样时必须与上一终点一致。 |
| `TcpPoseSequenceSegment` | `(poses: tuple[PoseTarget, ...], blend_radius=0.0)` | 至少一个带完整姿态的 pose。当前线性路径要求 `blend_radius == 0`；有限负值或正值都会被 request 校验拒绝。 |
| `TaskSpacePath` | `(segments: tuple[...])` | 非空、有序的 `TcpLineSegment` / `TcpPoseSequenceSegment` 序列。 |
| `LinearPosePathRequest` | `(current_q, path, tcp_frame_name=None, duration_s=None, sample_dt_s=None, avoid_collisions=False)`；`validate_structure()` | `current_q` 为 `(C,)`；本 facade 的 task position 使用 cuRobo robot-base planning frame。进入 cuRobo 采样前必须提供 `sample_dt_s`，通常注入 physics dt。结构错误抛 `ValueError`。 |
| `CollisionObject` | `(name, shape, pose, size, enabled=True, padding=0.0)`；`pose_matrix()`、`padded_size()` | `pose` 为齐次 `(4,4)`。`cuboid` size 是 `(x,y,z)`，`sphere` 是 `(radius,)`，`capsule` 是 `(radius,length)`，单位 m。Padding 增长 cuboid 两侧或半径。后端转换会拒绝不支持或非正几何。 |
| `FrameTransformer` | `(world_from_robot_base, world_from_env, robot_base_from_tcp=None)`；`from_root_pose(...)`、`pose_to_robot_base(...)`、`offset_to_robot_base(...)` | Transform 为齐次 `(4,4)`，命名遵循 `target_from_source`。Frame 为 `world`、`env`、`robot_base`、`tcp`；offset 只旋转不平移。缺少 TCP pose、未知 frame 或齐次矩阵末行错误抛 `ValueError`。 |
| `PoseInRobotBase` | `(position, orientation_wxyz)` | Frame 转换结果：robot-base 坐标中的 `(3,)` m position 与可选 `(4,)` `wxyz` orientation。 |

### 4.3 Result 与可执行 Linear Planning

| 入口 | 签名与契约 |
| --- | --- |
| `BatchIKResult` | `(joint_positions, success, position_error, orientation_error=None, status=())`；强制 `(N,C)`、`(N,)`、`(N,)`、可选 `(N,)` 和 `N` 个 status string。维度错误抛 `ValueError`。Position error 单位 m，orientation error 使用所选后端的度量。 |
| `PlanningDiagnostics` | `(status="", message="", metrics={})`；metrics 应是少量可打印数值，不能放 solver 或 tensor 对象。 |
| `IKResult` | `(joint_positions, success, position_error, orientation_error=None, message="", status="", num_solutions=1)`；关节向量使用后端顺序，消费前必须检查 `success`。 |
| `MotionResult` | `(path, trajectory, success, status, diagnostics=...)`；存在时 `path` 为 `(T,C)` 且使用后端顺序，`trajectory` 通常是项目 `JointTrajectory`。消费前必须检查 `success`。 |
| `LinearPlannerBackend` | `(joint_names, *, default_duration_s=1.0, default_sample_dt_s=None)`；`joint_names()`、`plan(request)` | 只生成确定性 joint interpolation。名称必须非空且唯一。接受 joint-goal `MotionRequest`，并要求 request 或构造函数提供正采样间隔；task-space 或 collision-aware request 返回失败 result。宽度/default 错误抛 `ValueError`。它不做 IK、collision、joint limit、velocity limit 或 acceleration limit 求解。 |

目标不可达或碰撞能力不足等 planning 失败通常返回 `success=False`；shape、名称或配置契约错误
抛 `ValueError`。后端选择与 collision 行为见
[cuRobo 使用与批量调度](../guides/motion-planning.md)。

## 5. cuRobo 后端

从 `linkerbot_sim.backends.curobo` import。Facade import 是 `pure`；构造 `CuroboContext`
或调用真实 solver 是 `cuRobo/CUDA`。除非 `CuroboJointMapping` 明确转换，所有关节数组都使用
context active C-space 顺序。

### 5.1 配置与 Profile（`pure`）

| 入口 | 构造函数 / 公共方法 | 契约 |
| --- | --- | --- |
| `SUPPORTED_CUROBO_DTYPES` | `frozenset({"float32"})` | 当前配置 parser 接受的精确 dtype 名。 |
| `CuroboTaskBundle` | 经过验证的 bundle DTO；使用 `CuroboTaskBundle.named(value)` 与 `validate_curobo_version(value)` | 只接受项目拥有的命名 task 文件集合。不支持的 bundle 抛 `ValueError`，不匹配的已安装 cuRobo release 抛 `RuntimeError`；不能自行组合 raw task path。 |
| `CuroboTcpFrame` | `(frame_name, parent_frame, xyz, rpy)`；`from_mapping(data, *, default_parent_frame, label)` | 相对 parent 的 fixed TCP；`xyz (3,)` 单位 m，`rpy (3,)` 单位 rad。名称必须非空，向量必须有限。 |
| `CuroboDeviceConfig` | `(device="cuda:0", tensor_dtype="float32", collision_geometry_dtype="float32", collision_gradient_dtype="float32", collision_distance_dtype="float32")`；`from_mapping(data)` | Parser 拒绝未知字段和 `SUPPORTED_CUROBO_DTYPES` 之外的 dtype；直到 context 构造时才探测 CUDA。 |
| `CuroboRobotConfig` | `(robot_config_path=None, urdf_path=None, base_link=None, flange_frame=None, tool_frames=(), default_tcp_frame=None, custom_tcp_frames=(), load_collision_spheres=True)`；`from_mapping`、`validate`、`resolved_tool_frames` | Robot YAML 与 URDF 至少提供一个。路径相对 checkout 解析；frame 唯一性与静态结构在 context 前校验，model membership 在 materialize/context 时校验。 |
| `CuroboIkConfig` | IK seed、tolerance、optimizer、regularization weight、`max_batch_size`、`multi_env`、`max_goalset`、self-collision 与 collision-cache capacity；`from_mapping`、`validate` | 缺省包含 32 个 seed、`0.002` m position tolerance、`0.01` rad orientation tolerance、batch 256、启用 CUDA graph。类型、有限范围、正容量与 key 均严格校验。 |
| `CuroboMotionPlannerConfig` | Planner warmup、IK/trajopt seed、tolerance、CUDA graph、batch/goal limit、self-collision 与 cache capacity；`from_mapping`、`validate` | 缺省 32 个 IK seed、4 个 trajopt seed、batch 256，并使用同一 tolerance；严格校验数值和容量。 |
| `CuroboConfig` | `(robot, task_bundle=..., device=..., ik=..., motion_planner=...)`；`from_mapping(data)`、`validate()` | 只在 cuRobo 已启用且使用受支持 arm planning group 时解析完整后端 mapping。未知字段及不一致 robot/TCP/capacity 设置抛 `ValueError`。 |
| `validate_curobo_profile` | `(data, *, source="<curobo profile>") -> dict` | 严格校验项目算法 profile，并返回独立顶层 dict；错误包含 `source`。 |
| `load_curobo_profile` | `(path) -> dict` | 读取 YAML 后执行同一严格校验；文件/YAML 错误原样传播。 |
| `merged_robot_config_with_curobo_profile` | `(robot_config, curobo_profile, *, profile_source=...) -> dict` | 返回新 deep merge，robot config 优先；不修改输入。 |
| `robot_curobo_config` | `(robot_config, *, curobo_profile=None, robot_source=..., curobo_profile_source=...) -> CuroboConfig` | 执行已验证 merge 并解析真实后端 config；`ValueError` 同时标明两个来源。 |
| `resolve_curobo_cache_dir` | `(cache_root=None, *, environ=None) -> Path` | 优先级为显式 root、`LINKERBOT_SIM_CACHE_ROOT`、`XDG_CACHE_HOME/linkerbot_sim`、`~/.cache/linkerbot_sim`；追加 `curobo` 并返回绝对路径，不创建目录。 |
| `materialize_curobo_config` | `(config, *, cache_root=None) -> CuroboConfig` | 存在 custom TCP 时，在 cache 下原子写入 content-addressed URDF 并返回替换后的不可变 config。要求 source URDF；XML、权限、校验与 I/O 错误传播。没有 custom TCP 时返回同一 config。 |
| `default_tcp_frame_name` | `(config) -> str | None` | 显式 default TCP，否则第一个 resolved tool frame。 |
| `resolve_tcp_frame_name` | `(context, *, tcp_frame_name=None, default_tcp_frame_name=None, label="tcp_frame_name") -> str` | 依次使用显式参数、调用方 default、context config；缺少、空或未知 frame 抛 `ValueError`。配合 duck-typed context 时该 helper 为 `pure`。 |

完整 YAML 所有权仍属于[配置参考](configuration.md)；这些 class 定义 Python 表示，不重复拥有
YAML 字段表。

### 5.2 Context、Solver 与 Collision（`cuRobo/CUDA`）

| 入口 | 构造函数 / 公共方法 | 资源、shape 与 result 契约 |
| --- | --- | --- |
| `import_curobo_module` | `() -> module` | 完成 Warp 检查后加载锁定的 `curobo` package。缺少 cuRobo 或传递依赖会转换成可执行的 `RuntimeError`；不持有 context。 |
| `CuroboContext` | `(config, *, cache_root=None)` | 探测配置的 Torch/CUDA、materialize TCP、加载 kinematics，并按需创建 `ik_solver`、`motion_planner`、`batch_motion_planner`。`joint_names()`/`frame_names()` 定义数组/frame 顺序。`compute_tcp_poses((N,C), tcp_frame_name=...) -> ((N,3),(N,4))` 返回 base-frame m 与 `wxyz`。`sync_collision_world(collision_objects=())` 替换 context 的当前 world snapshot 并更新已创建的 solver；`clear_collision_world()` 执行对应的空 world 同步。Capability 与组件 factory 使用同一 context。必须调用 `close()`；关闭某 solver 失败时仍保留其所有权，可重试 close。 |
| `CollisionCapability` | DTO 字段描述 robot sphere、scene checker、cache type、required/configured capacity 与 scene 同步；`available`、`missing_requirements` 是 property。 | 只有 robot model、checker、capacity、同步 scene 与 materialized fingerprint 全部存在时 `available` 才为 true。 |
| `CuroboCollisionWorld` | `(context, collision_objects=())`；`sync`、`update_solvers` 与计数 property | 重建完整 cuRobo scene snapshot，并更新已创建的 solver。Context 持有当前 world；调用方不能修改 `scene_cfg`。Shape/dimension 或 cache 超量抛 `ValueError`。 |
| `make_curobo_scene_cfg` | `(context, collision_objects) -> cuRobo Scene` | Enabled cuboid 直接转换，sphere/capsule 转成保守 cuboid。Pose 是 `[x,y,z,qw,qx,qy,qz]`，尺寸单位 m，并校验配置 cache capacity。返回的第三方对象是 opaque。 |
| `CuroboJointMapping` | 使用 `from_joint_names(*, cspace_joint_names, command_joint_names)`；property `cspace_width`、`command_width`；`command_to_cspace`、`cspace_to_command` | 按精确名称转换 `(N,D)`。缺名、宽度或行数错误抛 `ValueError`；非 C-space command 列从 `base_command_positions` 复制。该 class 自身为 `pure`。 |
| `CuroboForwardKinematics` | `(context)`；`joint_names`、`frame_names`、`compute_pose(q, frame)`、`compute_position`、`compute_orientation` | `q` 为 `(C,)`；输出 base-frame `(3,)` m position、`(4,)` `wxyz` orientation 与 `(3,3)` rotation。完整 pose 是带这三个命名属性的返回值。未知 frame/后端错误传播。 |
| `CuroboInverseKinematics` | `(context, *, tcp_frame_name=None)`；`joint_names`、`frame_names`、`solve(IKRequest) -> IKResult` | 创建 lazy IK solver。禁止 per-request tolerance override，应在 profile 配置。Collision capability 不足返回 `status="COLLISION_UNSUPPORTED"`；普通求解失败返回 `success=False`。Frame/seed width 错误抛 `ValueError`。 |
| `CuroboBatchIKSolver` | `(context, *, tcp_frame_name=None, command_joint_names=None)`；`solve(...)`、`compute_tcp_poses(...)` | Target 为 `(N,3)` m，可选 orientation 为 `(1,4)` 或 `(N,4)` `wxyz`；seed 为 `(N,C)`，有 mapping 时也可为 `(N,D)`。返回 `BatchIKResult`；失败行保留 seed position 并标记 `success=False`。宽度/frame/context 缺陷抛 `ValueError` 或 `RuntimeError`。 |
| `CuroboMotionPlanner` | `(context, *, tcp_frame_name=None)`；`planner`、`joint_names`、`plan(request)`、`close()` | 按需创建 scalar planner。Joint/pose goal 与 linear pose path 返回 `MotionResult`；请求的 collision 不可用时返回失败 result。Path 为 `(T,C)` rad，trajectory 时间单位 s。`close()` 会关闭共享 context，不能与仍需运行的其他 owner 共用。 |
| `plan_linear_pose_path` | `(context, request, *, tcp_frame_name) -> MotionResult` | 按 `sample_dt_s` 采样每段，position 以 m 线性插值、`wxyz` 以 Slerp 插值，再以此前解作 seed 顺序求 IK。无效/不支持路径、collision 不足或任一样本失败均返回失败 result。Frame 是 cuRobo robot-base local。 |

### 5.3 Trajectory Adapter（对 result-like 输入为 `pure`）

`joint_trajectory_from_curobo` 的签名是
`joint_trajectory_from_curobo(result_or_trajectory, *, joint_names,
sample_dt=None, phase="trajectory") -> JointTrajectory` 读取 cuRobo-like interpolated
trajectory、trajectory 或 JointState。Position 必须可化为单条 `(T,C)` trajectory，且
`len(joint_names) == C`；有显式时间时直接使用，否则必须给正 `sample_dt`。Velocity 与
acceleration 保持同 shape。缺少属性、存在 batch 维、宽度/时间/有限性错误抛 `ValueError`
或 `AttributeError`。函数自身不分配 CUDA 资源。

## 6. Controller

从 `linkerbot_sim.controllers` import。Facade import 与 setting/target 类型为 `pure`；
`JointController` 是 `Isaac main thread` 集成。

| 入口 | 签名与契约 |
| --- | --- |
| `ControlMode` | `Literal["position", "velocity", "effort"]`。 |
| `ControlMethod` | `Literal["implicit", "explicit", "direct"]`；支持 position/velocity 搭配 implicit 或 explicit，effort 搭配 direct。 |
| `ComponentControlSettings` | `(mode="position", method="implicit", stiffness=(1000,), damping=(50,), max_force=100, effort_limit=None, joint_friction=0.5, follower_stiffness=(50000,), follower_damping=(50,), follower_max_force=None, follower_joint_friction=None)`。每个 joint parameter 可为 scalar、精确长度 sequence 或精确 name map。`active_effort_limit(s)` 解析 limit；名称/长度/有限性错误抛 `ValueError`。 |
| `JointControlSettings` | `(default=..., arm=None, hand=None)`；`component(name, *, component=None)` 返回 arm/hand setting 或 `default`。 |
| `ControlTargets` | `(positions, velocities, efforts)`；构造相同 shape `(D,)` 的独立有限一维副本。单位是 articulation 原生单位：revolute 为 rad/rad/s 与 PhysX effort，prismatic 为 m/m/s 与 force。Shape 或有限性错误抛 `ValueError`。 |
| `JointController` | `(robot, *, joint_names, settings, mimic_path=None, component_mapping=None, native_mimic=False)`，在 articulation finalize 后构造。暴露 `command_indices`、`follower_indices`、`driven_indices`、`command_joint_names`、`runtime_follower_indices`。`configure_runtime()` 写 mode/gain/limit；`build_control_targets(command_*, base_positions=None)` 把 command `(C,)` 转成完整 `(D,)`；`targets_from_full_state` 校验 `(D,)`；`apply_targets(ArticulationAction, targets)` 分组下发 action。缺少 joint/file、宽度/setting 错误、controller capability 缺失或非有限 target 抛 `ValueError`、`RuntimeError` 或文件/XML 错误。 |

第一次 action 前必须调用 `configure_runtime()`。Command space 不包含 mimic follower；
Python 驱动的 follower 每帧根据 master 实际状态重算，native URDF mimic follower 不会收到重复
action。Controller 借用 articulation，不负责关闭。

## 7. Command Execution

从 `linkerbot_sim.execution` import。DTO 构造为 `pure`；所有 `run` 与 `execute_*` 调用均为
`Isaac main thread`，并借用全部资源。

| 入口 | 签名与契约 |
| --- | --- |
| `ExecutionRuntime` | `(articulation, simulation_world, articulation_action_type, joint_controller, simulation_app, render_enabled, drive_logger=None, state_observer=None, camera_observer=None)`。它是不拥有资源的 bundle；调用方关闭 World/app/logger/observer。 |
| `ExecutionStep` | Protocol，包含 `phase: str` 与 `run(runtime, step: int) -> int`；返回累计已完成 physics step。 |
| `SmoothCommandPositionTargetStep` | `(start_command, target_command, duration, phase, base_positions=None, should_stop=None)`；command array 为 `(C,)`，使用 controller command 顺序。Smoothstep duration 按 physics dt 量化且至少一步。 |
| `CommandPositionTrajectoryStep` | `(trajectory, should_stop=None)`；trajectory 已按每个 physics step 一行采样、没有初始样本，并使用 controller command-joint 列；本层不再次采样。 |
| `HoldCommandPositionTargetStep` | `(target_command, duration, phase, base_positions=None, should_stop=None)`；正 duration 量化为 step，非正 duration 运行到 app 退出或取消。没有 app 时必须提供能终止的 callback。 |
| `SwitchControlModeStep` | `(settings, phase="switch_control_mode")`；更新 controller setting/config，不推进 physics，并返回输入 step。 |

函数形式分别为：

- `execute_smooth_command_position_target`：
  `execute_smooth_command_position_target(*, articulation, simulation_world,
articulation_action_type, joint_controller, start_command, target_command,
duration, phase, simulation_app, render_enabled, step, base_positions=None,
should_stop=None, drive_logger=None, state_observer=None,
camera_observer=None) -> int`。
- `execute_command_position_trajectory`：
  `execute_command_position_trajectory(*, articulation, simulation_world,
articulation_action_type, joint_controller, trajectory, simulation_app,
render_enabled, step=0, should_stop=None, drive_logger=None,
state_observer=None, camera_observer=None, hold=False) -> int`。
- `execute_command_position_hold`：
  `execute_command_position_hold(*, articulation, simulation_world,
articulation_action_type, joint_controller, target_command, duration, phase,
simulation_app, render_enabled, step, base_positions=None, should_stop=None,
drive_logger=None, state_observer=None, camera_observer=None) -> int`。

取消只在 physics step 之间检查，并抛带 `.step` 的 `RuntimeError` subclass。如果 World 已成功
step、但 observer/logger 失败，也会抛带已完成 `.step` 的 `RuntimeError` subclass；必须从该值
继续，不能重放样本。其他 action、shape、World 与 observer 错误传播。

## 8. Object

从 `linkerbot_sim.objects` import。Parser 与 DTO 为 `pure`；接收 USD `stage` 的函数都是
`Isaac main thread`。

| 入口 | 签名与契约 |
| --- | --- |
| `ObjectSceneInstanceConfig` | `(name, object_profile, root_pose, runtime_handle=None, prim_path=None)`；使用 `from_mapping(data, *, index)` 和只读 property `default_prim_path` / `effective_prim_path`。名称/handle 必须非空，prim path 必须绝对；placement 通过 `RootPoseConfig` 使用 m/rad。 |
| `ObjectProfileConfig` | `(profile_name, name, kind, source, asset_path, ...)`；优先使用 `from_profile(name)` 或 `from_mapping(data, *, profile_name, source=None)`，表示一个严格校验的 rigid 或 dynamic-chain profile。 |
| `ObjectMaterialConfig` | `(static_friction=None, dynamic_friction=None, restitution=None, friction_combine_mode=None)`；`from_mapping(..., label=...)`、`has_overrides()`。系数有限且非负，restitution 不大于 1，combine mode 使用受支持的 PhysX 名称。 |
| `CapsuleRopeConfig` | `(asset_path=..., prim_path="/World/CapsuleRope", root_path="/CapsuleRope", physics=...)`；`from_mapping`、`asset_file`、`validate`。嵌套 physics 类型没有独立 facade 入口，应优先从 profile 解析。 |
| `RigidObjectPlanningCollisionConfig` | `(shape, size, xyz=(0,0,0), rpy=(0,0,0), enabled=True, padding=0)`；只描述简化 planning geometry。Size/offset/padding 单位 m，rotation 单位 rad；shape 为 cuboid/sphere/capsule。 |
| `RigidObjectPhysicsConfig` | `(static=False, material=None)`；描述 runtime PhysX override。 |
| `RigidObjectConfig` | `(name, asset_type, asset_path, prim_path, root_pose=..., physics=..., planning_collision=None, urdf_drive_type="none", import_config=...)`；优先用 `rigid_objects_from_env_config`，以一致校验依赖的 importer 类型。 |
| `AddedRigidObject` | `(name, asset_type, asset_path, prim_path, imported_path, static)`；stage 插入完成后的不可变摘要。 |
| `validate_object_profile` | `(data, *, source="<object profile>", profile_name="object") -> dict`；严格校验当前 profile，`ValueError` 包含来源。 |
| `load_object_profile` | `(path) -> dict`；读取并校验 YAML。 |
| `object_scene_instances_from_env_config` | `(env_config) -> tuple[ObjectSceneInstanceConfig, ...]`；校验 scene instance identity 与唯一性。 |
| `expanded_object_mapping` | `(instance, profile=None) -> dict`；把 placement/identity 与引用 profile 合成新 mapping；省略时加载 profile。 |
| `rigid_objects_from_env_config` | `(config) -> tuple[RigidObjectConfig, ...]`；只展开 `kind=rigid` instance。 |
| `add_capsule_rope_reference` | `(stage, config) -> dict[str, object]`；引用已有 USD 并返回命名 prim handle。缺少 asset/prim 与 USD 错误传播。 |
| `apply_capsule_rope_runtime_physics` | `(stage, config) -> {"collision_prims": int, "rigid_bodies": int}`；对已引用 rope 应用 material/solver override。 |
| `add_rigid_objects` | `(stage, objects) -> tuple[AddedRigidObject, ...]`；引用 USD 或调用 Isaac URDF importer，应用 pose/physics，并拒绝缺失资产、不支持类型、target 冲突或无效 stage 状态。 |

Object runtime 函数不生成资产。离线 builder 见[对象资产](../development/object-assets.md)；
PhysX 与 planning geometry 的区别见[碰撞模型](../guides/collision-models.md)。

## 9. Robot 元数据与 Capability

从 `linkerbot_sim.robots` import。所有入口都是 `pure`；articulation 名称可以作为普通 sequence
传入。

| 入口 | 签名与契约 |
| --- | --- |
| `RobotKind` | String enum：`ARM="arm"`、`HAND="hand"`、`ARM_HAND="arm_hand"`；`RobotKind.parse(value)`、`has_arm`、`has_hand`。非法值抛 `ValueError`。 |
| `robot_kind_from_profile` | `(profile) -> RobotKind`；要求 canonical 顶层 robot mapping 与 `robot.kind`。 |
| `PlanningBindingConfig` | `(enabled, planning_joint_group, has_robot_model)`；使用 `from_profile(profile, *, kind)`。启用 planning 要求 arm group 与非空 model；hand-only robot 不能启用。 |
| `PlanningCapability` | `(kind, backend_enabled, planning_joint_group, kinematics_binding_valid, arm_joint_mapping_valid, reasons=())`；property `supports_planning`，方法 `require(operation="planning")`。`require` 抛可诊断 `RuntimeError`；该 capability 不证明 scene-collision 可用。 |
| `JointGroup` | `(name, joint_names)`；`from_mapping(name, data)`、`indices_in(dof_names, *, allow_all=True) -> int ndarray`。保持精确名称匹配与顺序；缺名或非法输入抛 `ValueError`。 |
| `JointGroupLayout` | `(command_joint_names, arm=(), hand=(), passive=())`；`resolve(*, kind, command_joint_names, joint_groups, planning_joint_names=())`、`validate_kind`、`validate_planning_joints`、`names(group)`、`indices(group)`。Group 必须唯一、互不重叠、覆盖全部 command joint，并符合 robot kind；planning joint 集合必须等于 arm。 |

## 10. Sensor 配置

从 `linkerbot_sim.sensors` import `SceneSensorSettings`（`pure`）：

`SceneSensorSettings(cameras=())` 聚合解析后的 camera setting。
`from_env_config(config)` 严格解析 `sensors.cameras`；`enabled_cameras` 筛选启用项；
`has_output_consumers` 报告是否存在已启用 file/Foxglove sink；
`validate_single_scene_camera_scope()` 在 Single Scene 模式拒绝 Tiled Scene `env_ids` selector。配置错误抛
`ValueError`。

该 facade 不创建 camera、不采集 frame、也不发布数据。Camera 构建/采样要求
`Isaac main thread` 与启用 render 的 World step。文件输出使用 runtime camera-output policy。
Foxglove live 或 MCAP 还要求 `uv sync --all-extras`（`foxglove-sdk` visualization extra）、
已配置 sink，且内置 live server 只能绑定 loopback。RGB 与 depth 发布 RawImage；segmentation
modality 只在本地保存 NumPy array，并只发布 metadata。详见
[Camera 类型与传感器](../guides/cameras.md)、[Foxglove](../guides/foxglove.md)与
[输出与持久化](outputs.md)。

## 11. Snapshot

从 `linkerbot_sim.snapshots` import。数据与 compatibility 对象为 `pure`。Runtime capture、
descriptor 构建、restore、dispatch 与 clone 会读取或修改 runtime/PhysX 状态，因此都是
`Isaac main thread`。

### 11.1 数据与 Compatibility（`pure`）

| 入口 | 签名与契约 |
| --- | --- |
| `SNAPSHOT_SCHEMA` | 精确 discriminator string `linkerbot.snapshot`。 |
| `SnapshotMetadata` | `(source_runtime="", source_env_id=None, step=None, time_s=None, coordinate_frame="local", info={})`；`from_mapping`、`as_dict`。可选数值必须有限；Tiled Scene capture 使用 `env-local`，Single Scene 使用 `scene-local`。 |
| `RobotSnapshot` | `(label, robot_id, joint_names, joint_positions, joint_velocities, robot_profile=None, asset_fingerprint=None, command_joint_names=(), command_targets=None)`；`from_mapping`、`as_dict`。State array 为有限 `(J,)`，target 为 `(C,)`；revolute 为 rad/rad/s，prismatic 为 m/m/s。名称定义顺序且必须唯一。 |
| `ObjectSnapshot` | `(name, positions_local, orientations_wxyz, object_profile=None, linear_velocities=None, angular_velocities=None, body_names=(), body_*=None)`；`from_mapping`、`as_dict`。Root 为 `(3,)` m 与归一化 `(4,)` `wxyz`；velocity 为 `(3,)` m/s、rad/s。Body matrix 为 `(B,3)` / `(B,4)`，存在 body name 时必填。 |
| `SimulationSnapshot` | `(robots, objects={}, metadata=..., schema=SNAPSHOT_SCHEMA)`；`from_mapping`、`as_dict`。Mapping key 必须等于稳定 label/name，robot ID 唯一。Discriminator、未知顶层/robot 字段、shape、名称或有限性错误抛 `ValueError`。 |
| `SnapshotRestoreResult` | `(accepted, event="snapshot_restored", robots=(), objects=(), env_ids=(), partial=False, message="")`；`as_dict()`。`partial` 是 snapshot robot/object entry 被省略的 entry-level 指示，不编码更细粒度的 compatibility 语义。 |
| `RobotTargetDescriptor` | `(label, joint_names, robot_profile=None, asset_fingerprint=None, command_joint_names=())`；不可变 matching 输入。 |
| `ObjectTargetDescriptor` | `(name, object_profile=None, body_names=())`；不可变 matching 输入。 |
| `SnapshotTargetDescriptor` | `(runtime_kind, robots, objects={})`；描述目标 identity，不携带动态状态；runtime adapter 使用 `single_scene` 或 `tiled_scene`。 |
| `JointMapping` | `(source_indices, target_indices, names)`；成对一维 integer mapping，长度必须一致。 |
| `RobotCompatibilityMapping` | `(source_label, target_label, joints, command_joints=None)`。 |
| `ObjectCompatibilityMapping` | `(source_name, target_name, bodies=None)`。 |
| `SnapshotCompatibilityResult` | `(compatible, issues, robot_mappings={}, object_mappings={}, partial=False)`；`partial` 与上文具有相同的 entry-level 含义。 |
| `SnapshotCompatibilityError` | Required compatibility check/restore 抛出的 `ValueError` subclass。 |
| `check_snapshot_compatibility` | `(snapshot, target, *, label_map=None, strict=True) -> SnapshotCompatibilityResult`；计算 mapping，不写 runtime。 |
| `require_snapshot_compatibility` | 参数/返回值相同；不兼容时抛 `SnapshotCompatibilityError`。 |

### 11.2 Runtime Adapter（`Isaac main thread`）

| 入口 | 签名与契约 |
| --- | --- |
| `single_scene_target_descriptor` / `tiled_scene_target_descriptor` | `(runtime) -> SnapshotTargetDescriptor`；读取稳定 profile、fingerprint、joint/body name，不读动态状态。 |
| `get_single_scene_snapshot` | `(runtime) -> SimulationSnapshot`；采集包含任意数量 robot 的完整 Single Scene。 |
| `get_tiled_scene_snapshot` | `(runtime, *, env_id) -> SimulationSnapshot`；只采集一个 env，并移除 batch 行。非法 env 抛 `ValueError`/`IndexError`。 |
| `get_snapshot` | `(runtime, *, env_id=None) -> SimulationSnapshot`；按 canonical runtime shape 分派，Tiled Scene 必须给 `env_id`；未知 runtime 抛 `ValueError`。 |
| `set_single_scene_snapshot` | `(runtime, snapshot_or_mapping, *, label_map=None, strict=True) -> SnapshotRestoreResult`。 |
| `set_tiled_scene_snapshot` | `(runtime, snapshot_or_mapping, *, env_ids, label_map=None, strict=True) -> SnapshotRestoreResult`；把一个逻辑 snapshot 广播到非空、无重复、范围内的 env selection。 |
| `set_snapshot` | `(runtime, snapshot_or_mapping, *, env_ids=None, label_map=None, strict=True) -> SnapshotRestoreResult`；执行分派，Tiled Scene 必须给 `env_ids`。 |
| `clone_tiled_env_state` | `(runtime, *, source_env_id, target_env_ids, strict=True) -> SnapshotRestoreResult`；通过同一 adapter 采集 source，再恢复每个 target。 |

完整 payload 字段、strict/non-strict matching、`partial` 解释、restore transaction 与异常由
[Snapshot 数据与恢复参考](snapshots.md)统一定义。

## 12. Single Scene Interactive Runtime

从 `linkerbot_sim.app.interactive.single_scene` import。两个 callable 入口在调用时都是
`Isaac main thread`。

| 入口 | 签名与生命周期 |
| --- | --- |
| `main` | `(argv: Sequence[str] | None = None) -> None`；解析所选 Single Scene runtime profile，支持在 Kit 启动前输出 effective config，构造 runtime、运行 loop，并在 `finally` 关闭；状态写到 stdout。子进程使用优先调用 script entrypoint。 |
| `run_single_scene_interactive_motion` | `(runtime, *, stdin_enabled=True, tcp_jsonl_host=None, tcp_jsonl_port=None, websocket_host=None, websocket_port=None, state_stream_config=None, start_step=0, planner_backend="curobo", policy=None, interactive_settings=None, execution_settings=None, planner_settings=None, shutdown_settings=None) -> int`；运行 canonical Single Scene queue/timeline loop，返回累计 global step。 |

`run_single_scene_interactive_motion` 的调用方持有已创建 `SingleSceneRuntime`，即使 loop 报错也必须随后调用
`runtime.close()`。Loop 持有并停止自己启动的 transport/state stream；超时资源会移交 runtime，
供后续 close 重试。Runtime mutation、snapshot、会访问 runtime 状态的 timeline compile、camera
sample 与 World step 都留在 owner thread。Foxglove 输出必须满足第 10 节的可选依赖和 loopback
前提。精确消息与 terminal event 见 [Single Scene JSON 参考](single-scene-json.md)。

## 13. Tiled Scene Interactive Runtime 与 Transport

从 `linkerbot_sim.app.interactive.tiled_scene` import。Parser 与 queue/transport 所有权不需要 Isaac；
runtime 构造、dispatch 与 loop 执行需要 Isaac。

### 13.1 Runtime（`Isaac main thread`；选用 cuRobo 时另需 `cuRobo/CUDA`）

| 入口 | 签名与生命周期 |
| --- | --- |
| `TiledSceneRuntime` | 使用 `create(*, env_name, env_config, simulation_app, camera_output_settings, shutdown_settings, default_decimation, controller_bundle="default", planner_workers=2, max_pending_requests=64, max_completed_results=256, max_batch_problems=64, oversize_request_policy="split", failure_policy="hold_failed_env", cache_root=None, planner_request_defaults=None, command_defaults=None, playback_settings=None, planner_shutdown_timeout_s=30.0, planner_backend="curobo", curobo_profile="default", joint_batch_mode="auto", additional_output_path_plans=())`。它创建 World/scene、所选 camera、IK/planner service、buffer 与初始状态。 |
| `main` | `(argv=None) -> None`；解析 Tiled Scene runtime profile，构造 runtime/transport/telemetry，运行 loop，并执行有界 shutdown。子进程使用优先调用 script。 |
| `handle_tiled_interactive_message` | `(message: Mapping[str, object], runtime) -> dict`；在副作用前校验当前 Tiled Scene 消息与 selector，只同步调用一个 runtime 操作，把内部 label 转成公共 robot ID，并把预期错误收敛为 JSON-compatible `rejected` response。它不会把调用切换到 owner thread。 |
| `run_interactive_loop` | `(runtime, *, telemetry, request_queue, telemetry_rate_hz, idle_physics_policy="pause", idle_step_duration_s=None, queue_poll_timeout_s=0.1, event_publisher=None, transport_status_provider=None) -> None`；作为唯一 queue consumer，在 owner thread 串行 dispatch，执行配置的 idle step 与 telemetry，交付 response 后才释放 request admission。 |

Runtime 公共方法包括 `status()`、`robot_name_for_id(id)`、`idle_step()`、
`reset(env_ids)`、`step_action(action, *, env_ids, robot_names=None)`、
`get_state(*, env_ids, fields=None, include_efforts=False)`、
`set_state(state, *, env_ids)`、`get_snapshot(*, env_id)`、
`set_snapshot(snapshot, *, env_ids, label_map=None, strict=True)`、
`clone_state(*, source_env_id, target_env_ids, strict=True)`、
`load_trajectory(trajectory, *, env_ids, robot_name=None)`、
`step_trajectory(*, env_ids, robot_names=None, decimation=None)`、
`submit_plan(message, *, env_ids, robot_name=None)`、
`submit_hand_motion(message, *, env_ids, robot_name=None)`、
`planner_status(*, wait_timeout_s=0)`、`cancel_plan(...)`、`clear_completed(...)`、
`trajectory_status(...)`、`clear_trajectory(...)` 与 `close() -> bool`。State/action
array 保留 selected-env 行 `(E,...)`，并使用对应 runtime robot 的 command-joint 顺序。详细
payload 与 selector 规则属于 [Tiled Scene JSON 参考](tiled-scene-json.md)。

释放 Kit 前必须调用 `close()` 直到返回 true。它按依赖顺序关闭 planner、camera output、
cuRobo/IK 资源与 `SimulationApp`；false 表示至少一个有界关闭超时，runtime 仍保留资源所有权。

### 13.2 Pure Parser 与 Queue Admission

| 入口 | 签名与契约 |
| --- | --- |
| `parse_args` | `(argv=None) -> argparse.Namespace`；解析 Tiled Scene CLI，不构造 Isaac；非法 CLI 值由 `argparse` 通过 `SystemExit` 报告。 |
| `parse_tiled_action` | `(message, *, planner_defaults=None, command_defaults=None) -> TiledCommandAction`；只严格解析 canonical `type="step"` 消息，解析配置 default，校验有限数组/enum/shape，并返回供 `runtime.step_action` 消费的 action。非法输入抛 `ValueError`。 |
| `BoundedInteractiveRequestQueue` | `(*, capacity: int)`；`put`、`get`、`task_done`、`full`、`record_rejection`、`status`。Capacity 覆盖 data request 从 admission 到 response 交付的整个生命周期，不只是队列内元素。唯一 consumer 必须在同一线程按顺序配对 `get()` 与 `task_done()`。满队列与普通 `queue.Queue` 异常适用；control item 不占 data admission。 |

Queue item class 是 transport 实现值，不是可独立构造的接口。应通过下文 start 函数取得，并由
`run_interactive_loop` 消费，不能自行构造。

### 13.3 Transport 所有权（`pure`，只做后台 I/O）

| 入口 | 签名与生命周期 |
| --- | --- |
| `start_stdin_jsonl_reader` | `(request_queue, *, quit_on_eof, max_message_bytes=1048576, admission=None) -> reader handle`；启动可中断 reader thread。调用方保留 handle，在 shutdown 时调用 `stop(timeout_s=...) -> bool`，并在释放所有权前检查 `is_alive()`；超时后应重试 `stop`。Transport 校验 framing、长度与 UTF-8 后把文本入队，不调用 runtime；strict JSON parse 随后在 owner-thread `run_interactive_loop` 中执行。 |
| `start_tcp_jsonl_server` | `(request_queue, *, quit_event, host, port, max_message_bytes=1048576, max_connections=16, server_poll_interval_s=0.1, response_poll_interval_s=0.5, admission=None) -> ThreadingTCPServer`；bind 必须满足项目 loopback policy，每行对应一个 response。 |
| `stop_tcp_jsonl_server` | `(server, *, timeout_s=2.0) -> dict`；关闭 active connection，并有界停止 serve/shutdown/handler thread。必须检查 status；live resource 仍被持有，可重试调用。 |
| `start_websocket_server` | `(request_queue, *, quit_event, host, port, max_message_bytes=1048576, max_connections=16, event_queue_capacity=256, server_poll_interval_s=0.1, response_poll_interval_s=0.5, startup_timeout_s=5.0, admission=None) -> WebSocketServerHandle`；启动独立 asyncio thread。Startup 报错或超时时，helper 会先执行有界清理再重新抛出；清理失败或超时会作为附注保留在原始异常上。成功返回后由调用方持有 handle，并使用 `publish_event`、`status` 与有界 `stop`；检查 stop status，thread 仍存活时重试。 |

项目拥有的 `main` 组合会让 TCP 与 WebSocket 共用一个 connection admission。该 admission
class 没有已记录的公共构造函数，因此独立调用两个 start 函数会得到各自独立的 connection cap；
需要进程级统一上限时应使用 `main`。具体 admission 与 server-handle class 不是独立 facade
入口，调用方只使用本文记录的返回 handle 方法。所有内置 listener 都是无认证 loopback endpoint。
不能从 handler thread 调用 Isaac runtime；handler 只入队纯数据，由 owner thread dispatch。

## 14. 已记录的 Advanced Owner Path

本节模块没有统一 package facade。只有下列 fully qualified symbol 是可依赖 owner path；同一
模块中的其他名称不会因此自动成为公共接口。

### 14.1 Runtime 配置（`pure`）

| 精确 symbol | 签名与契约 |
| --- | --- |
| `linkerbot_sim.configs.profiles.profile_path` | `(group, name) -> Path`；group 为 `runtime`、`robot`、`env`、`object`、`curobo`、`logging`，name 必须是安全的单一 file stem。目录型 env 解析到 `base.yaml`；普通返回路径不保证已存在。 |
| `linkerbot_sim.configs.profiles.load_profile_yaml` | `(group, name) -> dict`；加载 checkout profile。Env、robot、object、cuRobo 与 logging group 会调用各自 domain validation；`runtime` 只返回 strict-YAML 数据，不解析 runtime schema，因此该 group 应使用 `load_runtime_profile`。Group/name/content 错误抛 `ValueError`，文件缺失抛 `FileNotFoundError`。 |
| `linkerbot_sim.configs.profiles.load_env_profile_yaml` | `(name) -> dict`；加载并校验单个 env YAML，或目录中的 `base.yaml` 与排序后的 per-env fragment，返回完整 merge 后 mapping。 |
| `linkerbot_sim.configs.runtime.RuntimeProfileConfig` | 使用 `RuntimeProfileConfig.from_mapping(data, *, profile_name=None, source_path=None)` 与 `as_dict()`。只接受当前顶层 `runtime` mapping，校验类型/范围/跨字段约束，复制输入且不创建 runtime 资源。 |
| `linkerbot_sim.configs.runtime.ResolvedRuntimeConfig` | 包含 `config`、叶子 `sources` 和确定性 `fingerprint`；`as_dict()` 返回 effective mapping。只读 property 暴露 `mode`、`profiles`、`simulation_app`、`execution`、`interactive`、`planner`、`playback`、`camera_output`、`telemetry`、`output`、`paths`、`shutdown`。 |
| `linkerbot_sim.configs.runtime.load_runtime_profile` | `(name) -> RuntimeProfileConfig`；安全 stem lookup 加严格 runtime 解析。 |
| `linkerbot_sim.configs.runtime.resolve_runtime_config` | `(profile, *, cli_overrides, env_config, expected_mode=None) -> ResolvedRuntimeConfig`；只应用已知且非 `None` 的 dotted override，校验所选 profile 与 mode/env 跨字段关系，并记录来源；不启动 Isaac，也不校验完整依赖图。 |
| `linkerbot_sim.configs.validator.ValidatedProfileGraph` | 冻结结果，包含 `runtime_profile`、已解析 `profile`、`resolved` 与排序 dependency name 的只读 mapping。 |
| `linkerbot_sim.configs.validator.validate_profile_graph` | `(*, runtime_profile, profile, resolved, env_config) -> ValidatedProfileGraph`；读取并校验 env、robot、controller、object、cuRobo 与 logging 依赖，不创建 Isaac/GPU/file-output。缺少依赖抛 `FileNotFoundError`，结构/capability/path 冲突抛 `TypeError` 或 `ValueError`。 |

调用 runtime factory 时应使用 resolved object 内的 nested setting，避免手工拼装未完整校验的
setting DTO。YAML 字段与 overlay 优先级仍由[配置参考](configuration.md)拥有。

### 14.2 Single Scene Factory 与 Parser

| 精确 symbol | 标签与契约 |
| --- | --- |
| `linkerbot_sim.app.runtime.single_scene_runtime.create_single_scene_runtime` | `Isaac main thread`；`(*, env="scene1", env_config=None, simulation_app, camera_output_settings, shutdown_settings, output_settings=None, curobo_profile="default", logging_profile="default_logger", controller_bundle="default", control_mode="position", cache_root=None, hold_app=False, status_prefix=None, additional_output_path_plans=(), session_factory=..., profile_loader=..., controller_bundle_loader=...) -> SingleSceneRuntime`。最后三个 injectable factory 是测试/组合 hook。函数在应用输出计划前统一校验，执行一次 World reset，失败时回滚已获取资源，成功后把所有权转给结果。 |
| `linkerbot_sim.app.runtime.single_scene_runtime.SingleSceneRuntime` | Factory 返回的 owning runtime。可依赖操作为只读 property `robots_by_id`、`robot_id_by_label`、`world`、`config_fingerprint`，以及 `robot(robot_id)`、`status()`、`close()`。`close()` 返回含 `stopped`/`live_resources` 的 report；未停止时重试。不能直接实例化 dataclass。 |
| `linkerbot_sim.app.interactive.single_scene.protocol.InteractiveMotionCommand` | `pure` frozen DTO，由 parser 返回。先按 `kind` 分支；可选 ID、reset、snapshot、timeline 字段只对对应 kind 有意义。 |
| `linkerbot_sim.app.interactive.single_scene.protocol.parse_interactive_motion_message` | `pure`；`(message, *, planner_defaults=None, command_defaults=None) -> InteractiveMotionCommand`。解析/规整一条 Single Scene 命令且不修改输入 mapping，应用严格字段/类型/有限向量校验与 default，不产生 queue/runtime 副作用。Protocol 缺陷抛 `ValueError`；可达性和 runtime robot ID 在主线程后续校验。 |

### 14.3 Trajectory Value 与 Builder（`pure`）

| 精确 symbol | 签名与契约 |
| --- | --- |
| `linkerbot_sim.trajectories.types.TrajectoryEval` | 冻结值，包含 `(position, velocity, acceleration, jerk, effort)`；每项 shape `(C,)`，使用 trajectory joint 顺序。 |
| `linkerbot_sim.trajectories.types.JointTrajectory` | Keyword constructor：`times (N,)`、`positions (N,C)`、`joint_names (C,)`，可选同 shape derivative/effort matrix 与 `phases (N,)`。复制有限 matrix，要求非空严格递增时间，省略 matrix 填零；不匹配抛 `ValueError`。`domain()`、`eval(t)`、`eval_all(t)` 把有限 query clamp 到端点，`len()` 为 `N`。时间 s，revolute position/derivative 使用 rad 系单位。 |
| `linkerbot_sim.trajectories.joint_trajectory_builder.joint_trajectory_from_positions` | `(*, times, positions, joint_names, phase="trajectory") -> JointTrajectory`；以有限差分计算 velocity、acceleration、jerk，不校验 robot limit 或动力学可行性。 |
| `linkerbot_sim.trajectories.retiming.trajectory_sample_times` | `(*, duration_s, sample_dt_s, include_start=False) -> ndarray`；返回至少一个正 tick time，并可选在开头加入零。当 `0 <= duration_s < sample_dt_s` 时，时间域扩展为一个完整的 `sample_dt_s` tick；否则末值等于 `duration_s`，需要时保留不足一 tick 的末段。有限性/范围错误抛 `ValueError`。 |
| `linkerbot_sim.trajectories.retiming.retime_joint_trajectory` | `(trajectory, *, duration_s: float | None, sample_dt_s: float | None, start_position=None, phase=None, include_start=False) -> JointTrajectory`；两个 timing value 都存在时，按累计 joint-path progress 重采样并重算 derivative。任一 timing value 为 `None` 时，`include_start=False` 原样返回同一个 trajectory 对象，`include_start=True` 则抛 `ValueError`。它是几何定时，不做 limit-aware optimization；start width 错误也抛 `ValueError`。 |

### 14.4 Logging 配置（`pure`）

| 精确 symbol | 签名与契约 |
| --- | --- |
| `linkerbot_sim.logging.config.JointLoggingConfig` | 冻结 Single Scene CSV setting。`should_write_step(step)` 应用 enable/decimation；`flush_interval_steps(physics_dt)` 把秒转换成至少一步。Tiled Scene entrypoint 不消费该 profile。 |
| `linkerbot_sim.logging.config.normalize_logging_profile_name` | `(value) -> str`；只接受 trim 后匹配 `[A-Za-z0-9][A-Za-z0-9_-]*` 的 stem，否则抛 `ValueError`。 |
| `linkerbot_sim.logging.config.joint_logging_config_from_mapping` | `(data, *, source_path=None) -> JointLoggingConfig`；严格校验当前字段/类型/范围，不创建文件。 |
| `linkerbot_sim.logging.config.load_joint_logging_profile` | `(name, *, logging_root=...) -> JointLoggingConfig`；安全 lookup 与严格解析，缺少文件抛 `FileNotFoundError`。 |
| `linkerbot_sim.logging.config.override_logging_config` | `(config, **updates) -> JointLoggingConfig`；忽略 `None`，返回 `dataclasses.replace`，未知字段抛 `TypeError`。不会重新运行 mapping 校验，因此只能传已校验值。 |

### 14.5 Telemetry DTO 与配置 Owner（`pure`）

| 精确 symbol | 签名与契约 |
| --- | --- |
| `linkerbot_sim.telemetry.state_snapshot.RobotJointStateSnapshot` | 一行 robot 状态：ID/label/name 与 `(J,)` rad、rad/s、rad/s2 array，以及可选 effort array。`effort_values(field)` 接受 `none`、`commanded`、`measured`、`applied`；`as_dict()` 把非有限 effort sample 转成 JSON null。等宽约束由调用方负责。 |
| `linkerbot_sim.telemetry.state_snapshot.ObjectPoseSnapshot` | `(name, prim_path, position_m (3,), orientation_wxyz (4,))`；`as_dict()` 输出一行 world pose。 |
| `linkerbot_sim.telemetry.state_snapshot.StateSnapshot` | `(step, time_s, robots, objects=(), phase=None)`；字段冻结、可跨线程传递的观测 payload，提供 JSON-compatible `as_dict()`。Dataclass 不复制 NumPy array，也不设置 write-protect；producer 必须传入已脱离 runtime 的数组，所有调用方在发布后都必须把它们视为不可变。它不同于可恢复的 `SimulationSnapshot`。 |
| `linkerbot_sim.telemetry.state_snapshot.StateStream` | `(*, capacity=1, drop_policy="latest")`；thread-safe、single-consumer、有界交接。`publish` 永不阻塞；`latest`、`wait_next`、`status`、`is_closed`、`close(*, discard_pending=False) -> None` 定义生命周期。Capacity/policy 错误抛 `ValueError`；其中不能放 Isaac 对象。 |
| `linkerbot_sim.telemetry.foxglove.FoxgloveTopicConfig` | `(joint_states="/joint_states", scene="/scene", state="/linkerbot/state")`；只表示 topic name，不加载可选 SDK。 |
| `linkerbot_sim.telemetry.foxglove.prepare_mcap_output` | `(path, *, existing_file_policy) -> OutputPathPlan | None`；只规划、不应用文件系统变更。`resume` 抛 `ValueError`；返回 plan 作为 opaque 输入传给 runtime factory。 |
| `linkerbot_sim.telemetry.tiled.config.TiledTelemetryConfig` | 经过验证的 selected-env/topic/buffer DTO。Selected ID 非空、唯一、非负，primary ID 必须在其中；decimation/capacity/policy/timeout 严格校验，非法值抛 `ValueError`。 |
| `linkerbot_sim.telemetry.tiled.config.parse_env_ids` | `(value: str) -> tuple[int, ...]`；解析逗号分隔 integer，并拒绝 blank string。随后构造 `TiledTelemetryConfig`，才能拒绝解析后的空 tuple，并校验唯一性、非负 ID 与 primary selection；runtime 组合层另行校验每个 ID 小于 `num_envs`。 |

Foxglove logger/sink、Single Scene sampler、CSV writer 与 Tiled Scene telemetry sink 仍是 runtime 拥有的实现。
除非本节新增精确 symbol，应通过已记录的 runtime setting 配置它们。

## 15. 错误与所有权规则

- 在创建 GPU 或 Isaac 资源前校验 shape、名称、frame、有限数值与配置。配置/编程缺陷抛异常；
  预期的 planner 无法完成通常用失败 result 或 rejected response 表达。
- NumPy 行列含义属于契约。必须保留显式 env、joint、body、sample 维度；只有方法明确允许时
  才能依赖 broadcasting。
- 接受 context、World、stage、articulation 或 runtime 的 class 通常只借用它。只有明确命名为
  `create`、`start`，或返回 owning handle 的方法会分配必须关闭的生命周期。
- Planner、transport、telemetry、camera-output 或 file-writer thread 不能读取 Isaac 对象；
  必须先在 owner thread 采集不可变 Python/NumPy 数据。
- Snapshot/reset/state restore rollback 失败或 shutdown 未完成后，应停止 mutation，并重建 runtime
  或完成关闭。详见[已知风险与设计约束](../operations/constraints.md)。
