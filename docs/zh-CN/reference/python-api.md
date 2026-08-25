# Python Facade 参考

语言：[中文](python-api.md) | [English](../../en/reference/python-api.md)

本页只列稳定 import 面。owner path 和 module map 用于维护导航，不构成兼容承诺。所有 facade
必须 lazy：普通 import 不启动 Kit、不初始化 CUDA，也不读取 YAML。

## `linkerbot_sim`

| 导出 | 语义 |
| --- | --- |
| `REPO_ROOT` | checkout 根路径 |

## `linkerbot_sim.configuration`

| 导出 | 语义 |
| --- | --- |
| `ComputeSettings` | mode root 中唯一 CUDA 设备编号的 owner |
| `MirrorConfig` | Mirror 已解析 immutable 根配置 |
| `KaleidoscopeConfig` | Kaleidoscope 已解析 immutable 根配置 |
| `KaleidoscopeEnvironmentSettings` | Kaleidoscope mode root 中环境数、路径命名与基准原点的唯一 owner |
| `KaleidoscopeViewportSettings` | 与训练根分离的 launch-only viewport 冷配置 |
| `NewtonCpuSettings` | Mirror Newton/CPU leaf；物理留在 CPU，根 compute 仍选择 cuRobo/RTX GPU |
| `NewtonCudaSettings` | Newton/CUDA physics leaf；设备和 world 数在 session 投影时派生 |
| `PhysicsEngine` | 公开 physics engine：`physx` 或 `newton` |
| `PhysicsExecution` | 公开 execution：`cpu` 或 `cuda` |
| `PhysicsSettings` | schema-valid physics leaf 的 strict union；产品根另行收窄 runtime 支持矩阵 |
| `PhysxCpuSettings` | Mirror PhysX/CPU leaf 配置 |
| `PhysxCudaSettings` | Kaleidoscope PhysX/CUDA leaf 配置 |
| `SkrlTrainingSettings` | 下游 skrl strict leaf；device 固定继承 environment |
| `load_mirror_config(source="physx_cpu", *, configs_root=None)` | 从唯一 catalog 组合 Mirror profile graph |
| `load_kaleidoscope_config(source="physx_cuda", *, configs_root=None)` | 从唯一 catalog 组合 Kaleidoscope profile graph |
| `load_kaleidoscope_viewport_config(source="kaleidoscope", *, configs_root=None)` | 严格加载 launch-only viewport profile；不进入 episode fingerprint |
| `load_skrl_training_settings(source="tblock_push_v1_ppo", *, configs_root=None)` | 通过 catalog 唯一 I/O 边界加载 skrl training profile |
| `semantic_config_payload(config)` | 返回排除 provenance 的 canonical JSON-compatible 语义图 |
| `semantic_config_fingerprint(config)` | 返回 validator 与 snapshot compatibility 共用的语义 SHA-256 |

Catalog 只在显式调用时执行 YAML I/O；配置对象不创建 runtime resource。

## `linkerbot_sim.mirror`

| 导出 | 语义 |
| --- | --- |
| `MirrorConfig` | 便于类型标注的配置根 |
| `MirrorRuntime` | 一个 session 与全部 Mirror resource 的 owner |
| `MirrorController` | owner-thread typed request dispatcher |
| `create_mirror_runtime(config, *, assembly_factory=None)` | 唯一 composition factory；队列容量只来自 strict control profile |
| `run_mirror(runtime, *, endpoints=(), poll_timeout_s=None, should_stop=None, on_ready=None, max_iterations=None, close_on_exit=True)` | 主线程 admission/physics/render loop；省略 poll timeout 时读取 strict control profile |

Wire DTO、camera coordinator、close report 和 snapshot schema 是参考实现细节；外部进程按
[Mirror JSON](mirror-json.md)集成，不依赖内部 module path。

### `MirrorRuntime`

`MirrorRuntime` 的 owner-thread 主要方法：

| 方法 | 合同 |
| --- | --- |
| `step(render=False)` | 只推进一次物理；需要时随后走统一 render/post-step observer |
| `render()` | 显式 render 并立即 capture 当前帧；未启用 rendering 时失败 |
| `get_state()` / `set_state(state, strict=True)` | 读取 owned 状态或事务式写入并使碰撞缓存失效 |
| `capture_snapshot()` / `restore_snapshot(...)` | 捕获或恢复版本化单场景快照 |
| `reset(hold_after_reset=True)` | 恢复初态并把 timeline 归零；默认按 `idle_step_duration_s` 为全部 arm/hand group 编译同步 hold，复用正常 executor/render 路径 |
| `get_control_mode()` | 查询 immutable initial/active mode、generation、支持集与全机器人 scope |
| `set_control_mode(mode, expected_generation=None)` | 在两次完整运动之间事务式切换全部机器人，不重建 runtime |
| `status()` | 返回产品、物理、场景、碰撞与关闭状态 |
| `close()` | 按依赖逆序幂等关闭，失败资源保留供重试 |

## `linkerbot_sim.kaleidoscope`

| 导出 | 语义 |
| --- | --- |
| `KaleidoscopeConfig` | strict PhysX CUDA / project-owned Newton 配置根 |
| `TorchKaleidoscopeEnv` | 原生 CUDA tensor 环境 |
| `GymnasiumKaleidoscopeAdapter` | 显式 NumPy VectorEnv 边界 |
| `KaleidoscopeEpisodeSnapshot` | owned CUDA episode snapshot |
| `KaleidoscopeTrainingPort` | training 唯一允许依赖的结构化 port |
| `ControlModeState` | immutable initial/active mode、generation、支持集和 scope |
| `ControlModeChange` | 幂等或真实模式变化的结果 |
| `ControlModeGenerationConflict` | optimistic generation 前置条件冲突 |
| `ControlModeIncompatibleError` | mode/trajectory 与固定 action 不兼容 |
| `ControlModeLockedError` | runtime phase 或 SAME_STEP transaction 禁止切换 |
| `ControlModeSwitchError` | 前向切换失败且已成功 rollback |
| `ControlModeRollbackError` | rollback 失败并触发 runtime 永久 fail-stop |
| `make_torch_env(...)` | 构造 native env |
| `make_viewport_env(..., viewport=None, viewport_profile="kaleidoscope")` | 为 PhysX/Newton 构造只显示 `selected_env` 的显式 human viewport env |
| `make_gymnasium_env(..., viewport_profile="kaleidoscope")` | 构造 Gymnasium adapter；仅 human render 使用 viewport profile |
| `register_gymnasium_envs()` | 幂等注册项目 env ID |

State/clone 的方法合同见 [Kaleidoscope API](kaleidoscope-api.md)。`make_torch_env()` 的训练 step
始终无渲染；`make_viewport_env()` 也只在调用方显式执行 `env.render()` 时更新一帧，不提供 camera、
SyntheticData、Replicator 或录制。

`TorchKaleidoscopeEnv` 原生提供 `get_control_mode()` / `set_control_mode()`；setter 不进入
`KaleidoscopeTrainingPort`、Gymnasium 或 skrl，训练始终保持初始 position 模式。

native bootstrap 和 `make_torch_env()` 不导入 Gymnasium；只有显式调用
`make_gymnasium_env()` 时才在函数内延迟导入 `GymnasiumKaleidoscopeAdapter`。因此未安装 training extra
不会污染原生 Torch/skrl 之外的基础 import，缺依赖只在 NumPy 边界明确报错。

## `linkerbot_sim.training.skrl`

| 导出 | 语义 |
| --- | --- |
| `SkrlTorchAdapter` | CUDA SAME_STEP 环境 adapter |
| `FinalObservationPPO` | 正确 bootstrap truncated terminal observation 的 PPO |
| `CudaRolloutMemory` | 不接受 CPU/Python selector 的 rollout memory |
| `make_skrl_trainer` | 从 training profile 构造 trainer |

Training facade 不导出 Isaac handle，也不接受 Gymnasium NumPy env 伪装为 native CUDA env。

## `linkerbot_sim.snapshots`

| 导出 | 语义 |
| --- | --- |
| `SceneSnapshot` | versioned CPU/NumPy scene snapshot |
| `load_scene_snapshot` | 从持久化文件读取并校验 schema |
| `save_scene_snapshot` | 原子保存 scene snapshot |
| `validate_scene_snapshot` | 只校验，不 mutation runtime |

Mirror adapter 和 Kaleidoscope episode snapshot 属于各产品，不由该 facade 混合分派。

## `linkerbot_sim.backends.curobo`

这是高级 capability facade。构造 context/IK/planner 的调用方必须显式拥有并关闭对应 resource；
Kaleidoscope 只组合 device batch kinematics，Mirror 才能组合 planner/collision world。具体 owner
和 runtime 要求见[源码模块图](../development/module-map.md)。

| 导出 | 语义 |
| --- | --- |
| `CuroboConfig` | 共享后端严格配置根 |
| `CuroboContext` | Mirror 的规划与碰撞 capability owner |
| `CuroboDeviceBatchIKSolver` | Kaleidoscope 使用的 CUDA batch IK adapter |
| `CuroboKinematicsContext` | 不创建 planner/collision world 的运动学 owner |
| `create_kinematics_context` | 构造 Kaleidoscope 窄运动学 capability |
| `curobo_config_from_profiles` | 将 typed robot/cuRobo profile 与 canonical CUDA device 投影为唯一数值后端配置 |

该 backend facade 不再提供旁路 YAML 读取入口。`linkerbot_sim.configuration` catalog 统一解析
`configs/curobo/`、注入 mode root CUDA device，再把 typed projection 交给数值后端。
