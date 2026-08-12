# 配置工作流

语言：[中文](configuration.md) | [English](../../en/guides/configuration.md)

## 先选产品，再改 leaf

1. 复制 `configs/modes/mirror/physx_cpu.yaml` 或 `configs/modes/kaleidoscope/physx_cuda.yaml`；
2. 只在 mode 文件修改 profile 引用，不内联 leaf 参数；
3. 在对应 owner 目录创建新 leaf；
4. 运行 `scripts/validate_mode_config.py`；
5. 配置通过后再启动 Isaac smoke。

## 必须闭合整张配置图

一次 mode load 不只读取 mode 和 leaf，还必须在同一个 `configs_root` 中继续解析：

```text
mode -> leaf profiles -> scene robot/object profiles -> effective controller bundles
```

controller bundle 的优先级是 scene 中机器人实例的 `controller_profile`、robot profile 默认值、
`physics.engine` 派生默认值。返回的 frozen config 已携带每个 instance 的 `resolved_profile` 和只读
`controller_bundles`；runtime 只消费这些对象，不会按名称回读仓库默认配置。自定义 `configs_root` 因而
必须包含所引用的 `robots/`、`objects/` 和 `controllers/`，缺文件会在启动 Kit 前失败。

`sources` 记录每个实际读取文件，适合审计来源；它不参与语义 fingerprint。配置指纹由 configuration
层统一生成，validator 与 Kaleidoscope snapshot compatibility 共用同一语义输入：相同配置复制到不同
绝对目录后指纹不变，机器人、对象或 controller 的有效内容变化后指纹会变化。

例如扩大 Kaleidoscope env 数时，只修改 mode root `environments.num_envs`；不要把环境数写入
scene/task/physics。更换 GPU 时只修改 mode root `compute.cuda_device`，不要在
Torch、cuRobo 或 training profile 再写 device。

机器人关节摩擦、刚体阻尼、接触 combine mode 和 per-body solver iteration 只修改
`configs/robots/<profile>.yaml` 的 `robot.physics.physx`；controller bundle 只保存控制增益和
effort/drive 限幅。对象的通用接触系数放在 `object.physics.material`，combine mode 与绳体 solver
放在 `object.physics.physx`。Newton composition 会在投影边界裁掉这些 PhysX leaf，不靠运行期告警
表达正常配置。

## 常见组合

### Mirror PhysX

`mirror/physx_cpu` 适合 GUI、camera、telemetry 和完整规划。物理使用 CPU 不代表 cuRobo 不能使用
GPU；cuRobo 与 render 从根 `compute.cuda_device` 派生设备，但不能伪装成 PhysX CUDA scene writer。
`curobo.kinematics.max_batch_size` 只属于 FK/IK。Mirror MotionPlanner 固定一次处理一个请求；其 leaf
仍拥有 warmup、graph、IK/trajopt seed、碰撞能力和 cache 容量。wire planning segment 只能覆盖
`duration_s`、`sample_dt_s`、`avoid_collisions` 和 `force_collision_refresh`，`coordination` 只能在
wrapper/timeline 顶层覆盖；`timeout_s` 不能由请求覆盖，始终读取
`planning.request_defaults.timeout_s`。

### Mirror Newton

`mirror/newton_cpu` 与 `mirror/newton_cuda` 复用相同 `control: mirror`、scene/curobo/planning/outputs 与根 compute；
`physics.engine` 自动派生 `newton` controller bundle。Newton runtime 可以是通用 multi-world
infrastructure，Mirror session 始终从产品语义派生一个 world。CPU execution 不使用 CUDA stream/graph，
但根 CUDA device 仍由 cuRobo 与 RTX 消费。

### Kaleidoscope

`kaleidoscope/physx_cuda` 组合 PhysX CUDA/Fabric，builder 内部固定使用 GridCloner、3.0 m 间距、
`replicate_physics=true`、`copy_from_source=true`、`enable_env_ids=true`；`kaleidoscope/newton_cuda` 组合项目自有 Newton，
builder 内部固定按最终 `num_envs` 创建等量独立 worlds、零间距共址。Kaleidoscope 没有 control slot，
两个后端的默认 controller bundle 均由 `physics.engine` 派生。
两种复制机制都保留为后端实现，但不再作为可独立选择的公开 profile，也不加载 Isaac Newton extension。
两个 mode profile 都只组合 headless GPU RL 所需能力。需要人工查看时，另行加载
`configs/visualization/kaleidoscope.yaml`；它不成为 mode slot，也不进入训练配置指纹。

task 只冻结 action 语义，不选择数值 backend。`joint_control`/`joint_delta` mode 必须省略
`profiles.curobo`，因此不创建
cuRobo；EE/linear action 对应的 mode 必须增加 `profiles.curobo: kaleidoscope_batch_ik`。该 profile 必须
省略 `motion_planner`，让 `kinematics.max_batch_size >=` 最终有效环境数、
`kinematics.collision_check=false`。canonical profile 省略 `kinematics.collision_cache`；保留合法
cache 也会在构造后端前被丢弃，因此无碰撞路径不会分配缓存。
即使显存充足也不要添加
trajectory planner/planning collision cache/avoidance，因为它们改变产品能力和显存上界。这里删除的是
规划碰撞查询与避障，不是物理接触；PhysX/Newton 仍计算机器人、工装和任务对象之间的真实 contact。

### Kaleidoscope viewport 冷配置

选择 `KaleidoscopeViewportSettings` 就固定进入 human-viewer 窗口边界；该配置独立拥有
`selected_env`、`render_every_n_steps`、render/window 尺寸、renderer/anti-aliasing/denoiser 和
scene visual。`make_viewport_env()` 可通过
`viewport_profile="kaleidoscope"` 选择它，或通过 `viewport` 直接接收已加载对象；Gymnasium
`render_mode="human"` 使用 `viewport_profile`。`make_torch_env()` 与训练 snapshot compatibility 不读取它。

`selected_env` 必须落在最终 `num_envs` 内。PhysX 与 Newton 都只为该环境维护 renderer-facing
状态，训练 physics tick 继续使用 `render=False`，并由调用方显式 `env.render()`。这个 profile 不允许
camera、SyntheticData、Replicator、录制或 image observation。

### PhysX GPU 显存预算

`configs/physics/physx/cuda.yaml` 的进程级 `memory` 映射对应完整 `GpuMemoryBudget`：

| 字段 | 如何使用 |
| --- | --- |
| `max_simulator_process_mib` | 当前 simulator PID 的 NVML 显存绝对上限 |
| `min_free_floor_mib` | 所选设备在所有审计阶段必须保留的空闲 MiB 下限 |
| `min_free_fraction_after_warmup` | warmup 后和稳态两端必须保留的空闲比例，范围 `(0, 1]` |
| `max_steady_growth_mib` | steady final 相对 steady baseline 的进程显存增长上限，可为 `0` |

这些字段不能放进 mode root、task 或 training profile，也不能用 Torch allocator 数字替代进程级
NVML 采样。修改后先 strict validate，再在目标 GPU 上运行：

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

该脚本只接受 Kaleidoscope `physx_cuda` profile；Newton 使用独立的 world capacity smoke。

## Scene 与 task 不混合

- Scene：稳定 USD 拓扑、robot/object instance、物理/渲染频率；
- Environments：mode root 唯一声明环境数、USD 路径命名和原点；
- 后端复制实现：PhysX 固定 GridCloner/env IDs，Newton 固定 multi-world；不暴露公开 selector；
- Task：observation/action/reward/done/randomization；
- Training：PPO 等下游算法。

这种拆分允许同一个 task 调整训练算法，也允许同一个 scene 在不改拓扑的情况下调整随机化。
Kaleidoscope catalog 会从同一配置根展开每个 object profile：scene 可有多个静态 rigid，但必须恰好有
一个非静态 rigid，并由 `task.dynamic_object` 命名；dynamic chain 会在启动 Kit 前失败，避免
snapshot/clone 静默漏掉动态状态。

## Fail-fast 示例

以下配置应在启动前失败：

- Kaleidoscope 引用 `physx/cpu`，或 mode root 出现已删除的 `profiles.replication`；
- Kaleidoscope scene 出现 camera/viewport，或 mode 出现 planning/outputs；viewport 只能位于独立的
  `configs/visualization/kaleidoscope.yaml` 冷配置；
- Mirror root 缺少 `compute.cuda_device`；
- physics leaf 重复声明 `cuda_device`；
- PhysX `memory` 缺少任一 `GpuMemoryBudget` 字段、数值类型错误或超出字段范围；
- task action 声明 backend/profile 字段，或 `joint_control`/`joint_delta` mode 带 `profiles.curobo`；
- EE/linear mode 缺少 `profiles.curobo`，或所选 cuRobo profile 声明 `motion_planner`、打开
  kinematics collision check、在 collision check 关闭时仍声明 collision cache；
- scene、physics 或 task 重复拥有 env count；
- YAML 有重复 key、字符串 boolean 或非有限数值。
- 自定义配置根缺少 scene 引用的 robot/object/controller 闭包。

字段全集见[配置参考](../reference/configuration.md)，源码 owner 见
[模块图](../development/module-map.md)。
