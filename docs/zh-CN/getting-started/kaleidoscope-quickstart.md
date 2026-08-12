# Kaleidoscope 快速入门

语言：[中文](kaleidoscope-quickstart.md) | [English](../../en/getting-started/kaleidoscope-quickstart.md)

Kaleidoscope 是 GPU-native 的强化学习环境。默认训练入口保持 headless；显式调试入口可为 PhysX
CUDA/Fabric 或项目自有 multi-world Newton 打开只显示一个环境的 viewport。两个后端均保留
任务物理接触、batch IK 和同步直线动作，但不创建 batch trajectory planner、planning collision
world、规划避障、camera、SyntheticData、Replicator 或录制资源。

## 1. 准备与校验

```bash
uv sync --extra simulation --extra training
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile newton_cuda
```

启动前确认 `torch.cuda.is_available()`，并确保配置中的唯一 device 是
所选 `configs/modes/kaleidoscope/*.yaml` 的 `compute.cuda_device`。

| profile | physics | 后端内部派生的环境实现 | 训练入口自动选择的 headless Kit |
| --- | --- | --- | --- |
| `physx_cuda` | `physx/cuda` | GridCloner、3.0 m 间距、env ID 隔离 | `apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` |
| `newton_cuda` | `newton/cuda` | multi-world、零间距、独立 worlds | `apps/linkerbot_sim.kaleidoscope.newton.python.kit` |

Newton Kit 由项目 runtime 直接导入 Newton/MuJoCo-Warp Python wheel 并拥有所有 worlds，不加载
Isaac Newton extension。mode root 的 `environments` 唯一持有环境数与路径命名，不存在公开 replication
profile。Kaleidoscope 根只含必选 `scene/physics/task`，没有 `profiles.control`。runtime 从最终
`num_envs` 派生 `world_count`；每个 env 对应独立 world。`physics.engine` 派生 `newton` 专用
controller bundle。

内置 mode 使用 `joint_control`，因此必须省略 `profiles.curobo`。自定义 EE/直线 task 仍只保存 action 语义；
与它配套的 mode root 必须增加 `profiles.curobo: kaleidoscope_batch_ik`。该数值 profile 只含无碰撞
kinematics，必须省略 `motion_planner`、关闭 collision check。canonical YAML 省略 `collision_cache`；
保留合法 cache 也会在运行时被丢弃，因此不会分配碰撞缓存。`joint_control`/`joint_delta` 错配 cuRobo，
或 EE/直线 action 缺少 cuRobo，都会在 Kit 启动前失败。

## 2. Native Torch 环境

```python
import torch

from linkerbot_sim.kaleidoscope import make_torch_env

env = make_torch_env(profile="physx_cuda", num_envs=256)
try:
    observations, info = env.reset()
    assert observations.device == env.device

    actions = torch.zeros(
        (env.num_envs, env.action_dim),
        device=env.device,
        dtype=torch.float32,
    )
    observations, rewards, terminated, truncated, info = env.step(actions)
finally:
    env.close()
```

Native API 禁止 CPU action、Python env ID list 和隐式 dtype/device 转换。done 行必须在下一次
native step 前 reset；native/debug 入口会同步读取一次 scalar，使这个可恢复的生命周期错误在 physics
推进前明确抛出。skrl 路径通过不透明 SAME_STEP token 完成该握手，不经过该校验，训练 rollout 仍无
scalar D2H。

将示例的 `profile` 改为 `"newton_cuda"` 即切换到 Newton，Torch API、action/observation
shape 及 state/snapshot/clone 合同不变。

## 3. Gymnasium 边界

```python
import gymnasium as gym

from linkerbot_sim.kaleidoscope import register_gymnasium_envs

register_gymnasium_envs()
env = gym.make_vec(
    "linkerbot/TBlockPush-Kaleidoscope-v1",
    num_envs=64,
    profile="physx_cuda",
)
try:
    observations, info = env.reset(seed=7)
finally:
    env.close()
```

Gymnasium adapter 返回 NumPy，因此会发生整批 D2H/H2D。吞吐优先时使用 native Torch 或
`linkerbot_sim.training.skrl`。

## 4. GPU state、snapshot 与 clone

```python
env = make_torch_env(profile="physx_cuda", num_envs=4)
try:
    env.reset()
    source = torch.tensor([0, 1], device=env.device, dtype=torch.int64)
    target = torch.tensor([2, 3], device=env.device, dtype=torch.int64)
    snapshot = env.snapshot(source)
    env.clone_state(source, target, include_rng=True)
    env.restore_snapshot(snapshot, target_env_ids=target)
finally:
    env.close()
```

selector 必须非空、唯一、范围有效且与环境同 device；clone source/target 不可重叠。返回 snapshot
拥有独立 CUDA storage。两个物理后端的 state capture、restore、snapshot 和 clone 都在设备内完成；
磁盘 checkpoint 是另一个显式冷 API，不应放进每步循环。

## 5. 显式打开单环境 viewport

维护入口对两个物理后端使用同一调用方式：

```bash
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile physx_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile newton_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
```

viewer 通过 `make_viewport_env()` 读取独立的
`configs/visualization/kaleidoscope.yaml`，分别选择
`linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit` 或
`linkerbot_sim.kaleidoscope.newton_viewport.python.kit`。`selected_env` 是唯一进入
renderer-facing USD 的 world；其余环境仍在 GPU 上推进，但不会为其维护 RTX 显示状态。

viewport 配置是 launch-only 冷边界，不进入 task/physics 配置图或 episode snapshot/clone
fingerprint。每个训练 decision 的 physics tick 仍固定调用 `step(render=False)`；viewer 只在
`render_every_n_steps` 到达时显式调用 `env.render()`。该能力是 human viewport，不提供 camera
observation、SyntheticData、Replicator、录制或图像 tensor。Gymnasium 调用方也可传
`render_mode="human"`，但仍承担其固有的 NumPy D2H/H2D 边界。

## 6. 运行真实物理与动作 smoke

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile physx_cuda --num-envs 2 --steps 2
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 2 \
  --exercise-training-adapters
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 1 \
  --action-mode ee_delta_position
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 1 \
  --action-mode ee_linear_path_position
```

smoke 通过正式 composition root 启动对应 Kit，并验证 reset/step、CUDA residency、snapshot restore
和 row-to-row `clone_state`。Newton canonical 命令额外走 Gymnasium/skrl SAME_STEP；两个 action
variant 会构造临时 mode 并显式加入 `profiles.curobo: kaleidoscope_batch_ik`，分别创建非碰撞 cuRobo
batch IK context，并要求所有机器人/环境的 IK 或固定 waypoint 直线动作成功；task 本身不选择 backend。
`just smoke-kaleidoscope` 固化了相同矩阵。

正式 Newton multi-world 的 256-world 容量可单独验证：

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 256 --steps 2
```

它还会把 T-block 放入每个 world 的 TCP，要求实时 contact world id 覆盖全部环境且没有跨 world contact。
这是功能与容量门禁，不代表任何未经采样的吞吐或显存峰值。
同一命令也可通过 `just smoke-kaleidoscope-newton-capacity` 执行。

## 7. 验收 PhysX GPU 显存预算

PhysX 引擎容量不属于项目配置面；`configs/physics/physx/cuda.yaml` 只声明完整的进程级
`GpuMemoryBudget`：

| 字段 | 门禁 |
| --- | --- |
| `max_simulator_process_mib` | simulator PID 的 NVML 显存上限 |
| `min_free_floor_mib` | 四阶段采样都必须保留的空闲 MiB 下限 |
| `min_free_fraction_after_warmup` | warmup 后与稳态两端必须保留的空闲比例 |
| `max_steady_growth_mib` | steady final 相对 baseline 的进程显存增长上限 |

在目标 GPU 上执行维护入口或等价脚本：

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

脚本依次采样 prelaunch、post-warmup、steady baseline/final，成功时输出
`LINKERBOT_PHYSX_GPU_MEMORY_BUDGET_OK`。它统计整个 simulator PID，不只统计 Torch allocator；只验收
`physx_cuda` profile，不能替代 Newton 256-world 容量 smoke。

完整仿真门禁可运行 `just test-simulation`。其中 `just smoke-runtime-kits` 独立启动七个正式 Kit
closure，`just smoke-mirror` 验证 Mirror 四个 profile；两项与 Kaleidoscope 双后端、Newton 容量及上述
显存门禁一起执行。

继续阅读：[Kaleidoscope API](../reference/kaleidoscope-api.md)、
[状态、快照与克隆](../reference/snapshots.md)、[配置参考](../reference/configuration.md)。
