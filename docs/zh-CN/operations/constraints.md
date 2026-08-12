# 约束与安全边界

语言：[中文](constraints.md) | [English](../../en/operations/constraints.md)

## 平台

- Linux x86-64、Python 3.12、Isaac Sim 6.0.1；
- 仿真环境使用 Kit `pxr`，CPU dev 环境使用 `usd-core`，二者不混装；
- EULA 必须由部署者通过 `OMNI_KIT_ACCEPT_EULA` 明确接受；
- 仓库是 checkout application，不支持 wheel/editable install。

## 产品排他

- Mirror：PhysX/CPU、Newton/CPU 或 Newton/CUDA，一个 world；
- Kaleidoscope：PhysX/CUDA 或项目自有 multi-world Newton/CUDA，默认训练入口均为 headless、
  同一 CUDA device 上的 GPU-native runtime；显式调试入口可为任一后端打开单环境 viewport；
- 两个 product root 互不 import；一个 runtime 只拥有一个 `IsaacSession`；
- Kaleidoscope 不创建 trajectory planner、planning avoidance/collision cache、camera、SyntheticData、
  Replicator、录制、transport 或 telemetry placeholder；Newton profile 是正式后端，不是 placeholder。

## 线程与资源

USD、Isaac view、physics step/render 和 runtime mutation 只在 owner thread。后台 ingress/output
worker 只能处理已冻结数据。资源关闭遵循 consumer → producer；close timeout 不转移所有权。

Mirror v2 与原生 Kaleidoscope 的控制模式切换只允许发生在两次完整运动/decision 之间，并且一次切换
覆盖全部 robot/env；不会重建 runtime owner。回滚失败会永久 fail-stop，reset 不能清除，只能 close
并由调用方重建。

Hybrid 力/位控制第一阶段只支持 Mirror PhysX CPU、至少 200 Hz（维护 profile 为 240 Hz）、每条请求一个
目标 robot、物理 TCP 绑定和 `reference_frame: world`。目标 arm 的全部 joint 使用显式 effort；位控方向
是显式笛卡尔阻抗，不是逐方向 implicit joint drive。hand 的 implicit position drive 与其它 robot 不被覆盖。

Hybrid 增益只能作为独立 owner-queued operation 在两次完整运动之间修改。每段 motion 冻结一个
`hybrid_parameter_generation`；`force_axes` 由每条 motion 独立选择。filter 与安全限幅不可通过 wire
修改。motion 必须引用当前 tare generation，reset 会失效 tare；恢复 controller 失败会永久 fail-stop
并请求 runtime shutdown。

## 数据

- 长度 m、角度 rad、公开 quaternion `wxyz`；
- Kaleidoscope hot path 禁止 CPU/NumPy selector 和隐式 device/dtype copy；
- Gymnasium、persistent checkpoint 与 human viewport physics-to-USD sync 是三类显式主机边界；
- Mirror scene snapshot 与 Kaleidoscope episode snapshot schema 不兼容；
- setter/restore 先完整 preflight；不可证明 rollback 时 fail-stop。

## 网络

Mirror TCP/WebSocket/Foxglove 只绑定 loopback，不提供认证、授权或 TLS。任何远程访问必须使用认证
代理或 SSH tunnel。Queue、连接、消息、planner、camera 和 output byte 均有显式上限。

## Newton

Newton runtime 不创建 Isaac World，直接拥有 Model/State/Control/Solver。Mirror 分配前断言一个
world；Kaleidoscope 则从最终 `num_envs` 派生 `world_count`，并为每个 env 创建独立 world。两者都使用
项目 Newton runtime，不加载 Isaac Newton extension。Mirror 每个 render frame 只执行一次
physics-to-USD sync；camera 不归 physics manager 所有。

机器人 profile 的逐组件 `gravity=false` 在 model finalize 前投影为 `mjc:gravcomp=1`，由
Newton 求解器通过 `mjc:gravcomp` 补偿；它不会关闭同 world 动态对象的场景重力。Newton 不支持运行期逐 link
切换，修改重力策略必须重建 runtime。

## GPU RL

PhysX builder 固定用 GridCloner 并启用 env ID；Newton builder 固定创建独立 worlds。它们是与物理
引擎绑定的内部复制实现，不是公开配置 selector。Task 物理接触保留，但规划碰撞和避障禁用。Native/skrl step 中
observation/action/reward/done/state/snapshot/clone/RNG 都必须驻留同一 GPU。

Kaleidoscope task 不选择数值 backend。EE/直线 mode 必须用可选 `profiles.curobo` 选择
kinematics-only profile，并关闭 collision check。canonical profile 省略
`kinematics.collision_cache`；保留的合法值也会被忽略，从而不分配碰撞缓存；纯关节
`joint_control`/`joint_delta` mode 必须省略该引用。

Kaleidoscope action variant、shape 与 tick 数在构造期冻结；运行时 control mode 初始为 position，只能在
完整 decision 之间全局切换。SAME_STEP 的 issued/stepped 阶段都拒绝切换。Gymnasium、skrl 与
`KaleidoscopeTrainingPort` 不暴露 setter。Snapshot restore 不自动切换模式：schema 2 要求 active mode
相同，schema 1 只表示 position，generation 不会被恢复。

显式 viewport 是独立冷边界：只显示 `selected_env`，配置不进入 episode fingerprint。训练 physics tick
始终 `render=False`，只有 `env.render()` 可执行一次 render-only 更新且不得推进 simulation time。

## PhysX GPU 显存门禁

`configs/physics/physx/cuda.yaml` 的 `physics.memory` 是完整的 `GpuMemoryBudget`，四个字段缺一不可：

| 字段 | 约束 |
| --- | --- |
| `max_simulator_process_mib` | NVML 归属到当前 simulator PID 的进程显存上限，覆盖 Kit、PhysX、Torch 与其它原生 CUDA allocator |
| `min_free_floor_mib` | prelaunch、warmup 后及稳态采样都必须保留的设备空闲显存绝对下限 |
| `min_free_fraction_after_warmup` | warmup 后、稳态起点与终点必须保留的设备空闲显存比例，范围 `(0, 1]` |
| `max_steady_growth_mib` | steady final 相对 steady baseline 的 simulator PID 显存最大增长量，允许为 `0` |

该门禁只适用于 Kaleidoscope `physx_cuda` profile，不是 Newton capacity 的替代品。它通过 NVML 统计整个
simulator 进程，同时报告 Torch allocated/reserved；采样失败、PID 不可见或预算越界都 fail closed。
在有 Isaac/CUDA 的仿真环境执行：

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

`just test-simulation` 会连同七个正式 Kit closure、Mirror 四个 profile、Kaleidoscope 双后端与 Newton
容量 smoke 一起执行这条显存门禁；普通 CPU `quality` 不隐式启动它。
