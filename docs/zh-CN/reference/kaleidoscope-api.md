# Kaleidoscope API 参考

语言：[中文](kaleidoscope-api.md) | [English](../../en/reference/kaleidoscope-api.md)

Kaleidoscope 没有 JSON/CLI 控制服务。它通过 Python 进程内 API 交换 CUDA tensor，并可在最外层
选择 Gymnasium NumPy adapter。

## 构造

```python
from linkerbot_sim.kaleidoscope import make_torch_env

env = make_torch_env(profile="physx_cuda", num_envs=256)
```

`profile` 解析 `KaleidoscopeConfig`；`num_envs` 只能覆盖 mode root `environments.num_envs`，必须是正整数。构造会
接受 `physx_cuda` 的 PhysX CUDA/Fabric 或 `newton_cuda` 的项目 multi-world Newton。
训练 mode graph 仍拒绝 CPU PhysX、render/camera、trajectory planner、planning
collision/avoidance、transport、playback 和 telemetry 配置；两个默认训练 Kit 均为 headless、
GPU-native，public tensor contract 相同。显示配置由下文独立的 viewport 冷边界持有。

| profile | physics leaf | 后端内部派生的环境实现 | Kit | physics owner |
| --- | --- | --- | --- | --- |
| `physx_cuda` | `physx/cuda` | GridCloner + env IDs | `linkerbot_sim.kaleidoscope.physx_cuda.python.kit` | PhysX CUDA/Fabric replicated scene |
| `newton_cuda` | `newton/cuda` | multi-world + 独立 worlds | `linkerbot_sim.kaleidoscope.newton.python.kit` | 项目 `NewtonRuntime` 的独立 worlds |

Newton Kit 不加载 Isaac Newton extension；Kaleidoscope 没有 control profile，`physics.engine` 直接派生
Newton 专用 controller bundle。
后端只能在构造前通过 profile 选择，runtime 内不能热切换。

内置 mode 使用 `joint_control`，必须省略 `profiles.curobo`。EE/直线 action 的 task 仍只保存动作语义，
配套 mode root 必须增加 `profiles.curobo: kaleidoscope_batch_ik`；catalog 对前者拒绝该引用，对后者要求
该引用存在。

## Native Torch contract

| 属性/方法 | 形状与语义 |
| --- | --- |
| `num_envs` | 环境数量 N |
| `device` | mode root 的唯一 CUDA device |
| `action_dim` | 每个 env 的 action 列数 A |
| `observation_dim` | 每个 env 的 observation 列数 O |
| `reset()` | `(N,O)` observation 与 dense CUDA info |
| `reset_idx(env_ids)` | K 行 reset；selector 为 CUDA int64 `(K,)` |
| `step(actions)` | actions 为 float32 CUDA `(N,A)`；返回 observation/reward/terminated/truncated/info |
| `get_control_mode()` | 返回 initial/active mode、generation、支持集和全局 scope |
| `set_control_mode(mode, expected_generation=None)` | 只在两次完整运动之间切换全部 robot/env |
| `render()` | 仅显式 viewport 环境可用；刷新一帧且不推进 physics time |
| `is_running()` | viewport 窗口仍打开时为 true |
| `close()` | 幂等关闭 task/view/session |

Native `step` 不自动 reset done 行。直接调用者必须先 `reset_idx`；native/debug 入口会同步读取一次
主机 scalar，让这个可恢复的生命周期错误在推进 physics 前明确抛出。skrl adapter 使用
`KaleidoscopeTrainingPort` 的 SAME_STEP token，不经过该检查，并在 reset 覆写 buffer 前复制 terminal
transition；训练 action、observation、reset 与 rollout 数据仍常驻 CUDA。

### Runtime 控制模式

native env 固定从 position 启动。同一 runtime 内可在完整 `step` 之间切换：

```python
state = env.get_control_mode()
change = env.set_control_mode(
    "velocity",
    expected_generation=state.generation,
)
```

真实变化成功后 generation 单调加一；切到当前模式幂等且不写 engine，但 expected generation 冲突
始终优先拒绝。切换 scope 固定为全部 robot、全部 env，不接受 selector。step/reset/close/另一切换期间，
以及 SAME_STEP token 的 issued 和 stepped 阶段都禁止切换。

事务会冻结当前 q，先写旧模式 neutral，再应用全部 robot profile，最后写新模式 neutral 并同步：
position neutral 是当前 q，velocity/effort neutral 是零。前向失败按逆 robot 顺序 rollback；rollback
失败会永久 fail-stop，只能关闭并重建 env。整个过程不重建 session、physics runtime、task、IK 或
CUDA context。

| 固定 action variant | position | velocity | effort |
| --- | --- | --- | --- |
| `joint_control` | 支持 | 直接有界速度 | 受 profile limit 约束的直接 effort |
| `joint_delta` 与全部 EE/直线 variant | 支持 | position reference 的有界差分 | 拒绝 |

切换不改变 action shape 或 action variant。`KaleidoscopeTrainingPort`、Gymnasium 和 skrl 不暴露
setter，训练保持初始 position 模式。

Gymnasium/skrl action space 与固定 action variant 一致：`joint_control`/`joint_delta` 使用
`[-task.action.clip, task.action.clip]`；EE/直线模式直接解释米、旋转向量或四元数，因此 Box 边界为无穷，
运行时仍要求所有实际输入为有限 `float32`。`ee_pose_full` 与 `ee_linear_path_full` 的每行 `wxyz`
四元数还必须满足 norm > `1e-8`；运行时会在 CUDA 上归一化合法四元数，无界 Box 不表示零四元数有效。

## Action mode

- `joint_control`：不创建 cuRobo，并支持 position/velocity/effort 三通道；
- `joint_delta`：不创建 cuRobo，支持 position/velocity；
- EE position/full-pose：一次批量 IK，失败行 hold + penalty + truncate；
- EE linear position/full-pose：在 GPU 上生成固定 waypoint，一次批量 waypoint IK，同步逐 tick 写入；
- 没有 batch trajectory planner、图搜索、collision cache 或 avoidance。

task action 不包含 backend/profile 字段。EE/直线 composition 由 mode root 选择
`curobo: kaleidoscope_batch_ik`；该数值 profile 必须省略 `motion_planner`，关闭
`kinematics.collision_check`。canonical YAML 省略 `kinematics.collision_cache`；保留合法值也不会传入
后端，因此不分配碰撞缓存；batch capacity 必须覆盖最终环境数。

## State extension

```python
state = env.get_state(env_ids, fields=("robot.q", "object.pose_local_wxyz"))
env.set_state(state, env_ids)
snapshot = env.snapshot(env_ids)
env.restore_snapshot(snapshot, target_env_ids=other_ids)
env.clone_state(env_ids, other_ids, include_rng=True)
```

两个 physics backend 绑定完全相同的核心 canonical 字段，组装后的 task 再追加
history/counter/RNG 字段：

| 字段 | shape 与语义 |
| --- | --- |
| `robot.q` | `(N, Q_full)`；按场景顺序拼接所有机器人的完整 articulation DOF，不只是受控关节。 |
| `robot.qd` | `(N, Q_full)`；与 `robot.q` 相同的完整 articulation DOF 速度。 |
| `robot.target` | `(N, Q_controlled)`；当前 active mode 的 engine target，单位由 snapshot mode metadata 判定。 |
| `robot.position_reference` | `(N, Q_controlled)`；所有模式下都保持 rad 单位的位置参考。 |
| `object.pose_local_wxyz` | `(N, 7)`；env-local XYZ 位置与 `wxyz` 四元数。 |
| `object.com_velocity` | `(N, 6)`；物体质心的线速度和角速度。 |

Newton 还绑定后端私有的 `solver.persistent` matrix，在 CUDA 上保存每个 world 的
SolverMuJoCo TIME、ACT 与 WARMSTART 状态，并参与默认 get/set、snapshot/restore、reset 与 clone。
PhysX 没有对应字段。完整 snapshot fingerprint 因而随后端不同，不能在 PhysX 与 Newton 之间恢复，
尽管上表核心字段由两者共享。

`object.*` 字段拥有所选 scene 中唯一的非静态 rigid object。严格配置会拒绝第二个动态刚体和任何
dynamic chain，因此 snapshot/clone 不会静默漏掉其他动态对象状态。

Selector 规则：同一 device、`torch.int64`、一维、唯一且范围有效。clone 要求 source/target 等长且
不重叠。默认返回 owned CUDA tensor；setter 完整预检后才调用 engine writer。writer 失败会把
state API 标为 poisoned，runtime 必须关闭重建。

Snapshot 包含物理、task/history/counter/RNG 等登记字段，可恢复 episode；普通 Mirror scene
snapshot 不具备这些字段。两个后端的 `get_state`/`set_state`、snapshot capture/restore 和
`clone_state` 都在 `env.device` 内完成；磁盘 checkpoint 通过显式 cold API 整批转 CPU，不进入
hot step。

schema 2 snapshot 记录 `control_mode` 与 `control_generation`。Restore 要求 runtime 已处于相同模式，
不会自动切 mode；generation 只用于来源诊断，不会回退 runtime generation。legacy schema 1 只代表
position；缺少 `robot.position_reference` 时从旧 `robot.target` 派生。observation
`command_target_error` 始终定义为 `position_reference - actual_q`，单位 rad，不是 velocity/effort
tracking error。velocity action 按 decision duration 推进 shadow reference；effort action 在每次
decision 开始时把它锚到当前 q。

## Gymnasium

```python
import gymnasium as gym
from linkerbot_sim.kaleidoscope import register_gymnasium_envs

register_gymnasium_envs()
env = gym.make_vec(
    "linkerbot/TBlockPush-Kaleidoscope-v1",
    num_envs=64,
    profile="physx_cuda",
)
```

注册是显式、幂等的；导入 facade 不修改全局 registry。adapter 支持 `disabled` 或 `same_step`
autoreset，返回 NumPy，并集中执行唯一允许的 D2H/H2D 转换。传入 `render_mode="human"` 时，
factory 改为构造下述 viewport 环境；`render()` 仍是显式调用，不会混入每步 adapter 转换。

## Human viewport 冷边界

```python
from linkerbot_sim.kaleidoscope import make_viewport_env

env = make_viewport_env(
    profile="physx_cuda",
    viewport_profile="kaleidoscope",
    num_envs=4,
)
try:
    observations, info = env.reset()
    # env.step(actions) 的内部 physics tick 始终使用 render=False。
    env.render()
finally:
    env.close()
```

`make_viewport_env` 通过 `viewport` 接收已加载的 `KaleidoscopeViewportSettings`，或者通过默认
`viewport_profile="kaleidoscope"` 加载 `configs/visualization/kaleidoscope.yaml`。选择该 profile
就固定创建 human-viewer 窗口；它只拥有 `selected_env`、渲染节奏、窗口/renderer 设置和 scene
visual，不属于 `KaleidoscopeConfig`，因此不会改变 episode
snapshot/clone fingerprint。PhysX 与 Newton 分别选择对应的 `*_viewport.python.kit`；只有
`selected_env` 被同步到 renderer-facing USD，其它环境继续在 GPU physics batch 中推进。

`env.step()` 不会隐式画帧，所有内部物理 tick 都是 `step(render=False)`；调用方按
`render_every_n_steps` 显式执行 `env.render()`。该 render-only 边界不推进 simulation time。
viewport Kit 只增加 RTX/human viewport，继续排除 camera、SyntheticData、Replicator、录制、图像
observation 和 telemetry。

## skrl

`linkerbot_sim.training.skrl` 提供 CUDA rollout memory、terminal-observation-aware PPO 和 Torch
adapter。training 层只依赖 public `KaleidoscopeTrainingPort`，不拥有 Isaac 或访问具体 task buffer。
该 port 和 skrl adapter 都不提供 `set_control_mode`。
trainer factory 会在分配 rollout memory/agent 前要求 policy 与 value model 都精确使用
`env.device`；它们的 Gymnasium Box `observation_space`、`state_space`、`action_space` 还必须在类型、
shape、dtype 与 bounds 上和环境一致。

## 生命周期

不要同时让两个 env 拥有同一个 session。关闭顺序是 task/view → `IsaacSession`；异常后仍应调用
`close()`。`make_torch_env()` 构造的 headless 训练环境调用 `render()` 会明确失败；只有
`make_viewport_env()` 或 Gymnasium `render_mode="human"` 环境支持它。
